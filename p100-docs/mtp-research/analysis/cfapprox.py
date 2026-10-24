# cfapprox.py [set ...]: CF_ANALYSIS step 4. Approximations of the per-setting tables, each scored by simulated test
# throughput vs the setting's own tables (out/cf/grid-<set>.json): loss = rate_approx / rate_own - 1 per grid setting;
# mean, worst, worst-10% (mean of the 48 lowest), and storage. Anchor choices and fits use dev/val only.
# Writes out/cf/approx-<set>.json and prints the comparison table.
import json, os, sys, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.isotonic import IsotonicRegression
import cfcore as C

OUT = C.OUT
TABLE_BYTES = 2 * 24 * 4 + 4  # one set: iso_q + iso_p1 float32 [24] + theta (the x grid is shared)


class Ev:
    """fast policy evaluation for one Set: rates on val and test for (setting gi, tables, theta)."""

    def __init__(self, setname):
        self.setname = setname
        self.S = C.Set(C.SETS[setname])
        self.rows = json.load(open(f"{OUT}/grid-{setname}.json"))
        self.own = [(np.array(r["tq"]), np.array(r["tp"]), r["theta"]) for r in self.rows]
        self.r_own = {C.TEST: np.array([r["r_own"] for r in self.rows])}
        self._cs = {}
        S = self.S
        self.r_own[C.VAL] = np.array([self.rate(gi, *self.kT(*self.own[gi]))[C.VAL] for gi in range(len(C.GRID))])

    def cs(self, gi):
        if gi not in self._cs:
            A = self.S.a(gi)
            Sv = np.cumprod(np.where(self.S.valid, A, 0.0), axis=1)
            self._cs[gi] = np.c_[np.zeros(self.S.n), np.cumsum(Sv, axis=1)].astype(np.float32)
        return self._cs[gi]

    def kT(self, tq, tp, theta):
        S = self.S
        k, st = S.k_of(S.score(tq, tp) < theta)
        T = S.c_d0 * (st > 0) + S.c_step * st + S.c_const - S.c_s * (C.D - k)
        return k, T

    def kT_score(self, sc, theta):
        S = self.S
        k, st = S.k_of(sc < theta)
        T = S.c_d0 * (st > 0) + S.c_step * st + S.c_const - S.c_s * (C.D - k)
        return k, T

    def rate(self, gi, k, T, splits=(C.VAL, C.TEST)):
        E = 1.0 + self.cs(gi)[np.arange(self.S.n), k].astype(float)
        return {s: self.S.rate(E, T, s) for s in splits}

    def loss(self, rates_test):
        return np.asarray(rates_test) / self.r_own[C.TEST] - 1


def stats(loss):
    l = np.sort(loss)
    n10 = max(1, len(l) // 10)
    return dict(mean=float(loss.mean()), worst=float(l[0]), worst10=float(l[:n10].mean()))


def dist(g, a):
    """simple distance between settings: temperature, top_p, presence scaled by the grid's ranges (top_k, min_p
    ignored)."""
    return abs(g[0] - a[0]) / 0.9 + abs(g[1] - a[1]) / 0.2 + abs(g[4] - a[4]) / 1.5


def lattice_weights(g, Ts, Ps, Rs):
    """multilinear weights of setting g on the lattice Ts x Ps x Rs (clamped to its box). Returns [(node, w)]."""
    def axis(v, xs):
        xs = list(xs)
        if len(xs) == 1:
            return [(0, 1.0)]
        v = min(max(v, xs[0]), xs[-1])
        j = int(np.searchsorted(xs, v, side="right") - 1)
        j = min(max(j, 0), len(xs) - 2)
        f = (v - xs[j]) / (xs[j + 1] - xs[j])
        return [(j, 1 - f), (j + 1, f)]
    out = []
    for (i, wi), (j, wj), (k, wk) in itertools.product(axis(g[0], Ts), axis(g[1], Ps), axis(g[4], Rs)):
        w = wi * wj * wk
        if w > 0:
            out.append(((Ts[i], Ps[j], Rs[k]), w))
    return out


def run(setname):
    E = Ev(setname)
    S = E.S
    G = C.GRID
    nG = len(G)
    res = {}

    def record(lab, rates_test, storage, extra=None):
        l = E.loss(rates_test)
        st = stats(l)
        st.update(storage=storage, loss=l.tolist())
        if extra:
            st.update(extra)
        res[lab] = st
        print(f"| {lab} | {st['mean'] * 100:+.2f}% | {st['worst'] * 100:+.2f}% | {st['worst10'] * 100:+.2f}% | {storage} |", flush=True)
        return st

    print(f"\n## {setname}: loss vs own tables (test, simulated; 480 settings)")
    print("| approximation | mean | worst | worst 10% | storage (bytes) |\n|---|---|---|---|---|")
    # (a) production's current policy
    record("a. production (chat/think tables by temperature, else p-min 0.2)", [r["r_prod"] for r in E.rows], 2 * TABLE_BYTES)
    record("(ref) p-min 0.2 everywhere", [r["r_pmin"] for r in E.rows], 0)
    # production's tables with the temperature gate removed: chat tables below 0.95, thinking tables from 0.95 up
    rt = []
    for gi, g in enumerate(G):
        tab = C.PROD_CHAT + (0.125,) if g[0] < 0.95 else C.PROD_THINK + (0.15,)
        rt.append(E.rate(gi, *E.kT(*tab))[C.TEST])
    record("a'. production's two sets, gate widened (chat T < 0.95, thinking T >= 0.95)", rt, 2 * TABLE_BYTES)
    # production's arrays with no gate at all: one set at every temperature
    for lab, tab in [("chat", C.PROD_CHAT + (0.125,)), ("thinking", C.PROD_THINK + (0.15,))]:
        k, T = E.kT(*tab)
        record(f"a2. production's {lab} tables/theta at every setting (no gate)", [E.rate(gi, k, T)[C.TEST] for gi in range(nG)], TABLE_BYTES)
    if len(S.parts) == 2:
        # content-keyed: thinking-off requests use the chat tables, thinking-on the thinking tables, at every setting
        k1, T1 = E.kT(*C.PROD_CHAT, 0.125)
        k2, T2 = E.kT(*C.PROD_THINK, 0.15)
        th = S.trace == 1
        k, T = np.where(th, k2, k1), np.where(th, T2, T1)
        record("a3. production's two sets keyed by thinking on/off (not temperature), no gate",
               [E.rate(gi, k, T)[C.TEST] for gi in range(nG)], 2 * TABLE_BYTES)

    # anchor matrix: rates of every grid setting under every candidate anchor's own tables+theta (k20, m0 anchors)
    cand = [gi for gi, g in enumerate(G) if g[2] == 20 and g[3] == 0.0]
    RV = np.zeros((len(cand), nG)); RT = np.zeros((len(cand), nG))
    for ai, a in enumerate(cand):
        k, T = E.kT(*E.own[a])
        for gi in range(nG):
            r = E.rate(gi, k, T)
            RV[ai, gi] = r[C.VAL]; RT[ai, gi] = r[C.TEST]
    LV = RV / E.r_own[C.VAL][None, :] - 1
    # (b0) one set for everything: the anchor with the best mean val loss
    b1 = int(np.argmax(LV.mean(1)))
    record(f"b. one table set for all (best on val: {C.sname(G[cand[b1]])})", RT[b1], TABLE_BYTES)
    # (b) nearest anchor, greedy on val
    D = np.array([[dist(G[gi], G[a]) for gi in range(nG)] for a in cand])
    chosen = []
    for K in range(1, 9):
        best = None
        for ai in range(len(cand)):
            if ai in chosen:
                continue
            A_ = chosen + [ai]
            asg = np.array(A_)[np.argmin(D[A_], axis=0)]
            obj = LV[asg, np.arange(nG)].mean()
            if best is None or obj > best[0]:
                best = (obj, ai)
        chosen.append(best[1])
        if K in (2, 4, 8):
            asg = np.array(chosen)[np.argmin(D[chosen], axis=0)]
            record(f"b. nearest anchor, K={K}: " + ", ".join(C.sname(G[cand[a]]) for a in chosen), RT[asg, np.arange(nG)],
                   K * TABLE_BYTES, dict(anchors=[G[cand[a]] for a in chosen]))
    # (c) multilinear interpolation over (temperature, top_p, presence) lattices, k20 m0 anchors
    lattices = {
        "T{0.3,1.2}": ([0.3, 1.2], [0.95], [0.75]),
        "T{0.3,0.7,1.2}": ([0.3, 0.7, 1.2], [0.95], [0.75]),
        "T{0.3,0.7,1.2} x pr{0,1.5}": ([0.3, 0.7, 1.2], [0.95], [0.0, 1.5]),
        "T{0.3,1.2} x p{0.8,1} x pr{0,1.5}": ([0.3, 1.2], [0.8, 1.0], [0.0, 1.5]),
        "T{0.3,0.7,1.2} x p{0.8,1} x pr{0,1.5}": ([0.3, 0.7, 1.2], [0.8, 1.0], [0.0, 1.5]),
        "T{0.3,0.5,0.7,1.0,1.2} x p{0.8,1} x pr{0,1.5}": ([0.3, 0.5, 0.7, 1.0, 1.2], [0.8, 1.0], [0.0, 1.5]),
        "T{0.3,0.5,0.7,1.0,1.2} x p{0.8,0.9,1} x pr{0,0.75,1.5}": ([0.3, 0.5, 0.7, 1.0, 1.2], [0.8, 0.9, 1.0], [0.0, 0.75, 1.5]),
        "full T(10) x p(4) x pr(3)": (C.TEMPS, C.TOPPS, C.PRESS),
    }
    for lab, (Ts, Ps, Rs) in lattices.items():
        rt, nodes = [], set()
        for gi, g in enumerate(G):
            tq = np.zeros(24); tp = np.zeros(24); th = 0.0
            for node, w in lattice_weights(g, Ts, Ps, Rs):
                a = C.GIDX[(node[0], node[1], 20, 0.0, node[2])]
                nodes.add(a)
                tq += w * E.own[a][0]; tp += w * E.own[a][1]; th += w * E.own[a][2]
            rt.append(E.rate(gi, *E.kT(tq, tp, th))[C.TEST])
        record(f"c. multilinear {lab}", rt, len(Ts) * len(Ps) * len(Rs) * TABLE_BYTES)
    # (c') the same lattice with top_k and min_p as extra axes (full grid lookup = own tables, the ceiling)
    # (f) single theta: own tables with one theta for all settings (best mean val ratio), and lattice with fixed theta
    curves = np.array([r["val_curve"] for r in E.rows])  # (nG, n_theta) val rates
    vr = curves / curves.max(1, keepdims=True)
    ti = int(np.argmax(vr.mean(0)))
    th1 = float(C.THETAS[ti])
    rt = [E.rate(gi, *E.kT(E.own[gi][0], E.own[gi][1], th1))[C.TEST] for gi in range(nG)]
    thetas = np.array([o[2] for o in E.own])
    record(f"f. own tables, one theta for all ({th1})", rt, 480 * 2 * 24 * 4 + 4,
           dict(theta=th1, own_theta_range=[float(thetas.min()), float(thetas.max())],
                own_theta_quartiles=np.percentile(thetas, [25, 50, 75]).tolist()))
    json.dump(res, open(f"{OUT}/approx-{setname}.json", "w"))
    return res


if __name__ == "__main__":
    for s in sys.argv[1:] or C.available():
        run(s)
