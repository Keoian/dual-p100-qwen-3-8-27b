# cfrecommend.py: the candidate policies for deployment, scored on every set (chat content, thinking content, pooled)
# with a bootstrap interval of the grid-mean loss vs own tables; an extrapolation check outside the grid's
# temperatures; and the recommended tables as C arrays (out/cf/recommended-c.txt).
import json, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.isotonic import IsotonicRegression
import cfcore as C, cfvec
from cfapprox import Ev, stats
from cfboot import mean_loss_ci, c_arrays

OUT = C.OUT
G = C.GRID
nG = len(G)


def iso(x, y, w):
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, y, sample_weight=w)


def marginal(E, gis):
    """one table set fit on dev rows pooled over the settings gis; theta = best mean val ratio over the whole grid."""
    S = E.S
    dev = S.fit_w > 0
    rq = S.q[S.rows_c, S.rows_d][dev]; rp = S.p1[S.rows_c, S.rows_d][dev]
    ys = np.concatenate([S.a_rows(gi)[dev] for gi in gis]); ws = np.tile(S.fit_w[dev], len(gis))
    tq = iso(np.tile(rq, len(gis)), ys, ws).predict(C.X24); tp = iso(np.tile(rp, len(gis)), ys, ws).predict(C.X24)
    return tq, tp


def best_theta(E, tables_of, gis):
    vr = np.zeros(len(C.THETAS))
    for gi in gis:
        tq, tp = tables_of(gi)
        vr += E.S.theta_curve(E.S.a(gi), tq, tp) / E.r_own[C.VAL][gi]
    return float(C.THETAS[int(np.argmax(vr))])


def t_interp(t, temps, tabs):
    t = min(max(t, temps[0]), temps[-1])
    j = min(int(np.searchsorted(temps, t, side="right") - 1), len(temps) - 2)
    f = (t - temps[j]) / (temps[j + 1] - temps[j])
    return tuple((1 - f) * np.asarray(tabs[j][i]) + f * np.asarray(tabs[j + 1][i]) for i in range(3))


def a_custom(S, g):
    """(n, D) a_S for a setting outside the grid (computed from the raw logits, like cfcore.a_matrix)."""
    out = np.zeros((S.n, C.D))
    off = 0
    t, tp, tk, mp, pr = g
    for p in S.parts:
        R = cfvec.Rows(p["c"])
        a, _, _ = R.accept(t, tk, tp, mp, pr)
        m = len(p["rr"])
        out[S.rows_c[off:off + m], S.rows_d[off:off + m]] = a
        off += m
    return out


def main():
    Es = {s: Ev(s) for s in ("chat", "think", "pooled")}
    Ep = Es["pooled"]
    # R2: one pooled marginal set (every 4th grid setting, both contents equally weighted), theta on pooled val
    tq2, tp2 = marginal(Ep, list(range(0, nG, 4)))
    th2 = best_theta(Ep, lambda gi: (tq2, tp2), range(nG))
    # R3: temperature-only tables at T 0.3 / 0.7 / 1.2 (pooled marginal over the 48 settings at each), theta per anchor
    temps3 = [0.3, 0.7, 1.2]
    tabs3 = []
    for t in temps3:
        gis = [gi for gi, g in enumerate(G) if g[0] == t]
        tq, tp = marginal(Ep, gis)
        tabs3.append((tq, tp, best_theta(Ep, lambda gi: (tq, tp), gis)))
    pc, pt = C.PROD_CHAT + (0.125,), C.PROD_THINK + (0.15,)

    def r0(g):  # production today
        t = np.float32(g[0])
        if np.float32(0.5) <= t <= np.float32(0.9):
            return pc
        if np.float32(0.95) <= t <= np.float32(1.05):
            return pt
        return None

    def r1(g):  # production arrays, chat set at every temperature outside the thinking window
        return pt if np.float32(0.95) <= np.float32(g[0]) <= np.float32(1.05) else pc

    cands = {
        "R0 production today (chat tables T 0.5-0.9, thinking 0.95-1.05, else p-min 0.2)": r0,
        "R1 production arrays, chat set at every T outside 0.95-1.05 (no p-min fallback)": r1,
        "R1b production chat set at every T (thinking set unused)": lambda g: pc,
        f"R2 one new set for all settings (pooled marginal fit, theta {th2})": lambda g: (tq2, tp2, th2),
        "R3 iso(q, T): new sets at T 0.3/0.7/1.2, linear in T (clamped)": lambda g: t_interp(g[0], temps3, tabs3),
    }
    res = {"R2": dict(tq=tq2.tolist(), tp=tp2.tolist(), theta=th2),
           "R3": dict(temps=temps3, tabs=[(a.tolist(), b.tolist(), c) for a, b, c in tabs3]), "scores": {}}
    print("| policy | set | mean loss vs own [95%] | worst | worst 10% |\n|---|---|---|---|---|")
    for lab, f in cands.items():
        for s, E in Es.items():
            def tab(gi, f=f, E=E):
                t = f(G[gi])
                if t is None:
                    S = E.S
                    return lambda: (lambda k, st: (k, S.c_d0 * (st > 0) + S.c_step * st + S.c_const - S.c_s * (C.D - k)))(
                        *S.k_of(np.nan_to_num(S.p1, nan=1.0) < 0.2))
                return t
            m, ci = mean_loss_ci(E, tab)
            rt = []
            for gi in range(nG):
                t = tab(gi)
                k, T = t() if callable(t) else E.kT(*t)
                rt.append(E.rate(gi, k, T)[C.TEST])
            st = stats(E.loss(rt))
            res["scores"][f"{lab} | {s}"] = dict(mean=m, ci=ci.tolist(), worst=st["worst"], worst10=st["worst10"])
            print(f"| {lab} | {s} | {m * 100:+.2f}% [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}] | {st['worst'] * 100:+.2f}% | "
                  f"{st['worst10'] * 100:+.2f}% |", flush=True)
    # extrapolation outside the grid's temperatures (pooled; own tables fit for each of these settings)
    print("\n## outside the grid: T 0.1 / 0.2 / 1.3 / 1.5 (pooled set; loss vs that setting's own tables, test)")
    print("| setting | own t/s | p-min 0.2 | R1 | R2 | R3 |\n|---|---|---|---|---|---|")
    S = Ep.S
    ext = []
    for t in (0.1, 0.2, 1.3, 1.5):
        for tp_ in (0.8, 0.95):
            for pr in (0.0, 1.5):
                g = (t, tp_, 20, 0.0, pr)
                A = a_custom(S, g)
                y = A[S.rows_c, S.rows_d]
                tq, tp = S.fit(a_rows=y)
                th, _ = S.choose_theta(A, tq, tp)
                r_own = S.rate(*S.pol_tables(A, tq, tp, th), C.TEST)
                rp = S.rate(*S.pol_pmin(A, 0.2), C.TEST) / r_own - 1
                r1_ = S.rate(*S.pol_tables(A, *r1(g)), C.TEST) / r_own - 1
                r2_ = S.rate(*S.pol_tables(A, tq2, tp2, th2), C.TEST) / r_own - 1
                r3_ = S.rate(*S.pol_tables(A, *t_interp(t, temps3, tabs3)), C.TEST) / r_own - 1
                ext.append((C.sname(g), r_own, rp, r1_, r2_, r3_))
                print(f"| {C.sname(g)} | {r_own:.2f} | {rp * 100:+.2f}% | {r1_ * 100:+.2f}% | {r2_ * 100:+.2f}% | {r3_ * 100:+.2f}% |", flush=True)
    res["extrapolation"] = ext
    json.dump(res, open(f"{OUT}/recommend.json", "w"), indent=1)
    with open(f"{OUT}/recommended-c.txt", "w") as f:
        x = "static const float spec_iso_x [24] = { " + ", ".join(f"{v:.3f}f" for v in C.X24[:12]) + ",\n" + " " * 39 + \
            ", ".join(f"{v:.3f}f" for v in C.X24[12:]) + " };"
        f.write("// R2: one table set for every target sampling setting (CF_ANALYSIS.md); replaces the chat and thinking sets\n")
        f.write(f"// and the temperature gate. Stop before drafted token d when prod_{{i<d}} iso_q(q_i) * iso_p1(p1_d) < {th2}.\n")
        f.write(x + "\n" + c_arrays(tq2, tp2) + "\n")
        f.write(f"static const float spec_stop_iso_theta = {th2}f;\n\n")
        f.write("// R3 (optional refinement): temperature-indexed sets at T = 0.3 / 0.7 / 1.2, linear in T, clamped outside.\n")
        f.write("static const float spec_iso_t    [3] = { " + ", ".join(f"{t:.1f}f" for t in temps3) + " };\n")
        for t, (a, b, c) in zip(temps3, tabs3):
            sfx = f"_t{int(round(t * 10)):02d}"
            f.write(c_arrays(a, b, sfx) + "\n")
        f.write("static const float spec_iso_theta_t[3] = { " + ", ".join(f"{c}f" for _, _, c in tabs3) + " };\n")
    print(open(f"{OUT}/recommended-c.txt").read())


if __name__ == "__main__":
    main()
