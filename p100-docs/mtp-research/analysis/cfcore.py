# cfcore.py: shared machinery of the counterfactual grid (CF_ANALYSIS.md): the setting grid, a_S per drafted row
# (cached), the isotonic table fit, the exact-safe lookup rule at n-max 4 (W 5) and its simulated throughput with the
# chat set's cost model (cache/cost.json, as sim.py), and the paired bootstrap over test prompts.
# A "Set" is one trace (cfchat / cfthink) or both pooled. Pooled: each trace's prompts are separate units; fits weight
# each trace's dev rows equally in total, rates weight each trace's prompts so both traces carry equal token mass.
import json, os, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.isotonic import IsotonicRegression
import cfload, cfvec

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "cf")
OUT = os.path.join(HERE, "out", "cf")
D = cfload.D
X24 = np.r_[0.0, 0.02, 0.05, np.round(np.arange(0.1, 0.951, 0.05), 3), 0.98, 0.995, 1.0]
assert len(X24) == 24
TEMPS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
TOPPS = [0.8, 0.9, 0.95, 1.0]
TOPKS = [20, 40]
MINPS = [0.0, 0.05]
PRESS = [0.0, 0.75, 1.5]
GRID = list(itertools.product(TEMPS, TOPPS, TOPKS, MINPS, PRESS))  # (temp, top_p, top_k, min_p, pres)
GIDX = {g: i for i, g in enumerate(GRID)}
CHAT_S = (0.7, 0.8, 20, 0.0, 1.5)
THINK_S = (1.0, 0.95, 20, 0.0, 0.0)
THETAS = np.round(np.arange(0.0, 0.5001, 0.0125), 4)
DEV, VAL, TEST = 0, 1, 2
COSTP = json.load(open(os.path.join(HERE, "cache", "cost.json")))
B_BOOT = 2000


def sname(g):
    t, tp, tk, mp, pr = g
    return f"T{t:.1f}/p{tp:g}/k{tk}/m{mp:g}/pr{pr:g}"


def read_prod_tables(path):
    rows = [l.split() for l in open(path) if l[:1].isdigit()]
    x = np.array([float(r[0]) for r in rows]); q = np.array([float(r[1]) for r in rows]); p = np.array([float(r[2]) for r in rows])
    assert np.allclose(x, X24)
    return q, p


PROD_CHAT = read_prod_tables(os.path.join(HERE, "out", "iso_tables.txt"))
PROD_THINK = read_prod_tables(os.path.join(HERE, "out", "think", "iso_tables.txt"))


def a_matrix(name):
    """(len(GRID), n_rows) float32 a_S for every grid setting, rows as cfvec.Rows(c) orders them; cached."""
    c, meta = cfload.load(name)
    p = os.path.join(CACHE, f"{name}.a.npy")
    src = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(p) and os.path.getmtime(p) > os.path.getmtime(src):
        return np.load(p, mmap_mode="r")
    R = cfvec.Rows(c)
    A = np.empty((len(GRID), len(R.x)), np.float32)
    nin = 0
    for i, (t, tp, tk, mp, pr) in enumerate(GRID):
        a, _, ex = R.accept(t, tk, tp, mp, pr)
        A[i] = a
        nin += int((~ex).sum())
    np.save(p, A)
    print(f"{name}: a_S cached for {len(GRID)} settings x {len(R.x)} rows; inexact (row, setting) pairs: {nin}", flush=True)
    return A


class Set:
    def __init__(self, names):
        self.names = list(names)
        parts = []
        for ti, nm in enumerate(self.names):
            c, meta = cfload.load(nm)
            ok = c["cf_ok"]
            R = cfvec.Rows(c)
            # map rows (R.rc, R.rd) of the kept cycles to a (n_kept, D) matrix
            cyc = np.where(ok)[0]
            pos = -np.ones(len(ok), np.int64); pos[cyc] = np.arange(len(cyc))
            parts.append(dict(c=c, cyc=cyc, rr=pos[R.rc], rd=R.rd, A=a_matrix(nm), ti=ti, own=tuple(meta["sampling"][0][:5])))
        self.parts = parts
        cat = lambda f: np.concatenate([f(p) for p in parts])
        self.n_real = cat(lambda p: np.minimum(p["c"]["n_real"][p["cyc"]], D))
        self.pos0 = cat(lambda p: p["c"]["pos0"][p["cyc"]]).astype(float)
        self.split = cat(lambda p: p["c"]["split"][p["cyc"]])
        self.cat_ = cat(lambda p: p["c"]["cat"][p["cyc"]])
        self.unit = cat(lambda p: p["c"]["pid"][p["cyc"]] + 72 * p["ti"])
        self.trace = cat(lambda p: np.full(len(p["cyc"]), p["ti"]))
        self.n_emit = cat(lambda p: p["c"]["n_emit"][p["cyc"]]).astype(float)
        self.q = cat(lambda p: p["c"]["d_q"][p["cyc"]])
        self.p1 = cat(lambda p: p["c"]["d_p1"][p["cyc"]])
        self.tid = cat(lambda p: p["c"]["d_tid"][p["cyc"]])
        self.tp = cat(lambda p: p["c"]["d_tp"][p["cyc"]])
        self.twin = cat(lambda p: p["c"]["d_twin"][p["cyc"]])
        self.did = cat(lambda p: p["c"]["d_id"][p["cyc"]])
        off = np.cumsum([0] + [len(p["cyc"]) for p in parts])
        self.rows_c = cat(lambda p: p["rr"] + off[p["ti"]])
        self.rows_d = cat(lambda p: p["rd"])
        self.n = len(self.n_real)
        self.nu = 72 * len(parts)
        self.valid = np.arange(D)[None, :] < self.n_real[:, None]
        # fit weights per row: each trace's dev rows sum to 1 (pooled); 1 per row for a single trace
        rw = []
        for p in parts:
            m = p["c"]["split"][p["cyc"]][p["rr"]] == DEV
            rw.append(np.where(m, 1.0 / max(m.sum(), 1), 0.0) if len(parts) > 1 else m.astype(float))
        self.fit_w = np.concatenate(rw)
        self.row_split = self.split[self.rows_c]
        # unit weights per split: equal token mass per trace (pooled)
        self.unit_w = {}
        for s in (DEV, VAL, TEST):
            w = np.zeros(self.nu)
            Ns = [self.n_emit[(self.trace == ti) & (self.split == s)].sum() for ti in range(len(parts))]
            for ti in range(len(parts)):
                w[72 * ti:72 * (ti + 1)] = Ns[0] / Ns[ti]
            self.unit_w[s] = w
        # cycle cost pieces at W = 5 (sim.cost with W = 5)
        P = COSTP
        ctx = self.pos0 / 1000.0
        self.c_const = P["V"]["5"] + P["k5"] * ctx + P["o5"]
        self.c_step = P["d1"] + P["dctx"] * ctx
        self.c_d0 = P["d0"]; self.c_s = P["s"]
        self._boot = {}

    def a(self, gi):
        """(n, D) a_S of grid setting gi (0 outside the drafted rows)."""
        out = np.zeros((self.n, D))
        off = 0
        for p in self.parts:
            m = len(p["rr"])
            out[self.rows_c[off:off + m], self.rows_d[off:off + m]] = p["A"][gi]
            off += m
        return out

    def a_rows(self, gi):
        return np.concatenate([np.asarray(p["A"][gi], float) for p in self.parts])

    def own_gi(self):
        # meta order (temp, top_k, top_p, min_p, pres) -> GRID order (temp, top_p, top_k, min_p, pres)
        return [GIDX[(float(o[0]), float(o[2]), int(o[1]), float(o[3]), float(o[4]))] for o in (p["own"] for p in self.parts)]

    # ---- fits
    def fit(self, gi=None, a_rows=None, w=None):
        """isotonic iso_q (x = q) and iso_p1 (x = p1) on dev rows -> 24-point tables."""
        y = self.a_rows(gi) if a_rows is None else a_rows
        w = self.fit_w if w is None else w
        m = w > 0
        xq = self.q[self.rows_c, self.rows_d][m]; xp = self.p1[self.rows_c, self.rows_d][m]
        iq = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(xq, y[m], sample_weight=w[m])
        ip = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(xp, y[m], sample_weight=w[m])
        return iq.predict(X24), ip.predict(X24)

    # ---- policies
    def score(self, tq, tp):
        """S_hat[:, d] = prod_{i<d} tq(q_i) * tp(p1_d), linear interpolation on the 24-point grid."""
        post = np.where(self.valid, np.interp(np.nan_to_num(self.q), X24, tq), 0.0)
        prior = np.where(self.valid, np.interp(np.nan_to_num(self.p1), X24, tp), 0.0)
        cp = np.c_[np.ones(self.n), np.cumprod(post, axis=1)[:, :-1]]
        return cp * prior

    def k_of(self, stopmask):
        hit = stopmask & self.valid
        first = np.where(hit.any(1), hit.argmax(1), D)
        k = np.minimum(first, self.n_real)
        steps = np.where(first < self.n_real, first + 1, D)
        return k, steps

    def ET(self, A, k, steps):
        """expected tokens and time per cycle; A = (n, D) a's (or survival-ready)."""
        S = np.cumprod(np.where(self.valid, A, 0.0), axis=1)
        cs = np.c_[np.zeros(self.n), np.cumsum(S, axis=1)]
        E = 1.0 + cs[np.arange(self.n), k]
        T = self.c_d0 * (steps > 0) + self.c_step * steps + self.c_const - self.c_s * (D - k)
        return E, T

    def agg(self, E, T, split):
        m = self.split == split
        u = self.unit[m]
        sE = np.bincount(u, E[m], self.nu); sT = np.bincount(u, T[m], self.nu)
        N = np.bincount(u, self.n_emit[m], self.nu); nc = np.bincount(u, minlength=self.nu)
        has = nc > 0
        tau = np.where(has, sT / np.maximum(sE, 1e-9), 0.0)
        return N, tau, has

    def rate(self, E, T, split, wb=None):
        N, tau, has = self.agg(E, T, split)
        w = self.unit_w[split] * has
        tr = COSTP["t_req"]
        if wb is None:
            return (w * N).sum() / ((w * N * tau).sum() + w.sum() * tr) * 1e6
        W = wb * w[None, :]
        return (W @ N) / (W @ (N * tau) + W.sum(1) * tr) * 1e6

    def boot_w(self, split, B=B_BOOT, seed=0):
        if split not in self._boot:
            rng = np.random.default_rng(seed)
            wb = np.zeros((B, self.nu))
            for ti in range(len(self.parts)):
                ids = np.unique(self.unit[(self.split == split) & (self.trace == ti)])
                for b in range(B):
                    wb[b] += np.bincount(rng.choice(ids, len(ids)), minlength=self.nu)
            self._boot[split] = wb
        return self._boot[split]

    def boot_gain(self, pol, base, split=TEST):
        wb = self.boot_w(split)
        g = self.rate(*pol, split) / self.rate(*base, split) - 1
        gs = self.rate(*pol, split, wb) / self.rate(*base, split, wb) - 1
        return g, np.percentile(gs, [2.5, 97.5])

    def pol_tables(self, A, tq, tp, theta):
        k, st = self.k_of(self.score(tq, tp) < theta)
        return self.ET(A, k, st)

    def pol_pmin(self, A, pmin=0.2):
        k, st = self.k_of(np.nan_to_num(self.p1, nan=1.0) < pmin)
        return self.ET(A, k, st)

    def theta_curve(self, A, tq, tp, split=VAL, thetas=THETAS):
        """rate on `split` for every theta (vectorized through the running min of S_hat)."""
        sc = self.score(tq, tp)
        M = np.minimum.accumulate(np.where(self.valid, sc, np.inf), axis=1)
        out = np.empty(len(thetas))
        for i, th in enumerate(thetas):
            k, st = self.k_of(M < th)
            out[i] = self.rate(*self.ET(A, k, st), split)
        return out

    def choose_theta(self, A, tq, tp, split=VAL):
        r = self.theta_curve(A, tq, tp, split)
        i = int(np.argmax(r))
        return float(THETAS[i]), r

    def prod_policy(self, g, A):
        """production's current policy for setting g: chat tables/0.125 for T in [0.5, 0.9], thinking tables/0.15 for
        T in [0.95, 1.05], else p-min 0.2."""
        t = np.float32(g[0])
        if np.float32(0.5) <= t <= np.float32(0.9):
            return self.pol_tables(A, *PROD_CHAT, 0.125), "chat tables, theta 0.125"
        if np.float32(0.95) <= t <= np.float32(1.05):
            return self.pol_tables(A, *PROD_THINK, 0.15), "thinking tables, theta 0.15"
        return self.pol_pmin(A, 0.2), "p-min 0.2"


SETS = {"chat": ["cfchat"], "think": ["cfthink"], "pooled": ["cfchat", "cfthink"]}


def available():
    """the sets whose traces are complete (cfthink: window13.log says done)."""
    log = f"{cfload.RUNS}/window13.log"
    done = os.path.exists(log) and "window13 done" in open(log).read()
    return ["chat", "think", "pooled"] if done else ["chat"]
