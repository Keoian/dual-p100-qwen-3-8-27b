# sim.py: policy simulation over trace cycles (shared by cost.py, policies.py, lenient.py).
# A policy picks, per cycle, k = drafted tokens kept (0..N) and the number of draft steps decoded, at a verify width
# W = N+1 that is fixed for the server process (padded verify). Expected committed tokens E = 1 + sum_{j<=k} prod_{i<=j} a_i,
# expected time T from the cost model, throughput = tokens / time aggregated as ratio of sums (BASELINES' committed t/s).
import json, os, numpy as np
from load import CACHE, DMAX

_P = None


def params():
    global _P
    if _P is None:
        _P = json.load(open(os.path.join(CACHE, "cost.json")))
    return _P


def survival(c, key="a"):
    a = np.nan_to_num(c[key][:, :DMAX])
    valid = np.arange(DMAX)[None, :] < c["n_real"][:, None]
    a = np.where(valid, a, 0.0)
    return np.cumprod(a, axis=1)


def expected_tokens(S, k):
    """E = 1 + sum_{j<k} S[:, j] for per-cycle k (array)."""
    cs = np.c_[np.zeros(len(S)), np.cumsum(S, axis=1)]
    return 1.0 + cs[np.arange(len(S)), np.minimum(k, S.shape[1])]


def rule_stop(stop_mask, c, nmax):
    """Generic 'look' stop rule: step d (0-based) is decoded, then stop_mask[:, d] True means its token is NOT added.
    Returns (k, steps). Cut cycles (n_real < nmax) keep at most n_real tokens and decode nmax steps."""
    n = len(c["n_real"])
    sm = np.asarray(stop_mask[:, :nmax], bool)
    nr = np.minimum(c["n_real"], nmax)
    valid = np.arange(nmax)[None, :] < nr[:, None]
    hit = sm & valid
    first = np.where(hit.any(1), hit.argmax(1), nmax)
    k = np.minimum(first, nr)
    steps = np.where(first < nr, first + 1, nmax)
    return k, steps


def rule_pmin(c, nmax, pmin):
    return rule_stop(np.nan_to_num(c["d_p1"], nan=1.0) < pmin, c, nmax)


def rule_fixed(c, k):
    kk = np.minimum(c["n_real"], k)
    return kk, np.full(len(kk), k)


def cost(steps, W, n_act, ctx_k):
    """Vectorised cycle time (us). W may be scalar or array; ctx_k = pos0/1000."""
    p = params()
    W = np.broadcast_to(np.asarray(W), np.shape(steps)).astype(int)
    V = np.array([p["V"][str(w)] for w in W.ravel()]).reshape(W.shape)
    s = np.where((W >= 2) & (W <= 8), p["s"], 0.0)
    kc = np.where(W <= 5, p["k5"] * W / 5.0, p["k9"] * W / 9.0)
    oth = p["o5"] + (p["o9"] - p["o5"]) * (W - 5) / 4.0
    return p["d0"] * (steps > 0) + p["d1"] * steps + p["dctx"] * ctx_k * steps + V - s * (W - n_act) + kc * ctx_k + oth


def evaluate(c, k, steps, W, S=None, extra_us=None):
    """Per-cycle expected tokens and time (us) for a policy."""
    if S is None:
        S = survival(c)
    E = expected_tokens(S, k)
    T = cost(steps, W, k + 1, c["pos0"] / 1000.0)
    if extra_us is not None:
        T = T + extra_us
    return E, T


def per_prompt(c, E, T, mask):
    """Aggregate per prompt id: (sumE, sumT, N tokens in trace, cycles)."""
    pid = c["pid"][mask]
    n = int(c["pid"].max()) + 1
    return (np.bincount(pid, E[mask], n), np.bincount(pid, T[mask], n),
            np.bincount(pid, c["n_emit"][mask].astype(float), n), np.bincount(pid, minlength=n).astype(float))


def rate_from(agg, w=None):
    """tokens/s: N tokens at sumT/sumE us per token plus t_req per request; w = prompt weights (bootstrap counts)."""
    sE, sT, N, ncyc = agg
    has = ncyc > 0
    if w is None:
        w = has.astype(float)
    tr = params()["t_req"]
    time = (w * N * np.where(has, sT / np.maximum(sE, 1e-9), 0)).sum() + (w * has).sum() * tr
    return (w * N).sum() / time * 1e6


def rate(c, E, T, mask):
    return rate_from(per_prompt(c, E, T, mask))


def boot_gain(c, pol, base, mask, B=4000, seed=0):
    """Paired bootstrap over prompts (resample prompt ids within the mask with replacement).
    pol, base = (E, T). Returns point gain, (lo, hi) 95% percentile CI, as fractions."""
    a1 = per_prompt(c, pol[0], pol[1], mask); a0 = per_prompt(c, base[0], base[1], mask)
    ids = np.where(a1[3] > 0)[0]
    g = rate_from(a1) / rate_from(a0) - 1
    rng = np.random.default_rng(seed)
    gs = np.empty(B)
    n = int(a1[3].shape[0])
    for b in range(B):
        w = np.bincount(rng.choice(ids, len(ids)), minlength=n).astype(float)
        gs[b] = rate_from(a1, w) / rate_from(a0, w) - 1
    return g, np.percentile(gs, [2.5, 97.5])


def boot_rate(c, pol, mask, B=4000, seed=0):
    a1 = per_prompt(c, pol[0], pol[1], mask)
    ids = np.where(a1[3] > 0)[0]
    rng = np.random.default_rng(seed)
    n = int(a1[3].shape[0])
    rs = [rate_from(a1, np.bincount(rng.choice(ids, len(ids)), minlength=n).astype(float)) for _ in range(B)]
    return rate_from(a1), np.percentile(rs, [2.5, 97.5])


def oracle(c, N, W, S=None, iters=50):
    """Per-cycle best k in 0..N given the true a's, steps = k (an oracle needs no look-ahead step).
    The objective is rate_from's: sum_p N_p / sum_p N_p*sT_p/sE_p, so each cycle maximises E - R_p*T with R_p its
    prompt's own rate (Dinkelbach, per prompt). Returns k, steps."""
    if S is None:
        S = survival(c)
    n = len(S)
    ks = np.arange(N + 1)
    E = np.stack([expected_tokens(S, np.full(n, kk)) for kk in ks], 1)
    T = np.stack([cost(np.full(n, kk), W, np.full(n, kk + 1), c["pos0"] / 1000.0) for kk in ks], 1)
    pid = c["pid"]; npid = int(pid.max()) + 1
    k = np.full(n, min(N, 2))
    for _ in range(iters):
        e = E[np.arange(n), k]; t = T[np.arange(n), k]
        Rp = np.bincount(pid, e, npid) / np.maximum(np.bincount(pid, t, npid), 1e-9)
        kn = (E - Rp[pid][:, None] * T).argmax(1)
        if (kn == k).all():
            break
        k = kn
    return k, k.copy()
