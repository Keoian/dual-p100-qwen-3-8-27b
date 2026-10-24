# nmax_smoke.py: would the lookup stop rule pay with a deeper cap (n-max 5-7, verify width 6-8, inside the
# inactive-column skip's width limit)? Simulated on the n-max 8 traces (acceptance doesn't depend on the cap), theta
# tuned on val per cap, test reported. Then the break-even: the per-inactive-column saving s (cost model: 3.97 ms;
# a width step costs ~8.7 ms) at which a deeper cap beats n-max 4.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import sim, heads as HD, policies as PO
X = PO.Ctx("n8p0", 8)
iso_q = HD.isotonic_q(); iso_p = HD.isotonic_q(key="d_p1")
shp = X.c["d_q"].shape
post = np.where(np.isnan(X.c["d_q"]), 0.0, iso_q.predict(np.nan_to_num(X.c["d_q"]).ravel()).reshape(shp))
prior = np.where(np.isnan(X.c["d_p1"]), 0.0, iso_p.predict(np.nan_to_num(X.c["d_p1"]).ravel()).reshape(shp))
score = HD.exact_chain(prior, post)
P = sim.params(); s0 = P["s"]
V = {w: P["V"][str(w)] for w in range(2, 10)}
print(f"set {os.environ.get('MTP3_TRACESET', 'chat')}: cost model V(W) ms " + " ".join(f"{w}:{V[w]/1000:.1f}" for w in range(4, 10))
      + f"; skip saves {s0/1000:.2f} ms per inactive column")
def best(N):
    return PO.tune_threshold(X, score, [N])
prod = PO.fixed_policies(X, [4])[(4, 0.2)]
r_prod = X.rate(prod, PO.TEST)
for s in [s0, 5000.0, 6000.0, 7000.0, 8000.0, 8700.0]:
    P["s"] = s
    rows = []
    for N in (4, 5, 6, 7):
        r, n, th, pol = best(N)
        rows.append((N, th, X.rate(pol, PO.TEST)))
    base = rows[0][2]
    print(f"  s {s/1000:4.2f} ms: " + "  ".join(f"n-max {N} (theta {th}) {100*(t/r_prod-1):+5.1f}%" for N, th, t in rows)
          + "  [vs production p-min 0.2, n-max 4; test]")
P["s"] = s0
# Best case for engineering: every round costs what an unpadded batch of exactly its real width costs (variable-width
# verify, or a perfect skip at any width): V(n_act + 1) instead of V(W) - s*(W - n_act).
_cost0 = sim.cost
def cost_exact(steps, W, n_act, ctx_k):
    base = _cost0(steps, W, n_act, ctx_k)
    Wn = np.broadcast_to(np.asarray(W), np.shape(steps)).astype(int)
    na = np.asarray(n_act).astype(int)
    Vpad = np.array([P["V"][str(w)] for w in Wn.ravel()]).reshape(Wn.shape) - np.where((Wn >= 2) & (Wn <= 8), s0, 0.0) * (Wn - na)
    Vex = np.array([P["V"][str(max(2, w))] for w in np.minimum(na, 9).ravel()]).reshape(na.shape)
    return base - Vpad + Vex
sim.cost = cost_exact
rows = []
for N in (4, 5, 6, 7, 8):
    r, n, th, pol = best(N)
    rows.append((N, th, X.rate(pol, PO.TEST)))
print("  exact-width verify (best case): " + "  ".join(f"n-max {N} (theta {th}) {100*(t/r_prod-1):+5.1f}%" for N, th, t in rows))
sim.cost = _cost0
