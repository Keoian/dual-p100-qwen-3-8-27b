# policies.py (brief Q3): simulated stop policies on held-out prompts. SIMULATION, not measurement.
# Fit on dev (heads), choose thresholds / settings on val, report on test; paired bootstrap over test prompts.
# States: trace-n4p0 for policies with n-max <= 4 (closest to on-policy; production replays within +-2% of BASELINES),
# trace-n8p0 for every n-max up to 8. Gains are always paired on the same states.
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import sim, heads as HD, features as FT
from load import CATS

DEV, VAL, TEST = 0, 1, 2
PMINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
THETAS = np.round(np.arange(0.0, 0.951, 0.025), 3)
REBUILD_US = 20000.0  # verify-graph rebuild when the width changes (RECONNAISSANCE §1.4: 16-26 ms)


class Ctx:
    def __init__(self, run, Nmax):
        self.run = run; self.Nmax = Nmax
        self.c, self.F, self.G = HD.data(run)
        self.S = sim.survival(self.c)
        self.split = self.c["split"]
        self.m = {s: self.split == s for s in (DEV, VAL, TEST)}
        self.m["devval"] = self.split < 2

    def ev(self, k, steps, W, extra=None):
        return sim.evaluate(self.c, k, steps, W, self.S, extra)

    def rate(self, pol, mask):
        return sim.rate(self.c, pol[0], pol[1], self.m[mask] if isinstance(mask, (int, str)) else mask)


def fixed_policies(X, Ns):
    out = {}
    for N in Ns:
        for pm in PMINS:
            k, st = sim.rule_pmin(X.c, N, pm)
            out[(N, pm)] = X.ev(k, st, N + 1)
    return out


def threshold_policy(X, score, N, theta):
    """stop at the first step d whose monotone score[:, d] < theta (token d not added)."""
    k, st = sim.rule_stop(np.nan_to_num(score[:, :N], nan=0.0) < theta, X.c, N)
    return X.ev(k, st, N + 1)


def tune_threshold(X, score, Ns, sel=VAL):
    best = None
    for N in Ns:
        for th in THETAS:
            pol = threshold_policy(X, score, N, th)
            r = X.rate(pol, sel)
            if best is None or r > best[0]:
                best = (r, N, th, pol)
    return best


def per_group_pmin(X, groups, Ns, sel, cap=None):
    """Per-group p-min (and depth cap) at a common n-max N: groups = int array per cycle. Chosen on sel prompts,
    each group independently (the objective is additive over cycles at a fixed rate; we pick per group by its own rate)."""
    best = None
    for N in Ns:
        k = np.zeros(len(X.c["task"]), int); st = np.zeros(len(k), int); choice = {}
        for g in np.unique(groups):
            gm = groups == g
            bestg = None
            for pm in PMINS:
                for capg in (range(1, N + 1) if cap else [N]):
                    kk, ss = sim.rule_pmin(X.c, capg, pm)
                    ss = np.where(kk < capg, ss, capg)
                    E, T = X.ev(kk, ss, N + 1)
                    mm = gm & (X.m[sel] if isinstance(sel, (int, str)) else sel)
                    if mm.sum() == 0:
                        continue
                    r = E[mm].sum() / T[mm].sum()
                    if bestg is None or r > bestg[0]:
                        bestg = (r, pm, capg, kk, ss)
            if bestg is None:
                kk, ss = sim.rule_pmin(X.c, N, 0.2); bestg = (0, 0.2, N, kk, ss)
            choice[int(g)] = (bestg[1], bestg[2])
            k[gm] = bestg[3][gm]; st[gm] = bestg[4][gm]
        pol = X.ev(k, st, N + 1)
        r = X.rate(pol, sel)
        if best is None or r > best[0]:
            best = (r, N, choice, pol)
    return best


def per_category_width(X, sel, Ns):
    """Oracle Task-Type head with per-request width: each category gets its own (n-max, p-min); +1 rebuild/request."""
    c = X.c
    k = np.zeros(len(c["task"]), int); st = np.zeros(len(k), int); W = np.zeros(len(k), int); choice = {}
    first = (c["cyc"] == 0).astype(float)
    for g in range(len(CATS)):
        gm = c["cat"] == g
        bestg = None
        for N in Ns:
            for pm in PMINS:
                kk, ss = sim.rule_pmin(c, N, pm)
                E, T = X.ev(kk, ss, N + 1, extra=first * REBUILD_US)
                mm = gm & X.m[sel]
                r = E[mm].sum() / T[mm].sum()
                if bestg is None or r > bestg[0]:
                    bestg = (r, N, pm, kk, ss)
        choice[CATS[g]] = (bestg[1], bestg[2])
        k[gm] = bestg[3][gm]; st[gm] = bestg[4][gm]; W[gm] = bestg[1] + 1
    return choice, X.ev(k, st, W, extra=first * REBUILD_US)


def report_line(X, lab, pol, base_prod, base_fixed, base_hind, note=""):
    r, (lo, hi) = sim.boot_rate(X.c, pol, X.m[TEST])
    g1, ci1 = sim.boot_gain(X.c, pol, base_prod, X.m[TEST])
    g2, ci2 = sim.boot_gain(X.c, pol, base_fixed, X.m[TEST])
    g3, ci3 = sim.boot_gain(X.c, pol, base_hind, X.m[TEST])
    print(f"| {lab} | {r:.2f} | {g1 * 100:+.1f}% [{ci1[0] * 100:+.1f}, {ci1[1] * 100:+.1f}] | {g2 * 100:+.1f}% [{ci2[0] * 100:+.1f}, {ci2[1] * 100:+.1f}] "
          f"| {g3 * 100:+.1f}% [{ci3[0] * 100:+.1f}, {ci3[1] * 100:+.1f}] | {note} |", flush=True)
    return g1, ci1


def run(trace, Ns, heads):
    X = Ctx(trace, max(Ns))
    c = X.c
    print(f"\n# states: trace-{trace}; n-max choices {Ns}; test = {len(np.unique(c['pid'][X.m[TEST]]))} prompts, "
          f"{X.m[TEST].sum()} cycles (val {X.m[VAL].sum()}, dev {X.m[DEV].sum()})")
    fx = fixed_policies(X, range(1, max(Ns) + 1))
    prod = fx[(4, 0.2)]
    # best fixed: chosen on val among all (n-max, p-min) with n-max <= max(Ns); hindsight: best on test itself
    key_val = max(fx, key=lambda k: X.rate(fx[k], VAL))
    key_dv = max(fx, key=lambda k: X.rate(fx[k], "devval"))
    key_hind = max(fx, key=lambda k: X.rate(fx[k], TEST))
    bfix, bhind = fx[key_val], fx[key_hind]
    print(f"best fixed on val: n-max {key_val[0]} p-min {key_val[1]}; on dev+val: {key_dv}; hindsight best on test: {key_hind}")
    print("\n| policy (SIMULATED, test prompts) | t/s [95% CI] | vs production (n4 p0.2) | vs best fixed (val-chosen) | vs hindsight-best fixed on test | chosen on val |")
    print("|---|---|---|---|---|---|")
    report_line(X, "production n-max 4 p-min 0.2", prod, prod, bfix, bhind)
    report_line(X, f"best fixed (val): n-max {key_val[0]} p-min {key_val[1]}", bfix, prod, bfix, bhind)
    report_line(X, f"hindsight best fixed on test: n-max {key_hind[0]} p-min {key_hind[1]}", bhind, prod, bfix, bhind)
    for N in range(1, max(Ns) + 1):
        report_line(X, f"fixed depth {N} (p-min 0)", fx[(N, 0.0)], prod, bfix, bhind)
    for N in sorted({4, max(Ns)}):
        for pm in [0.0, 0.1, 0.2, 0.4, 0.6]:
            report_line(X, f"p-min rule n-max {N} p-min {pm}", fx[(N, pm)], prod, bfix, bhind)
    res = {}  # label -> (policy, val rate, realistic?)

    def add(lab, pol, note="", realistic=True):
        res[lab] = (pol, X.rate(pol, VAL), realistic)
        report_line(X, lab, pol, prod, bfix, bhind, note)

    # ---- exact-safe cheap heuristics (decision on token d uses step d's distribution and earlier tokens only)
    pcum = np.exp(X.F["cum_logp1"])
    iso_q = HD.isotonic_q(); iso_p = HD.isotonic_q(key="d_p1")
    shp = X.c["d_q"].shape
    post_iso = np.where(np.isnan(X.c["d_q"]), 0.0, iso_q.predict(np.nan_to_num(X.c["d_q"]).ravel()).reshape(shp))
    prior_iso = np.where(np.isnan(X.c["d_p1"]), 0.0, iso_p.predict(np.nan_to_num(X.c["d_p1"]).ravel()).reshape(shp))
    ex_iso = HD.exact_chain(prior_iso, post_iso)
    for lab, sc in [("p_cum = prod top-1 p (heuristic)", pcum),
                    ("iso chain: prod_{i<d} iso(q_i) x iso(p1_d), 1-D lookups fit on dev (heuristic)", ex_iso)]:
        b = tune_threshold(X, sc, Ns)
        add(lab, b[3], f"n-max {b[1]}, theta {b[2]}")
    # rolling acceptance proxy: p-min by EMA of first-token acceptance (3 bins)
    ema = X.G["ema_acc1"]; gr = np.digitize(ema, [0.6, 0.85])
    b = per_group_pmin(X, gr, Ns, VAL, cap=True)
    add("rolling-acceptance p-min/cap (3 EMA bins)", b[3], f"n-max {b[1]}, {b[2]}")
    # realistic task-type proxy: structured (fence / JSON / tool markup) vs prose
    struct = ((X.G["in_fence"] + X.G["json_mode"] + X.G["tool_mode"]) > 0).astype(int)
    b = per_group_pmin(X, struct, Ns, VAL, cap=True)
    add("task-type PROXY (structured vs prose) p-min/cap", b[3], f"n-max {b[1]}, {b[2]}")
    best = None
    for N in Ns:
        ths = {}
        k = np.zeros(len(c["task"]), int); st = np.zeros(len(k), int)
        for g in (0, 1):
            gm = struct == g; bg = None
            for th in THETAS:
                kk, ss = sim.rule_stop(ex_iso[:, :N] < th, c, N); E, T = X.ev(kk, ss, N + 1)
                mm = gm & X.m[VAL]; r = E[mm].sum() / T[mm].sum()
                if bg is None or r > bg[0]:
                    bg = (r, th, kk, ss)
            ths[g] = float(bg[1]); k[gm] = bg[2][gm]; st[gm] = bg[3][gm]
        pol = X.ev(k, st, N + 1); r = X.rate(pol, VAL)
        if best is None or r > best[0]:
            best = (r, N, ths, pol)
    add("task-type PROXY x iso chain, theta per state (prose, structured)", best[3], f"n-max {best[1]}, theta {best[2]}")
    # ---- oracle task-type head: category labels
    b = per_group_pmin(X, c["cat"], Ns, VAL, cap=True)
    add("ORACLE category p-min/cap, common width (chosen on val)", b[3],
        f"n-max {b[1]}, " + ", ".join(f"{CATS[g]}:{v}" for g, v in b[2].items()), realistic=False)
    b = per_group_pmin(X, c["cat"], Ns, "devval", cap=True)
    add("ORACLE category p-min/cap, common width (chosen on dev+val)", b[3],
        f"n-max {b[1]}, " + ", ".join(f"{CATS[g]}:{v}" for g, v in b[2].items()), realistic=False)
    ch, pol = per_category_width(X, "devval", range(1, max(Ns) + 1))
    add("ORACLE category (n-max, p-min) per request, +20 ms rebuild", pol, ", ".join(f"{k}:{v}" for k, v in ch.items()), realistic=False)
    # ---- learned Fidelity heads, exact-safe: prior model at step d x token-aware model for earlier tokens
    Sh = {}
    for lab, fprior, fpost, kind in heads:
        hp = HD.Head(fprior, kind, cond=True); hq = HD.Head(fpost, kind, cond=True)
        Shat = HD.exact_chain(hp.predict_grid(trace, X.Nmax), hq.predict_grid(trace, X.Nmax)); Sh[lab] = Shat
        b = tune_threshold(X, Shat, Ns)
        add(lab, b[3], f"n-max {b[1]}, theta {b[2]}", realistic="ORACLE" not in lab)
    lab0 = [l for l in Sh if "realistic" in l and "DIFF" not in l and "ORACLE" not in l][0]
    # oracle category x exact realistic head: per-category (n-max, theta), per-request width
    first = (c["cyc"] == 0).astype(float)
    k = np.zeros(len(c["task"]), int); st = np.zeros(len(k), int); W = np.zeros(len(k), int); ch = {}
    for g in range(len(CATS)):
        gm = c["cat"] == g; bg = None
        for N in Ns:
            for th in THETAS:
                kk, ss = sim.rule_stop(Sh[lab0][:, :N] < th, c, N); E, T = X.ev(kk, ss, N + 1, extra=first * REBUILD_US)
                mm = gm & X.m[VAL]; r = E[mm].sum() / T[mm].sum()
                if bg is None or r > bg[0]:
                    bg = (r, N, th, kk, ss)
        ch[CATS[g]] = (bg[1], float(bg[2])); k[gm] = bg[3][gm]; st[gm] = bg[4][gm]; W[gm] = bg[1] + 1
    add("ORACLE category x exact realistic head: per-category (n-max, theta), per-request width", X.ev(k, st, W, extra=first * REBUILD_US),
        ", ".join(f"{a}:{v}" for a, v in ch.items()), realistic=False)
    # ---- NOT EXACT (look at the sampled token's own q; would change the output distribution), for reference only
    qd = np.nan_to_num(X.c["d_q"], nan=1.0)
    best = None
    for N in Ns:
        for th in THETAS:
            k, st = sim.rule_stop(qd < th, c, N); pol = X.ev(k, st, N + 1); r = X.rate(pol, VAL)
            if best is None or r > best[0]:
                best = (r, N, th, pol)
    add("NOT EXACT q-min: stop when the sampled token's q < theta", best[3], f"n-max {best[1]}, theta {best[2]}", realistic=False)
    b = tune_threshold(X, np.cumprod(post_iso, axis=1), Ns)
    add("NOT EXACT prod iso(q_i) through d", b[3], f"n-max {b[1]}, theta {b[2]}", realistic=False)
    hq = HD.Head("realistic", "hgb", cond=True)
    b = tune_threshold(X, HD.chain(hq.predict_grid(trace, X.Nmax)), Ns)
    add("NOT EXACT HGB realistic token-aware chain", b[3], f"n-max {b[1]}, theta {b[2]}", realistic=False)
    # ---- oracles: per-cycle best depth given the true a's, at each fixed width (an upper bound for any stop rule)
    for N in Ns:
        k, st = sim.oracle(c, N, N + 1, X.S)
        add(f"ORACLE per-cycle depth, n-max {N} (W {N + 1}), true a's", X.ev(k, st, N + 1), realistic=False)
        st2 = np.where(k < np.minimum(c["n_real"], N), k + 1, k)
        add(f"ORACLE per-cycle depth, n-max {N}, paying the look-ahead step", X.ev(k, st2, N + 1), realistic=False)
    # ---- headline: the realistic (exact-safe, label-free) policy with the best VAL rate, reported on test
    cand = {l: v for l, v in res.items() if v[2]}
    cand["best fixed (val)"] = (bfix, X.rate(bfix, VAL), True)
    top = sorted(cand, key=lambda l: -cand[l][1])
    print("\nrealistic exact-safe policies ranked by VAL rate: " + "; ".join(f"{l} {cand[l][1]:.2f}" for l in top[:7]))
    bl = top[0]
    g, ci = sim.boot_gain(c, cand[bl][0], prod, X.m[TEST])
    g2, ci2 = sim.boot_gain(c, cand[bl][0], bfix, X.m[TEST])
    g3, ci3 = sim.boot_gain(c, cand[bl][0], bhind, X.m[TEST])
    print(f"HEADLINE ({trace}, n-max in {Ns}): best realistic policy on val = {bl}: test gain vs production {g * 100:+.1f}% "
          f"[{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}], vs best fixed (val) {g2 * 100:+.1f}% [{ci2[0] * 100:+.1f}, {ci2[1] * 100:+.1f}], "
          f"vs hindsight-best fixed {g3 * 100:+.1f}% [{ci3[0] * 100:+.1f}, {ci3[1] * 100:+.1f}]")
    hb = "iso chain: prod_{i<d} iso(q_i) x iso(p1_d), 1-D lookups fit on dev (heuristic)"
    pairs_ = [(l, hb) for l in Sh if "ORACLE" not in l]
    pairs_ += [("FIDELITY+DIFFICULTY (exact): HGB +prev-cycle teacher", lab0),
               ("FIDELITY+ORACLE category (exact): HGB +category", lab0),
               ("ORACLE category x exact realistic head: per-category (n-max, theta), per-request width", lab0),
               ("task-type PROXY x iso chain, theta per state (prose, structured)", hb),
               (hb, "p_cum = prod top-1 p (heuristic)"),
               ("NOT EXACT HGB realistic token-aware chain", lab0)]
    for a_, b_ in pairs_:
        if a_ == b_:
            continue
        g, ci = sim.boot_gain(c, res[a_][0], res[b_][0], X.m[TEST])
        print(f"  paired: {a_}  vs  {b_}: {g * 100:+.1f}% [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]")
    return X, res


if __name__ == "__main__":
    heads = [("FIDELITY (exact): LR draft features", "prior", "draft", "lr"),
             ("FIDELITY (exact): HGB draft features", "prior", "draft", "hgb"),
             ("FIDELITY (exact): HGB realistic (draft+rolling+proxy)", "prior+roll+proxy", "realistic", "hgb"),
             ("FIDELITY+DIFFICULTY (exact): HGB +prev-cycle teacher", "prior+roll+proxy+diff", "realistic+diff", "hgb"),
             ("FIDELITY+ORACLE category (exact): HGB +category", "prior+roll+proxy+cat", "realistic+cat", "hgb")]
    which = sys.argv[1:] or ["n4p0", "n8p0"]
    for tr in which:
        if tr == "n4p0":
            run("n4p0", [2, 3, 4], heads)
        elif tr == "n8p0":
            run("n8p0", [2, 3, 4, 5, 6, 7, 8], heads)
        elif tr == "n8p0-le4":
            run("n8p0", [2, 3, 4], heads)
