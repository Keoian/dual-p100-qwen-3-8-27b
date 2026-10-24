# sens.py: how much do the headline comparisons depend on the cost model? Re-tune thresholds on val under each
# perturbed model and report the test gain vs production (n-max 4, p-min 0.2). trace-n4p0 states, n-max 2..4.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import copy, numpy as np
import sim, heads as HD
import policies as PO

TEST, VAL = 2, 1


def main():
    X = PO.Ctx("n4p0", 4)
    c = X.c
    hp = HD.Head("prior+roll+proxy", "hgb", cond=True); hq = HD.Head("realistic", "hgb", cond=True)
    Sh = HD.exact_chain(hp.predict_grid("n4p0", 4), hq.predict_grid("n4p0", 4))
    iso_q = HD.isotonic_q(); iso_p = HD.isotonic_q(key="d_p1"); shp = c["d_q"].shape
    post = np.where(np.isnan(c["d_q"]), 0.0, iso_q.predict(np.nan_to_num(c["d_q"]).ravel()).reshape(shp))
    prior = np.where(np.isnan(c["d_p1"]), 0.0, iso_p.predict(np.nan_to_num(c["d_p1"]).ravel()).reshape(shp))
    ex_iso = HD.exact_chain(prior, post)
    base = copy.deepcopy(sim.params())
    Ns = [2, 3, 4]
    print("| cost model | best fixed (val) vs prod | iso chain vs prod | exact HGB head vs prod | oracle n4 vs prod | prod t/s |")
    print("|---|---|---|---|---|---|")
    for lab, mod in [("as fitted", {}), ("s x 0.5", {"s": 0.5}), ("s x 0.75", {"s": 0.75}), ("s x 1.25", {"s": 1.25}),
                     ("s = 0 (no partial verify)", {"s": 0.0}), ("draft step d1 x 0.75", {"d1": 0.75}), ("draft step d1 x 1.25", {"d1": 1.25}),
                     ("V(W) all +10%", {"V": 1.1}), ("V(W) all -10%", {"V": 0.9})]:
        p = copy.deepcopy(base)
        for k, f in mod.items():
            if k == "V":
                p["V"] = {w: v * f for w, v in p["V"].items()}
            else:
                p[k] = p[k] * f
        sim._P = p
        fx = PO.fixed_policies(X, range(1, 5))
        prod = fx[(4, 0.2)]
        kv = max(fx, key=lambda k: X.rate(fx[k], VAL))
        out = [sim.boot_gain(c, fx[kv], prod, X.m[TEST], B=1000)]
        for sc in (ex_iso, Sh):
            b = PO.tune_threshold(X, sc, Ns)
            out.append(sim.boot_gain(c, b[3], prod, X.m[TEST], B=1000))
        k, st = sim.oracle(c, 4, 5, X.S)
        out.append(sim.boot_gain(c, X.ev(k, st, 5), prod, X.m[TEST], B=1000))
        print(f"| {lab} | " + " | ".join(f"{g * 100:+.1f}% [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]" for g, ci in out)
              + f" | {X.rate(prod, TEST):.2f} | (best fixed: {kv})")
    sim._P = base
    # alternative estimator: position-weighted (each trace cycle stands for the n_emit tokens it covered; time per token
    # T/E of the policy at that state). It brackets the ratio estimator in the BASELINES replay (see calib).
    fx = PO.fixed_policies(X, range(1, 5)); prod = fx[(4, 0.2)]
    w = c["n_emit"].astype(float)
    pw = lambda pol: (w, w * pol[1] / pol[0])
    kv = max(fx, key=lambda k: sim.rate(c, *pw(fx[k]), X.m[VAL]))
    rows = [("best fixed (val, position-weighted)", fx[kv])]
    for lab, sc in (("iso chain", ex_iso), ("exact HGB head", Sh)):
        best = None
        for N in Ns:
            for th in PO.THETAS:
                pol = PO.threshold_policy(X, sc, N, th); r = sim.rate(c, *pw(pol), X.m[VAL])
                if best is None or r > best[0]:
                    best = (r, N, th, pol)
        rows.append((f"{lab} (n-max {best[1]}, theta {best[2]}, tuned position-weighted)", best[3]))
    k, st = sim.oracle(c, 4, 5, X.S); rows.append(("oracle n4", X.ev(k, st, 5)))
    print(f"\nposition-weighted estimator, test, vs production ({sim.rate(c, *pw(prod), X.m[TEST]):.2f} t/s):")
    for lab, pol in rows:
        g, ci = sim.boot_gain(c, pw(pol), pw(prod), X.m[TEST], B=1000)
        print(f"  {lab}: {g * 100:+.1f}% [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}] (best fixed = {kv})")
    # what the headline policy does: depth distribution vs production, test prompts
    fx = PO.fixed_policies(X, range(1, 5)); prod = fx[(4, 0.2)]
    b = PO.tune_threshold(X, Sh, Ns)
    k, st = sim.rule_stop(np.nan_to_num(Sh[:, :b[1]], nan=0.0) < b[2], c, b[1])
    kp, stp = sim.rule_pmin(c, 4, 0.2)
    m = X.m[TEST]
    print(f"\nexact HGB head (n-max {b[1]}, theta {b[2]}) on test: drafted tokens kept per cycle {k[m].mean():.2f} (production {kp[m].mean():.2f}); "
          f"depth histogram {np.bincount(k[m], minlength=5) / m.sum()} (production {np.bincount(kp[m], minlength=5) / m.sum()})")
    E, T = b[3]; E0, T0 = prod
    print(f"  expected tokens/cycle {E[m].mean():.3f} vs {E0[m].mean():.3f}; cycle time {T[m].mean() / 1000:.2f} ms vs {T0[m].mean() / 1000:.2f} ms")
    for cn_i, cn in enumerate(["casual", "code", "creative", "json", "longctx", "reasoning", "switch", "tools"]):
        mm = m & (c["cat"] == cn_i)
        if mm.sum():
            print(f"  {cn:9s} test: head {sim.rate(c, E, T, mm):.2f} t/s vs production {sim.rate(c, E0, T0, mm):.2f} ({(sim.rate(c, E, T, mm) / sim.rate(c, E0, T0, mm) - 1) * 100:+.1f}%), kept {k[mm].mean():.2f} vs {kp[mm].mean():.2f}")


if __name__ == "__main__":
    main()
