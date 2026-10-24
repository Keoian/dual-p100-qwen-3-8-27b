# cfgrid.py [repro|grid] : CF_ANALYSIS step 2.
# repro: S = each trace's own setting reproduces production's tables (chat from cfchat, thinking from cfthink), with
#        the bootstrap band of the fit over dev prompts and the throughput of production's vs the refit tables.
# grid:  for every setting of cfcore.GRID and every available set (chat, think, pooled): own tables (dev), theta (val),
#        test rates of (i) production's policy for S, (ii) p-min 0.2, (iii) own tables, with the paired bootstrap of
#        (iii) vs (i). Writes out/cf/grid-<set>.json (tables, thetas, rates) and prints a summary.
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import cfcore as C, cfload

OUT = C.OUT


def tables_str(tq, tp):
    return " ".join(f"{v:.3f}" for v in tq), " ".join(f"{v:.3f}" for v in tp)


def fit_band(S, gi, B=200, seed=0):
    """bootstrap over dev prompts of the 24-point tables: (lo, hi) per point, 95%."""
    y = S.a_rows(gi)
    rng = np.random.default_rng(seed)
    ru = S.unit[S.rows_c]
    dev_units = np.unique(S.unit[S.split == C.DEV])
    Q, P = [], []
    for b in range(B):
        cnt = np.bincount(rng.choice(dev_units, len(dev_units)), minlength=S.nu)
        w = S.fit_w * cnt[ru]
        tq, tp = S.fit(a_rows=y, w=w)
        Q.append(tq); P.append(tp)
    Q, P = np.array(Q), np.array(P)
    return np.percentile(Q, [2.5, 97.5], 0), np.percentile(P, [2.5, 97.5], 0)


def legacy_identity(name):
    """production's own fit procedure (heads.isotonic_q: every dev cycle incl. grammar, recorded a, n-max 4 + 8
    traces) with the cf trace standing in for the n-max 4 one: does the loader give back production's tables?"""
    import load as L
    os.environ["MTP3_TRACESET"] = "chat" if name == "cfchat" else "think"
    import importlib
    importlib.reload(L)
    c8, _, _ = L.load("n8p0")
    c, _ = cfload.load(name)
    xs_q, xs_p, ys = [], [], []
    nr = np.minimum(c["n_real"], C.D)
    rc = np.repeat(np.arange(len(nr)), nr); rd = np.concatenate([np.arange(k) for k in nr])
    m = c["split"][rc] == C.DEV
    q = c["d_q"][rc, rd]; tp = c["t_p"][rc, rd]
    a = np.where(np.isnan(tp), 0.0, np.minimum(1.0, np.nan_to_num(tp) / np.where(q > 0, q, 1.0)))
    xs_q.append(q[m]); xs_p.append(c["d_p1"][rc, rd][m]); ys.append(a[m])
    nr8 = np.minimum(c8["n_real"], 8)
    rc = np.repeat(np.arange(len(nr8)), nr8); rd = np.concatenate([np.arange(k) for k in nr8])
    m = c8["split"][rc] == C.DEV
    xs_q.append(c8["d_q"][rc, rd][m]); xs_p.append(c8["d_p1"][rc, rd][m]); ys.append(np.nan_to_num(c8["a"][rc, rd][m]))
    from sklearn.isotonic import IsotonicRegression
    iq = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(np.concatenate(xs_q), np.concatenate(ys))
    ip = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(np.concatenate(xs_p), np.concatenate(ys))
    return np.round(iq.predict(C.X24), 3), np.round(ip.predict(C.X24), 3)


def repro():
    res = {}
    for sname_, name, g, prod, th_prod in [("chat", "cfchat", C.CHAT_S, C.PROD_CHAT, 0.125),
                                           ("think", "cfthink", C.THINK_S, C.PROD_THINK, 0.15)]:
        if sname_ not in C.available():
            print(f"{name}: not complete, skipped")
            continue
        S = C.Set([name])
        gi = C.GIDX[g]
        A = S.a(gi)
        tq, tp = S.fit(gi)
        (qlo, qhi), (plo, phi) = fit_band(S, gi)
        th, curve = S.choose_theta(A, tq, tp)
        lq, lp = legacy_identity(name)
        print(f"\n## {name} at its own setting {C.sname(g)} vs production's {sname_} tables")
        print(f"legacy fit (heads.isotonic_q procedure, {name} in place of the n-max 4 trace, + n-max 8 trace): "
              f"max |diff| vs production iso_q {np.abs(lq - prod[0]).max():.3f}, iso_p1 {np.abs(lp - prod[1]).max():.3f}")
        print("x      prod_q  cf_q   [95% band]       prod_p1 cf_p1  [95% band]")
        for i, x in enumerate(C.X24):
            print(f"{x:5.3f}  {prod[0][i]:.3f}   {tq[i]:.3f}  [{qlo[i]:.3f}, {qhi[i]:.3f}]   {prod[1][i]:.3f}   {tp[i]:.3f}  [{plo[i]:.3f}, {phi[i]:.3f}]")
        outq = int(((prod[0] < qlo - 1e-9) | (prod[0] > qhi + 1e-9)).sum()); outp = int(((prod[1] < plo - 1e-9) | (prod[1] > phi + 1e-9)).sum())
        print(f"max |cf - prod|: iso_q {np.abs(tq - prod[0]).max():.3f} (mean {np.abs(tq - prod[0]).mean():.3f}), "
              f"iso_p1 {np.abs(tp - prod[1]).max():.3f} (mean {np.abs(tp - prod[1]).mean():.3f}); production points outside "
              f"the cf fit's 95% band: iso_q {outq}/24, iso_p1 {outp}/24; mean band width q {np.mean(qhi - qlo):.3f}, p1 {np.mean(phi - plo):.3f}")
        th_p, curve_p = S.choose_theta(A, *prod)
        pol_prod = S.pol_tables(A, *prod, th_prod)
        pol_cf = S.pol_tables(A, tq, tp, th)
        g_, ci = S.boot_gain(pol_cf, pol_prod)
        print(f"theta on val: cf tables {th} (production {th_prod}; production's tables on this val: {th_p}); "
              f"test rate production tables {S.rate(*pol_prod, C.TEST):.2f}, cf tables {S.rate(*pol_cf, C.TEST):.2f}: "
              f"{g_ * 100:+.2f}% [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]")
        res[name] = dict(cf_q=tq.tolist(), cf_p1=tp.tolist(), theta=th, legacy_q=lq.tolist(), legacy_p1=lp.tolist(),
                         gain=g_, ci=ci.tolist())
    json.dump(res, open(f"{OUT}/repro.json", "w"), indent=1)


def grid(sets=None):
    for setname in sets or C.available():
        S = C.Set(C.SETS[setname])
        rows = []
        for gi, g in enumerate(C.GRID):
            A = S.a(gi)
            tq, tp = S.fit(gi)
            th, curve = S.choose_theta(A, tq, tp)
            pol_own = S.pol_tables(A, tq, tp, th)
            (pol_prod, prod_lab) = S.prod_policy(g, A)
            pol_pm = S.pol_pmin(A, 0.2)
            r_own, r_prod, r_pm = (S.rate(*p, C.TEST) for p in (pol_own, pol_prod, pol_pm))
            gain, ci = S.boot_gain(pol_own, pol_prod)
            Am = A[S.valid]
            rows.append(dict(g=list(g), name=C.sname(g), tq=tq.tolist(), tp=tp.tolist(), theta=th,
                             val_curve=curve.tolist(), r_own=r_own, r_prod=r_prod, r_pmin=r_pm, prod=prod_lab,
                             gain=gain, ci=ci.tolist(), mean_a=float(Am.mean()),
                             mean_a_dev=float(A[S.valid & (S.split == C.DEV)[:, None]].mean())))
            if gi % 48 == 0:
                print(f"{setname}: {gi}/{len(C.GRID)} {C.sname(g)} own {r_own:.2f} prod {r_prod:.2f} ({prod_lab}) "
                      f"pmin {r_pm:.2f} gain {gain * 100:+.2f}% theta {th}", flush=True)
        json.dump(rows, open(f"{OUT}/grid-{setname}.json", "w"))
        summarize(setname, rows)


def summarize(setname, rows=None):
    rows = rows or json.load(open(f"{OUT}/grid-{setname}.json"))
    g = np.array([r["gain"] for r in rows]); lo = np.array([r["ci"][0] for r in rows])
    pm = np.array([r["r_own"] / r["r_pmin"] - 1 for r in rows])
    print(f"\n## {setname}: own tables vs production's policy over {len(rows)} settings (test, simulated)")
    print(f"gain mean {g.mean() * 100:+.2f}%, median {np.median(g) * 100:+.2f}%, max {g.max() * 100:+.2f}%, min {g.min() * 100:+.2f}%;"
          f" settings with CI above 0: {(lo > 0).sum()}; own vs p-min 0.2 mean {pm.mean() * 100:+.2f}%")
    for lab in sorted(set(r["prod"] for r in rows)):
        m = np.array([r["prod"] == lab for r in rows])
        print(f"  production uses {lab}: {m.sum()} settings, own-table gain mean {g[m].mean() * 100:+.2f}% "
              f"(min {g[m].min() * 100:+.2f}, max {g[m].max() * 100:+.2f})")
    print("  by temperature (mean gain over the 48 settings each):",
          ", ".join(f"T{t:.1f} {np.mean([r['gain'] for r in rows if r['g'][0] == t]) * 100:+.1f}%" for t in C.TEMPS))
    print("  largest gains:")
    for i in np.argsort(-g)[:8]:
        r = rows[i]
        print(f"    {r['name']}: {r['gain'] * 100:+.2f}% [{r['ci'][0] * 100:+.1f}, {r['ci'][1] * 100:+.1f}] own {r['r_own']:.2f} "
              f"prod {r['r_prod']:.2f} ({r['prod']}) theta {r['theta']}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "grid"
    if what == "repro":
        repro()
    elif what == "grid":
        grid(sys.argv[2:] or None)
    elif what == "summary":
        for s in sys.argv[2:] or C.available():
            summarize(s)
