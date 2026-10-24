# fidelity.py (brief Q3d metrics, Q3e, Q5): how well do draft-side features predict a_d on held-out (test) prompts?
# Trained on dev cycles of both traces; evaluated on test cycles (pooled n4p0 depths 1-4 + n8p0 depths 1-8).
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.metrics import roc_auc_score
import heads as HD
from load import CATS

RUNS = (("n4p0", 4), ("n8p0", 8))


def test_pairs():
    out = []
    for run, D in RUNS:
        c, F, G = HD.data(run)
        rc, rd = HD.pairs(run, D, 2)
        out.append((run, rc, rd))
    return out


def gather(fn):
    return np.concatenate([fn(run, rc, rd) for run, rc, rd in test_pairs()])


def main(B=1000):
    a = gather(lambda run, rc, rd: np.nan_to_num(HD.data(run)[0]["a"][rc, rd]))
    pid = gather(lambda run, rc, rd: HD.data(run)[0]["pid"][rc])
    depth = gather(lambda run, rc, rd: rd + 1)
    cat = gather(lambda run, rc, rd: HD.data(run)[0]["cat"][rc])
    preds = {}
    preds["draft q (raw)"] = gather(lambda run, rc, rd: HD.data(run)[0]["d_q"][rc, rd])
    preds["draft top-1 p (raw)"] = gather(lambda run, rc, rd: HD.data(run)[0]["d_p1"][rc, rd])
    iso = HD.isotonic_q()
    preds["isotonic(q) fit on dev"] = iso.predict(preds["draft q (raw)"])
    isop = HD.isotonic_q(key="d_p1")
    preds["EXACT-SAFE isotonic(top-1 p) fit on dev"] = isop.predict(preds["draft top-1 p (raw)"])
    models = [("EXACT-SAFE LR prior", "prior", "lr"), ("EXACT-SAFE LR prior+category (oracle)", "prior+cat", "lr"),
              ("EXACT-SAFE HGB prior", "prior", "hgb"), ("EXACT-SAFE HGB prior+rolling+proxy", "prior+roll+proxy", "hgb"),
              ("EXACT-SAFE HGB prior+rolling+proxy+difficulty", "prior+roll+proxy+diff", "hgb"),
              ("EXACT-SAFE HGB prior+rolling+proxy+category (oracle)", "prior+roll+proxy+cat", "hgb"),
              ("LR draft", "draft", "lr"), ("LR draft+category (oracle)", "draft+cat", "lr"),
              ("HGB draft", "draft", "hgb"), ("HGB draft+rolling", "draft+roll", "hgb"),
              ("HGB draft+proxy", "draft+proxy", "hgb"), ("HGB draft+difficulty", "draft+diff", "hgb"),
              ("HGB realistic (draft+rolling+proxy)", "realistic", "hgb"),
              ("HGB realistic+difficulty", "realistic+diff", "hgb"),
              ("HGB realistic+category (oracle)", "realistic+cat", "hgb"),
              ("HGB draft+category (oracle)", "draft+cat", "hgb")]
    for lab, fs, kind in models:
        h = HD.Head(fs, kind, cond=False)
        preds[lab] = gather(lambda run, rc, rd, h=h: h.predict_pairs(run, rc, rd))
    print(f"## per-token prediction of a_d on test prompts ({len(a)} (cycle, depth) pairs, {len(np.unique(pid))} prompts)")
    print("| predictor | AUC (a_d > 0.5) | Brier vs a_d | soft log-loss | mean pred (mean a = %.3f) | Brier diff vs isotonic(q), 95%% CI |" % a.mean())
    print("|---|---|---|---|---|---|")
    ids = np.unique(pid)
    rng = np.random.default_rng(1)
    boots = [rng.choice(ids, len(ids)) for _ in range(B)]
    idx_of = {p: np.where(pid == p)[0] for p in ids}
    ref = preds["isotonic(q) fit on dev"]
    for lab, pr in preds.items():
        m = HD.token_metrics(pr, a)
        d = []
        for bs in boots:
            ii = np.concatenate([idx_of[p] for p in bs])
            d.append(np.mean((pr[ii] - a[ii]) ** 2) - np.mean((ref[ii] - a[ii]) ** 2))
        lo, hi = np.percentile(d, [2.5, 97.5])
        dd = np.mean((pr - a) ** 2) - np.mean((ref - a) ** 2)
        print(f"| {lab} | {m['auc']:.3f} | {m['brier']:.4f} | {m['logloss']:.4f} | {m['mean_pred']:.3f} | {dd:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    # by depth for q vs realistic
    print("\n## AUC by depth (test): raw q / isotonic(q) / HGB realistic / HGB realistic+category / EXACT-SAFE top-1 p / EXACT-SAFE HGB prior+rolling+proxy")
    for d in range(1, 9):
        m = depth == d
        y = (a[m] > 0.5).astype(int)
        if len(np.unique(y)) < 2:
            continue
        print(f"  d={d} n={m.sum():5d}: " + " / ".join(f"{roc_auc_score(y, preds[k][m]):.3f}" for k in
              ["draft q (raw)", "isotonic(q) fit on dev", "HGB realistic (draft+rolling+proxy)", "HGB realistic+category (oracle)",
               "draft top-1 p (raw)", "EXACT-SAFE HGB prior+rolling+proxy"]))
    print("\n## AUC within category (test): isotonic(q) / HGB realistic / EXACT-SAFE top-1 p / EXACT-SAFE HGB prior+rolling+proxy")
    for k, cn in enumerate(CATS):
        m = cat == k
        y = (a[m] > 0.5).astype(int)
        if m.sum() < 30 or len(np.unique(y)) < 2:
            continue
        print(f"  {cn:9s} n={m.sum():5d}: {roc_auc_score(y, preds['isotonic(q) fit on dev'][m]):.3f} / {roc_auc_score(y, preds['HGB realistic (draft+rolling+proxy)'][m]):.3f}"
              f" / {roc_auc_score(y, preds['draft top-1 p (raw)'][m]):.3f} / {roc_auc_score(y, preds['EXACT-SAFE HGB prior+rolling+proxy'][m]):.3f}"
              f"   mean a {a[m].mean():.3f}  mean iso(q) {preds['isotonic(q) fit on dev'][m].mean():.3f}  mean HGB {preds['HGB realistic (draft+rolling+proxy)'][m].mean():.3f}")
    for lab in ["draft q (raw)", "isotonic(q) fit on dev", "HGB realistic (draft+rolling+proxy)", "draft top-1 p (raw)",
                "EXACT-SAFE HGB prior+rolling+proxy"]:
        print(f"\n## reliability, {lab} (test): bin | n | mean predicted | mean a_d")
        for lo, hi, n, mp, ma in HD.reliability(preds[lab], a):
            print(f"  [{lo:.1f},{hi:.1f}) {n:6d}  {mp:.3f}  {ma:.3f}")
    # Q3e: does previous-cycle teacher information predict the next cycle's first-token acceptance and survival?
    print("\n## Difficulty signal (Q3e): previous cycle's teacher info vs this cycle's acceptance (all splits, n4p0 / n8p0)")
    from scipy.stats import spearmanr
    for run, D in RUNS:
        c, F, G = HD.data(run)
        S = HD.sim.survival(c)
        m = c["cyc"] > 0
        a1 = np.nan_to_num(c["a"][:, 0])
        sumS = S[:, :4].sum(1)
        for f in ["prev_rH", "prev_rp1", "prev_tH0", "prev_tp1_0", "prev_acc1", "ema_acc1"]:
            r1 = spearmanr(G[f][m], a1[m]).correlation; r2 = spearmanr(G[f][m], sumS[m]).correlation
            # partial: within category and controlling for this cycle's draft top-1 p at step 1
            print(f"  {run} {f:11s} spearman with a_1 {r1:+.3f}, with S_1+..+S_4 {r2:+.3f}"
                  f"; vs draft p1 at step 1: {spearmanr(F['p1'][m, 0], a1[m]).correlation:+.3f}")


if __name__ == "__main__":
    main()
