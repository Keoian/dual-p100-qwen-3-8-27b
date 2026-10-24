# tasktype.py (brief Q5): does the category explain acceptance beyond the draft's own q/H? Does acceptance shift
# within a "switch" reply, and does the draft's own confidence already track the shift?
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import heads as HD, features as FT, sim
from load import CATS

BASE = ["logq", "logp1", "H", "margin", "depth"]


def design(run, D, splits, names):
    c, F, G = HD.data(run)
    rc, rd = HD.pairs(run, D, splits)
    X = FT.matrix(c, F, G, names, rc, rd)
    return X, np.nan_to_num(c["a"][rc, rd]), c["pid"][rc], c["cat"][rc]


def fit_eval(names, B=1000):
    Xtr, ytr, ptr, ctr = [np.concatenate(z) if z[0].ndim == 1 else np.vstack(z) for z in
                          zip(*[design(r, D, (0, 1), names) for r, D in (("n4p0", 4), ("n8p0", 8))])]
    Xte, yte, pte, cte = [np.concatenate(z) if z[0].ndim == 1 else np.vstack(z) for z in
                          zip(*[design(r, D, 2, names) for r, D in (("n4p0", 4), ("n8p0", 8))])]
    sc = StandardScaler().fit(Xtr)
    m = HD.soft_fit(LogisticRegression(C=1.0, max_iter=3000), sc.transform(Xtr), ytr)
    pr = m.predict_proba(sc.transform(Xte))[:, 1]
    return pr, yte, pte, cte


def ll(p, a):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(a * np.log(p) + (1 - a) * np.log(1 - p))


def main():
    print("## Q5a: logistic regression (soft labels) trained on dev+val, scored on test prompts")
    res = {}
    for lab, names in [("category only", FT.CAT), ("q/p1/H/margin/depth", BASE), ("q/p1/H/margin/depth + category", BASE + FT.CAT),
                       ("exact-safe p1/H/margin/depth", ["logp1", "H", "margin", "depth"]),
                       ("exact-safe p1/H/margin/depth + category", ["logp1", "H", "margin", "depth"] + FT.CAT)]:
        res[lab] = fit_eval(names)
        pr, a, pid, cat = res[lab]
        print(f"  {lab:42s} log-loss {ll(pr, a).mean():.4f}  Brier {np.mean((pr - a) ** 2):.4f}  AUC {roc_auc_score(a > 0.5, pr):.3f}")
    rng = np.random.default_rng(2)
    for x, y in [("q/p1/H/margin/depth + category", "q/p1/H/margin/depth"), ("exact-safe p1/H/margin/depth + category", "exact-safe p1/H/margin/depth")]:
        p1, a, pid, _ = res[x]; p0 = res[y][0]
        d = ll(p1, a) - ll(p0, a)
        ids = np.unique(pid); idx = {p: np.where(pid == p)[0] for p in ids}
        bs = [np.mean(np.concatenate([d[idx[p]] for p in rng.choice(ids, len(ids))])) for _ in range(1000)]
        print(f"  log-loss change from adding category to [{y}]: {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] (95% CI, prompts)")
    pr, a, pid, cat = res["q/p1/H/margin/depth"]
    print("\n  calibration-in-the-large by category (test), category-blind model: mean a_d vs mean predicted")
    for k, cn in enumerate(CATS):
        m = cat == k
        print(f"    {cn:9s} n={m.sum():5d} mean a {a[m].mean():.3f} predicted {pr[m].mean():.3f} diff {a[m].mean() - pr[m].mean():+.3f}")

    print("\n## Q5b: within 'switch' replies (all splits, trace-n4p0 and n8p0): acceptance by segment and by reply third")
    for run in ("n4p0", "n8p0"):
        c, F, G = HD.data(run)
        S = sim.survival(c)
        iso = HD.isotonic_q(key="d_p1")
        pred1 = iso.predict(np.nan_to_num(c["d_p1"][:, 0]))
        sw = c["cat"] == CATS.index("switch")
        fence = G["in_fence"] > 0
        json_ = G["json_mode"] > 0
        print(f"  {run}: segment | cycles | mean a_1 | E[S1+..+S4] | iso(top-1 p at step 1), category-blind")
        for lab, m in [("switch, prose", sw & ~fence), ("switch, inside ``` fence", sw & fence),
                       ("code category (reference)", c["cat"] == CATS.index("code")),
                       ("casual category (reference)", c["cat"] == CATS.index("casual"))]:
            print(f"    {lab:30s} {m.sum():5d}  {np.nan_to_num(c['a'][m, 0]).mean():.3f}  {S[m, :4].sum(1).mean():.3f}  {pred1[m].mean():.3f}")
        # by reply third, per switch prompt
        print(f"  {run}: per switch prompt, E[S1+..+S4] in reply thirds (first/middle/last) and fence share")
        for pid in np.unique(c["pid"][sw]):
            m = sw & (c["pid"] == pid)
            pr_ = c["posrel"][m]; t = np.digitize(pr_, np.percentile(pr_, [33.3, 66.7]))
            v = [S[m][t == j, :4].sum(1).mean() for j in range(3)]
            p = [pred1[m][t == j].mean() for j in range(3)]
            print(f"    pid {pid:2d}: actual {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}   iso(p1) {p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}   fence share {fence[m].mean():.2f}")
        # variance decomposition: how much of the variance of sum S (per cycle) does segment explain, vs the draft's p1 at step 1
        y = S[sw, :4].sum(1)
        seg = fence[sw].astype(int)
        r2_seg = 1 - sum(((y[seg == g] - y[seg == g].mean()) ** 2).sum() for g in (0, 1)) / ((y - y.mean()) ** 2).sum()
        X = np.c_[np.ones(sw.sum()), np.log(np.nan_to_num(c["d_p1"][sw, :4], nan=1) + 1e-6)]
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r2_p = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        X2 = np.c_[X, seg]; b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
        r2_b = 1 - ((y - X2 @ b2) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  {run}: switch cycles, R^2 of E[S1..S4]: fence segment alone {r2_seg:.3f}; draft log top-1 p (4 steps) {r2_p:.3f}; both {r2_b:.3f}")


if __name__ == "__main__":
    main()
