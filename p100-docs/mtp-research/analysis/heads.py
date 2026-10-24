# heads.py: the Fidelity predictor of a_d (and its variants), trained on dev cycles of both traces, with soft labels.
# A soft label y in [0,1] is fed to a classifier as two rows (y=1 with weight y, y=0 with weight 1-y), which makes the
# log-loss objective the cross-entropy against y.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import features as FT
import sim

SETS = {
    # exact-safe (decision on token d before looking at it): step-d distribution + earlier tokens
    "prior": FT.PRIOR + FT.CTX,
    "prior+roll+proxy": FT.PRIOR + FT.CTX + FT.ROLL + FT.PROXY,
    "prior+roll+proxy+diff": FT.PRIOR + FT.CTX + FT.ROLL + FT.PROXY + FT.DIFF,
    "prior+roll+proxy+cat": FT.PRIOR + FT.CTX + FT.ROLL + FT.PROXY + FT.CAT,
    "prior+cat": FT.PRIOR + FT.CTX + FT.CAT,
    # token-aware (also sees the sampled token's q): a valid predictor of a_d once token d is in the prefix
    "draft": FT.DRAFT + FT.CTX,
    "realistic": FT.DRAFT + FT.CTX + FT.ROLL + FT.PROXY,
    "realistic+diff": FT.DRAFT + FT.CTX + FT.ROLL + FT.PROXY + FT.DIFF,
    "realistic+cat": FT.DRAFT + FT.CTX + FT.ROLL + FT.PROXY + FT.CAT,
    "draft+cat": FT.DRAFT + FT.CTX + FT.CAT,
    "draft+roll": FT.DRAFT + FT.CTX + FT.ROLL,
    "draft+proxy": FT.DRAFT + FT.CTX + FT.PROXY,
    "draft+diff": FT.DRAFT + FT.CTX + FT.DIFF,
}

_cache = {}


def data(run):
    if run not in _cache:
        c, F, G = FT.build(run)
        _cache[run] = (c, F, G)
    return _cache[run]


def pairs(run, Dmax, split=None):
    c, F, G = data(run)
    rc, rd = FT.all_pairs(c, Dmax)
    if split is not None:
        keep = np.isin(c["split"][rc], np.atleast_1d(split))
        rc, rd = rc[keep], rd[keep]
    return rc, rd


def soft_fit(model, X, y, w=None):
    w = np.ones(len(y)) if w is None else w
    X2 = np.vstack([X, X]); y2 = np.r_[np.ones(len(y)), np.zeros(len(y))]
    w2 = np.r_[w * y, w * (1 - y)]
    keep = w2 > 0
    model.fit(X2[keep], y2[keep], sample_weight=w2[keep])
    return model


class Head:
    """Predicts a_d. cond=True trains the conditional acceptance (weight S_{d-1}), the right factor for the
    chain S_d = prod a_i; cond=False the plain E[a_d | features]."""

    def __init__(self, fset, kind="hgb", cond=True, runs=(("n4p0", 4), ("n8p0", 8)), train_splits=(0,)):
        self.names = SETS[fset]; self.kind = kind; self.cond = cond
        Xs, ys, ws = [], [], []
        for run, D in runs:
            c, F, G = data(run)
            rc, rd = pairs(run, D, train_splits)
            S = sim.survival(c)
            Sprev = np.where(rd > 0, S[rc, np.maximum(rd - 1, 0)], 1.0)
            Xs.append(FT.matrix(c, F, G, self.names, rc, rd)); ys.append(np.nan_to_num(c["a"][rc, rd]))
            ws.append(Sprev if cond else np.ones(len(rc)))
        X = np.vstack(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
        if kind == "lr":
            self.sc = StandardScaler().fit(X)
            self.m = soft_fit(LogisticRegression(C=1.0, max_iter=2000), self.sc.transform(X), y, w)
        else:
            self.sc = None
            self.m = soft_fit(HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=4,
                                                             min_samples_leaf=40, l2_regularization=1.0,
                                                             random_state=0), X, y, w)

    def predict_pairs(self, run, rc, rd):
        c, F, G = data(run)
        X = FT.matrix(c, F, G, self.names, rc, rd)
        if self.sc is not None:
            X = self.sc.transform(X)
        return self.m.predict_proba(X)[:, 1]

    def predict_grid(self, run, D):
        """(n_cycles, D) predicted a for every drafted depth (NaN where not drafted)."""
        c, F, G = data(run)
        rc, rd = FT.all_pairs(c, D)
        out = np.full((len(c["task"]), D), np.nan)
        out[rc, rd] = self.predict_pairs(run, rc, rd)
        return out


def chain(ahat):
    """Token-aware chain prod_{i<=d} ahat_i. NOT exact as a stop rule (it looks at token d itself)."""
    return np.cumprod(np.nan_to_num(ahat, nan=0.0), axis=1)


def exact_chain(prior, post):
    """Exact-safe survival estimate for the decision on token d: prod_{i<d} post_i (earlier tokens, seen) x prior_d."""
    post = np.nan_to_num(post, nan=0.0); prior = np.nan_to_num(prior, nan=0.0)
    cp = np.c_[np.ones(len(post)), np.cumprod(post, axis=1)[:, :-1]]
    return cp * prior


def token_metrics(pred, a):
    y = (a > 0.5).astype(int)
    eps = 1e-6
    pc = np.clip(pred, eps, 1 - eps)
    return dict(auc=roc_auc_score(y, pred), brier=np.mean((pred - a) ** 2),
                logloss=-np.mean(a * np.log(pc) + (1 - a) * np.log(1 - pc)), mean_pred=pred.mean(), mean_a=a.mean())


def reliability(pred, a, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        m = (pred >= edges[i]) & (pred < edges[i + 1] if i < bins - 1 else pred <= 1)
        if m.sum():
            rows.append((edges[i], edges[i + 1], int(m.sum()), pred[m].mean(), a[m].mean()))
    return rows


def isotonic_q(runs=(("n4p0", 4), ("n8p0", 8)), key="d_q"):
    xs, ys = [], []
    for run, D in runs:
        c, F, G = data(run)
        rc, rd = pairs(run, D, 0)
        xs.append(c[key][rc, rd]); ys.append(np.nan_to_num(c["a"][rc, rd]))
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(np.concatenate(xs), np.concatenate(ys))
