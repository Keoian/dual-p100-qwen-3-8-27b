# cfrepro_decomp.py chat|think: why the cf refit sits above production tables (the n-max 8 trace depth 5-8 rows). CF_ANALYSIS.md.
import sys, os
os.environ["MTP3_TRACESET"] = sys.argv[1]
import numpy as np, cfcore as C, cfload, load as L
from sklearn.isotonic import IsotonicRegression
name = "cfchat" if sys.argv[1] == "chat" else "cfthink"
prod = C.PROD_CHAT if sys.argv[1] == "chat" else C.PROD_THINK
def fit(xq, xp, y):
    return (IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(xq, y).predict(C.X24),
            IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(xp, y).predict(C.X24))
c, _ = cfload.load(name)
nr = np.minimum(c["n_real"], 4); rc = np.repeat(np.arange(len(nr)), nr); rd = np.concatenate([np.arange(k) for k in nr])
q = c["d_q"][rc, rd]; tp = c["t_p"][rc, rd]; p1 = c["d_p1"][rc, rd]
a = np.where(np.isnan(tp), 0.0, np.minimum(1.0, np.nan_to_num(tp) / q)); dev = c["split"][rc] == 0
c8, _, _ = L.load("n8p0")
nr8 = np.minimum(c8["n_real"], 8); rc8 = np.repeat(np.arange(len(nr8)), nr8); rd8 = np.concatenate([np.arange(k) for k in nr8])
m8 = c8["split"][rc8] == 0; q8 = c8["d_q"][rc8, rd8]; p18 = c8["d_p1"][rc8, rd8]; a8 = np.nan_to_num(c8["a"][rc8, rd8])
for lab, m in [("+ n-max 8 trace, all depths (production)", m8), ("+ n-max 8 trace, depths 1-4 only", m8 & (rd8 < 4))]:
    tq, tpp = fit(np.r_[q[dev], q8[m]], np.r_[p1[dev], p18[m]], np.r_[a[dev], a8[m]])
    print(f"{name} dev {lab}: max |diff| vs production q {np.abs(tq-prod[0]).max():.3f} p1 {np.abs(tpp-prod[1]).max():.3f}; mean signed diff q {np.mean(tq-prod[0]):+.3f}")
tq, tpp = fit(q[dev], p1[dev], a[dev])
print(f"{name} dev only (all cycles, recorded a): mean signed diff vs production q {np.mean(tq-prod[0]):+.3f}, p1 {np.mean(tpp-prod[1]):+.3f}")
print("n-max 8 dev rows at depth 5-8:", int((m8 & (rd8 >= 4)).sum()), "of", int(m8.sum()), "; mean a depth 1-4 %.3f, 5-8 %.3f" % (a8[m8 & (rd8 < 4)].mean(), a8[m8 & (rd8 >= 4)].mean()))
