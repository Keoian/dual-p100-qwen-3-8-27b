# cfboot.py: bootstrap interval of an approximation's grid-mean loss vs own tables (resampling test prompts once per
# replicate, the same weights for every grid setting, so the interval reflects prompt noise of the whole-grid mean).
import numpy as np
import cfcore as C


def mean_loss_ci(E, tables_of, B=1000):
    """E: cfapprox.Ev; tables_of(gi) -> (tq, tp, theta) or a callable returning (k, T). Returns mean loss, (lo, hi)."""
    S = E.S
    wb = S.boot_w(C.TEST)[:B]
    tot = np.zeros(B); pt = 0.0
    for gi in range(len(C.GRID)):
        t = tables_of(gi)
        k, T = t() if callable(t) else E.kT(*t)
        ko, To = E.kT(*E.own[gi])
        cs = E.cs(gi)
        Ea = 1.0 + cs[np.arange(S.n), k].astype(float)
        Eo = 1.0 + cs[np.arange(S.n), ko].astype(float)
        ra = S.rate(Ea, T, C.TEST, wb); ro = S.rate(Eo, To, C.TEST, wb)
        tot += ra / ro - 1
        pt += S.rate(Ea, T, C.TEST) / S.rate(Eo, To, C.TEST) - 1
    tot /= len(C.GRID)
    return pt / len(C.GRID), np.percentile(tot, [2.5, 97.5])


def c_arrays(tq, tp, suffix=""):
    def arr(name, v):
        s = f"static const float {name:<11s}[24] = {{ "
        parts = [f"{x:.3f}f" for x in v]
        lines = [", ".join(parts[:12]), ", ".join(parts[12:])]
        return s + lines[0] + ",\n" + " " * 39 + lines[1] + " };"
    return "\n".join([arr("spec_iso_q" + suffix, tq), arr("spec_iso_p1" + suffix, tp)])
