# depth.py (brief Q1): MTP quality vs draft depth on its own drafted prefix, judged by the teacher on the same prefix.
# Data: trace-n8p0 (n-max 8, p-min 0: every cycle drafts 8 tokens unless cut), cross-check trace-n4p0 (depths 1-4).
import numpy as np
from load import load, CATS, DMAX

np.set_printoptions(precision=3, suppress=True)


def curves(c, mask, D):
    a = np.nan_to_num(c["a"][mask, :D]); nr = c["n_real"][mask]
    full = nr >= D
    a = a[full]
    S = np.cumprod(a, axis=1)
    g = (c["d_id"][mask][full, :D] == c["t_argmax"][mask][full, :D])           # sampled draft == teacher raw argmax
    gt = (c["d_top1id"][mask][full, :D] == c["t_argmax"][mask][full, :D])      # draft's own argmax == teacher argmax
    G = np.cumprod(g, axis=1).astype(bool)
    out = dict(n=int(full.sum()), Ea=a.mean(0), Es=S.mean(0),
               cond=S.mean(0) / np.r_[1.0, S.mean(0)[:-1]],
               agree=g.mean(0), agree_top1=gt.mean(0), Eq=c["d_q"][mask][full, :D].mean(0),
               Ep1=c["d_p1"][mask][full, :D].mean(0))
    # agreement at d given all earlier sampled drafts agreed with the teacher argmax, and given some earlier one did not
    ag_given = [g[:, 0].mean()] + [g[G[:, d - 1], d].mean() for d in range(1, D)]
    ag_not = [np.nan] + [g[~G[:, d - 1], d].mean() if (~G[:, d - 1]).any() else np.nan for d in range(1, D)]
    n_given = [len(g)] + [int(G[:, d - 1].sum()) for d in range(1, D)]
    # acceptance at d given every earlier token had a >= 0.99 (an effectively certain prefix) vs not
    P = np.cumprod(a >= 0.99, axis=1).astype(bool)
    a_given = [a[:, 0].mean()] + [a[P[:, d - 1], d].mean() for d in range(1, D)]
    a_not = [np.nan] + [a[~P[:, d - 1], d].mean() for d in range(1, D)]
    out.update(ag_given=np.array(ag_given), ag_not=np.array(ag_not), n_given=np.array(n_given),
               a_given=np.array(a_given), a_not=np.array(a_not), n_cert=np.array([len(a)] + [int(P[:, d - 1].sum()) for d in range(1, D)]))
    # distribution of a_d
    bins = [(0, 0), (1e-12, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1 - 1e-12), (1, 1)]
    dist = np.array([[((a[:, d] >= lo) & (a[:, d] <= hi)).mean() for (lo, hi) in bins] for d in range(D)])
    out["dist"] = dist
    return out


def show(name, o, D):
    print(f"\n## {name}: {o['n']} cycles with all {D} drafted")
    print("| depth d | " + " | ".join(str(d + 1) for d in range(D)) + " |")
    print("|---|" + "---|" * D)
    rows = [("E[a_d] (MTP quality on its own prefix)", o["Ea"]), ("P(accept d | accepted 1..d-1) = E[S_d]/E[S_d-1]", o["cond"]),
            ("survival E[prod a_i]", o["Es"]), ("greedy agreement: sampled draft == teacher argmax", o["agree"]),
            ("draft top-1 == teacher argmax", o["agree_top1"]),
            ("agreement at d given 1..d-1 all agreed", o["ag_given"]), ("agreement at d given an earlier one disagreed", o["ag_not"]),
            ("E[a_d] given a_i >= 0.99 for all i < d", o["a_given"]), ("E[a_d] given some earlier a_i < 0.99", o["a_not"]),
            ("draft q of the sampled token, mean", o["Eq"]), ("draft top-1 p, mean", o["Ep1"])]
    for lab, v in rows:
        print(f"| {lab} | " + " | ".join("-" if np.isnan(x) else f"{x:.3f}" for x in v) + " |")
    print("| n (all earlier agreed) | " + " | ".join(str(x) for x in o["n_given"]) + " |")
    print("| n (earlier a >= 0.99) | " + " | ".join(str(x) for x in o["n_cert"]) + " |")
    print("\nDistribution of a_d (fraction of cycles): a=0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) | a=1")
    for d in range(D):
        print(f"  d={d + 1}: " + "  ".join(f"{x:.3f}" for x in o["dist"][d]))


if __name__ == "__main__":
    for run, D in [("n8p0", 8), ("n4p0", 4)]:
        c, _, _ = load(run)
        allm = np.ones(len(c["task"]), bool)
        o = curves(c, allm, D)
        show(f"trace-{run}, all prompts", o, D)
        if run == "n8p0":
            print("\n## per category, trace-n8p0: E[a_d] by depth, then survival sum (expected accepted drafts at n-max 8) and E[S_4]")
            print("| category | n | " + " | ".join(f"a{d + 1}" for d in range(D)) + " | sum S (n8) | S4 | S8 | agree d1 | agree d8 given all agreed |")
            print("|---|---|" + "---|" * (D + 5))
            for k, cat in enumerate(CATS):
                o = curves(c, c["cat"] == k, D)
                print(f"| {cat} | {o['n']} | " + " | ".join(f"{x:.2f}" for x in o["Ea"]) +
                      f" | {o['Es'].sum():.2f} | {o['Es'][3]:.3f} | {o['Es'][7]:.3f} | {o['agree'][0]:.2f} | {o['ag_given'][7]:.2f} |")
            # same split by position in reply (first 100 tokens vs later)
            print("\n## by position in reply (trace-n8p0), E[a_d]")
            for lo, hi in [(0, 50), (50, 150), (150, 400)]:
                m = (c["posrel"] >= lo) & (c["posrel"] < hi)
                o = curves(c, m, D)
                print(f"  posrel [{lo},{hi}) n={o['n']}: " + " ".join(f"{x:.3f}" for x in o["Ea"]) + f"  sumS {o['Es'].sum():.2f}")
