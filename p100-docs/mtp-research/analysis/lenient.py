# lenient.py (brief Q4, plan Mode B, APPROXIMATE): accept a drafted token whenever the teacher gives it
# p_draft >= lambda * p_top1 (or p_draft >= tau), else fall back to the exact p/q test. Teacher rows are teacher-forced
# along the drafted prefix, so a'_d is defined after a lenient accept too. Not applied under a grammar (tool calls) and
# never to tokens the draft cannot propose. Speed/divergence frontier only; real quality needs generation.
# divergence = expected committed tokens that the lenient rule accepted where the exact test (same uniform) rejected,
# divided by expected committed tokens per cycle (correction/bonus tokens included in the denominator).
import numpy as np
import sim
from load import load, CATS, DMAX


def lenient_E(c, k, cond):
    a = np.nan_to_num(c["a"][:, :DMAX]); valid = np.arange(DMAX)[None, :] < np.minimum(c["n_real"], k)[:, None]
    L = cond & valid & (~c["grammar"][:, None])
    a2 = np.where(L, 1.0, a) * valid
    S2 = np.cumprod(a2, axis=1)
    reach = np.c_[np.ones(len(a)), S2[:, :-1]]
    div = (reach * L * (1 - a) * valid).sum(1)
    E = 1 + S2.sum(1)
    after = np.zeros(len(a))  # tokens committed after a lenient-only accept within the cycle (on a diverged prefix)
    return E, div, after


def main():
    out = []
    for run, settings in [("n4p0", [(4, 0.2), (4, 0.5)]), ("n8p0", [(4, 0.2), (8, 0.2), (8, 0.0)])]:
        c, _, _ = load(run)
        S = sim.survival(c)
        p = np.nan_to_num(c["t_p"][:, :DMAX]); p1 = np.nan_to_num(c["t_p1"][:, :DMAX], nan=1.0)
        for N, pm in settings:
            k, st = sim.rule_pmin(c, N, pm)
            E0, T = sim.evaluate(c, k, st, N + 1, S)
            allm = np.ones(len(k), bool)
            r0 = sim.rate(c, E0, T, allm)
            print(f"\n## trace-{run} states, n-max {N} p-min {pm}: exact {E0.mean():.3f} tokens/cycle, {r0:.2f} t/s (simulated, all 72 prompts)")
            print("| rule | tokens/cycle | t/s (sim) | gain | divergent fraction of committed tokens | json | tools | code | creative |")
            print("|---|---|---|---|---|---|---|---|---|")
            rules = [(f"lambda {l}", p >= l * p1) for l in (0.9, 0.7, 0.5, 0.3, 0.2, 0.1)] + \
                    [(f"tau {t}", p >= t) for t in (0.5, 0.3, 0.2, 0.1, 0.05)]
            for lab, cond in rules:
                cond = cond & (p > 0)
                E, div, _ = lenient_E(c, k, cond)
                r = sim.rate(c, E, T, allm)
                cats = []
                for cn in ["json", "tools", "code", "creative"]:
                    m = c["cat"] == CATS.index(cn)
                    cats.append(f"{div[m].sum() / E[m].sum() * 100:.1f}% (+{(E[m].mean() / E0[m].mean() - 1) * 100:.0f}% tok)")
                print(f"| {lab} | {E.mean():.3f} | {r:.2f} | {(r / r0 - 1) * 100:+.1f}% | {div.sum() / E.sum() * 100:.1f}% | " + " | ".join(cats) + " |")
            # upper bound: accept everything
            E, div, _ = lenient_E(c, k, np.ones_like(p, bool))
            print(f"| accept every non-grammar draft (bound) | {E.mean():.3f} | {sim.rate(c, E, T, allm):.2f} | {(sim.rate(c, E, T, allm) / r0 - 1) * 100:+.1f}% | {div.sum() / E.sum() * 100:.1f}% | | | | |")
    # with lenient acceptance, is a deeper fixed draft now worth it? (n8p0 states, fixed depth k, p-min 0.2, lambda 0.3 / 0.5)
    c, _, _ = load("n8p0"); S = sim.survival(c)
    p = np.nan_to_num(c["t_p"][:, :DMAX]); p1 = np.nan_to_num(c["t_p1"][:, :DMAX], nan=1.0)
    print("\n## n8p0 states: simulated t/s by n-max (p-min 0.2), exact vs lenient (divergent fraction in brackets)")
    for lab, cond in [("exact", np.zeros_like(p, bool)), ("lambda 0.5", (p >= 0.5 * p1) & (p > 0)), ("lambda 0.3", (p >= 0.3 * p1) & (p > 0)),
                      ("lambda 0.1", (p >= 0.1 * p1) & (p > 0))]:
        row = []
        for N in range(1, 9):
            k, st = sim.rule_pmin(c, N, 0.2)
            _, T = sim.evaluate(c, k, st, N + 1, S)
            E, div, _ = lenient_E(c, k, cond)
            row.append(f"{N}:{sim.rate(c, E, T, np.ones(len(k), bool)):.1f} [{div.sum() / E.sum() * 100:.0f}%]")
        print(f"  {lab:10s} " + "  ".join(row))


if __name__ == "__main__":
    main()
