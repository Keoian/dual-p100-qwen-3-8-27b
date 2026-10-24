# calib.py: does the simulation reproduce BASELINES' measured fixed-policy sweep?
# Simulate each B-run setting on the trace states of the same prompt subset (dev+val, no longctx), from both traces,
# and compare committed t/s and tokens per cycle with the measured runs. Measured runs are unpaired (different texts).
import numpy as np
from load import load, CATS
import sim
from cost import b_runs

SETTINGS = [("B3-n1", 1, 0.2), ("B3-n2", 2, 0.2), ("B3-n3", 3, 0.2), ("B1r", 4, 0.2), ("B1", 4, 0.2),
            ("B4-p0.0", 4, 0.0), ("B4-p0.1", 4, 0.1), ("B4-p0.4", 4, 0.4), ("B4-p0.6", 4, 0.6),
            ("B3-n6", 6, 0.2), ("B3-n8", 8, 0.2)]


def main():
    traces = {r: load(r)[0] for r in ["n8p0", "n4p0"]}
    print("| run | n-max | p-min | measured t/s | sim t/s (n8p0 states) | diff | sim t/s (n4p0 states) | diff |"
          " measured tok/cycle | sim tok/cycle (n8p0) | measured drafted/cycle | sim drafted/cycle (n8p0) | which parts are calibrated on this run |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, n, pm in SETTINGS:
        b = b_runs(lab)
        out = []
        for r, c in traces.items():
            m = (c["cat"] != CATS.index("longctx")) & (c["split"] < 2)
            if r == "n4p0" and n > 4:
                out.append(None); continue
            k, st = sim.rule_pmin(c, n, pm)
            E, T = sim.evaluate(c, k, st, n + 1)
            out.append((sim.rate(c, E, T, m), E[m].mean(), k[m].mean()))
        cal = {1: "V(2)", 2: "V(3)", 3: "V(4)", 6: "V(7)"}.get(n, "none (V5, s, V9 from traces)")
        f = lambda o: "-" if o is None else f"{o[0]:.2f}"
        dd = lambda o: "-" if o is None else f"{(o[0] / b['tps'] - 1) * 100:+.1f}%"
        print(f"| {lab} | {n} | {pm} | {b['tps']:.2f} | {f(out[0])} | {dd(out[0])} | {f(out[1])} | {dd(out[1])} | "
              f"{b['tpc']:.3f} | {out[0][1]:.3f} | {b['dpc']:.3f} | {out[0][2]:.3f} | {cal} |")
    # fixed depth, p-min 0, the same subset, for the shape
    print("\nfixed depth k (n-max k, p-min 0), dev+val no longctx:")
    for r, c in traces.items():
        m = (c["cat"] != CATS.index("longctx")) & (c["split"] < 2)
        row = []
        for n in range(1, 9 if r == "n8p0" else 5):
            k, st = sim.rule_fixed(c, n)
            E, T = sim.evaluate(c, k, st, n + 1)
            row.append(f"{n}:{sim.rate(c, E, T, m):.2f}")
        print(f"  {r}: " + "  ".join(row))
    # position-weighted (renewal) estimator as a sensitivity check, n8p0 states
    c = traces["n8p0"]; m = (c["cat"] != CATS.index("longctx")) & (c["split"] < 2)
    row = []
    for n in range(1, 9):
        k, st = sim.rule_pmin(c, n, 0.2)
        E, T = sim.evaluate(c, k, st, n + 1)
        w = c["n_emit"][m]
        row.append(f"{n}:{w.sum() / (w * T[m] / E[m]).sum() * 1e6:.2f}")
    print("  position-weighted estimator (n8p0, p-min 0.2, no t_req): " + "  ".join(row))


if __name__ == "__main__":
    main()
