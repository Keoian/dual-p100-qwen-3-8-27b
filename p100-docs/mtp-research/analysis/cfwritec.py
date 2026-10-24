# cfwritec.py: out/cf/recommended-c.txt from out/cf/recommend.json (the recommendation R1, then the alternatives R2, R3).
import json
import numpy as np
import cfcore as C
from cfboot import c_arrays

R = json.load(open(f"{C.OUT}/recommend.json"))
x = ("static const float spec_iso_x [24] = { " + ", ".join(f"{v:.3f}f" for v in C.X24[:12]) + ",\n" + " " * 39 +
     ", ".join(f"{v:.3f}f" for v in C.X24[12:]) + " };")
o = []
o.append("// CF_ANALYSIS.md, recommendation R1: keep production's two table sets (values below are the production arrays,")
o.append("// unchanged); only the gate changes. Thinking set + theta 0.15 for target temperature in [0.95, 1.05] (as today);")
o.append("// chat set + theta 0.125 for EVERY other temperature (today: only [0.5, 0.9], p-min 0.2 elsewhere). No new arrays.")
o.append("// Lookup: linear interpolation on spec_iso_x (clamped). Stop before drafted token d when")
o.append("//   prod_{i<d} iso_q(q_i) * iso_p1(p1_d) < theta.")
o.append(x)
o.append("// chat set (production, out/iso_tables.txt), theta 0.125")
o.append(c_arrays(*C.PROD_CHAT))
o.append("// thinking set (production, out/think/iso_tables.txt), theta 0.15")
o.append(c_arrays(*C.PROD_THINK, "_think"))
o.append("")
r2 = R["R2"]
o.append(f"// Alternative R2: one new set for every setting (pooled marginal fit over the grid, both contents), theta {r2['theta']};")
o.append("// replaces both sets and the gate. Simulated: -0.09% vs own tables (pooled), but -0.23% / -0.37% vs production's")
o.append("// tables at the two card settings, so not recommended over R1.")
o.append(c_arrays(np.array(r2["tq"]), np.array(r2["tp"]), "_one"))
o.append(f"static const float spec_stop_iso_theta_one = {r2['theta']}f;")
o.append("")
r3 = R["R3"]
o.append("// Alternative R3: temperature-indexed sets at T = 0.3 / 0.7 / 1.2 (pooled marginal fits), linear in T, clamped")
o.append("// outside [0.3, 1.2]; theta 0.175 at every anchor. Simulated -0.01% vs own tables (pooled), worst -0.12%.")
o.append("static const float spec_iso_t[3] = { " + ", ".join(f"{t:.1f}f" for t in r3["temps"]) + " };")
for t, (a, b, c) in zip(r3["temps"], r3["tabs"]):
    o.append(c_arrays(np.array(a), np.array(b), f"_t{int(round(t * 10)):02d}"))
o.append("static const float spec_iso_theta_t[3] = { " + ", ".join(f"{c}f" for _, _, c in r3["tabs"]) + " };")
open(f"{C.OUT}/recommended-c.txt", "w").write("\n".join(o) + "\n")
print("\n".join(o))
