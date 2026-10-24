# cfcard.py: R2 / R3 / own vs production tables at the two card settings (CF_ANALYSIS.md). Needs out/cf/recommend.json.
import sys
import json, numpy as np, cfcore as C
from cfrecommend import t_interp
R = json.load(open(f"{C.OUT}/recommend.json"))
r2 = (np.array(R["R2"]["tq"]), np.array(R["R2"]["tp"]), R["R2"]["theta"])
tabs3 = [(np.array(a), np.array(b), c) for a, b, c in R["R3"]["tabs"]]
for setname, g, prod in [("chat", C.CHAT_S, C.PROD_CHAT + (0.125,)), ("think", C.THINK_S, C.PROD_THINK + (0.15,))]:
    S = C.Set(C.SETS[setname]); gi = C.GIDX[g]; A = S.a(gi)
    row = json.load(open(f"{C.OUT}/grid-{setname}.json"))[gi]
    pp = S.pol_tables(A, *prod)
    for lab, tab in [("R2", r2), ("R3", t_interp(g[0], R["R3"]["temps"], tabs3)), ("own", (np.array(row["tq"]), np.array(row["tp"]), row["theta"])),
                     ("chat set (R1 at T1.0 would use thinking set)", C.PROD_CHAT + (0.125,))]:
        gg, ci = S.boot_gain(S.pol_tables(A, *tab), pp)
        print(f"{setname} content @ {C.sname(g)}: {lab} vs production tables {gg*100:+.2f}% [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]")
