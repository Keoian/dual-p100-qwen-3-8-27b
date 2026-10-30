import sys, numpy as np, collections
sys.path.insert(0, "/work/bench/mtp5"); sys.path.insert(0, "/work/bench/mtp3/analysis")
import parity_sim as P
from load import detok
for name in ("cfchat", "cfthink"):
    Dt = P.Data(name)
    m = collections.Counter(); hi_tot = 0.0
    for N in (98304,):
        sel = Dt.tid >= N
        for r, c in zip(*np.nonzero(sel)):
            m[int(Dt.tid[r, c])] += float(Dt.pS[r, c])
    tot = sum(m.values())
    print(name, f"served mass on ids>=98304 summed over {len(Dt.x)} rows: {tot:.1f}")
    for i, v in m.most_common(10):
        print(f"   id {i} {detok([i])!r}: {v:.1f} ({v / tot * 100:.1f}%)")
    mm = collections.Counter()
    sel = (Dt.tid >= 81920) & (Dt.tid < 98304)
    for r, c in zip(*np.nonzero(sel)):
        mm[int(Dt.tid[r, c])] += float(Dt.pS[r, c])
    t2 = sum(mm.values())
    print(f"   81920..98303: total {t2:.1f}; top ids " + ", ".join(f"{detok([i])!r} {v:.1f}" for i, v in mm.most_common(8)))
