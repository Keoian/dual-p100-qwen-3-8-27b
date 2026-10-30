# vocab_sim.py: SIMULATION on the cf traces. The MTP draft scores only output rows < LLAMA_MTP_DRAFT_VOCAB (81920), so
# target mass on higher ids is overlap the draft can never get. Upper bound of the gain from a wider draft vocabulary:
# sum of the served p over ids >= N, per drafted row; and how often the committed token is such an id.
import sys, numpy as np
sys.path.insert(0, "/work/bench/mtp5")
import parity_sim as P
for name in ("cfchat", "cfthink"):
    Dt = P.Data(name)
    out = [f"{name}: rows {len(Dt.x)}"]
    for N in (81920, 98304, 131072):
        m = (Dt.tid >= N) & (Dt.pS > 0)
        mass = (Dt.pS * (Dt.tid >= N)).sum(1)
        out.append(f"N={N}: served mass on ids>=N mean {mass.mean():.5f} (rows with any {(mass > 0).mean() * 100:.2f}%), "
                   f"draft top-5 ids>=N {(Dt.dtid >= N).any(1).mean() * 100:.2f}%")
    em = Dt.c["d_id"][Dt.c["d_id"] >= 0]
    print("\n  ".join(out))
    print(f"  recorded drafted tokens with id >= 81920: {(Dt.x >= 81920).mean() * 100:.3f}% (the draft cannot propose them; check)")
