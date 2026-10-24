# summ10.py: window 10 summary. Per depth: production (P) and the rule (L) over both rounds (even + odd questions),
# committed tok/s over all tokens, acceptance; L vs P with a 95% bootstrap over questions (same questions in both arms).
import json, random, os, collections
R = "/work/bench/mtp3/runs"
def load(l):
    p = f"{R}/{l}.jsonl"
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []
def tps(rs): return sum(r["predicted_n"] for r in rs) / sum(r["predicted_ms"] for r in rs) * 1000
P = [r for l in ("A10-Pe", "A10-Po") for r in load(l) if not r["probe"]]
L = [r for l in ("A10-Le", "A10-Lo") for r in load(l) if not r["probe"]]
for r in load("A10-Pe"):
    if r["probe"]:
        print(f"probe {r['depth']}: a follow-up without restore reprocessed {r['prompt_n']} tokens ({r['prompt_ms']/1000:.1f} s)")
depths = sorted({r["depth"] for r in P}, key=lambda d: int(d[:-1]))
for d in depths + ["ALL"]:
    bp = {(r["depth"], r["q"]): r for r in P if d in ("ALL", r["depth"])}
    bl = {(r["depth"], r["q"]): r for r in L if d in ("ALL", r["depth"])}
    ids = sorted(set(bp) & set(bl))
    if not ids:
        continue
    line = []
    for lab, b in (("P", bp), ("L", bl)):
        rs = [b[i] for i in ids]
        dn = sum(r["draft_n"] for r in rs); da = sum(r["draft_acc"] for r in rs)
        line.append(f"{lab} {tps(rs):6.2f} t/s acc {da/max(dn,1):.3f} ({sum(r['predicted_n'] for r in rs)} tok)")
    ratio = tps([bl[i] for i in ids]) / tps([bp[i] for i in ids])
    rng = random.Random(1); bs = []
    for _ in range(2000):
        s = [rng.choice(ids) for _ in ids]
        bs.append(tps([bl[i] for i in s]) / tps([bp[i] for i in s]))
    bs.sort()
    print(f"{d:5s} n={len(ids):2d}  " + "  ".join(line) +
          f"  L vs P {100*(ratio-1):+.1f}% (95% {100*(bs[50]-1):+.1f}..{100*(bs[1949]-1):+.1f}%)")
