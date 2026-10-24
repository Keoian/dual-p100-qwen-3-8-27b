# summ9.py: window 9 summary. Per arm: committed tok/s over all tokens, acceptance, finish reasons; per mode the rule
# (L) vs production (P), with a 95% bootstrap interval over prompts (resampling prompt ids, the same ids in both arms).
import json, random, collections, os, sys
R = "/work/bench/mtp3/runs"
def load(l):
    p = f"{R}/{l}.jsonl"
    rs = [json.loads(x) for x in open(p)] if os.path.exists(p) else []
    return [r for r in rs if not os.environ.get("SPLIT") or r["split"] == os.environ["SPLIT"]]
def tps(rs): return sum(r["predicted_n"] for r in rs) / sum(r["predicted_ms"] for r in rs) * 1000
pref = sys.argv[1] if len(sys.argv) > 1 else "A9"
for mode, name in (("M", "medium effort (budget sampler, per-token rule)"), ("N", "no effort (block verify)")):
    P, L = load(f"{pref}-{mode}P"), load(f"{pref}-{mode}L")
    if not P or not L:
        print(mode, "missing"); continue
    print(f"== {name}" + (f" [split {os.environ['SPLIT']}]" if os.environ.get("SPLIT") else ""))
    for lab, rs in (("P", P), ("L", L)):
        dn = sum(r["draft_n"] for r in rs); da = sum(r["draft_acc"] for r in rs)
        fin = collections.Counter(r["finish"] for r in rs)
        print(f"  {lab}: {tps(rs):6.2f} t/s  {sum(r['predicted_n'] for r in rs):6d} tok  acc {da/max(dn,1):.3f}  "
              f"drafted/token {dn/sum(r['predicted_n'] for r in rs):.3f}  finish {dict(fin)}")
    bp = {r["id"]: r for r in P}; bl = {r["id"]: r for r in L}
    ids = sorted(set(bp) & set(bl))
    ratio = tps([bl[i] for i in ids]) / tps([bp[i] for i in ids])
    rng = random.Random(1); bs = []
    for _ in range(2000):
        s = [rng.choice(ids) for _ in ids]
        bs.append(tps([bl[i] for i in s]) / tps([bp[i] for i in s]))
    bs.sort()
    print(f"  L vs P {100*(ratio-1):+.1f}%  (95% over prompts {100*(bs[50]-1):+.1f}% .. {100*(bs[1949]-1):+.1f}%)")
    cats = collections.defaultdict(list)
    for i in ids: cats[bp[i]["category"]].append(i)
    print("  by category: " + "  ".join(f"{c} {100*(tps([bl[i] for i in v])/tps([bp[i] for i in v])-1):+.0f}%" for c, v in sorted(cats.items())))
