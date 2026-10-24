# spad.py: verify cost per real width, padded (width 5, inactive columns skipped) vs unpadded (LLAMA_SPEC_PAD=0).
# Full cycles only (no end-of-request cut). "ideal" = unpadded verify minus its extra graph build/enqueue time over
# the padded arm's: what a cached graph per width could reach.
import json, statistics, collections
R = "/work/bench/mtp3/runs"
def load(name):
    out = collections.defaultdict(list)
    for l in open(f"{R}/trace-{name}.jsonl"):
        r = json.loads(l)
        if r.get("task", 0) == 0:
            continue
        nr = r["n_real"]; npd = r["n_draft_padded"]; t = r["timing"]
        if name == "padon" and npd != 4:
            continue
        out[nr + 1].append((t["verify_us"], t["verify_enqueue_us"], t["cycle_us"], r["pos0"]))
    return out
P, U = load("padon"), load("padoff")
med = lambda xs: statistics.median(xs)
print("active cols | padded W5: n  verify ms  enqueue ms | unpadded W=cols: n  verify ms  enqueue ms | ideal ms | ideal vs padded")
for w in range(1, 6):
    p, u = P.get(w, []), U.get(w, [])
    if not p or not u:
        print(f"{w:11d} | missing (padded {len(p)}, unpadded {len(u)})"); continue
    pv, pe = med([x[0] for x in p]) / 1000, med([x[1] for x in p]) / 1000
    uv, ue = med([x[0] for x in u]) / 1000, med([x[1] for x in u]) / 1000
    ideal = uv - max(0.0, ue - pe)
    print(f"{w:11d} | {len(p):6d}  {pv:8.2f}  {pe:9.2f} | {len(u):6d}  {uv:8.2f}  {ue:9.2f} | {ideal:7.2f} | {100*(ideal/pv-1):+6.1f}%")
for name, d in (("padded", P), ("unpadded", U)):
    rows = [x for v in d.values() for x in v]
    print(f"{name}: {len(rows)} cycles, mean cycle {statistics.mean(x[2] for x in rows)/1000:.2f} ms, mean pos0 {statistics.mean(x[3] for x in rows):.0f}")
