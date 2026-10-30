# analyze_k1.py: windowK1 (MEASURED, exact acceptance). Part 1 knobs vs the baseline on the same 72 prompts and seed:
# speed (paired bootstrap over prompts; the two ABBA runs of a config are pooled per prompt), acceptance, accepted tokens
# per cycle, drafted per committed token, acceptance by draft depth (LLAMA_SPEC_LOG lines per request) and by category,
# greedy identity vs the production binary, VRAM peaks at depth.
#   python3 analyze_k1.py > out-k1.md
import glob, json, os, re
import numpy as np
R = "/work/bench/mtp5/runs"
CONF = {"B": "baseline (production exact path)", "P": "1c sampler parity (LLAMA_SPEC_DRAFT_REQ=1)",
        "K": "1a draft KV q8_0/q8_0", "M": "1b MTP block Q8_0 (from BF16)",
        "H": "1b' draft LM-head copy q8_0 (VOCAB_TYPE=same)", "W": "1b' draft LM-head copy over 98304 rows"}


def load(lab):
    p = f"{R}/{lab}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def speclog(lab):
    p = f"{R}/speclog-{lab}.txt"
    if not os.path.exists(p):
        return None
    out = []
    for l in open(p):
        f = l.split()
        if len(f) >= 5:
            out.append((int(f[3]), int(f[4])))  # accepted, drafted
    return out


def depth_stats(cyc):
    """acceptance at depth d = P(draft d accepted | drafted d tokens and the first d-1 accepted)."""
    a = np.array([c[0] for c in cyc]); n = np.array([c[1] for c in cyc])
    out = []
    for d in range(1, 5):
        reach = (n >= d) & (a >= d - 1)
        out.append((a[reach] >= d).mean() if reach.any() else float("nan"))
    return out, a.mean(), n.mean()


def arm_rows(mode, conf):
    """all runs of a config (ABBA: <mode>-<conf>1, <mode>-<conf>2), each a list of rows."""
    runs = []
    for k in (1, 2):
        rs = load(f"{mode}-{conf}{k}")
        if rs:
            runs.append((f"{mode}-{conf}{k}", rs))
    return runs


def per_prompt(runs):
    tok, ms = {}, {}
    for _, rs in runs:
        for r in rs:
            tok[r["id"]] = tok.get(r["id"], 0) + r["predicted_n"]
            ms[r["id"]] = ms.get(r["id"], 0) + r["predicted_ms"]
    return tok, ms


def boot(b, x, ids, rng, B=4000):
    a = np.array([b[0][i] for i in ids]); at = np.array([b[1][i] for i in ids])
    c = np.array([x[0][i] for i in ids]); ct = np.array([x[1][i] for i in ids])
    g = (c.sum() / ct.sum()) / (a.sum() / at.sum()) - 1
    bs = []
    for _ in range(B):
        k = rng.integers(0, len(ids), len(ids))
        bs.append((c[k].sum() / ct[k].sum()) / (a[k].sum() / at[k].sum()) - 1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return g, lo, hi


def main():
    rng = np.random.default_rng(0)
    print("# windowK1: Part 1 knobs under exact acceptance (MEASURED on the GPUs, production settings)\n")
    for mode, title in (("C", "Chat (72 prompts, thinking off, cap 300)"), ("T", "Thinking (72 prompts, medium effort, cap 3000)")):
        base = arm_rows(mode, "B")
        if not base:
            continue
        print(f"## {title}\n")
        print("| config | runs | t/s (pooled) | median / mean reply t/s | vs baseline (95% CI) | acceptance | accepted per cycle | "
              "drafted per committed token | acceptance by depth 1/2/3/4 |")
        print("|---|---|---|---|---|---|---|---|---|")
        bp = per_prompt(base)
        cats = {}
        for conf in CONF:
            runs = arm_rows(mode, conf)
            if not runs:
                continue
            rows = [r for _, rs in runs for r in rs]
            n = sum(r["predicted_n"] for r in rows); ms = sum(r["predicted_ms"] for r in rows)
            per = [r["predicted_n"] / r["predicted_ms"] * 1000 for r in rows if r["predicted_n"] >= 8]
            dn = sum(r["draft_n"] for r in rows); da = sum(r["draft_acc"] for r in rows)
            cyc = []
            for lab, rs in runs:
                sl = speclog(lab)
                if sl:   # only the suite's requests (their line ranges), not the warm-up
                    cyc += [c for r in rs if r.get("log0") is not None for c in sl[r["log0"]:r["log1"]]]
            ds, mean_acc, mean_dr = depth_stats(cyc) if cyc else ([float("nan")] * 4, float("nan"), float("nan"))
            xp = per_prompt(runs)
            ids = sorted(set(bp[0]) & set(xp[0]))
            if conf == "B":
                cmp = "—"
                if len(base) == 2:   # drift between the two baseline runs (A ... A of ABBA)
                    g, lo, hi = boot(per_prompt(base[:1]), per_prompt(base[1:]), ids, rng)
                    cmp = f"run 2 vs run 1: {g * 100:+.1f}% ({lo * 100:+.1f}..{hi * 100:+.1f})"
            else:
                g, lo, hi = boot(bp, xp, ids, rng)
                cmp = f"{g * 100:+.1f}% ({lo * 100:+.1f}..{hi * 100:+.1f})"
            print(f"| {CONF[conf]} | {len(runs)} | {n / ms * 1000:.2f} | {np.median(per):.2f} / {np.mean(per):.2f} | {cmp} | "
                  f"{da / max(dn, 1):.4f} | {mean_acc:.3f} | {dn / max(n, 1):.3f} | " + " / ".join(f"{v:.3f}" for v in ds) + " |")
            for r in rows:
                cats.setdefault(conf, {}).setdefault(r["category"], [0, 0])
                cats[conf][r["category"]][0] += r["draft_acc"]; cats[conf][r["category"]][1] += r["draft_n"]
        print("\nAcceptance by category:\n")
        allc = sorted({c for v in cats.values() for c in v})
        print("| config | " + " | ".join(allc) + " |")
        print("|---|" + "---|" * len(allc))
        for conf, v in cats.items():
            print(f"| {conf} | " + " | ".join(f"{v[c][0] / max(v[c][1], 1):.3f}" if c in v else "" for c in allc) + " |")
        print()
    # greedy identity
    print("## Greedy identity (24 test prompts, temperature 0, cap 200; hash of reasoning + content + tool calls)\n")
    ref = load("G-prod")
    if ref:
        refh = {r["id"]: r["hash"] for r in ref}
        for p in sorted(glob.glob(f"{R}/G-*.jsonl")):
            lab = os.path.basename(p)[:-6]
            rs = load(lab)
            same = sum(refh.get(r["id"]) == r["hash"] for r in rs)
            print(f"- {lab}: {same}/{len(rs)} identical to the production binary")
    print()
    # VRAM
    print("## VRAM at production depth (250k chat restored, 1440p image + follow-up; peak MiB per GPU of 16,384)\n")
    for p in sorted(glob.glob(f"{R}/peak-V-*.csv")):
        lab = os.path.basename(p)[5:-4]
        g = {0: 0, 1: 0}
        for l in open(p):
            f = [x.strip() for x in l.split(",")]
            if len(f) == 2 and f[0].isdigit():
                g[int(f[0])] = max(g[int(f[0])], int(f[1]))
        print(f"- {lab}: GPU0 {g[0]:,} / GPU1 {g[1]:,} MiB (free at peak {16384 - g[0]:,} / {16384 - g[1]:,})")


if __name__ == "__main__":
    main()
