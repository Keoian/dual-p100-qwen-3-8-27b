# cfload.py: parse the LLAMA_SPEC_TRACE_TOPK traces (trace-cfchat / trace-cfthink + their T-*.jsonl rows) into flat
# numpy arrays, cached as cache/cf/<name>.npz. Same structure as load.py (one row per verify cycle, per-depth arrays
# (n_cycles, D) padded with NaN / -1), plus per drafted row the target's raw top-K logits (K = 128), lse_all, and
# the penalty-window membership of each listed id (teacher row d's window = (pen_hist + drafted[0..d-1])[-64:]).
# Only the drafted rows 0..n_real-1 are kept (the bonus row has no drafted token; acceptance never needs it).
# Flags per cycle: grmr (grammar sampler), forced (any teacher row the reasoning budget was forcing), short teacher.
# Counterfactual sets use cycles with cf_ok = ~grmr & ~forced & full teacher.
#   python3 cfload.py [cfchat] [cfthink]   (prints counts)
import json, os, sys
from collections import Counter
import numpy as np

R = "/work/bench/mtp3"
RUNS = f"{R}/runs"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "cf")
CATS = ["casual", "code", "creative", "json", "longctx", "reasoning", "switch", "tools"]
SPLITS = ["dev", "val", "test"]
D = 4      # n-max of the cf traces
K = 128    # LLAMA_SPEC_TRACE_TOPK
NPEN = 64  # penalty_last_n of the grid (= the recorded one)


def _parse(name):
    T = [json.loads(l) for l in open(f"{RUNS}/T-{name}.jsonl")]
    tmap = {t["task"]: t for t in T}
    prompts = json.load(open(f"{R}/suite/prompts.json"))
    pid_of = {p["id"]: i for i, p in enumerate(prompts)}
    cyc = []  # per record: dict of small arrays
    samp = Counter()
    for line in open(f"{RUNS}/trace-{name}.jsonl"):
        r = json.loads(line)
        if r["task"] not in tmap:  # task 0 = warm-up
            continue
        t = tmap[r["task"]]
        s = r["sampling"]
        samp[tuple(s[k] for k in ("temp", "top_k", "top_p", "min_p", "penalty_present", "penalty_repeat",
                                  "penalty_freq", "penalty_last_n"))] += 1
        nr = r["n_real"]
        tch = r["teacher"]
        dr = r["draft"]
        e = dict(task=r["task"], t_us=r["t_us"], pid=pid_of[t["id"]], cat=CATS.index(t["category"]),
                 split=SPLITS.index(t["split"]), pos0=r["pos0"], n_real=nr, n_pad=r["n_draft_padded"],
                 grmr=bool(r.get("grmr", r["grammar"])), grammar=bool(r["grammar"]),
                 forced=any(x.get("forced", False) for x in tch), n_teacher=len(tch),
                 n_acc=r["n_accepted"], n_emit=len(r["emitted"]), rule=r["rule"])
        ids = np.full(D, -1, np.int64); q = np.full(D, np.nan); p1 = np.full(D, np.nan); H = np.full(D, np.nan)
        tid = np.full((D, 5), -1, np.int64); tp = np.full((D, 5), np.nan)
        for d, st in enumerate(dr[:min(nr, D)]):
            ids[d] = st["id"]; q[d] = st["q"]; H[d] = st["H"]
            p1[d] = st["top"][0][1]
            for j, (a, b) in enumerate(st["top"][:5]):
                tid[d, j] = a; tp[d, j] = b
        rid = np.full((D, K), -1, np.int64); rlg = np.full((D, K), -np.inf, np.float32)
        lse = np.full(D, np.nan); pdr = np.full(D, np.nan); frc = np.zeros(D, bool); win = np.zeros((D, K), bool)
        win_top = np.zeros((D, 5), bool)
        pen = list(r.get("pen_hist") or [])
        for d in range(min(nr, D, len(tch))):
            x = tch[d]
            if "p_draft" in x:
                pdr[d] = x["p_draft"]
            frc[d] = bool(x.get("forced", False))
            raw = x.get("raw")
            if raw:
                kk = min(len(raw), K)
                rid[d, :kk] = [a for a, _ in raw[:kk]]
                rlg[d, :kk] = np.array([b for _, b in raw[:kk]], np.float32)
                lse[d] = x.get("lse_all", np.nan)
            w = set((pen + [int(v) for v in ids[:d]])[-NPEN:])
            win[d] = np.isin(rid[d], list(w)) if w else False
            win_top[d] = np.isin(tid[d], list(w)) if w else False
        e.update(d_id=ids, d_q=q, d_p1=p1, d_H=H, d_tid=tid, d_tp=tp, r_id=rid, r_lg=rlg, r_lse=lse, t_p=pdr,
                 t_forced=frc, r_win=win, d_twin=win_top, pen_len=len(pen))
        cyc.append(e)
    cyc.sort(key=lambda e: (e["task"], e["t_us"]))
    out = {}
    for k in cyc[0]:
        if k == "rule":
            out[k] = np.array([e[k] for e in cyc])
        else:
            out[k] = np.stack([np.asarray(e[k]) for e in cyc])
    out["cf_ok"] = ~out["grmr"] & ~out["forced"] & (out["n_teacher"] >= np.minimum(out["n_real"], D))
    # cycle index within its request, position relative to the first cycle
    cy = np.zeros(len(cyc), np.int64)
    for i in range(1, len(cyc)):
        cy[i] = cy[i - 1] + 1 if out["task"][i] == out["task"][i - 1] else 0
    out["cyc"] = cy
    meta = dict(sampling=[list(k) + [v] for k, v in samp.items()], n_req=len(T),
                req_predicted_n=[t["predicted_n"] for t in T], req_task=[t["task"] for t in T])
    return out, meta


def load(name):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{name}.npz")
    pm = os.path.join(CACHE, f"{name}.meta.json")
    src = f"{RUNS}/trace-{name}.jsonl"
    if os.path.exists(p) and os.path.getmtime(p) > os.path.getmtime(src):
        z = np.load(p)
        return {k: z[k] for k in z.files}, json.load(open(pm))
    out, meta = _parse(name)
    np.savez_compressed(p, **out)
    json.dump(meta, open(pm, "w"))
    return out, meta


def report(name):
    c, meta = load(name)
    n = len(c["task"])
    ok = c["cf_ok"]
    nrows = np.minimum(c["n_real"], D)
    print(f"{name}: {n} cycles, {int(nrows.sum())} drafted rows, {meta['n_req']} requests; sampling {meta['sampling']}")
    print(f"  excluded: grammar (grmr) {int(c['grmr'].sum())} cycles / {int(nrows[c['grmr']].sum())} rows; "
          f"forced-row cycles {int(c['forced'].sum())} / {int(nrows[c['forced']].sum())} rows "
          f"(both {int((c['grmr'] & c['forced']).sum())}); short teacher {int((c['n_teacher'] < nrows).sum())}; "
          f"kept {int(ok.sum())} cycles / {int(nrows[ok].sum())} rows")
    for s in range(3):
        m = ok & (c["split"] == s)
        print(f"  {SPLITS[s]}: {int(m.sum())} cycles, {len(np.unique(c['pid'][m]))} prompts, {int(c['n_emit'][m].sum())} emitted")
    print("  rules:", dict(Counter(c["rule"].tolist())))


if __name__ == "__main__":
    for nm in sys.argv[1:] or ["cfchat"]:
        report(nm)
