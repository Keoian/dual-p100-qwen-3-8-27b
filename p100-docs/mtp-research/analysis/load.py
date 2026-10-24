# load.py: parse the LLAMA_SPEC_TRACE JSONL files and their run_suite rows into flat numpy arrays (cached as .npz).
# One row per verify cycle; per-depth arrays are (n_cycles, DMAX) padded with NaN (or -1 for ids).
# a[c, d] = min(1, p_draft/q) is the per-token p/q rule's acceptance probability of drafted token d given drafted 1..d-1
# (teacher-forced along the drafted prefix, so it is defined past the real rejection too).
# Grammar positions use the plain compare in the server (accept iff the target's sample equals the draft); there the
# acceptance probability is p_draft, not min(1, p/q). The trace does not say which grammar-request positions had the
# grammar active, so a_plain = p_draft is stored as well and the grammar flag is kept (2% of cycles).
import json, os, sys, numpy as np

R = "/work/bench/mtp3"
RUNS = os.environ.get("MTP3_RUNS_DIR", f"{R}/runs")  # trace + harness files (override for dry runs)
CACHE0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")  # chat set's cache; vocab.json for both sets
# Trace set, chosen by MTP3_TRACESET: unset or "chat" = thinking-off traces n4p0/n8p0 + identity run I, cache/, out/;
# "think" = thinking-on traces tn4p0/tn8p0, no identity run, cache/think/, out/think/ (run_think.sh). The code keeps
# the chat names: "n4p0"/"n8p0" mean the n-max 4 / n-max 8 trace of the selected set (FILE maps them to the files).
# The cost model is timing, not sampling: the think set reads the chat set's cost.json unless MTP3_THINK_REFIT_COST=1.
TRACESET = os.environ.get("MTP3_TRACESET") or "chat"
assert TRACESET in ("chat", "think"), TRACESET
CACHE = CACHE0 if TRACESET == "chat" else os.path.join(CACHE0, "think")
FILE = {"n4p0": "tn4p0", "n8p0": "tn8p0"} if TRACESET == "think" else {}
TRACE_RUNS = ["n4p0", "n8p0"] + (["I"] if TRACESET == "chat" else [])
REFIT_COST = TRACESET == "chat" or os.environ.get("MTP3_THINK_REFIT_COST") == "1"
COST = os.path.join(CACHE if REFIT_COST else CACHE0, "cost.json")
DMAX = 8
CATS = ["casual", "code", "creative", "json", "longctx", "reasoning", "switch", "tools"]
SPLITS = ["dev", "val", "test"]


def trace_file(run):
    return f"{RUNS}/trace-{FILE.get(run, run)}.jsonl"


def rows_file(run):
    """run_suite rows of a trace run (the identity run's are I-on.jsonl)."""
    return f"{RUNS}/" + ("I-on.jsonl" if run == "I" else f"T-{FILE.get(run, run)}.jsonl")


def _parse(run):
    T = [json.loads(l) for l in open(rows_file(run))]
    tmap = {t["task"]: t for t in T}
    prompts = json.load(open(f"{R}/suite/prompts.json"))
    pid_of = {p["id"]: i for i, p in enumerate(prompts)}
    recs = [json.loads(l) for l in open(trace_file(run))]
    recs = [r for r in recs if r["task"] in tmap]  # task 0 = server warm-up request
    recs.sort(key=lambda r: (r["task"], r["t_us"]))
    n = len(recs)
    f = lambda: np.full((n, DMAX), np.nan)
    i_ = lambda: np.full((n, DMAX), -1, dtype=np.int64)
    out = dict(task=np.zeros(n, np.int64), pid=np.zeros(n, np.int64), cat=np.zeros(n, np.int64),
               split=np.zeros(n, np.int64), cyc=np.zeros(n, np.int64), pos0=np.zeros(n, np.int64),
               posrel=np.zeros(n, np.int64), n_real=np.zeros(n, np.int64), n_pad=np.zeros(n, np.int64),
               grammar=np.zeros(n, bool), n_acc=np.zeros(n, np.int64), n_emit=np.zeros(n, np.int64),
               n_teacher=np.zeros(n, np.int64), n_draft_entries=np.zeros(n, np.int64),
               d_id=i_(), d_q=f(), d_p1=f(), d_p2=f(), d_p5=f(), d_H=f(), d_istop=f(), d_top1id=i_(),
               t_p=f(), t_p1=f(), t_H=f(), t_argmax=i_(), t_top1id=i_(), t_logit=f(), t_logitmax=f(),
               b_p1=np.full(n, np.nan), b_H=np.full(n, np.nan),     # bonus row (index n_real) of the teacher
               r_p1=np.full(n, np.nan), r_H=np.full(n, np.nan),     # row n_acc: the one the emitted last token came from
               a=f(), a_plain=f())
    for k in ["draft_us", "pad_us", "verify_us", "verify_enqueue_us", "catchup_us", "accept_us", "trace_us",
              "cycle_us", "gap_us"]:
        out["tm_" + k] = np.full(n, np.nan)
    emitted = []
    last_task, cyc, first_pos = None, 0, 0
    for c, r in enumerate(recs):
        t = tmap[r["task"]]
        if r["task"] != last_task:
            last_task, cyc, first_pos = r["task"], 0, r["pos0"]
        out["task"][c] = r["task"]; out["pid"][c] = pid_of[t["id"]]
        out["cat"][c] = CATS.index(t["category"]); out["split"][c] = SPLITS.index(t["split"])
        out["cyc"][c] = cyc; cyc += 1
        out["pos0"][c] = r["pos0"]; out["posrel"][c] = r["pos0"] - first_pos
        nr = r["n_real"]; out["n_real"][c] = nr; out["n_pad"][c] = r["n_draft_padded"]
        out["grammar"][c] = r["grammar"]; out["n_acc"][c] = r["n_accepted"]; out["n_emit"][c] = len(r["emitted"])
        out["n_teacher"][c] = len(r["teacher"]); out["n_draft_entries"][c] = len(r["draft"])
        for d, s in enumerate(r["draft"][:min(nr, DMAX)]):
            out["d_id"][c, d] = s["id"]; out["d_q"][c, d] = s["q"]
            tp = [x[1] for x in s["top"]]
            out["d_p1"][c, d] = tp[0]; out["d_p2"][c, d] = tp[1] if len(tp) > 1 else 0.0
            out["d_p5"][c, d] = sum(tp); out["d_H"][c, d] = s["H"]
            out["d_istop"][c, d] = float(s["id"] == s["top"][0][0]); out["d_top1id"][c, d] = s["top"][0][0]
        tch = r["teacher"]
        for d in range(min(nr, DMAX)):
            if d < len(tch) and "p_draft" in tch[d]:
                e = tch[d]
                out["t_p"][c, d] = e["p_draft"]; out["t_p1"][c, d] = e["top1"][1]; out["t_H"][c, d] = e["H"]
                out["t_argmax"][c, d] = e["argmax"]; out["t_top1id"][c, d] = e["top1"][0]
                out["t_logit"][c, d] = e["logit_draft"]; out["t_logitmax"][c, d] = e["logit_max"]
                q = out["d_q"][c, d]
                out["a"][c, d] = min(1.0, e["p_draft"] / q) if q > 0 else 0.0
                out["a_plain"][c, d] = e["p_draft"]
            else:  # grammar rejected an earlier drafted token on the forced prefix: no rows from here on
                out["a"][c, d] = 0.0; out["a_plain"][c, d] = 0.0
        if len(tch) > nr:
            out["b_p1"][c] = tch[nr]["top1"][1]; out["b_H"][c] = tch[nr]["H"]
        na = r["n_accepted"]
        if na < len(tch):
            out["r_p1"][c] = tch[na]["top1"][1]; out["r_H"][c] = tch[na]["H"]
        for k, v in r["timing"].items():
            out["tm_" + k][c] = v
        emitted.append(r["emitted"])
    req = dict(task=np.array([t["task"] for t in T]), pid=np.array([pid_of[t["id"]] for t in T]),
               cat=np.array([CATS.index(t["category"]) for t in T]), split=np.array([SPLITS.index(t["split"]) for t in T]),
               predicted_n=np.array([t["predicted_n"] for t in T]), predicted_ms=np.array([t["predicted_ms"] for t in T]),
               draft_n=np.array([t["draft_n"] for t in T]), draft_acc=np.array([t["draft_acc"] for t in T]),
               prompt_n=np.array([t["prompt_n"] for t in T]))
    return out, emitted, req


def load(run):
    """run in {'n4p0','n8p0','I'}; returns (cycles dict of arrays, emitted list, request dict).
    The identity run I exists only in the chat set (the think set's cost refit borrows it), so it is cached in CACHE0."""
    cache = CACHE0 if run == "I" else CACHE
    os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, f"{FILE.get(run, run)}.npz")
    pe = os.path.join(cache, f"{FILE.get(run, run)}.emitted.json")
    src = trace_file(run)
    if os.path.exists(p) and os.path.getmtime(p) > os.path.getmtime(src):
        z = np.load(p)
        cyc = {k[2:]: z[k] for k in z.files if k.startswith("c_")}
        req = {k[2:]: z[k] for k in z.files if k.startswith("r_")}
        return cyc, json.load(open(pe)), req
    cyc, emitted, req = _parse(run)
    np.savez_compressed(p, **{"c_" + k: v for k, v in cyc.items()}, **{"r_" + k: v for k, v in req.items()})
    json.dump(emitted, open(pe, "w"))
    return cyc, emitted, req


def prompts():
    return json.load(open(f"{R}/suite/prompts.json"))


_vocab = None
_bdec = None


def detok(ids):
    """GPT-2 byte-level BPE detokenizer over cache/vocab.json (built by vocab.py)."""
    global _vocab, _bdec
    if _vocab is None:
        _vocab = json.load(open(os.path.join(CACHE0, "vocab.json")))
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        m = 0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256 + m); m += 1
        _bdec = {chr(c): b for b, c in zip(bs, cs)}
    out = bytearray()
    for i in ids:
        s = _vocab[i]
        if s.startswith("<|") or s in ("<think>", "</think>", "<tool_call>", "</tool_call>"):
            out += s.encode(); continue
        for ch in s:
            b = _bdec.get(ch)
            if b is None:
                out += ch.encode()
            else:
                out.append(b)
    return out.decode("utf-8", "replace")


if __name__ == "__main__":
    for run in sys.argv[1:] or TRACE_RUNS:
        c, e, r = load(run)
        print(FILE.get(run, run), len(c["task"]), "cycles", len(r["task"]), "requests")
