# run_suite.py LABEL: run the workload suite against the server on :8090 (temp key), one JSONL row per request.
# env: SPLITS (default dev,val), SEEDS (default 11), MAXTOK (default 400), SAMPLING (nothink | kmic | greedy),
#      CATS (comma list, default all), KEYFILE (default /work/qwen-api-keys.work), THINK (off | on; sent as
#      chat_template_kwargs.enable_thinking so servers without --reasoning off match production's default),
#      REASONING_EFFORT (unset: none sent; low | medium: production's LLAMA_THINK_BY_BUDGET turns it into a 512/2048-token
#      thinking budget, i.e. a reasoning-budget sampler and the per-token verify rule; no effort = block verify)
# Natural-length replies (no ignore_eos: it is a logit bias and changes the sampler path). Prints per-category and
# overall committed tok/s (tokens / decode time), per-reply median, draft acceptance, prompt time.
import json, os, sys, time, hashlib, statistics, urllib.request, collections
lab = sys.argv[1]
KEY = [l.strip() for l in open(os.environ.get("KEYFILE", "/work/qwen-api-keys.work")) if l.strip() and not l.startswith("#")][0]
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
splits = os.environ.get("SPLITS", "dev,val").split(",")
seeds = [int(s) for s in os.environ.get("SEEDS", "11").split(",")]
cats = os.environ.get("CATS")
maxtok = int(os.environ.get("MAXTOK", "400"))
S = {"nothink": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
     "kmic": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),
     "greedy": dict(temperature=0.0, top_k=1, presence_penalty=0.0)}[os.environ.get("SAMPLING", "nothink")]
suite = [p for p in json.load(open("/work/bench/mtp3/suite/prompts.json"))
         if p["split"] in splits and (not cats or p["category"] in cats.split(","))]
os.makedirs("/work/bench/mtp3/runs", exist_ok=True)
out = open(f"/work/bench/mtp3/runs/{lab}.jsonl", "w")
rows = []
for seed in seeds:
    for p in suite:
        b = dict(messages=[{"role": "user", "content": p["prompt"]}], max_tokens=maxtok, seed=seed, **S,
                 chat_template_kwargs={"enable_thinking": os.environ.get("THINK", "off") == "on"})
        if os.environ.get("CACHE_PROMPT") == "0":  # no reuse of an earlier request's prompt state (RAM cache)
            b["cache_prompt"] = False
        if os.environ.get("REASONING_EFFORT"):
            b["reasoning_effort"] = os.environ["REASONING_EFFORT"]
        if "tools" in p:
            b["tools"] = p["tools"]
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
                       json.dumps(b).encode(), H), timeout=1800).read())
        wall = time.time() - t0
        try:  # the slot's last task id links this row to LLAMA_SPEC_TRACE records ("task")
            sl = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8090/slots", headers=H), timeout=30).read())
            task_id = sl[0].get("id_task")
        except Exception:
            task_id = None
        m = r["choices"][0]["message"]; t = r["timings"]
        text = (m.get("reasoning_content") or "") + "\u241e" + (m.get("content") or "") + json.dumps(m.get("tool_calls") or [])
        row = dict(label=lab, id=p["id"], category=p["category"], split=p["split"], seed=seed, wall_s=wall, task=task_id,
                   prompt_n=t.get("prompt_n"), prompt_ms=t.get("prompt_ms"), predicted_n=t.get("predicted_n"),
                   predicted_ms=t.get("predicted_ms"), draft_n=t.get("draft_n", 0), draft_acc=t.get("draft_n_accepted", 0),
                   finish=r["choices"][0].get("finish_reason"), hash=hashlib.md5(text.encode()).hexdigest()[:8], text=text)
        rows.append(row); out.write(json.dumps(row) + "\n"); out.flush()
def summ(rs):
    n = sum(r["predicted_n"] for r in rs); ms = sum(r["predicted_ms"] for r in rs)
    dn = sum(r["draft_n"] for r in rs); da = sum(r["draft_acc"] for r in rs)
    med = statistics.median(r["predicted_n"] / r["predicted_ms"] * 1000 for r in rs if r["predicted_n"] >= 8)
    return f"{n:6d} tok {n/ms*1000:6.2f} t/s (median {med:6.2f}) acc {da/max(dn,1):.3f} ({da}/{dn})"
by = collections.defaultdict(list)
for r in rows: by[r["category"]].append(r)
for c, rs in sorted(by.items()): print(f"{lab:14s} {c:9s} {summ(rs)}")
print(f"{lab:14s} {'ALL':9s} {summ(rows)}  wall {sum(r['wall_s'] for r in rows):.0f} s", flush=True)
