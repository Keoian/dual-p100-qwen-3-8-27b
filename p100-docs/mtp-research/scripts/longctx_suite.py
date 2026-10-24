# longctx_suite.py LABEL: long-context decode on the saved slot library (/work/bench/slots), one JSONL row per request.
# Each question starts from its depth's document end: the slot is restored before every request (restore 4-17 s), so
# arms see identical states. Questions: /work/bench/slots/questions.json ({depth: [16 questions]}).
# env: DEPTHS (default 32k,64k,128k,200k), QIDX (question indices, "even" | "odd" | "all" | comma list; default all),
#      SAMPLING (nothink | kmic), THINK (off | on), MAXTOK (default 400), SEED (default 11), KEYFILE,
#      PROBE=1: at each depth, after the first question, ask the second WITHOUT a restore and record how many prompt
#      tokens it reprocessed (the rollback to the document's end goes through a context checkpoint); that row is
#      marked probe and kept out of the speed summary.
import json, os, sys, time, hashlib, statistics, urllib.request
sys.path.insert(0, "/work/bench/slots"); import longctx
lab = sys.argv[1]
KEY = [l.strip() for l in open(os.environ.get("KEYFILE", "/work/qwen-api-keys.work")) if l.strip() and not l.startswith("#")][0]
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
depths = os.environ.get("DEPTHS", "32k,64k,128k,200k").split(",")
qsel = os.environ.get("QIDX", "all")
maxtok = int(os.environ.get("MAXTOK", "400")); seed = int(os.environ.get("SEED", "11"))
S = {"nothink": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
     "kmic": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0)}[os.environ.get("SAMPLING", "nothink")]
Q = json.load(open("/work/bench/slots/questions.json"))
def idx(n):
    if qsel == "all": return list(range(n))
    if qsel == "even": return list(range(0, n, 2))
    if qsel == "odd": return list(range(1, n, 2))
    return [int(x) for x in qsel.split(",")]
def chat(msgs):
    b = dict(messages=msgs, max_tokens=maxtok, seed=seed, **S, chat_template_kwargs={"enable_thinking": os.environ.get("THINK", "off") == "on"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
                   json.dumps(b).encode(), H), timeout=3600).read())
    return r, time.time() - t0
os.makedirs("/work/bench/mtp3/runs", exist_ok=True)
out = open(f"/work/bench/mtp3/runs/{lab}.jsonl", "w")
rows = []
for d in depths:
    base = longctx.messages(d)
    for n, i in enumerate(idx(len(Q[d]))):
        # the probe asks this question first without a restore (the slot still holds the previous question and answer),
        # then the question is asked again from a restored slot like every other one
        for probe in ((True, False) if (os.environ.get("PROBE") == "1" and n == 1) else (False,)):
            if not probe:
                rs = longctx.restore(d, KEY)
            q = Q[d][i]
            r, wall = chat(base + [{"role": "user", "content": q}])
            m = r["choices"][0]["message"]; t = r["timings"]
            text = (m.get("reasoning_content") or "") + "␞" + (m.get("content") or "")
            row = dict(label=lab, depth=d, q=i, probe=probe, restore_s=None if probe else rs.get("wall_s"), wall_s=wall,
                       prompt_n=t.get("prompt_n"), prompt_ms=t.get("prompt_ms"), predicted_n=t.get("predicted_n"),
                       predicted_ms=t.get("predicted_ms"), draft_n=t.get("draft_n", 0), draft_acc=t.get("draft_n_accepted", 0),
                       finish=r["choices"][0].get("finish_reason"), hash=hashlib.md5(text.encode()).hexdigest()[:8], text=text)
            rows.append(row); out.write(json.dumps(row) + "\n"); out.flush()
            if probe:
                print(f"{lab} {d} probe: follow-up without restore reprocessed prompt_n {row['prompt_n']} "
                      f"({row['prompt_ms']:.0f} ms)", flush=True)
def summ(rs):
    n = sum(r["predicted_n"] for r in rs); ms = sum(r["predicted_ms"] for r in rs)
    dn = sum(r["draft_n"] for r in rs); da = sum(r["draft_acc"] for r in rs)
    return f"{n:6d} tok {n/ms*1000:6.2f} t/s acc {da/max(dn,1):.3f} prompt_n med {statistics.median(r['prompt_n'] for r in rs)}"
for d in depths:
    rs = [r for r in rows if r["depth"] == d and not r["probe"]]
    print(f"{lab:14s} {d:5s} {summ(rs)}  restore med {statistics.median(r['restore_s'] for r in rs):.1f} s", flush=True)
print(f"{lab:14s} ALL   {summ([r for r in rows if not r['probe']])}  wall {sum(r['wall_s'] for r in rows):.0f} s", flush=True)
