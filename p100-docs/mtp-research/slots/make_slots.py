# make_slots.py build | verify   (server on 127.0.0.1:8090, key from KEYFILE, default /work/qwen-api-keys.work)
#
# build:  the nested long-context slot library. A fixed corpus (corpus.txt) is cut into chunks at token targets; depth
#         k's conversation is [system, user part 1, ..., user part k]. Each conversation renders (chat template, as a
#         chat request with one more user message would) to a prefix of the next, so the prefill runs once over
#         ~250k tokens and the slot is saved at every depth: /work/slots/longctx-<N>k.bin (+ .draft .spec .ckpt).
#         messages-<N>k.json holds the exact messages; a test appends its own user message to them.
# verify: for each depth, restore the slot, send one question appended, and check that only the question was
#         processed (timings.prompt_n). Records restore time and prompt time.
import json, os, sys, time, hashlib, urllib.request
D = "/work/bench/slots"
KEY = [l.strip() for l in open(os.environ.get("KEYFILE", "/work/qwen-api-keys.work")) if l.strip() and not l.startswith("#")][0]
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
BASE = "http://127.0.0.1:8090"
TARGETS = [16000, 25000, 32000, 64000, 128000, 200000, 250000]  # rendered prompt tokens, approximately
SYSTEM = ("You are a helpful assistant. The user shares a long reference document in numbered parts, then asks "
          "questions about it. Answer from the document.")
Q = "In one sentence, what is the last part of the reference document about?"

def post(path, body, timeout=7200):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), H)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def render(msgs):
    # the prompt a chat request with these messages plus one more user message renders, cut before that message
    full = post("/apply-template", {"messages": msgs + [{"role": "user", "content": "@@Q@@"}]})["prompt"]
    i = full.rindex("@@Q@@")
    j = full.rindex("<|im_start|>user", 0, i)
    return full[:j]

def ntok(text):
    return len(post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"])

def build():
    corpus = open(f"{D}/corpus.txt").read()
    toks = post("/tokenize", {"content": corpus, "add_special": False})["tokens"]
    print(f"corpus {len(corpus)} chars, {len(toks)} tokens", flush=True)
    msgs = [{"role": "system", "content": SYSTEM}]
    base = ntok(render(msgs))
    pos, prev_prefix, manifest = 0, "", []
    for k, target in enumerate(TARGETS, 1):
        need = target - ntok(render(msgs)) - 12            # ~12 tokens of message wrapper and part header
        chunk = post("/detokenize", {"tokens": toks[pos:pos + need]})["content"]
        pos += need
        msgs.append({"role": "user", "content": f"Reference document, part {k}:\n\n{chunk}"})
        prefix = render(msgs)
        assert prefix.startswith(prev_prefix), f"depth {k}: rendered prefix is not an extension of the previous one"
        prev_prefix = prefix
        t0 = time.time()
        r = post("/completion", {"prompt": prefix, "n_predict": 0, "cache_prompt": True})
        t_prefill = time.time() - t0
        name = f"longctx-{target // 1000}k"
        t0 = time.time()
        s = post(f"/slots/0?action=save", {"filename": f"{name}.bin"})
        t_save = time.time() - t0
        json.dump(msgs, open(f"{D}/messages-{target // 1000}k.json", "w"))
        files = {f: os.path.getsize(f"/work/slots/{f}") for f in os.listdir("/work/slots") if f.startswith(name + ".bin")}
        row = dict(name=name, target=target, n_tokens=s.get("n_saved"), prompt_processed=r.get("timings", {}).get("prompt_n"),
                   prefill_s=round(t_prefill, 1), save_s=round(t_save, 1), files=files, bytes=sum(files.values()),
                   prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest()[:16], messages=f"messages-{target // 1000}k.json")
        manifest.append(row)
        print(json.dumps(row), flush=True)
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), corpus="corpus.txt",
                corpus_sha256=hashlib.sha256(open(f"{D}/corpus.txt", "rb").read()).hexdigest()[:16],
                server=os.environ.get("SLOT_SERVER_DESC", ""), question_for_verify=Q, slots=manifest)
    json.dump(meta, open(f"{D}/manifest.json", "w"), indent=1)

def verify():
    meta = json.load(open(f"{D}/manifest.json"))
    for row in meta["slots"]:
        msgs = json.load(open(f"{D}/{row['messages']}"))
        t0 = time.time()
        r = post("/slots/0?action=restore", {"filename": row["name"] + ".bin"})
        t_restore = time.time() - t0
        c = post("/v1/chat/completions", {"messages": msgs + [{"role": "user", "content": Q}], "max_tokens": 60,
                                          "temperature": 0.0, "top_k": 1})
        t = c["timings"]
        row.update(restore_s=round(t_restore, 1), n_restored=r.get("n_restored"), verify_prompt_n=t.get("prompt_n"),
                   verify_prompt_ms=round(t.get("prompt_ms", 0)), verify_reply=c["choices"][0]["message"]["content"][:160])
        print(json.dumps({k: row[k] for k in ("name", "n_tokens", "restore_s", "n_restored", "verify_prompt_n", "verify_prompt_ms", "verify_reply")}), flush=True)
    json.dump(meta, open(f"{D}/manifest.json", "w"), indent=1)

{"build": build, "verify": verify}[sys.argv[1]]()
