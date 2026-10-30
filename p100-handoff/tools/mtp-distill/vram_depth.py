# vram_depth.py (task 1a/1b VRAM): production depth for one server config. Restores a ~250k-token saved chat
# (/work/slots/<slotfile>), sends a 1440p small-text screenshot (3600 image tokens) plus a question, then a text
# follow-up that decodes with MTP drafts at depth. The window script samples nvidia-smi every 200 ms meanwhile.
#   python3 vram_depth.py <slotfile> <label>        (temp key /work/qwen-api-keys.work)
import base64, json, subprocess, sys, time, urllib.request
sys.path.insert(0, "/work/bench/slots")
import longctx
slotfile, lab = sys.argv[1], sys.argv[2]
KEY = [l.strip() for l in open("/work/qwen-api-keys.work") if l.strip() and not l.startswith("#")][0]
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}


def post(path, body, timeout=1800):
    return json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8090" + path, json.dumps(body).encode(), H),
                                             timeout=timeout).read())


def vram():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                          text=True).stdout.split()


t0 = time.time(); r = post("/slots/0?action=restore", {"filename": slotfile})
print(f"[{lab}] restored {r.get('n_restored')} tokens in {time.time() - t0:.1f} s, VRAM {vram()}", flush=True)
msgs = longctx.messages("250k")
b = base64.b64encode(open("/work/bench/vision1440/img0.png", "rb").read()).decode()
msgs.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}},
             {"type": "text", "text": "Ignore the reference parts. Transcribe every 'Row X: CODE' line of this screenshot as JSON."}]})
body = dict(messages=msgs, max_tokens=120, temperature=0.0, top_k=1, cache_prompt=True, **{"speculative.accept": "exact"})
t0 = time.time(); r = post("/v1/chat/completions", body); tm = r["timings"]
print(f"[{lab}] image at depth {r['usage']['prompt_tokens']}: new {tm['prompt_n']} tokens in {tm['prompt_ms'] / 1000:.1f} s, "
      f"decode {tm['predicted_per_second']:.1f} t/s, draft {tm.get('draft_n_accepted')}/{tm.get('draft_n')}, VRAM {vram()}", flush=True)
msgs.append({"role": "assistant", "content": r["choices"][0]["message"]["content"]})
msgs.append({"role": "user", "content": "Now summarize part 0 of the reference material in about 150 words."})
body = dict(messages=msgs, max_tokens=250, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, seed=7,
            cache_prompt=True, **{"speculative.accept": "exact"})
r = post("/v1/chat/completions", body); tm = r["timings"]
print(f"[{lab}] follow-up at depth {r['usage']['prompt_tokens']}: decode {tm['predicted_per_second']:.1f} t/s, "
      f"draft {tm.get('draft_n_accepted')}/{tm.get('draft_n')}, VRAM {vram()}", flush=True)
