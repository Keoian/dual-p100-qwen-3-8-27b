# longctx.py: use the saved long-context slots in tests (see README.md here).
#
#   import sys; sys.path.insert(0, "/work/bench/slots"); import longctx
#   longctx.restore("128k", key)                  # load the slot into slot 0 of the server on :8090 (~12 s)
#   msgs = longctx.messages("128k") + [{"role": "user", "content": "your question"}]
#   ... POST /v1/chat/completions with msgs: only the question is processed (timings.prompt_n ~ 27)
#
# CLI: python3 longctx.py list | restore <depth> [keyfile]
import json, os, sys, time, urllib.request
D = "/work/bench/slots"
BASE = os.environ.get("LONGCTX_BASE", "http://127.0.0.1:8090")

def depths():
    return [r["name"].split("-")[1] for r in json.load(open(f"{D}/manifest.json"))["slots"]]

def messages(depth):
    return json.load(open(f"{D}/messages-{depth}.json"))

def restore(depth, key):
    req = urllib.request.Request(f"{BASE}/slots/0?action=restore", json.dumps({"filename": f"longctx-{depth}.bin"}).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    r["wall_s"] = round(time.time() - t0, 1)
    return r

if __name__ == "__main__":
    if sys.argv[1] == "list":
        for r in json.load(open(f"{D}/manifest.json"))["slots"]:
            print(f"{r['name']:14s} {r['n_tokens']:7d} tokens  {r['bytes']/2**30:5.1f} GiB  restore {r.get('restore_s', '?')} s")
    elif sys.argv[1] == "restore":
        kf = sys.argv[3] if len(sys.argv) > 3 else "/work/qwen-api-keys.work"
        key = [l.strip() for l in open(kf) if l.strip() and not l.startswith("#")][0]
        print(json.dumps(restore(sys.argv[2], key)))
