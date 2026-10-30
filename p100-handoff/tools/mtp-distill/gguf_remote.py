# gguf_remote.py: read a remote GGUF's header over HTTP range requests and fetch selected tensors' raw bytes, without
# downloading the file. Used for task 1b: the BF16 source of the MTP block (blk.64.*) from unsloth's BF16 GGUF.
#   python3 gguf_remote.py list <repo> <file>            tensor infos
#   python3 gguf_remote.py fetch <repo> <file> <prefix> <outdir>   raw bytes per tensor + infos.json
import json, os, struct, sys, urllib.request
from huggingface_hub import hf_hub_url

GGUF_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


class Buf:
    def __init__(self, url):
        self.url, self.data, self.pos = url, b"", 0

    def need(self, n):
        while len(self.data) < self.pos + n:
            self.data += rng(self.url, len(self.data), len(self.data) + (8 << 20))

    def read(self, n):
        self.need(n)
        v = self.data[self.pos:self.pos + n]
        self.pos += n
        return v

    def u(self, fmt):
        n = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.read(n))[0]

    def s(self):
        n = self.u("Q")
        return self.read(n).decode("utf-8", "replace")

    def val(self, t):
        if t == 8:
            return self.s()
        if t == 9:
            et, n = self.u("I"), self.u("Q")
            return [self.val(et) for _ in range(n)]
        return self.u(GGUF_TYPES[t])


def header(url):
    b = Buf(url)
    assert b.read(4) == b"GGUF"
    ver, nt, nkv = b.u("I"), b.u("Q"), b.u("Q")
    kv = {}
    for _ in range(nkv):
        k = b.s(); t = b.u("I"); kv[k] = b.val(t)
    tens = []
    for _ in range(nt):
        name = b.s(); nd = b.u("I"); dims = [b.u("Q") for _ in range(nd)]; typ = b.u("I"); off = b.u("Q")
        tens.append(dict(name=name, dims=dims, type=typ, offset=off))
    align = kv.get("general.alignment", 32)
    data0 = (b.pos + align - 1) // align * align
    return kv, tens, data0


def nbytes(t):
    import gguf
    qt = gguf.GGMLQuantizationType(t["type"])
    bs, ts = gguf.GGML_QUANT_SIZES[qt]
    n = 1
    for d in t["dims"]:
        n *= d
    return n // bs * ts


if __name__ == "__main__":
    sys.path.insert(0, "/work/src/llama.cpp/gguf-py")
    cmd, repo, fn = sys.argv[1:4]
    url = hf_hub_url(repo, fn)
    kv, tens, data0 = header(url)
    if cmd == "list":
        print("kv keys", len(kv), "split", kv.get("split.no"), "of", kv.get("split.count"), "data0", data0)
        for t in tens:
            print(t["name"], t["type"], t["dims"], t["offset"])
    elif cmd == "fetch":
        prefix, outdir = sys.argv[4], sys.argv[5]
        os.makedirs(outdir, exist_ok=True)
        infos = []
        for t in tens:
            if not t["name"].startswith(prefix):
                continue
            n = nbytes(t)
            a = data0 + t["offset"]
            raw = b""
            while len(raw) < n:
                raw += rng(url, a + len(raw), a + min(n, len(raw) + (64 << 20)))
            open(os.path.join(outdir, t["name"] + ".bin"), "wb").write(raw)
            infos.append(dict(t, nbytes=n))
            print("fetched", t["name"], t["type"], t["dims"], n, flush=True)
        json.dump(dict(repo=repo, file=fn, tensors=infos), open(os.path.join(outdir, "infos.json"), "w"), indent=1)
