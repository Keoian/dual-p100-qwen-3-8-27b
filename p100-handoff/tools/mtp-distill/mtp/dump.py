# dump.py: reader for LLAMA_MTP_DUMP files (common/speculative.cpp, CHANGES §25). Records, little-endian:
#   int32 kind (1 catch-up row, 2 draft step), seq, step (-1 for catch-up), pos, token, n_embd, float h_in[n_embd];
#   kind 2 adds int32 k, k x (int32 id, float logit), float lse, int32 drafted (-1: this step stopped drafting);
#   kind 3 (target raw output at a decoded row, n_embd 0, no h): the same top-k (128) logits, lse, -1
import numpy as np


def read(path, keep_h=True):
    buf = np.fromfile(path, np.uint8)
    off = 0
    recs = []
    n = len(buf)
    while off + 24 <= n:
        kind, seq, step, pos, tok, ne = (int(x) for x in np.frombuffer(buf, np.int32, 6, off))   # Python ints: offsets pass 2 GiB
        off += 24
        h = np.frombuffer(buf, np.float32, ne, off) if keep_h else None
        off += 4 * ne
        r = dict(kind=int(kind), seq=int(seq), step=int(step), pos=int(pos), tok=int(tok), h=h)
        if kind in (2, 3):
            k = int(np.frombuffer(buf, np.int32, 1, off)[0]); off += 4
            pairs = np.frombuffer(buf, np.uint8, 8 * k, off).view(np.int32).reshape(k, 2); off += 8 * k
            r["ids"] = pairs[:, 0].copy()
            r["logits"] = pairs[:, 1].copy().view(np.float32)
            r["lse"] = float(np.frombuffer(buf, np.float32, 1, off)[0]); off += 4
            r["drafted"] = int(np.frombuffer(buf, np.int32, 1, off)[0]); off += 4
        recs.append(r)
    return recs


def requests(recs):
    """split into requests: a catch-up row at position 0 starts one (the dump run uses cache_prompt = false)."""
    out, cur = [], []
    for r in recs:
        if r["kind"] == 1 and r["pos"] == 0 and cur:
            out.append(cur); cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def committed(req):
    """last catch-up row per position (rows past a rejection are rewritten by the next verify) -> tokens, h, positions."""
    last = {}
    for r in req:
        if r["kind"] == 1:
            last[r["pos"]] = r
    P = sorted(last)
    assert P == list(range(len(P))), "gap in the committed positions"
    return (np.array([last[p]["tok"] for p in P], np.int64), np.stack([last[p]["h"] for p in P]), np.array(P, np.int64))


def cycles(req):
    """draft cycles: lists of consecutive kind-2 records starting at step 0."""
    out, cur = [], []
    for r in req:
        if r["kind"] != 2:
            continue
        if r["step"] == 0 and cur:
            out.append(cur); cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out
