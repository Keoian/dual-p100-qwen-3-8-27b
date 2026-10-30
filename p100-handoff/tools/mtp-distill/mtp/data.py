# data.py: LLAMA_MTP_DUMP -> distillation shards (one .npz per request). What the head needs to be trained the way it runs:
#   tok   int32 (n,)        committed tokens, positions 0..n-1 (prompt + reply)
#   h     float16 (n, d)    the head's input at each position: the target's h at pos-1 (row 0: zeros)
#   tq    int32 (n, K) / tl float16 (n, K) / tlse float32 (n,): the target's raw top-K logits + full lse at each committed
#                           position (predicting pos+1), from the verify / prefill row; -1 where none was logged (prompt)
#   win   int32 (n, 64)     presence window (the last 64 committed tokens up to and including pos), -1 padded
#   cyc_pos0 int32 (m,), cyc_tok int32 (m, 4), cyc_n int32 (m,): each draft cycle's position and its drafted tokens
#   cyc_tq / cyc_tl / cyc_tlse (m, 4, K): the target's raw top-K at each drafted row (teacher-forced on the drafted prefix,
#                           i.e. the teacher for depth 2..4 on the head's own rollout), -1 where the row was not verified
# The served distribution (request chain: temperature, top-k, top-p, presence) is applied at training time from these
# raw logits, so one shard serves any sampling setting. sampling = the request's settings (from the run).
import os
import numpy as np
from . import dump as D

K = 64


def _top(rec, k=K):
    ids = np.full(k, -1, np.int32); lg = np.zeros(k, np.float16)
    n = min(k, len(rec["ids"]))
    ids[:n] = rec["ids"][:n]; lg[:n] = rec["logits"][:n]
    return ids, lg, np.float32(rec["lse"])


def compact_request(req):
    tok, h, pos = D.committed(req)
    n = len(tok)
    tq = np.full((n, K), -1, np.int32); tl = np.zeros((n, K), np.float16); tlse = np.full(n, np.nan, np.float32)
    last3 = {}
    for r in req:
        if r["kind"] == 3:
            last3[r["pos"]] = r
    for p, r in last3.items():
        if p < n and r["tok"] == tok[p]:
            tq[p], tl[p], tlse[p] = _top(r)
    win = np.full((n, 64), -1, np.int32)
    for p in range(n):
        w = tok[max(0, p - 63):p + 1]
        win[p, :len(w)] = w
    cyc = []
    i = 0; recs = req; m = len(recs)
    while i < m:
        if recs[i]["kind"] == 2 and recs[i]["step"] == 0:
            c = []
            while i < m and recs[i]["kind"] == 2 and (not c or recs[i]["step"] > 0):
                c.append(recs[i]); i += 1
            ver = {}
            while i < m and recs[i]["kind"] == 3:
                ver[recs[i]["pos"]] = recs[i]; i += 1
            drafted = [r["drafted"] for r in c if r["drafted"] >= 0]
            p0 = c[0]["pos"]
            ct = np.full((4, K), -1, np.int32); cl = np.zeros((4, K), np.float16); cs = np.full(4, np.nan, np.float32)
            for s in range(min(len(drafted) + 1, 4)):   # row s: input drafted[s-1] (row 0: id_last); s = n is the bonus row
                r = ver.get(p0 + s)
                if r is not None and (s == 0 or r["tok"] == drafted[s - 1]):
                    ct[s], cl[s], cs[s] = _top(r)
            cyc.append((p0, drafted, ct, cl, cs))
        else:
            i += 1
    mc = len(cyc)
    out = dict(tok=tok.astype(np.int32), h=h.astype(np.float16), tq=tq, tl=tl, tlse=tlse, win=win,
               cyc_pos0=np.array([c[0] for c in cyc], np.int32),
               cyc_tok=np.array([(c[1] + [-1] * 4)[:4] for c in cyc], np.int32).reshape(mc, 4),
               cyc_n=np.array([len(c[1]) for c in cyc], np.int32),
               cyc_tq=np.stack([c[2] for c in cyc]) if cyc else np.zeros((0, 4, K), np.int32),
               cyc_tl=np.stack([c[3] for c in cyc]) if cyc else np.zeros((0, 4, K), np.float16),
               cyc_tlse=np.stack([c[4] for c in cyc]) if cyc else np.zeros((0, 4), np.float32))
    return out


def compact(dump_path, out_dir, sampling, skip=(0,)):
    """one shard per request (request indices in skip, e.g. the warm-up, are left out); returns the shard paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ri, req in enumerate(D.requests(D.read(dump_path))):
        if ri in skip:
            continue
        s = compact_request(req)
        p = os.path.join(out_dir, f"req{ri:05d}.npz")
        np.savez(p, **s, sampling=np.array([sampling["temp"], sampling["top_k"], sampling["top_p"], sampling["min_p"],
                                            sampling["pres"]], np.float32))
        paths.append(p)
    return paths
