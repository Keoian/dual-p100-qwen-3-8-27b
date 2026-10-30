# parity.py (task 2a): the torch MTP head (mtp/head.py) against llama.cpp's own draft logits, from an LLAMA_MTP_DUMP run.
# For every dumped draft step: the head's KV context is rebuilt from the dump's committed catch-up rows (teacher-forced,
# target h), depth d > 1 appends the cycle's earlier draft rows with llama.cpp's own h_in ("tf": each step checked on
# its own) or the torch head's own h_out ("chain": the whole draft chain in torch). Compared on the dumped top-64:
#   max/mean |d logit|, |d lse|, top-1 agreement, KL(q_llama || q_torch) over the draft's T 1 top-20 (what it samples).
#   python3 parity.py <dump> [gguf] [--kv q4_0|q8_0|f32] [--mtp-src DIR] [--max-req N] [--device cpu|cuda]
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp import dump as D, weights as W
from mtp.head import MTPHead


def draft_lm_head(G, draft_vocab=81920, draft_type="q4_0"):
    """the MTP draft's own LM head as llama.cpp builds it at load (src/llama-model.cpp, LLAMA_MTP_DRAFT_VOCAB /
    LLAMA_MTP_DRAFT_VOCAB_TYPE): the first draft_vocab rows of output.weight, re-quantized from its values to q4_0
    ("same": output's own type). draft_vocab 0: the full output head."""
    from mtp.head import q4_0_roundtrip
    W0 = torch.from_numpy(G.tensor(G.spec.lm_head))
    if draft_vocab and draft_vocab < W0.shape[0]:
        W0 = W0[:draft_vocab].clone()
        if draft_type == "q4_0":
            for i in range(0, W0.shape[0], 8192):
                W0[i:i + 8192] = q4_0_roundtrip(W0[i:i + 8192])
    return W0


def load(gguf, kv, mtp_src, device, lm_dtype=torch.float32, emulate=None, draft_vocab=81920, draft_type="q4_0"):
    G = W.GGUFHead(gguf)
    ov = W.bf16_overrides(mtp_src, G.prefix) if mtp_src else None
    block = G.block(ov)
    t0 = time.time()
    lm = draft_lm_head(G, draft_vocab, draft_type).to(device=device, dtype=lm_dtype)
    print(f"draft LM head: {G.type_of(G.spec.lm_head)} rows {lm.shape[0]} as {draft_type if draft_vocab else 'is'} "
          f"in {time.time() - t0:.0f} s", flush=True)
    cache = {}

    def embd(tokens):
        t = tokens.cpu().numpy()
        miss = sorted({int(x) for x in t} - cache.keys())
        if miss:
            rows = G.tensor(G.spec.embd, rows=np.array(miss))
            for i, m in enumerate(miss):
                cache[m] = rows[i]
        return torch.from_numpy(np.stack([cache[int(x)] for x in t])).to(device)

    types = G.block_types()
    if mtp_src:
        types = {k: ("F32" if v in ("Q6_K", "Q8_0") and k in ov else v) for k, v in types.items()}
    head = MTPHead(G.spec, G.hp, block, embd, lm, kv_type=kv, emulate=emulate, types=types,
                   lm_type=("Q4_0" if draft_vocab and draft_type == "q4_0" else G.type_of(G.spec.lm_head))).to(device)
    return G, head


def topk_metrics(rec, lg_t, k20=20):
    ids = torch.as_tensor(rec["ids"], dtype=torch.long, device=lg_t.device)
    ll = torch.as_tensor(rec["logits"])
    lt = lg_t[ids].cpu()
    lse_t = float(torch.logsumexp(lg_t, 0))
    d = (lt - ll).abs()
    # the draft's sampling list: top-20 at T 1 (renormalized), as production samples it
    pl = torch.softmax(ll[:k20], 0); pt = torch.softmax(lt[:k20], 0)
    kl = float((pl * (pl.clamp_min(1e-30).log() - pt.clamp_min(1e-30).log())).sum())
    top1 = int(torch.argmax(lg_t).cpu()) == int(rec["ids"][0])
    return dict(maxd=float(d.max()), meand=float(d.mean()), dlse=abs(lse_t - rec["lse"]), top1=top1, kl=kl,
                tv=float(0.5 * (pl - pt).abs().sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("gguf", nargs="?", default="/work/models/Qwen3.8-27B-UD-Q6_K.gguf")
    ap.add_argument("--kv", default="q4_0")
    ap.add_argument("--mtp-src", default=None)
    ap.add_argument("--max-req", type=int, default=1000)
    ap.add_argument("--skip", default="", help="comma list of request indices to leave out (e.g. a NaN run)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--emulate", default=None, help="cpu: ggml-cpu's activation quantization (parity vs a CPU run)")
    ap.add_argument("--draft-vocab", type=int, default=81920, help="LLAMA_MTP_DRAFT_VOCAB of the run (0: full head)")
    ap.add_argument("--draft-type", default="q4_0", help="LLAMA_MTP_DRAFT_VOCAB_TYPE of the run (q4_0 | same)")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    G, head = load(a.gguf, a.kv, a.mtp_src, a.device, emulate=a.emulate, draft_vocab=a.draft_vocab, draft_type=a.draft_type)
    recs = D.read(a.dump)
    reqs = D.requests(recs)[:a.max_req]
    skip = {int(x) for x in a.skip.split(",") if x}
    M = {"tf": [], "chain": []}
    for ri, req in enumerate(reqs):
        if ri in skip:
            continue
        tok, h, pos = D.committed(req)
        dev = a.device
        T = torch.from_numpy(tok).to(dev); Hh = torch.from_numpy(h).to(dev); P = torch.from_numpy(pos).to(dev)
        hout, (K, V, PP) = head.forward_seq(T, Hh, P)
        for cyc in D.cycles(req):
            p0 = cyc[0]["pos"]
            assert cyc[0]["tok"] == tok[p0] and np.array_equal(cyc[0]["h"], h[p0]), "draft step 0 input != committed row"
            cache = (K[:p0], V[:p0], PP[:p0])
            cache_c = cache
            hc = None
            for r in cyc:
                tt = torch.tensor([r["tok"]], device=dev); pp = torch.tensor([r["pos"]], device=dev)
                # tf: llama.cpp's own h_in at this step
                ho, cache = head.step(tt, torch.from_numpy(r["h"][None]).to(dev), pp, cache)
                m = topk_metrics(r, head.logits(ho)[0]); m.update(step=r["step"], req=ri, pos=r["pos"])
                M["tf"].append(m)
                # chain: the torch head's own h_out from the previous step (step 0 = target h)
                hin = torch.from_numpy(r["h"][None]).to(dev) if r["step"] == 0 else hc
                hc, cache_c = head.step(tt, hin, pp, cache_c)
                m = topk_metrics(r, head.logits(hc)[0]); m.update(step=r["step"], req=ri, pos=r["pos"])
                M["chain"].append(m)
        n = len(M["tf"])
        print(f"request {ri}: {len(tok)} positions, {n} draft steps so far; tf top-1 "
              f"{np.mean([m['top1'] for m in M['tf']]):.4f}, max|dlogit| {max(m['maxd'] for m in M['tf']):.3f}", flush=True)
    lines = [f"# parity: torch MTP head vs llama.cpp draft logits ({a.dump}, KV {a.kv}, MTP src {a.mtp_src or 'GGUF'})", ""]
    lines.append("| mode | depth | steps | top-1 agree | max abs dlogit (top-64) | mean abs dlogit | p99 abs dlogit | mean abs dlse | mean KL top-20 | mean TV top-20 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for mode in ("tf", "chain"):
        for dsel in (None, 0, 1, 2, 3):
            ms = [m for m in M[mode] if dsel is None or m["step"] == dsel]
            if not ms:
                continue
            md = np.array([m["maxd"] for m in ms])
            lines.append(f"| {mode} | {'all' if dsel is None else dsel + 1} | {len(ms)} | {np.mean([m['top1'] for m in ms]):.4f} | "
                         f"{md.max():.4f} | {np.mean([m['meand'] for m in ms]):.4f} | {np.percentile(md, 99):.4f} | "
                         f"{np.mean([m['dlse'] for m in ms]):.4f} | {np.mean([m['kl'] for m in ms]):.2e} | {np.mean([m['tv'] for m in ms]):.4f} |")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        json.dump(M, open(a.out + ".json", "w"))


if __name__ == "__main__":
    main()
