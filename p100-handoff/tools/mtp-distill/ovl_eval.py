# ovl_eval.py: OFFLINE, PAIRED. Exact-acceptance overlap sum(min(p, q)) on the drafted rows of an LLAMA_MTP_DUMP run, for
# draft variants computed with the torch head (mtp/head.py) on the SAME rows: p = the target's served distribution at the
# row (the verify's kind-3 record at that position: top-128 raw logits + the request's chain and presence window), q = the
# draft's sampling list (production: top-20 at T 1; "req": the request's chain, 1c). Teacher-forced on llama.cpp's own
# draft inputs (drafted prefix and h_in), so variants differ only in the head's numerics / sampling.
# Batched: depth 1 = one causal pass over the committed sequence; depth d > 1 = rows attending to the committed cache
# (<= pos0) plus their cycle's earlier draft rows. Head variants re-score the same hidden states.
#   python3 ovl_eval.py <dump> --reqs chat:0-24,think:25-49 [--device cpu|cuda] [--out md]
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp import dump as D, weights as W
from mtp.head import MTPHead
import parity

SAMPLING = {"chat": dict(temp=0.7, top_k=20, top_p=0.8, min_p=0.0, pres=1.5),
            "think": dict(temp=1.0, top_k=20, top_p=0.95, min_p=0.0, pres=0.0)}
MQ8 = "/work/bench/mtp5/models/Qwen3.8-27B-UD-Q6_K-mtpq8.gguf"


def served(ids, logits, window, S):
    """llama.cpp's default chain on top-k raw logits: presence -> top-k -> top-p (T 1) -> min-p -> temperature."""
    lg = logits.astype(np.float64).copy()
    if S["pres"] and window:
        lg = lg - S["pres"] * np.isin(ids, list(window))
    o = np.argsort(-lg, kind="stable"); lg = lg[o]; ids = ids[o]
    lg, ids = lg[:S["top_k"]], ids[:S["top_k"]]
    p1 = np.exp(lg - lg[0]); p1 /= p1.sum()
    keep = len(lg)
    if S["top_p"] < 1:
        c = np.cumsum(p1); hit = np.nonzero(c >= S["top_p"])[0]
        keep = int(hit[0]) + 1 if len(hit) else len(lg)
    if S["min_p"] > 0:
        keep = min(keep, max(1, int((lg >= lg[0] + np.log(S["min_p"])).sum())))
    lg, ids = lg[:keep], ids[:keep]
    e = np.exp((lg - lg[0]) / S["temp"]); return ids, e / e.sum()


def draft_q(v, i, mode, S, window):
    """the draft's list from its top-20 logits v (desc) / ids i: 'prod' T 1; 'req' the request's chain."""
    if mode == "prod":
        e = np.exp(v - v[0]); return i, e / e.sum()
    return served(i, v, window, S)


def overlap(pi, pp, qi, qq):
    d = dict(zip(pi.tolist(), pp.tolist()))
    return float(sum(min(d.get(int(t), 0.0), float(w)) for t, w in zip(qi, qq)))


def cycles_with_teacher(req):
    """[(cycle records (drafted steps only), {pos: kind-3 verify record})] in file order."""
    out = []; i = 0; n = len(req)
    while i < n:
        if req[i]["kind"] == 2 and req[i]["step"] == 0:
            cyc = []
            while i < n and req[i]["kind"] == 2 and (not cyc or req[i]["step"] > 0):
                cyc.append(req[i]); i += 1
            ver = {}
            while i < n and req[i]["kind"] == 3:
                ver[req[i]["pos"]] = req[i]; i += 1
            out.append(([r for r in cyc if r["drafted"] >= 0], ver))
        else:
            i += 1
    return out


def parse_reqs(spec, nreq):
    m = {}
    for part in spec.split(","):
        mode, rng = part.split(":"); a, b = rng.split("-")
        for k in range(int(a), min(int(b), nreq)):
            m[k] = mode
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("--reqs", required=True, help="mode:first-last,... (request index ranges)")
    ap.add_argument("--gguf", default="/work/models/Qwen3.8-27B-UD-Q6_K.gguf")
    ap.add_argument("--device", default="cpu", help="device of the head blocks"); ap.add_argument("--emulate", default=None)
    ap.add_argument("--head-device", default=None, help="device of the LM-head variants (default: --device)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = a.device; hdev = a.head_device or dev
    G = W.GGUFHead(a.gguf)
    t0 = time.time()
    Wq8 = torch.from_numpy(G.tensor(G.spec.lm_head))
    HEADS = {"q4_0/81920": parity.draft_lm_head(G, 81920, "q4_0").to(hdev), "q8_0/81920": Wq8[:81920].clone().to(hdev),
             "q4_0/98304": parity.draft_lm_head(G, 98304, "q4_0").to(hdev), "q8_0/full": Wq8.to(hdev)}
    print(f"heads built in {time.time() - t0:.0f} s", flush=True)
    cache = {}

    def embd(tokens):
        t = tokens.cpu().numpy()
        miss = sorted({int(x) for x in t} - cache.keys())
        if miss:
            rws = G.tensor(G.spec.embd, rows=np.array(miss))
            for k, m in enumerate(miss):
                cache[m] = rws[k]
        return torch.from_numpy(np.stack([cache[int(x)] for x in t])).to(dev)

    G8 = W.GGUFHead(MQ8)
    MODELS = {"Q6_K/q4_0": (G.block(), G.block_types(), "q4_0"), "Q6_K/q8_0": (G.block(), G.block_types(), "q8_0"),
              "Q6_K/f16": (G.block(), G.block_types(), "f16"), "Q8_0/q4_0": (G8.block(), G8.block_types(), "q4_0")}
    mods = {k: MTPHead(G.spec, G.hp, b, embd, HEADS["q4_0/81920"], kv_type=kv, emulate=a.emulate, types=ty,
                       lm_type="Q4_0").to(dev) for k, (b, ty, kv) in MODELS.items()}
    # variants: label -> (model, head, q mode)
    V = {"production (Q6_K block, KV q4_0, head q4_0/81920, T1 top-20)": ("Q6_K/q4_0", "q4_0/81920", "prod"),
         "1c sampler parity (request chain)": ("Q6_K/q4_0", "q4_0/81920", "req"),
         "1a draft KV q8_0/q8_0": ("Q6_K/q8_0", "q4_0/81920", "prod"),
         "1a draft KV f16/f16": ("Q6_K/f16", "q4_0/81920", "prod"),
         "1b MTP block Q8_0 (mtpq8.gguf)": ("Q8_0/q4_0", "q4_0/81920", "prod"),
         "1b' draft head q8_0 (81920 rows; VOCAB_TYPE=same)": ("Q6_K/q4_0", "q8_0/81920", "prod"),
         "1b' draft head q4_0, 98304 rows": ("Q6_K/q4_0", "q4_0/98304", "prod"),
         "1b' full q8_0 head (DRAFT_VOCAB=0)": ("Q6_K/q4_0", "q8_0/full", "prod")}
    recs = D.read(a.dump)
    reqs = D.requests(recs)
    modes = parse_reqs(a.reqs, len(reqs))
    R = {m: {lab: [] for lab in V} for m in ("chat", "think")}
    DEP = {"chat": [], "think": []}
    AGREE = {"chat": [], "think": []}
    for ri, req in enumerate(reqs):
        if ri not in modes:
            continue
        mode = modes[ri]; S = SAMPLING[mode]
        tok, h, pos = D.committed(req)
        cyc = cycles_with_teacher(req)
        rows = []   # (cycle index, step, record, teacher, window)
        for ci, (cr, ver) in enumerate(cyc):
            drafted = [r["drafted"] for r in cr]
            p0 = cr[0]["pos"] if cr else None
            for r in cr:
                t = ver.get(r["pos"])
                s = r["step"]
                if t is None or (s > 0 and t["tok"] != drafted[s - 1]):
                    continue
                seq = list(tok[:p0 + 1]) + drafted[:s]
                rows.append((ci, s, r, t, set(seq[-64:])))
        if not rows:
            continue
        T = torch.from_numpy(tok).to(dev); Hh = torch.from_numpy(h).to(dev); P = torch.from_numpy(pos).to(dev)
        H_rows = {}   # model -> (n_rows, d) h_out per row
        for mk, m in mods.items():
            hout, (K, Vv, PP) = m.forward_seq(T, Hh, P)
            hr = torch.zeros(len(rows), G.hp.n_embd, device=dev)
            kv_of = {}   # (cycle, step) -> (k, v) of that draft row
            for s in range(4):
                idx = [j for j, x in enumerate(rows) if x[1] == s]
                if not idx:
                    continue
                if s == 0:
                    hr[idx] = hout[torch.tensor([rows[j][2]["pos"] for j in idx], device=dev)]
                    for j in idx:
                        p0 = rows[j][2]["pos"]
                        kv_of[(rows[j][0], 0)] = (K[p0], Vv[p0])
                    continue
                ok = [j for j in idx if all((rows[j][0], t) in kv_of for t in range(1, s))]
                tt = torch.tensor([rows[j][2]["tok"] for j in ok], device=dev)
                hi = torch.from_numpy(np.stack([rows[j][2]["h"] for j in ok])).to(dev)
                pp = torch.tensor([rows[j][2]["pos"] for j in ok], device=dev)
                cut = pp - s
                Ke = Ve = None
                if s > 1:
                    Ke = torch.stack([torch.stack([kv_of[(rows[j][0], t)][0] for t in range(1, s)]) for j in ok])
                    Ve = torch.stack([torch.stack([kv_of[(rows[j][0], t)][1] for t in range(1, s)]) for j in ok])
                ho, kk, vv = m.draft_rows(tt, hi, pp, K, Vv, PP, cut, Ke, Ve)
                hr[ok] = ho
                for n_, j in enumerate(ok):
                    kv_of[(rows[j][0], s)] = (kk[n_], vv[n_])
            H_rows[mk] = hr
        # score every variant
        top = {}
        for lab, (mk, hk, qm) in V.items():
            key = (mk, hk)
            if key not in top:
                L = H_rows[mk].to(hdev) @ HEADS[hk].t()
                v, i = torch.topk(L, 20, dim=1)
                top[key] = (v.double().cpu().numpy(), i.cpu().numpy())
            v, i = top[key]
            for j, (ci, s, r, t, win) in enumerate(rows):
                pi, pp_ = served(t["ids"], t["logits"], win, S)
                qi, qq = draft_q(v[j], i[j], qm, S, win)
                R[mode][lab].append(overlap(pi, pp_, qi, qq))
        DEP[mode] += [s for _, s, *_ in rows]
        # sanity: the torch production variant vs llama.cpp's own logged draft (top-1 and TV over its T 1 top-20)
        v, i = top[("Q6_K/q4_0", "q4_0/81920")]
        for j, (ci, s, r, t, win) in enumerate(rows):
            ql = np.exp(r["logits"][:20] - r["logits"][0]); ql /= ql.sum()
            qt = np.exp(v[j] - v[j][0]); qt /= qt.sum()
            d = dict(zip(r["ids"][:20].tolist(), ql.tolist()))
            AGREE[mode].append((int(i[j][0]) == int(r["ids"][0]), 0.5 * (sum(abs(d.get(int(x), 0) - w) for x, w in zip(i[j], qt)) +
                                sum(w for x, w in d.items() if x not in set(i[j].tolist())))))
        print(f"request {ri} ({mode}): {len(rows)} rows; overlap production {np.mean(R[mode][list(V)[0]]):.4f}", flush=True)
    lines = [f"# ovl_eval: {a.dump} (OFFLINE, paired, teacher-forced on llama.cpp's draft inputs)", ""]
    rng = np.random.default_rng(0)
    for mode in ("chat", "think"):
        base = np.array(R[mode][list(V)[0]])
        if not len(base):
            continue
        dep = np.array(DEP[mode]); ag = np.array(AGREE[mode])
        lines += [f"## {mode}: {len(base)} drafted rows; torch production variant vs llama.cpp's logged draft: top-1 agree "
                  f"{ag[:, 0].mean():.4f}, mean TV {ag[:, 1].mean():.4f}", "",
                  "| variant | overlap | vs production (95% CI, rows) | by depth 1/2/3/4 |", "|---|---|---|---|"]
        for lab in V:
            x = np.array(R[mode][lab]); d = x - base
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            lines.append(f"| {lab} | {x.mean():.4f} | {d.mean():+.4f} ({lo:+.4f}..{hi:+.4f}) | "
                         + " / ".join(f"{x[dep == k].mean():.3f}" if (dep == k).any() else "-" for k in range(4)) + " |")
        lines.append("")
    txt = "\n".join(lines); print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        json.dump({m: {k: v for k, v in R[m].items()} for m in R}, open(a.out + ".json", "w"))


if __name__ == "__main__":
    main()
