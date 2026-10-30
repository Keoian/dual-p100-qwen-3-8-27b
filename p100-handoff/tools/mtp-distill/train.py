# train.py (task 2c; NOT RUN beyond --smoke until the operator's OK): on-policy distillation of the MTP head on shards from
# mtp/data.py. Only the MTP block trains (spec.trainable); the trunk's embedding and the draft's LM head stay frozen.
#   depth 1: one causal pass over each request's committed sequence (the head's own KV context, as at inference), loss at
#            every reply position against the target's served distribution there
#   depth 2..4: each logged draft cycle replayed on its drafted tokens with the head's OWN hidden state chained (gradients
#            flow through the chain), loss against the target's served distribution on that drafted prefix
# Loss: total variation TV(p_served, q) = 1 - exact acceptance of a sampled draft, plus lam * KL(p_served || q) for dense
# gradients on tokens the draft ranks low; q = softmax(z / T_draft) over the draft head's rows (T_draft = the temperature
# the draft samples at: the request's under parity sampling). Validation: expected overlap per depth on held-out requests,
# with the deployed sampling list (top-20 at T_draft), every --eval-every steps; checkpoints = trainable tensors only.
#   python3 train.py --train DIR --val DIR --out DIR [--steps N] [--lr 1e-5] [--lam 0.1] [--device cuda:0] [--smoke]
import argparse, glob, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp import weights as W
from mtp.head import MTPHead
import parity


def served_t(tq, tl, tlse, win, S, device):
    """batched served distribution from raw top-K: (ids (n,k), p (n,k)) as torch, llama.cpp's chain order."""
    ids = torch.as_tensor(tq.astype(np.int64), device=device)
    lg = torch.as_tensor(tl.astype(np.float32), device=device)
    valid = ids >= 0
    lg = torch.where(valid, lg, torch.full_like(lg, -1e30))
    if S["pres"]:
        w = torch.as_tensor(win.astype(np.int64), device=device)
        inwin = (ids[:, :, None] == w[:, None, :]).any(-1) & valid
        lg = lg - S["pres"] * inwin.float()
    lg, o = torch.sort(lg, 1, descending=True)
    ids = torch.gather(ids, 1, o)
    k = int(S["top_k"])
    lg, ids = lg[:, :k], ids[:, :k]
    p1 = torch.softmax(lg, 1)
    keep = torch.ones_like(p1, dtype=torch.bool)
    if S["top_p"] < 1:
        c = torch.cumsum(p1, 1)
        keep = (c - p1) < S["top_p"]           # the smallest prefix whose mass reaches top_p
    l2 = torch.where(keep, lg / S["temp"], torch.full_like(lg, -1e30))
    return ids, torch.softmax(l2, 1)


def tv_kl(logits, ids, p, T_draft):
    """TV(p, q) and KL(p || q) with q = softmax(logits / T_draft) over the head's rows; p on ids (n,k)."""
    lq = torch.log_softmax(logits.float() / T_draft, -1)
    lq_p = torch.gather(lq, 1, ids.clamp_min(0)).clamp_min(-30.0)   # tokens the draft head cannot score: constant
    q_p = lq_p.exp()
    tv = 0.5 * ((p - q_p).abs().sum(1) + (1.0 - q_p.sum(1)))   # mass of q outside p's support counts fully
    kl = (p * (torch.log(p.clamp_min(1e-12)) - lq_p)).sum(1)
    return tv, kl


def overlap_list(logits, ids, p, T_draft, topk=20):
    """expected exact acceptance with the deployed sampling list: top-20 of the head at T_draft, renormalized."""
    v, i = torch.topk(logits.float(), topk, dim=1)
    q = torch.softmax(v / T_draft, 1)
    hit = (i[:, :, None] == ids[:, None, :])
    pq = (hit.float() * p[:, None, :]).sum(-1)
    return torch.minimum(pq, q).sum(1)


class Shards:
    def __init__(self, d):
        self.paths = sorted(glob.glob(os.path.join(d, "*.npz")))

    def __len__(self):
        return len(self.paths)

    def get(self, i):
        z = np.load(self.paths[i])
        return {k: z[k] for k in z.files}


def request_loss(head, s, S, T_draft, lam, device, max_cycles=None, eval_only=False):
    """per-request losses: depth-1 rows (reply positions with a teacher) and depth 2..4 on the logged rollouts."""
    tok = torch.as_tensor(s["tok"].astype(np.int64), device=device)
    h = torch.as_tensor(s["h"].astype(np.float32), device=device)
    pos = torch.arange(len(tok), device=device)
    hout, (K, V, P) = head.forward_seq(tok, h, pos)
    out = {}
    m = ~np.isnan(s["tlse"])
    m[-1] = False
    rows = np.nonzero(m)[0]
    if len(rows):
        ids, p = served_t(s["tq"][rows], s["tl"][rows], s["tlse"][rows], s["win"][rows], S, device)
        lg = head.logits(hout[torch.as_tensor(rows, device=device)])
        tv, kl = tv_kl(lg, ids, p, T_draft)
        out[1] = (tv, kl, overlap_list(lg.detach(), ids, p, T_draft))
    # rollouts
    n_c = len(s["cyc_pos0"]) if max_cycles is None else min(max_cycles, len(s["cyc_pos0"]))
    if n_c:
        sel = np.arange(len(s["cyc_pos0"]))
        if n_c < len(sel):
            sel = np.sort(np.random.choice(sel, n_c, replace=False))
        p0 = s["cyc_pos0"][sel]; ctok = s["cyc_tok"][sel]; cn = s["cyc_n"][sel]
        h_prev = hout[torch.as_tensor(p0, device=device)]                 # depth-1 output (own, differentiable)
        Ke = Ve = None
        alive = np.ones(len(sel), bool)
        for d in range(1, 4):
            alive &= (cn >= d) & ~np.isnan(s["cyc_tlse"][sel, d])   # input = drafted[d-1]; teacher = verify row d
            idx = np.nonzero(alive)[0]
            if not len(idx):
                break
            ti = torch.as_tensor(idx, device=device)
            tt = torch.as_tensor(ctok[idx, d - 1].astype(np.int64), device=device)
            pp = torch.as_tensor((p0[idx] + d).astype(np.int64), device=device)
            ho, kk, vv = head.draft_rows(tt, h_prev[ti], pp, K, V, P, pp - d,
                                         None if Ke is None else Ke[ti], None if Ve is None else Ve[ti])
            win = np.full((len(idx), 64), -1, np.int32)   # window: committed up to pos0, then the drafted prefix
            for j, c in enumerate(idx):
                w = list(s["tok"][max(0, p0[c] - 63 + d):p0[c] + 1]) + list(ctok[c, :d])
                w = w[-64:]; win[j, :len(w)] = w
            ids, p = served_t(s["cyc_tq"][sel[idx], d], s["cyc_tl"][sel[idx], d], s["cyc_tlse"][sel[idx], d], win, S, device)
            lg = head.logits(ho)
            tv, kl = tv_kl(lg, ids, p, T_draft)
            out[d + 1] = (tv, kl, overlap_list(lg.detach(), ids, p, T_draft))
            # next depth: carry this depth's h and K/V, rows not alive get zeros (masked out next round)
            hn = torch.zeros_like(h_prev); hn[ti] = ho
            kz = torch.zeros(len(sel), 1, *kk.shape[1:], device=device, dtype=kk.dtype); kz[ti, 0] = kk
            vz = torch.zeros(len(sel), 1, *vv.shape[1:], device=device, dtype=vv.dtype); vz[ti, 0] = vv
            Ke = kz if Ke is None else torch.cat([Ke, kz], 1)
            Ve = vz if Ve is None else torch.cat([Ve, vz], 1)
            h_prev = hn
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train"); ap.add_argument("--val"); ap.add_argument("--out", default="/work/bench/mtp5/ckpt")
    ap.add_argument("--gguf", default="/work/models/Qwen3.8-27B-UD-Q6_K.gguf")
    ap.add_argument("--init", default=None, help="GGUF whose MTP block initializes the head (default --gguf)")
    ap.add_argument("--draft-vocab", type=int, default=81920); ap.add_argument("--draft-type", default="q4_0")
    ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lam", type=float, default=0.1); ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max-cycles", type=int, default=256); ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--t-draft", default="request", help="'request' (parity sampling) or a number (e.g. 1.0)")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--amp", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one forward/backward on one shard, no update saved")
    a = ap.parse_args()
    dev = a.device
    G = W.GGUFHead(a.gguf); Gi = W.GGUFHead(a.init) if a.init else G
    lm = parity.draft_lm_head(G, a.draft_vocab, a.draft_type).to(dev)
    E = {}

    def embd(tokens):
        t = tokens.cpu().numpy()
        miss = sorted({int(x) for x in t} - E.keys())
        if miss:
            rws = G.tensor(G.spec.embd, rows=np.array(miss))
            for k, m in enumerate(miss):
                E[m] = torch.from_numpy(rws[k])
        return torch.stack([E[int(x)] for x in t]).to(dev)

    head = MTPHead(G.spec, G.hp, Gi.block(), embd, lm, kv_type=None, trainable=False).to(dev)
    params = []
    for k, prm in head.p.items():
        rel = k.replace("__", ".")
        prm.requires_grad_(rel in G.spec.trainable)
        if prm.requires_grad:
            params.append(prm)
    print(f"trainable tensors {len(params)}, parameters {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, a.steps)))))
    scaler = torch.cuda.amp.GradScaler(enabled=a.amp)
    tr = Shards(a.train) if a.train else None
    va = Shards(a.val) if a.val else None

    def S_of(s):
        t, k, tp, mp, pr = [float(x) for x in s["sampling"]]
        return dict(temp=t, top_k=int(k), top_p=tp, min_p=mp, pres=pr)

    def T_of(S):
        return S["temp"] if a.t_draft == "request" else float(a.t_draft)

    def evaluate():
        head.eval(); acc = {d: [] for d in range(1, 5)}
        with torch.no_grad():
            for i in range(len(va)):
                s = va.get(i); S = S_of(s)
                for d, (tv, kl, ov) in request_loss(head, s, S, T_of(S), a.lam, dev, max_cycles=None).items():
                    acc[d].append(ov.cpu().numpy())
        head.train()
        return {d: float(np.concatenate(v).mean()) for d, v in acc.items() if v}

    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train.log"), "a")
    if va is not None and not a.smoke:
        ev = evaluate(); print(f"step 0 held-out overlap by depth {ev}", flush=True); logf.write(json.dumps(dict(step=0, val=ev)) + "\n")
    order = list(range(len(tr)))
    t0 = time.time()
    for step in range(1 if a.smoke else a.steps):
        if step % len(order) == 0:
            random.shuffle(order)
        s = tr.get(order[step % len(order)]); S = S_of(s)
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
            out = request_loss(head, s, S, T_of(S), a.lam, dev, max_cycles=a.max_cycles)
        terms = [(tv.float().mean() + a.lam * kl.float().mean()) * len(tv) for tv, kl, _ in out.values()]
        nrow = sum(len(tv) for tv, _, _ in out.values())
        loss = sum(terms) / max(nrow, 1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if a.smoke:
            print(f"smoke: rows {nrow}, loss {float(loss):.4f}, grad norm {float(gn):.3f}, "
                  + ", ".join(f"d{d} overlap {float(ov.mean()):.3f}" for d, (_, _, ov) in out.items()), flush=True)
            return
        scaler.step(opt); scaler.update(); sched.step()
        if step % 10 == 0:
            msg = dict(step=step, loss=float(loss), rows=nrow, gnorm=float(gn), lr=sched.get_last_lr()[0],
                       ov={d: float(ov.mean()) for d, (_, _, ov) in out.items()}, s=round(time.time() - t0))
            print(json.dumps(msg), flush=True); logf.write(json.dumps(msg) + "\n")
        if va is not None and (step + 1) % a.eval_every == 0:
            ev = evaluate()
            print(f"step {step + 1} held-out overlap by depth {ev}", flush=True)
            logf.write(json.dumps(dict(step=step + 1, val=ev)) + "\n"); logf.flush()
            torch.save({k.replace("__", "."): v.detach().cpu() for k, v in head.p.items() if v.requires_grad},
                       os.path.join(a.out, f"head-{step + 1:06d}.pt"))


if __name__ == "__main__":
    main()
