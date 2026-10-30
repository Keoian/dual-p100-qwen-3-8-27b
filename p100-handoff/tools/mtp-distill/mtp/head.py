# head.py: torch reimplementation of the MTP draft head (spec.py documents the math). One module serves the parity check
# (fp32, frozen, llama.cpp's KV quantization emulated) and distillation (trainable block, frozen embedding/LM head).
import math
import torch
import torch.nn.functional as F


def hadamard(n, device=None, dtype=torch.float32):
    """ggml_gen_hadamard: Sylvester construction, normalized (H @ H = I, symmetric)."""
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(n)).to(device=device, dtype=dtype)


def q4_0_roundtrip(x):
    """ggml quantize_row_q4_0_ref + dequantize along the last dim (blocks of 32), fp16 scale."""
    s = x.shape
    b = x.reshape(-1, 32)
    idx = b.abs().argmax(1, keepdim=True)
    mx = torch.gather(b, 1, idx)
    d = (mx / -8.0)
    d16 = d.to(torch.float16).to(torch.float32)
    idd = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp((b * idd + 8.5).to(torch.int8).to(torch.float32), max=15.0)  # (int8_t) truncation, then MIN(15, .)
    return ((q - 8.0) * d16).reshape(s)


def q8_0_roundtrip(x):
    s = x.shape
    b = x.reshape(-1, 32)
    amax = b.abs().amax(1, keepdim=True)
    d = amax / 127.0
    idd = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.round(b * idd)
    return (q * d.to(torch.float16).to(torch.float32)).reshape(s)


def q8_k_roundtrip(x):
    """ggml quantize_row_q8_K_ref (blocks of 256, iscale -127/max signed, fp32 scale) + dequantize."""
    s = x.shape
    b = x.reshape(-1, 256)
    idx = b.abs().argmax(1, keepdim=True)
    mx = torch.gather(b, 1, idx)
    isc = torch.where(mx != 0, -127.0 / mx, torch.zeros_like(mx))
    q = torch.clamp(torch.round(isc * b), max=127.0)
    d = torch.where(mx != 0, 1.0 / isc, torch.zeros_like(mx))
    return (q * d).reshape(s)


KVQ = {None: None, "f32": None, "q4_0": q4_0_roundtrip, "q8_0": q8_0_roundtrip,
       "f16": lambda x: x.to(torch.float16).to(torch.float32)}
# activation quantization ahead of a quantized matmul, per weight type, as ggml-cpu's vec_dot_type
ACT_CPU = {"Q6_K": q8_k_roundtrip, "Q5_K": q8_k_roundtrip, "Q4_K": q8_k_roundtrip, "Q8_0": q8_0_roundtrip,
           "Q4_0": q8_0_roundtrip}
# ggml-cuda's mmvq / mmq: activations as q8_1 (blocks of 32, d = amax/127) for every quantized weight type
ACT_GPU = {k: q8_0_roundtrip for k in ("Q6_K", "Q5_K", "Q4_K", "Q8_0", "Q4_0")}
ACT = {"cpu": ACT_CPU, "gpu": ACT_GPU}


class MTPHead(torch.nn.Module):
    def __init__(self, spec, hp, block, embd_fn, lm_head, kv_type=None, trainable=False, emulate=None, types=None,
                 lm_type=None):
        """block: {relative tensor name: np array}; embd_fn(tokens) -> (n, d) float tensor; lm_head: (vocab, d) tensor.
        emulate="cpu": quantize activations ahead of each quantized matmul (types = {name: GGML type name}) and the
        query of a quantized-K attention, as ggml-cpu does (parity checks against a CPU run only)."""
        super().__init__()
        self.emulate, self.types, self.lm_type = emulate, types or {}, lm_type
        self.spec, self.hp = spec, hp
        self.p = torch.nn.ParameterDict({k.replace(".", "__"): torch.nn.Parameter(torch.as_tensor(v), requires_grad=trainable)
                                         for k, v in block.items()})
        self.embd_fn = embd_fn
        self.lm_head = lm_head
        self.kv_type = kv_type
        d = hp.head_dim
        inv = hp.rope_base ** (-torch.arange(0, hp.n_rot, 2, dtype=torch.float64) / hp.n_rot)
        self.register_buffer("inv_freq", inv.to(torch.float32), persistent=False)
        self.register_buffer("Hk", hadamard(spec.kv_rot_k), persistent=False)
        self.register_buffer("Hv", hadamard(spec.kv_rot_v), persistent=False)

    def w(self, name):
        return self.p[name.replace(".", "__")]

    def mm(self, x, name):
        if self.emulate in ACT and self.types.get(name) in ACT[self.emulate]:
            x = ACT[self.emulate][self.types[name]](x)
        return x @ self.w(name).t()

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.hp.eps) * w

    def rope(self, x, pos):
        """x (n, heads, head_dim); NeoX pairs (i, i + n_rot/2) over the first n_rot dims."""
        nr = self.hp.n_rot
        ang = pos.to(torch.float32)[:, None] * self.inv_freq[None, :]          # (n, nr/2)
        c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
        x1, x2 = x[..., :nr // 2], x[..., nr // 2:nr]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, x[..., nr:]], -1)

    def pre_attn(self, tokens, h_in, pos):
        """per-row: z (residual), q (rotated, normed), gate, k, v as stored (quantization emulated)."""
        hp = self.hp
        e = self.embd_fn(tokens)
        en, hn = self.rms(e, self.w("nextn.enorm.weight")), self.rms(h_in, self.w("nextn.hnorm.weight"))
        x = torch.cat([en, hn] if self.spec.concat_embd_first else [hn, en], -1)
        z = self.mm(x, "nextn.eh_proj.weight")
        a = self.rms(z, self.w("attn_norm.weight"))
        n = a.shape[0]
        qf = self.mm(a, "attn_q.weight").view(n, hp.n_head, 2 * hp.head_dim)
        q, gate = qf[..., :hp.head_dim], qf[..., hp.head_dim:]
        k = self.mm(a, "attn_k.weight").view(n, hp.n_head_kv, hp.head_dim)
        v = self.mm(a, "attn_v.weight").view(n, hp.n_head_kv, hp.head_dim)
        q = self.rms(q, self.w("attn_q_norm.weight"))
        k = self.rms(k, self.w("attn_k_norm.weight"))
        q, k = self.rope(q, pos), self.rope(k, pos)
        quant = KVQ[self.kv_type]
        if quant is not None:
            # stored Hadamard-rotated and quantized; the attention sees rotated Q against rotated K, and rotated V
            Hk, Hv = self.Hk, self.Hv
            nv = hp.head_dim // Hv.shape[0]
            q = q @ Hk
            if self.emulate in ACT:
                q = q8_0_roundtrip(q)   # flash attention with quantized K: Q as q8_0 (cpu) / q8_1 (cuda vec kernel)
            k = quant(k @ Hk)
            v = quant((v.reshape(n, hp.n_head_kv, nv, Hv.shape[0]) @ Hv).reshape(n, hp.n_head_kv, hp.head_dim))
        return z, q, gate, k, v

    def post_attn(self, z, att, gate):
        """att (n, heads, head_dim) attention output (rotated back if needed) -> h_out (n, d)."""
        n = z.shape[0]
        o = self.mm((att * torch.sigmoid(gate)).reshape(n, -1), "attn_output.weight")
        r = z + o
        m = self.rms(r, self.w("post_attention_norm.weight"))
        f = self.mm(F.silu(self.mm(m, "ffn_gate.weight")) * self.mm(m, "ffn_up.weight"), "ffn_down.weight")
        y = r + f
        return self.rms(y, self.w("nextn.shared_head_norm.weight"))

    def attend(self, q, K, V, q_pos, k_pos):
        """q (n, H, D) against K/V (m, Hkv, D) with causal mask k_pos <= q_pos."""
        hp = self.hp
        g = hp.n_head // hp.n_head_kv
        Kx = K.repeat_interleave(g, dim=1).transpose(0, 1)      # (H, m, D)
        Vx = V.repeat_interleave(g, dim=1).transpose(0, 1)
        Q = q.transpose(0, 1)                                    # (H, n, D)
        mask = (k_pos[None, :] <= q_pos[:, None])
        s = (Q @ Kx.transpose(1, 2)) / math.sqrt(hp.head_dim)
        s = s.masked_fill(~mask[None], float("-inf"))
        out = (torch.softmax(s, -1) @ Vx).transpose(0, 1)       # (n, H, D)
        if KVQ[self.kv_type] is not None:
            nv = hp.head_dim // self.Hv.shape[0]
            out = (out.reshape(out.shape[0], hp.n_head, nv, -1) @ self.Hv).reshape(out.shape)
        return out

    def forward_seq(self, tokens, h_in, pos, chunk=256):
        """teacher-forced causal pass over a whole sequence. Returns h_out (n, d) and the K/V cache (as stored)."""
        z, q, gate, k, v = self.pre_attn(tokens, h_in, pos)
        outs = []
        for i in range(0, len(tokens), chunk):
            sl = slice(i, i + chunk)
            outs.append(self.attend(q[sl], k[:i + chunk], v[:i + chunk], pos[sl], pos[:i + chunk]))
        h = self.post_attn(z, torch.cat(outs, 0), gate)
        return h, (k, v, pos)

    def step(self, tokens, h_in, pos, cache):
        """rows appended after a cache (k, v, pos): attention over cache + these rows (causal)."""
        z, q, gate, k, v = self.pre_attn(tokens, h_in, pos)
        K = torch.cat([cache[0], k]); V = torch.cat([cache[1], v]); P = torch.cat([cache[2], pos])
        h = self.post_attn(z, self.attend(q, K, V, pos, P), gate)
        return h, (K, V, P)

    def draft_rows(self, tokens, h_in, pos, Kc, Vc, Pc, cutoff, Ke=None, Ve=None, chunk=128):
        """a batch of draft-step rows, each attending to the committed cache rows with position <= its cutoff (pos0)
        plus its own cycle's earlier draft rows Ke/Ve (n, e, Hkv, D) and itself. Returns h_out and the rows' own k, v."""
        hp = self.hp
        z, q, gate, k, v = self.pre_attn(tokens, h_in, pos)
        g = hp.n_head // hp.n_head_kv
        outs = []
        for i in range(0, len(tokens), chunk):
            sl = slice(i, i + chunk)
            qq = q[sl]                                                         # (n, H, D)
            Kx = Kc.repeat_interleave(g, dim=1); Vx = Vc.repeat_interleave(g, dim=1)   # (m, H, D)
            sc = torch.einsum("nhd,mhd->nhm", qq, Kx) / math.sqrt(hp.head_dim)
            sc = sc.masked_fill(~(Pc[None, None, :] <= cutoff[sl][:, None, None]), float("-inf"))
            own_k = k[sl][:, None]; own_v = v[sl][:, None]                     # (n, 1, Hkv, D)
            ek = own_k if Ke is None else torch.cat([Ke[sl], own_k], 1)
            ev = own_v if Ve is None else torch.cat([Ve[sl], own_v], 1)
            ekx = ek.repeat_interleave(g, dim=2); evx = ev.repeat_interleave(g, dim=2)  # (n, e, H, D)
            se = torch.einsum("nhd,nehd->nhe", qq, ekx) / math.sqrt(hp.head_dim)
            w = torch.softmax(torch.cat([sc, se], -1), -1)
            o = torch.einsum("nhm,mhd->nhd", w[..., :sc.shape[-1]], Vx) + torch.einsum("nhe,nehd->nhd", w[..., sc.shape[-1]:], evx)
            if KVQ[self.kv_type] is not None:
                nv = hp.head_dim // self.Hv.shape[0]
                o = (o.reshape(o.shape[0], hp.n_head, nv, -1) @ self.Hv).reshape(o.shape)
            outs.append(o)
        h = self.post_attn(z, torch.cat(outs, 0), gate)
        return h, k, v

    def logits(self, h):
        """the draft's logits: lm_head rows (a draft vocab head of the first n rows scores only those; the rest -inf)."""
        if self.emulate in ACT and self.lm_type in ACT[self.emulate]:
            h = ACT[self.emulate][self.lm_type](h)
        lg = h @ self.lm_head.t()
        if lg.shape[-1] < self.hp.n_vocab:
            lg = torch.cat([lg, lg.new_full(lg.shape[:-1] + (self.hp.n_vocab - lg.shape[-1],), float("-inf"))], -1)
        return lg
