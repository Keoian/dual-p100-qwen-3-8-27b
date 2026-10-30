# spec.py: everything model-specific about an MTP (NextN) draft head, in one place. The rest of the pipeline (weights.py,
# head.py, dump.py, parity.py, the harness and the distillation loop) reads only a HeadSpec. A new model (e.g. Qwen 4's
# 27B) should need a new entry here, plus head.py changes only if its block has a structure not covered by the flags.
#
# Qwen3.5/3.8 dense MTP head, as llama.cpp computes it (src/models/qwen35.cpp, graph_mtp; checked 2026-10-29):
#   inputs per position t: token x_t, and h_in = the target's h_nextn at t-1 (its final hidden state AFTER output_norm),
#     or, at draft depth > 1, the head's own h_nextn from the previous step (after shared_head_norm)
#   e = token_embd[x_t] (the trunk's shared embedding, Q6_K in production)
#   z = eh_proj( concat( rms(e) * enorm, rms(h_in) * hnorm ) )           embedding half FIRST, eh_proj [2d -> d]
#   full-attention decoder block (the trunk's every-4th-layer type):
#     a = rms(z) * attn_norm
#     q|gate = attn_q(a) per head interleaved [q 256 | gate 256] x 24 heads; k = attn_k(a), v = attn_v(a) (4 KV heads)
#     q, k: per-head rms * q_norm / k_norm (256); RoPE NeoX-style on the first 64 of 256 dims (IMRoPE with equal t/h/w
#       positions for text), base 1e7; GQA 24/4, scale 1/sqrt(256), causal over the head's own KV cache
#     o = attn_output( attn(q, k, v) * sigmoid(gate) ); r = z + o
#     f = ffn_down( silu(ffn_gate(m)) * ffn_up(m) ), m = rms(r) * post_attention_norm; y = r + f
#   h_out = rms(y) * shared_head_norm         (the h_in of the next draft depth)
#   logits = output.weight @ h_out            (the trunk's shared LM head, Q8_0 in production)
#   The head's KV cache holds, for every committed position, the K/V computed from (x_t, target h_{t-1}) (catch-up);
#   draft depth d adds its own rows at pos0+1..pos0+d-1 (own h). Quantized KV (q4_0 here) is stored Hadamard-rotated:
#   K (and Q) by a 256x256 Walsh-Hadamard matrix per head, V by 64x64 blocks (the output is rotated back).
from dataclasses import dataclass, field


@dataclass
class HeadSpec:
    name: str
    arch: str                      # GGUF general.architecture, prefix of the hparam keys
    block: str                     # format of the MTP block's tensor prefix, given the trunk's layer count L
    embd: str = "token_embd.weight"
    lm_head: str = "output.weight"
    eh_proj: str = "nextn.eh_proj.weight"
    enorm: str = "nextn.enorm.weight"
    hnorm: str = "nextn.hnorm.weight"
    head_norm: str = "nextn.shared_head_norm.weight"
    concat_embd_first: bool = True
    gated_q: bool = True           # attn_q emits [q | gate] per head; output * sigmoid(gate)
    qk_norm: bool = True
    rope_neox: bool = True
    ffn: str = "swiglu"
    # trainable tensors (relative to the block prefix): the MTP block itself; embedding and LM head are the trunk's
    trainable: tuple = ("nextn.eh_proj.weight", "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                        "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight", "post_attention_norm.weight",
                        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "nextn.enorm.weight",
                        "nextn.hnorm.weight", "nextn.shared_head_norm.weight")
    kv_rot_k: int = 256            # Hadamard size for quantized K (largest power of 2 dividing head_dim, >= 64)
    kv_rot_v: int = 64


QWEN35_MTP = HeadSpec(name="qwen3.8-27b-mtp", arch="qwen35", block="blk.{L}.")

SPECS = {"qwen35": QWEN35_MTP}


@dataclass
class HParams:
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_rot: int
    rope_base: float
    eps: float
    n_ff: int
    n_vocab: int
    L: int                          # trunk layers; the MTP block is blk.L


def hparams_from_gguf(reader, spec):
    f = lambda k: reader.fields[f"{spec.arch}.{k}"].contents()
    n_nextn = f("nextn_predict_layers")
    assert n_nextn == 1, "one MTP block supported"
    L = f("block_count") - n_nextn
    vocab = len(reader.fields["tokenizer.ggml.tokens"].contents())
    return HParams(n_embd=f("embedding_length"), n_head=f("attention.head_count"), n_head_kv=f("attention.head_count_kv"),
                   head_dim=f("attention.key_length"), n_rot=f("rope.dimension_count"), rope_base=float(f("rope.freq_base")),
                   eps=float(f("attention.layer_norm_rms_epsilon")), n_ff=f("feed_forward_length"), n_vocab=vocab, L=L)
