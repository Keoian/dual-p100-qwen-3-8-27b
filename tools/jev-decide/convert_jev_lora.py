#!/usr/bin/env python3
"""JEV-27B PEFT LoRA (adapter/) -> llama.cpp GGUF LoRA for qwen35 (Qwen3.8-27B).
Explicit mapping + the same V-head reorder that conversion/qwen.py applies to the base weights
(_LinearAttentionVReorderBase): rows of in_proj_qkv's V part and in_proj_z (-> lora_b), columns of out_proj (-> lora_a)."""
import sys, json, re, argparse
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf
from safetensors.numpy import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True, help="JEV-27B adapter/ dir (adapter_config.json, adapter_model.safetensors)")
ap.add_argument("--config", required=True, help="JEV-27B config.json (the Qwen3.8 text config)")
ap.add_argument("--outtype", choices=["f32", "f16"], default="f32")
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg = json.load(open(a.config)); acfg = json.load(open(a.adapter + "/adapter_config.json"))
assert acfg["peft_type"] == "LORA" and not acfg["use_dora"] and not acfg["use_rslora"] and acfg["bias"] == "none"
nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
hk, hv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
vpk = nv // nk

def reorder_v(t, dim, head_dim):  # identical to _LinearAttentionVReorderBase._reorder_v_heads
    shape = list(t.shape); dim %= len(shape)
    new = shape[:dim] + [nk, vpk, head_dim] + shape[dim + 1:]
    perm = list(range(len(new))); perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.ascontiguousarray(t.reshape(new).transpose(perm)).reshape(shape)

MAP = {"linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate", "linear_attn.out_proj": "ssm_out",
       "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
       "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
sd = load_file(a.adapter + "/adapter_model.safetensors")
pat = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.(\w+\.\w+)\.lora_([AB])\.weight$")
pairs = {}
for k, v in sd.items():
    m = pat.match(k); assert m, k
    pairs.setdefault((int(m[1]), m[2]), {})[m[3]] = v
dt = np.float32 if a.outtype == "f32" else np.float16
w = gguf.GGUFWriter(a.out, "qwen35")
w.add_string("general.type", "adapter"); w.add_string("adapter.type", "lora")
w.add_float32("adapter.lora.alpha", float(acfg["lora_alpha"]))
w.add_string("general.name", "JEV-27B System 1 LoRA (autotrust/JEV-27B@51740a88 adapter/)")
n = 0
for (il, mod), ab in sorted(pairs.items()):
    A, B = ab["A"], ab["B"]
    assert A.shape[0] == B.shape[1] == acfg["r"]
    if mod == "linear_attn.in_proj_qkv":
        qk = 2 * nk * hk
        B = np.concatenate([B[:qk], reorder_v(B[qk:], 0, hv)], 0)
    elif mod == "linear_attn.in_proj_z":
        B = reorder_v(B, 0, hv)
    elif mod == "linear_attn.out_proj":
        A = reorder_v(A, 1, hv)
    name = f"blk.{il}.{MAP[mod]}.weight"
    w.add_tensor(name + ".lora_a", np.ascontiguousarray(A.astype(dt)))
    w.add_tensor(name + ".lora_b", np.ascontiguousarray(B.astype(dt)))
    n += 1
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {n} LoRA pairs to {a.out}")
