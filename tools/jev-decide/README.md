# llama-jev-decide: JEV-27B System 1 decisions

[autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) turns Qwen3.8-27B into a calibrated decision engine with
a rank-16 backbone LoRA and a linear 24-slot head (`noul` yes/no, `score` 0-5, `choice` A-P) over the last token's
final-norm hidden state. This directory runs it on the unmodified Qwen3.8-27B GGUF.

## 1. Convert the LoRA (once)

Download only the System 1 files (`adapter/`, `head.safetensors`, `calibration.json`, `config.json`), not the bf16
backbone shards, then:

```bash
python3 tools/jev-decide/convert_jev_lora.py --adapter JEV-27B/adapter --config JEV-27B/config.json \
    --outtype f16 --out jev-27b-lora-f16.gguf
```

The converter maps the PEFT modules to qwen35 tensors (`in_proj_qkv -> attn_qkv`, `in_proj_z -> attn_gate`,
`out_proj -> ssm_out`, `q/k/v/o_proj -> attn_q/k/v/output`, MLP 1:1) and applies the same V-head reorder the base
converter applies to the base weights. f16 vs f32 LoRA: KL 6.6e-7 on 300 decisions.

## 2. Run decisions over a JSONL file

```bash
llama-jev-decide -m Qwen3.8-27B-UD-Q6_K.gguf --lora jev-27b-lora-f16.gguf -ngl 99 -sm tensor -fa 1 -c 8192 -b 1024 -ub 1024 \
    -ctk q4_0 -ctv q4_0 \
    --jev-head JEV-27B/head.safetensors --jev-calib JEV-27B/calibration.json --jev-in rows.jsonl --jev-out out.jsonl
```

Input rows: `{"id", "kind", "state", "question", "options"}` (the `SargeDev/jev-distill-corpus-v3` schema). Output rows:
`{"id", "n_tokens", "logits", "probs", "ms"}`, `logits` being the raw head slots of the kind, `probs` after the per-kind
temperature. Prompts use the `bare-v1` template, tokenized as one string without BOS; each row starts from an empty
context. The hidden state is read with `llama_set_embeddings_nextn` (masked), not embeddings mode, which would make
every prompt token an output.

The server exposes the same thing as `POST /v1/decide` (`--jev-lora`, `--jev-head`, `--jev-calib`).

## Accuracy on 2x P100 (UD-Q6_K backbone, f16 LoRA, f16 KV)

Stratified 3,000-row subset of `test_set_30k` (1,000 per kind), against the published bf16 numbers:

| source / kind | KL ours | KL published | top-1 ours | top-1 published |
|---|---|---|---|---|
| yuri_v3 / choice | 0.0256 | 0.025 | 0.910 | 0.906 |
| yuri_v3 / noul | 0.0043 | 0.004 | 0.969 (AUROC 0.996) | 0.960 (AUROC 0.995) |
| yuri_v3 / score | 0.0201 | 0.021 | 0.892 (MAE 0.097) | 0.891 (MAE 0.098) |
| yuri_v1 / noul | 0.0000 | 0.000 | - | - |
| openjev_v2 / noul (n=121) | 0.0065 | 0.003 | 1.000 | 0.999 |
| openjev_v2 / choice (n=111) | 0.173 | 0.146 | 0.928 | 0.885 |

Reweighted to the full set's composition: KL ~0.0195 vs 0.0185. ECE (15 bins, soft) 0.0031 vs 0.0011 published:
calibration is ~3x off on this backbone, hence the planned per-kind temperature refit. ~690 ms per decision at ~107
tokens.

These numbers are with an f16 decision cache. The server's decision context defaults to q4_0 (KL 3.5e-4 vs f16 on 300
rows, 6 argmax flips, all near-ties, top-2 gap <= ~0.03); q8_0 is KL 2.9e-6 with no flips. Pass the same `-ctk/-ctv` to this tool to
evaluate what the server runs. f16/q8_0 need `03da0202b`: before it, prompts above ~5.9k tokens could give NaN under
`-sm tensor` (a race in the GEMM-attention softmax).
