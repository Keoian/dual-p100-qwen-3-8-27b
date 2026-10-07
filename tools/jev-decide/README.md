# llama-jev-decide: JEV-27B System 1 decisions

[autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) turns Qwen3.8-27B into a calibrated decision engine with
a rank-16 backbone LoRA and a linear 24-slot head (`noul` yes/no, `score` 0-5, `choice` A-P) over the last token's
final-norm hidden state. This directory runs it on the unmodified Qwen3.8-27B GGUF.

## 1. Convert the LoRA (once)

Download only the System 1 files (`adapter/`, `head.safetensors`, `calibration.json`, `config.json`; pinned revision
`51740a8891c2a8baefd969237fd44187b3e3a115`), not the bf16 backbone shards. The converter needs
`pip install numpy safetensors` (no torch) and imports the repo's `gguf-py`, so run it from this checkout:

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
    -ctk q8_0 -ctv q8_0 \
    --jev-head JEV-27B/head.safetensors --jev-calib JEV-27B/calibration.json --jev-in rows.jsonl --jev-out out.jsonl
```

Input rows: `{"id", "kind", "state", "question", "options"}` (the `SargeDev/jev-distill-corpus-v3` schema). Output rows:
`{"id", "n_tokens", "logits", "probs", "ms"}`, `logits` being the raw head slots of the kind, `probs` after the per-kind
temperature. Prompts use the `bare-v1` template, tokenized as one string without BOS; each row starts from an empty
context. The hidden state is read with `llama_set_embeddings_nextn` (masked), not embeddings mode, which would make
every prompt token an output.

The server exposes the same thing as `POST /v1/decide` (alias `/decide`): `--jev-lora`, `--jev-head` (required with
`--jev-lora`), `--jev-calib`, and for the decision context `--jev-ctx` (8192), `--jev-batch` (512), `--jev-ctk/--jev-ctv`
(q4_0 built-in default; the tyler-port production setup uses q8_0). Request and response: QUICKSTART "JEV System 1".

## Accuracy on 2x P100

**Full `test_set_30k`, all 29,955 rows** (2026-10-07), through the production server's `/v1/decide`: UD-Q6_K backbone,
f16 LoRA, q8_0 decision cache, 262k System 2 and vision loaded; published = bf16 (autotrust `reports/eval_27b_bundle.md`):

| metric | ours | published |
|---|---:|---:|
| KL(target ‖ model), all rows | 0.0186 | 0.0185 |
| KL excluding yuri_v1 placeholders (27,695) | 0.0201 | 0.0201 |
| KL on Jev-labelled rows (yuri_v3, 25,376) | 0.0167 | ≈0.017 |
| JS | 0.0049 | 0.0048 |
| ECE (15 bins) | 0.0012 | 0.0011 |
| noul AUROC / Brier (soft) | 0.9961 / 0.0013 | 0.9961 / 0.0013 |
| score MAE (expected) / RPS | 0.0979 / 0.0077 | 0.0976 / 0.0077 |
| choice KL / top-1 (all rows) | 0.0366 / 0.900 | 0.0364 / 0.904 |
| by source/kind KL: yuri_v3 choice / noul / score | 0.0249 / 0.0043 / 0.0210 | 0.025 / 0.004 / 0.021 |
| by source/kind KL: openjev_v2 choice / noul, yuri_v1 noul | 0.1467 / 0.0030, 0.0000 | 0.146 / 0.003, 0.000 |

0 NaN; 627 ms median, 697 mean, 1.0 s p95 per decision (106 tokens mean). RPS is the unnormalized sum over the
cumulative distribution. Top-1 counts tied targets toward the first option; over unique-argmax rows only it is 0.912.

Earlier (10-06), stratified 3,000-row subset of `test_set_30k` (1,000 per kind), against the published bf16 numbers:

| source / kind | KL ours | KL published | top-1 ours | top-1 published |
|---|---|---|---|---|
| yuri_v3 / choice | 0.0256 | 0.025 | 0.910 | 0.906 |
| yuri_v3 / noul | 0.0043 | 0.004 | 0.969 (AUROC 0.996) | 0.960 (AUROC 0.995) |
| yuri_v3 / score | 0.0201 | 0.021 | 0.892 (MAE 0.097) | 0.891 (MAE 0.098) |
| yuri_v1 / noul | 0.0000 | 0.000 | - | - |
| openjev_v2 / noul (n=121) | 0.0065 | 0.003 | 1.000 | 0.999 |
| openjev_v2 / choice (n=111) | 0.173 | 0.146 | 0.928 | 0.885 |

Reweighted to the full set's composition: KL ~0.0195 vs 0.0185. ECE 0.0031 on 3,000 rows: ECE shrinks ~1/sqrt(n),
and the full set indeed gives 0.0012. The 3k numbers are with an f16 decision cache. q8_0 (served in production) is KL 2.9e-6 vs f16 with no flips on 300
rows; q4_0 (the server's built-in default) is KL 3.5e-4, 6 argmax flips, all near-ties (top-2 gap <= ~0.03). Pass the
served `-ctk/-ctv` to this tool to evaluate what the server runs. f16/q8_0 need `03da0202b`: before it, prompts above ~5.9k tokens could give NaN under
`-sm tensor` (a race in the GEMM-attention softmax).
