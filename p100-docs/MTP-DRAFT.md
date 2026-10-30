# The MTP draft under exact acceptance: what was measured, what could still pay, and the training it would take

State 2026-10-30. Production runs the **exact** accept rule again since 10-30 (CHANGES §25): every token keeps the
27B's own odds. This document collects what limits MTP speed under that rule, the speed-ups still on the table, and
what retraining the MTP head would take. Numbers are MEASURED on the reference box (two P100s, 150 W, production build
and settings) unless a line says estimate, computed or simulated; the proposal's expected gains, odds, data sizes
and schedule are estimates. Per-attempt logs and raw results stay on the box: `/work/bench/LOG.md` (10-24 to 10-30),
`/work/bench/mtp5/` (dumps, `out-k1.md` with per-category tables, parity and overlap results), `/work/bench/riddle/`
(the riddle runs), and the full reports `/work/MTP_ACCURACY.md`, `/work/MTP_STEPFATHER.md`, `/work/MTP_RATIO_R1.md`,
`/work/MTP_RATIO_SWITCH.md`. The tooling is in `p100-handoff/tools/mtp-distill/`.

## Speeds at exact

| workload | exact | `ratio:1.5` (default 10-29 to 10-30) | no MTP |
|---|---|---|---|
| chat, 72-prompt suite (thinking off, 300-token cap) | **51.6 t/s** (10-30) | 53.8 (10-29: +4.1% vs exact's 51.7 that night) | ~30 (10-24, 48-prompt chat suite: 30.3) |
| thinking, 72-prompt suite (medium effort) | **51.4 t/s** (10-30) | 55.6 (10-29: +7.6% vs exact's 51.7 that night) | not measured |
| 32k / 64k / 128k / 200k context, chat (8 questions per depth, 10-28) | 53.1 / 48.7 / 41.3 / 37.8 | not measured | — |

- Plain `tg256` (no MTP) is 31.1 t/s.
- The suites and the long-context questions are different prompt sets: compare within a row.
- `esc+0.2` ran 9-10% faster than exact at every depth from 32k to 200k (10-28); `ratio:1.5` was not measured at depth.
- All of these run the bf16 exchange wire (`AR_BF16=1`, CHANGES §21: MTP replies +6.8%, KLD +1.1%). It rounds each
  GPU's partial sums to bf16, so it is the one remaining lossy setting beyond the model's own quantization and the
  q8_0/q4_0 cache. `AR_BF16=0` restores the f32 wire; exact plus the f32 wire is about 48 t/s in chat (estimate: the
  two were measured separately, never together).
- Exact means the same distribution, not the same text for a given seed: the random draws are consumed differently,
  and the model's own logits depend slightly on how the verify batches positions (see "Greedy identity" below).

## Why exact: the stepfather case

On 10-28 a thinking reply (effort xhigh, card sampling: temperature 1.0, top-p 0.95, top-k 20) answered a family
riddle wrong: "Riley's father is your stepfather", repeated about 20 times in the thinking and copied into the
answer. The run was recovered from the RAM prompt cache (a slot save holds the token ids) and probed on the
production model:

| next token after... | the 27B's own probabilities, teacher-forced |
|---|---|
| "Riley's father is" | " a" 29%, " Riley" 21%, **" your" 17%**, "..." 10%, " some" 7% |
| "Riley's father is your" | **" step" 45%**, " father" 24%, "..." 20% |
| "Riley's father is your step" | **"father" 61%**, "-" 12% |

The slip is the model's own: a 1-in-6 pick at one word, which it then reasoned from. It was not confident at the
fork. Resampled from "Riley's father", same seeds, alternating rules:

| rule | "is your stepfather" | runs |
|---|---|---|
| `exact` | 8.0% (10-28), 6.0% and 2.7% (10-29, two sessions); pooled 5.6% | 500 |
| `esc+0.2` (production 10-26 to 10-29) | 22.0%, 11.0%, 10.0%; pooled 14.0% (Fisher p = 1e-5 vs exact) | 500 |
| `ratio:1.5` (production 10-29 to 10-30) | 8.5% (vs 6.0% exact the same night: not distinguishable, 200 runs each) | 200 |

- **Why the veto rule amplified it:** `esc+0.2` keeps the head's token wherever the model allows it. The head's stock
  phrase " step" + "father" is made of tokens the model rates as plausible, so the model's usual way out ("your
  father's new wife's husband...") got squeezed. Rare-word and off-top counts can't see an error built from top
  tokens (FINDINGS "Approximate accept rules and the model's own slips").
- **`ratio:1.5` bounds it:** no token more than 1.5x as likely as under the model's own sampling. Its extra error on
  the riddle is small and unproven: confirming a 6.0% vs 8.5% gap takes about 1,700 runs per rule (80% power,
  p < 0.05), and rates move between server sessions (2.7-8.0% under exact), so the runs must pair within a session.
- **Exact removes the question:** the riddle rate is the 27B's own, 3-8% from that point.

**What MTP work can and can't do here.** Under exact, the draft head cannot change what the model writes: a perfect
head and a useless one give the same riddle rate. Everything below is speed. A better head would shrink what
`ratio:1.5` adds (its boost comes from rows where the head is surer than the model, which training corrects), but
showing that on the riddle (say 8.5% -> 7.25%) takes about 7,300 runs per arm, on one riddle. The levers that act on
the model's own slip are different ones: a lower thinking temperature (shrinks a 17% branch; Qwen's card warns that
low temperatures loop in thinking), or two or three answers and a check or vote (a slip made 6% of the time would
fall to about 1% with a three-way vote if the answers were independent; they aren't fully, so less). Neither is built.

## What limits the draft (MEASURED 10-29/30)

### Where the time goes (per-cycle draft log, chat)

- A verify cycle costs 39.8 ms plus 0.25 ms per 1k tokens of context, and +3.9 ms per drafted row (the padded
  verify skips inactive columns). Each draft step costs 1.28 ms.
- A committed token is worth about 19.4 ms, so a drafted row pays only if its prefix survives with probability
  above about 0.27. That is what the stop tables (CHANGES §22) decide.
- Under exact the chance a drafted token is kept is sum(min(p, q)) = 1 - TV(p, q): the overlap between the model's
  served distribution p and the draft's q. Per drafted row it is **0.78 in chat and 0.74 in thinking** (19k rows
  logged on a server in the production configuration). That gap, 22-26%, is the main limit on exact-mode speed.

### The draft's own LM head

At load, llama.cpp copies output rows 0..81,919 and re-quantizes them to Q4_0 for the draft (`LLAMA_MTP_DRAFT_VOCAB`,
`LLAMA_MTP_DRAFT_VOCAB_TYPE`, OPTLOG 226). Every id at or above 81,920 gets -inf, so **the draft can never propose
`<|im_end|>` or `</think>`.** The model's served mass above row 81,920 is 1.44% per row in chat and 1.08% in
thinking; above 98,304 it is 0.48% / 0.32%, half of it those two tokens. Coverage matters, precision doesn't:

| draft head variant (offline, paired on the same rows) | overlap gain, chat / thinking |
|---|---|
| Q4_0 over 98,304 rows | +0.0041 / +0.0039 |
| full vocabulary, Q8_0 (`LLAMA_MTP_DRAFT_VOCAB=0`) | +0.0102 / +0.0086 |
| Q8_0 over the same 81,920 rows | +0.0008 / +0.0011 |

### Cheap knobs, all under exact (72 prompts per mode; paired bootstrap over prompts)

| change | offline overlap gain, chat / thinking | chat t/s (95% CI) | thinking t/s | VRAM, GPU1 peak at depth | verdict |
|---|---|---|---|---|---|
| baseline (production's exact path) | — | 51.57 | 51.37 | 16,099 MiB (285 free) | — |
| draft sampled with the request's own settings (`LLAMA_SPEC_DRAFT_REQ=1`) | +0.0138 / +0.0035 | +0.7% (-1.3..+2.6) | -0.2% (-2.6..+2.3) | same | not as is: needs a stop-table refit |
| draft KV q8_0 instead of q4_0 | +0.0002 / +0.0009 | -0.6% (-3.0..+1.4) | not run | +128 MiB | reject |
| draft KV f16 | same as q8_0 | — | — | does not fit (computed: ~+390 MiB per GPU) | skip |
| MTP block Q8_0 from the BF16 source (new GGUF) | +0.0002 / +0.0002 | -2.1% (-4.3..-0.1) | -1.0% | +54 MiB | reject |
| draft LM head copy Q8_0 | +0.0008 / +0.0011 | -1.8% (-3.8..+0.1) | -1.7% | +100 MiB | reject |
| draft LM head copy over 98,304 rows | +0.0041 / +0.0039 | +0.9% (-0.5..+2.4) | +0.3% | +22 MiB | below the +2% bar; best small candidate |

- The VRAM column is GPU1's peak with the 250k chat restored (without its draft cache), then a 1440p image and a
  follow-up decode; the full-fill check was skipped, which matters for any change that adds draft-side memory (A
  below has 263 MiB of headroom on this measure).
- The precision knobs move overlap by at most 0.1%, and their extra cost per draft step (+0.1-0.2 ms) makes them
  net losses.
- Sampling the draft the way the request samples raised accepted tokens per cycle by 6.7% in chat (1.871 -> 1.997),
  but the stop tables, fit on temperature-1.0 drafts, then drafted 15% more rows (2.75 -> 3.17 per cycle), and the
  extra rows ate the gain. It stays exact (the draft's q is the list it samples from).
- The Q8_0 MTP-block GGUF has the same lineage as production (identical norms, production's Q8_0 `attn_k`/`attn_v`
  byte-identical to a Q8_0 of the BF16; 860 of 866 tensors hash-identical). It was rejected and stays on the box only
  until the operator deletes it.

### Greedy identity

At temperature 0, a build or setting that keeps the drafting schedule gives byte-identical replies (24/24 test
prompts, with the random tool-call ids ignored; 21/24 on raw hashes). Changing only the stock head's draft lengths (`LLAMA_SPEC_P_CUM=0.6`) changed 11 of 24
greedy replies at near-ties ("they're wrong" vs "they've made a mistake"). So the model's own logits at a position
depend on which verify batch and row computed it (likely the recurrent state's re-evaluation after rejections and
the verify kernels). That predates this work; making the target batch-invariant would be its own project. Any change
that alters acceptance therefore changes some greedy replies (the knobs above changed 4-6 of 24: 18-20 identical;
the request-sampled draft, inactive at T 0, 24/24) without changing the model's math, and under sampling exact acceptance still writes the model's distribution for whichever schedule ran.

## Proposal: speed-ups that keep exact output

Ranked by cost. None is built beyond what is listed; each needs the operator's OK and a GPU window.

| | change | work | expected | odds of clearing +2% (judgment) |
|---|---|---|---|---|
| A | draft LM head over 98,304 rows plus appended rows for `<\|im_end\|>` and `</think>` | small code change in the draft-head copy; env already exists for the row count | +1-1.5% (98,304 alone measured +0.9% chat, n.s.; the two tokens are half the remaining mass) | ~25% |
| B | draft sampled with the request's settings, plus stop tables refit on those drafts | refit with the existing CF tooling (branch `mtp-adaptive-research`), one window | +1-3% chat, ~0 thinking (estimate: committed tokens per cycle measured +4.4%, less whatever extra drafting the refit keeps) | ~35% |
| C | retrain (distill) the MTP head on the 27B's own outputs | six night windows, see below | median +3-5% | ~55%; +5% or more ~30%; +10% or more ~15% |

A and B are independent of C and of each other; C's trained head would also need B's refit. A better head also helps
any approximate rule (and shrinks what `ratio` adds), but the point here is to make exact fast enough that an
approximate rule isn't needed: `ratio:1.5` measured 4.1% faster than exact in chat and 7.6% in thinking (10-29).

Why C's odds are modest: the head is one transformer layer guessing two tokens ahead from the model's last hidden
state, with far less computation than the 27B, so much of its 22-26% mismatch can't be trained away; the trained
weights are re-quantized to 4-6 bits for serving; and B showed that gains get eaten unless the stop tables are refit.
A +10% result is the optimistic end, not the median.

## Training it would take (C)

### The head, as llama.cpp computes it

`src/models/qwen35.cpp` graph_mtp, written down in `mtp-distill/mtp/spec.py` (the only model-specific file):

- **Inputs at position t:** token x_t and h_in. h_in is the 27B's final hidden state at t-1 after `output_norm`; at
  draft depth > 1 it is the head's own previous output after `shared_head_norm`.
- **Projection:** z = eh_proj(concat(rms(token_embd[x_t]) * enorm, rms(h_in) * hnorm)), 10240 -> 5120, embedding half first.
- **One full-attention block (blk.64):** attn_q gives [q | gate] per head (24 x 512), attn_k / attn_v 4 KV heads x 256,
  per-head RMS q/k norms, NeoX RoPE on the first 64 of 256 dims (base 1e7), GQA 24/4, causal over the head's own KV
  cache, output gated by sigmoid(gate), then a SwiGLU FFN (17408), and h_out = rms(y) * shared_head_norm.
- **Logits:** the draft LM head copy (above) times h_out.
- **KV cache:** catch-up writes the head's K/V for every verified position; depth d adds rows at pos0+1..pos0+d-1.
  The cache is q4_0, stored Hadamard-rotated (K and Q by a 256x256 Walsh-Hadamard per head, V by 64x64 blocks).
- The head shares only the embedding table and the output matrix with the trunk.

### Reimplementation and parity (done)

A PyTorch copy of the head (`mtp-distill/mtp/head.py`) matches llama.cpp's own draft logits, logged with
`LLAMA_MTP_DUMP`:

| test | draft steps | top-1 agree | mean \|dlogit\| | TV (top-20) |
|---|---|---|---|---|
| unquantized (F32 block and head, f32 draft KV, llama.cpp CPU) | 1,006 | 100% | 0.0017 | 0.0002 |
| production precision on CPU, quantization emulated | 127 | 100% | 0.036 | 0.0055 |
| production GPU server | 2,505 | 98.7% | 0.080 | 0.012 |

The mapping is exact; at production precision the remainder is llama.cpp's activation quantization and the Q4_0 head,
small next to the 22-26% gap training targets. (One of the 15 CPU-only test requests gave the *target* NaN on every row: a test model with an F32 MTP block and output
matrix, CPU only, 2 threads, q8_0/q4_0 cache, flash attention on. Not checked on the production model or upstream;
production runs CUDA.)

### Data

- **Prompts:** about 4,000, written from per-category templates: reasoning 30%, code 20%, casual 10%, creative 10%,
  json/extraction 8%, multi-part 8%, tools 7% (other tool schemas than the eval suite's), long documents 7% (8k-30k
  tokens). 70% thinking (medium effort, some high), 30% chat. No production chat content.
- **Disjoint from the 72-prompt eval suite:** exact-match check, plus rejection of any prompt with word-3-gram Jaccard
  >= 0.3 against a suite prompt or sharing its topic. 10% of prompts held out for validation.
- **Long documents:** public-domain texts (Project Gutenberg) if the operator OKs the download, else written ones.
- **Collection:** the production config under exact with `LLAMA_MTP_DUMP` on. Each position keeps the head's input h
  (fp16), the 27B's raw top-64 logits and log-sum-exp, the presence window, and per draft cycle the drafted tokens with
  the 27B's verify rows as teachers for depths 2-4. 10.8 KiB per position compacted (measured; the raw dump is about 32 KB: 2.3 GB for 73.5k positions).
- **Before collection:** the dump costs roughly 19% of decode in chat and 25% in thinking (not a paired measurement:
  the dump run used the 24 test prompts with the prompt cache off and a 1,200-token thinking cap, against the 72-prompt
  baseline). Not profiled; the likely costs are the partial sort over all 248,320 ids plus a full-vocabulary exp sum
  per record, and the fp32 h of the catch-up records (77% of the bytes; draft records 21%). nth_element, fp16 h and no
  h in draft records should bring it under 10% (estimate). The reasoning budget's forced
  tokens can't be rebuilt from raw logits, so the server should also record the sampler's actual candidate list per
  verify row.
- **Size:** a pilot of about 300k positions plus 50k held out (about 4 GB), then 2-3M (about 30 GB).

### Training

- **Trainable:** the MTP block, eh_proj and its two input norms (425M parameters). Frozen: the embedding, the trunk
  and the draft LM head it will be served with. It starts from Qwen's weights: adaptation, not training from scratch.
- **Loss:** TV(p_served, q) + 0.1 KL(p_served || q). Under exact acceptance expected acceptance is exactly 1 - TV, so
  TV is the objective itself; the small KL term gives gradient to tokens the draft ranks low, where TV's is zero.
  q = softmax(z / T_draft), with p rebuilt from the raw logits and the request's sampling settings.
- **Depths:** depth 1 teacher-forced on every reply position with the head's own KV context built causally, as at
  inference; depths 2-4 on the logged rollouts, chaining the head's own differentiable hidden state, taught by the
  verify's distribution on the drafted prefix. The second collection round uses the pilot head, so rollouts stay
  close to on-policy.
- **Schedule:** AdamW lr 1e-5 (5e-6 and 2e-5 also tried in the pilot), warmup, cosine, gradient clip 1.0; early stop
  on held-out overlap per depth. Success for the pilot: held-out overlap up at every depth beyond bootstrap noise.
- **Compute (estimate):** about 9 GFLOP per reply position; about 0.5 h per 1M positions per epoch per P100 in fp32,
  about half that with fp16 autocast (no bf16 on Pascal). Memory about 12 GB per card (weights, gradients and AdamW
  state 6.8 GB, the frozen draft head 0.8-1.7 GB, activations up to about 3 GB at 4k tokens). Pilot about 1 h; full run
  (3M positions x 2 epochs, both cards) about 2-3 h.
- **Status:** `train.py` is written and passed a one-step smoke test (424.7M trainable parameters, gradients through
  the chained depths). No training has run.

### Export, evaluation, keep bar

- The trained tensors go into a new copy of the GGUF (never the production file), at Q6_K through ggml's own
  quantizer or Q8_0 (gguf-py's Q8_0 is byte-identical to ggml's). Every non-MTP tensor's SHA-256 is checked against
  production and the new file's hash recorded. `parity.py` runs on the export.
- A/B under exact, stock vs trained head, both modes, 72 prompts, with greedy identity and VRAM at production depth,
  plus a stop-table refit check.
- Keep bar: +2% or more with the CI excluding 0 in at least one mode, no significant loss in the other.

### GPU windows

Training and serving can't share the cards (production fills both), so every step is downtime with the chat saved
and restored:

| night | work | downtime |
|---|---|---|
| 1 | pilot collection (about 350k positions, 2.5 h), pilot training (1 h), export and a quick A/B | ~6 h |
| 2-4 | full collection, about 1M positions per night (round 2 uses the pilot head) | ~6 h each |
| 5 | full training, export, parity on the export | ~4 h |
| 6 | full A/B: stock vs trained, both modes, greedy identity | ~5 h |

About 33 h of downtime over six nights.

### Portable to the next Qwen

`mtp/spec.py` holds every model-specific part (tensor names, hparam keys, structure flags, trainable set, Hadamard
sizes); a new MTP head needs a new HeadSpec. The dump hooks sit in the generic MTP draft code
(`common/speculative.cpp`), not in the model graph, and `parity.py`, `ovl_eval.py`, `train.py` and `data.py` read
only the spec and the dump.

## Hooks and tools

Server env switches in `common/speculative.cpp` and `tools/server/server-context.cpp`, all off by default (CHANGES §25):

| switch | what it does | cost when on |
|---|---|---|
| `LLAMA_MTP_DUMP=<path>` | binary log of the head's inputs (h, token, position) and top-64 draft logits per draft step, and the 27B's top-128 logits for every verified row; format in the code comment | roughly -19% decode in chat, -25% in thinking; ~32 KB per position; appends, so use a new path per run; draft records valid only with the CPU draft sampler (`-sm tensor` forces it, else `--no-spec-draft-backend-sampling`) |
| `LLAMA_SPEC_DRAFT_REQ=1` | under exact too, sample each draft token with the request's own chain (presence, top-k, top-p, min-p, temperature) | its speed is in the knob table above; the stop tables stay on (and over-draft) |
| `LLAMA_SPEC_LOG=<path>` | now also writes the drafted count per cycle (column 5, before the top-1 probabilities) | as before |

`p100-handoff/tools/mtp-distill/` (README there): the PyTorch head and dump reader (`mtp/`), parity and paired overlap
checks (`parity.py`, `ovl_eval.py`), the trainer (`train.py`), the Q8_0 GGUF writer with lineage and hash checks
(`make_mtp_q8.py`, `gguf_remote.py`), the offline simulations, and the window scripts that measured the table above.
Written for the reference box's layout (`/work/...`); torch 2.6 with CUDA 12.4 supports sm_60.

## Decisions pending (the operator's)

1. A and B above: go or no-go.
2. C: go or no-go on the pilot (night 1), with the prompts, schedule and loss above.
3. The Project Gutenberg download for long-document prompts.
4. Deleting the rejected Q8_0 MTP-block GGUF (22 GB, on the box only:
   `/work/bench/mtp5/models/Qwen3.8-27B-UD-Q6_K-mtpq8.gguf`, SHA-256
   `a84a3bd2c6dea955efcb5a9edc6efb603d32546005dbb4d5f410cf3680771880`).
