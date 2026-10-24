# Phase 0 reconnaissance: native MTP in the dual-P100 fork

Read-only code and log audit, 2026-10-24. Branch `mtp-adaptive-research` at `9acfc08e2`. The code is the same as on
`p100-optimizations-keoian`; the last code commit is `c73976485`, and later commits are deploy/docs only. Nothing was
built or run on the GPUs. The only things computed were two small Python passes over `/work/bench/spec-log.txt`.

Legend: statements are **read** from the cited code or log unless tagged **[I]**, which means inferred, estimated or
derived. Paths are relative to `/work/src/llama.cpp` unless absolute. "Kmic board" means the P2P board behind
`OPTLOG.md`. "This board" means the H270, where GPU1 sits on a PCH Gen3 x4 link with no P2P, at 150 W.

---

## 0. The plan's question, answered

**Can native MTP produce successive tokens without new backbone hidden states?** Yes, mechanically, and it already
does inside every draft. Draft step 1 is anchored on the target: it takes the target's final-normed hidden row at
`pos0-1` and the token the target just sampled. Each later step `k >= 2` consumes three things the MTP made itself:
- its own previous output row: `h_nextn`, the MTP block's output after `shared_head_norm` (`qwen35.cpp:655-662`);
- the token it just drafted;
- its own KV entries, written by the earlier steps of the same draft, at position `pos0+k-1`.

The feedback code is `common/speculative.cpp:1815` and `:1910-1916`. No backbone computation happens between draft
steps. Depth is bounded in code only by `--spec-draft-n-max`. The multi-head cap at `:1536-1542` applies only to
chained-head models; Qwen3.5/3.8 has exactly one MTP block (`qwen35.cpp:490`).

**What it cannot do is commit tokens without the target.** Every committed token must still pass through the target,
because that pass is the only way to advance:
- the target's 16-layer attention KV;
- its 48-layer delta-net (GDN) recurrent state;
- the target hidden row that anchors the next draft. The MTP's KV for committed positions is also rewritten from target
  rows by the catch-up pass.

No code path advances target state any other way, and GDN state is a nonlinear function of every token. Making MTP
"autonomous" therefore can only change two things:
- **when** the target runs: defer it and process a longer burst in one wider pass;
- **whether its verdict is enforced**: lenient or approximate acceptance using the target probabilities that pass
  already computes.

It cannot change **whether** the target runs. A truly backbone-free continuation across anchors would need both of
these, and the repo has neither:
- (a) a learned state transition, for example predicting target-like `h` from MTP `h` (the plan's optional
  "hidden-state correction head");
- (b) some way to advance target KV/GDN state without running the target. This does not exist and is not a training
  problem.

The MTP head's training objective (one step from backbone `h`, or recursive) cannot be determined from the repo. The
measured self-fed accuracy is the evidence: conditional acceptance is 0.78 / 0.82 / 0.80 / 0.73 at depths 1-4 (§7).
There is no data for depth > 4 at this board's current settings.

**Decision-gate recommendation [I]:** option (b), verified adaptive MTP, plus lenient acceptance as the explicit
approximate mode. A "free-running burst" here is just a long draft committed by one target pass. Phase 2's
quality-vs-run-length curve can be measured with existing machinery: a draft of L <= n_max, plus the verify pass's
teacher logits on the identical drafted prefix (trace hook T2, §6). No new pathway is needed for that. Build an
autonomous-state pathway (a) only if Phase 2 shows high teacher agreement well past depth 8.

---

## 1. Data-flow map: one MTP cycle in llama-server

```
 prefill ubatches ─► ctx_tgt decode (all 64 trunk layers; h_nextn rows of EVERY token -> host)
                    └► common_speculative_process: ctx_dft "catch-up" decode, KV-only, pairs (x_p, h_tgt[p-1])
 first token sampled from target logits ─► slot.sampled = id_last, pos0 = pos_next
┌───────────────────────────── cycle ─────────────────────────────────────────────────────────────┐
│ DRAFT (ctx_dft, 1 token per step, CUDA-graph captured under GRAPHS_PRE_VOLTA=3)                   │
│  step1: (id_last, h_tgt[pos0-1]) @pos0 ─► MTP block ─► h1, logits1 ─► host top-K ─► d1 (stop?)  │
│  step k: (d_{k-1}, h_{k-1} = MTP's own output) @pos0+k-1 ─► ... ─► d_k     (own KV, no backbone) │
│ pad draft to n_max (repeat last token); erase ctx_dft KV >= pos0                                  │
│ VERIFY (ctx_tgt): batch [id_last, d1..d_n, pad..] width n_max+1, every row an output,              │
│        n_active = n_real+1 (matvec / GLU / q4p skip pad columns; W <= 8 only)                     │
│        GDN writes n_max+1 state snapshots; all h_nextn rows + all logits -> host                  │
│ CATCH-UP (ctx_dft, KV-only) over the verify batch, pairs (x_p, h_tgt[p-1]); stash verify rows     │
│ ACCEPT (host): greedy compare | per-token p/q | block verify ─► accepted[0..a], bonus/correction    │
│ ROLLBACK: seq_rm(pos_next) on both contexts: target KV cells + GDN rs_idx -> snapshot slot;       │
│           draft KV cells; MTP anchor row := verify row a                                           │
│ STREAM: process_token() for each accepted token (only verified tokens are ever sent)               │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**1.1 Prompt prefill.**
- The server decodes prompt batches into `ctx_tgt` (`server-context.cpp:4080`). The target runs with
  `embeddings_nextn` unmasked (`speculative.cpp:1528`), so every token's `h_nextn` row is copied to host
  asynchronously (`llama-context.cpp:1664-1670`, `2085-2093`). Each row is 5120 f32 (20 KiB). The host buffer is sized
  `n_embd*n_batch` (`:2203-2206`).
- On the target, `h_nextn = output_norm(final layer output)`: the same tensor that feeds the LM head
  (`src/models/qwen35.cpp:207-223`).
- After each target decode, `common_speculative_process` (`server-context.cpp:4146-4153` -> `speculative.cpp:1600-1726`)
  builds a draft batch of the same tokens and positions. Rows are shifted right by one, so token `x_p` pairs with
  `h_tgt[p-1]` (`:1648-1649`). The first row is looked up by position through `h_at()` (`:1381-1399`, `:1662`): it
  checks the pending row, then the previous one, then the last verify rows. An unknown row is zeros.
- It decodes `ctx_dft` with no outputs. The MTP graph then stores K/V and returns, with no attention, FFN or head
  (`qwen35.cpp:595-626`, `LLAMA_MTP_KV_ONLY`, OPTLOG 207). This is how the MTP KV cache holds teacher-forced entries
  for every committed position.
- Batches carrying embeddings (images) are skipped (`speculative.cpp:1606`). After an image the MTP cache has holes,
  and the next first row is zeros [I from `h_at`].

**1.2 Draft step 1** (`speculative.cpp:1728-1771`, server setup at `server-context.cpp:3394-3401`):
- One token: `id_last` (the target's last sampled token) at `pos0 = pos_next`, with `h = h_at(seq, pos0-1)`. That is
  the target row of the last committed token, kept on the host in `pending_h`.
- MTP graph (`qwen35.cpp:486-675`): `enorm(embed(tok))` and `hnorm(h)` are concatenated to 10240 and passed through
  `eh_proj` to 5120. Then comes one gated full-attention block over the draft's own KV (`ctx_dft`, `-ctkd/-ctvd`,
  M-RoPE at `pos`), then the FFN, then `shared_head_norm`, giving `h_nextn` (`:655-662`).
- The LM head is `output_draft`, a q4_0 copy of the first 81920 vocab rows (`:667-668`; `llama-model.cpp:1911-1971`).
  The other logits are filled with -inf (`llama-context.cpp:1997-2010`).
- Logits go to host. The CPU sampler (chain `top_k(K)` + `dist`, `speculative.cpp:1502-1510`) gives `cur_p`: the top-K
  candidates with p = softmax over those K at T=1.
- Backend sampling is refused under `-sm tensor` (`llama-context.cpp:1290-1293`), because the logits are
  vocab-sharded (OPTLOG 88).

**1.3 Draft steps 2..n** (`speculative.cpp:1773-1931`):
- After step i, `h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last)` (`:1815`) is the MTP's own `h_nextn`. The
  next batch is `(id, pos0+i+1, h_row)` (`:1910-1916`).
- The draft KV for `pos0..pos0+i` was written by the previous steps of this draft. Positions advance by 1.
- Stop conditions, evaluated after step i's decode and before its token is added (`:1834-1840`):
  - top-1 `p < p_min`;
  - `p_cum*p < p_cum_min(pos0)`;
  - `result.size() == n_max` (`:1891-1895`).
- Sampled mode (`:1845-1884`, `LLAMA_SPEC_SAMPLE_TEMP`): the token is drawn from the top-K renormalized at T, with
  optional top-p. Its distribution `q` is pushed to `dp.dists`. The stop test still uses the top-1 p.
- `n_min` discards drafts that are too short (`:1948-1950`).
- After drafting, the server erases the draft's KV at `>= pos0` (`server-context.cpp:3446`). The draft-made KV entries
  live only for the duration of one draft.

**1.4 Verify batch** (`server-context.cpp:3419-3478`, `539-578`, `4071-4089`):
- Fixed-width verify (`LLAMA_SPEC_PAD`, on when `-np 1`, `:1455-1459`): a short draft is padded to `n_max` by repeating
  its last token, or `id_last` if it is empty. `spec_n_real` is the real count (`:3428-3433`).
- The batch is `[id_last, d1..d_n_max]`, every row with output=true (`:574-576`). `batch.n_active = spec_n_real+1`
  (`:560`), passed as `llama_set_n_active_tokens` around `llama_decode(ctx_tgt)` (`:4071-4084`).
- The CUDA backend honours it only for 2..8-token ubatches (`llama-context.cpp:1485`), and only in the q6_K and other
  matvecs (`mmvq.cu:2047`), the fused fp16 GLU (`ggml-cuda.cu:4913`, `4980`) and q4p attention (`fattn-q4p.cuh:1274`).
  Exchanges, GDN, norms and other FA kernels compute the full width (`ggml-cuda/active-tokens.cuh`).
- Why the width is fixed: a change of width rebuilds the 64-layer tensor-parallel graph, ~26 ms, later 16-25 ms
  (OPTLOG 192, 198).
- The draft's top-1 p is calibrated, so a depth-scheduled `p_cum` cutoff trims drafts deep in the context
  (`speculative.cpp:1436-1456`). It is 0 below 16k, ramps to 0.22 at 48k+, and is active only with padding.

**1.5 Acceptance** (`server-context.cpp:4313-4358`, `common/sampling.cpp:810-1061`):
- **Greedy drafts** (`dists` empty): `common_sampler_sample_and_accept_n` (`sampling.cpp:810-838`). At each position
  the target samples its own token with the full user chain. The draft token is accepted iff equal; the first mismatch
  ends the cycle, and the target's sample is emitted. This is exact for any target sampler. With padding it uses only
  the real prefix (`server-context.cpp:4341-4346`).
- **Sampled drafts, per-token rule** (`sampling.cpp:989-1061`): `p_x` is the draft token's probability in the target's
  post-chain `cur_p`, and `q_x` comes from `dists`. Accept iff `u*q_x < p_x`. On rejection, draw from normalized
  `max(p-q,0)` over the target candidates. If everything is accepted, the bonus is sampled at position n. The RNG
  `slot.spec_dist_rng` is seeded per request from the sampler seed (`server-context.cpp:2010`).
- **Block verification** (Sun et al., `sampling.cpp:852-907`, used at `:930-985`):
  - It is on by default (`LLAMA_SPEC_BLOCK_VERIFY`), but only for a stateless target chain: no grammar, reasoning
    budget, repeat/freq/presence penalties, DRY or mirostat (`:937-938`).
  - Production's thinking-off sampling has presence 1.5, so production uses the per-token rule. Thinking requests carry
    a budget sampler and tool requests a lazy grammar (`sampling.cpp:397`), so they also fall back to per-token.
  - Kmic's sampling (presence 0) uses block verify.
- **Replay path** (`server-context.cpp:4326-4333`, `4366-4397`): used only when the target rollback type is FULL. For
  qwen35 `n_rollback <= n_max = n_rs_seq` always, so it is unreachable [I from `:4360-4364`].
- **Synthetic acceptance** (`--spec-synth-len/-rates`, `server-context.cpp:66-100`, `4347-4352`): benchmark only;
  accepts at random regardless of the target.

**1.6 Rollback** (`server-context.cpp:4360-4401`, `4430-4436`):
- `n_rollback = spec_draft.size()+1-accepted.size()`, which counts padding.
- `slot.mem.seq_rm(slot.id, pos_next, -1)` (`:4436`) goes to `common_memory::seq_rm` (`common/common.cpp:1652-1657`),
  which calls both contexts:
  - **Target attention KV:** cells at `>= pos_next` are removed (`llama_memory_hybrid::seq_rm`,
    `llama-memory-hybrid.cpp:143-150`).
  - **Target GDN state:** every target ubatch writes `K = n_rs_seq+1` per-token state snapshots: slot 0 is the final
    state, slot s is s tokens back (`ggml.h:2642-2659`, `src/models/delta-net-base.cpp:546-603`, conv states
    `:497-521`). `seq_rm` with `1 <= rollback <= n_rs_seq` sets `rs_idx` and moves the tail position. It is single use
    and needs no copy (`llama-memory-recurrent.cpp:193-201`). The next graph reads its initial state from snapshot
    slot `rs_idx` (`s_copy`, `:1306-1323`).
  - `n_rs_seq` is set from `draft.n_max` (`common/common.h:397-403` -> `common.cpp:1729`). The ubatch splitter keeps
    each sequence's last `n_rs_seq+1` tokens together (`llama-memory-recurrent.cpp:443-445`).
  - The rollback type is RS (`common.cpp:1595-1597`), so no per-cycle checkpoint is taken for the target
    (`server-context.cpp:3452-3477`).
  - **Draft context:** plain KV `seq_rm`. It has no recurrent state, because `n_rs_seq = 0` (`speculative.cpp:2809`).
    The catch-up wrote KV for rejected and padded positions too, and this removes them.
  - **MTP anchor row:** `common_speculative_accept` sets `pending_h = verify_h[min(a, rows-1)]`, the target row of the
    last accepted token, and `pending_pos` (`speculative.cpp:1986-1999`).
  - **Sampler:** the accept functions advance `slot.smpl`. `smpl_save` (`server-context.cpp:4317`) restores it only on
    the replay path.

**1.7 Streaming.** The accepted tokens go through `process_token` (`server-context.cpp:4438-4456` -> `:2040`, which
calls `send_partial_response` at `:2084`). Only verified tokens leave the server, once per cycle, and none is ever
retracted. In MTP mode the per-token `prob` is a placeholder (`result.prob = 1.0f; // set later`, `:4443`; no
`populate_token_probs`). So **the API returns no target probabilities for speculative tokens** [I: `n_probs` is ignored
on that path].

**1.8 State across requests.**
- Context checkpoints (`-ctxcp`, ~150 MiB each, which is one recurrent-state copy: `srv-v.log` "size = 149.646 MiB"),
  slot files (`<slot>.ckpt`, `<slot>.spec`) and RAM-cache entries all carry the MTP anchor row and its position
  (`get_state`/`set_state`, `speculative.cpp:1958-1984`; server `:348-360`, `2597`, `2833`, `2949`, `3754`).
- This is the `bd72ea071` / `73725dd56` fix for MTP nondeterminism.

---

## 2. Symbols and paths

| symbol | path:line | role |
|---|---|---|
| `common_speculative_impl_draft_mtp` | `common/speculative.cpp:1335-2013` | MTP driver: `process` 1600, `draft` 1728, `get/set_state` 1958, `accept` 1986 |
| `h_at`, `pending_h/prev_h/verify_h` | `speculative.cpp:1352-1399` | target row by position (anchor for step 1 and catch-up row 0) |
| `p_cum_min` | `speculative.cpp:1436-1456` | depth-scheduled cumulative-p cutoff (pad only) |
| `sample_temp`, `sample_top_p` | `speculative.cpp:1408-1424`, `1845-1884` | sampled drafts, `dp.dists` |
| `common_speculative_draft_params` | `common/speculative.h:52-76` | per-seq draft request/result: `n_max`, `pos0`, `id_last`, `result`, `dists` |
| `common_speculative_draft/accept` | `speculative.cpp:3080-3163`, `3165-3199` | wrappers (truncate to `dp.n_max`, per-pos stats) |
| `common_speculative_n_max` | `speculative.cpp:2581-2629` | startup n_max -> pad width, output limits |
| `need_n_rs_seq` | `common/common.h:397-403` | target `n_rs_seq = draft.n_max` |
| `llama_model_qwen35::graph` / `graph_mtp` | `src/models/qwen35.cpp:139-225` / `486-675` | target `h_nextn` / MTP block + KV-only catch-up |
| `output_draft` (`LLAMA_MTP_DRAFT_VOCAB`) | `src/llama-model.cpp:1911-1971`, `llama-context.cpp:1997-2010` | 81920-row q4_0 draft head |
| `build_recurrent_attn`, conv snapshots | `src/models/delta-net-base.cpp:470-603` | K = n_rs_seq+1 GDN/conv snapshots |
| `llama_memory_recurrent::seq_rm`, `s_copy` | `src/llama-memory-recurrent.cpp:193-201`, `1306-1323` | snapshot rollback |
| `common_context_can_seq_rm` | `common/common.cpp:1588-1625` | RS / PART / FULL rollback type |
| `server_slot::handle_last_sampled_token` | `tools/server/server-context.cpp:539-578` | builds the padded verify batch |
| draft scheduling | `server-context.cpp:3335-3478` | draft params, `common_speculative_draft`, padding, draft-KV erase |
| `decode()` | `server-context.cpp:4036-4158` | `llama_set_n_active_tokens`, target decode, MTP catch-up |
| verify / accept / rollback / stream | `server-context.cpp:4302-4459` | acceptance rules, `seq_rm`, `process_token` |
| `common_sampler_sample_and_accept_n(_dist)`, `common_spec_block_verify` | `common/sampling.cpp:810-1061` | greedy compare, p/q rule, block verify |
| `llama_set_n_active_tokens` -> CUDA | `llama-context.cpp:1480-1492`, `ggml-cuda/active-tokens.cuh`, `ggml-cuda.cu:6855-6866` | fixed-width verify |
| CUDA graph policy | `ggml/src/ggml-cuda/ggml-cuda.cu:3079-3110`, `5563-5570` | `GGML_CUDA_GRAPHS_PRE_VOLTA` |
| FA routing | `ggml-cuda/fattn.cu:696-717`, `818-857`; `fattn-gemm.cu:1730-1736`; `fattn-q4p.cuh:1198-1236` | q4p (1-5 q), vec, tile, GEMM |
| host-staged AllReduce | `ggml-cuda/allreduce.cu:415-460` | no-P2P path, `GGML_CUDA_AR_BF16_THRESHOLD` |

---

## 3. Depth: what bounds it, and n-max 8 / 16 / 32

**There is no code cap on n_max for qwen35** (`arg.cpp:4184-4192` checks only `>= 0`). The bounds are side effects:

| bound | where | at n_max = N |
|---|---|---|
| recurrent snapshots | `n_rs_seq = N` (`common.h:402`). RS buffer = (N+1) groups of 74.8 MiB per GPU; 374.06 MiB/GPU at N=4 (`/work/bench/srv-v.log`) | N=8: 673 MiB/GPU (+299); N=16: 1272 (+898); N=32: 2469 (+2095) [I: linear in N+1, as allocated at `llama-memory-recurrent.cpp:101`] |
| snapshot traffic | each target ubatch writes min(W, N+1) state copies (`delta-net-base.cpp:589-603`) | grows linearly with N+1 per verify; size it with nvprof in Phase 1 [I] |
| fixed verify width | pad to N (`server-context.cpp:3428-3433`) | W = N+1 every cycle |
| partial compute of the padded batch | `llama-context.cpp:1485`: only 2 <= W <= 8 | N >= 8: the whole padded width is computed every cycle, even after an early stop |
| attention kernel by width | q4p: W <= 5 (`fattn-q4p.cuh:1225`). Production (q8_0 K): vec up to `FA_VEC_MAX_Q`=16, GEMM from `FA_GEMM_MINQ`=17 at kv >= 4096. q4_0/q4_0: tile with direct q4_0 dequant for W >= 6 (`fattn.cu:696-717`), GEMM from 128 | N=4 is the last q4p width. N=8 moves to vec (prod) or tile (q4_0); N=16 moves to GEMM (prod) [I: vec/tile/GEMM cost at verify widths unmeasured on this board] |
| matvec by width | fp16 exact 2..5; fp16 6..16 (OPTLOG 276); >16 MMQ/fold GEMM | W=9 and W=17 cross kernel families |
| CUDA graphs | mode 3 captures single-token graphs only (`ggml-cuda.cu:3079-3110`) | draft steps are always 1 token, so this is no bound. The verify runs eagerly at any width |
| graph shape | verify graph fixed at W, reused; the draft graph is 1 token | no bound. The scheduler reserves up to `-ub`, and the verify must fit one ubatch (`-ub >= N+1`) |
| outputs | target `n_outputs_max_per_seq = N+1` (`server-context.cpp:50-60`, `1204-1205`) | ~1 MiB of host logits per row [I] |
| context remaining | `get_n_draft_max` (`server-context.cpp:517-536`), truncation in `speculative.cpp:3125-3129` | near `n_predict`/`n_ctx` the MTP still drafts N tokens and the excess is cut |
| per-request override | `tools/server/server-schema.cpp:198` (`#if 0`) | n_max/p_min are fixed per server process |
| draft cost | N single-token MTP decodes, ~1.4 ms each at short context (this board, §7) to ~2.7 ms at 260k (Kmic) | N=16: ~22-45 ms of drafting per cycle [I] |

**Can the server run n-max 8 or 16 today without code changes?**
- **n-max 8: functionally yes [I].** No assertion or kernel limit blocks W=9. The costs:
  - a full 9-wide verify every cycle, because the padding is no longer partially skipped;
  - vec/tile attention at W=9;
  - +299 MiB per GPU.
  Production has ~280 MiB free on GPU1 at full depth with a 1440p image (QUICKSTART VRAM section; memory note: peak
  16,101/16,384 MiB). So at 262k with an image it **likely runs out of memory at depth [I]**. Text-only q4_0/q4_0 at
  -ub 2048 has ~1.7-1.9 GiB free (QUICKSTART, estimated on Kmic's board) and should fit [I].
- **n-max 16: functionally yes [I]** (W=17 uses the prefill-style GEMM paths), +898 MiB per GPU. Not in production's
  262k+vision envelope. Possibly fits text-only q4_0/q4_0 at 262k, or anything at reduced context [I].
- **n-max 32:** +2.1 GiB per GPU of snapshots. It does not fit at 262k on either config [I]. It would need a smaller
  `-c`, or snapshots decoupled from n_max.
- **Last measured on this board:** n-max 5 and 6 fell to 27.6/27.9 t/s against 37.7 at n-max 4, because the verify
  left the exact fp16 kernels (`/work/bench/LOG.md:156`, 10-05). Kmic's 6..16-column matvec (OPTLOG 276) merged later,
  and nothing has been re-measured since.
- **What larger n_max would want in code:**
  - raise or remove the `<= 8` active gate, and make vec/tile/GEMM FA honour active columns;
  - decouple `n_rs_seq` from `n_max`, for experiments where the rollback depth can be smaller;
  - re-tune `p_cum_min`, which was tuned for n_max 4.

---

## 4. Controls and their exact semantics

| control | default -> production | semantics (read) | per-cycle change safe? |
|---|---|---|---|
| `--spec-draft-n-max N` | 3 -> 4 | max drafted tokens per cycle. Also sets target `n_rs_seq`, the pad width and the output limits at startup | **No**: fixed at init. `dp.n_max` can only truncate after drafting |
| `--spec-draft-n-min N` | 0 | draft shorter than N is discarded (`speculative.cpp:1948`). With padding the cycle still verifies a full-width batch with `n_real=0` | safe but wasteful; OPTLOG 87: no effect |
| `--spec-draft-p-min P` | 0 -> 0.2 | **only a draft-continuation threshold on the draft's own top-1 probability**: softmax over its top-K candidates (`LLAMA_SPEC_DRAFT_TOPK`), T=1, 81920-row draft vocab. Below P, that step's token is not added and drafting ends (`:1834-1840`). Never used in verification. In sampled mode it tests top-1 p, not the sampled token's q. The help text says "(greedy)" but it applies in both modes | yes, if plumbed; it is a `const` member today |
| `p_cum` schedule / `LLAMA_SPEC_P_CUM=p` | 0 below 16k, ramp to 0.22 at 48k | stop once the product of drafted top-1 p's would fall below the threshold. Only with padded verify (`:1442-1456`); the env var fixes it | yes, if plumbed |
| `LLAMA_SPEC_PAD` | on (with `-np 1`) | fixed-width verify + `n_active` (`server-context.cpp:1455-1459`). `=0`: the width follows the draft, and each change rebuilds the graph (~20-26 ms) | startup |
| `LLAMA_SPEC_SAMPLE_TEMP=T` | 0 (greedy) -> 1.0 | draft samples from its top-K at temperature T and records q; the server then uses the p/q rule (`server-context.cpp:4325`). Lossless (OPTLOG 202) | yes if plumbed: the rule is chosen per cycle from `dists` |
| `LLAMA_SPEC_DRAFT_TOPK=K` | 10 -> 20 | draft candidate count: support of q and normalization of the top-1 p used by p_min/p_cum (`:1507`). The unused backend chain hardcodes 10 (`:1517`) | startup (sampler built in ctor) |
| `LLAMA_SPEC_DRAFT_TOPP=P` | 1.0 | truncates sampled q to its top-p (`:1857-1872`). OPTLOG 233: no gain | startup |
| `LLAMA_SPEC_BLOCK_VERIFY` | 1 | block verification for sampled drafts under a stateless chain; `0` = per-token rule, `2` = paired stats (`sampling.cpp:930-985`) | decided per call |
| `LLAMA_MTP_DRAFT_VOCAB=N` / `_TYPE` | 81920 / q4_0 | draft head over ids < N; the rest get -inf. **The draft can never propose `<|im_end|>` (248046), `<think>` (248068)** or other ids >= 81920 (ids from `srv-v.log`). End of turn and the think/tool markers are always rejection points | load time |
| `LLAMA_MTP_KV_ONLY` | 1 | catch-up stores K/V only | load time |
| `LLAMA_SPEC_LOG=path` | off | one line per cycle: `pos0 draft_us end-of-draft->accept_us n_accepted p1..pk` (top-1 p of every decoded step, the stopping one included) (`:1463-1475`, `2001-2011`) | diagnostic |
| `LLAMA_SPEC_PROFILE=1` | off | synchronized catch-up and draft-step times every 64 drafts. **It adds syncs** | diagnostic |
| `LLAMA_TL=1`, `LLAMA_TRACE=N` | off | stderr timeline marks D0/D1/C0/C1/V0-V2/P0/P1/A0/A1; accept lines | diagnostic |
| `--spec-synth-len/-rates` | off | random acceptance for benchmarking. **Hazard [I]**: with padding on, it can accept padded copies whose matvec columns were never computed (`server-context.cpp:4347-4352` uses the padded `spec_draft`). Use `LLAMA_SPEC_PAD=0`, or p_min=0 below 16k so drafts are never padded | startup |
| `--spec-draft-p-split` | 0.1 | **not used by draft-mtp** | n/a |
| `--spec-draft-backend-sampling` | on | refused under `-sm tensor` (`llama-context.cpp:1290-1293`); CPU sampling | n/a |
| `-ngld` | | **inert for draft-mtp**: the MTP context is built on the target model, so the draft model params are unused (`speculative.cpp:2832-2842`, `2721-2728`) | n/a |
| `-ubd`, `-ctkd`, `-ctvd` | 0 / f16 / f16 -> 64 / q4_0 / q4_0 | draft ubatch cap (bounds its KQ-mask buffer, `:2747-2754`); MTP-layer KV types (288 MiB total at q4_0, 262k) | startup |
| `GGML_CUDA_GRAPHS_PRE_VOLTA` | unset -> 3 | unset: no CUDA graphs on Pascal. Any nonzero value enables them (`ggml-cuda.cu:5563-5570`). `3`: only graphs with >1 node whose matmuls and FA are single-token, i.e. MTP draft steps and plain decode (`:3079-3110`). **Other values (1, 2, 4) capture multi-token verifies too and freeze the padded verify's active count, which gives wrong output** (OPTLOG 212, 220) [I: no guard in current code] | startup |
| `GGML_CUDA_FA_VEC_MAX_Q`, `FA_GEMM_MINQ`, `FA_GEMM_MINKV`, `FA_Q4P` | 2/128/4096/on -> 16/17/-/on | FA routing by width (§3). Production needs 16/17 with q8_0 K, or the tile kernel's whole-cache f16 conversion runs out of memory at depth | startup |

**Per-cycle changes without invalidating anything:** draft length (any stop at or below the startup n_max), p_min /
p_cum, and greedy vs sampled drafting can all change every cycle:
- the draft's KV is erased before the verify (`server-context.cpp:3446`);
- target rollback covers any accepted count up to `n_rs_seq`;
- the acceptance rule is picked per cycle from what the draft recorded.

The output distribution is the target's under all three exact rules: the greedy compare, the per-token p/q rule and
block verify (OPTLOG 202/234; `p100-handoff/tools/block-verify-test`). Seeded text changes with the policy because RNG
use changes, but its distribution does not. Wall-clock cost does depend on the policy. With padding, a shorter draft
saves only the matvec/q4p columns, and only at W <= 8. Without padding, every change of width rebuilds the graph.

---

## 5. Signals at draft-decision time, and after verify

| signal | where / in scope | on device or host | cost [I unless cited] |
|---|---|---|---|
| draft top-K candidates (ids, logits, p over top-K), top-1 p, margin, top-K entropy | `cur_p` at `speculative.cpp:1817` | host | free (already computed) |
| full draft logits (81920 valid) | `llama_get_logits_ith(ctx_dft, i_last)` | host (copied every step) | full-vocab entropy ~0.1 ms/step |
| sampled q and chosen id | `q`, `id`, `dp.dists` (`:1845-1884`) | host | free |
| step index, `pos0` (depth), `p_cum` | `i`, `dp.pos0`, `p_cum[seq_id]` | host | free |
| MTP hidden row h_k (5120 f32) | `h_row` (`:1815`) | host (needed to feed the next step) | free; norms/cosines ~µs |
| target anchor row (pos0-1), verify rows | `pending_h`, `verify_h` | host (DtoH of every target row) | free |
| target per-layer inputs | `llama_set_embeddings_layer_inp(lid)` (`src/llama-ext.h:117-121`, `llama-context.cpp:1040-1045`, `2076`) | extra DtoH per enabled layer | n_tokens x 20 KiB per layer, plus graph outputs; unmeasured |
| acceptance history | cumulative only: `slot.n_accepted_per_pos`, `impl->n_acc_tokens_per_pos`, `slot.stats` | host | a per-cycle ring buffer is trivial; it is not exposed to the MTP impl |
| draft calibration | P(accept \| reached) ~= draft top-1 p (OPTLOG 192); this board: p in [0.9,1) -> 0.96 (n=445), [0.5,0.6) -> 0.43 (n=46) (computed from `spec-log.txt`) | | strongest cheap feature known |
| sampling settings | `slot.task->params.sampling` | host (server only; the MTP impl does not see them) | plumbing needed |
| prompt/task features | `slot.task` (tools, `json_schema`/grammar, template kwargs), `slot.generated_text`, `n_decoded` | host (server only) | free |
| always-wrong positions | target tokens >= 81920 (EOG, think/tool markers) | | free (the token id is known after verify) |

**After verify, all on the host and synchronized:**
- the target's post-chain distribution `cur_p` at each position it sampled. With the per-token rule that is up to the
  first rejection; with block verify, every position (`P[i]`);
- `p(draft token)`: `p_x` (`sampling.cpp:1012-1018`). Greedy path: look the draft token up in `cur_p`;
- raw target logits for every real verify row: `llama_get_logits_ith(ctx_tgt, spec_i_batch[i])`. Padded rows are
  garbage.

So p(draft) is available almost free up to the rejection point. Teacher p beyond it, on the drafted prefix, needs one
sampler call per extra position on a cloned sampler: a prefiltered top-k scan, ~36 µs per row (CHANGES §10
`66bbd1212`), plus the chain [I]. Target argmax is one more logit scan per row.

---

## 6. Trace hooks for Phase 1: env-gated, default off, nothing in the GPU path

All proposed points run on the host after a sync that already exists. None adds a device sync or touches `slot.smpl`
or `slot.spec_dist_rng`.

- **T1 draft side** (extends the existing `LLAMA_SPEC_LOG` record, `spec_log_rec`, `speculative.cpp:1463-1475`):
  - After `:1817` (`cur_p`), record per step: `i`; chosen `id`; top-N (id, logit, p) for N <= 5; `q(id)` (`q[k].p` in
    sampled mode, else `cur_p->data[0].p`); the stop decision and its reason (p_min, p_cum or n_max, from `pc_next`,
    `p_cum_min(pos0)` and `result.size()`); optionally `||h_row||` and cos(h_row, h_prev).
  - In scope: `seq_id, i, dp (pos0, id_last, dists, result), cur_p, id, pc_next, sample_temp, h_row, h_last`.
  - Expose it to the server with one new optional field in `common_speculative_draft_params`
    (`speculative.h:52-76`), e.g. `std::vector<trace_step> * trace = nullptr`.
- **T2 verify side** (`server-context.cpp`, right after `accepted` is computed at `:4352`, before `spec_n_real` is
  reset at `:4353`):
  - In scope: `slot.spec_draft` (padded), `n_real_draft`, `dists`, `use_dist`, `accepted`, `smpl_save` (the
    pre-verify sampler clone), `slot.spec_i_batch`, `slot.ctx_tgt`, `slot.prompt.tokens.pos_next()` (still `pos0`
    here), `slot.task` (id, sampling params).
  - In trace mode, clone `smpl_save` once. For i = 0..n_real-1:
    - `common_sampler_sample(clone, ctx_tgt, spec_i_batch[i])`;
    - read `p(draft[i])`, the target's top-1 id and p, and the candidate entropy from the clone's candidates;
    - take the raw-logit argmax;
    - then `common_sampler_accept(clone, draft[i], true)`.
  - This gives **teacher p and argmax on the identical drafted prefix at every real position, past the rejection
    too.** That is the label the plan's Fidelity head needs.
  - Also record `n_accepted = accepted.size()-1`, the rejection position (= n_accepted if < n_real), the emitted
    correction or bonus token, and which rule ran (greedy, per-token or block).
- **T3 timing:** store `ggml_time_us()` in slot fields at the existing `tl_mark` sites:
  - D0/D1, draft (`:3411/3415`);
  - C0/C1, padding and draft-KV erase (`:3419/3479`);
  - V0/V2, verify enqueue to sync (`:4072/4088`);
  - P0/P1, catch-up (`:4149/4153`);
  - A0/A1, accept (`:4311/4406`).

  These are host clocks; the draft and accept spans already include their syncs.
- **T4 output:** one JSONL record per cycle, written after `common_speculative_accept` (`:4401`) to a `FILE*` with a
  large buffer, flushed every N records and at slot release. Gate it with e.g. `LLAMA_SPEC_TRACE=<path>`; the existing
  `LLAMA_SPEC_LOG` is already the same idea.

Cost: T1 and T3 are negligible. T2 adds n_real sampler calls per cycle, ~0.1-0.5 ms [I]. Keep trace runs out of
timed benchmarks, as the plan says. Byte-identity check: a trace-on and a trace-off run with the same seed must give
the same text.

---

## 7. Cost model: measured numbers, with sources

| quantity | value | source | board |
|---|---|---|---|
| plain decode token (W=1) | 32.2 ms (tg128 31.1 t/s, bf16 wire); 33.1 ms (tg256 30.22, f32, 10-08) | `/work/bench/STATE-2026-10-24.md:37`; `LOG.md:452` | this |
| forward pass W=5 | 51.2 ms (pp5 97.7, bf16); 55.0 ms (pp5 90.8, f32) | `STATE-2026-10-24.md:37` | this |
| forward pass W=9 / 16 / 24 | 91.5 / 145.9 / 243 ms (pp9 98.4, pp16 109.7, pp24 98.7; f32, 10-08) | `LOG.md:452` | this |
| exchange cost | ~15 µs + 0.65 µs/KB per exchange; AllReduce ~10% of a decode token, ~18% of a 5-token batch; GPUs 96-97% busy | `p100-docs/FINDINGS.md:338-345` | this |
| verify by width, 2k | W2-W5: 39.5/46.3/53.1/63.8 ms (OPTLOG 198); 46/53/58/64 (CHANGES §10); W5 ~48.2 later (OPTLOG 223) | OPTLOG 198, 223; CHANGES §10 | Kmic |
| verify by width, 260k | W2-W5: 60/75/91/106 (OPTLOG 198); 75/91/106/~120 (CHANGES §10) | same | Kmic |
| wide passes | B8 77, B10 92, B12 108, B15 149 ms | OPTLOG 276 | Kmic |
| width-change rebuild | ~26 ms (1.8 build + 15.2 alloc + 8.8 meta), 16-25 ms after fixes | OPTLOG 192, 198 | Kmic |
| draft step | 2k: 1.33-1.45 ms (81920 q4_0 head); 260k: 2.68 ms (98304 q6_K head); wall ~4.3 ms at 260k after the mask fast path | OPTLOG 226; CHANGES §10 | Kmic |
| draft, this board | 4 steps: median 5.6 ms (~1.4 ms/step), ctx <= 10k, 10-05 build; 241/244 cycles drafted all 4 steps | `/work/bench/spec-log.txt` (computed) | this |
| verify + catch-up + sampling, this board | median 68.5 ms at W=5 (10-05 build, f32 wire) | same | this |
| catch-up | 5.4 ms/cycle at 2k before KV-only (OPTLOG 177); KV-only: -4 ms at 260k (207); enqueue 2.34 -> 1.11 ms (208) | OPTLOG 177/207/208 | Kmic |
| host phases per cycle | 260k: verify enqueue 65 + sync 41 ms; catch-up enqueue 2.3; accept 0.5; to draft 1.1 (later 0.55); 4 x (1.05 enqueue + 2.45 wait/sample). 2k: 1.45 / 0.44 / 0.55 / 0.89 + 1.44 ms. Draft enqueue later 0.61 -> 0.45 | OPTLOG 207, 208, 223 | Kmic |
| GPU idle in a cycle | 260k: GPUs 76-78% busy; ~5.8 ms after the verify, 2-3.5 ms before each draft step, ~4.5 ms before the verify | OPTLOG 198 | Kmic |
| host sampling | prefiltered top-k ~36 µs vs ~650 µs full build; logits DtoH ~100 µs/row | CHANGES §10 (`66bbd1212`); OPTLOG 88 | Kmic |
| end to end, production | MTP replies 52.80 t/s (bf16 wire), acceptance ~0.47; per-request acceptance 0.27-0.82, mean len 2.03-4.28 | `STATE-2026-10-24.md`; `/work/serve-8090.log` | this |

**Acceptance by position:**

| source | setting | unconditional pos 1/2/3/4 | conditional |
|---|---|---|---|
| `/work/bench/spec-log.txt` (10-05, 244 cycles, ctx <= 10k) | n_max 4, p_min 0.2; draft mode not recorded | 0.78 / 0.64 / 0.51 / 0.37 | 0.78 / 0.82 / 0.80 / 0.73 (computed) |
| `srv-v.log` task 228 (10-05) | temp 0 requests (`ember.completion`) | 0.787 / 0.639 / 0.492 / 0.344 | |
| `srv-ram-build-abhead.log` task 260 (10-06) | temp 0 | 0.783 / 0.633 / 0.533 / 0.367 | |
| OPTLOG "MTP flag tuning", 79, 87 (Kmic, greedy, C++ quicksort prompt) | n-max 3-7 at p-min 0.75: 95.3/93.2/87.9/83.8/75.7% of drafts accepted. p-min 0.2: n-max 4 78.2%, 5 71.9%, 6 (n-min 3) 67.8% | marginal accepted at depth 5 and 6 ~0.47 each [I: assumes full-length drafts] | |
| OPTLOG 124 (Kmic, depth) | 81k: n_draft 6 78.5%; 229k: n_draft 4 81.1%, 6 69.7%; 8 timed out | | |
| OPTLOG 177 (Kmic, real-world sampling) | ~0.55 acceptance, 3.98 drafts per cycle, 2.7-3.97 tokens per cycle | | |

Nothing measures acceptance at depth > 7 on this fork, and nothing past depth 4 on this board with current code.

**Derived bound for a forced-accept burst [I]:** one target pass at W=16 on this board is ~146 ms (9.1 ms/token).
Draft steps cost ~1.4-2.7 ms. So a 15-token burst plus one 16-wide pass is ~(15x2 + 146)/16 ~ 11 ms/token, about
90 t/s if every token is kept. The current verified cycle is ~3 tokens per ~55-60 ms, about 50 t/s. This is the most
Mode B could gain, before any quality loss.

---

## 8. State correctness: tests, and what Mode B can save

**Existing checks:**

| test / script | invariant checked |
|---|---|
| `/work/bench/nd_compare.py` | temp-0 200-token continuation on a 40k prefix, done three ways: full reprocess, cached prefix, slot save/restore. **cached == restored**, including MTP state; full reprocess may differ (batch-split numerics, known and expected) |
| `/work/bench/restore_loop.py` | 4 restores of one slot file plus the same request -> 1 distinct output |
| `/work/bench/determinism.py`, `nd_short.py` | repeats without restore / `cache_prompt:false` -> 1 distinct output |
| `/work/bench/ramcache_test.py` | RAM prompt cache path == live slot (greedy target and drafts) |
| `/work/bench/ember.py restore` | restored continuation byte-identical, and only the suffix reprocessed |
| `tests/test-recurrent-state-rollback.cpp` | `n_rs_seq=8`: snapshot rollback then replay vs a context that never advanced (logit max-diff, NMSE); multi-sequence split replay; restore into a dirty context. The partial-rollback-after-checkpoint case is disabled (TODO at `:402-411`: rollback is single use and needs a preceding ubatch) |
| `tests/test-save-load-state.cpp`, `test-state-restore-fragmented.cpp` | state save/load round trips |
| `p100-handoff/tools/block-verify-test/bvtest.cpp` | Monte Carlo: block verify reproduces the target distribution (1.85-2.21 SE null; planted bug 337 SE) |
| `tools/gate.sh` | PPL band, FA eval, op suite. It stops at a false "fattn-fold-sass.h is stale"; run PPL directly |
| server-level (OPTLOG 198; STATE 10-24) | padded == unpadded text at 2k when draft lengths don't vary; seeded sampled-draft runs 5/5 identical |

Gaps:
- no test with n_max > 4;
- no test of synthetic acceptance combined with padding;
- no test that compares a committed burst with teacher-forced decoding of the same tokens. That would be the same code
  path as a verify, so equality holds up to batch-split numerics: different W means different kernels.

**Can "commit unverified tokens" (Mode B) skip target compute? No.** The verify batch is the processing pass. It
advances target KV and GDN state by all n+1 tokens and produces every row's `h_nextn` for the MTP's next anchor and
its catch-up.

What verification costs on top of processing the same tokens:
1. **Logits at every position:** every row is an output (`:574-576`). That means W columns of the 5120x248320 LM head
   and W rows of ~970 KiB logits to host, vocab-sharded over the x16 and x4 links. A pure processing pass needs logits
   only at its last row; the `h` rows are needed either way.
2. **Host sampling per position:** up to the rejection, or at every position under block verify.
3. **Rollback machinery:** (n_max+1) GDN snapshots written per verify (VRAM and bandwidth, §3) and the `seq_rm` /
   `rs_idx` bookkeeping, which is µs.
4. **Compute on rejected or padded tokens:** padding is partly skipped (matvec and q4p, W <= 8). Rejected tokens are
   fully wasted.

The MTP catch-up is needed in either mode. Mode B can therefore save (4) and some of (1)-(3), but never the forward
pass. A free-running burst of L tokens plus a re-anchor is computationally the same as an L-token draft verified and
force-accepted in one target pass.

**The realistic Mode B is lenient acceptance.** At the existing accept site, the target's p for every real position
is already on the host (after T2-style sampling past the rejection, or by block verify). A rule that accepts more than
min(1, p/q) allows:
- accept if the draft token is in the target's top-k/top-p, or if p >= alpha*q, or if the target top-1 p is low (a
  tie);
- a cap on consecutive lenient accepts.

It raises tokens per cycle at zero extra GPU cost. The bonus token is then drawn from the verify logits at the last
accepted position, which were computed on the drafted prefix. Constraints:
- label it approximate;
- keep sampler state consistent through `common_sampler_accept`;
- never apply it when a grammar is active (tools or JSON schema) or to tokens >= 81920;
- remember it changes the output distribution.

Its upper bound at n_max 4: 5 tokens per cycle against ~3 today. Deferring and batching the target pass over a longer
MTP-only burst is the other lever (§7 bound).

---

## 9. The "Discord reference" command against this fork

It is Kmic's QUICKSTART command from before `-b 2048` became the documented default: the current QUICKSTART
(`p100-docs/QUICKSTART.md:29-37`) differs only by `-b 2048`.

| item | honored? | effect on this fork / board |
|---|---|---|
| `GGML_CUDA_P2P=1` | read, **inert here** | peer access is enabled only where `cudaDeviceCanAccessPeer`=1 (`ggml-cuda.cu:408-420`, VMM grants `:640-656`, one-kernel exchange needs both directions `:1352`). This board has 0 both ways (`/work/bench/LOG.md:6`), so the host-staged AllReduce is chosen automatically (`allreduce.cu:415-424`). Since the 10-06 hard reset Tyler's rule is "P2P retest only while he watches" (`LOG.md:274`), so drop it from the adapted command |
| `GGML_CUDA_GRAPHS_PRE_VOLTA=3` | yes | single-token graphs (draft steps, plain decode); same as production |
| `LLAMA_SPEC_SAMPLE_TEMP=1.0` | yes | sampled drafts plus p/q verification. With this sampling (presence 0, no grammar/budget) **block verify is on**; production's thinking-off presence 1.5 uses the per-token rule |
| `LLAMA_SPEC_DRAFT_TOPK=20` | yes | q over 20 candidates; also the normalization of the top-1 p that p-min tests |
| `-ngl 99 -sm tensor -fa 1` | yes | |
| `-ctk q4_0 -ctv q4_0` | yes | KV 2304 MiB/GPU at 262k (`srv-v.log`). Attention: q4p for 1-5 queries, then tile with direct q4_0 dequant, then GEMM/fold from 128 queries; FA env overrides not needed [I from `fattn.cu:696-717`, `fattn-tile.cuh:1418`]. Production uses q8_0 K (KLD 0.00274 -> 0.00116, QUICKSTART table) |
| `-c 262144` | yes | |
| `-b 32768` | yes | same GPU passes as `-b 2048` (QUICKSTART:90). A dropped 64k prompt holds the slot ~33 s instead of ~2 s. Host `embd_nextn` buffer 5120x32768 f32 ~640 MiB (`llama-context.cpp:2203-2206`; OPTLOG 193 ~671 MB) |
| `-ub 2048` | yes | **memory: feasible text-only [I].** This board served q4_0/q4_0, 262k, `-ub 2048`, `-b 32768` with an F16 projector on 10-06 (`/work/bench/SUMMARY.md:150-156`). QUICKSTART estimates ~1.7-1.9 GiB free on GPU0 text-only. Production uses 1024 only because of q8_0 K plus a 1440p image |
| `-np 1` | yes | required for padded verify (`server-context.cpp:1455`) |
| `--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2` | yes | `n_rs_seq`=4. p-min is the draft-continuation threshold of §4, **not** a 20% acceptance threshold and unrelated to any auto-accept risk threshold. The p_cum depth schedule adds to it |
| `-ngld 99` | **inert** for draft-mtp | |
| `-ubd 64 -ctkd q4_0 -ctvd q4_0` | yes | draft ubatch cap; MTP KV 144 MiB/GPU |
| `--jinja`, no `--reasoning` | yes | reasoning `auto` (`arg.cpp:3721-3735`), so the template default applies: **Qwen3.8 thinks by default** (QUICKSTART:21). Replies start with `<think>`. No budget sampler unless a request sends a budget, so block verify applies. Matched comparisons with production must send `enable_thinking` explicitly (`run_suite.py` THINK=) |
| `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0` | yes | the model card's thinking set; presence defaults to 0 |
| `--host 0.0.0.0 --port 8080` | yes | no API key; use 127.0.0.1 for tests |
| implied defaults | | `-ctxcp 32`, `--cache-ram 8192`; **f32 exchange wire** (`GGML_CUDA_AR_BF16_THRESHOLD` default 0 on Pascal without P2P, `allreduce.cu:453`; production runs bf16, +6.8% MTP replies); padded verify on; draft vocab 81920; block verify on |

The screenshot supplies no throughput or acceptance data. The model is "Qwen3.8-27B-Q6_K", while this machine has
`Qwen3.8-27B-UD-Q6_K.gguf` (Unsloth dynamic: delta-net alpha/beta are Q8_0, FINDINGS:335). Record that as a divergence.

**Adapted command (B2'), for a separate run:**

```bash
GGML_CUDA_GRAPHS_PRE_VOLTA=3 LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20 \
/work/src/llama.cpp/build-vision/bin/llama-server \
  -m /work/models/Qwen3.8-27B-UD-Q6_K.gguf \
  -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
  -c 262144 -b 32768 -ub 2048 -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 \
  -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 \
  --jinja --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 \
  --host 127.0.0.1 --port 8091
```

Divergences from the transcript:
- `GGML_CUDA_P2P=1` dropped (inert, plus Tyler's rule);
- model path;
- bind address and port.

Optional second arm: `GGML_CUDA_AR_BF16_THRESHOLD=1`, to separate the wire effect. It needs both GPUs, so production
on :8090 must be stopped. That is Tyler's call.

**Differences from production** (`/work/bench/think/prod-command.txt`):
- KV cache: q8_0 K in production vs q4_0;
- `-ub` 1024 vs 2048;
- `-b` 2048 vs 32768;
- thinking: `--reasoning off` + `LLAMA_THINK_BY_BUDGET=1` in production vs thinking on;
- sampling: `LLAMA_SAMPLING_BY_MODE=1` (thinking off 0.7/0.8/20/0, presence 1.5, i.e. the per-token rule) vs
  1.0/0.95/20/0 (block verify);
- exchange wire: bf16 vs f32;
- FA routing envs 16/17;
- `-ctxcp 32 --cache-ram 24576`;
- vision (Q8_0 projector on CUDA1, 3600 image tokens);
- API key, alias, slot-save path, port 8090;
- both run at a 150 W power limit.

---

## 10. Capabilities and gaps

| capability | status |
|---|---|
| MTP-only autoregressive rollout up to n_max | **exists** (`speculative.cpp:1910-1916`) |
| commit tokens without a target pass | **impossible** in this engine |
| exact verification: greedy, p/q, block | exists; block verify only for stateless chains |
| per-cycle draft length / p_min / greedy-vs-sampled | possible with small plumbing; distribution-safe |
| per-request spec params | disabled (`server-schema.cpp:198`) |
| n_max > 4 | runs [I], but costs full width at W >= 9 and +75 MiB/GPU per extra token of n_max |
| cheap draft signals (top-K, p, margin, h) | on host, free |
| teacher p on drafted prefix, all positions | logits on host; needs a clone-sampler pass (T2) |
| target probs via API in MTP mode | **missing** (`result.prob = 1.0f`) |
| per-cycle trace | partial (`LLAMA_SPEC_LOG`); T1-T4 proposed |
| backend sampling / device-chained draft steps | blocked by vocab-sharded logits under `-sm tensor` (OPTLOG 88, 198) |
| draft proposes EOG/think/tool tokens | **never** (draft vocab 81920) |
| rollback depth > n_rs_seq | falls to the checkpoint path, unreachable for qwen35 |
| draft-step time on this board with current code | **unmeasured** (no `LLAMA_SPEC_PROFILE` run on the no-P2P board) |

---

## 11. Feasibility verdict per intervention

| intervention | verdict | notes |
|---|---|---|
| Adaptive depth (0..n_max per cycle) | **Feasible now**, exact | hook: the stop test at `speculative.cpp:1834`. Savings are partial: with padding only matvec and q4p skip columns (W <= 8); exchanges, GDN and norms run the full width. Without padding, each change of width costs a ~20-26 ms graph rebuild. Depth above the startup n_max needs a restart |
| Adaptive p-min / p_cum | **Feasible**, exact | make the thresholds per-cycle inputs in `common_speculative_draft_params` |
| Greedy vs sampled drafting per cycle | **Feasible**, exact | make `sample_temp` per cycle; the server already picks the rule from `dists` (`server-context.cpp:4325`). The gain depends on target temperature (OPTLOG 196: sampled loses at T 0.3; 202: wins at T 1.0) |
| Lenient / approximate acceptance (Mode B) | **Feasible, approximate**, zero GPU cost | at the accept site (`server-context.cpp:4334-4352` / `sampling.cpp`). Exclude grammar paths and tokens >= 81920. Gain is bounded by the rejection rate, ~+50% tokens per cycle at n_max 4 if everything were accepted |
| Free-running long MTP bursts (8-32) | **Mechanism exists; constrained** | n_max 8 OK at reduced VRAM margin; 16 needs ~+0.9 GiB/GPU; 32 doesn't fit at 262k [I]. Every burst still needs a target pass of width L+1. Quality past depth 4 is unknown: measure it with trace T2 before building anything |
| Mode B that skips target compute | **Not possible** | §8 |
| Trace hooks T1-T4 | **Feasible**, env-gated | host only; T2 adds ~0.1-0.5 ms per cycle [I]; keep out of timed runs |
| Learned three-head controller inputs | available cheaply on host | draft top-K, h_k, depth, position, history, task features. Teacher labels come from T2 |

---

## 12. Blockers and risks

1. **Target compute cannot be skipped.** Mode B reduces to lenient acceptance and/or deferred, batched target passes.
2. **Depth is tied to VRAM.** `n_rs_seq = n_max` costs 74.8 MiB per GPU per unit. Production has ~280 MiB free on
   GPU1 at 262k with an image.
3. **The padded verify is partial only for W <= 8** (`llama-context.cpp:1485`), and only for matvec, GLU and q4p.
   Above that, every cycle pays the full width.
4. **Kernel regime changes at W=6, 9 and 17** (§3). n-max 5 and 6 measured -27% on this board before Kmic's 6..16
   matvec, and nothing has been re-measured since.
5. **Per-request speculative params are disabled.** Every sweep point is a server restart, or a code change for
   per-cycle control.
6. **Host round trip per draft step.** Logits are vocab-sharded under `-sm tensor`, so neither backend sampling nor
   device-chained draft steps is available (OPTLOG 88, 198).
7. **Block verify is off under production's thinking-off sampling** (presence 1.5) and with budgets or grammars.
   Results under Kmic's sampling won't transfer one-to-one.
8. **`GGML_CUDA_GRAPHS_PRE_VOLTA` other than 3**, and **synthetic acceptance with padding**, can both give wrong
   output [I from code + OPTLOG 212/220]. Avoid them in experiments.
9. **Missing measurements on this board:** draft-step time, verify time by width at depth, and acceptance past depth
   4, all with current code.
10. **Production shares the GPUs.** Every Phase 1 run needs :8090 stopped, which is Tyler's call. Benchmark hygiene:
    `pkill -x llama-server`, never `pkill -f`; alternate arms ABAB; watch the 80 C rule.

**Existing harnesses for Phase 1:**
- `/work/bench/mtp3/run_suite.py` (natural-length suite, JSONL);
- `/work/bench/ember.py` (ttft/latency/restore);
- `tools/depth-bench.py` (depth staircase);
- `/work/bench/forks/paired.py` (per-request pairing, ~0.3% noise);
- diagnostics `LLAMA_SPEC_LOG`, `LLAMA_SPEC_PROFILE`, `LLAMA_TL`, `GGML_CUDA_OP_PROFILE`;
- server log lines "draft acceptance" and "acc per pos".
