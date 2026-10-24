# LLAMA_SPEC_TRACE record schema

Branch `mtp-adaptive-research`, commit `6928a6854` (2026-10-24); the stop rule's `"iso"` stop value from `8d0d6b6dd`, the
counterfactual fields from `ffdc8e49d`. Built and run: same-seed output identical with the trace on and off; +2.3% decode time
while on.

Enable with `LLAMA_SPEC_TRACE=<path>` on `llama-server`. The server reads it once, when speculative decoding
initializes, and appends to the file. Unset means off. Output is JSONL: one object per line, one line per verify
cycle per slot. It is buffered (1 MiB) and flushed every 64 records, at slot release and at process exit.
A `kill -9` loses up to 63 records.

Code: `tools/server/server-context.cpp` (`server_spec_trace`, `server_spec_trace_cycle`),
`common/speculative.cpp` (MTP draft loop), `common/speculative.h` (`common_speculative_trace_step`).

`LLAMA_SPEC_TRACE_TOPK=<K>` (default 0) adds the raw logits and penalty window that recomputing the target's
distribution under another sampling setting needs ([below](#llama_spec_trace_topk-counterfactual-fields)). Unset or 0,
records are byte-for-byte the format above.

## Example (values illustrative)

```json
{"task":41,"slot":0,"t_us":91234567890,"pos0":25311,"n_ctx_used":25311,"n_draft_padded":4,"n_real":3,
 "sampling":{"temp":0.7,"top_k":20,"top_p":0.8,"min_p":0,"penalty_present":1.5,"penalty_repeat":1,"penalty_freq":0,"penalty_last_n":64},
 "draft_temp":1,"rule":"dist-pertoken","grammar":false,
 "draft":[{"id":279,"q":0.93,"top":[[279,0.93],[264,0.03],[11,0.01],[13,0.01],[323,0.005]],"H":0.36},
          {"id":3363,"q":0.71,"top":[[3363,0.71],[1888,0.2],[5,0.04],[17,0.02],[90,0.01]],"H":0.92},
          {"id":13,"q":0.55,"top":[[13,0.55],[11,0.3],[323,0.1],[198,0.02],[220,0.01]],"H":1.1},
          {"id":198,"q":0.12,"top":[[198,0.12],[271,0.1],[220,0.09],[13,0.08],[11,0.07]],"H":2.9,"stop":"p_min","thr":0.2}],
 "teacher":[{"p_draft":0.97,"top1":[279,0.97],"H":0.15,"logit_draft":24.1,"logit_max":24.1,"argmax":279},
            {"p_draft":0.64,"top1":[3363,0.64],"H":0.8,"logit_draft":19.2,"logit_max":19.2,"argmax":3363},
            {"p_draft":0.2,"top1":[11,0.71],"H":0.9,"logit_draft":17.0,"logit_max":18.3,"argmax":11},
            {"top1":[220,0.4],"H":1.7,"logit_max":15.9,"argmax":220}],
 "n_accepted":2,"emitted":[279,3363,11],
 "timing":{"draft_us":4300,"pad_us":35,"verify_us":51200,"verify_enqueue_us":31000,"catchup_us":1100,
           "accept_us":420,"trace_us":700,"cycle_us":57900,"gap_us":850}}
```

## Top level

| field | type | meaning |
|---|---|---|
| `task` | int | `slot.task->id` |
| `slot` | int | slot id |
| `t_us` | int | `ggml_time_us()` when the record was written, which is also the end of the accept (A1) |
| `pos0` | int | position of `id_last`, the first verify row. This is the code's `dp.pos0`, the same value `LLAMA_SPEC_LOG` writes. **Drafted token `i` (0-based) sits at `pos0+1+i`.** The first drafted token is therefore at `pos0+1`, not `pos0` |
| `n_ctx_used` | int | tokens in the slot's context before this cycle's verify batch. Equals `pos0` for text-only prompts. Images can make the two differ |
| `n_draft_padded` | int | width of the verified draft, padding included (`spec_draft.size()`). The verify batch is this plus 1 |
| `n_real` | int | real drafted tokens. Equals `n_draft_padded` when the draft was not padded |
| `sampling` | object | the request's target sampling: `temp`, `top_k`, `top_p`, `min_p`, `penalty_present`, `penalty_repeat`, `penalty_freq`, `penalty_last_n`, from `slot.task->params.sampling` (after `LLAMA_SAMPLING_BY_MODE` defaults) |
| `draft_temp` | float | `LLAMA_SPEC_SAMPLE_TEMP`. 0 means greedy drafts |
| `rule` | string | the acceptance rule that ran this cycle: `dist-block` (block verification), `dist-pertoken` (p/q rule), `greedy` (target sample vs draft token), `replay` (checkpoint replay, unreachable for qwen35), `synth` (`--spec-synth-*`). Computed the same way the server's branches and `common_sampler_sample_and_accept_n_dist` choose it |
| `grammar` | bool | the target sampler has a grammar or a reasoning-budget sampler. With a grammar, `dist-pertoken` uses the plain compare at grammar positions, and the teacher may stop early (see below) |
| `draft` | array | per draft step (below). Empty for `replay` |
| `teacher` | array | per verify row 0..`n_real` (below) |
| `n_accepted` | int | accepted drafted tokens: `accepted.size()-1` |
| `emitted` | int[] | token ids committed this cycle: the accepted drafts, then the correction or bonus token. Streaming can stop partway if a stop condition fires |
| `timing` | object | host µs (below) |

## `draft[i]`

Entries `0..n_real-1` are the real drafted tokens. If drafting stopped on `p_min` or `p_cum`, there is one more
entry at index `n_real` (also on `iso`): the step whose decode stopped drafting. Its token was **not** added, and its `id` is the
draft's top-1. In total there are `min(steps decoded, n_real+1)` entries.

| field | type | meaning |
|---|---|---|
| `id` | int | drafted token: the argmax, or the sampled token in sampled mode |
| `q` | float | the draft's probability of `id`. In sampled mode this is the q the token was drawn from (after `draft_temp` and `LLAMA_SPEC_DRAFT_TOPP`), the value recorded in `dp.dists`. Otherwise it is the top-1 p |
| `top` | [[id,p],...] | the draft's top-5 candidates: top-K (`LLAMA_SPEC_DRAFT_TOPK`), p = softmax over those K at T=1. This is the distribution `p_min` and `p_cum` test. `top` is always at T=1. In sampled mode `q` belongs to `id`, which need not be `top[0]` |
| `H` | float | entropy (nats) over the same top-K at T=1 |
| `stop` | string, optional | present only on the step where drafting stopped. `iso`: the lookup stop rule (`LLAMA_SPEC_STOP_ISO`) estimated the drafted prefix's survival below theta. `p_min`: top-1 p < `--spec-draft-p-min`. `p_cum`: the product of top-1 p's would fall below the depth-scheduled cutoff. `n_max`: the draft reached `--spec-draft-n-max`; this entry's token was added and it is entry `n_real-1`. `cut`: the token was drafted but removed afterwards, by `--spec-draft-n-min` or by the remaining-context limit (`dp.n_max`) |
| `thr` | float, optional | the threshold behind `stop`: theta (`iso`), p_min, the `p_cum` cutoff at this depth, or n_max. Absent for `cut` |

`p_cum` itself is not written. It equals the product of `top[0][1]` over the earlier entries, multiplied by this
entry's `top[0][1]`.

## `teacher[i]`

Row `i` is the target's output at verify row `i` (`spec_i_batch[i]`), which predicts the token after row `i`'s
token, so row `i` scores `draft[i]`. Row `n_real` is the bonus position, after the last real drafted token, and has
no drafted token.

The rows come from a **clone of the pre-verify sampler** (`smpl_save`). The clone is teacher-forced along the
drafted tokens: sample at row `i`, then accept `draft[i]` into the clone. Rows past the first rejection therefore
hold the distribution the target would have had on the drafted prefix. Row 0 always equals the real verify's first
distribution. `slot.smpl`, `slot.spec_dist_rng` and the draft RNG are never touched.

| field | type | meaning |
|---|---|---|
| `p_draft` | float | probability of `draft[i]` in the target's post-chain candidate distribution, 0 if the token is not a candidate. This is exactly the `p_x` the p/q rule uses. Absent on the bonus row |
| `top1` | [id,p] | the target's most probable candidate |
| `H` | float | entropy (nats) of the target's candidate distribution |
| `logit_draft` | float | raw target logit of `draft[i]` (`llama_get_logits_ith`), before penalties and temperature. Absent on the bonus row |
| `logit_max` | float | raw max logit over the full vocabulary (plain scan) |
| `argmax` | int | raw argmax, i.e. the temp-0 target token ignoring penalties. Not in the original field list; it comes free with the scan |

Candidate set: with the sampler prefilter (`prefilter_k`), the candidates are the top-k after penalties, not the
whole vocabulary. That is the target's real sampling distribution, the one verification tests, so `p_draft = 0`
means "outside the target's top-k after penalties".

The teacher is shorter than `n_real+1` only when a grammar rejected a drafted token on the forced prefix (an
exception, caught). There are no rows past that point.

## `timing` (host µs, `ggml_time_us`, taken next to the existing `tl_mark` sites)

| field | marks | meaning |
|---|---|---|
| `draft_us` | D1−D0 | the `common_speculative_draft` call (it includes the per-step syncs and host sampling, plus a few µs of T1 trace work). Absent when this cycle did not draft (`replay`) |
| `pad_us` | C1−C0 | padding, draft-KV erase, checkpoints |
| `verify_us` | V2−V0 | target verify decode, enqueue through sync |
| `verify_enqueue_us` | V1−V0 | `llama_decode` return (graph build plus enqueue) |
| `catchup_us` | P1−P0 | MTP catch-up (`common_speculative_process`) |
| `accept_us` | A1−A0−`trace_us` | host acceptance (sampling, rule, rollback, `common_speculative_accept`), **excluding** the teacher pass |
| `trace_us` | | the teacher pass (clone, n_real+1 sampler calls, logit scans). Pure trace overhead |
| `cycle_us` | A1−D0 (or A1−C0 without a draft) | wall time from draft start to accept end. **It includes `trace_us`** |
| `gap_us` | D0 − previous A1 | host time between cycles: the previous cycle's streaming (`process_token`), the trace write, and the server loop. Absent on a task's first cycle |

D, C, V and P are batch-wide phases (one draft call and one verify decode serve every slot in the batch), so they are
stamped once on the server's trace state. Only A0 and A1 are per slot. V and P are taken from the decode chunk at
batch offset 0, where the generating slots' verify rows always sit. This is exact for `-np 1`. With several
generating slots, every slot's record shows the same D/C/V/P spans.

## LLAMA_SPEC_TRACE_TOPK: counterfactual fields

Added 2026-10-24 on `mtp-adaptive-research` (`ffdc8e49d`). Run in window 13 (K 128): `counterfactual.py selfcheck` reproduced
the server's p of every drafted token within 5.0e-7 (thinking off, 19,949 rows) and 5.3e-7 (thinking on, 72,943 rows). With `LLAMA_SPEC_TRACE_TOPK=K`, K > 0
(clamped to the vocabulary size), every record gets the fields below, so that `analysis/counterfactual.py` can replay
the target's sampler chain under any `temp`/`top_k`/`top_p`/`min_p`/penalty setting and give
p_S(x) for each drafted token x, hence the p/q acceptance probability min(1, p_S(x)/q(x)) (q does not depend on the
target's sampling). The server reads K once, with `LLAMA_SPEC_TRACE`.

```json
{"task":41, ..., "rule":"dist-pertoken","grammar":true,"grmr":false,"pen_hist":[13,279,3363, ...,11,279],
 "draft":[...],
 "teacher":[{"p_draft":0.97,"top1":[279,0.97],"H":0.15,"logit_draft":24.1,"logit_max":24.1,"argmax":279,
             "raw":[[279,24.0981445],[264,19.5512695], ...],"lse_all":24.1290359},
            ...,
            {"p_draft":0,"top1":[248069,1],"H":0,"logit_draft":15.2,"logit_max":21.7,"argmax":13,
             "raw":[[13,21.6982422], ...],"lse_all":22.0310116,"forced":true}],
 ...}
```

Per record (after `grammar`):

| field | type | meaning |
|---|---|---|
| `grmr` | bool | the target sampler has a grammar sampler. `grammar` is also true for a reasoning-budget sampler alone (every thinking request with a budget), so `grmr` is the one that says whether a grammar could have constrained a row |
| `pen_hist` | int[] | the penalty window before this cycle's drafted tokens: the last `penalty_last_n` text tokens of the slot's context up to and including `id_last` (the token at `pos0`), oldest first. Shorter when the context is. Media positions are skipped, as `init_sampler` skips them when it feeds the prompt to the sampler, so these are exactly the tokens the penalty sampler's ring buffer holds before the verify. The window for teacher row `i` is `(pen_hist + draft[0..i-1].id)[-penalty_last_n:]`, the clone having accepted the drafted tokens before row `i`. Present whenever `sampling.penalty_last_n > 0`, **also when the penalty weights are neutral** (presence 0, repeat 1, frequency 0, the thinking-on defaults), so a counterfactual setting can add a penalty with a window up to this long. Absent when `penalty_last_n` is 0 |

Per `teacher[i]` (every row, the bonus row included; after `argmax`):

| field | type | meaning |
|---|---|---|
| `raw` | [[id,logit],...] | the K largest **raw** target logits of the row (`llama_get_logits_ith`, the same row `logit_max` scans), before any sampler, in descending order; equal logits in ascending id order (the order the sampler's top-k prefilter keeps). Logits are written with 9 significant digits, so each reads back as the exact float32 |
| `lse_all` | float | log(Σ_v e^logit_v) over the **whole** raw row (248,320 entries), 9 digits. exp(logit − `lse_all`) is the softmax at T = 1, before penalties. e^x comes from a degree-6 polynomial (relative error < 3e-7 per term, a vectorizable stand-in for `expf`), summed in double: `lse_all` is good to 3e-7 (measured: 4e-9 on a 248,320-entry row). At another temperature Σ_v e^(logit_v/T) does not follow from it, only its top-K part; see below |
| `forced` | true, optional | present only when the reasoning-budget sampler was forcing its end sequence at this row (its state before the row's sample). The row's distribution is then a point mass on the forced token, which is `top1`, whatever the sampling setting; `raw` still holds the model's logits |

With K > 0 `argmax` and `logit_max` come from the selection instead of the plain scan: the same values (ties: the
lowest id).

**Size.** About 20 bytes per `raw` entry: K = 128 adds ~2.5 KB per row, so ~12 KB per cycle at n_max 4 and ~22 KB at
n_max 8 (a 20k-cycle run: 250-450 MB), against ~1.5 KB per record without it.

**Cost.** Per row the selection (one pass with a K-entry heap) and the log-sum-exp pass replace the max scan. Measured
on a 248,320-float row with this build's flags (-O3, SSE2), while production was serving: selection 0.20-0.23 ms,
log-sum-exp 0.45 ms, JSON 0.03-0.06 ms (K 64-128), less the 0.24 ms scan: **+0.45 ms per row**, i.e. +2.3 ms per cycle
at n_max 4 and +4 ms at n_max 8, all inside `trace_us` (an `expf` loop would cost 0.9 ms per row instead of 0.45).
That is roughly +4% (n_max 4) to +7% (n_max 8) of a 57 ms cycle on top of the trace's own +2.3%: keep timed runs
without it.

**Choosing K.** The top_k after penalties is exact from the list when at least top_k listed tokens stay above the
list's smallest logit after the penalties, which K ≥ top_k + (distinct penalized tokens) guarantees. For the card
settings (top_k 20, presence 1.5 over 64 tokens) K = 84 always suffices; **K = 128** leaves margin for top_k up to 64
with a 64-token window. `counterfactual.py` checks the condition per row (`info["exact"]`).

**What a setting may change offline.** `temp`, `top_k`, `top_p`, `min_p`, `min_keep`, `penalty_present`,
`penalty_repeat`, `penalty_freq`, and `penalty_last_n` up to the recorded one (longer only while the whole context fits
in `pen_hist`), in the default sampler order or another (`samplers`). With `top_k` ≤ 0 the whole-vocabulary softmaxes
take the tokens outside the list from `lse_all`: exact at T = 1, which covers top_p and min_p in the default order
(they run before temperature); dist normalizing the whole vocabulary at T ≠ 1 (top_k ≤ 0, top_p ≥ 1, min_p ≤ 0) is
an estimate. Not replayable, and not recorded: logit biases (`ignore_eos` is one; so are a vocabulary's suppress tokens,
which also turn the prefilter off), dry, top_n_sigma, typical, xtc, dynatemp, mirostat, a grammar. The record's
`sampling` object does not hold `min_keep` or the sampler order; the suite's requests use the defaults (0, the default
order).

**Check.** `python3 analysis/counterfactual.py selfcheck <trace.jsonl>` recomputes `p_draft` at each row's own sampling
for every row of a cycle without `grmr` and prints the max and 99th-percentile error and the fraction above 1e-4.
`p_draft` is written with 6 digits, so the floor is ~5e-7. Before any such trace existed, its chain was checked against
this fork's C++ samplers (`llama_sampler_init_penalties/top_k/top_p/min_p/temp_ext/dist` from `libllama`, CPU only, as
`common_sampler_init` chains them) on 400 random 3,000-token rows and settings (temp 0-1.3, top_k 0-40, top_p,
min_p, min_keep, presence/repeat/frequency over 0-64 tokens): identical candidate sets, max |Δp| 1.5e-8, and the same
from the top 128 plus `lse_all` on the 364 rows it flagged exact. The prefilter's subset gave the same p as the whole
vocabulary in every case.

## Not recorded

- A cycle that restores a speculative checkpoint (FULL rollback) returns before the accept, so it writes no record.
  Its replay cycle is recorded with `rule: "replay"`. This path is unreachable for qwen35.
- Draft-side `||h||` or cos(h, h_prev) (optional in RECONNAISSANCE §6) are not recorded.
- Cycles of non-MTP draft types produce records, but with an empty `draft` array (only the MTP draft fills it).
- Without `LLAMA_SPEC_TRACE_TOPK`: raw logits other than `logit_draft`/`logit_max`, the penalty window, the reasoning
  budget's forcing state, and whether `grammar` means a grammar sampler or only a reasoning budget.
