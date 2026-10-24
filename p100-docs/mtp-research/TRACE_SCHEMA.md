# LLAMA_SPEC_TRACE record schema

Branch `mtp-adaptive-research`, commit `6928a6854` (2026-10-24). Built and run: same-seed output identical with the trace on and off; +2.3% decode time while on.

Enable with `LLAMA_SPEC_TRACE=<path>` on `llama-server`. The server reads it once, when speculative decoding
initializes, and appends to the file. Unset means off. Output is JSONL: one object per line, one line per verify
cycle per slot. It is buffered (1 MiB) and flushed every 64 records, at slot release and at process exit.
A `kill -9` loses up to 63 records.

Code: `tools/server/server-context.cpp` (`server_spec_trace`, `server_spec_trace_cycle`),
`common/speculative.cpp` (MTP draft loop), `common/speculative.h` (`common_speculative_trace_step`).

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
entry at index `n_real`: the step whose decode stopped drafting. Its token was **not** added, and its `id` is the
draft's top-1. In total there are `min(steps decoded, n_real+1)` entries.

| field | type | meaning |
|---|---|---|
| `id` | int | drafted token: the argmax, or the sampled token in sampled mode |
| `q` | float | the draft's probability of `id`. In sampled mode this is the q the token was drawn from (after `draft_temp` and `LLAMA_SPEC_DRAFT_TOPP`), the value recorded in `dp.dists`. Otherwise it is the top-1 p |
| `top` | [[id,p],...] | the draft's top-5 candidates: top-K (`LLAMA_SPEC_DRAFT_TOPK`), p = softmax over those K at T=1. This is the distribution `p_min` and `p_cum` test. `top` is always at T=1. In sampled mode `q` belongs to `id`, which need not be `top[0]` |
| `H` | float | entropy (nats) over the same top-K at T=1 |
| `stop` | string, optional | present only on the step where drafting stopped. `p_min`: top-1 p < `--spec-draft-p-min`. `p_cum`: the product of top-1 p's would fall below the depth-scheduled cutoff. `n_max`: the draft reached `--spec-draft-n-max`; this entry's token was added and it is entry `n_real-1`. `cut`: the token was drafted but removed afterwards, by `--spec-draft-n-min` or by the remaining-context limit (`dp.n_max`) |
| `thr` | float, optional | the threshold behind `stop`: p_min, the `p_cum` cutoff at this depth, or n_max. Absent for `cut` |

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

## Not recorded

- A cycle that restores a speculative checkpoint (FULL rollback) returns before the accept, so it writes no record.
  Its replay cycle is recorded with `rule: "replay"`. This path is unreachable for qwen35.
- Draft-side `||h||` or cos(h, h_prev) (optional in RECONNAISSANCE §6) are not recorded.
- Cycles of non-MTP draft types produce records, but with an empty `draft` array (only the MTP draft fills it).
