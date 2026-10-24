# Adaptive MTP research (branch `mtp-adaptive-research`, 2026-10-24)

The question, from the owner's plan ([PLAN.md](PLAN.md)): can three small heads (Task Type, Difficulty, Fidelity)
vary native MTP drafting on the fly well enough to beat the already-optimized fixed policy on this dual-P100
Qwen3.8-27B Q6_K setup? And if unverified MTP bursts are feasible, what quality-vs-speed frontier does
confidence-gated re-anchoring give?

## Answer

| head or idea | verdict | evidence |
|---|---|---|
| Fidelity: when to stop drafting | **built, measured: +4.7%** (held-out prompts +8.0%) | two isotonic lookup tables replace the p-min test (`LLAMA_SPEC_STOP_ISO=0.125`, commit `8d0d6b6dd`); exact output distribution |
| Task Type | **not worth building** | true category labels add +0.0% on top of the Fidelity rule; the draft's own confidence already tracks task switches inside a reply |
| Difficulty | **not worth building** | the previous cycle predicts the next (Spearman -0.22), but adds +0.1% once the current draft distribution is known |
| a trained Fidelity model (gradient boosting) | marginal | +0.3% over the lookup tables in simulation |
| Mode B: commit drafts without verification | **cannot skip target compute here** | every committed token must pass through the target to advance its KV and its 48-layer recurrent state, and that pass is the verify |
| lenient acceptance (the realistic Mode B, approximate) | open; needs a quality evaluation | simulated +3.9% at 1.9% divergent tokens, +5.5% at 2.7%, +9.2% at 4.5%; never under a grammar (tools/JSON schema) |
| deeper drafts / free-running MTP | no gain at today's verify cost | acceptance on the MTP's own prefix doesn't drift (0.68-0.77 on an accepted prefix out to depth 8), but survival decays geometrically and verify width above 5 costs more than it returns |

Upper bound: an oracle that knew every draft's acceptance and picked the best depth per cycle gains +17% (simulated).
The measured rule recovers about a quarter of that.

## What was done

1. **Phase 0, code audit** ([RECONNAISSANCE.md](RECONNAISSANCE.md)).
   - Drafting deeper than 1 is already an MTP-only rollout: the MTP layer feeds back its own hidden state, token and KV.
   - `--spec-draft-p-min` only stops drafting; it is compared with the draft's own top-1 probability.
   - There is no per-request control of draft length.
   - n-max 16 aborts at warm-up (`GGML_ASSERT(obj_new)`, ggml.c:1805, a graph-metadata limit; not investigated).
2. **Phase 1, fixed-policy baselines** ([BASELINES.md](BASELINES.md)): 15 runs over a 72-prompt, 8-workload suite
   (casual, creative, code, reasoning, JSON, task switching, tool calls, ~7k-token long context; dev / val / test).
   - The production policy sits on a broad plateau: n-max 2-4 and p-min 0-0.6 are all within ~1%.
   - n-max 1 is -12%, 6 is -18%, 8 is -31%. Greedy drafts are equal. No MTP is 30.2 t/s (MTP is 1.65x).
   - The "Discord reference" config, adapted, runs 49.0 t/s vs production's 51.5 under the same sampling (the bf16
     exchange wire); acceptance is identical.
   - Per category, n-max 2 beats 4 on creative and casual and loses on code, JSON and tools.
3. **Trace instrumentation** ([TRACE_SCHEMA.md](TRACE_SCHEMA.md), commit `6928a6854`): `LLAMA_SPEC_TRACE=<path>`
   writes one JSON line per verify cycle.
   - Each line holds the draft's per-step distribution and the target's distribution at every drafted position,
     teacher-forced along the drafted prefix, past the rejection too. That is the label the Fidelity head needs.
   - Same-seed output with the trace on and off is byte-identical.
   - Traces were collected at n-max 4 and 8 with p-min 0: 5,117 and 4,476 cycles.
4. **Offline analysis** ([ANALYSIS.md](ANALYSIS.md), scripts in [analysis/](analysis/)).
   - A cost model fitted on the traces reproduces the measured p-min sweep within 0.4% and the n-max sweep's shape
     within ±2.5%.
   - Policies were replayed with dev-fit / val-tuned / test-reported splits and a paired bootstrap over prompts.
   - The analysis found that a stop rule must not look at the drafted token's own q, or the p/q rule stops being
     exact.
5. **Phase 4, measured A/B** (commit `8d0d6b6dd`): production vs the best fixed p-min vs the lookup rule, two rounds
   in alternating order, all 72 prompts.

   | arm | all prompts | held-out test |
   |---|---|---|
   | production (p-min 0.2) | 49.30 t/s | 49.38 |
   | best fixed (p-min 0.5) | 49.46 | 49.31 |
   | lookup rule (`LLAMA_SPEC_STOP_ISO=0.125`) | **51.61** | **53.32** |

   - Each arm reproduced its exact tokens on the second round, so these differences are not run-to-run noise.
   - The rule drafts 28% fewer tokens at acceptance 0.68 (production 0.51). The gain comes from not verifying padded
     columns on cycles whose late drafts would fail: each inactive column saves ~4 ms of the 5-wide verify.

## Using it

- `LLAMA_SPEC_STOP_ISO=0.125` on the server enables the rule; unset, behaviour is unchanged.
- The tables were fit on thinking-off sampling (0.7/0.8/20/0, presence 1.5) with sampled drafts at T 1.0, the
  production setup. Other sampling regimes (thinking 1.0/0.95, greedy drafts) weren't measured.
- `LLAMA_SPEC_TRACE=<path>` collects traces for refitting. `scripts/run_suite.py` runs the suite, and
  `analysis/run_all.sh` rebuilds every analysis table (CPU, ~2.5 min).

## Limits

- Small suite: 72 prompts, one seed, up to 300 tokens, contexts up to ~8.5k tokens. Long contexts (64k+) and
  thinking mode weren't covered.
- The held-out +8.0% rests on 24 prompts; the all-prompt +4.7% is the safer number.
- Lenient acceptance has a simulated speed/divergence frontier only. Real quality needs generation and task scoring.

## Files

| file | contents |
|---|---|
| [PLAN.md](PLAN.md) | the owner's research plan |
| [RECONNAISSANCE.md](RECONNAISSANCE.md) | Phase 0: data flow, controls, signals, cost model, feasibility per intervention |
| [BASELINES.md](BASELINES.md) | Phase 1 fixed-policy sweep |
| [TRACE_SCHEMA.md](TRACE_SCHEMA.md) | trace format |
| [ANALYSIS.md](ANALYSIS.md) | offline analysis: depth curve, cost model, policies, heads, lenient acceptance, task type |
| `analysis/` | the analysis scripts |
| `scripts/` | the suite runner and the test-window scripts (they stop and restore the box's production server) |
| `suite/prompts.json` | the 72-prompt suite with categories and splits |
| `slots/` | the long-context slot library's tooling: saved 16k-250k contexts that restore in 2-20 s |
