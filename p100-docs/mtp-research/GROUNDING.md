# Grounding the stop rule: thinking on, long context, and tables per thinking mode (2026-10-24 evening)

The rule of [README.md](README.md) was fit and measured on thinking-off chat at contexts of 8.5k tokens or less.
These four test windows measured it where it hadn't been, and fixed what they found. Production (the default branch
`p100-optimizations-keoian`) runs the result: CHANGES §22 there, commits `c44158c77`, `a6c045269`, `504d4d353`,
`e0b402541`, and `f5fbd2542` (the chat tables at every other temperature, [CF_ANALYSIS.md](CF_ANALYSIS.md)).

## Exactness, checked in code first

- **Per-token rule with a reasoning budget** (`reasoning_effort` low/medium adds a budget sampler, which rules out
  block verification). The stop decision for step d uses only step d's draft distribution and the tokens drafted
  before it, so the draft length is a stopping time and the p/q rule stays exact. Forced end-of-think tokens are a
  one-hot target, handled for any draft length.
- **Block verification** (thinking with no effort level). A block cut at d verifies like a full-length block whose
  draft emits, after d, a token of target probability 0. Its keep probability is 0, h at the cut is keep_d, and the
  residual there is the target. That is what `common_spec_block_verify` computes for any draft length.

## Window 9: thinking on (test split, 24 prompts, card thinking sampling 1.0 / 0.95 / 20 / 0)

| path | production | rule | gain (95% over prompts) |
|---|---|---|---|
| medium effort (budget sampler, per-token rule), cap 3000 | 49.11 t/s | 48.70 | -0.8% (-4.7..+1.9) |
| no effort level (block verify), cap 1500 | 48.59 | 49.49 | +1.9% (-1.1..+5.6) |

Per cycle (medium) the rule drafted 2.40 tokens instead of 3.90 and lost 0.32 accepted ones (thinking off: 2.70
instead of 3.84, 0.11 lost). Production's acceptance is the same in both modes (0.50-0.51). So the tables, fit at
target T 0.7 / top-p 0.8 / presence 1.5, underestimate acceptance at T 1.0 / top-p 0.95 and stop too early.

## Window 10: long context (thinking off, production sampling, 16 questions per depth)

Questions about the saved slot library's documents (`slots/questions.json`); the slot was restored before every
question; two rounds, even and odd questions, in alternating order.

| depth | production | rule | gain (95%) |
|---|---|---|---|
| 32k | 49.28 t/s | 52.54 | +6.6% (+3.0..+10.6) |
| 64k | 46.54 | 49.45 | +6.2% (+0.2..+12.4) |
| 128k | 40.07 | 41.17 | +2.8% (-0.6..+6.1) |
| 200k | 35.20 | 36.32 | +3.2% (-1.2..+7.7) |
| all 64 | 41.83 | 43.84 | **+4.8% (+2.1..+7.4)** |

Side result: a follow-up question without a restore reprocessed only itself (37-43 tokens, 0.6-1.0 s, through a
context checkpoint at the document's end) and gave the identical reply at every depth. Restores between questions
are unnecessary.

## The temperature gate (commit `30ad85064` here, `c44158c77` on the default branch)

The server passes each request's target temperature to the draft (`common_speculative_draft_params::target_temp`).
The rule applies only in `LLAMA_SPEC_STOP_ISO_TEMP` (default 0.5,0.9 on this branch; the default branch's is 0.001,1e9
since f5fbd2542). Window 11 checked it with the rule on:
- thinking-on replies equal production's, 48 of 48 (token and accept counts per request; text apart from random
  tool-call ids);
- thinking-off replies equal the ungated rule's, 72 of 72.

## Thinking tables (window 11 traces, window 12 A/B)

- **Traces:** thinking on, medium effort, card sampling, p-min 0, n-max 4 and 8, all 72 prompts: 18,530 and 17,845
  cycles (as loaded for the analysis; the trace files hold 18,532 and 17,846 records, the warm-up's among them).
- **Fit:** `analysis/run_think.sh` (`MTP3_TRACESET=think`; the default path still reproduces every output
  byte for byte).
- **Tables:** acceptance at a given draft probability is higher than the chat tables say: 0.251 vs 0.115 at q 0.02,
  0.458 vs 0.321 at q 0.45.
- **Simulated** on held-out prompts: +3.4% at theta 0.15. n-max 4 stays best.

| window 12 (vs production with the chat rule, thinking on p-min) | production | thinking tables 0.15 | gain (95%) |
|---|---|---|---|
| medium effort, all 72 prompts | 50.09 t/s | 51.57 | **+2.9% (+1.4..+4.6)** |
| medium effort, held-out 24 | 48.20 | 50.35 | +4.5% (+1.4..+6.9) |
| no effort (block verify), held-out 24 | 48.41 | 49.67 | +2.6% (-0.6..+5.6) |

On the default branch this is `LLAMA_SPEC_STOP_ISO_THINK=<theta>` with `LLAMA_SPEC_STOP_ISO_THINK_TEMP` (default
0.95,1.05), commit `504d4d353`. This research branch keeps the trace hooks, the chat tables and the gate; it doesn't
carry the thinking tables.

## Testing notes

- **Arms are deterministic.** Production arms in windows 9, 11 and 12 reproduced each other's token counts
  exactly.
- **The RAM prompt cache breaks bit-identity across modes.** A thinking-on request right after the thinking-off
  request for the same prompt reuses its prefix state (`prompt_n` 2). That state was computed in a different
  batch, so the text can differ. Comparisons across runs should use `CACHE_PROMPT=0` (`scripts/run_suite.py`).
- **Thinking replies were shorter than planned:** 100-2,300 tokens, ~17k tokens per 24-prompt arm. Intervals over
  24 prompts are about ±3%; window 12 used all 72 prompts for medium effort.
