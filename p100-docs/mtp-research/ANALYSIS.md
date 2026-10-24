# mtp3 offline analysis (Phases 1-3): can cheap signals beat the fixed MTP policy?

2026-10-24. CPU-only analysis (numpy / scipy / scikit-learn) of the window-6 traces. Nothing ran on the GPUs.
Scripts are in `analysis/`; `bash analysis/run_all.sh` rebuilds every table in about 2.5 min on one core. Raw outputs are in `analysis/out/`.
**Everything in §4-§7 is a SIMULATION.** Each policy is replayed on recorded verify cycles, with a cost model fitted to the traces and checked against BASELINES. None of it is a measured end-to-end result.

## 0. Bottom line

**Headline number.** The best realistic policy is the one chosen on val among the label-free policies that keep the output distribution exact. On the 24 held-out test prompts it gains **+5.7% simulated throughput over production (n-max 4, p-min 0.2), 95% CI [+4.2, +7.0]**. That policy is an exact-safe Fidelity stop rule, n-max 4, theta 0.225. The CI comes from a paired bootstrap over prompts.
- Against the best fixed policy (n-max 4 with p-min raised to 0.5) the gain is +3.5% [+2.7, +4.3].
- On the second state set (trace-n8p0) the same pipeline gives +5.1% [+3.2, +6.7].
- The replay overestimated the two measured early-stopping runs (p-min 0.4 and 0.6) by 1.7-2.5%. A measured A/B should therefore expect roughly **+3 to +5%**.
- Two 1-D lookup tables plus a threshold get almost all of the gain: +5.4% [+4.0, +6.5]. A trained model adds only +0.3% [0.0, +0.6] on top of them.

**Oracle bound.** The oracle knows every true a_d and picks the best depth for each cycle at the same width (n-max 4, W=5). It reaches **+17.0% [+13.8, +19.5]** over production, or +14.9% if it also pays for the look-ahead draft step that any real stop rule pays. At n-max 5, 6, 7 and 8 the oracle gets less: +9.4%, +3.2%, -2.5% and -27.5%. The cause is the verify width: each width step above 5 costs more than perfect stopping can recover.

| head | verdict | evidence (test prompts, simulated unless noted) |
|---|---|---|
| Fidelity (stop rule from predicted survival of the drafted prefix) | **Build it, in its cheap form.** Use two 1-D lookups and a threshold; it must not look at the sampled token's own q (§4.2) | Exact lookup rule: +5.4% [+4.0, +6.5] over production and +3.2% [+2.4, +4.0] over the best fixed. HGB: +0.3% [0.0, +0.6] more. AUC 0.86 for a_d > 0.5 |
| Difficulty (previous cycle's teacher entropy and top-1 p) | **No** | It does predict the next cycle (Spearman -0.22 with a_1), but adds nothing beyond the draft's own distribution: Brier unchanged, throughput +0.1% [-0.1, +0.2] |
| Task-Type, oracle (true category labels) | **No** | Inside the Fidelity rule: +0.0% [-0.0, +0.1]. Alone, per-category p-min: +0.2% over best fixed. Per-category n-max per request: +1.3% [-0.2, +2.9] over best fixed, and it loses to the Fidelity rule |
| Task-Type, realistic proxy (code fence, `{`/`[` start, tool markup, rolling acceptance) | **No** | Proxy p-min: +0.1% over best fixed. Proxy x lookup rule: -0.1% vs the lookup rule alone. Rolling acceptance: -0.5% vs best fixed |

**Lenient acceptance** (approximate, Mode B), at production settings:

| rule | throughput gain | divergent committed tokens |
|---|---|---|
| lambda 0.5 | +3.9% | 1.9% |
| lambda 0.3 | +5.5% | 2.7% |
| lambda 0.1 | +9.2% | 4.5% |
| tau 0.05 | +10.0% | 4.9% |

It stacks with the stop rule. n-max 4 stays the best depth even under leniency.

**Free-running depth.**
- Mean acceptance on the MTP's own prefix, E[a_d], falls from 0.76 at d=1 to 0.44 at d=8.
- Given an accepted prefix, the conditional acceptance stays at 0.68-0.77 and agreement at 0.67-0.76. There is no self-feeding drift.
- Survival therefore decays geometrically: 0.76, 0.52, 0.36, 0.26, 0.19, 0.15, 0.11, 0.08.
- Depth past 4 never pays at today's verify cost. JSON and tools survive longest (S_8 0.43-0.45), creative writing shortest (0.003).

## 1. Data, and the trace schema checked against the records

| file | records used | requests | note |
|---|---|---|---|
| trace-n4p0 + T-n4p0 | 5,115 cycles | 72 (dev/val/test 24 each) | n-max 4, p-min 0 |
| trace-n8p0 + T-n8p0 | 4,474 cycles | 72 | n-max 8, p-min 0 |
| trace-I + I-on | 1,416 cycles | 21 (dev, sweep categories) | n-max 4, p-min 0.2: real p-min stops, so the verify runs at 1-5 active columns |

`TRACE_SCHEMA.md` matches the records field for field (its "not built or run yet" header is stale), apart from these findings:
- **Task 0** is the server warm-up request. It is absent from the T files and was dropped.
- **`emitted` never contains a reply's first token**, the one sampled from the prompt logits. The task-type proxies rebuild it from the T row's text.
- **Cut cycles:**
  - The end-of-request cycles (37 in n4p0, 76 in n8p0) decoded all n-max draft steps.
  - Their verify was **not padded**: `n_draft_padded == n_real < n-max`. That means a narrower graph and verify times 6-13 ms higher.
  - The simulation ignores this end-of-request effect.
- **Grammar:**
  - `grammar` is true on every cycle of a tool request (111 and 76 cycles).
  - The teacher is shorter only there (20 and 24 cycles). Those positions get a = 0.
  - The server uses the plain compare, P = p_draft, at grammar positions. The analysis uses min(1, p/q); the difference is 0.01 tokens per cycle on those cycles.
- **Draft `q` and `top`:** `q` equals the top-5 p of the sampled id exactly whenever that id is in the top 5. 4.4% of sampled ids lie outside the top 5 but inside the top 20.
- **Teacher argmax:** the raw argmax differs from the post-chain top-1 in 5.9% of rows, because of the presence penalty. Greedy agreement uses the raw argmax.
- **Consistency of a_d = min(1, p_draft/q):**
  - Expected accepted drafts per cycle vs realized `n_accepted`, on non-grammar cycles: 1.956 vs 1.948 (n4p0), 2.359 vs 2.337 (n8p0), 1.998 vs 1.981 (I).
  - On all cycles the differences are -0.008 ± 0.007 and -0.024 ± 0.011. Depth by depth they agree within 0.01.
- **Overhead:** the trace adds `trace_us` (1.2 ms per cycle at n-max 4, 2.1 ms at n-max 8). `predicted_ms` equals sum(cycle_us + gap_us) plus 9.4 ms per request. Costs exclude `trace_us`.
- **The two traces are different texts.** The n4p0 dev+val subset without longctx is token-identical to the measured B4-p0.0 run.

## 2. Q1: free-running MTP quality vs depth (n8p0, 4,398 cycles with all 8 drafted)

Definitions: a_d is the p/q acceptance probability of drafted token d, with the teacher forced along the drafted prefix, past the real rejection too. S_d = prod_{i<=d} a_i.

| depth d | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| E[a_d] | .761 | .647 | .583 | .551 | .510 | .499 | .472 | .440 |
| P(acc d \| acc 1..d-1) = E[S_d]/E[S_d-1] | .761 | .683 | .684 | .724 | .741 | .762 | .747 | .770 |
| survival E[S_d] | .761 | .519 | .355 | .257 | .191 | .145 | .109 | .084 |
| greedy agreement (sampled == teacher raw argmax) | .726 | .628 | .575 | .546 | .503 | .499 | .477 | .437 |
| draft top-1 == teacher argmax | .783 | .681 | .625 | .569 | .538 | .526 | .507 | .470 |
| agreement given 1..d-1 all agreed | .726 | .666 | .690 | .737 | .751 | .759 | .757 | .758 |
| agreement given an earlier disagreement | - | .528 | .467 | .450 | .422 | .440 | .431 | .398 |
| E[a_d] given every earlier a_i >= 0.99 | .761 | .696 | .715 | .754 | .769 | .782 | .784 | .807 |
| share with a_d = 0 / a_d = 1 | .16/.66 | .28/.57 | .34/.51 | .37/.48 | .41/.44 | .41/.42 | .44/.39 | .47/.36 |

trace-n4p0 reproduces depths 1-4: E[a_d] .774 / .675 / .596 / .572.

**Where it falls off.** The steep part is d=1 -> 3. a_d is bimodal at every depth (mostly 0 or 1).

**Drift.** On a prefix the teacher accepts, conditional acceptance and agreement do not fall with depth; they rise a little, because easy spans are selected. The decline of E[a_d] comes entirely from cycles where an earlier drafted token already left the target's distribution; there, later tokens are accepted at only 0.41-0.55. So there is no measurable self-feeding drift out to depth 8. What limits depth is the compounding of a per-step acceptance of about 0.7-0.75.

Per category (n8p0; E[a_d], then expected accepted drafts at n-max 8 = sum S, and S_8):

| category | a1 | a2 | a4 | a8 | sum S | S_8 |
|---|---|---|---|---|---|---|
| json | .94 | .90 | .85 | .77 | 5.46 | .433 |
| tools | .86 | .84 | .70 | .66 | 5.13 | .447 |
| code | .89 | .82 | .74 | .58 | 3.86 | .184 |
| reasoning | .83 | .75 | .67 | .53 | 3.17 | .132 |
| longctx | .76 | .67 | .58 | .44 | 2.38 | .058 |
| switch | .77 | .63 | .48 | .39 | 2.21 | .066 |
| casual | .72 | .61 | .54 | .45 | 1.96 | .029 |
| creative | .66 | .50 | .38 | .29 | 1.25 | .003 |

By position in the reply, sum S is 2.61 / 2.46 / 2.28 for tokens 0-50 / 50-150 / 150+.

## 3. Q2: cost model, and the replay of BASELINES

The cycle time is `T = d0 + d1*steps + verify(W, n_active, ctx) + other(W)`, production with no trace:

| term | value | source |
|---|---|---|
| draft | 0.86 ms + 1.11 ms per decoded step (+6 µs/step per 1k ctx) | traces I, n4p0, n8p0 |
| verify, W=5 | 54.4 ms, **-3.97 ms per inactive column** (bootstrap CI 3.67-4.16), +0.30 ms per 1k ctx | trace-I (37.1 / 44.0 / 46.4 / 51.1 / 54.0 ms at 1-5 active columns) + n4p0 |
| verify, W=9 | 89.5 ms + 1.16 ms per 1k ctx; no column skipping at W > 8 | n8p0 |
| verify, W = 2 / 3 / 4 / 7 | 38.8 / 43.7 / 49.8 / 72.2 ms | calibrated on B3 end to end |
| verify, W = 6 / 8 | 63.3 / 80.8 ms | interpolated |
| other (pad + catch-up + accept + gap) | 1.16 ms at W5, 1.37 ms at W9 | traces |
| per request | 9.4 ms | |

For reference, plain decode (B0) costs 33.2 ms per token.

**Independent check.** The B4 p-min sweep was not fitted. With its measured drafted tokens per cycle, the model gives its measured ms per cycle within 0.4%:

| run | measured | model |
|---|---|---|
| p-min 0.0 | 61.20 | 60.95 |
| p-min 0.1 | 60.84 | 60.95 |
| p-min 0.2 (B1r) | 60.56 | 60.48 |
| p-min 0.4 | 56.97 | 56.91 |
| p-min 0.6 | 52.88 | 52.82 |
| greedy drafts | 60.46 | 60.42 |

B1, the same text as B1r, measured 59.60: run-to-run noise is about 1.6%.

**Replay of BASELINES** (dev+val, no longctx; sim on the trace states):

| run | n-max | p-min | measured | sim, n4p0 states | sim, n8p0 states | tokens per cycle, measured / sim (n4p0) |
|---|---|---|---|---|---|---|
| B3-n1 | 1 | .2 | 43.42 | 42.89 (-1.2%) | 42.53 (-2.1%) | 1.815 / 1.774 |
| B3-n2 | 2 | .2 | 49.60 | 49.23 (-0.7%) | 48.42 (-2.4%) | 2.375 / 2.324 |
| B3-n3 | 3 | .2 | 49.65 | 49.66 (0.0%) | 48.60 (-2.1%) | 2.735 / 2.707 |
| B1r | 4 | .2 | 49.48 | 49.43 (-0.1%) | 48.30 (-2.4%) | 2.996 / 2.992 |
| B1 | 4 | .2 | 50.28 | 49.43 (-1.7%) | 48.30 (-3.9%) | |
| B4-p0.0 | 4 | 0 | 48.94 | 49.21 (+0.5%) | 48.09 (-1.7%) | 2.995 / 3.000 |
| B4-p0.4 | 4 | .4 | 49.96 | 50.83 (+1.7%) | 49.32 (-1.3%) | 2.846 / 2.928 |
| B4-p0.6 | 4 | .6 | 49.30 | 50.51 (+2.5%) | 49.30 (0.0%) | 2.607 / 2.732 |
| B3-n6 | 6 | .2 | 40.75 | - | 40.17 (-1.4%) | 3.252 / 3.159 (n8p0) |
| B3-n8 | 8 | .2 | 33.91 | - | 33.35 (-1.7%) | 3.389 / 3.360 (n8p0) |

**How well the replay matches.**
- The shape is reproduced: the n-max 2-4 plateau at about 49-50, n-max 1 at about 43, n-max 6 at about 40, n-max 8 at about 33.
- n4p0 states are within ±2.5%. n8p0 states read 0-2.4% low, because an n-max 8 run starts more cycles after rejections, at harder positions.
- In relative terms the two state sets agree within about 0.7 points.
- **Known bias:** early-stopping replays (p-min 0.4 and 0.6) are 1.7-2.5% optimistic. Real runs stopped more often than the replay (2.26 vs 2.39 drafted tokens per cycle at p-min 0.6), because low-confidence stops cluster. The adaptive rules stop about as often as p-min 0.5.
- A position-weighted (renewal) estimator brackets this: it replays p-min 0.6 2.5% low and n-max 1-3 2% high, and gives the same headline gains (§4.4).

Fixed depth k (p-min 0, n8p0 states), simulated t/s:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| t/s | 42.6 | 48.4 | 48.5 | 48.1 | 43.6 | 39.7 | 36.4 | 33.4 |

The drop at W=6 is the verify leaving the q4p and exact-fp16 kernels. At W=9 the padded verify also stops skipping inactive columns.

## 4. Q3: simulated policies on held-out prompts

### 4.1 Method
- **Splits.** The heads are fit on dev cycles of both traces. Thresholds, n-max and per-group settings are chosen on val. Everything is reported on test: 24 prompts, 1,700 cycles (n4p0) and 1,481 (n8p0).
- **Simulation.**
  - "Look" rules decode step d, then decide whether to add token d, as p-min does today.
  - The verify width is fixed at n-max+1, with n_active = k+1.
  - Expected tokens: E = 1 + sum_{j<=k} S_j. Time comes from the cost model.
  - Throughput is computed per prompt as tokens / (tokens × sum T / sum E + 9.4 ms).
- **Uncertainty.** A paired bootstrap over test prompts (4,000 resamples) gives the CIs. They cover prompt sampling only, not the model error in §3.
- **Baselines.**
  - production = n-max 4, p-min 0.2;
  - "best fixed" = the best of n-max 1-8 × p-min 0-0.7 chosen on val: n-max 4, p-min 0.5 in both state sets, which is also the hindsight best on test.
  - The best measured fixed policies (B4-p0.4 at 49.96, B3-n3 at 49.65) are within noise of production.

### 4.2 Exactness constraint
With sampled drafts and the per-token p/q rule, the output distribution stays the target's only if the decision to add drafted token d depends on two things alone:
- the step-d draft **distribution** (top-K p's, entropy);
- the earlier tokens, which are already in the prefix whenever position d matters.

A rule that inspects the sampled token's own q (q-min, prod q through d, a model that takes q_d or "sampled == top-1") biases the output.
- Monte Carlo, 400k draws, with q = (0.9, 0.1), p = (0.5, 0.5), skipping the token when q < 0.2: the output is (0.549, 0.451).
- So the realistic heads use exact-safe inputs only at step d. The survival estimate is S_hat_d = prod_{i<d} post(i) × prior(d).
- The token-aware versions, which are not exact, are listed for reference. They are 0.3-0.7% faster.

Production's p-min rule is exact-safe.

### 4.3 Results on test (SIMULATED)

**Production and fixed policies.** On n4p0 states production simulates at 48.76 t/s; on n8p0 states at 49.25.

| policy | t/s (n4p0) | vs production | vs best fixed | n8p0: vs production | chosen on val |
|---|---|---|---|---|---|
| best fixed: n-max 4, p-min 0.5 | 49.79 | +2.1% [+0.9, +3.2] | 0 | +2.2% [+1.0, +3.5] | |
| p-min 0.4 / 0.6 | 49.77 / 49.34 | +2.1 / +1.2% | -0.0 / -0.9% | +2.0 / +1.2% | |
| fixed depth 1/2/3/4 | 42.65 / 48.87 / 49.11 / 48.54 | -12.5 / +0.2 / +0.7 / -0.5% | | -12.7 / -0.2 / +0.6 / -0.4% | |
| fixed depth 5/6/7/8 | | | | -10.1 / -18.1 / -25.2 / -31.6% | |
| p-min rule n-max 8, p-min 0..0.6 | | | | -31.6 .. -38.2% | |

**Cheap exact-safe heuristics.**

| policy | t/s (n4p0) | vs production | vs best fixed | n8p0: vs production | chosen on val |
|---|---|---|---|---|---|
| p_cum = prod top-1 p | 50.62 | +3.8% [+2.5, +4.9] | +1.7% [+1.0, +2.4] | +3.8% [+2.3, +5.1] | theta 0.4 |
| **lookup rule: prod_{i<d} iso_q(q_i) × iso_p1(p1_d)** | **51.38** | **+5.4% [+4.0, +6.5]** | **+3.2% [+2.4, +4.0]** | +4.7% [+2.8, +6.4] | theta 0.125 |

**Learned Fidelity heads (exact-safe).**

| policy | t/s (n4p0) | vs production | vs best fixed | n8p0: vs production | chosen on val |
|---|---|---|---|---|---|
| LR, draft features | 51.50 | +5.6% [+4.1, +6.8] | +3.4% | +5.3% [+3.4, +6.9] | theta 0.2 |
| HGB, draft features | 51.55 | +5.7% [+4.2, +7.0] | +3.5% | +5.0% [+3.2, +6.5] | theta 0.175 |
| HGB realistic (draft + rolling + proxy) | 51.51 | +5.6% [+4.1, +6.9] | +3.5% [+2.6, +4.3] | +5.2% [+3.4, +6.8] | theta 0.2 |
| **HGB realistic + Difficulty (headline: best on val)** | **51.55** | **+5.7% [+4.2, +7.0]** | **+3.5% [+2.7, +4.3]** | **+5.1% [+3.2, +6.7]** | theta 0.225 |

**Task-type proxies (realistic) and oracle category heads.**

| policy | t/s (n4p0) | vs production | vs best fixed | n8p0: vs production | chosen on val |
|---|---|---|---|---|---|
| rolling acceptance: p-min/cap by EMA | 49.52 | +1.6% [+0.3, +2.7] | -0.5% | +2.1% | |
| PROXY p-min/cap (structured vs prose) | 49.83 | +2.2% [+1.1, +3.2] | +0.1% | +2.2% | |
| PROXY × lookup rule | 51.35 | +5.3% [+3.9, +6.4] | +3.1% | +4.7% | |
| ORACLE category p-min/cap, common width | 49.86 | +2.3% [+0.9, +3.5] | +0.2% [-0.4, +0.9] | +2.6% | creative capped at 2 |
| ORACLE category (n-max, p-min) per request, +20 ms rebuild | 50.42 | +3.4% [+1.0, +5.7] | +1.3% [-0.2, +2.9] | +3.3% [+0.7, +5.9] | |
| ORACLE category inside the HGB head | 51.54 | +5.7% [+4.2, +6.9] | +3.5% | +5.2% | |
| ORACLE category × HGB head, per-request width | 51.45 | +5.5% [+3.3, +7.5] | +3.3% | +5.0% | |

**Not exact (they look at the sampled token's own q).** Reference only.

| policy | t/s (n4p0) | vs production | n8p0: vs production |
|---|---|---|---|
| q-min on the sampled token | 50.74 | +4.1% | +4.1% |
| prod iso(q) through d | 51.74 | +6.1% | +5.3% |
| HGB token-aware chain | 51.89 | +6.4% [+4.8, +7.7] | +5.5% |

**Oracles.**

| policy | t/s (n4p0) | vs production | vs best fixed | n8p0: vs production |
|---|---|---|---|---|
| per-cycle oracle, n-max 2 / 3 | 51.85 / 54.62 | +6.3 / +12.0% | | +5.6 / +11.5% |
| **per-cycle oracle, n-max 4** | **57.05** | **+17.0% [+13.8, +19.5]** | **+14.6% [+12.1, +16.5]** | +16.7% [+13.5, +19.3] |
| same, paying the look-ahead step | 56.02 | +14.9% [+12.1, +17.1] | +12.5% | +14.6% |
| per-cycle oracle, n-max 5 / 6 / 7 / 8 | | | | +9.4 / +3.2 / -2.5 / -27.5% |

All val-chosen adaptive policies picked n-max 4 even when n-max 5-8 was allowed. The oracle sees each token's realized a, so it bounds every stop rule, exact or not.

**Paired differences on test:**

| comparison | n4p0 | n8p0 |
|---|---|---|
| lookup rule vs p_cum | +1.5% [+1.0, +2.0] | +0.8% [+0.3, +1.4] |
| HGB realistic vs lookup rule | +0.3% [-0.0, +0.5] | +0.5% [+0.2, +0.8] |
| LR vs lookup rule | +0.2% [-0.1, +0.5] | +0.6% [+0.1, +1.0] |
| + Difficulty vs HGB realistic | +0.1% [-0.1, +0.2] | -0.1% [-0.3, +0.1] |
| + category vs HGB realistic | +0.0% [-0.0, +0.1] | +0.0% [-0.1, +0.1] |
| proxy × lookup rule vs lookup rule | -0.1% [-0.2, +0.1] | |
| ORACLE category × head vs head | -0.1% [-1.1, +0.9] | |
| token-aware HGB (not exact) vs exact HGB | +0.7% [+0.5, +1.0] | +0.3% [-0.1, +0.6] |

**What the head does** (HGB realistic, theta 0.2, test, vs production):

| quantity | head | production |
|---|---|---|
| drafted tokens kept per cycle | 2.70 | 3.91 |
| expected tokens per cycle | 2.867 | 2.972 (-3.5%) |
| cycle time | 55.4 ms | 60.9 ms (-9.0%) |

- Depth histogram: k = 1/2/3/4 in 21 / 26 / 16 / 38% of cycles. Production keeps 4 in 95%.
- The gain is fewer padded columns verified on cycles that will fail, not more tokens.
- By category: creative +8.9%, casual +7.8%, switch +6.1%, longctx +5.2%, reasoning +3.8%, tools +2.5%, code +1.9%, json +0.2%.

### 4.4 Sensitivity (n4p0, thresholds re-tuned on val each time)

| cost model | best fixed | lookup rule | HGB head | oracle n4 |
|---|---|---|---|---|
| as fitted | +2.1% | +5.4% | +5.6% | +17.0% |
| column saving s × 0.75 / × 1.25 | +0.9 / +4.3% | +3.4 / +8.3% | +3.4 / +8.3% | +13.1 / +21.3% |
| s × 0.5 | +1.1% | +1.9% | +1.9% | +9.6% |
| s = 0 | +1.3% | +1.3% | +1.3% | +3.5% |
| draft step × 0.75 / × 1.25 | +1.9 / +2.3% | +5.2 / +5.5% | +5.5 / +5.8% | +16.2 / +17.8% |
| every V(W) ±10% | +1.6 / +3.1% | +4.6 / +6.3% | +4.7 / +6.7% | +15.3 / +19.1% |
| position-weighted estimator | +3.6% (n-max 2) | +5.5% [+4.0, +6.8] | +5.8% [+4.1, +7.3] | +16.5% |

- The gain lives in the padded verify's column skipping. Its fitted value, 3.97 ms per inactive column, is tight and validated on the B4 sweep.
- Without column skipping, stop rules are worth nothing beyond a fixed depth.
- The choice of estimator does not matter.

## 5. Q3d: Fidelity predictor (test, 18,494 (cycle, depth) pairs)

| predictor of a_d | AUC (a_d > 0.5) | Brier | soft log-loss | Brier vs isotonic(q) [CI] |
|---|---|---|---|---|
| draft q, raw | .847 | .1602 | .661 | +.029 [+.022, +.034] |
| draft top-1 p, raw | .833 | .1843 | .717 | +.053 |
| isotonic(q) | .847 | .1315 | .473 | 0 |
| exact-safe isotonic(top-1 p) | .833 | .1380 | .491 | +.007 |
| exact-safe HGB prior | .858 | .1266 | .459 | -.005 |
| exact-safe HGB prior + rolling + proxy | .859 | .1259 | .457 | -.006 [-.008, -.004] |
| exact-safe, + Difficulty / + category | .859 / .859 | .1260 / .1260 | | -.006 |
| LR, token-aware | .862 | .1254 | .464 | -.006 |
| HGB, token-aware | .870 | .1199 | .443 | -.012 |
| HGB, token-aware realistic | .871 | .1194 | .442 | -.012 [-.014, -.011] |
| same, + Difficulty / + category | .871 / .870 | .1194 / .1197 | | -.012 |

**Reliability.** The raw q is badly calibrated as an acceptance probability:
- over-confident in the middle and top: q in [0.8, 0.9) gives mean a 0.55, and [0.9, 1] gives 0.86 at a mean q of 0.985;
- under-confident at the bottom: q < 0.1 gives 0.13.
- The isotonic map and the HGB are calibrated within ±0.02 in every decile. Full tables are in `out/fidelity.txt`, and the rule's two lookups are in `out/iso_tables.txt`.

**Ranking.** AUC falls with depth, for example the token-aware HGB from 0.889 at d=1 to 0.84-0.86 at d=4-8. The model beats q by 0.01-0.02 at every depth; the largest gain is in tools (0.92 -> 0.98) and creative (0.79 -> 0.82). Most of the power is the draft's own q or top-1 p; the model adds calibration across depth and cumulative context.

## 6. Q3e: does the previous cycle predict the next? (Difficulty)

Spearman correlations with the next cycle's a_1 and with S_1+..+S_4:

| previous-cycle signal | n4p0 (a_1 / S_1..4) | n8p0 (a_1 / S_1..4) |
|---|---|---|
| teacher entropy at the last emitted row | -0.22 / -0.33 | -0.19 / -0.29 |
| that row's top-1 p | +0.21 / +0.31 | +0.18 / +0.27 |
| teacher row-0 entropy | -0.12 / -0.19 | |
| EMA of first-token acceptance | +0.16 / +0.28 | +0.13 / +0.23 |

For comparison, this cycle's draft top-1 p at step 1 has Spearman +0.60 with a_1.

So difficulty persists across cycles, but it is redundant once the draft distribution is known:
- per token: Brier .1199 -> .1198 (token-aware), .1259 -> .1260 (exact-safe);
- throughput: +0.1% [-0.1, +0.2] (n4p0), -0.1% (n8p0).

The draft step is cheap (1.1 ms) and its distribution is the better difficulty signal.

## 7. Q4: lenient acceptance (APPROXIMATE)

The rule: accept drafted token d when teacher p_draft >= lambda × top-1 p (or p_draft >= tau); otherwise use the exact test.
- It is never applied under a grammar, so tool requests are excluded.
- **Divergent tokens** = the expected committed tokens accepted where the exact test, with the same uniform, would have rejected, as a share of committed tokens.
- All 72 prompts, n4p0 states, at n-max 4 / p-min 0.2 (exact: 2.974 tokens per cycle, 48.83 t/s).

| rule | tokens per cycle | t/s | gain | divergent tokens | json | code | creative |
|---|---|---|---|---|---|---|---|
| lambda 0.9 | 3.035 | 49.89 | +2.2% | 1.1% | .2% | .8% | 1.6% |
| lambda 0.7 | 3.051 | 50.18 | +2.8% | 1.4% | .2% | .9% | 2.1% |
| lambda 0.5 | 3.081 | 50.71 | +3.9% | 1.9% | .4% | 1.2% | 2.7% |
| lambda 0.3 | 3.127 | 51.52 | +5.5% | 2.7% | .5% | 1.6% | 4.0% |
| lambda 0.2 | 3.170 | 52.28 | +7.1% | 3.5% | .8% | 1.9% | 5.1% |
| lambda 0.1 | 3.230 | 53.32 | +9.2% | 4.5% | 1.1% | 2.2% | 6.6% |
| tau .3/.2/.1/.05 | 3.073 / 3.126 / 3.203 / 3.251 | | +3.5 / +5.4 / +8.2 / +10.0% | 1.8 / 2.7 / 4.0 / 4.9% | | | |
| accept every non-grammar draft (bound) | 4.869 | 79.86 | +63.6% | 26.6% | | | |

- **Stacking.** On top of p-min 0.5, lambda 0.3 adds +4.8% at 2.5% divergent tokens.
- **Depth.** Leniency does not make depth pay. On n8p0 states with p-min 0.2 and lambda 0.1:

| n-max | 3 | 4 | 5 | 8 |
|---|---|---|---|---|
| t/s | 52.9 | 53.3 | 48.8 | 37.7 |

- These are token-level shares only; quality needs generation and task scoring.

## 8. Q5: task-type evidence

**Category beyond q and H.** Logistic regression with soft labels, trained on dev+val, scored on test:

| features | log-loss | Brier | AUC |
|---|---|---|---|
| category only | .646 | .201 | .632 |
| q / p1 / H / margin / depth | .4736 | .1281 | .858 |
| + category | .4709 | .1272 | .859 |
| exact-safe | .4905 | .1350 | .846 |
| exact-safe + category | .4879 | .1340 | .847 |

- Adding category lowers log-loss by 0.0026 [0.0004, 0.0053]. That is detectable but small, and it is calibration-in-the-large: json +.087, code +.045, reasoning +.033, creative -.026, tools -.068.
- In the stop-rule simulation it is worth +0.0%.

**Within "switch" replies** (n4p0; n8p0 in brackets):

| segment | cycles | mean a_1 | E[S_1..4] | category-blind iso(top-1 p at step 1) |
|---|---|---|---|---|
| switch, prose | 472 | .740 [.747] | 1.74 [1.65] | .57 [.59] |
| switch, inside a code fence | 95 | .938 [.952] | 2.91 [3.03] | .81 [.85] |
| code category (reference) | 641 | .887 | 2.74 | .78 |
| casual category (reference) | 824 | .751 | 1.70 | .60 |

- **Acceptance shifts strongly within a reply.** A code block inside a switch reply behaves like the code category, and the prose around it like casual. Reply thirds vary by up to 2.5x within one prompt.
- **The draft's own confidence tracks the shift.** For the per-cycle E[S_1..4] of switch cycles:

| predictor | R², n4p0 | R², n8p0 |
|---|---|---|
| segment label alone | .10 | .10 |
| draft log top-1 p over 4 steps | .41 | .33 |
| both | .42 | .36 |

## 9. Caveats

1. **Simulation, not measurement.**
   - Absolute replay error is ±2.5%.
   - Early-stopping replays read 1.7-2.5% optimistic, and the recommended rule stops about as often. Read +5.7% as about +3 to +5% measured.
   - The CIs cover prompt sampling only.
2. **Small suite.**
   - 24 test prompts, 3 per category, one seed, 300-token cap, contexts of 8.5k at most.
   - longctx test prompts are code; dev and val longctx are prose.
3. **Cost model.**
   - Verify at W 2-4 and 7 is calibrated on B3, and W 6 and 8 are interpolated.
   - The column saving is measured only at W=5. The gain needs n_active column skipping.
4. **Sampling regime.** Thinking-off sampling only (per-token rule, sampled drafts at T 1.0). Block verify, greedy drafts and thinking budgets are not covered.
5. **Off-policy features.** Rolling and Difficulty features come from the trace's own trajectory.
6. **Grammar positions** use min(1, p/q) instead of p_draft (2% of cycles).
7. **The oracle** uses the realized a; the exact-safe ceiling is lower.

## 10. Recommendation

1. **Implement the exact-safe lookup rule in place of `p < p_min`** (`common/speculative.cpp:1834`):
   - S_hat = prod_{i<d} iso_q(q_i) × iso_p1(p1_d) < 0.125 at n-max 4;
   - two 24-entry tables (`out/iso_tables.txt`), env-gated, default off; the earlier q's are already in `dp.dists`;
   - A/B it measured against production and against p-min 0.5;
   - expected +3 to +5%.
2. **Do not build** the Task-Type or Difficulty heads, or a trained Fidelity model unless the lookup version measures well.
3. **Lenient acceptance** is the only lever beyond about +5%. It is approximate and needs a quality evaluation, and grammar paths must stay exact.
4. **More depth is not useful** until wide verifies get cheaper per column above W=5.

## 11. Files (`/work/bench/mtp3/analysis/`)

| file | purpose |
|---|---|
| `run_all.sh` | reproduces everything |
| `vocab.py` | tokenizer strings from the GGUF metadata, read-only |
| `load.py` | trace and run parsing |
| `features.py` | per (cycle, depth) features; exact-safe vs token-aware sets |
| `heads.py` | soft-label LR/HGB heads, chains |
| `depth.py` | Q1 |
| `cost.py`, `calib.py` | Q2 |
| `sim.py` | simulator, oracle, bootstrap |
| `policies.py` | Q3 |
| `fidelity.py` | Q3d/e, Q5a in part |
| `sens.py` | sensitivity |
| `lenient.py` | Q4 |
| `tasktype.py` | Q5 |
| `iso_tables.py` | the rule's lookups |
| `out/*.txt` | all outputs |
