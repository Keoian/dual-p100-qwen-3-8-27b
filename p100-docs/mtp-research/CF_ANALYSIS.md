# Stop-rule tables at any sampling setting: counterfactual analysis (2026-10-24)

All throughput numbers are **simulated**, not measured. They come from the existing pipeline's cost model (`cache/cost.json`)
and its policy rules (n-max 4, W 5, dev fit / val choice / test report, paired bootstrap over test prompts). Scripts:
`analysis/cf*.py`; run all of them with `analysis/run_cf.sh`. Outputs: `analysis/out/cf/`.

## Method

- **Traces.** `trace-cfchat` (thinking off, 0.7 / 0.8 / 20 / 0, presence 1.5) and `trace-cfthink` (thinking on, medium
  effort, 1.0 / 0.95 / 20 / 0, presence 0). Both are n-max 4, p-min 0, 72 prompts, with `LLAMA_SPEC_TRACE_TOPK=128`. They
  are token-identical to the earlier n4p0 / tn4p0 traces (5116 of 5117 and 18531 of 18532 cycles).
- **Exclusions.** Cycles with a grammar sampler (`grmr`) are left out: 111 chat and 296 think cycles. These are all 9
  `tools` prompts, so each split keeps 21 prompts. Cycles with a forced row are also left out: 21 think cycles (84 rows).
  Kept: chat 5,004 cycles / 19,941 drafted rows; think 18,213 / 72,851.
- **a_S.** For each grid setting S (480 of them), p_S(x) is recomputed for every drafted row, teacher-forced along the
  drafted prefix. The penalty window is `pen_hist` plus the earlier drafted tokens. Then a_S = min(1, p_S/q).
  `cfvec.py` is `counterfactual.py`'s chain, vectorized.
- **Tables.** Isotonic iso_q and iso_p1 are fit on dev rows and sampled on the production 24-point x grid. Theta is chosen
  on val from 0 to 0.5 in steps of 0.0125. The simulator interpolates the tables linearly, which is assumed to match
  `spec_iso_lookup`.
- **Pooled set.** Both traces' prompts are separate units. Each trace's dev rows get equal total fit weight, and each
  trace gets equal token mass in rates. The bootstrap is stratified by trace.
- **Loss** = rate(approximation) / rate(S's own tables) - 1 on test. It is summarized as the mean over the 480 settings,
  the worst setting, and the mean of the worst 48 ("worst 10%").

## Validation

| check | result |
|---|---|
| `counterfactual.py selfcheck` | chat 19,949 rows, max abs(p - p_draft) 5.0e-7; think 72,943 rows (22 forced), max 5.3e-7; 0 rows inexact from the list |
| `cfvec` vs `counterfactual.row_probs` at random (row, grid setting) pairs | chat 1,547, think 1,572 pairs: max abs(dp) 0 (bit-identical), 0 exactness-flag mismatches |
| top_k filled inside the top-128 after penalties | every row at every grid setting, so every a_S is exact |
| simulator vs `sim.py` (p-min 0 / 0.2 / 0.5, production tables) | identical rates |
| production's fit procedure (`heads.isotonic_q`, n-max 4 trace replaced by the cf trace, + n-max 8 trace) | gives back production's chat and thinking tables exactly (max diff 0.000) |

**Reproduction at each trace's own setting** (cf trace dev rows only, at its own S, vs production's arrays):

| set | max diff iso_q / iso_p1 | mean signed diff | theta own / production | test rate, cf tables vs production tables |
|---|---|---|---|---|
| chat @ 0.7/0.8/20/0/1.5 | 0.105 / 0.100 | +0.04 | 0.175 / 0.125 | -0.11% [-0.30, +0.08] |
| thinking @ 1.0/0.95/20/0/0 | 0.065 / 0.124 | +0.03 | 0.175 / 0.15 | -0.01% [-0.08, +0.05] |

- The refit is higher than production by about 0.04, and 8-15 of 24 points lie outside its 95% dev-bootstrap band.
- The cause is production's fit data, not noise or the loader. Production also pooled the n-max 8 trace, half of whose
  dev rows are depths 5-8, where acceptance is lower (chat 0.48 vs 0.65 at depths 1-4).
- Dropping those rows from production's procedure reproduces the +0.04 shift (`out/cf/repro-decomp.txt`).
- Theta absorbs the shift (0.175 instead of 0.125 / 0.15), so throughput is equal. On val, production's arrays still
  choose 0.125 and 0.15.

## Content-shift check (step 3)

Tables fit on the other trace's content at the same S, scored on this trace's test split:

| scored on | setting | tables from | max diff iso_q / iso_p1 | loss, other trace's theta [95%] | loss, theta re-chosen here |
|---|---|---|---|---|---|
| think | thinking 1.0/0.95/20/0/0 | chat | 0.102 / 0.136 | -0.21% [-0.53, +0.26] | -0.01% [-0.15, +0.13] |
| chat | chat 0.7/0.8/20/0/1.5 | think | 0.100 / 0.115 | -0.14% [-0.49, +0.20] | +0.08% [-0.14, +0.30] |
| chat | thinking | think | 0.102 / 0.136 | -0.17% [-0.41, +0.10] | +0.09% [-0.04, +0.20] |
| think | chat | chat | 0.100 / 0.115 | -0.03% [-0.24, +0.20] | +0.02% [-0.15, +0.12] |
| chat, all 480 S | | think | median 0.105 / 0.120 | mean -0.07%, worst -0.40% | |
| think, all 480 S | | chat | median 0.105 / 0.120 | mean -0.23%, worst -0.96%, worst 10% -0.61% | |

- Content moves the tables by up to about 0.1, mostly at mid q (0.2-0.75): there, thinking content accepts 0.03-0.10 more.
- At low q the setting dominates, through top_p. At q 0.02 the thinking setting gives 0.248 from chat content and 0.254
  from thinking content, against 0.093 at the chat setting (0.203 at T 0.7 with top_p 0.95 instead of 0.8).
- Throughput barely notices either shift: at most -0.25% on average.
- **One trace type suffices for any setting** to within about 0.25% (worst 1%). Thinking content scored with chat-fit
  tables is the weaker direction.

## Grid summary (step 2): own tables vs production's current policy, test

| T | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 |
|---|---|---|---|---|---|---|---|---|---|---|
| production before f5fbd2542 uses | p-min | p-min | chat | chat | chat | chat | chat | thinking | p-min | p-min |
| chat content | +5.8% | +5.5% | -0.1% | -0.1% | -0.0% | -0.0% | +0.0% | +0.4% | +4.2% | +4.1% |
| think content | +6.6% | +5.9% | +0.1% | +0.2% | +0.3% | +0.4% | +0.4% | +0.0% | +4.0% | +4.0% |
| pooled | +6.5% | +5.8% | -0.0% | +0.0% | +0.1% | +0.1% | +0.2% | +0.2% | +4.1% | +4.0% |

Each cell is the mean over the 48 settings at that T.

| set | own vs production, grid mean | in the p-min region (192 S) | in production's table windows (288 S) | own vs p-min 0.2 | own t/s range |
|---|---|---|---|---|---|
| chat | +1.98% | +4.89% (+2.94..+6.15) | -0.22..+0.89 | +4.81% | 49.97-52.82 |
| think | +2.19% | +5.12% (+2.55..+6.85) | -0.10..+0.69 | +4.82% | 46.73-49.89 |
| pooled | +2.09% | +5.08% (+2.70..+6.84) | -0.10..+0.42 | +4.82% | 48.46-51.29 |

- **Production leaves the most on the table where it falls back to p-min 0.2**, at T <= 0.4 and T >= 1.1. Own tables
  gain +4-7% there.
- The largest gain is at T 0.3, top_p 0.8, presence 1.5: +6.8% [+5.8, +7.8] pooled. Within the p-min region, the gain is
  larger with presence 1.5 than with 0 (+5.4% vs +4.8%) and with top_p 0.8 than with 1.0 (+5.5% vs +4.8%).
- Inside production's windows the gain is noise-level (at most +0.9%).
- Acceptance hardly depends on the setting: mean a is 0.63-0.69 over the whole grid. top_k 20 vs 40 changes the own-table
  rate by at most 0.4%, min_p 0 vs 0.05 by at most 0.7%.

## Approximations of the family (step 4): loss vs own tables, test, mean / worst / worst 10% (%)

| approximation | storage (B) | chat | think | pooled |
|---|---|---|---|---|
| a. production before f5fbd2542 (gate by T, else p-min 0.2) | 392 | -1.88 / -5.79 / -5.62 | -2.09 / -6.41 / -6.19 | -1.99 / -6.40 / -6.08 |
| a. production's chat set at every T | 196 | +0.08 / -0.33 / -0.14 | -0.28 / -0.86 / -0.65 | -0.10 / -0.50 / -0.35 |
| a. production's thinking set at every T | 196 | -0.61 / -1.40 / -1.23 | -0.23 / -0.83 / -0.78 | -0.41 / -1.34 / -1.13 |
| a. production's two sets, chat T < 0.95, thinking T >= 0.95 | 392 | -0.01 / -0.88 / -0.56 | -0.16 / -0.69 / -0.52 | -0.08 / -0.42 / -0.32 |
| a. production's two sets keyed by thinking on/off | 392 | | | -0.08 / -0.54 / -0.42 |
| b. one set (the best grid setting on val) | 196 | +0.04 / -0.35 / -0.17 | -0.03 / -0.37 / -0.22 | -0.11 / -0.32 / -0.25 |
| b. nearest anchor, K = 2 | 392 | +0.06 / -0.35 / -0.17 | -0.01 / -0.37 / -0.17 | -0.08 / -0.32 / -0.24 |
| b. nearest anchor, K = 4 | 784 | +0.05 / -0.35 / -0.19 | -0.01 / -0.24 / -0.10 | -0.05 / -0.25 / -0.16 |
| b. nearest anchor, K = 8 | 1568 | +0.07 / -0.35 / -0.16 | -0.01 / -0.24 / -0.10 | -0.03 / -0.15 / -0.11 |
| c. multilinear in T {0.3, 1.2} | 392 | -0.12 / -0.72 / -0.53 | -0.04 / -0.30 / -0.22 | -0.03 / -0.22 / -0.15 |
| c. T {0.3, 0.7, 1.2} | 588 | -0.08 / -0.72 / -0.50 | -0.03 / -0.30 / -0.20 | -0.02 / -0.22 / -0.16 |
| c. T {0.3, 0.7, 1.2} x presence {0, 1.5} | 1176 | -0.01 / -0.31 / -0.25 | -0.02 / -0.25 / -0.16 | -0.01 / -0.20 / -0.11 |
| c. T(3) x top_p {0.8, 1} x presence {0, 1.5} | 2352 | -0.02 / -0.55 / -0.27 | -0.00 / -0.14 / -0.07 | +0.01 / -0.09 / -0.06 |
| c. T(5) x top_p(3) x presence(3) | 8820 | +0.02 / -0.29 / -0.12 | +0.00 / -0.22 / -0.07 | +0.01 / -0.07 / -0.04 |
| c. full T(10) x top_p(4) x presence(3) | 23520 | -0.00 / -0.41 / -0.12 | -0.00 / -0.22 / -0.07 | +0.00 / -0.07 / -0.03 |
| d. one set fit over the whole grid (marginal) | 196 | -0.02 / -0.52 / -0.31 | -0.12 / -0.51 / -0.32 | -0.09 / -0.33 / -0.25 |
| d. iso(q, T): marginal sets at T 0.3/0.7/1.2, linear in T | 588 | +0.02 / -0.26 / -0.19 | -0.04 / -0.23 / -0.17 | -0.01 / -0.12 / -0.09 |
| d. iso(q, T) at all 10 temperatures | 1960 | +0.04 / -0.26 / -0.16 | -0.04 / -0.18 / -0.15 | -0.03 / -0.20 / -0.12 |
| d. PCA family (mean + 1 component, linear in the fields) | 456 | -0.00 / -0.45 / -0.27 | -0.02 / -0.16 / -0.12 | -0.01 / -0.15 / -0.09 |
| d. PCA family, 2 components | 684 | +0.02 / -0.36 / -0.21 | -0.00 / -0.10 / -0.06 | +0.01 / -0.09 / -0.05 |
| e. analytic (reference tables x reshaped-draft ratio), uniform tail | 196 | -0.38 / -1.31 / -1.02 | -0.04 / -0.77 / -0.49 | -0.60 / -1.73 / -1.42 |
| e. the same + correction tables | 392 | -0.29 / -1.18 / -0.91 | -0.02 / -0.75 / -0.47 | -0.57 / -1.62 / -1.36 |
| f. own tables, one theta (0.175) for all | 92164 | +0.02 / -0.57 / -0.27 | +0.01 / -0.09 / -0.06 | +0.00 / -0.09 / -0.05 |

**Notes on the rows**
- Every table-based approximation is within 0.3% of own tables on average. The tables matter much less than whether the
  rule is on at all.
- **b.** Anchors are chosen greedily on val. Distance: abs(dT)/0.9 + abs(dtop_p)/0.2 + abs(dpresence)/1.5; top_k and
  min_p are ignored, with every anchor at top_k 20, min_p 0. **c.** Anchors are at top_k 20, min_p 0, and theta is
  interpolated too. Including top_k and min_p is not worth it: the full T x top_p x presence lattice is already within
  0.0% on average.
- **e. Analytic.** The draft's T = 1 top-20 is rebuilt from the recorded top-5. The unrecorded mass is spread uniformly
  over 15 tokens ("packed", the most concentrated tail, changes the mean by at most 0.05% and the worst case by up to
  0.4%). S's penalties, top_p, min_p and temperature are applied to it, and the drawn token's ratio
  min(1, p_hat/q) and the overlap sum min(q, p_hat) scale the reference setting's tables.
  - The top-5 misses 4.6% of the draft's mass on average (p90 16%). The drawn token is outside the top-5 in 4.9% of rows.
  - The tail is not the main error; the model is. Treating the target as the draft exaggerates the setting's effect 2-3x.
    For example, at T 0.7 / top_p 1 / presence 0 vs the chat setting, the predicted mean change is +0.095 and the actual
    +0.034. At T 0.3 / top_p 1 / presence 0 it predicts +0.07 for an actual +0.004.
  - In C++ the full top-20 would be available, but the model error would remain. Not recommended.
- **f. Theta** hardly needs to vary. Own thetas span 0.15-0.21 (median 0.175 in every set), and a single 0.175
  costs 0.00-0.02% on average.

## Candidates for deployment, with 95% intervals of the grid-mean loss vs own tables

| candidate | chat: mean [95%] / worst | think: mean [95%] / worst | pooled: mean [95%] / worst |
|---|---|---|---|
| R0 production before f5fbd2542 | -1.88 [-2.32, -1.37] / -5.79 | -2.09 [-2.37, -1.85] / -6.41 | -1.99 [-2.28, -1.68] / -6.40 |
| **R1 production arrays; thinking set for T in [0.95, 1.05], chat set at every other T** | **+0.05 [-0.07, +0.16] / -0.88** | **-0.24 [-0.46, +0.08] / -0.86** | **-0.10 [-0.20, +0.03] / -0.50** |
| R1b production chat set at every T | +0.08 [-0.04, +0.21] / -0.33 | -0.28 [-0.54, +0.11] / -0.86 | -0.10 [-0.23, +0.06] / -0.50 |
| R2 one new set for every S (pooled marginal fit, theta 0.175) | -0.04 [-0.22, +0.17] / -0.31 | -0.13 [-0.34, +0.15] / -0.56 | -0.09 [-0.16, -0.01] / -0.33 |
| R3 new sets at T 0.3/0.7/1.2, linear in T (theta 0.175) | +0.04 [-0.11, +0.21] / -0.27 | -0.05 [-0.21, +0.18] / -0.29 | -0.01 [-0.03, +0.02] / -0.12 |

**At the two card settings, against production's tables** (where nearly all traffic is):
- R2: -0.23% [-0.45, +0.05] on chat content, -0.37% [-0.75, +0.19] on think content.
- R3: -0.26% [-0.49, -0.01] and -0.20% [-0.47, +0.20].
- R1 is identical to production there.

**Gain of R1 over production before f5fbd2542:** +2.0% on the grid mean; +6.4 / +5.9% at T 0.3 / 0.4 and +3.8% at T 1.1 / 1.2
(pooled); 0.0% from 0.5 to 1.0.

**Outside the grid** (pooled, against that setting's own tables):

| range | p-min 0.2 | R1 | R2 | R3 |
|---|---|---|---|---|
| T 0.1-0.2 (T 0.1 is close to the greedy limit) | -6.5 to -7.4% | -0.05 to -0.25% | -0.36 to -0.59% | +0.01 to +0.13% |
| T 1.3-1.5 | -3.3 to -4.7% | -0.02 to -0.38% | -0.05 to -0.20% | -0.10 to 0.00% |

## Recommendation

**R1: keep both production table sets and drop the p-min fallback.**
- Use the thinking set (theta 0.15) for target T in [0.95, 1.05], as today, and the chat set (theta 0.125) at every other
  temperature.
- It needs no new arrays and no fit, and it is identical to production at both card settings.
- Mean loss vs own tables: chat +0.05%, think -0.24%, pooled -0.10%.
- Worst setting: -0.88% (chat content at T 1.0 / top_p 0.8 / presence 1.5, which is production's unchanged thinking
  window) and -0.86% (thinking content at T 1.2 / top_p 1.0).
- Simulated gain over production: +4-7% for requests at T <= 0.4 or >= 1.1.

R3 is the refinement if non-card temperatures become common: within 0.01% of own tables pooled, worst -0.12%, 588 B. It
costs about 0.2% at the card settings unless production's two sets stay as overrides there.

**C++ spec (R1)**, in `common/speculative.cpp` (MTP draft loop):
- **Arrays:** unchanged. `spec_iso_x[24]` (shared), the chat `spec_iso_q[24]` / `spec_iso_p1[24]`, and the thinking pair
  (production names may differ). That is 5 x 96 B = 480 B, plus 2 thetas. Values: `out/cf/recommended-c.txt`.
- **Selection,** per request from `dparams.target_temp` (`slot.task->params.sampling.temp`, already passed):
  - `t >= 0.95f && t <= 1.05f`: thinking set, theta 0.15;
  - otherwise (also t < 0, unknown): chat set, theta 0.125.
  - As shipped (default branch f5fbd2542), greedy requests (0 <= T < 0.001, not in the grid) keep the p-min test.
- **Env:** the same as widening `LLAMA_SPEC_STOP_ISO_TEMP` to all temperatures, provided the thinking window
  (`LLAMA_SPEC_STOP_ISO_THINK_TEMP`) is checked first.
- **Lookup:** as today. Linear interpolation on `spec_iso_x`, s_iso *= iso_q(q_id) after each drafted token, and stop
  before step d when s_iso * iso_p1(p1_d) < theta.

**C++ spec (R3, optional):**
- **Arrays:** `float spec_iso_q_t[3][24]`, `spec_iso_p1_t[3][24]`, `spec_iso_t[3] = {0.3, 0.7, 1.2}`, theta 0.175.
  That is 576 B + 12 B.
- **Per request:** tc = clamp(t, 0.3, 1.2), j = tc < 0.7 ? 0 : 1, f = (tc - T[j]) / (T[j+1] - T[j]). Build two 24-entry
  scratch tables (1-f)*tab[j] + f*tab[j+1] once per request, then look up as today.

## Caveats

- Simulation only. Window 9 measured the chat tables at the thinking setting at -0.8% (-4.7..+1.9) vs p-min, where this
  simulator says +2.7% [+1.4, +4.7]. So measure R1 at one non-card temperature (e.g. T 0.3 and 1.2, thinking off) before
  trusting the +4-7%.
- Content shift: the text is the one recorded at each trace's own setting. The cross-content check bounds what a
  different text does to the tables (0.25% on average).
- Grammar (tools) requests are not covered: 9 of 72 prompts are excluded.
- n-max 4 only. 21 test prompts per trace. A target T <= 0 uses the greedy verify; T 0.1 stands in for it.
- The lookup was assumed to interpolate linearly; checked afterwards: `spec_iso_lookup` is piecewise-linear between the
  table's points and clamped at both ends, so the assumption holds.

Runtime: about 15 min of single-threaded CPU at nice 10, at most 2 processes. Caches take 244 MB in
`analysis/cache/cf/`.

## Measured afterwards (2026-10-24 evening)

- **R1 on the box** (window 14, default branch `f5fbd2542`, thinking off, 72 prompts, production sampling with the
  temperature overridden; order P R R P): T 0.3 47.65 -> 50.81 t/s, **+6.6%** (95% over prompts +4.7..+8.7);
  T 1.2 48.39 -> 51.33, **+6.1%** (+3.8..+8.2). Simulated +6% / +3.8%. In production since 22:49; a smoke test
  there reproduced the A/B replies token for token at T 0.3 and production's at the card settings.
- **Deeper caps** (`analysis/nmax_smoke.py`, simulated on the n-max 8 traces): the rule at n-max 5 / 6 / 7 is
  -2.5 / -8.6 / -14.2% vs production (thinking off; thinking on -5.1 / -11.1 / -16.3%), n-max 4 +4.7 / +3.2%.
  Even with a verify of exactly the real width, n-max 4 stays best: a filled column past width 5 costs ~8.7 ms and
  a 5th draft token is worth ~0.2 expected tokens per round.
- **Exact-width verify** (window 15, `scripts/spad.py`: padded vs `LLAMA_SPEC_PAD=0`): unpadded verify minus its
  extra graph build is only 0.3-1.2% cheaper than the padded one at the same real width (2 columns 42.26 vs
  42.77 ms); the padded skip is already ~perfect. End to end unpadded is -11% (a graph rebuild per width change).
  The simulator's cost model is right for padded verify (within ~1 ms) but its calibrated narrow-width costs V2-V4
  were too low, which is where a simulated +3% for exact width came from.
