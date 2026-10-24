# Three-Head Adaptive MTP: Research and Implementation Plan

**Audience:** Claude Code CLI working inside the user's existing inference-engine fork  
**Status:** Research specification and implementation roadmap; not an implementation design  
**Primary objective:** Test whether three lightweight, trained decision heads can improve end-to-end generation speed by choosing when and how to trust native MTP drafting, with explicit control over quality loss.  
**Secondary objective:** Learn whether an MTP head can produce useful, longer self-directed runs between full-model executions.

> **IMPORTANT FOR CLAUDE CODE:** Explore the actual repository before designing or editing anything. Do not assume upstream architecture, names, internal APIs, kernel layouts, feature support, model dimensions, MTP recurrence mechanics, or cache invariants. Where this document describes behavior, treat it as a **research goal or hypothesis**, not as a claim that the fork already supports it. Derive implementation choices from the code, model metadata, measurements, and tests you find. Report blockers rather than silently substituting a different experiment.

---

## 0. Project context and research question

The target environment is an existing customized inference-engine fork running **Qwen 3.8 27B Q6_K** with native MTP on **two Tesla P100 16 GB GPUs**. The owner reports roughly **40 generated tokens/sec** in a recent optimized configuration, with an approximately **four-token MTP draft length** proving preferable in earlier throughput experiments. The fork reportedly has **chunked cache snapshots** and **deterministic cache restoration**. All of these are *context supplied by the owner*, not verified current facts; measure and document them before treating them as baselines.

We want to study a **three-head inference controller**:

1. **Task-Type Head:** Recognize what kind of work the model is doing, including switches *within* a response (code, explanation, creative writing, casual chat, reasoning, structured output, tool calls, etc.).
2. **Difficulty Head:** Estimate how much full-backbone computation or verification a given span likely needs; this is an operational estimate, not just a subject-matter classification.
3. **Fidelity Head:** Estimate whether a proposed MTP continuation will agree with, or remain acceptably close to, what the Q6_K teacher would produce under the **same prefix and sampling conditions**.

Together, they inform an **adaptive decoding policy** selecting such controls as:

- How many tokens to draft before checking or re-anchoring.
- Whether drafting and/or target sampling is greedy or probabilistic, where supported.
- Whether to use normal target-model verification, confidence-gated approximate acceptance, or a full-model-only path.
- Auto-acceptance risk thresholds, a maximum unverified run length, and the conditions that trigger a new full-model anchor.
- Whether certain spans (especially tool/API structures, code-sensitive syntax, or uncertain situations) require strict verification.

These are **possible controls, not guaranteed existing knobs**. Discover which can be altered correctly and cheaply in the fork.

### Two separate success modes

**Mode A — verified speculative decoding:** The full model verifies proposed tokens, preserving its intended decoding distribution when the appropriate acceptance algorithm is used. Gain comes from more efficient drafting/batched verification, not from skipping necessary target computation.

**Mode B — approximate adaptive decoding:** A controller sometimes commits draft tokens *without* exact target verification, then optionally re-anchors. This may skip target work, but **cannot guarantee Q6_K-equivalent tokens, distribution, reasoning, or output quality**. Label it as approximate in interfaces, experiments, and reports.

A useful result could be faster verified decoding, faster approximate decoding at an acceptable measured quality loss, or a rigorous finding that neither helps on this hardware.

---

## 1. Operating rules for Claude Code

- **Explore before prescribing:** Inspect the fork's actual MTP, sampling, batching, model-state, rollback, cache, and output-streaming paths. Trace them end-to-end. Document what is found with repository paths, relevant symbols, and test commands. Do not infer support from upstream documentation.
- **Measure before optimizing:** Reproduce the existing performance and correctness baseline on the owner's real workload/hardware before changing behavior. Record commit, model and quantization, flags, drivers, GPU clocks/power, context length, and test environment.
- **No fabricated specifics:** Do not make up classes, APIs, training corpora, model dimensions, extra VRAM, acceptance rates, results, or estimated speedups. Select head architectures, frameworks, thresholds, and file locations *after discovery and measurement*.
- **Preserve the baseline:** Keep a reversible, default-off research path. Normal Q6_K+MTP must remain usable; protect cache restore determinism and existing optimizations.
- **Separate observation from intervention:** Add tracing/profiling first; make behavioral changes in independently testable steps.
- **Prove state correctness:** The model includes nontrivial sequence state. Validate both attention-cache and recurrent-state handling from the actual fork before trusting replay, rewind, or re-anchoring. A copied token sequence alone is not proof of equivalent internal state.
- **Count all costs:** Throughput must include task classification, head execution, draft generation, verification, token replay, re-anchoring, rollback, synchronization, and any GPU/CPU transfers.
- **Do not confuse sampling with verification:** Standard probabilistic speculative acceptance needs the target-model probability and draft-model probability of the proposed token (with correct residual sampling on rejection). A learned confidence threshold that accepts without the target distribution is *approximate decoding*, not exact probabilistic verification.
- **Don't expose retractable tokens:** Once a token or tool call is streamed externally, it cannot simply be rolled back. Stage unverified output when rollback is possible, or explicitly operate in irrevocable approximate mode. Never execute speculative tool calls.
- **Use small, reviewable increments:** Make one behavioral change at a time, with correctness tests and A/B measurements. Maintain a decision log of findings and abandoned paths.

---

## 2. Phase 0 — Repository reconnaissance and feasibility audit

**Goal:** Establish what is possible, where the real costs lie, and whether native MTP can operate without a new full-model hidden state.

Tasks:

- [ ] Identify the current branch/commit, upstream lineage, build and benchmark commands, and relevant local patches.
- [ ] Trace prompt prefill, regular decode, native MTP draft generation, draft-token verification, acceptance/rejection, and output streaming.
- [ ] Verify exactly how the MTP head obtains its inputs, handles token positions, updates its own intermediate state, and produces logits.
- [ ] Determine whether repeated MTP-only decoding is currently possible. If not, characterize what state transition, feature input, or training would be required; **do not assume feeding its output back is valid**.
- [ ] Determine whether draft length and sampling policies can be changed per batch or per token without invalidating caches or distributions.
- [ ] Trace KV cache and any recurrent-state checkpoint/restore/replay paths; test whether a proposed re-anchor produces the same future full-model result as uninterrupted teacher-forced processing of identical tokens.
- [ ] Inventory accessible, inexpensive signals: relevant hidden states, draft logits/log-probs, entropy, top-token margins, previous acceptance history, sampling settings, timing, and task-context features. Measure retrieval/transfer cost for every proposed head input.
- [ ] Profile decode vs. prefill/batched verification vs. MTP/logit projection; inspect VRAM budget and bottlenecks across both P100s.
- [ ] Map hard constraints: maximum context, state ownership, CPU/GPU boundaries, synchronization, GPU occupancy, memory headroom, supported output formats, and existing automated tests.

**Deliverable:** `RECONNAISSANCE.md` (or another repository-appropriate path) containing an accurate data-flow map, concrete code references, benchmark reproduction instructions, capabilities/gaps table, and a feasibility verdict for each proposed intervention. No architectural refactor in this phase.

**Decision gate:** If free-running MTP is unsupported, explicitly choose whether to (a) first implement/train an autonomous recurrent drafting pathway, (b) pursue verified adaptive MTP only, or (c) stop that subproject. Never quietly describe normal verified drafting as independent MTP generation.

---

## 3. Phase 1 — Baselines, instrumentation, and trace collection

**Goal:** Create trustworthy evidence that later experiments can be compared against.

- [ ] Run baseline full-model decoding, current production/default MTP settings, and a small fixed-depth MTP sweep compatible with the discovered implementation.
- [ ] Instrument per-token/per-batch draft count, accepted count, rejection position, target passes, batch size, token latency, TTFT, prefill, and end-to-end output throughput.
- [ ] Distinguish **draft tok/s**, **committed tok/s**, and **user-visible tok/s**. Record wall-clock totals and GPU/CPU/memory utilization.
- [ ] Capture draft-vs-teacher distributions where feasible, including per-position teacher log-probability of drafted token, teacher argmax, entropy, margins, and any actual verifier decision.
- [ ] Record teacher-consistent hidden states at carefully defined matching positions if available; note how these states are obtained and what computation/transfer they cost.
- [ ] Validate output correctness and cache determinism after rejection and restoration, not just numerical performance.
- [ ] Build repeatable scripts to capture traces without changing the baseline's sampling behavior. Keep trace generation out of timed benchmarks unless that overhead would exist in production.

### External fixed-config comparison: Discord reference (October 21, 2026)

An outside experimenter shared the following **reported** Qwen 3.8 27B Q6_K / native-MTP launch configuration in a Discord screenshot. Treat it as a **configuration to reproduce if compatible**, not as a verified result, an endorsed setting, or proof that every flag works in this fork. The GGUF path is intentionally a placeholder. The shell formatting below simply groups the displayed environment variables and flags into one command; it is **not an implementation prescription**.

```bash
GGML_CUDA_P2P=1 \
GGML_CUDA_GRAPHS_PRE_VOLTA=3 \
LLAMA_SPEC_SAMPLE_TEMP=1.0 \
LLAMA_SPEC_DRAFT_TOPK=20 \
./build-opt/bin/llama-server \
  -m /path/to/Qwen3.8-27B-Q6_K.gguf \
  -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
  -c 262144 -b 32768 -ub 2048 -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 \
  -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 \
  --jinja --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 \
  --host 0.0.0.0 --port 8080
```

**Mandatory interpretation checks before a comparison:**

- [ ] Verify all options and environment variables in the **actual installed fork**, including what `LLAMA_SPEC_SAMPLE_TEMP`, `LLAMA_SPEC_DRAFT_TOPK`, `GGML_CUDA_GRAPHS_PRE_VOLTA`, `-ubd`, `-ctkd`, and `-ctvd` do, whether they are honored, and whether their semantics differ across commits.
- [ ] Confirm that `--spec-draft-p-min 0.2` controls **whether drafting continues based on draft confidence** (or document the actual discovered behavior); do not call it a *20% main-model acceptance threshold*. In normal verified speculation, target verification still decides acceptance. Its relationship to the proposed controller's **unverified auto-acceptance risk threshold** must stay explicit.
- [ ] Distinguish **draft sampling**, **target sampling**, and **acceptance/verification algorithms**. `--temp 1.0` or `LLAMA_SPEC_SAMPLE_TEMP=1.0` alone does not establish that the verifier is operating in a specific probabilistic mode. Confirm semantics in code.
- [ ] Test whether `GGML_CUDA_P2P=1` actually enables GPU peer transfers on this machine's topology; setting an environment variable does not establish operational P2P. Similarly validate the pre-Volta graphs setting on the P100s.
- [ ] Verify memory feasibility of the 262,144-token context and cache formats on the **current** dual-P100 arrangement; avoid silently reducing context only for one configuration. Use matched *effective* context lengths in paired tests.
- [ ] Preserve this exact transcript as an external reference while making any machine-dependent adjustments (model path, bind address/port, supported flags, context/caches) in a **separately documented adapted command**. For local tests, prefer a localhost bind unless LAN access is needed.
- [ ] Record original post attribution/date and that the screenshot alone supplies **no validated throughput or acceptance data**.

#### Fixed-policy and controller benchmark matrix

Use the same prompt suites, effective context, target sampling policy, quantization/cache state, hardware conditions, and scoring procedure wherever possible. When a setting cannot be held fixed, make it an explicit variable and report its effects. All listed variants are **candidate experiments subject to Phase 0 support**.

| ID | Comparison | Question answered |
| --- | --- | --- |
| B0 | No MTP, full Q6_K | What is the uncompromised sequential-decoding baseline? |
| B1 | Owner's **current optimized fixed** MTP setup | Are we improving the existing ~4-token configuration rather than an artificially weak baseline? |
| B2 | **Discord reference** above, reproduced or explicitly marked unsupported | How does `n-max=4`, `p-min=0.2`, and the posted sampling/runtime configuration behave on this fork? |
| B3 | Fixed verified MTP **draft-depth sweep** (e.g., 2, 4, 8, 12, 16; select only supported values) | Where is the throughput / accepted-tokens optimum for each workload? |
| B4 | Fixed verified MTP **draft-confidence-cutoff sweep** (include 0.2 and higher values where valid) | Does loosening/tightening the draft stop criterion improve real throughput or merely increase rejected work? |
| B5 | Fixed **greedy versus stochastic drafting** and appropriately configured target sampling/verification | How do sampling and acceptance behavior affect throughput and distribution fidelity? |
| B6 | Untrained heuristic adapting draft depth/cutoff from observable signals | How much value is obtainable without trained heads? |
| B7 | Trained **three-head controller, verified mode** | Do the learned heads beat the best *matched fixed* policy and simple heuristics? |
| B8 | Static confidence-gated approximate mode versus trained **three-head approximate mode**, **only if independent drafting and safe state handling are proven** | Does learned auto-acceptance achieve a better measured quality-speed frontier? |

**Minimum ablations:** Vary one control at a time before jointly tuning draft length, draft confidence cutoff, and sampling mode; include an explicitly matched B2-vs-B7 comparison. Benchmark per workload and context regime, not just aggregate tok/s. Select settings on validation prompts, then score on held-out prompts. Report accepted draft tokens per target pass, rejection depth, fraction of proposals stopped before verification, total full-model work, full output speed, head/controller overhead, and task quality. Keep verified and unverified results in separate tables; more drafts or a lower cutoff do **not** automatically mean greater acceptance or better net speed.

### Workload suite

Use a deliberately diverse set with separate development, validation, and held-out tests:

| Workload | Important outcomes |
| --- | --- |
| Casual chat / short answers | Fluency, responsiveness, relevance |
| Long creative writing | Coherence, repetition, continuity |
| Software development | Compile/test success, patch correctness, syntax |
| Technical reasoning | Answer accuracy, consistency, error accumulation |
| Tool selection and structured arguments | Exact format/schema validity, appropriate tool/action choices |
| Constrained JSON and format adherence | Parse/schema pass rate and field correctness |
| Long-context questions / retrieval | Correct reference to context and stability at varying positions |
| Task-switching conversations | Correct policy changes when task type changes mid-response |

Use both short and long contexts that fit the real configuration. Fix seeds where applicable; run multiple seeds for stochastic modes. Record power, thermal, and warm/cold-cache effects. Include naturally occurring long outputs, not just synthetic generation loops.

**Deliverable:** Baseline report, profiling traces, reusable benchmark commands, and a machine-readable experimental dataset with a documented schema and provenance.

**Decision gate:** Decide which operations actually dominate the cost. If target re-anchoring or head feature extraction is too expensive, redesign before investing in training.

---

## 4. Phase 2 — Prove the free-running MTP hypothesis

**Goal:** Answer whether, and for how long, MTP can generate independently without severe drift.

Only proceed when the repo audit establishes a technically meaningful free-running mechanism; adding a learned state transition may be a separate research task.

Experiments:

- [ ] Start with a trusted full-model anchor and generate approximate continuations of increasing lengths (candidate depths: 4, 8, 16, 32, 64; larger only if justified).
- [ ] Test greedy and stochastic drafting if meaningful and supported. Separate *MTP confidence* from *full-model agreement*.
- [ ] Evaluate the full teacher under the **same drafted prefixes**, and record per-position divergence. Do **not** compare a teacher's independently sampled story to a draft story and count every textual difference as a mistake.
- [ ] Measure self-repetition, entropy collapse, incoherence, syntax errors, degradation with depth, and outright instability.
- [ ] Analyze whether the failure appears linked to missing backbone information, representation drift, feedback/input mismatch, or numerical error. Avoid asserting a cause without an ablation.
- [ ] Measure the cost of committing a burst and synchronizing the full-model attention/recurrent state by whatever method Phase 0 proves correct.

**Deliverable:** A plotted **quality vs. unverified-run-length vs. net-throughput** tradeoff, plus examples of representative failure modes.

**Decision gate:** If MTP-only recurrence cannot produce useful spans or the replay cost erases savings, pivot to adaptive *verified* drafting and stop relying on auto-accept for the near-term speedup.

---

## 5. Phase 3 — Dataset and training objectives for the three heads

**Goal:** Train the smallest practical predictors using signals available at the moment their decisions must be made.

The owner wants **three small, distinct heads**. Initially train/evaluate them independently. They may later share computation if profiling demonstrates a benefit, but preserve three separately inspectable outputs.

### Head A: Task Type

**Question:** What mode of work is this span doing *now*?

- Candidate labels include: casual conversation, creative writing, code, structured tool/JSON output, factual/technical explanation, multi-step reasoning, and mixed/transition.
- Prefer multi-label or hierarchical labels if the collected data demonstrates overlapping tasks.
- Train from prompt/output annotations and any inexpensive contextual or hidden-state signals available when a decision is needed.
- Evaluate within-response switching, not only classification of the user prompt.
- Measure false classifications that might relax verification on sensitive structured output.

### Head B: Difficulty

**Question:** How much risk or teacher computation does this local span require to preserve a selected quality target?

- Define operational ground truth using observed teacher-vs-draft behavior, task-level success, and the effect of changing verification/re-anchor frequency.
- Avoid equating long prompts, high entropy, or the words "complex reasoning" with ground-truth difficulty.
- Predict an interpretable ordered score or risk category that correlates with extra verification being *worth its cost*.
- Assess calibration across task types, context lengths, and transitions.

### Head C: Fidelity

**Question:** How likely is this specific continuation to match or remain acceptable to the teacher before the next anchor?

- Use teacher-forced evaluation on the **identical draft prefix** to produce token-level and burst-level supervision.
- Keep distinct labels for greedy exact-token agreement, probabilistic teacher preference, and downstream quality. They are not interchangeable.
- Test prediction of mismatch *before* it happens, including confidently wrong drafts, long bursts, and out-of-distribution prompts.
- Calibrate risk estimates rather than evaluating classification accuracy alone. Report precision/recall at the operating thresholds and reliability curves.
- Explicitly prevent teacher-only logits, future target hidden states, or post-verification acceptance information from leaking into deployment-time predictor inputs.

### General data/training requirements

- [ ] Capture varied prefixes, draft depths, sampling modes, and context lengths rather than only successful short greedy batches.
- [ ] Preserve sequence/group splits by prompt and source (including repository or document where relevant) to avoid leakage across train/validation/test.
- [ ] Address class imbalance and selection bias: easy high-acceptance tokens must not dominate fidelity training.
- [ ] Confirm inputs at inference time are available at the chosen decision frequency without running expensive full-model passes solely for the heads.
- [ ] Benchmark alternative small head sizes against memory, compute, synchronization, and calibration cost; choose based on data.
- [ ] Maintain a versioned labeling protocol, dataset manifest, and reproducible training/evaluation commands.

**Deliverable:** Head training/evaluation report with held-out calibration, confusion matrices where relevant, latency and VRAM overhead, error analysis, and a decision to retain or discard each head.

**Decision gate:** If a head fails to beat inexpensive heuristics on held-out data *or* costs more than it saves, do not include it in the production policy.

---

## 6. Phase 4 — Deterministic controller before learned policy

**Goal:** Combine the head outputs into an understandable, measurable inference policy *without* conflating controller quality with head quality.

Implement a simple, configurable policy using the measured outputs. Exact controls and frequency depend on the code audit.

Research actions to evaluate:

- Choose MTP draft/burst length (including zero/no MTP).
- Choose greedy vs. stochastic *drafting* where relevant; separately control the target sampler according to user/model settings.
- Adjust confidence/risk thresholds for approximate acceptance, and cap maximum consecutive unverified tokens.
- Select verification/re-anchor strategy at boundaries proven to be state-safe.
- Force verified mode for strict tool/schema paths and when confidence calibration is poor or the input is out of distribution.
- Use recent outcomes to shorten or lengthen batches, where those outcomes actually exist; do not claim access to unobserved target acceptance on skipped-verification steps.

Begin with policy ablations:

1. Fixed baseline (existing verified MTP behavior).
2. Adaptive draft length without any learned heads (cheap heuristic baseline).
3. Task Type only.
4. Difficulty only.
5. Fidelity only.
6. Every two-head combination.
7. All three heads.
8. All three with approximate auto-accept and re-anchoring, **only if Phase 2 established a viable path**.

Allow the system to revert to standard Q6_K/MTP decoding on classifier uncertainty, excessive risk, unstable state, memory pressure, or output-format constraints. Treat target-model verification and approximate acceptance as explicitly different modes; don't change sampling semantics silently.

**Deliverable:** Documented controller policy, configuration toggles, telemetry, tests, and an ablation table showing the *incremental value* of each head.

---

## 7. Phase 5 — State synchronization, rollback, and output safety

**Goal:** Guarantee correct full-model state when transitioning back to Q6_K, and avoid invalid speculative side effects.

- [ ] Determine which state must advance for every committed token and prove that the chosen anchor/replay mechanism handles all relevant state correctly.
- [ ] Verify rollback against a reference decode with the exact same retained token prefix, model settings, and sampling state.
- [ ] Confirm repeatability after cache snapshot restore and after a partial rejection; measure replay and restore latency separately.
- [ ] Establish a defined relationship between speculative-but-not-committed tokens, displayed tokens, internal teacher state, and tool outputs.
- [ ] Buffer tokens during any interval that may be rolled back. Do not fire tools, write files, or externally commit actions from a speculative branch.
- [ ] Test forced EOS, context rollover, prompt changes, cancellation, variable batch sizes, streaming interruptions, and transition between task types.
- [ ] If approximate tokens are irreversibly streamed, document that later re-anchoring does not retroactively make them teacher-approved.

**Deliverable:** State-transition invariants, replay/restore correctness tests, rollback stress suite, and exact accounting for synchronization overhead.

---

## 8. Phase 6 — Controlled optimization and final evaluation

Only tune performance **after** the correctness, trace, and head-training stages are reproducible.

### Required comparisons

- Existing 27B Q6_K native-MTP configuration.
- October 21, 2026 Discord fixed-setting reference (`n-max=4`, `p-min=0.2`, plus displayed runtime and sampling flags), after documenting compatibility/adaptations.
- Matched single-variable MTP draft-depth and draft-confidence-cutoff sweeps, and greedy-versus-stochastic sampling experiments with verified acceptance semantics.
- A **paired B2-vs-B7** evaluation: external fixed configuration versus trained three-head verified controller under matched conditions; compare separately against the owner's optimized B1.
- No-MTP full-model decoding.
- Fixed-depth verified MTP sweep.
- Heuristic adaptive verified MTP.
- Three-head adaptive **verified** MTP.
- MTP-only approximate bursts with static confidence threshold (if viable).
- Three-head adaptive **approximate** decoding (if viable).
- Any reduced-cost MTP output projection or recursive second-head path pursued later, as its *own* ablation.

### Metrics

**Speed and cost:** End-to-end committed/user-visible tok/s, per-token latency and p95, TTFT, verification frequency, accepted tokens per target pass, fraction of full-model work avoided (if any), prefill/replay costs, wall time, GPU utilization/power, memory consumption, and head overhead.

**Quality and safety of execution:** Teacher-forced token agreement, distribution/calibration metrics where sampling is stochastic, factual/reasoning accuracy, code test success, JSON/schema validity, correct tool decisions, creative-writing coherence, long-context consistency, and repetition/degeneration rate. Include failures and regressions by task type.

**Methodology:** Paired prompt comparisons, multiple random seeds where applicable, warm/cold cache separation, held-out prompts, report of uncertainty, and statistical variability. The optimized configuration must be chosen on validation data and scored once on a reserved test set.

### Success gates

1. **Correctness:** Baseline default mode remains unchanged; cache restore/replay tests pass; verified decoding behaves as intended under supported sampling modes.
2. **Learnability:** Each retained head adds measurable predictive value beyond cheap heuristics with acceptable runtime overhead.
3. **Net value:** At least one controller variant improves end-to-end output speed versus the *current optimized* baseline on a defined workload, after all overhead.
4. **Quality accounting:** Any approximate speedup is accompanied by an explicit, measured quality penalty and a per-task failure profile. Never call an approximate result "Q6_K-equivalent" without evidence.
5. **Practicality:** Required memory and power fit the actual P100 setup without displacing critical model or cache components.

Avoid setting an arbitrary speedup percentage as a pass condition before profiling. Report negative results, including cases where a simpler heuristic matches or beats the trained heads.

---

## 9. Optional follow-on branches (do not implement by default)

Only open these investigations if measurements identify a reason:

- **Head-for-head / recursive MTP:** Train a still-smaller speculative producer for native MTP, then evaluate whether intermediary checking saves enough work. Requires explicit proof of the expected data flow and training target.
- **Hidden-state correction head:** Predict recursive MTP state error relative to teacher-conditioned representations. Study whether it lengthens stable rollouts. This would be a *fourth* research component, not one of the requested three control heads.
- **Draft trees:** Test branching candidates if multi-token verification shows meaningful batching opportunities.
- **Reduced-vocabulary or cheaper projection:** Only if profiling confirms MTP output projection is a major cost and correctness can be measured.
- **Distilled drafter or low-bit 27B proxy:** Compare only if spare compute/memory and net throughput justify a separate drafter.
- **Layer/neuron sparsity (virtual MoE):** A separate model-execution research effort with model-quality and state-correctness questions; not part of this initial three-head implementation.
- **Learned policy optimization:** Consider contextual bandits or other optimization *after* deterministic policy ablations establish reliable measurements and constrained quality objectives.

---

## 10. Suggested checkpoints and artifacts

Claude Code should adapt artifact paths and naming to repository conventions discovered in Phase 0; the following are **deliverables, not assumed existing files**.

| Checkpoint | Evidence to hand back before moving on |
| --- | --- |
| C0: Repository audit | Architecture trace, relevant symbols, feasibility map, verified baseline instructions |
| C1: Benchmark harness | Reproducible benchmark and trace schema; baseline plots and known limitations |
| C2: Independent MTP test | Actual self-drafting feasibility, depth-vs-quality-vs-speed curves, failure analysis |
| C3: Three heads | Training pipeline, versioned data, held-out metrics, feature/compute cost |
| C4: Controller | Exact/approx mode separation, configurable policies, exhaustive ablations |
| C5: Synchronization | Tested replay/restore invariants, rollback and streaming safety |
| C6: Final comparison | Workload-by-workload speed-quality frontier and a documented recommended configuration |

At each checkpoint:

- State **what was observed**, **what remains hypothetical**, and **what changed**.
- Show actual measured values and the commands or scripts that produced them.
- Identify whether continuation is worthwhile; recommend a pivot if evidence warrants one.
- Keep commits focused, make experiments reversible, and preserve the baseline.

---

## 11. First instructions to execute in Claude Code CLI

1. Read this research plan completely.
2. Explore the **current working fork**; don't assume its internal design matches upstream llama.cpp or other public implementations.
3. Produce the Phase 0 architecture/feasibility report with repository-specific references and a clear answer to: **Can native MTP produce successive tokens without new backbone hidden states, and if not, what does making it autonomous actually require?**
4. Identify the minimally invasive instrumentation needed for Phase 1, and where it can live without disturbing timing-critical paths.
5. Reproduce the existing benchmark and relevant cache-state correctness tests.
   Include the Discord October 21 fixed configuration as an **additional reproducible baseline** if supported; record every divergence from the screenshot and do not conflate `--spec-draft-p-min` with the proposed auto-accept policy.
6. Propose a repository-grounded milestone sequence with dependencies, candidate tests, expected risks, and rollback strategy; clearly mark any blockers.
7. Then work through the research phases, using actual results to choose subsequent experiments. **Do not invent implementation-specific details to fill gaps.**

### Final research question

> **Can task-type, difficulty, and fidelity prediction provide enough information to vary native MTP behavior on the fly, allowing a dual-P100 Qwen 3.8 27B Q6_K system to produce more useful tokens per second than its already-optimized fixed-depth MTP baseline? And if unverified MTP bursts are feasible, what quality-versus-speed frontier results from confidence-gated re-anchoring?**
