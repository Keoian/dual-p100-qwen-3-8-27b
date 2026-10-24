# Handoff

Where the work stands, and what's worth doing next. For the gates, see `tools/gate.sh` and the "Gates" and "Operator
rules" items of the tyler-port section below (the operator's mission brief lives outside the repo, `/work/CLAUDE.md`
on the build box).
For the results and changes, see `p100-docs/`; to bring the production setup up elsewhere, `p100-docs/deploy/`.

## Branch `p100-optimizations-keoian` (state 2026-10-24; default branch since 10-22, production on the reference box)

- Branch contents (CHANGES §18-22):
  - all of the base `p100-optimizations` up to its tip d90ec36a3 (2026-10-17, unchanged since), merged;
  - `tyler-port` and `kv-q8q4-262k-vision1440-q8proj`, both ancestors of this branch now (historical names; work
    happens here);
  - a background pre-fault of a spare context checkpoint, `-b 2048` in the deploy script;
  - thinking off as the template's own generation prompt and sampling by thinking mode (aa942a7d3), the penalty
    sampler visiting only the window's tokens (0d920143d), the card's presence_penalty 1.5 with thinking off;
  - three commits brought over from the PRs: the GGML_ASSERT fix (400aac811), 42 MUL_MAT tests (d30893442), `-b 2048`
    in QUICKSTART (504afc465);
  - the README rewrite (0d682248f) and a deploy recipe that brings up this setup (a6cf418f6);
  - ideas from other P100 forks (CHANGES §21, 10-24): recurrent states cleared with FILL (03bb2a866), `seq_rm` over
    the used cells only (5974d909e), the sampler prefilter with penalties (910d2cfb8), the q4p q4_0-K spill fix
    (c73976485), and the bf16 exchange wire in the start script (`AR_BF16=1`, ef2df6c23). Production runs all of it
    since 10-24;
  - the MTP draft's learned stop rule (CHANGES §22, 10-24): `LLAMA_SPEC_STOP_ISO` with its temperature gate
    (c44158c77), on in the start script (`STOP_ISO=0.125`, a6c045269); tables for thinking requests (504d4d353),
    on in the start script (`STOP_ISO_THINK=0.15`, e0b402541), and the chat tables for every other positive
    temperature (f5fbd2542). Production runs all of it since 10-24 evening.
- Production on :8090 is `p100-docs/deploy/start-qwen-vision.sh` (which runs `start-qwen.sh`) from `build-vision`:
  - `-c 262144 -sm tensor`, KV q8_0 K / q4_0 V, `-ub 1024 -b 2048 -np 1`;
  - MTP drafts n-max 4, `LLAMA_SPEC_SAMPLE_TEMP=1.0`, `LLAMA_SPEC_DRAFT_TOPK=20`; the draft stops by the lookup rule,
    one pair of tables per thinking mode chosen by the target temperature (`STOP_ISO_THINK=0.15` for 0.95-1.05,
    `STOP_ISO=0.125` for any other T > 0, p-min 0.2 for greedy): chat +4.7%, long context +4.8%, thinking +2.9%,
    T 0.3 / 1.2 +6.6% / +6.1%, exact;
  - `-ctxcp 32 --cache-ram 24576` (48 GB host; on 32 GB `CTXCP=4 CACHE_RAM=8192`);
  - Q8_0 projector on GPU1, `IMAGE_MAX_TOKENS=3600`, `GGML_CUDA_FA_VEC_MAX_Q=16`, `GGML_CUDA_FA_GEMM_MINQ=17`;
  - `--reasoning off` with `LLAMA_THINK_BY_BUDGET=1` and `LLAMA_SAMPLING_BY_MODE=1`: thinking off 0.7 / 0.8 / 20 / 0
    with presence_penalty 1.5 (the card's), thinking 1.0 / 0.95 / 20 / 0 / 0; CLI sampling (for `/completion`)
    0.7 / 0.8 / 20 / 0;
  - `AR_BF16=1` (`GGML_CUDA_AR_BF16_THRESHOLD=1`): the decode and verify exchanges cross PCIe as bf16 (MTP replies
    +6.8%, KLD +1.1%; `AR_BF16=0` for exact f32);
  - JEV off, API key file, alias `qwen3.8-27b`, `--slot-save-path`, 150 W cap, no `GGML_CUDA_P2P`;
  - binaries: `build-vision/bin` built 10-24 from f5fbd2542's code (snapshot `/work/bench/mtp3/bin-r1`),
    `400aac811` included;
  - rollback: the stop rule off on its own: restart with `STOP_ISO=0 STOP_ISO_THINK=0`. Binaries before the chat
    tables served every temperature: `/work/bench/mtp3/bin-think/bin/*` (or `LLAMA_SPEC_STOP_ISO_TEMP=0.5,0.9`);
    without the thinking tables: `/work/bench/mtp3/bin-prod-iso/bin/*`. The binaries before the rule:
    `/work/bench/forks/exp2/bin/*` (c73976485's code). Back to 10-23's production: copy
    `/work/bench/think/bin-prod-0d682248f/*` into `build-vision/bin` and restart with `AR_BF16=0 STOP_ISO=0`. The bf16
    wire off on its own: restart with `AR_BF16=0`;
  - after a reboot the installed Unraid script runs `/work/start-qwen-vision.sh`, which brings all of this back. It
    passes no env, so a lasting `AR_BF16=0` or `STOP_ISO=0` belongs in `/work/start-qwen.sh`'s default.
  - Numbers: README "Measured" (production build and settings, 2026-10-22; decode re-measured 10-24).
- Contributions back to the base: 12 PRs to Kmic-68/llama.cpp `p100-optimizations`, **#5-#16, still open on 10-24**
  (table in CHANGES §20). Each is a branch `p100/<type>-<name>` cut from d90ec36a3 and validated there (identical
  logits for every code PR, gate PPL 2.6074, full op suite with all but the first merged; FINDINGS "Validating PRs").
  Earlier: #4 (the softmax barrier, 03da0202b) was merged; Kmic's #3 (KaneQuah's fold prefill for q8_0/f16) was merged
  and superseded our own q8_0-K fold prefill.
- What's left:
  - Kmic's `tools/gate.sh` stops at a false "fattn-fold-sass.h is stale" on his tip and here (FINDINGS "Ideas from
    other forks"): run its perplexity step directly until the check hashes only the two SASS kernels.
  - Candidates for PRs to Kmic from CHANGES §21: the q4p spill fix (his q4_0 default is the config it speeds up), the
    sampler prefilter, the `seq_rm` bound and the FILL fix; from §22, the MTP draft's stop rule. Not opened.
  - `p100-docs/deploy/start-qwen.sh`'s comment on `STOP_ISO` still names the 0.5-0.9 window; since f5fbd2542 the
    chat tables serve every T > 0 outside the thinking window (the script's behaviour is right). Fix it with the next
    start-script change (the box's `/work/start-qwen.sh` must stay identical).
  - Adaptive-MTP leftovers (branch `mtp-adaptive-research`, FINDINGS "The MTP draft's stop rule"): lenient acceptance
    (approximate; needs a quality evaluation, never under a grammar) and n-max 16 aborting at warm-up
    (`GGML_ASSERT(obj_new)`, ggml.c). Deeper drafts and an exact-width verify were measured or simulated: they don't
    pay.
  - A guard for `LLAMA_THINK_BY_BUDGET` on Apertus, whose on/off reply starts are identical (the splice would leave
    thinking on): offered, not written. Production (Qwen3.8 only) is unaffected.
  - When Kmic merges or changes the PRs: merge his tip (never rebase) and repeat the checks of the 10-21 merge
    (CHANGES §18: op suite, PPL for both cache types, logits against the previous build, ABBA speed).
  - The container predates `--init` (zombie `llama-server` processes after a restart) until it is recreated with
    `run-container.sh`.
  - Deleting the fork branch `p100-optimizations` is pending the owner. The old `fix/ perf/ docs/` branch copies are
    still on the fork.
  - Older open items: CHANGES "Known gaps", and the parked slot-autosave design below (ask first).

## Branch `kv-q8q4-262k-vision1440-q8proj` (state 2026-10-18; production then, now part of `p100-optimizations-keoian`)

- Seven commits on tyler-port `67a7a1472` (CHANGES §18, FINDINGS "q8_0 K cache, 1440p images", QUICKSTART "1440p
  images"). Production on :8090 is `p100-docs/deploy/start-qwen-vision.sh` from `build-vision`: 262k, KV q8_0 K /
  q4_0 V, `-ub 1024`, Q8_0 projector on GPU1, images up to 1440p area, JEV off.
- Why: the operator wanted small text in screenshots legible (the 1024-token cap made 9-12 px text unreadable), the
  most accurate cache that fits at full context, and the fastest accurate projector; JEV was dropped to make room.
- Rollback on the box: `bash /work/start-qwen.sh.pre-vision` (tyler-port build-exp, JEV + q4_0/q4_0). The default
  `start-qwen.sh` gained `CTK`/`CTV`/`UB` switches with the old defaults.
- Host RAM 48 GB since 10-18: the script keeps 32 context checkpoints and a 24 GiB RAM prompt cache (branching back 3 turns
  in a 58k chat: 77.5 s -> 0.5 s). On a 32 GB host set `CTXCP=4 CACHE_RAM=8192`.
- Margins: GPU1 peaks at 16,101 of 16,384 MiB (1440p image at 259k). `-ub 2048` and image caps above 4096 do not
  fit at 262k. Required env: `GGML_CUDA_FA_VEC_MAX_Q=16`, `GGML_CUDA_FA_GEMM_MINQ=17` (set by the script).
- Open: image prompt time is now mostly the language model's prefill of 3,600 image tokens (~9.5 s); the vision PV
  GEMM wastes ~44% of its tiles at head size 72; q8_0 K decode at depth costs per byte read; better V types
  (q5_x, 4.6x lower KLD) need V support in q4p/fold and ~0.4 GiB per card more.
- Process lessons from this round:
  - Test servers on the shared :8090 got a temporary API key, because outside clients (an agent harness) poll the
    endpoint and skew or interrupt measurements; the real key file was left untouched and restored at the end.
  - Wait loops built on `pgrep -f <script>` match their own command line and never end: wait on PIDs or names.
  - Splitting a mixed working tree into commits with `git apply --cached --unidiff-zero` misplaced a pure-insertion
    hunk by 20 lines; stage hunks exactly by line number (or with context) and re-read each commit's diff.
  - Probe VRAM at full depth unless you have shown nothing grows with depth; compare cache types by KLD, not greedy.

## Branch `tyler-port` (state 2026-10-07; supersedes the sections below for this branch)

- `tyler-port` = `p100-optimizations` at `ae35056eb` + gated commits, pushed to Keoian/dual-p100-qwen-3-8-27b.
  Board: ASUS H270, GPU1 on a PCH Gen3 x4 link, **no P2P** (`cudaDeviceCanAccessPeer` 0) -> exchanges are
  host-staged; `-sm tensor` still wins. Power limit 150 W per card (see below).
- Round 1 (CHANGES §15, FINDINGS "tyler-port"): host-staged AllReduce and prefill exchange overlap without P2P; fast
  9-64-token batches; server checkpoint handling for chat turns; **byte-identical MTP restore**; slot files carry the
  context checkpoints. Base -> branch: chat turn on a 26k prefix ~1.6-2.0 s -> ~0.56 s; pp2048 404 -> 466; tg256
  26.7 -> 27.8 (at 180 W).
- Phase 3 (CHANGES §16, FINDINGS "tyler-port Phase 3", QUICKSTART "JEV"): **JEV-27B System 1** decisions via
  `POST /v1/decide` in a separate context on the shared model (LoRA bound to that context only); the full
  `test_set_30k` (29,955 rows) matches the published metrics (KL 0.0186 vs 0.0185, ECE 0.0012 vs 0.0011); System 2 byte-identical with decisions interleaved. Found and
  fixed on the way: a **use-after-free in the meta backend's subgraph arena** (`847fa0515`) and a **missing barrier in
  `fattn_gemm_softmax`** (`03da0202b`, NaN/nondeterminism with f16/q8_0 KV on long prompts; likely also the old
  "virtual devices NaN" gap). Upstreaming brief for Kmic: bench `PR_fattn_gemm_softmax_race.md`.
- Gates: PPL 2.6074 at every commit (band 2.6209 +/- 0.0199), full op suite for kernel commits.
- **Operator rules, until lifted:** no JEV with `GGML_CUDA_P2P=1`; P2P retests only while the operator watches;
  a P2P confirmation run (GGML_CUDA_P2P=1, small depth, while the operator watches) is
  saved for later. The operator approved 262k with JEV after the 10-07 deep test passed, and chose q8_0 for the served decision
  cache (the server's built-in default stays q4_0).
- **Power:** one host hard reset (no log) during a 262k deep prefill with both cards at 180 W; not reproducible at
  <= 64k; the same 262k run passed at `nvidia-smi -i 0,1 -pl 150`. The limit resets on reboot: set it at startup.
  tg256 at 150 W: 26.8 before the 10-08 Kmic merge, 30.2 after (CHANGES §17).
- Records outside the repo (build box): `/work/bench/HANDOFF.md` (start there), `LOG.md`, `SUMMARY.md`,
  `PHASE3_JEV_SYSTEM1.md`; JEV data/tools in `/work/jev/`.
- **Production (10-07):** JEV (q8_0 decision cache) + vision on GPU1 at 262k, after a full-depth deep test passed
  (CHANGES §16). Off switch on the build box: `JEV=0` for the start script.
- **Next steps:** (1) re-run the System 2 determinism/latency harnesses with JEV loaded (greedy reference, MTP restore,
  RAM cache, Ember TTFT; not run since Phase 3); (2) full `test_set_30k` eval with the served
  decision-cache type: DONE 10-07, matches published; (3) open the PR to Kmic for `03da0202b`: DONE, merged as Kmic's PR #4 (CHANGES §20); (4) temperatures: checked on the full set, ECE 0.0012 vs 0.0011, no refit needed; (5) Ember routing thresholds; (6) state-prefix reuse across
  decisions (latency); (7) round-1
  leftovers: slot autosave (ask first), fold tile for ~129-383-token batches.

### tyler-port: how to measure (lessons that cost time)
- Server latency: scripted identical prompts, ABAB order, warm the server first (the first request after a start is
  slower and can differ); compare the server's own `prompt_ms`, not whole-conversation totals (sampled replies vary).
- Sampler changes: natural-length replies, never `ignore_eos`, which is an EOG logit bias and turns the top-k
  prefilter off in both arms. Same-text arms can be paired per request from the server logs
  (`/work/bench/forks/paired.py`).
- Every kernel change: greedy compare on fixed prompts plus a small-batch KLD (`llama-perplexity -b N -ub N
  --kl-divergence`), not only llama-bench, which once blessed a kernel that read the wrong buffer.
- Intermittent NaN or drift: compare identical runs for bit-identity first; nondeterminism in a single-stream kernel
  chain means a race. memcheck misses host use-after-free and shared-memory ordering races; `CUDA_LAUNCH_BLOCKING=1`
  deadlocks the host-staged AllReduce; `MALLOC_PERTURB_=165` rules host-heap corruption in or out.
- Debug symbols without a rebuild: re-run `ninja -t commands bin/libggml-base.so.*` with `-g` into a separate dir and
  put it first in `LD_LIBRARY_PATH` (it wins over RUNPATH); the same trick A/Bs any library build.
- Tool shells: `pkill -x llama-server`, never `pkill -f` (it matches the shell's own command line). Long runs in the
  background with a synced step log (`p100-docs/deploy/diag/monitor.sh` for VRAM/power/PCIe/AER).
- Thermal/power: benches on hot cards read up to 20% low; alternate A/B order and cool between pairs.

### tyler-port: parked design, slot autosave across restarts (ask the operator before building)
Slot files are client-driven and the RAM prompt cache dies with the process. Sketched: `--slot-autosave NAME`
(+ `--slot-autosave-idle S`, ~10 s); restore at startup through the existing slot-restore path (it already restores
`.draft/.spec/.ckpt`); save when slot 0 has been idle S seconds and changed since the last save, via an idle callback
in `server_queue::start_loop`'s 1 s wait; snapshot to host synchronously (`llama_state_seq_get_data_ext`), write on a
background thread in the `llama_state_seq_save_file` format, temp file + rename, best-effort save on SIGTERM. Cost on
the reference box (share writes 228 MB/s, reads 538 MB/s): ~1.1 GB per save at 26k context (~5 s), ~6 GB at 256k
(~26 s) -- tens of GB written per day, so a longer idle threshold or save-on-shutdown only may be preferable.

## State (2026-10-01)

- Branch `p100-optimizations` (fast-forwarded from `goal/prefill300`), upstream last merged at
  `f46bc30cb` (2026-09-22). The public fork is at `781cb4220` until the user pushes.
- **Sampling is the model card's:** temp 1.0, top-k 20, top-p 0.95, min-p 0.
- Gates (OPTLOG 273): tg256 32.58, perplexity 2.6101, full op suite 16324/16324 on both GPUs, both
  SASS headers current (`u2cubin.py --check`, `facubin.py --check`, run by `tools/gate.sh`).
- KLD against an fp32 run: 0.001186 (09-26 release: 0.001215). Same top token 98.86% (98.80%).
- Prefill at 260k with vision loaded, `-ub 2048`, 1479-token question: ~153 / 155 t/s (09-26
  release, at its `-ub 1024`: 120 / 124). `pp2048` at 0 context 493 (release 385; stock ~250).
- Depth sweep (OPTLOG 274, QUICKSTART): fill 0 -> 260k in 24.9 min of prefill; MTP decode 54 t/s
  at 2k, 29-35 at 260k.
- `qwen-server` runs `-ub 2048` with or without `--mmproj` (compact mask, OPTLOG 248). GPU0 min
  free at 262k with vision: 732-792 MiB, with the desktop's ~392 MiB on it.
- Round 4 (prefill at depth, OPTLOG 236-272) is written up in CHANGES §14 and, with the
  analysis and every dead end, in `p100-handoff/GOAL-PREFILL300.md`.
- **Next steps:**
  1. The user pushes `p100-optimizations` and refreshes the release bundle
     (`p100-handoff/release-sync/refresh-build.sh 2026-09-26`, then `sync.sh`).
  2. Prefill at depth is at the 175 W power cap: the three big kernels need less energy per flop
     (fewer shared-memory reads per HFMA2), not fewer stalls. Estimated ceiling ~176 t/s at 260k.
     Cheap leftovers: one transfer for the 768 checkpoint copies (~60 ms), preload cuBLAS
     (147 ms on the first request), the final partial batch and its syncs (~220 ms).

## What changed on 2026-09-24/25 (OPTLOG 201-218)

- Sampled MTP drafts verified with the speculative-sampling rule (lossless; +15% tokens per cycle
  at 2k). `LLAMA_SPEC_SAMPLE_TEMP=1.0`, `LLAMA_SPEC_DRAFT_TOPK=20`, set by qwen-server / depth-bench.
- fp16 q6_K verify matvec (`mmvq-f16.cu`), 2-5 columns: -10..16% per call, and closer to fp32
  (verify-path KLD against an all-fp32 run 0.00354 -> 0.00274).
- 5-token q4p attention over half the GQA group per block, DPT 8 for 15/18 rows: -16% per call at 260k.
- The MTP catch-up stores K/V only (-4 ms per cycle at 260k).
- CUDA graphs for single-token graphs only (`GGML_CUDA_GRAPHS_PRE_VOLTA=3`, now the default).
- Host: one sync per split and async uploads for user inputs, no per-cycle prompt copy, an SSE2
  one-pass mask fill.
- Delta-net: the recurrent-state and conv-state gathers are read in place by their consumers, and
  the 5 conv-state rollback snapshot copies per layer run as one launch.
- fp16 verify matvec on every q6_K matrix of >= 16 rows (split K below 256 rows); 6 blocks per SM.
- Draft cutoff at depth 0.22 (was 0.3): neutral within noise; left at 0.22.
- Tools: `p100-handoff/tools/mmvq-harness/` (a kernel in ~7 s), `tools/quick.sh` (a verify pass in
  35 s), depth-bench `--server-prefix` (nsys/pmp), `LLAMA_TL=1` host timeline, `GGML_CUDA_GRAPH_DEBUG=1`.

## Where the MTP cycle goes now (nsys, 260k, warm, ~128 ms)

GPU0: attention 52 ms (the 5-token verify calls 40, the 1-token draft steps 4.6), matvecs 47 (fp16
path 37, integer 8, including 4.7 of draft-step vocabulary projection), ~1900 small-op launches
~10 ms, idle 14 ms (~4 at the 128 tensor-parallel exchanges, the rest host work between graphs).
A draft step is ~1.1 ms of host (the 260k-column mask fill and upload, input copies) plus ~2.5 ms of
GPU. At 2k a verify pass is ~51 ms (quick.sh), ~30 of it the fp16 matvec.

Levers measured and not worth it (details in OPTLOG): fp16 in attention (loses accuracy against
the fp32 q4p), 2-blocks-per-SM attention configs, draft vocabulary caps, top-p on the draft
distribution, n_max 3/5, p_min, p_cum thresholds, all-graph CUDA graphs (freezes the fixed-width
verify's token count). Open: a persistent device-side KQ mask (~1.5-2% at 260k), fusing the
delta-net state gather into its kernel (~1% at 2k), overlapping draft-step host prep with the GPU.

nsys works on Pascal (2022.4). If the importer fails, run
`/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i X.qdstrm` and then
`nsys export -t sqlite`. `perf` is locked here (`perf_event_paranoid` 4). For a host CPU profile
use `tools/pmp/`, an LD_PRELOAD sampler; its header has the usage.

## Open threads

1. **Per-GPU enqueue threads in `ggml-backend-meta.cpp`.** One host thread enqueues GPU0's subgraph
   before GPU1's, so GPU0 waits ~200 us for GPU1 at the first all-reduce of every graph. Two
   threads would fix that and double the enqueue rate, which matters in the draft steps. It needs host-side ordering
   between the two threads at every all-reduce (the peer copy's event must be recorded before
   the other side's stream waits on it). Worth maybe 2-4% at short context.
2. **Chain the MTP draft steps on the device.** Each step waits on the host for the previous
   step's token and hidden state. Feeding them device-side (the top-1 candidate and `h_nextn`
   into the next step's inputs), with one read-back at the end, would save ~4 ms per cycle at 2k
   and ~10 at 260k. The meta backend has no `cpy_tensor_async`, and the logits are
   vocabulary-split across the GPUs, which is the tricky part. The per-width graph cache this
   thread used to describe is no longer needed: the fixed-width verify (8736a7ef3) removed the
   rebuilds another way.
3. **A possible timing-dependent result in an earlier binary.** One build gave three different
   260k texts across normal and profiled runs. The current build agrees with itself, async and
   under `CUDA_LAUNCH_BLOCKING=1`, on every case tried. See OPTLOG 192. A repeat-until-diverge
   test at 260k would settle it.
4. **q4p occupancy.** Every variant runs one 256-thread block per SM (233-255 registers). At 30
   rows the PV accumulators alone are 120 registers. A design that keeps fewer rows per thread,
   or splits PV across two blocks, might approach the ~70% FFMA efficiency the instruction mix
   allows, against ~44% now. At 30 rows it is latency-bound (same time at 1189 and 1328 MHz),
   so more loads in flight should matter more than fewer instructions. Measure in the server,
   not only in test-backend-ops (OPTLOG 190).
5. **Short-prompt prefill (time to first token per chat turn).** A 9-127 token prompt takes
   ~0.5 s of GPU at any depth: cuBLAS's 256x128-tile HGEMM at ~4 TFLOPS on a skinny GEMM, plus a
   full f16 dequant of the weights each call (OPTLOG 195). A narrow-tile kernel would help. Any
   replacement must match ALGO6's accuracy (attempt 153). Largely done on tyler-port: fp16 mat-vec chunks
   (`0ff1cd3b1`) and 32/64-column fold tiles (`a8847f5ed`), CHANGES §15.
6. **`GGML_CUDA_DEVICES` above the physical GPU count isn't reproducible** (NaN in 4 of 8 runs at
   3 virtual devices). It follows the GEMM attention path. It's debug-only, and two physical GPUs
   are bit-stable. OPTLOG attempt 153 §8c. Probably the softmax `red[]` race fixed in `03da0202b` (not re-run).
7. **Fuse the all-reduce widen into the ADD** (~+1% prefill). It needs an accumulating-copy path
   in `ggml-backend-meta.cpp`.
8. ~~`gated_delta_net` in prefill~~: done, chunked (`77e05601d`, +3.8%).
9. ~~Deepest prefill regressed ~10%~~: obsolete; 260k prefill is now ~153 t/s.

## Closed: don't re-sweep without new information

| axis | result |
|---|---|
| attention occupancy (384 threads, occupancy 2-4, doubled warps) | neutral or worse, three ways |
| `nbatch_K` 64/128/256 | 128 is best on narrow tiles; 256 is +44% |
| `nbatch_fa` 32/64/128 | 64 |
| Q-column reuse (`cpw` 1 vs 2) | identical to 0.005% |
| wide loads in the q4_0 dequant | scalar wins by 7% |
| mmvq register caps, unrolling, prefetch pipelines | all worse; not latency-bound |
| double-buffered mmvq staging | shared memory then limits occupancy |
| internal AllReduce on Pascal | −17% (PCIe); re-measured exact (no BF16 wire) in OPTLOG 189: still ~1% slower |
| CUDA graphs for small batches only | no gain under `-sm tensor` (OPTLOG 186) |
| 5-column q6_K matvec: fp32 or fp16 rewrites, I2F removal | at its floor (OPTLOG 179, 187) |
| MMQ on Pascal | no DP4A, ~4x ALU disadvantage |
| `-sm layer`, for prefill or decode | tensor split wins both |
| f32 GEMM output | halves GEMM throughput |

## Notes for kernel work

- The mmvq geometry and staging live in `mmvq.cu`, under `GGML_CUDA_MMVQ_PASCAL`. Build times:
  `mmvq.cu` alone is ~90 s, while touching `vecdotq.cuh` rebuilds ~200 instances (~25 min).
  Put sweep knobs in the source, not in `-D` flags.
- Fast isolated benchmark (a few seconds):
  `test-backend-ops perf -o MUL_MAT -b CUDA0 -p 'type_a=q6_K,type_b=f32,m=4096,n=1,k=14336'`.
  Always confirm on the real model. The isolated shape has pointed the wrong way before.
- Casting a shared-memory pointer through `uintptr_t` loses the address space, and ptxas silently
  emits generic loads. Derive aligned pointers with `char *` arithmetic.
- Correctness harnesses and proofs are in `p100-handoff/tools/` (bit-exactness replays for the
  fastdiv, DP4A, q4_0 dequant and norm changes). `p100-handoff/VERIFICATION.md` is the numerical
  audit.

## Round 4, 2026-09-27 to 10-01 (resume here)

Prefill at depth: 260k 120 -> ~153 t/s, `pp2048` 385 -> 493, all exact. See CHANGES §14,
`p100-handoff/GOAL-PREFILL300.md` and OPTLOG 236-273. Fast loops for the fold attention SASS:
`p100-handoff/tools/sass-gemm/fa-exp.sh` (~40 s per experiment); prefill A/B screen:
`tools/ab-pp.sh` (~1 min).

## Round 3, 2026-09-26

Ideas from other projects (six research agents; OPTLOG 230-235). Kept, all exact: one-kernel P2P
AllReduce (GGML_CUDA_AR_P2P), FFN gate+up+SWIGLU fusion in mmvq-f16 (GGML_CUDA_FUSE_FFN_GLU), lazy
sched hash reset, block verification of MTP drafts (LLAMA_SPEC_BLOCK_VERIFY, =2 prints the paired
gain). Gates: tg256 31.91 (warm), PPL 2.6096, full ops 16324/16324 x2, KLD identical to shipped.
Server vs the 192fd789a build: cycle 2k -5%, 260k -6%. Next candidates, in order: Lamport push
AllReduce with the ADD+RMS_NORM epilogue fused (TRT-LLM); q6_K LOP3 repack (big, high effort).
Parked: PR #1 on the public fork (Q4_K vdr 4 by mewsian) -- revisit with a Q4_K_M model.

## Round 2, 2026-09-26

Hot cards (70-78 C), production flags, depth-bench --restore --extra-chars 4100, 2 seeds x 2 questions:

| | old build (72-77 C) | now | goal |
|---|---|---|---|
| 260k prefill | 87.7-91.6 t/s | 101.6-107.9 t/s | 95-100 (met) |
| 2k decode | 68.6-69.7 ms/cycle | 49.4 t/s avg (58-63 ms/cycle) | 50 (~1% short, within heat noise) |
| 260k decode | 116.8-120.6 ms/cycle | 29.4-30.2 t/s avg (95-104 ms/cycle) | 30 (at the line) |

Merged today (OPTLOG 226-228): MTP draft head over the first 81920 tokens in q4_0
(`LLAMA_MTP_DRAFT_VOCAB`, 0 = off; draft step 2.05 -> ~1.4 ms), fold prefill chunk 1024 (VRAM for
the draft head; GPU0 min free at 262k with prefill now ~642-646 MiB, was 696-702), goal2/vattn2
(q4p verify attention 2.44 -> 1.93 ms/call at 260k; verify KLD 0.001127), goal2/ops2 + ops2b
(bit-exact launch fusions, verify pass ~-1.6 ms). Tried and dropped: CUDA graphs for the verify
(slower, +100 MiB), big-path matvec prefetch/x-reorder (not faster), n-max 3/5 (model says worse).

Gates on this build: tg256 31.17 t/s, PPL 2.6096, FA eval OK (full suite not run).

Next: the 177 TP exchange ADDs per verify pass (fold into the next fused add+norm across meta
subgraphs); the q6_K 5-col matvec (~36 of ~46 ms/pass) is still the bulk and resisted two attempts.
Cleaned up 09-26: goal worktrees removed, merged goal*/ branches deleted. Kept: branch goal/pfattn
(6ba2952d5, P layout for the fold prefill, op ~276 -> 270 ms, FA eval OK, not measured end to end;
drop its OPTLOG hunk if merged) and stash@{0} (its old WIP). qwen-server now runs -ub 1024
(OPTLOG 229: -ub 2048 no longer fits at 262k with the draft head).

## Paused 2026-09-25 late (resume here)

Goal the user set: hot cards, production serving flags, MTP on: 260k prefill (1k chunk)
95-100 t/s, 2k decode 50 t/s, 260k decode 30 t/s, math no less accurate.

**Status: prefill met; decode not proven on hot cards.**

| (1479 new tokens, depth-bench --restore) | old build (72-77 C) | new, 62-66 C | new, 79 C | goal |
|---|---|---|---|---|
| 260k prefill | 84.8 t/s | 122.5 | 101.7 | 95-100 (met) |
| 260k decode | 26.9 t/s (123.7 ms/cycle) | 33.8 (101.4) | 25.9 (126.2) | 30 |
| 2k decode | 42.6 t/s (71.7 ms/cycle) | 50.0 (60.0) | 38.6 (75.0) | 50 |

Card temperature moves decode by +/-20%, more than the gains. What the kernel-level data
supports: 2k verify pass -3% (~2 ms/cycle); 260k q4p verify call -10.7%, draft call -5%
(~5 ms/cycle). Accuracy: prefill KLD vs all-fp32 0.00121 (was 0.00125); verify path (-ub 5,
3 chunks) 0.00117 (was ~0.00114, within noise).

Merged into p100-optimizations today (OPTLOG 223-225, not pushed):
- 223 (goal/dec, bit-exact): fp16 q6_K matvec scales once per warp; 7 blocks/SM when it saves a
  wave; host trims (fusion-check cache `GGML_CUDA_FUSE_CACHE`, device cache, 1-token attention in
  graph mode 3); residual ADD fused into RMS_NORM+MUL (`GGML_CUDA_FUSE_ADD_NORM=0` off).
- 224 (goal/vattn): q4p PV reduce-scatter at 15/18 rows, L2 prefetch off at 15; 1-token kernel
  reduce-scatter + 2 blocks/SM (`GGML_CUDA_Q4P_OLD=1` = previous config, for A/B).
- 225 (goal/pfattn): fold-path prefill attention in fattn-gemm.cu, default on for a q4_0 cache
  (`GGML_CUDA_FA_FOLD=0` = old cuBLAS path; `_CHUNK` 2048, `_SPLIT` 2). Op at kv=262144 nb=1024
  ~318 -> ~270 ms. Scratch ~165 MB/GPU at -ub 1024 (was ~71); GPU0 min free at 262k 696-702 MiB.

Not done / next:
1. **Measure decode fairly first**: strict ABBA old/new at one temperature (or pre-heat both
   arms), several seeds. depth-bench.py now has `--extra-chars N` for prefill at a snapshot's depth.
2. Not merged: goal/pfattn `6ba2952d5` (P layout; FA eval 4019/4019, not measured end to end).
   It also appends its own "Attempt 223" to OPTLOG.md; on merge, drop that hunk (attempt 225
   already covers the fold prefill work) and add the P-layout step to 225.
   Not committed: vattn's SEL-free reduce-scatter + mask preload, `wt-vattn/vt/q4p-c6.cuh`
   (op test nb=5 2405 -> 2278 us; needs FA eval + in-server profile).
3. Decode ideas from the dec agent, unstarted: fuse the delta-net prologue (~190 launches/pass,
   ~0.5 ms), fold the exchange ADD into the residual ADD, draft top-k sampling on the GPU.
   The fp16 matvec is ~90% instruction-bound (~44 instr per 8 weights, 20 are math).
4. Gates not run on this build: tg256, full op suite (`./tools/gate.sh --full`). Then the README
   results table and the release refresh (`qwen-server` still runs d3a650552).
5. Cleanup: worktrees `/mnt/fast/p100-scratch/wt-{pfattn,vattn,dec}` (NTFS: git shows every file
   as a mode change; commit by explicit path only), branches goal/*, a stale stash in wt-pfattn.
   Agent reports: /mnt/fast/p100-scratch/goal-*.md, status-*.md; results in .../goal/*.jsonl.
