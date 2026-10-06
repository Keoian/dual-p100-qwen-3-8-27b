# Handoff

Where the work stands, and what's worth doing next. For the project rules and gates, see
`CLAUDE.md`. For the results and changes, see `p100-docs/`.

## Branch `tyler-port` (state 2026-10-06; supersedes the sections below for this branch)

- `tyler-port` = `p100-optimizations` at `ae35056eb` + 14 gated code commits + docs, pushed to
  Keoian/dual-p100-qwen-3-8-27b. Board: ASUS H270, GPU1 on a PCH Gen3 x4 link, **no P2P**
  (`cudaDeviceCanAccessPeer` 0) -> exchanges are host-staged; `-sm tensor` still wins.
- What it adds (CHANGES §15, FINDINGS "tyler-port"): host-staged AllReduce and prefill exchange
  overlap without P2P; fast 9-64-token batches (fp16 mat-vec chunks <=28 columns, 32/64-column fold
  GEMM tiles, bit-identical to u2); server checkpoint handling for chat turns (no re-copy, no extra
  split, no MADV_HUGEPAGE, buffer recycling); **byte-identical MTP restore** (the MTP draft's
  hidden-row carry-over is matched by position and saved with checkpoints, slot files and the RAM
  prompt cache); slot files carry the context checkpoints.
- Gates: PPL 2.6074 on this board at every commit (band 2.6209 +/- 0.0199), full op suite for every
  kernel commit. Base -> branch: chat turn on a 26k prefix ~1.6-2.0 s -> ~0.56 s server prompt time;
  pp16 37 -> 80; pp2048 404 -> 466; tg256 26.7 -> 27.8.
- Serving (QUICKSTART "Boards without P2P"): the server command plus `-ctxcp 4`, and for chat apps
  `--alias --reasoning off --image-max-tokens 1024 --api-key-file`, vision with `mmproj-F16.gguf`.
- Records outside the repo (build box, `/work/bench/`): `HANDOFF.md` (start there), `SUMMARY.md`,
  `LOG.md` (every attempt), `RESULTS.md`, `patches/` (rejected diffs), A/B and repro scripts.
- **Next steps:** (1) optional idle-time autosave of the slot so the last cache state survives a
  restart (designed, not built: ~1 GB written per 26k-context turn, ask first); (2) or restore a named
  Ember prefix slot at startup; (3) perf leftovers: fold tile for ~129-383-token batches, a 16-wide
  tile for 9-16 tokens, a replay-safe way to drop the ~75 ms end-of-prompt checkpoint decode.

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
   replacement must match ALGO6's accuracy (attempt 153).
6. **`GGML_CUDA_DEVICES` above the physical GPU count isn't reproducible** (NaN in 4 of 8 runs at
   3 virtual devices). It follows the GEMM attention path. It's debug-only, and two physical GPUs
   are bit-stable. OPTLOG attempt 153 §8c.
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
