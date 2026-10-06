# Findings

What worked, what didn't, what transfers to other Pascal cards, and the measurement mistakes
that cost the most time. The change list is [CHANGES.md](CHANGES.md); the raw record is
[`../OPTLOG.md`](../OPTLOG.md).

## What binds on a P100

- **The decode matvec is bound by global memory instructions, not bandwidth.** A variant with
  fewer total load/store operations was 5% *slower* because it swapped shared loads for global
  ones. Global loads cost far more than shared ones.
- **The activation, not the weights, is the bottleneck.** Every block re-reads the q8_1
  activation, so its cost scales with block count. Staging it in shared memory was the largest
  single win.
- **Occupancy isn't the limiter.** Forcing fewer registers makes it monotonically slower, even
  with zero spills. So did three separate attempts to raise occupancy in the attention kernel.
- **Achievable bandwidth is ~605 GB/s per card**, not the 732 on the spec sheet. Decode reaches
  ~490 GB/s effective. ECC costs nothing on HBM2, so leave it on.
- **Long prefill is bound by power, not instructions.** The fold GEMM and attention kernels run
  the cards at their 175 W cap, where the clock settles near 1290 MHz and they reach ~13 TFLOPS of
  the fp16 pipe. A lean test kernel on random data does 15.4 TFLOPS in the same power, so the
  remaining gap is energy per flop (shared-memory and register traffic per multiply-add), not
  stalls. Removing yields and bank conflicts measured flat or slower. Benchmark these on random
  data: on zeros the same kernel draws less power and reads 17-18 TFLOPS.
- **Alignment is a wall.** Every quantized block is 2 mod 4 bytes, so a 4-byte quant word needs
  two loads. Four designs moved that cost around, and none removed it. Only repacking the weights
  would.

## What worked

**Dequantize the KV cache into shared memory, not global.** Upstream converts a quantized KV cache
to f16 in full on every attention call. At 262144 context that's 4.15 ms per call, ~66 ms per
token, to re-convert a cache that changed by a few positions. Dequantizing each tile as it's
staged removed it and freed 512 MiB per card. The result is numerically equal to upstream's for
every finite scale. It isn't bit-identical: some zeros differ in sign, and an infinite scale
gives a signed infinity where upstream gives NaN.

**Fit the tile to the batch.** The attention tile ladder offered widths of 1, 2, 4 and 8 queries
per KV head. A 5-token MTP verify was padded into two 4-token tiles, so 37% of the work was
padding. The width has to be a multiple of the GQA fold (6), and columns per warp must be a
power of two. 36 satisfies both.

**Bound fp16 accumulation error.** The attention output accumulated over the whole KV cache in a
`half2` register, and the error grew as the square root of the context:

| KV length | `half2` accumulator | per-tile fp32 fold |
|---|---|---|
| 4096 | 3.3e-06 | 2.9e-06 |
| 65536 | 2.8e-05 | 3.1e-06 |
| 262144 | not measured | 3.0e-06 |

Folding into fp32 once per tile keeps the fast inner loop and costs 2.4%. The fix often suggested
for P100s, extending the sm_61 `FAST_FP16_AVAILABLE` exemption to sm_60, converts far more to
fp32. It doesn't fit the 36-wide tile in shared memory, and costs 17-90%.

Don't also "fix" `fattn-vec.cuh`. Its `half2 VKQ` declaration looks like the same bug, but it's
compiled only for HIP. CUDA builds already accumulate in `float2`. Trying it cost 16% of decode
for nothing.

**Don't materialise the causal mask.** The f16 attention mask is n_kv × n_tokens: 1 GiB per card at
262k context and `-ub 2048`, two-thirds of the compute buffer. With one sequence in position order,
query t sees exactly the first L₀ + t cells, so a list of prefix lengths carries the same
information. That freed the room for `-ub 2048` with vision, bit-exact.

**Hide the tensor-parallel exchange.** In prefill each card waited for the other's partial sums,
about 7% of the time (nsys). Splitting the matmul into token chunks and sending each chunk while
the next one computes recovered most of it (+4.5%), and filling what's left of the wait with the
next layer's weight dequant gained a bit more, both bit-identical.

**Pick the cuBLAS algorithm on Pascal.** The default picks a long-chain fp16 accumulator from
~256 rows up. `ALGO6` is 10x more accurate at every shape measured, and faster at 512-1024 rows.

**CUDA graphs work on Pascal.** Upstream disables them by architecture. They're +6.7% on the
speculative path but −2% on plain decode, so they ship opt-in.

## What was reverted, and why it's interesting

| attempt | result | lesson |
|---|---|---|
| KQ dot product accumulated in `half2` | −20% time, reverted | Perplexity couldn't tell (2.6097 either way), but per-lane error rose ×1.22, exactly as predicted. Both forms round twice; the second rounding just lands on a larger value. *Where* rounding happens matters, not only how often |
| internal AllReduce on Pascal | −17% | it assumes NVLink, and these are PCIe cards |
| doubling the shared-memory stage | slower | shared memory then limits occupancy (14 → 9 blocks/SM) |
| wide loads in the q4_0 dequant | +7% time | a q4_0 block is 18 bytes, so its quants are never 8-byte aligned |
| vec kernel thread remap | abandoned | passed 3949/3949 op tests and still produced NaN in real inference. Run perplexity first, not last |
| MoE `mmid` threshold tuning | no effect | this model is dense. Check the path runs before tuning it |
| `n_draft` 4 → 6 at depth | +1.9% | acceptance falls from 81% to 70% |

## Two races the op suite couldn't see

Both came from the fork's own work, both are fixed, and both passed all ~14600 `test-backend-ops`
cases for weeks. The suite runs ops one at a time with host syncs between them, which is exactly
the condition under which a cross-stream race can't happen.

1. The GEMM attention softmax wrote probabilities over scores it was still reading. It corrupted
   a few rows per long prompt and could NaN in fp16.
2. A tensor-parallel peer copy could overwrite the all-reduce buffer before the other card's ADD
   had read it. This hit decode and MTP. With the race forced, acceptance fell from 79% to 43%.

Races need repeated unsynchronised runs, deliberate delay injection, or an in-op self-check.

## Serving at long context

**Checkpoints only help if the next query extends them.** Qwen3.8 has 48 recurrent layers, and a
recurrent state can't be rewound. Restoring a saved slot and resending the original prompt still
reprocesses everything, because the saved state includes generated tokens. The recipe: prefill
with `"n_predict": 0`, save, then restore and send the same text plus a suffix. At full depth that
processed 10 tokens instead of 259,229.

## How the measurements lied

These cost more time than any kernel bug.

1. **The wrong corpus.** The perplexity band belongs to one file. Another reads 2.7566 on every
   build, stock included, and it was twice mistaken for a regression. The gate now lives in
   `tools/gate.sh`, so the corpus can't drift from the number.
2. **Warm cards.** One build read 30.8 cold and 24.9 straight after a perplexity run. Discard a
   warmup, and interleave A/B within one session.
3. **`-n 128` at long context** amortises a 2-3 s first-token cost over too few tokens. It
   produced 12.2 t/s when the truth was 21.5. Use 512 or more.
4. **A stale constant.** An old note said bandwidth was ~196 GB/s. The real figure is ~490, and
   the wrong one "proved" a target impossible.
5. **A short prompt's VRAM peak.** "~250 MiB of headroom" was measured at 8% context fill. At a
   full prompt the margin was negative. Take the *minimum* of free memory over a whole long run;
   early samples look safe and mean nothing.
6. **The wrong library.** Copied builds load `build-opt`'s library through RUNPATH. Two different
   kernels once agreed to seven digits, which should have been the tell.
7. **Stale objects.** Editing a CUDA header doesn't always rebuild the template instances that
   include it, so a benchmark measured the previous binary. Touch the includers after a header edit.
8. **Skipped cases.** `test-backend-ops perf` silently skips large-KV cases when VRAM is taken,
   and still prints "passed".
9. **A negative result is only valid for its workload.** CUDA graphs were rejected twice on plain
   decode before they measured +6.7% on the speculative path.
10. **A clean merge isn't a correct merge.** Upstream's new `launch_fattn` parameter shifted one
    of our arguments into a `bool` with no compiler warning. See CHANGES §9.

## tyler-port: a board without P2P, chat turns, and exact restore (2026-10-04 to 10-06)

Measured on an ASUS H270 board whose second x16 slot is a PCH Gen3 x4 link; the cards cannot reach each other
(`cudaDeviceCanAccessPeer` 0 both ways). CHANGES §15 lists the commits.

**What worked, and why.**
- *Without P2P, small exchanges are latency, not bandwidth.* A staged peer copy costs ~33 us per 4 KB, so decode's
  many small AllReduces dominate. A host-staged AllReduce with pinned buffers and a spin wait (upstream's
  allreduce.cu idea, adapted for pre-Volta) took decode from 26.66 to 27.39 t/s. For prefill, overlapping the exchange
  with the matmul (Kmic's chunked sender, previously only reachable with P2P) was worth +22% at pp512.
- *Short batches were the real cost of a chat turn.* A 10-40-token turn went through the 128-column fold GEMM tile,
  ~0.43 s per pass of the model whatever N was. Running q6_K/q5_K as fp16 mat-vec column chunks (<=28 columns now) and
  giving the fold GEMM 32/64-column tiles (<=64) cut it to what the column count costs. Both keep or improve accuracy
  (the narrow tile is bit-identical to the 128-column u2 kernel; ub32 KLD went down 6% with the cap at 28).
- *Server-side, the turn's time was mostly checkpoint handling.* Not re-copying a just-restored checkpoint, not
  splitting the prompt at the last user message, and recycling checkpoint buffers took the server's prompt time for a
  short turn from ~1.6-2.0 s to ~0.56 s. The biggest single item was a host setting: with THP `defrag=madvise`,
  `MADV_HUGEPAGE` on the 150 MiB checkpoint buffers triggered synchronous compaction (~200 ms stalls per turn), and
  as memory fragmented over a day every server number drifted 745 -> 1050 ms. Dropping the advice fixed both.
- *MTP determinism* (CHANGES §15): a speculative draft that pairs tokens with the target's hidden rows has state of its
  own that must travel with every snapshot of the target (checkpoints, slot files, prompt cache). Symptoms of missing
  it: greedy output that depends on earlier requests and repeats exactly across restarts. A first patch had been
  applied to the EAGLE3 driver instead of the MTP one, which is why it "had no effect".

**What failed, and why it's interesting.**
- *Reading the end-of-prompt checkpoint from the MTP rollback snapshot.* With MTP the target keeps per-token recurrent
  snapshots for the last 4 tokens of every batch, so the n-4 checkpoint can be read after one decode of the whole prompt
  instead of a separate decode of the last 4 tokens: -4% prompt time, and the snapshot is right (bit-exact in a single
  ubatch; at 2.6k tokens it is closer to a separately decoded state than two batch splits are to each other). Rejected:
  the first answer to a prompt then comes from a one-pass state and every replay from snapshot + 4 tokens, so replays no
  longer reproduce the first answer byte for byte. The separate decode is what makes both paths compute the same thing.
  Also: in hybrid memory `seq_pos_min` is the recurrent cell's position, so a checkpoint taken "back in time" must move
  `pos_min` back too, or the restore lookup silently never uses it.
- *A faster kernel that changed the text.* fp16 q8_0 mat-vec: +12% pp9 in llama-bench, slower over a 10-turn server
  run; int8 chunking of every type: faster, KLD +87%. Chunk width 8 instead of 5: slower, the fp16 kernels are
  ALU-bound per column.
- *Bonsai donor items* mostly did not transfer: its cuBLAS algo picks are superseded by the fold GEMM (accuracy), its
  f16 split-KV flash-decoding is no faster than q4_0 at 26k context, and two items were already in Kmic's tree.

**How the measurements lied (new traps).**
- llama-bench blessed a kernel that read the conv state from the wrong buffer (+1%, greedy diverged at character 70).
  Every kernel change needs a greedy compare and a small-batch KLD, not just speed.
- Kmic's plain `gemm_fold_kernel` is *not* bit-identical to the u2 kernel it backs up: its integer fold keeps the sums
  at 2^-112 scale, so outputs near zero land in fp32 subnormals (PPL at ub48 6.7149 vs 6.6833). Mirror u2's arithmetic.
- Whole-conversation totals vary with the sampled replies; compare scripted, identical prompts (`ember.py ttft`) in
  ABAB order. The first request after a server start is slower and can differ.
- A phone chat app that hides `reasoning_content` showed nothing for a minute and aborted on image questions; the
  server was fine. Serve chat apps with `--reasoning off` (requests can opt back in) and cap image tokens.

## What transfers to other Pascal cards

- **Direct KV-cache dequantization** helps any pre-Volta GPU with a quantized KV cache, and more
  as context grows. It's the change most worth proposing upstream.
- **cuBLAS algorithm choice.** Check it on any pre-Volta card before trusting the default.
- **CUDA graphs** on Pascal help workloads that issue many small kernels.
- **The DP4A emulation** applies to all of sm_60. sm_61 cards (GTX 10-series, P40) have real DP4A.

What doesn't transfer directly: the tile configurations assume head size 256 with GQA 6, the
draft-length advice follows from that tile geometry, and the AllReduce result is a PCIe result.

## Where the remaining time is

**Prefill at 260k**, one 1479-token question (GPU0): fold attention ~5.8 s (PV 2.9, QK 2.9), the
weight GEMM ~2.8 s, and ~0.5 s of everything else (the last partial batch, syncs, dequant,
checkpoints). At the 175 W cap the exact-math ceiling is about 176 t/s against 153 measured.

**Decode at depth.**

Plain decode at 229k context is 46.6 ms per token: 22.9 for weights and everything else, 23.7 for
attention. At that shape an f16 KV cache runs at the bandwidth limit (480 GB/s), while q4_0 manages
99 GB/s despite reading a quarter of the bytes. The gap is dequant work and the shared-memory round
trip. Closing it needs a kernel that dequantizes K into registers, which is new work, not tuning.
