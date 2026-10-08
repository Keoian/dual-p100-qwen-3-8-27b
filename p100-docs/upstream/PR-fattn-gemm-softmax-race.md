# PR brief: missing `__syncthreads()` in `fattn_gemm_softmax` (pre-Volta cuBLAS-GEMM attention)

Hand this file to a fresh Claude Code instance (status 2026-10-07: PR not opened yet). It contains everything needed to open the PR: where the code lives,
the bug, the one-line fix, the exact patch, the evidence, and the PR text.

## 0. Where the PR goes (read first)

- The buggy file `ggml/src/ggml-cuda/fattn-gemm.cu` is **not in ggml-org/llama.cpp**. It was added in
  **Kmic-68/llama.cpp, branch `p100-optimizations`** (commit `738022bda`, author Kaden, 2026-09-01,
  "cuda: cuBLAS-GEMM flash attention for pre-Volta long context"). So the PR target is
  **`Kmic-68/llama.cpp` → base branch `p100-optimizations`**, not ggml-org.
- The fix is already committed and pushed in Tyler's fork: **`Keoian/dual-p100-qwen-3-8-27b`, branch `tyler-port`,
  commit `03da0202b`** ("cuda: barrier before the GEMM-attention softmax reuses its shared reduction slots ...").
- The patch was checked with `git apply --check` against `kmic/p100-optimizations` @ `ae35056eb` (its head on
  2026-10-06): **applies cleanly**. `tyler-port` has other commits on top of Kmic's branch, so do NOT open the PR from
  `tyler-port`. Make a branch from `kmic/p100-optimizations` with only this change (steps in section 6).
- `Keoian/dual-p100-qwen-3-8-27b` is a GitHub fork of Kmic's fork, so a PR from a new branch there to
  `Kmic-68/llama.cpp:p100-optimizations` works directly.

## 1. The bug in one paragraph

`fattn_gemm_softmax` (one block of 256 threads per query row and head) does a block-wide max reduction through
`__shared__ float red[8]`. Thread 0 writes the result to `red[0]`, there is a `__syncthreads()`, and every thread reads
`vmax = red[0]`. Pass 2 then computes the probabilities and does a block-wide **sum** reduction **through the same
`red[]`**, starting with `if (tid % WARP_SIZE == 0) red[tid/WARP_SIZE] = sum;`. Nothing makes the threads wait between
"read the max from `red[0]`" and "warp 0 writes its partial sum into `red[0]`". If warp 0 finishes its pass-2 loop
before some other warp has executed `vmax = red[0]`, that warp reads warp 0's partial sum as the row max. It then
computes `p = exp(v - m)` with the wrong `m` for all of its keys:

- with `GGML_CUDA_FA_GEMM_PREC=32` (fp32 scores/probabilities): small, run-to-run nondeterministic errors;
- with the default fp16 path: `p` can exceed 65504 when stored as half, which gives `inf`, then `inf * 0` / `inf - inf`
  gives **NaN** in the attention output, and the NaN spreads to the logits.

It is a timing race, so it is intermittent: the same input fails on one run and passes on the next.

## 2. The fix (5 lines, 1 of them code)

```diff
--- a/ggml/src/ggml-cuda/fattn-gemm.cu
+++ b/ggml/src/ggml-cuda/fattn-gemm.cu
@@ -168,6 +168,11 @@ static __global__ void fattn_gemm_softmax(
         }
         __syncthreads();
         vmax = red[0];
+        // every thread must have read the max out of red[0] before pass 2 reuses red for the row sum:
+        // without this barrier warp 0 can store its partial sum into red[0] first, a late warp then
+        // takes that sum as the row max, and exp(v - m) is wrong for its keys -- run-to-run drift in
+        // fp32, and in fp16 a probability that overflows to inf and NaNs the output
+        __syncthreads();
     }

     const float m_old = m_state[h*nt + t];
```

The kernel's surrounding structure, for context (abridged, after the fix):

```cuda
__shared__ float red[block_size/WARP_SIZE];
// pass 1: row max
...  vmax = warp_reduce_max(vmax);
if (block_size > WARP_SIZE) {
    if (tid % WARP_SIZE == 0) { red[tid/WARP_SIZE] = vmax; }
    __syncthreads();
    vmax = tid < block_size/WARP_SIZE ? red[tid] : -FLT_MAX/2.0f;
    vmax = warp_reduce_max(vmax);
    if (tid == 0) { red[0] = vmax; }
    __syncthreads();
    vmax = red[0];
    __syncthreads();            // <-- NEW: everyone has read red[0] before it is reused
}
// pass 2: p = exp(v - m_new), row sum
...  sum = warp_reduce_sum(sum);
if (block_size > WARP_SIZE) {
    if (tid % WARP_SIZE == 0) { red[tid/WARP_SIZE] = sum; }   // <-- used to race with "vmax = red[0]"
    ...
```

History: the race has been there since the path was added (`738022bda`). `cb6024e6b` ("GEMM attention softmax writes
probabilities out of place -- the in-place write raced") chased intermittent NaN/drift in this same kernel. Moving
S and P to separate buffers made it rarer but didn't remove it. This shared-memory race is the likely remaining cause.
The source comments in that commit describe a NaN at 4096 context that "did not reproduce".

## 3. Who is affected

The kernel is only called by the generic branch of `ggml_cuda_flash_attn_ext_gemm`, which is used when **all** of
these hold:
- pre-Volta GPU (cc < 7.0), `GGML_CUDA_FA_GEMM` not set to 0 (the path is on by default);
- KV length `nkv >= 4096` (`GGML_CUDA_FA_GEMM_MINKV`) and `>= 128` query rows in the ubatch (long prompt prefill);
- KV cache type **not q4_0** (f16, q8_0, q5_x, ...). q4_0 K/V return early into the fold kernels
  (`ggml_cuda_fa_fold_usable`), which have barriers between every write and read of their own `sm.red`.

Decode and speculative verify (few rows) never enter the GEMM path. So the default q4_0-KV configuration in Kmic's
QUICKSTART is unaffected. f16 or q8_0 KV with long prompts is affected.

## 4. Evidence

Hardware: 2x Tesla P100-PCIE-16GB (sm_60, no P2P between the cards), CUDA 12, `-sm tensor`, driver 580.
Model: Qwen3.8-27B UD-Q6_K. Power limit 150 W during the A/B (set for an unrelated host issue).
Repro: 13 prompts of 5,829-7,901 tokens (one decision each), 3 passes = 39 decodes, context 8192, `-b 512 -ub 512`,
fresh KV (memory cleared) per prompt. "NaN" = non-finite final hidden state; "rows differ" = how many of the 13
prompts gave different logits across the 3 passes (identical input, identical process).

### 4.1 Before the fix: bisect with env switches (f16 KV, base model)

| configuration | NaN | rows whose logits differ between passes |
|---|---|---|
| default (fp16 GEMM attention) | **10/39** | **8/13** |
| `GGML_CUDA_FA_GEMM=0` (upstream tile kernel) | 0/39 | 0/13 |
| `GGML_CUDA_FA_GEMM_PREC=32` (fp32 GEMM attention) | 0/39 | **11/13** |
| `CUDA_LAUNCH_BLOCKING=1` | n/a: deadlocks (the host-staged all-reduce spin-waits on the other GPU) | |

Reading: the bug is inside the GEMM attention path (tile is clean and deterministic). It is nondeterministic in both
precisions, so it's a race, not overflow on particular data. fp16 turns it into NaN.

Other observations before the fix:
- q8_0 KV: **8/39 NaN** (same path). q4_0 KV: 0/39 (fold kernels, different code).
- `MALLOC_PERTURB_=165` (poisons freed host memory) did not change the NaN rate, so host memory corruption is ruled out.
- `compute-sanitizer --tool memcheck` on shorter runs: 0 errors. memcheck doesn't detect shared-memory ordering races.
- Every NaN prompt computed finite logits in other runs/configs, and all other prompts agreed closely across configs.

### 4.2 After the fix (same repro)

| configuration | NaN | rows whose logits differ between passes |
|---|---|---|
| f16 KV, base model | **0/39** (was 10/39) | **0/13** (was 8/13) |
| q8_0 KV + a LoRA adapter | **0/39** (was 8/39) | **0/13** |

So the GEMM attention path is deterministic again.

### 4.3 No cost for anything else

- Default q4_0-KV configuration, A/B of the fixed vs the pre-fix `libggml-cuda.so`, same build otherwise, ABAB order,
  150 W cap, `llama-bench -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ub 2048 -p 2048 -n 256 -d 8192 -r 2`:

  | | fixed | pre-fix |
  |---|---|---|
  | pp2048 @ depth 8192 | 408.96, 406.82 | 407.63, 406.64 |
  | tg256 @ depth 8192 | 26.36, 26.37 | 26.33, 26.35 |

  (Expected: q4_0 never calls this kernel. The barrier itself is one `__syncthreads()` per softmax block.)
- Kmic's gate (`tools/gate.sh --full`, corpus `p100-handoff/ppl-orig.txt`): **PPL 2.6074 ± 0.0198**, unchanged
  (band 2.6209 ± 0.0199); tg256 26.74 at 150 W. The full op suite had one failing case,
  `MUL_MAT(type_a=q5_1, type_b=f32, m=16, n=1, k=32)` ERR 0.000539 > 0.0005. That is a random-input tolerance flake in
  an unrelated op: it passed 5/5 reruns on the fixed library and 5/5 on the pre-fix library.

## 5. Reproducing it

**The perplexity recipe suggested here earlier does not reproduce it** (run 2026-10-08 on Kmic's e48e240a8 without the
fix, Qwen3.8-27B Q6_K, 2x P100 `-sm tensor`, f16 KV, `-b 512 -ub 512`):

| form | runs | result without the fix |
|---|---|---|
| `llama-perplexity -c 8192`, per-chunk values compared | 3 | identical, no `nan` |
| same, logits saved with `--kl-divergence-base`, then `--kl-divergence` | 1 + 3 | KLD 0 (rounding), top-1 100% |
| `-c 6000` (each chunk ends on a 368-row batch past 4096 KV) | 1 + 3 | KLD 0, top-1 100% |

So perplexity-style prefill rarely loses this race; Kmic's own comment above `fattn_gemm_softmax` records one
4096-context perplexity run that went NaN once and did not reproduce, which fits.

**Where it does reproduce:** `llama-jev-decide` on branch `tyler-port` (tools/jev-decide), which runs one long prompt
(5,829-7,901 tokens) per row with the memory cleared between rows, `-c 8192 -b 512 -ub 512 -ctk f16 -ctv f16`, 13 rows x 3
passes, no LoRA: 10/39 rows NaN and 8/13 rows with different logits between passes. `GGML_CUDA_FA_GEMM=0` (tile
kernel): 0/39 NaN, 0/13 differ. With the barrier: 0/39 NaN, 0/13 differ. q8_0 KV + the JEV LoRA: 8/39 NaN before, 0 after (§4).

The case for the fix stands on the code (§2): the barrier is missing between the last read of `red[0]` and the next
write to `red[]`, which `__syncthreads` semantics require regardless of how often the timing loses.

## 6. Steps for the new Claude Code instance

```bash
# in a clone of Keoian/dual-p100-qwen-3-8-27b (it has remotes origin=Keoian fork, kmic=Kmic-68/llama.cpp)
git fetch kmic
git fetch origin tyler-port
git switch -c fix/fattn-gemm-softmax-race kmic/p100-optimizations
git cherry-pick 03da0202b        # touches only ggml/src/ggml-cuda/fattn-gemm.cu (+5 lines)
# the cherry-picked message mentions JEV/Phase 3 context: reword it with section 7's title/body
git commit --amend               # only on this new, unpushed branch
git push origin fix/fattn-gemm-softmax-race
gh pr create --repo Kmic-68/llama.cpp --base p100-optimizations \
  --head Keoian:fix/fattn-gemm-softmax-race --title "<section 7 title>" --body-file <section 7 body>
```

Notes:
- If `git cherry-pick` conflicts (Kmic's branch moved), apply section 2's diff by hand: add one `__syncthreads();`
  right after `vmax = red[0];` inside the `if (block_size > WARP_SIZE)` block of pass 1 in `fattn_gemm_softmax`.
- Tyler's project rules: never force-push; never push to a remote other than `origin` (Keoian). The PR is opened
  with `gh` from `origin`'s branch, which is fine. Ask Tyler before opening it if anything looks different from
  this brief.
- Build caveat from Kmic's docs: sm_60 only (`-DCMAKE_CUDA_ARCHITECTURES=60`). After editing a `.cu`/`.cuh`, rebuild
  `ggml-cuda`.

## 7. Suggested PR title and body

**Title:** `cuda: barrier before fattn_gemm_softmax reuses its shared reduction slots (fixes intermittent NaN / nondeterminism with f16/q8_0 KV)`

**Body:**

> Reproduced the NaN commented on in `ggml/src/ggml-cuda/fattn-gemm.cu` on line 115
> (https://github.com/Kmic-68/llama.cpp/blob/e48e240a8c9549f88c60b4cd47c8cd44bbf22f36/ggml/src/ggml-cuda/fattn-gemm.cu#L115,
> "a 4096-context perplexity run once went NaN ... and did not reproduce"). This addresses that issue.
>
> Why: `fattn_gemm_softmax` reads the block's row max back from `__shared__ red[0]` and then reuses `red[]` for the
> row-sum reduction with no barrier in between. Warp 0 can store its partial sum into `red[0]` before a slower warp
> has read the max. That warp then uses the sum as the max, and `exp(v - m)` is wrong for its keys. In the fp16 path a
> probability can overflow half to inf and the attention output becomes NaN; in fp32 (`GGML_CUDA_FA_GEMM_PREC=32`) it
> shows up as run-to-run drift. The out-of-place change in `cb6024e6b` removed a different race and doesn't touch this
> one. One `__syncthreads()` after `vmax = red[0]` fixes it.
>
> Affected: the generic branch of `ggml_cuda_flash_attn_ext_gemm` (pre-Volta, nkv >= 4096, >= 128 query rows,
> KV type other than q4_0, e.g. f16 / q8_0). q4_0 KV goes to the fold kernels and is unaffected. Decode is
> unaffected.
>
> Tested on 2x P100 (`-sm tensor`, Qwen3.8-27B Q6_K, f16 KV, prompts of 5.8k-7.9k tokens, 13 prompts x 3 passes):
> 10/39 NaN and 8/13 prompts nondeterministic before, 0/39 and 0/13 after. Perplexity runs rarely lose the race
> (3 repeated `-c 8192` f16-KV runs were bit-identical without the fix), which is likely why it didn't reproduce. No cost to the default q4_0 path
> (pp2048@8k 408.96/406.82 vs 407.63/406.64, tg256@8k 26.36/26.37 vs 26.33/26.35); `tools/gate.sh --full` PPL 2.6074
> unchanged.

## 8. Reference

- Fixed in: `Keoian/dual-p100-qwen-3-8-27b` `tyler-port` @ `03da0202b` (2026-10-06).
- Raw logs and repro inputs are on the reference build box only (`/work/bench/LOG.md` 10-06 entries, `/work/jev/runs/`,
  `/work/jev/data/long_rep.jsonl`); the numbers above are the summary. Repo write-up: CHANGES.md §16, FINDINGS.md.
