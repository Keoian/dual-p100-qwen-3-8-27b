# Archive

Superseded releases, each packed whole. None of them is the build to run. That's `../build/`.

| archive | what it is |
|---|---|
| `2026-09-06-bundle.tar.zst` | the first release (48 commits on upstream `f280b2698`). **It carries two data races** that were fixed afterwards: the GEMM attention softmax (avoidable with `GGML_CUDA_FA_GEMM=0`) and a tensor-parallel peer copy that corrupts decode and MTP, which no flag avoids. Kept only so OPTLOG's before/after numbers can be reproduced |
| `2026-09-16-bundle.tar.zst` | free of those two races, on the same upstream base. Superseded by later tuning. Like every build since `738022bda` until `03da0202b` (2026-10-06), it still has the softmax `red[]` race, which only affects non-q4_0 KV caches |
| `2026-09-19-bundle.tar.zst` | the last release on upstream `f280b2698`, before the first upstream merge. It's the "before" in CHANGES §9 |
| `2026-09-22-bundle.tar.zst` | `61b9684e1`, the first build on upstream `f46bc30cb` (CHANGES §9) |
| `2026-09-23-bundle.tar.zst` | `4a991193e`: q4p latency work and faster graph rebuilds (CHANGES §10) |
| `2026-09-23b-bundle.tar.zst` | `6ea4edbdb`: fixed-width verify and depth-scheduled draft length (CHANGES §10) |
| `2026-09-23c-bundle.tar.zst` | `d3a650552`: KQ-mask fast path under M-RoPE (CHANGES §10) |
| `2026-09-26-bundle.tar.zst` | `192fd789a`: fp16 math with fp32 folds, fold prefill attention (CHANGES §11, §12) |
| `2026-09-26b-bundle.tar.zst` | `ceefc4a46`: round 3, P2P AllReduce, FFN GLU fusion, block verification (CHANGES §13) |
| `2026-09-26c-bundle.tar.zst` | `781cb4220`, the last release before round 4. It's the "09-26 release" in CHANGES §14. Vision there needs `-ub 1024` |

To unpack one:

    tar --zstd -xf 2026-09-19-bundle.tar.zst

Run unpacked binaries with `LD_LIBRARY_PATH` pointing at their own directory. Otherwise they
load whatever `libggml-cuda.so` their RUNPATH names.
