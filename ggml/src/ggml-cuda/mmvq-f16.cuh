#pragma once

#include "common.cuh"

// fp16 multi-column q6_K matvec for Pascal (MTP verify widths). Returns false if it does not handle
// the shape, in which case the caller runs the integer path. See mmvq-f16.cu.
bool ggml_cuda_mmvq_f16_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                            ggml_tensor * dst, int64_t ncols);

// The gated delta net's alpha and beta matvecs (same activation, 2..5 columns, < 256 rows together)
// with the gate softplus(alpha + b) * m and sigmoid(beta) applied in the epilogue, in one launch.
// Bit-identical to MUL_MAT, ADD, SOFTPLUS, MUL, MUL_MAT, SIGMOID. GGML_CUDA_FUSE_GDN_GATE=0 disables.
bool ggml_cuda_mmvq_f16_gdn_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_a, const ggml_tensor * mm_b,
                                 const ggml_tensor * b, const ggml_tensor * m, ggml_tensor * gate_out, ggml_tensor * beta_out,
                                 int64_t ncols);

// The FFN's gate and up matvecs (same activation, same shape, 2..5 columns, >= 3072 rows) and the
// SWIGLU after them, in one launch. Bit-identical to MUL_MAT, MUL_MAT, GLU(SWIGLU).
// GGML_CUDA_FUSE_FFN_GLU=0 disables.
bool ggml_cuda_mmvq_f16_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_gate, const ggml_tensor * mm_up,
                            ggml_tensor * dst, int64_t ncols);

// Forget the cached fp16 activation (called at the start of every graph compute, like the q8_1 cache).
void ggml_cuda_mmvq_f16_invalidate(ggml_backend_cuda_context & ctx);

// whether q5_K takes the fp16 path (GGML_CUDA_MMVQ_F16_Q5K, default on)
bool ggml_cuda_mmvq_f16_q5_K_on();
