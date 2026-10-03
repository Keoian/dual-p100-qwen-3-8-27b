#include "mmvq-f16.cuh"
#include "unary.cuh"
#include "mmvq.cuh"

#include <unordered_map>

// q6_K x f32 matvec for 2..16 columns on Pascal, in fp16 with short chains folded into fp32.
// 2..5 columns are one speculative sequence's verify; 6..16 are several sequences verified together
// (parallel server slots). Wider than MMVQ_F16_MAXK, the columns run as two launches.
//
// The integer path (mmvq.cu) is ALU-bound here: sm_60 has no DP4A, and its emulation costs ~2.2
// instructions per multiply-add. This path converts each weight pair to fp16 once (PRMT, HSUB2,
// HMUL2) and shares it across the columns, then does one HFMA2 per two multiply-adds.
//
// Numerics, per column c and window w of 1024 values (4 q6_K blocks):
//   x    -> half, prescaled by a power of 2 so |x| < 1 (the scaling is exact; one rounding of x
//           to 11 bits, against the integer path's rounding of x to 8 bits in q8_1)
//   w    = (d*1024) * sc * (q - 32) in half: one rounding of the weight
//   each lane runs a fused fp16 chain of 16 HFMA2 over its window, and the two lanes are added and
//   folded into an fp32 accumulator once per window.
// Against a double-precision reference on this model's real weights: NMSE ~4.6e-7, where the
// integer path's q8_1 activations give ~1.3e-4 (OPTLOG 203, 206).
//
// Layout constraints: each row must hold an even number of q6_K blocks, so that every row and every
// 4-block window starts 4-byte aligned and block b's 2-byte phase (b & 1) is known at compile time.

#define MMVQ_F16_NW  2   // warps per block
#define MMVQ_F16_RPW 4   // rows per warp
#define MMVQ_F16_NBF 4   // q6_K blocks per window (fold)
static constexpr int MMVQ_F16_WIN = MMVQ_F16_NBF*256;
#define MMVQ_F16_MAXK 12 // widest single launch: wider, the 4-row tile runs out of registers

// Columns are independent (each one's arithmetic never touches another's), so splitting a batch into
// launches over column ranges gives the same results.
template <typename F>
static void mmvq_f16_columns(const int64_t ncols, F && f) {
    const int64_t n0 = ncols <= MMVQ_F16_MAXK ? ncols : (ncols + 1)/2;
    for (int64_t c0 = 0; c0 < ncols; c0 += n0) {
        f(c0, std::min(n0, ncols - c0));
    }
}

template <typename F>
static void mmvq_f16_nc(const int64_t nc, F && f) {
    switch (nc) {
        case 2:  f(std::integral_constant<int, 2>{});  break;
        case 3:  f(std::integral_constant<int, 3>{});  break;
        case 4:  f(std::integral_constant<int, 4>{});  break;
        case 5:  f(std::integral_constant<int, 5>{});  break;
        case 6:  f(std::integral_constant<int, 6>{});  break;
        case 7:  f(std::integral_constant<int, 7>{});  break;
        case 8:  f(std::integral_constant<int, 8>{});  break;
        case 9:  f(std::integral_constant<int, 9>{});  break;
        case 10: f(std::integral_constant<int, 10>{}); break;
        case 11: f(std::integral_constant<int, 11>{}); break;
        case 12: f(std::integral_constant<int, 12>{}); break;
        default: GGML_ABORT("mmvq_f16: %d columns", (int) nc);
    }
}

// one warp per (window, column)
static __global__ void mmvq_f16_prep(const float * __restrict__ X, const int64_t sx, __half * __restrict__ XS,
                                     float * __restrict__ S, const int K) {
    const int w = blockIdx.x*blockDim.y + threadIdx.y, c = blockIdx.y, nw = (K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN;
    if (w >= nw) {
        return;
    }
    const float * x = X + c*sx + (int64_t) w*MMVQ_F16_WIN;
    constexpr int PER = MMVQ_F16_WIN/32;
    const int len = min(MMVQ_F16_WIN, K - w*MMVQ_F16_WIN);
    float v[PER];
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < PER/4; ++j) {
        float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (j*128 < len) {
            f = __ldg((const float4 *) x + j*32 + threadIdx.x);
        }
        v[4*j] = f.x; v[4*j + 1] = f.y; v[4*j + 2] = f.z; v[4*j + 3] = f.w;
        m = fmaxf(m, fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w))));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, o));
    }
    int e = 0;
    if (m > 0.0f) {
        frexpf(m, &e);
    }
    const float inv = ldexpf(1.0f, -e);
    __half * xs = XS + (int64_t) c*K + (int64_t) w*MMVQ_F16_WIN;
#pragma unroll
    for (int j = 0; j < PER/4; ++j) {
        if (j*128 >= len) {
            break;
        }
        const __half2 a = __floats2half2_rn(v[4*j]*inv, v[4*j + 1]*inv), b = __floats2half2_rn(v[4*j + 2]*inv, v[4*j + 3]*inv);
        uint2 u;
        u.x = *(const uint32_t *) &a;
        u.y = *(const uint32_t *) &b;
        ((uint2 *) xs)[j*32 + threadIdx.x] = u;
    }
    if (threadIdx.x == 0) {
        S[c*nw + w] = ldexpf(1.0f, e - 10); // also undoes the 1024 on the weights
    }
}

// 32-bit field at byte offset OFF of block B, relative to a 4-aligned window base
template <int B, int OFF>
static __device__ __forceinline__ uint32_t mmvq_f16_fld(const uint32_t * wb, const int lane_words) {
    constexpr int A = 210*B + OFF;
    if constexpr ((A & 3) == 0) {
        return wb[A/4 + lane_words];
    } else {
        return __byte_perm(wb[A/4 + lane_words], wb[A/4 + 1 + lane_words], 0x5432);
    }
}

template <int NC, int RPW, int NBLK, int B = 0>
static __device__ __forceinline__ void mmvq_f16_blocks(
        const uint32_t * const * wb, const uint2 * const * scs, const __half * const * xw, __half2 (&t)[NC][RPW],
        const int P, const int ql_w, const int qh_w, const int vh_shift, const int sp) {
    if constexpr (B < NBLK) {
        const __half2 k1056 = __float2half2_rn(1056.0f);
        __half2 xa[NC][2], xb[NC][2];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const __half * xs = xw[c] + B*256;
            const uint2 a = __ldg((const uint2 *) (xs + P)), bb = __ldg((const uint2 *) (xs + P + 64));
            xa[c][0] = *(const __half2 *) &a.x;  xa[c][1] = *(const __half2 *) &a.y;
            xb[c][0] = *(const __half2 *) &bb.x; xb[c][1] = *(const __half2 *) &bb.y;
        }
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const uint32_t vl = mmvq_f16_fld<B, 0>(wb[i], ql_w);
            const uint32_t vh = mmvq_f16_fld<B, 128>(wb[i], qh_w) >> vh_shift;
            // scales so and so+4, precomputed per window by mmvq_f16_scales (the same half ops)
            const uint2 sAB = scs[i][B*8 + sp];
            const __half2 sA = *(const __half2 *) &sAB.x, sB = *(const __half2 *) &sAB.y;
            const uint32_t qa = (vl & 0x0F0F0F0F) | ((vh << 4) & 0x30303030);
            const uint32_t qb = ((vl >> 4) & 0x0F0F0F0F) | (vh & 0x30303030);
            // half(1024 + q) per byte, minus 1056, times the scale
            const uint32_t h0 = __byte_perm(qa, 0x64646464u, 0x5140), h1 = __byte_perm(qa, 0x64646464u, 0x5342);
            const uint32_t h2 = __byte_perm(qb, 0x64646464u, 0x5140), h3 = __byte_perm(qb, 0x64646464u, 0x5342);
            const __half2 w0 = __hmul2(__hsub2(*(const __half2 *) &h0, k1056), sA);
            const __half2 w1 = __hmul2(__hsub2(*(const __half2 *) &h1, k1056), sA);
            const __half2 w2 = __hmul2(__hsub2(*(const __half2 *) &h2, k1056), sB);
            const __half2 w3 = __hmul2(__hsub2(*(const __half2 *) &h3, k1056), sB);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                __half2 u = B == 0 ? __hmul2(w0, xa[c][0]) : __hfma2(w0, xa[c][0], t[c][i]);
                u = __hfma2(w1, xa[c][1], u);
                u = __hfma2(w2, xb[c][0], u);
                t[c][i] = __hfma2(w3, xb[c][1], u);
            }
        }
        mmvq_f16_blocks<NC, RPW, NBLK, B + 1>(wb, scs, xw, t, P, ql_w, qh_w, vh_shift, sp);
    }
}

// A block's 16 scales, as the (sA, sB) half2 pairs its lanes use: lane l of the warp computes pair
// p = l % 8 (scales j and j + 4, j = p < 4 ? p : p + 4) of block l / 8 of the window, once, instead
// of every lane recomputing its pair for every block (~12 instructions per lane, block and row).
// The half arithmetic is exactly the per-lane version's: sc2 = (half(1152 + sc) - 1152) * (d*1024).
template <int RPW>
static __device__ __forceinline__ void mmvq_f16_scales(const uint32_t * const * wb, uint2 (*scst)[MMVQ_F16_NBF*8],
                                                       const int nblk, const int lane) {
    const int B = lane >> 3, p = lane & 7, j = p < 4 ? p : p + 4;
    if (B >= nblk) {
        return;
    }
    const __half2 k1152 = __float2half2_rn(1152.0f), k1024 = __float2half2_rn(1024.0f);
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        const uint8_t * bb = (const uint8_t *) wb[i] + 210*B;
        const uint32_t sab = (uint32_t) bb[192 + j] | ((uint32_t) bb[196 + j] << 8);
        const uint32_t sh  = __byte_perm(sab ^ 0x8080u, 0x64646464u, 0x5140); // half(1024 + 128 + sc)
        const uint32_t d16 = *(const uint16_t *) (bb + 208);
        const uint32_t dd  = d16 | (d16 << 16);
        const __half2 d2  = __hmul2(*(const __half2 *) &dd, k1024);
        const __half2 sc2 = __hmul2(__hsub2(*(const __half2 *) &sh, k1152), d2);
        const __half2 sA  = __low2half2(sc2), sB = __high2half2(sc2);
        scst[i][B*8 + p] = make_uint2(*(const uint32_t *) &sA, *(const uint32_t *) &sB);
    }
}

// The gated delta net's gate and beta in one launch (GATE): rows [0, rows1) are alpha's (W, Y),
// giving softplus(alpha + b) * m; the rest are beta's (W2, Y2), giving sigmoid(beta). The epilogues
// are the unfused ADD, SOFTPLUS, MUL and SIGMOID kernels' operations, so results are bit-identical.
struct mmvq_f16_gate {
    const uint8_t * W2;
    float *         Y2;
    const float *   b;
    const float *   m;
    int             rows1;
};

static __device__ __forceinline__ float mmvq_f16_softplus(const float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x)); // op_softplus (unary.cu)
}

// GLU: the FFN's gate and up matvecs and the SWIGLU after them in one launch. Each warp takes RPW/2
// output rows: its first RPW/2 row slots walk the gate matrix (W), the rest the same rows of up
// (gate.W2), and the epilogue writes silu(gate) * up with the unfused SWIGLU kernel's expression.
// Every row's arithmetic is the unfused kernel's, so the result is bit-identical.
template <int NC, int RPW, int NWT, bool KS, int MINB = 12/NWT, bool GATE = false, bool GLU = false>
__launch_bounds__(NWT*WARP_SIZE, MINB) // 12/NWT: 168 registers, no spills: 6 blocks per SM (OPTLOG 210)
static __global__ void mmvq_f16_q6_K(const uint8_t * __restrict__ W, const int64_t row_bytes, const __half * __restrict__ XS,
                                     const float * __restrict__ S, float * __restrict__ Y, const int64_t sy,
                                     int rows, const int K, const mmvq_f16_gate gate = {}) {
    static_assert(!GATE || (KS && RPW == 1), "GATE: split-K, one row per block");
    static_assert(!GLU || (!KS && !GATE && RPW % 2 == 0), "GLU: row-parallel, gate/up row pairs");
    constexpr int ORW = GLU ? RPW/2 : RPW; // output rows per warp
    constexpr int RPB = NWT*RPW, WB = MMVQ_F16_NBF*210, NU = (WB + 15 + 15)/16;
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int nb = K/256, nw = (nb + MMVQ_F16_NBF - 1)/MMVQ_F16_NBF;
    // KS (split K): the block's warps share its RPW rows and take every NWT-th window each
    int row0 = KS ? blockIdx.x*RPW : GLU ? (blockIdx.x*NWT + wid)*ORW : blockIdx.x*RPB + wid*RPW;
    bool second = false;
    if constexpr (GATE) {
        if (row0 >= gate.rows1) {
            second = true;
            row0 -= gate.rows1;
            rows -= gate.rows1;
            W = gate.W2;
            Y = gate.Y2;
        } else {
            rows = gate.rows1;
        }
    }

    __shared__ uint4 wst[NWT][RPW][NU];

    // this lane's 8 values of a block: P..P+3 (scale so) and P+64..P+67 (scale so+4)
    const int iqs = lane, P = 128*(iqs/16) + 4*(iqs % 16);
    const int ql_w = iqs, qh_w = 8*(iqs/16) + iqs % 8, vh_shift = 2*((iqs % 16)/8), so = 8*(iqs/16) + (iqs % 16)/4;
    const int sp = so < 8 ? so : so - 4; // this lane's scale pair (so, so + 4), see mmvq_f16_scales
    __shared__ uint2 scst[NWT][RPW][MMVQ_F16_NBF*8];
    const uint2 * scs[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        scs[i] = scst[wid][i];
    }

    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    const char * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        if constexpr (GLU) {
            rp[i] = (const char *) (i < ORW ? W : gate.W2) + (int64_t) min(row0 + i % ORW, rows - 1)*row_bytes;
        } else {
            rp[i] = (const char *) W + (int64_t) min(row0 + i, rows - 1)*row_bytes + (KS ? (int64_t) wid*WB : 0);
        }
    }
    const __half * xw[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        xw[c] = XS + (int64_t) c*K + (KS ? (int64_t) wid*MMVQ_F16_WIN : 0);
    }

    for (int win = KS ? wid : 0; win < nw; win += KS ? NWT : 1) {
        const int nblk = min(MMVQ_F16_NBF, nb - win*MMVQ_F16_NBF);
        __syncwarp();
        const uint32_t * wb[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const char * g = rp[i];
            rp[i] += KS ? (int64_t) NWT*WB : WB;
            const int m = (int) ((uintptr_t) g & 15); // a multiple of 4
            wb[i] = (const uint32_t *) wst[wid][i] + (m >> 2);
            const uint4 * g16 = (const uint4 *) (g - m);
            const int nu = (m + nblk*210 + 15)/16;
#pragma unroll
            for (int r = 0; r < (NU + 31)/32; ++r) {
                const int k = r*32 + lane;
                if (k < nu) {
                    wst[wid][i][k] = __ldg(g16 + k);
                }
            }
        }
        __syncwarp();
        mmvq_f16_scales<RPW>(wb, scst[wid], nblk, lane);
        __syncwarp();
        __half2 t[NC][RPW];
        switch (nblk) {
            case 4: mmvq_f16_blocks<NC, RPW, 4>(wb, scs, xw, t, P, ql_w, qh_w, vh_shift, sp); break;
            default: mmvq_f16_blocks<NC, RPW, 2>(wb, scs, xw, t, P, ql_w, qh_w, vh_shift, sp); break; // nb % 4 == 2
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            xw[c] += KS ? NWT*MMVQ_F16_WIN : MMVQ_F16_WIN;
            const float s = __ldg(S + c*nw + win);
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
            }
        }
    }
    if constexpr (KS) {
        // per-warp partials, then warp 0 adds them in a fixed order
        __shared__ float red[NWT][NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                float v = acc[c][i];
#pragma unroll
                for (int o = 16; o; o >>= 1) {
                    v += __shfl_xor_sync(0xFFFFFFFF, v, o);
                }
                if (lane == 0) {
                    red[wid][c][i] = v;
                }
            }
        }
        __syncthreads();
        if (wid == 0 && lane < NC*RPW) {
            const int c = lane / RPW, i = lane % RPW;
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < NWT; ++w) {
                v += red[w][c][i];
            }
            if (row0 + i < rows) {
                if constexpr (GATE) {
                    const int r = row0 + i;
                    v = second ? 1.0f / (1.0f + expf(-v)) : mmvq_f16_softplus(v + gate.b[r]) * gate.m[r];
                }
                Y[c*sy + row0 + i] = v;
            }
        }
        return;
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW]; // op_swiglu (unary.cu)
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// Every other weight type: one kernel, one small unpack per type.
//
// The q6_K kernel above is scheduled around q6_K's 210-byte block. This one takes any type through a
// per-type unpack of 8 consecutive weights of a block (a "chunk"), written from ggml's reference
// dequantization (ggml-quants.c), and shares the rest: the prescaled fp16 activation and its cache,
// the per-window fp16 chains folded into fp32, the column split. Numerics per weight:
//   q    -> half(1024 + q) by byte permute (exact), minus the bias (exact)
//   w    = (q - bias) * half(d*1024*sc) [+ half(-dmin*1024*m)]: one or two roundings, as on q6_K
// Lane l takes chunks l, l + 32, l + 64, l + 96 of each 1024-value window: 16 HFMA2 per window and
// column, as on q6_K, then the fold into fp32 with the window's activation scale.

static __device__ __forceinline__ uint32_t mmvq_f16_ld2(const uint8_t * p) { // 2-byte aligned
    const uint16_t * q = (const uint16_t *) p;
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}
static __device__ __forceinline__ uint32_t mmvq_f16_ld4(const uint8_t * p) { // 4-byte aligned
    return *(const uint32_t *) p;
}
// 4 bits (bit k of b) -> bit 4 of byte k
static __device__ __forceinline__ uint32_t mmvq_f16_spread4(const uint32_t b) {
    return ((b & 1) << 4) | ((b & 2) << 11) | ((b & 4) << 18) | ((b & 8) << 25);
}
// 4 nibble indices (one per byte) -> kvalues_iq4nl[i] + 128 per byte
static __device__ __forceinline__ uint32_t mmvq_f16_iq4(const uint32_t v) {
    const uint32_t x = v & 0x07070707, y = x | (x >> 4);
    const uint32_t sel = (y & 0xFF) | ((y >> 8) & 0xFF00);
    const uint32_t lo = __byte_perm(0x3f2d1801u, 0x766a5d4fu, sel), hi = __byte_perm(0xa6998d81u, 0xf1d9c5b5u, sel);
    const uint32_t m = ((v >> 3) & 0x01010101) * 0xFF;
    return (lo & ~m) | (hi & m);
}
// 8 bytes (q per byte, element order) -> 8 weights (q - bias)*s, or (q - 1024 + 1024 - bias)*s + mn
template <bool MIN>
static __device__ __forceinline__ void mmvq_f16_w8(const uint32_t v0, const uint32_t v1, const float bias, const float s, const float mn,
                                                   __half2 (&w)[4]) {
    const __half2 kb = __float2half2_rn(1024.0f + bias), s2 = __float2half2_rn(s);
    const uint32_t h[4] = { __byte_perm(v0, 0x64646464u, 0x5140), __byte_perm(v0, 0x64646464u, 0x5342),
                            __byte_perm(v1, 0x64646464u, 0x5140), __byte_perm(v1, 0x64646464u, 0x5342) };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const __half2 q = __hsub2(*(const __half2 *) &h[k], kb);
        if constexpr (MIN) {
            w[k] = __hfma2(q, s2, __float2half2_rn(mn));
        } else {
            w[k] = __hmul2(q, s2);
        }
    }
}
static __device__ __forceinline__ void mmvq_f16_sm_k4(const int j, const uint8_t * q, int & d, int & m) { // get_scale_min_k4
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}
static __device__ __forceinline__ float mmvq_f16_h(const uint8_t * p) {
    return __half2float(*(const __half *) p);
}

template <ggml_type T> struct mmvq_f16_t;

template <> struct mmvq_f16_t<GGML_TYPE_Q4_0> {
    static constexpr int QK = 32, BS = 18;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const uint8_t * qs = b + 2 + 8*(sub & 1);
        const int sh = 4*(sub >> 1);
        mmvq_f16_w8<false>((mmvq_f16_ld2(qs) >> sh) & 0x0F0F0F0F, (mmvq_f16_ld2(qs + 4) >> sh) & 0x0F0F0F0F, 8.0f, mmvq_f16_h(b)*1024.0f, 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q4_1> {
    static constexpr int QK = 32, BS = 20;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const uint8_t * qs = b + 4 + 8*(sub & 1);
        const int sh = 4*(sub >> 1);
        mmvq_f16_w8<true>((mmvq_f16_ld4(qs) >> sh) & 0x0F0F0F0F, (mmvq_f16_ld4(qs + 4) >> sh) & 0x0F0F0F0F, 0.0f,
                          mmvq_f16_h(b)*1024.0f, mmvq_f16_h(b + 2)*1024.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q5_0> {
    static constexpr int QK = 32, BS = 22;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const uint8_t * qs = b + 6 + 8*(sub & 1);
        const int sh = 4*(sub >> 1);
        const uint32_t hb = mmvq_f16_ld2(b + 2) >> (8*sub); // element e's bit 4 is qh bit e
        mmvq_f16_w8<false>(((mmvq_f16_ld2(qs) >> sh) & 0x0F0F0F0F) | mmvq_f16_spread4(hb),
                           ((mmvq_f16_ld2(qs + 4) >> sh) & 0x0F0F0F0F) | mmvq_f16_spread4(hb >> 4), 16.0f, mmvq_f16_h(b)*1024.0f, 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q5_1> {
    static constexpr int QK = 32, BS = 24;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const uint8_t * qs = b + 8 + 8*(sub & 1);
        const int sh = 4*(sub >> 1);
        const uint32_t hb = mmvq_f16_ld4(b + 4) >> (8*sub);
        mmvq_f16_w8<true>(((mmvq_f16_ld4(qs) >> sh) & 0x0F0F0F0F) | mmvq_f16_spread4(hb),
                          ((mmvq_f16_ld4(qs + 4) >> sh) & 0x0F0F0F0F) | mmvq_f16_spread4(hb >> 4), 0.0f,
                          mmvq_f16_h(b)*1024.0f, mmvq_f16_h(b + 2)*1024.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q8_0> {
    static constexpr int QK = 32, BS = 34;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const uint8_t * qs = b + 2 + 8*sub;
        mmvq_f16_w8<false>(mmvq_f16_ld2(qs) ^ 0x80808080, mmvq_f16_ld2(qs + 4) ^ 0x80808080, 128.0f, mmvq_f16_h(b)*1024.0f, 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_IQ4_NL> {
    static constexpr int QK = 32, BS = 18;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const uint8_t * qs = b + 2 + 8*(sub & 1);
        const int sh = 4*(sub >> 1);
        mmvq_f16_w8<false>(mmvq_f16_iq4((mmvq_f16_ld2(qs) >> sh) & 0x0F0F0F0F), mmvq_f16_iq4((mmvq_f16_ld2(qs + 4) >> sh) & 0x0F0F0F0F),
                           128.0f, mmvq_f16_h(b)*1024.0f, 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_IQ4_XS> {
    static constexpr int QK = 256, BS = 136;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int ib = sub >> 2, t = sub & 3;
        const uint8_t * qs = b + 8 + 16*ib + 8*(t & 1);
        const int sh = 4*(t >> 1);
        const int ls = ((b[4 + ib/2] >> 4*(ib % 2)) & 0xF) | (((*(const uint16_t *) (b + 2) >> 2*ib) & 3) << 4);
        mmvq_f16_w8<false>(mmvq_f16_iq4((mmvq_f16_ld4(qs) >> sh) & 0x0F0F0F0F), mmvq_f16_iq4((mmvq_f16_ld4(qs + 4) >> sh) & 0x0F0F0F0F),
                           128.0f, mmvq_f16_h(b)*1024.0f*(float) (ls - 32), 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q2_K> {
    static constexpr int QK = 256, BS = 84;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int n = sub >> 4, j = (sub >> 2) & 3, l0 = 8*(sub & 3);
        const uint8_t * qs = b + 16 + 32*n + l0;
        const int sc = b[8*n + 2*j + (l0 >= 16)];
        mmvq_f16_w8<true>((mmvq_f16_ld4(qs) >> 2*j) & 0x03030303, (mmvq_f16_ld4(qs + 4) >> 2*j) & 0x03030303, 0.0f,
                          mmvq_f16_h(b + 80)*1024.0f*(float) (sc & 0xF), -mmvq_f16_h(b + 82)*1024.0f*(float) (sc >> 4), w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q3_K> {
    static constexpr int QK = 256, BS = 110;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int n = sub >> 4, j = (sub >> 2) & 3, l0 = 8*(sub & 3);
        const uint8_t * qs = b + 32 + 32*n + l0, * hm = b + l0;
        const int hs = 4*n + j;
        const uint32_t v0 = ((mmvq_f16_ld2(qs)     >> 2*j) & 0x03030303) | (((mmvq_f16_ld2(hm)     >> hs) & 0x01010101) << 2);
        const uint32_t v1 = ((mmvq_f16_ld2(qs + 4) >> 2*j) & 0x03030303) | (((mmvq_f16_ld2(hm + 4) >> hs) & 0x01010101) << 2);
        const int is = 8*n + 2*j + (l0 >= 16);
        const uint8_t * sc = b + 96;
        const int s6 = (is < 8 ? sc[is] & 0xF : sc[is - 8] >> 4) | (((sc[8 + is % 4] >> 2*(is/4)) & 3) << 4);
        mmvq_f16_w8<false>(v0, v1, 4.0f, mmvq_f16_h(b + 108)*1024.0f*(float) (s6 - 32), 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q4_K> {
    static constexpr int QK = 256, BS = 144;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int j = sub >> 3, t = sub & 7, is = 2*j + (t >> 2), sh = 4*(t >> 2);
        const uint8_t * qs = b + 16 + 32*j + 8*(t & 3);
        int sc, m;
        mmvq_f16_sm_k4(is, b + 4, sc, m);
        mmvq_f16_w8<true>((mmvq_f16_ld4(qs) >> sh) & 0x0F0F0F0F, (mmvq_f16_ld4(qs + 4) >> sh) & 0x0F0F0F0F, 0.0f,
                          mmvq_f16_h(b)*1024.0f*(float) sc, -mmvq_f16_h(b + 2)*1024.0f*(float) m, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q5_K> {
    static constexpr int QK = 256, BS = 176;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int j = sub >> 3, t = sub & 7, is = 2*j + (t >> 2), sh = 4*(t >> 2);
        const uint8_t * qs = b + 48 + 32*j + 8*(t & 3), * qh = b + 16 + 8*(t & 3);
        int sc, m;
        mmvq_f16_sm_k4(is, b + 4, sc, m);
        const uint32_t v0 = ((mmvq_f16_ld4(qs)     >> sh) & 0x0F0F0F0F) | (((mmvq_f16_ld4(qh)     >> is) & 0x01010101) << 4);
        const uint32_t v1 = ((mmvq_f16_ld4(qs + 4) >> sh) & 0x0F0F0F0F) | (((mmvq_f16_ld4(qh + 4) >> is) & 0x01010101) << 4);
        mmvq_f16_w8<true>(v0, v1, 0.0f, mmvq_f16_h(b)*1024.0f*(float) sc, -mmvq_f16_h(b + 2)*1024.0f*(float) m, w);
    }
};

// RPW row slots per warp, NWT warps per block. GLU: slots [0, RPW/2) walk W (gate), the rest the
// same rows of W2 (up), and the epilogue writes silu(gate) * up as the SWIGLU kernel does.
// Each warp stages its rows' window bytes in shared memory with coalesced 16-byte loads (as the q6_K
// kernel does); the copy keeps each byte's address mod 16, so the types' 2- and 4-byte fields stay
// aligned. Then the unpacks read from shared memory.
template <ggml_type T, int NC, int RPW, int NWT, bool GLU>
__launch_bounds__(NWT*WARP_SIZE)
static __global__ void mmvq_f16_gen(const uint8_t * __restrict__ W, const uint8_t * __restrict__ W2, const int64_t row_bytes,
                                    const __half * __restrict__ XS, const float * __restrict__ S, float * __restrict__ Y,
                                    const int64_t sy, const int rows, const int K) {
    using tr = mmvq_f16_t<T>;
    constexpr int ORW = GLU ? RPW/2 : RPW, SUBS = tr::QK/8;
    constexpr int WB = (MMVQ_F16_WIN/tr::QK)*tr::BS, NU = (WB + 15 + 15)/16;
    __shared__ uint4 wst[NWT][RPW][NU];
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int row0 = (blockIdx.x*NWT + wid)*ORW;
    if (row0 >= rows) {
        return;
    }
    const int nchunk = K/8, nw = (K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN;
    const uint8_t * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        rp[i] = (GLU && i >= ORW ? W2 : W) + (int64_t) min(row0 + i % ORW, rows - 1)*row_bytes;
    }
    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    for (int win = 0; win < nw; ++win) {
        const int wbytes = min((int64_t) WB, row_bytes - (int64_t) win*WB);
        const uint8_t * wb[RPW];
        __syncwarp();
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const uint8_t * g = rp[i] + (int64_t) win*WB;
            const int m = (int) ((uintptr_t) g & 15);
            wb[i] = (const uint8_t *) wst[wid][i] + m;
            const uint4 * g16 = (const uint4 *) (g - m);
            const int nu = (m + wbytes + 15)/16;
#pragma unroll
            for (int r = 0; r < (NU + 31)/32; ++r) {
                const int k = r*32 + lane;
                if (k < nu) {
                    wst[wid][i][k] = __ldg(g16 + k);
                }
            }
        }
        __syncwarp();
        __half2 t[NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                t[c][i] = __float2half2_rn(0.0f);
            }
        }
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const int kw = jj*32 + lane, k = win*128 + kw;
            if (k < nchunk) {
                const int blk = kw / SUBS, sub = kw % SUBS;
                __half2 x[NC][4];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const uint4 u = __ldg((const uint4 *) (XS + (int64_t) c*K + 8*k));
                    x[c][0] = *(const __half2 *) &u.x; x[c][1] = *(const __half2 *) &u.y;
                    x[c][2] = *(const __half2 *) &u.z; x[c][3] = *(const __half2 *) &u.w;
                }
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
                    __half2 w[4];
                    tr::w8(wb[i] + blk*tr::BS, sub, w);
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        __half2 u = __hfma2(w[0], x[c][0], t[c][i]);
                        u = __hfma2(w[1], x[c][1], u);
                        u = __hfma2(w[2], x[c][2], u);
                        t[c][i] = __hfma2(w[3], x[c][3], u);
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float s = __ldg(S + c*nw + win);
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW];
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}

static bool mmvq_f16_gen_type(const ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
            return true;
        default:
            return false;
    }
}

// The fewest columns at which the generic kernel beats the integer path (test-backend-ops at 8704x5120,
// re-measured after the Pascal integer dot products, OPTLOG 281). The K-quants' unpack still costs the
// integer path per column and it loses from 3-5 columns; the 32-value quants and IQ4 keep the integer
// path up to 8, past which the integer side is MMQ (2-3x slower than this kernel).
static int mmvq_f16_gen_min_cols(const ggml_type t) {
    // GGML_CUDA_MMVQ_F16_GEN_MIN: override for every type (measurement)
    static const int env = [] { const char * s = getenv("GGML_CUDA_MMVQ_F16_GEN_MIN"); return s ? atoi(s) : 0; }();
    if (env > 0) {
        return env;
    }
    switch (t) {
        case GGML_TYPE_Q2_K:
            return 3;
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
            return 4;
        case GGML_TYPE_Q3_K:
            return 5;
        default:
            return MMVQ_MAX_BATCH_SIZE + 1;
    }
}

template <typename F>
static void mmvq_f16_gen_dispatch(const ggml_type t, F && f) {
    switch (t) {
        case GGML_TYPE_Q4_0:   f(std::integral_constant<ggml_type, GGML_TYPE_Q4_0>{});   break;
        case GGML_TYPE_Q4_1:   f(std::integral_constant<ggml_type, GGML_TYPE_Q4_1>{});   break;
        case GGML_TYPE_Q5_0:   f(std::integral_constant<ggml_type, GGML_TYPE_Q5_0>{});   break;
        case GGML_TYPE_Q5_1:   f(std::integral_constant<ggml_type, GGML_TYPE_Q5_1>{});   break;
        case GGML_TYPE_Q8_0:   f(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{});   break;
        case GGML_TYPE_IQ4_NL: f(std::integral_constant<ggml_type, GGML_TYPE_IQ4_NL>{}); break;
        case GGML_TYPE_IQ4_XS: f(std::integral_constant<ggml_type, GGML_TYPE_IQ4_XS>{}); break;
        case GGML_TYPE_Q2_K:   f(std::integral_constant<ggml_type, GGML_TYPE_Q2_K>{});   break;
        case GGML_TYPE_Q3_K:   f(std::integral_constant<ggml_type, GGML_TYPE_Q3_K>{});   break;
        case GGML_TYPE_Q4_K:   f(std::integral_constant<ggml_type, GGML_TYPE_Q4_K>{});   break;
        case GGML_TYPE_Q5_K:   f(std::integral_constant<ggml_type, GGML_TYPE_Q5_K>{});   break;
        default: GGML_ABORT("mmvq_f16_gen: type %s", ggml_type_name(t));
    }
}

// columns per launch: up to 8 (wider, split in two)
template <typename F>
static void mmvq_f16_gen_columns(const int64_t ncols, F && f) {
    const int64_t n0 = ncols <= 8 ? ncols : (ncols + 1)/2;
    for (int64_t c0 = 0; c0 < ncols; c0 += n0) {
        const int64_t n = std::min(n0, ncols - c0);
        switch (n) {
            case 1: f(c0, std::integral_constant<int, 1>{}); break;
            case 2: f(c0, std::integral_constant<int, 2>{}); break;
            case 3: f(c0, std::integral_constant<int, 3>{}); break;
            case 4: f(c0, std::integral_constant<int, 4>{}); break;
            case 5: f(c0, std::integral_constant<int, 5>{}); break;
            case 6: f(c0, std::integral_constant<int, 6>{}); break;
            case 7: f(c0, std::integral_constant<int, 7>{}); break;
            case 8: f(c0, std::integral_constant<int, 8>{}); break;
            default: GGML_ABORT("mmvq_f16_gen: %d columns", (int) n);
        }
    }
}

// GLU: W2 = up (same shape as W)
static void mmvq_f16_gen_launch(const ggml_type type, const uint8_t * W, const uint8_t * W2, const int64_t row_bytes,
                                const __half * xs, const float * sc, float * Y, const int64_t sy, const int64_t rows,
                                const int64_t K, const int64_t ncols, const bool glu, cudaStream_t stream) {
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    mmvq_f16_gen_dispatch(type, [&](auto tt) {
        constexpr ggml_type T = decltype(tt)::value;
        mmvq_f16_gen_columns(ncols, [&](const int64_t c0, auto ncc) {
            constexpr int NC = decltype(ncc)::value;
            constexpr int RPW = NC <= 4 ? 4 : 2, NWT = 2;
            if (glu) {
                const int g = (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
                mmvq_f16_gen<T, NC, 2*RPW, NWT, true><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                    W, W2, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
            } else {
                const int g = (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
                mmvq_f16_gen<T, NC, RPW, NWT, false><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                    W, nullptr, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
            }
        });
    });
    CUDA_CHECK(cudaGetLastError());
}

// The prescaled fp16 activation, reused across consecutive matmuls that read the same src1 (gate
// and up, the q/k/v projections), like the integer path's q8_1 cache. Keyed on the src1 node, whose
// contents are fixed within one graph evaluation; cleared at the start of each one. One persistent
// buffer per CUDA context (contexts are per device).
struct mmvq_f16_cache {
    const ggml_tensor * src1 = nullptr;
    const void *        data = nullptr;
    int64_t             ncols = 0, K = 0;
    void *              buf  = nullptr;
    size_t              cap  = 0;
    int                 device = -1; // the buffer's device: a freed context's address can be reused by one on another GPU
};
static std::unordered_map<const ggml_backend_cuda_context *, mmvq_f16_cache> mmvq_f16_caches;

void ggml_cuda_mmvq_f16_invalidate(ggml_backend_cuda_context & ctx) {
    auto it = mmvq_f16_caches.find(&ctx);
    if (it != mmvq_f16_caches.end()) {
        it->second.src1 = nullptr;
        it->second.data = nullptr;
    }
}

static bool mmvq_f16_shape_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int64_t ncols);
static void mmvq_f16_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t ncols, int64_t K,
                             __half ** xs_ptr, float ** sc_ptr);

bool ggml_cuda_mmvq_f16_gdn_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_a, const ggml_tensor * mm_b,
                                 const ggml_tensor * b, const ggml_tensor * m, ggml_tensor * gate_out, ggml_tensor * beta_out,
                                 const int64_t ncols) {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_FUSE_GDN_GATE");
        return !s || atoi(s) != 0;
    }();
    const ggml_tensor * src1 = mm_a->src[1];
    const int64_t K = mm_a->src[0]->ne[0], ra = mm_a->src[0]->ne[1], rb = mm_b->src[0]->ne[1];
    auto vec_ok = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->ne[0] == ra && ggml_nelements(t) == ra && ggml_is_contiguous(t);
    };
    // outputs: ra (rb) floats per column, contiguous
    auto out_ok = [&](const ggml_tensor * t, int64_t r) {
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == r*mm_a->ne[1];
    };
    if (!enabled || mm_a->src[0]->type != GGML_TYPE_Q6_K || mm_b->src[0]->type != GGML_TYPE_Q6_K ||
            mm_b->src[1] != src1 || mm_b->src[0]->ne[0] != K || ra + rb >= 256 || ra != rb ||
            !mmvq_f16_shape_ok(mm_a->src[0], src1, mm_a, ncols) || !mmvq_f16_shape_ok(mm_b->src[0], src1, mm_b, ncols) ||
            !vec_ok(b) || !vec_ok(m) || !out_ok(gate_out, ra) || !out_ok(beta_out, rb)) {
        return false;
    }
    __half * xs_ptr;
    float  * sc_ptr;
    mmvq_f16_prepare(ctx, src1, ncols, K, &xs_ptr, &sc_ptr);
    mmvq_f16_gate g;
    g.W2 = (const uint8_t *) mm_b->src[0]->data;
    g.Y2 = (float *) beta_out->data;
    g.b  = (const float *) b->data;
    g.m  = (const float *) m->data;
    g.rows1 = (int) ra;
    const uint8_t * W = (const uint8_t *) mm_a->src[0]->data;
    float * Y = (float *) gate_out->data;
    const int g_blocks = (int) (ra + rb);
    const dim3 bdk(WARP_SIZE, 4);
    cudaStream_t stream = ctx.stream();
    const int64_t rbytes = mm_a->src[0]->nb[1];
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    mmvq_f16_columns(ncols, [&](const int64_t c0, const int64_t n) {
        mmvq_f16_gate gc = g;
        gc.Y2 = g.Y2 + c0*rb;
        mmvq_f16_nc(n, [&](auto nc) {
            constexpr int NC = decltype(nc)::value;
            mmvq_f16_q6_K<NC, 1, 4, true, 3, true><<<g_blocks, bdk, 0, stream>>>(W, rbytes, xs_ptr + c0*K, sc_ptr + c0*nw, Y + c0*ra, ra, (int) (ra + rb), (int) K, gc);
        });
    });
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_mmvq_f16_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_gate, const ggml_tensor * mm_up,
                            ggml_tensor * dst, const int64_t ncols) {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_FUSE_FFN_GLU");
        return !s || atoi(s) != 0;
    }();
    const ggml_tensor * src1 = mm_gate->src[1];
    const ggml_tensor * wg = mm_gate->src[0], * wu = mm_up->src[0];
    const int64_t K = wg->ne[0], rows = wg->ne[1];
    // only the big row-parallel shapes (the unfused kernel's 2 warps x 4 rows band)
    if (!enabled || wu->type != wg->type || mm_up->src[1] != src1 || wu->ne[0] != K || wu->ne[1] != rows || wu->nb[1] != wg->nb[1] || rows < 3072 ||
            !mmvq_f16_shape_ok(wg, src1, mm_gate, ncols) || !mmvq_f16_shape_ok(wu, src1, mm_up, ncols) ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || dst->ne[0] != rows || ggml_nrows(dst) != mm_gate->ne[1]) {
        return false;
    }
    __half * xs_ptr;
    float  * sc_ptr;
    mmvq_f16_prepare(ctx, src1, ncols, K, &xs_ptr, &sc_ptr);
    mmvq_f16_gate g;
    g.W2 = (const uint8_t *) wu->data;
    const uint8_t * W = (const uint8_t *) wg->data;
    float * Y = (float *) dst->data;
    const int64_t sy = dst->nb[1]/sizeof(float);
    constexpr int ORW = MMVQ_F16_RPW/2;
    cudaStream_t stream = ctx.stream();
    if (wg->type != GGML_TYPE_Q6_K) {
        mmvq_f16_gen_launch(wg->type, W, g.W2, wg->nb[1], xs_ptr, sc_ptr, Y, sy, rows, K, ncols, true, stream);
        return true;
    }
    const int64_t rb = wg->nb[1];
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    mmvq_f16_columns(ncols, [&](const int64_t c0, const int64_t n) {
        mmvq_f16_nc(n, [&](auto nc) {
            constexpr int NC = decltype(nc)::value;
            // more than 5 columns: one warp per block, 8 blocks per SM (the 2-warp tile's 168 registers spill)
            constexpr int NWT = NC <= 5 ? MMVQ_F16_NW : 1, MINB = NC <= 5 ? 6 : 8;
            const int nblk = (int) ((rows + NWT*ORW - 1)/(NWT*ORW));
            mmvq_f16_q6_K<NC, MMVQ_F16_RPW, NWT, false, MINB, false, true><<<nblk, dim3(WARP_SIZE, NWT), 0, stream>>>(
                W, rb, xs_ptr + c0*K, sc_ptr + c0*nw, Y + c0*sy, sy, (int) rows, (int) K, g);
        });
    });
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static bool mmvq_f16_shape_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int64_t ncols) {
    // GGML_CUDA_MMVQ_F16=0 turns the fp16 path off
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_MMVQ_F16");
        return !s || atoi(s) != 0;
    }();
    // GGML_CUDA_MMVQ_F16_MINROWS: the smallest row count that takes this path
    static const int64_t min_rows = [] { const char * s = getenv("GGML_CUDA_MMVQ_F16_MINROWS"); return s ? (int64_t) atoll(s) : (int64_t) 16; }();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int64_t K = src0->ne[0], rows = src0->ne[1];
    // q6_K: its own kernel (K must hold an even number of blocks); the other types: the generic kernel
    const bool q6 = src0->type == GGML_TYPE_Q6_K;
    if (!q6 && mmvq_f16_gen_type(src0->type) && ncols < mmvq_f16_gen_min_cols(src0->type)) {
        return false;
    }
    return !(!enabled || cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc) || !(q6 || mmvq_f16_gen_type(src0->type)) ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ncols < (q6 ? 2 : 1) || ncols > MMVQ_F16_MAX_COLS ||
            K % (q6 ? 512 : 256) != 0 || rows < min_rows || src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            src0->nb[1] != ggml_row_size(src0->type, K) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            (src1->nb[1] % 16) != 0 || ((uintptr_t) src1->data % 16) != 0 || ((uintptr_t) src0->data % 4) != 0);
}

static void mmvq_f16_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const int64_t ncols, const int64_t K,
                             __half ** xs_out, float ** sc_out) {
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    cudaStream_t stream = ctx.stream();

    mmvq_f16_cache & cache = mmvq_f16_caches[&ctx];
    if (cache.device != ctx.device) {
        // a stale entry from a destroyed context on another device: its buffer is not usable here
        if (cache.buf) {
            ggml_cuda_set_device(cache.device);
            CUDA_CHECK(cudaFree(cache.buf));
            ggml_cuda_set_device(ctx.device);
        }
        cache = mmvq_f16_cache();
        cache.device = ctx.device;
    }
    const size_t xs_bytes = (size_t) ncols*K*sizeof(__half);
    const size_t need     = xs_bytes + (size_t) ncols*nw*sizeof(float);
    const bool hit = cache.src1 == src1 && cache.data == src1->data && cache.ncols == ncols && cache.K == K;
    if (!hit) {
        if (cache.cap < need) {
            if (cache.buf) {
                CUDA_CHECK(cudaFree(cache.buf)); // synchronizes; only when the buffer grows
            }
            CUDA_CHECK(cudaMalloc(&cache.buf, need));
            cache.cap = need;
        }
        cache.src1 = src1; cache.data = src1->data; cache.ncols = ncols; cache.K = K;
    }
    __half * xs_ptr = (__half *) cache.buf;
    float  * sc_ptr = (float *) ((char *) cache.buf + xs_bytes);
    if (!hit) {
        const dim3 pb(WARP_SIZE, 4), pg((nw + 3)/4, ncols);
        mmvq_f16_prep<<<pg, pb, 0, stream>>>((const float *) src1->data, src1->nb[1]/sizeof(float), xs_ptr, sc_ptr, (int) K);
    }
    *xs_out = xs_ptr;
    *sc_out = sc_ptr;
}

bool ggml_cuda_mmvq_f16_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                            ggml_tensor * dst, const int64_t ncols) {
    const int64_t K = src0->ne[0], rows = src0->ne[1];
    // Where it measured faster than the integer path (OPTLOG 206), test-backend-ops at 5 columns, rows x K:
    // 8704x5120 154 -> 133 us, 6144x5120 115 -> 97, 5120x5120 98 -> 89, 3072x5120 67 -> 61,
    // 5120x8704 161 -> 146, 5120x3072 61 -> 57. K must hold an even number of q6_K blocks.
    if (!mmvq_f16_shape_ok(src0, src1, dst, ncols)) {
        return false;
    }
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    cudaStream_t stream = ctx.stream();
    __half * xs_ptr;
    float  * sc_ptr;
    mmvq_f16_prepare(ctx, src1, ncols, K, &xs_ptr, &sc_ptr);
    const int64_t sy = dst->nb[1]/sizeof(float);
    const uint8_t * W = (const uint8_t *) src0->data;
    float * Y = (float *) dst->data;
    if (src0->type != GGML_TYPE_Q6_K) {
        mmvq_f16_gen_launch(src0->type, W, nullptr, src0->nb[1], xs_ptr, sc_ptr, Y, sy, rows, K, ncols, false, stream);
        return true;
    }
    // big matrices: 2 warps x 4 rows per block; mid-size: 1 row per warp; small (< 256 rows): the 4
    // warps of a block split K over one row, so there are enough blocks and warps to cover the GPU
    int64_t c0 = 0, nc = ncols; // the column range of the current launch
    auto launch = [&](auto rpw, auto nwt, auto ks, auto minb) {
        constexpr int  RPW  = decltype(rpw)::value;
        constexpr int  NWT  = decltype(nwt)::value;
        constexpr bool KS   = decltype(ks)::value;
        constexpr int  MINB = decltype(minb)::value;
        const dim3 bdk(WARP_SIZE, NWT);
        const int g = KS ? (int) ((rows + RPW - 1)/RPW) : (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
        mmvq_f16_nc(nc, [&](auto ncc) {
            constexpr int NC = decltype(ncc)::value;
            mmvq_f16_q6_K<NC, RPW, NWT, KS, MINB><<<g, bdk, 0, stream>>>(W, src0->nb[1], xs_ptr + c0*K, sc_ptr + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
        });
    };
    using I = std::integral_constant<int, 1>;
    mmvq_f16_columns(ncols, [&](const int64_t c0_, const int64_t n_) {
    c0 = c0_;
    nc = n_;
    if (nc > 5) {
        // several sequences' verify: 6..12 columns per launch. The 2-warp tiles' 168 registers spill
        // here; one warp per block with up to 255 registers does not (8704x5120 at 10 columns: 213 us
        // against 367 for the 2-warp tile, which spills, and ~300 for 2-row tiles).
        if (rows >= 3072) {
            launch(std::integral_constant<int, MMVQ_F16_RPW>{}, I{}, std::false_type{}, std::integral_constant<int, 8>{});
        } else if (rows >= 256) {
            launch(std::integral_constant<int, 2>{}, I{}, std::false_type{}, std::integral_constant<int, 12>{});
        } else {
            launch(I{}, std::integral_constant<int, 4>{}, std::true_type{}, std::integral_constant<int, 3>{});
        }
        return;
    }
    if (rows >= 3072) {
        // 6 or 7 blocks per SM (168 or 128 registers): take 7 when it needs fewer waves over the
        // SMs (3072 rows: 2 -> 1 wave, 49 vs 57 us at 5 columns; 6144: 3 -> 2; 8704: 4 -> 3).
        const int64_t nblk = (rows + MMVQ_F16_NW*MMVQ_F16_RPW - 1)/(MMVQ_F16_NW*MMVQ_F16_RPW);
        const int64_t nsm  = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        const int64_t w6 = (nblk + 6*nsm - 1)/(6*nsm), w7 = (nblk + 7*nsm - 1)/(7*nsm);
        if (w7 < w6) {
            launch(std::integral_constant<int, MMVQ_F16_RPW>{}, std::integral_constant<int, MMVQ_F16_NW>{}, std::false_type{}, std::integral_constant<int, 7>{});
        } else {
            launch(std::integral_constant<int, MMVQ_F16_RPW>{}, std::integral_constant<int, MMVQ_F16_NW>{}, std::false_type{}, std::integral_constant<int, 6>{});
        }
    } else if (rows >= 256) {
        // one warp of 2 rows per block: at 512 rows x 5120 and 5 columns (wk/wv under -sm tensor)
        // 13.3 us against 17.7 for 2 warps of 1 row (test-backend-ops, prep included). The rows'
        // arithmetic doesn't depend on the grouping, so the results are identical.
        // GGML_CUDA_MMVQ_F16_MID=0 restores 2 warps x 1 row.
        static const bool mid2 = [] { const char * s = getenv("GGML_CUDA_MMVQ_F16_MID"); return !s || atoi(s) != 0; }();
        if (mid2) {
            launch(std::integral_constant<int, 2>{}, I{}, std::false_type{}, std::integral_constant<int, 16>{});
        } else {
            launch(I{}, std::integral_constant<int, MMVQ_F16_NW>{}, std::false_type{}, std::integral_constant<int, 12/MMVQ_F16_NW>{});
        }
    } else {
        launch(I{}, std::integral_constant<int, 4>{}, std::true_type{}, std::integral_constant<int, 3>{});
    }
    });
    CUDA_CHECK(cudaGetLastError());
    return true;
}
