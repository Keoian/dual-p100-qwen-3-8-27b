#pragma once

// Pascal (sm_60, no DP4A) dot products for the integer matvec (mmvq.cu only, so editing this
// header rebuilds one object). Each replaces its vecdotq.cuh counterpart when the build targets
// sm_60 alone (GGML_CUDA_MMVQ_PASCAL); see get_vec_dot_q_cuda / get_vdr_mmvq in mmvq.cu.
// The q6_K and q3_K tunings that predate this file live in vecdotq.cuh itself.

#include "vecdotq.cuh"

// ---------------------------------------------------------------------------------------------
// q2_K / q3_K (agent kq23).
//
// Both types store 2-bit (q3_K: 2+1-bit) quants, so the generic path spends almost all of its
// instructions on the emulated dp4a (8 per int) and on per-lane scale/float work, not on bytes.
// These versions build dp4a's 16-bit operand halves directly from the quants and feed them to
// the two XMADs, skipping the PRMT unpack of the weight side, and they raise vdr so the scales,
// the q8_1 `ds` and the int->float conversions are paid once per group instead of once per int.
// The q8_1 side keeps dp4a's PRMT sign-extension (it is shared by the warp's two rows).
//
// Halves are paired as (byte0, byte2) and (byte1, byte3); the activation is paired the same way.

// c += a.lo*b.lo + a.hi*b.hi, signed 16-bit halves (the multiply half of ggml_cuda_dp4a on sm_60)
static __device__ __forceinline__ int p100_mad16x2(const int a, const int b, int c) {
    asm("{ .reg .s16 al,ah,bl,bh;\n\t"
        "mov.b32 {al,ah}, %1;\n\t"
        "mov.b32 {bl,bh}, %2;\n\t"
        "mad.wide.s16 %0, al, bl, %0;\n\t"
        "mad.wide.s16 %0, ah, bh, %0;\n\t}"
        : "+r"(c) : "r"(a), "r"(b));
    return c;
}

// q8_1 int -> sign-extended 16-bit halves (byte0, byte2) and (byte1, byte3)
static __device__ __forceinline__ void p100_sext_b02_b13(const int u, int & b02, int & b13) {
    asm("prmt.b32 %0, %1, 0, 0xA280;" : "=r"(b02) : "r"(u));
    asm("prmt.b32 %0, %1, 0, 0xB391;" : "=r"(b13) : "r"(u));
}

// N consecutive ints from a 2-byte-aligned address: N+1 aligned loads and a funnel shift each,
// instead of get_int_b2's two 16-bit loads and a merge per int.
template <int N>
static __device__ __forceinline__ void p100_load_ints_b2(const uint8_t * p, int * out) {
    const int     mis = (int) ((uintptr_t) p & 2);
    const int *   w   = (const int *) (p - mis);
    int r[N + 1];
#pragma unroll
    for (int k = 0; k <= N; ++k) {
        r[k] = w[k];
    }
#pragma unroll
    for (int k = 0; k < N; ++k) {
        out[k] = __funnelshift_r(r[k], r[k + 1], 8*mis);
    }
}

// q2_K. Element = dm.x*sc*q - dm.y*m, q in 0..3, one (sc, m) per 16 values.
//
// Each weight half is the quant left in place, (v & (3 << 2i)), so it carries a factor 4^i that
// is undone exactly in float; one LOP per half, no shifts beyond one v>>8 per int.
// The min term needs the plain sum of the q8_1 values per 16-value group. Rather than a second
// dp4a against m, the sign-extended activation halves already built for the XMADs are summed into
// one packed accumulator. As an int32, a pair (lo, hi) reads (lo mod 2^16) + 2^16*hi; flipping
// bit 15 makes that lo + 2^15 + 2^16*hi for either sign of lo, so after removing the known
// 2^15-per-term offset the sum is exactly sum(lo) + 2^16*sum(hi), decoded once per group. That work depends only on the activation, so it is shared by
// the warp's rows. All integer arithmetic is exact; only the float accumulation order differs.
#define VDR_Q2_K_Q8_1_MMVQ_P100 4

static __device__ __forceinline__ float vec_dot_q2_K_q8_1_p100(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    constexpr int vdr = VDR_Q2_K_Q8_1_MMVQ_P100;
    static_assert(vdr == 1 || vdr == 2 || vdr == 4, "q2_K p100: vdr must divide QI8_1/2");

    const block_q2_K * bq2_K = (const block_q2_K *) vbq + kbx;

    const int n    = iqs / QI8_1;        // which 128-value half of the block
    const int lane = iqs % QI8_1;        // int index inside each q8_1 block
    const int s    = lane / (QI8_1/2);   // which 16-value scale group

    int acc[QR2_K]  = { 0 }; // 4^i * sum(q*y)
    int ysum[QR2_K] = { 0 }; // packed sums of the activation halves, see above

#pragma unroll
    for (int l = 0; l < vdr; ++l) {
        const int v  = get_int_b4(bq2_K->qs, iqs + l);
        const int v8 = v >> 8;
#pragma unroll
        for (int i = 0; i < QR2_K; ++i) {
            const int mask = 0x00030003 << (2*i);
            int b02, b13;
            p100_sext_b02_b13(get_int_b4(bq8_1[QR2_K*n + i].qs, lane + l), b02, b13);
            acc[i]  = p100_mad16x2(v  & mask, b02, acc[i]);
            acc[i]  = p100_mad16x2(v8 & mask, b13, acc[i]);
            ysum[i] = ysum[i] + (b02 ^ 0x8000) + (b13 ^ 0x8000);
        }
    }

    // scale bytes 8n + s + 2i: bytes s, s+2 of word 2n and of word 2n+1
    const int sc0 = get_int_b4(bq2_K->scales, 2*n + 0) >> (8*s);
    const int sc1 = get_int_b4(bq2_K->scales, 2*n + 1) >> (8*s);

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sw = ((i >> 1) ? sc1 : sc0) >> (16*(i & 1));
        const int sc = sw & 0xF;
        const int m  = (sw >> 4) & 0xF;

        // |sum(lo)| <= 2*vdr*127 < 2^15, so it sign-extends back exactly from 16 bits
        const int yp  = ysum[i] - 2*vdr*0x8000;
        const int ylo = (yp << 16) >> 16;
        const int S   = ylo + ((yp - ylo) >> 16);

        const float d8 = __low2float(bq8_1[QR2_K*n + i].ds);
        // |acc| <= 64*3*127*4*vdr, times sc << (6 - 2i): below 2^24, exact in float
        sumf_d += d8 * (float) (acc[i] * (sc << (6 - 2*i)));
        sumf_m += d8 * (float) (S * m);
    }

    const float2 dm = __half22float2(bq2_K->dm);
    return dm.x * (1.0f/64.0f) * sumf_d - dm.y * sumf_m; // 1/64: exact, a power of two
}

// q3_K. Element = d*(sc - 32)*(q - 4*nh), q in 0..3 from qs, nh = NOT the hmask bit.
//
// Each weight half is built as 64*(q - 4*nh) by one two-source PRMT: the low byte is q placed in
// bits 6-7 of a byte, the high byte is the sign replica of a byte whose bit 7 holds nh, so a set
// nh contributes 0xFF00 = -256. No bias, no __vsubss4, no sign fix-up; 1/64 is undone at the end.
#define VDR_Q3_K_Q8_1_MMVQ_P100 4

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_p100(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    constexpr int vdr = VDR_Q3_K_Q8_1_MMVQ_P100;
    static_assert(vdr == 1 || vdr == 2 || vdr == 4, "q3_K p100: vdr must divide QI8_1/2");

    const block_q3_K * bq3_K = (const block_q3_K *) vbq + kbx;

    const int n    = iqs / QI8_1;
    const int lane = iqs % QI8_1;
    const int s    = lane / (QI8_1/2);

    // block_q3_K is 110 bytes, so its fields are only 2-byte aligned
    int vl[vdr];
    int vh[vdr];
    p100_load_ints_b2<vdr>(bq3_K->qs    + 4*iqs,  vl);
    p100_load_ints_b2<vdr>(bq3_K->hmask + 4*lane, vh);

    int acc[QR3_K] = { 0 };

#pragma unroll
    for (int l = 0; l < vdr; ++l) {
        const int h = ~vh[l] >> (4*n); // bit i of each byte: 1 where 4 is subtracted
#pragma unroll
        for (int i = 0; i < QR3_K; ++i) {
            const int A = (vl[l] << (6 - 2*i)) & 0xC0C0C0C0; // q in bits 6-7 of each byte
            const int H = h << (7 - i);                      // nh in bit 7 of each byte
            int a02, a13;
            asm("prmt.b32 %0, %1, %2, 0xE2C0;" : "=r"(a02) : "r"(A), "r"(H));
            asm("prmt.b32 %0, %1, %2, 0xF3D1;" : "=r"(a13) : "r"(A), "r"(H));
            int b02, b13;
            p100_sext_b02_b13(get_int_b4(bq8_1[QR3_K*n + i].qs, lane + l), b02, b13);
            acc[i] = p100_mad16x2(a02, b02, acc[i]);
            acc[i] = p100_mad16x2(a13, b13, acc[i]);
        }
    }

    // 6-bit scale j = 8n + s + 2i: low nibble in byte (s + 2i) at shift 4n,
    // high pair in byte 8 + s + 2(i&1) at shift 2*(2n + i/2)
    int sw[3];
    p100_load_ints_b2<3>(bq3_K->scales, sw);
    const int lo0 = sw[0] >> (8*s + 4*n);
    const int lo1 = sw[1] >> (8*s + 4*n);
    const int hi  = sw[2] >> (8*s + 4*n);

    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int lo = (((i >> 1) ? lo1 : lo0) >> (16*(i & 1))) & 0xF;
        const int hb = (hi >> (16*(i & 1) + 2*(i >> 1))) & 3;
        const int sc = (lo | (hb << 4)) - 32;
        // |acc| <= 64*4*127*4*vdr, times |sc| <= 32: below 2^24, so the float product is exact
        sumf += __low2float(bq8_1[QR3_K*n + i].ds) * (float) (sc * acc[i]);
    }

    const float d = bq3_K->d;
    return d * (1.0f/64.0f) * sumf; // 1/64 undoes the placement above; exact, a power of two
}

// ---------------------------------------------------------------------------------------------
// q4_K / q5_K (agent kq45).
//
// Element = dm.x*sc*q - dm.y*m, q in 0..15 (q4_K) or 0..31 (q5_K), one (sc, m) per 32 values.
// vdr 4: a thread owns 16 consecutive values of each of the two sub-blocks 2j / 2j+1 that share
// one 32-byte quant chunk, so it reads its quants with one 16-byte shared load (both blocks are
// 16-byte multiples), decodes the scales, the dm and the two q8_1 ds once per 16 values instead
// of once per 8, and converts to float once per sub-block.
//
// The weight side never goes through dp4a's PRMT unpack: masking an int with 0x000F000F gives the
// 16-bit halves (q0, q2) and with 0x0F000F00 gives (256*q1, 256*q3), which feed the XMADs
// directly. The q8_1 side is permuted to match, (256*y0, 256*y2) and sext (y1, y3), so every
// product lands at a uniform 256*q*y and one accumulator serves the sub-block. The plain sum of
// y for the min term is summed from the same halves as packed integers (see t02/t13 below). All integer arithmetic is exact; 1/256 is folded into the final float scale (exact,
// a power of two). Only the float accumulation order differs from the generic path.
#define VDR_Q4_K_Q8_1_MMVQ_P100 4
#define VDR_Q5_K_Q8_1_MMVQ_P100 4

// q8_1 int -> (256*y0, 256*y2) and sext (y1, y3) as 16-bit halves
static __device__ __forceinline__ void p100_k45_y(const int u, int & y02, int & y13) {
    asm("prmt.b32 %0, %1, 0, 0x2404;" : "=r"(y02) : "r"(u));
    asm("prmt.b32 %0, %1, 0, 0xB391;" : "=r"(y13) : "r"(u));
}

// chunk j's two sub-block scales from the three 32-bit words of block.scales (same branchless
// decode as vec_dot_q4_K_q8_1, which reads the same bytes as three 16-bit loads)
static __device__ __forceinline__ void p100_k45_scales(const int w1, const int w2, const int w3, const int j, int sc[2], int m[2]) {
    const int jm = j & 1;
    const uint32_t s0 = ((uint32_t) w1 >> (16*jm)) & 0xffff;
    const uint32_t s2 = ((uint32_t) w2 >> (16*jm)) & 0xffff;
    const uint32_t s4 = ((uint32_t) w3 >> (16*jm)) & 0xffff;
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    const uint32_t a0 = ((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi);
    const uint32_t a1 = ((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi);
    sc[0] = a0 & 0xff; sc[1] = (a0 >> 8) & 0xff;
    m[0]  = a1 & 0xff; m[1]  = (a1 >> 8) & 0xff;
}

template <bool q5>
static __device__ __forceinline__ float p100_k45_dot(
    const uint8_t * blk, const block_q8_1 * __restrict__ bq8_1, const int iqs) {

    // block_q4_K is dm, scales[12], qs[128] (144 bytes); block_q5_K is dm, scales[12], qh[32],
    // qs[128] (176 bytes). Both are multiples of 16, so the staged blocks are 16-byte aligned and
    // dm + scales come in with one 16-byte load, as do each thread's quants and high bits.
    const uint8_t * qs = blk + (q5 ? 48 : 16);
    const uint8_t * qh = blk + 16;
    const int4 hdr = *(const int4 *) blk;

    const int j = iqs / 8;        // 32-byte chunk: sub-blocks 2j (low nibbles), 2j+1 (high)
    const int h = (iqs / 4) % 2;  // which 16 values of each sub-block

    const int4 v4 = *(const int4 *) (qs + 32*j + 16*h);
    const int  v[4] = { v4.x, v4.y, v4.z, v4.w };
    int hb[4] = { 0, 0, 0, 0 };
    if constexpr (q5) {
        const int4 h4 = *(const int4 *) (qh + 16*h);
        hb[0] = h4.x >> (2*j); hb[1] = h4.y >> (2*j); hb[2] = h4.z >> (2*j); hb[3] = h4.w >> (2*j);
    }

    // q8_1 blocks 2j (A) and 2j+1 (B): ints 4h..4h+3 of each, plus both ds. (Fetching these with
    // 8-byte loads -- 6 instead of 10 -- measured 10% slower: bank conflicts.)
    int ua[4], ub[4], dsa, dsb;
    {
        const char * yp = (const char *) bq8_1 + 2*j*(int) sizeof(block_q8_1);
        const int * pa = (const int *) (yp + 4 + 16*h);
        const int * pb = (const int *) (yp + 40 + 16*h);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            ua[l] = pa[l];
            ub[l] = pb[l];
        }
        dsa = *(const int *) yp;
        dsb = *(const int *) (yp + 36);
    }

    int accA = 0, accB = 0; // 256*sum(q*y)
    // Plain sums of y for the min term, from the halves already built for the XMADs (shared by
    // the warp's rows). Flipping bit 15 of a register of 16-bit halves (lo, hi) makes it exactly
    // 2^16*hi + lo + 2^15 for either sign of lo (no borrow), so such registers add as plain
    // integers. y13 = (y1, y3) is used as is; y02 = (256*y0, 256*y2) is flipped and then shifted
    // down by 8, which is exact (the low byte is zero) and gives 2^16*y2 + y0 + 128. So
    // t = 2^16*(S2 + S3) + (S0 + S1) + offset; the offset is preloaded.
    int tA = -4*(0x8000 + 0x80), tB = -4*(0x8000 + 0x80);

#pragma unroll
    for (int l = 0; l < 4; ++l) {
        int a02, a13, b02, b13;
        p100_k45_y(ua[l], a02, a13);
        p100_k45_y(ub[l], b02, b13);

        int lo02 = v[l] & 0x000F000F;
        int lo13 = v[l] & 0x0F000F00;
        const int t = v[l] >> 4;
        int hi02 = t & 0x000F000F;
        int hi13 = t & 0x0F000F00;
        if constexpr (q5) {
            const int HA = hb[l] << 4; // bit 2j   -> bit 4 of each byte
            const int HB = hb[l] << 3; // bit 2j+1 -> bit 4 of each byte
            lo02 |= HA & 0x00100010;
            lo13 |= HA & 0x10001000;
            hi02 |= HB & 0x00100010;
            hi13 |= HB & 0x10001000;
        }

        accA = p100_mad16x2(lo02, a02, accA);
        accA = p100_mad16x2(lo13, a13, accA);
        accB = p100_mad16x2(hi02, b02, accB);
        accB = p100_mad16x2(hi13, b13, accB);

        tA += ((a02 ^ 0x8000) >> 8) + (a13 ^ 0x8000);
        tB += ((b02 ^ 0x8000) >> 8) + (b13 ^ 0x8000);
    }
    // |S0 + S1| <= 8*128 < 2^15, so it sign-extends back exactly from 16 bits
    const int loA = (tA << 16) >> 16, loB = (tB << 16) >> 16;
    const int sumA = (loA + ((tA - loA) >> 16)) << 8; // 256*S
    const int sumB = (loB + ((tB - loB) >> 16)) << 8;

    int sc[2], m[2];
    p100_k45_scales(hdr.y, hdr.z, hdr.w, j, sc, m);

    const float d8a = __half2float(__ushort_as_half((unsigned short) (dsa & 0xffff)));
    const float d8b = __half2float(__ushort_as_half((unsigned short) (dsb & 0xffff)));
    const float2 dmf = __half22float2(*(const half2 *) &hdr.x);

    const float sumf_d = d8a * (float) (accA * sc[0]) + d8b * (float) (accB * sc[1]);
    const float sumf_m = d8a * (float) (sumA * m[0])  + d8b * (float) (sumB * m[1]);

    return (1.0f/256.0f) * (dmf.x*sumf_d - dmf.y*sumf_m);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_p100(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q4_K * bq = (const block_q4_K *) vbq + kbx;
    static_assert(sizeof(block_q4_K) == 144, "q4_K layout");
    return p100_k45_dot<false>((const uint8_t *) bq, bq8_1, iqs);
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_p100(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q5_K * bq = (const block_q5_K *) vbq + kbx;
    static_assert(sizeof(block_q5_K) == 176, "q5_K layout");
    return p100_k45_dot<true>((const uint8_t *) bq, bq8_1, iqs);
}

// ---------------------------------------------------------------------------------------------
// q4_0 / q4_1 / q5_0 / q5_1 / q8_0 / iq4_nl / iq4_xs (agent leg).
//
// The 32-value types originally split a block over 2-4 lanes (vdr 2), so each lane paid the
// staging-loop overhead, the block-scale load and the float epilogue for only 8-16 values.
// Here one lane owns a whole block (vdr = qi): one scale load and one I2F/FFMA per block, and
// two to four times the work per trip of the staging loop. Integer partial sums are exact; only
// the float grouping changes (one FFMA per block instead of one per half block).
// Blocks that start with a lone ggml_half are 2 mod 4 aligned and use p100_load_ints_b2.

// q8_1 int -> sign-extended 16-bit halves (byte0, byte1) and (byte2, byte3), as ggml_cuda_dp4a
static __device__ __forceinline__ void p100_sext_b01_b23(const int u, int & b01, int & b23) {
    asm("prmt.b32 %0, %1, 0, 0x9180;" : "=r"(b01) : "r"(u));
    asm("prmt.b32 %0, %1, 0, 0xB3A2;" : "=r"(b23) : "r"(u));
}

// 256 * sum(x*y) over one 4-bit block: v = 4 ints of nibbles (low nibbles are values 0..15,
// high nibbles 16..31), y = the matching q8_1 quants. Each weight half is the nibble masked in
// place, so it carries a factor 1, 16 or 256 that is folded back exactly with two shifted adds;
// that is 5 instructions per int to unpack 8 weights instead of 7 (mask, shift, 4 PRMT).
// The caller folds the 256 into the float scale (a power of two, so the result is unchanged).
static __device__ __forceinline__ int p100_dot_q4_block_x256(const int * v, const int8_t * yqs) {
    int s1 = 0, s16 = 0, s256 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int y0_02, y0_13, y1_02, y1_13;
        p100_sext_b02_b13(get_int_b4(yqs, i + 0), y0_02, y0_13);
        p100_sext_b02_b13(get_int_b4(yqs, i + 4), y1_02, y1_13);
        s1   = p100_mad16x2(v[i] & 0x000F000F,        y0_02, s1);   // values 4i, 4i+2
        s256 = p100_mad16x2(v[i] & 0x0F000F00,        y0_13, s256); // values 4i+1, 4i+3
        s16  = p100_mad16x2(v[i] & 0x00F000F0,        y1_02, s16);  // values 4i+16, 4i+18
        s256 = p100_mad16x2((v[i] >> 4) & 0x0F000F00, y1_13, s256); // values 4i+17, 4i+19
    }
    return (s1 << 8) + (s16 << 4) + s256;
}

#define VDR_Q4_0_Q8_1_MMVQ_P100 4

static __device__ __forceinline__ float vec_dot_q4_0_q8_1_p100(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q4_0 * bq = (const block_q4_0 *) vbq + kbx;
    GGML_UNUSED(iqs); // vdr == qi: always 0

    int v[4];
    p100_load_ints_b2<4>(bq->qs, v);
    const int sumi256 = p100_dot_q4_block_x256(v, bq8_1->qs);

    const float2 ds8f = __half22float2(bq8_1->ds);
    // the ds8f.y term subtracts 8 from each quant value
    return __half2float(bq->d) * (sumi256 * (ds8f.x * (1.0f/256.0f)) - 8.0f * ds8f.y);
}

#define VDR_Q4_1_Q8_1_MMVQ_P100 4

static __device__ __forceinline__ float vec_dot_q4_1_q8_1_p100(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q4_1 * bq = (const block_q4_1 *) vbq + kbx;
    GGML_UNUSED(iqs);

    int v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = get_int_b4(bq->qs, i);
    }
    const int sumi256 = p100_dot_q4_block_x256(v, bq8_1->qs);

#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(bq->dm, bq8_1->ds));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;
#else
    const float2 dm4f = __half22float2(bq->dm);
    const float2 ds8f = __half22float2(bq8_1->ds);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
#endif // FAST_FP16_AVAILABLE
    return sumi256 * (d4d8 * (1.0f/256.0f)) + m4s8;
}

// 256 * sum(x*y) over one 5-bit block (low nibbles as for q4, 5th bits in qh). The four 5th bits
// of a quartet are spread to bit 4 of each byte with one multiply (bit j -> bit 8j+4; the cross
// terms land on distinct bits, so nothing carries) instead of four shift+mask pairs, and the
// weight halves are then masked in place as in p100_dot_q4_block_x256 (factor 1 or 256).
static __device__ __forceinline__ int p100_dot_q5_block_x256(const int * vl, const int qh, const int8_t * yqs) {
    int s1 = 0, s256 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t a  = (uint32_t) qh >> (4*i);
        const int      hb = (int) (((a        & 0xF) * 0x02040810u) & 0x10101010u); // values 4i..4i+3
        const int      hc = (int) ((((a >> 16) & 0xF) * 0x02040810u) & 0x10101010u); // values 4i+16..4i+19
        const int      x0 = ((vl[i] >> 0) & 0x0F0F0F0F) | hb;
        const int      x1 = ((vl[i] >> 4) & 0x0F0F0F0F) | hc;
        int y0_02, y0_13, y1_02, y1_13;
        p100_sext_b02_b13(get_int_b4(yqs, i + 0), y0_02, y0_13);
        p100_sext_b02_b13(get_int_b4(yqs, i + 4), y1_02, y1_13);
        s1   = p100_mad16x2(x0 & 0x001F001F, y0_02, s1);
        s256 = p100_mad16x2(x0 & 0x1F001F00, y0_13, s256);
        s1   = p100_mad16x2(x1 & 0x001F001F, y1_02, s1);
        s256 = p100_mad16x2(x1 & 0x1F001F00, y1_13, s256);
    }
    return (s1 << 8) + s256;
}

#define VDR_Q5_0_Q8_1_MMVQ_P100 4

static __device__ __forceinline__ float vec_dot_q5_0_q8_1_p100(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q5_0 * bq = (const block_q5_0 *) vbq + kbx;
    GGML_UNUSED(iqs);

    int w[5]; // qh, then the 4 qs ints; contiguous from offset 2
    p100_load_ints_b2<5>(bq->qh, w);
    const int sumi256 = p100_dot_q5_block_x256(w + 1, w[0], bq8_1->qs);

    const float2 ds8f = __half22float2(bq8_1->ds);
    // the ds8f.y term subtracts 16 from each quant value
    return __half2float(bq->d) * (sumi256 * (ds8f.x * (1.0f/256.0f)) - 16.0f * ds8f.y);
}

#define VDR_Q5_1_Q8_1_MMVQ_P100 4

static __device__ __forceinline__ float vec_dot_q5_1_q8_1_p100(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q5_1 * bq = (const block_q5_1 *) vbq + kbx;
    GGML_UNUSED(iqs);

    int vl[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vl[i] = get_int_b4(bq->qs, i);
    }
    const int sumi256 = p100_dot_q5_block_x256(vl, get_int_b4(bq->qh, 0), bq8_1->qs);

#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(bq->dm, bq8_1->ds));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;
#else
    const float2 dm5f = __half22float2(bq->dm);
    const float2 ds8f = __half22float2(bq8_1->ds);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
#endif // FAST_FP16_AVAILABLE
    return sumi256 * (d5d8 * (1.0f/256.0f)) + m5s8;
}

#define VDR_Q8_0_Q8_1_MMVQ_P100 8

static __device__ __forceinline__ float vec_dot_q8_0_q8_1_p100(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q8_0 * bq = (const block_q8_0 *) vbq + kbx;
    GGML_UNUSED(iqs);

    int v[8];
    p100_load_ints_b2<8>((const uint8_t *) bq->qs, v);

    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sumi = ggml_cuda_dp4a(v[i], get_int_b4(bq8_1->qs, i), sumi);
    }

    const float d8_0 = __half2float(bq->d);
    const float d8_1 = __low2float(bq8_1->ds);
    return d8_0*d8_1 * ((float) sumi);
}

// prmt.b32 in its default mode, without __byte_perm's 3-bit selector masking
static __device__ __forceinline__ uint32_t p100_prmt(const uint32_t a, const uint32_t b, const uint32_t s) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
    return r;
}

// iq4: dot of one int of 8 table indices (as get_int_from_table_16) with q8_1 ints u0 (values
// 0..3) and u1 (values 16..19 of the group). The two byte-order fixups and dp4a's four weight-side
// PRMTs are replaced by four sign-extending PRMTs straight from the looked-up bytes, and the
// selectors are masked once per int instead of once per PRMT (__byte_perm masks every selector).
static __device__ __forceinline__ int p100_dot_iq4_int(const int q4, const int u0, const int u1, int sumi) {
    const uint32_t * table32 = (const uint32_t *) kvalues_iq4nl;
    const uint32_t qm  = (uint32_t) q4 & 0x77777777;                            // 3-bit table index
    const uint32_t sel = 0x32103210 | (((uint32_t) q4 & 0x88888888) >> 1);      // bit 3: low/high half
    uint32_t t[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const uint32_t low  = p100_prmt(table32[0], table32[1], qm >> (16*k));
        const uint32_t high = p100_prmt(table32[2], table32[3], qm >> (16*k));
        t[k] = p100_prmt(low, high, sel >> (16*k));
    }
    // t[0] = values (0, 16, 1, 17), t[1] = values (2, 18, 3, 19)
    int x0_01, x0_1617, x1_23, x1_1819, y0_01, y0_23, y1_01, y1_23;
    p100_sext_b02_b13(t[0], x0_01, x0_1617);
    p100_sext_b02_b13(t[1], x1_23, x1_1819);
    p100_sext_b01_b23(u0, y0_01, y0_23);
    p100_sext_b01_b23(u1, y1_01, y1_23);
    sumi = p100_mad16x2(x0_01,   y0_01, sumi);
    sumi = p100_mad16x2(x1_23,   y0_23, sumi);
    sumi = p100_mad16x2(x0_1617, y1_01, sumi);
    sumi = p100_mad16x2(x1_1819, y1_23, sumi);
    return sumi;
}

#define VDR_IQ4_NL_Q8_1_MMVQ_P100 4

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1_p100(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq4_nl * bq = (const block_iq4_nl *) vbq + kbx;
    GGML_UNUSED(iqs);

    int v[4];
    p100_load_ints_b2<4>(bq->qs, v);

    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sumi = p100_dot_iq4_int(v[i], get_int_b4(bq8_1->qs, i), get_int_b4(bq8_1->qs, i + 4), sumi);
    }

    const float d = __half2float(bq->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

#define VDR_IQ4_XS_Q8_1_MMVQ_P100 4

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1_p100(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq4_xs * bq = (const block_iq4_xs *) vbq + kbx;
    const block_q8_1 * by = bq8_1 + iqs/4;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sumi = p100_dot_iq4_int(get_int_b4(bq->qs, iqs + j), get_int_b4(by->qs, j), get_int_b4(by->qs, j + 4), sumi);
    }

    const int ls = ((bq->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;

    const float d = __half2float(bq->d) * __low2float(by->ds);
    return d * sumi;
}
// ---- end leg ---------------------------------------------------------------------------------
