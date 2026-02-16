// Copyright (c) 2025-2026, Brandon Lehmann
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of
//    conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list
//    of conditions and the following disclaimer in the documentation and/or other
//    materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors may be
//    used to endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
// THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/**
 * @file fp10_avx2.h
 * @brief AVX2 radix-2^25.5 field element operations using scalar int64_t.
 *
 * This header exists to solve a specific MSVC problem: on x64, field
 * multiplication normally uses __int128 (GCC/Clang) or _umul128 (MSVC) for
 * the 51-bit x 51-bit products. On MSVC, the _umul128 results are returned
 * through a uint128_emu struct, which the optimizer can't keep in registers
 * -- so force-inlining fp_mul into curve bodies causes massive register
 * spilling and a 15-30% regression. The radix-2^25.5 representation sidesteps
 * this entirely: products are at most 26 x 26 = 52 bits, fitting in plain
 * int64_t with no 128-bit arithmetic needed.
 *
 * The representation uses 10 limbs in alternating 26/25-bit widths (same as
 * the portable 32-bit implementation, but stored in int64_t for the wider
 * accumulators). Multiplication is a 10x10 schoolbook with pre-multiplied
 * 19*g terms for the wrap-around and pre-doubled odd-indexed f limbs. The
 * interleaved carry chain normalizes back to 26/25-bit limbs.
 *
 * Also provides fp51-to-fp10 and fp10-to-fp51 conversion functions used by
 * the AVX2 fp10-throughout scalarmult path.
 */

#ifndef RANSHAW_X64_AVX2_FP10_AVX2_H
#define RANSHAW_X64_AVX2_FP10_AVX2_H

#include "ranshaw_ct_barrier.h"
#include "fp_ops.h"
#include "ranshaw_platform.h"
#include "x64/fp51.h"

#include <immintrin.h>

#if defined(_MSC_VER)
#define FP10_AVX2_FORCE_INLINE __forceinline
#else
#define FP10_AVX2_FORCE_INLINE inline __attribute__((always_inline))
#endif

typedef int64_t fp10[10];

static const int64_t FP10_MASK26 = (1LL << 26) - 1;
static const int64_t FP10_MASK25 = (1LL << 25) - 1;

/**
 * @brief Convert fp51 (radix-2^51, uint64_t[5]) to fp10 (radix-2^25.5, int64_t[10]).
 *
 * Carry-propagates the fp51 input first to ensure each limb is <=51 bits,
 * then splits each 51-bit limb cleanly into a 26-bit even limb and a 25-bit
 * odd limb. After carry propagation: limb[i] = 26+25 = 51 bits, so the
 * split is exact with no overlap.
 */
static FP10_AVX2_FORCE_INLINE void fp51_to_fp10(fp10 out, const fp_fe src)
{
    // Carry-propagate to ensure each limb is <=51 bits
    uint64_t t[5], c;
    t[0] = src[0];
    c = t[0] >> 51;
    t[0] &= FP51_MASK;
    t[1] = src[1] + c;
    c = t[1] >> 51;
    t[1] &= FP51_MASK;
    t[2] = src[2] + c;
    c = t[2] >> 51;
    t[2] &= FP51_MASK;
    t[3] = src[3] + c;
    c = t[3] >> 51;
    t[3] &= FP51_MASK;
    t[4] = src[4] + c;
    c = t[4] >> 51;
    t[4] &= FP51_MASK;
    t[0] += c * 19;
    c = t[0] >> 51;
    t[0] &= FP51_MASK;
    t[1] += c;

    // Split each 51-bit limb into (26-bit, 25-bit) pair
    out[0] = (int64_t)(t[0] & 0x3FFFFFFULL);
    out[1] = (int64_t)(t[0] >> 26);
    out[2] = (int64_t)(t[1] & 0x3FFFFFFULL);
    out[3] = (int64_t)(t[1] >> 26);
    out[4] = (int64_t)(t[2] & 0x3FFFFFFULL);
    out[5] = (int64_t)(t[2] >> 26);
    out[6] = (int64_t)(t[3] & 0x3FFFFFFULL);
    out[7] = (int64_t)(t[3] >> 26);
    out[8] = (int64_t)(t[4] & 0x3FFFFFFULL);
    out[9] = (int64_t)(t[4] >> 26);
}

/**
 * @brief Convert fp10 (radix-2^25.5, int64_t[10]) to fp51 (radix-2^51, uint64_t[5]).
 *
 * Performs carry propagation first to ensure limbs are in canonical range,
 * then packs pairs of limbs back into 51-bit limbs.
 */
static FP10_AVX2_FORCE_INLINE void fp10_to_fp51(fp_fe out, const fp10 src)
{
    int64_t t[10], c;

    t[0] = src[0];
    t[1] = src[1];
    t[2] = src[2];
    t[3] = src[3];
    t[4] = src[4];
    t[5] = src[5];
    t[6] = src[6];
    t[7] = src[7];
    t[8] = src[8];
    t[9] = src[9];

    // Carry-propagate to canonical range: even limbs [0, 2^26), odd limbs [0, 2^25)
    c = t[0] >> 26;
    t[1] += c;
    t[0] &= FP10_MASK26;
    c = t[1] >> 25;
    t[2] += c;
    t[1] &= FP10_MASK25;
    c = t[2] >> 26;
    t[3] += c;
    t[2] &= FP10_MASK26;
    c = t[3] >> 25;
    t[4] += c;
    t[3] &= FP10_MASK25;
    c = t[4] >> 26;
    t[5] += c;
    t[4] &= FP10_MASK26;
    c = t[5] >> 25;
    t[6] += c;
    t[5] &= FP10_MASK25;
    c = t[6] >> 26;
    t[7] += c;
    t[6] &= FP10_MASK26;
    c = t[7] >> 25;
    t[8] += c;
    t[7] &= FP10_MASK25;
    c = t[8] >> 26;
    t[9] += c;
    t[8] &= FP10_MASK26;
    c = t[9] >> 25;
    t[0] += c * 19;
    t[9] &= FP10_MASK25;
    c = t[0] >> 26;
    t[1] += c;
    t[0] &= FP10_MASK26;

    // Pack pairs of limbs into 51-bit limbs. After carry propagation:
    //   t[0]: 26 bits, t[1]: 25 bits -> 26+25 = 51 bits -> fp51[0]
    //   t[2]: 26 bits, t[3]: 25 bits -> fp51[1]
    //   t[4]: 26 bits, t[5]: 25 bits -> fp51[2]
    //   t[6]: 26 bits, t[7]: 25 bits -> fp51[3]
    //   t[8]: 26 bits, t[9]: 25 bits -> fp51[4]
    out[0] = (uint64_t)t[0] | ((uint64_t)t[1] << 26);
    out[1] = (uint64_t)t[2] | ((uint64_t)t[3] << 26);
    out[2] = (uint64_t)t[4] | ((uint64_t)t[5] << 26);
    out[3] = (uint64_t)t[6] | ((uint64_t)t[7] << 26);
    out[4] = (uint64_t)t[8] | ((uint64_t)t[9] << 26);
}

/**
 * @brief fp10 addition: h = f + g (no carry propagation).
 */
static FP10_AVX2_FORCE_INLINE void fp10_add(fp10 h, const fp10 f, const fp10 g)
{
    h[0] = f[0] + g[0];
    h[1] = f[1] + g[1];
    h[2] = f[2] + g[2];
    h[3] = f[3] + g[3];
    h[4] = f[4] + g[4];
    h[5] = f[5] + g[5];
    h[6] = f[6] + g[6];
    h[7] = f[7] + g[7];
    h[8] = f[8] + g[8];
    h[9] = f[9] + g[9];
}

/**
 * @brief fp10 subtraction: h = f - g with bias to keep positive.
 */
static FP10_AVX2_FORCE_INLINE void fp10_sub(fp10 h, const fp10 f, const fp10 g)
{
    // Add 2*p to avoid underflow. p = 2^255 - 19.
    // In radix-2^25.5: 2p limbs are (0x7FFFFDA, 0x3FFFFFE, 0x7FFFFFE, 0x3FFFFFE, ..., 0x3FFFFFE)
    // Simplified: add a large enough multiple of p, then carry-propagate.
    h[0] = f[0] + 0x7FFFFDA - g[0];
    h[1] = f[1] + 0x3FFFFFE - g[1];
    h[2] = f[2] + 0x7FFFFFE - g[2];
    h[3] = f[3] + 0x3FFFFFE - g[3];
    h[4] = f[4] + 0x7FFFFFE - g[4];
    h[5] = f[5] + 0x3FFFFFE - g[5];
    h[6] = f[6] + 0x7FFFFFE - g[6];
    h[7] = f[7] + 0x3FFFFFE - g[7];
    h[8] = f[8] + 0x7FFFFFE - g[8];
    h[9] = f[9] + 0x3FFFFFE - g[9];

    int64_t c;
    c = h[0] >> 26;
    h[1] += c;
    h[0] &= FP10_MASK26;
    c = h[1] >> 25;
    h[2] += c;
    h[1] &= FP10_MASK25;
    c = h[2] >> 26;
    h[3] += c;
    h[2] &= FP10_MASK26;
    c = h[3] >> 25;
    h[4] += c;
    h[3] &= FP10_MASK25;
    c = h[4] >> 26;
    h[5] += c;
    h[4] &= FP10_MASK26;
    c = h[5] >> 25;
    h[6] += c;
    h[5] &= FP10_MASK25;
    c = h[6] >> 26;
    h[7] += c;
    h[6] &= FP10_MASK26;
    c = h[7] >> 25;
    h[8] += c;
    h[7] &= FP10_MASK25;
    c = h[8] >> 26;
    h[9] += c;
    h[8] &= FP10_MASK26;
    c = h[9] >> 25;
    h[0] += c * 19;
    h[9] &= FP10_MASK25;
}

/**
 * @brief fp10 negation: h = -f (mod p).
 */
static FP10_AVX2_FORCE_INLINE void fp10_neg(fp10 h, const fp10 f)
{
    fp10 zero = {0};
    fp10_sub(h, zero, f);
}

/**
 * @brief fp10 copy: h = f.
 */
static FP10_AVX2_FORCE_INLINE void fp10_copy(fp10 h, const fp10 f)
{
    h[0] = f[0];
    h[1] = f[1];
    h[2] = f[2];
    h[3] = f[3];
    h[4] = f[4];
    h[5] = f[5];
    h[6] = f[6];
    h[7] = f[7];
    h[8] = f[8];
    h[9] = f[9];
}

/**
 * @brief fp10 conditional move: if b, then t = u.
 */
static FP10_AVX2_FORCE_INLINE void fp10_cmov(fp10 t, const fp10 u, int64_t b)
{
    int64_t mask = -(int64_t)ranshaw_ct_barrier_u64((uint64_t)b);
    t[0] ^= mask & (t[0] ^ u[0]);
    t[1] ^= mask & (t[1] ^ u[1]);
    t[2] ^= mask & (t[2] ^ u[2]);
    t[3] ^= mask & (t[3] ^ u[3]);
    t[4] ^= mask & (t[4] ^ u[4]);
    t[5] ^= mask & (t[5] ^ u[5]);
    t[6] ^= mask & (t[6] ^ u[6]);
    t[7] ^= mask & (t[7] ^ u[7]);
    t[8] ^= mask & (t[8] ^ u[8]);
    t[9] ^= mask & (t[9] ^ u[9]);
}

/**
 * @brief fp10 schoolbook multiplication: h = f * g (mod 2^255-19).
 *
 * Uses the portable schoolbook algorithm with 10x10 limb products. Each product
 * is at most 26+26=52 bits (signed), fitting in int64_t. Odd-indexed g limbs
 * are pre-multiplied by 2 where needed for the reduction step (factor 19*2=38
 * for wrap-around terms).
 *
 * This is essentially the same algorithm as portable fp_mul but using int64_t
 * accumulators instead of int32_t limbs and int64_t products. The key
 * advantage over fp51_mul for MSVC: no 128-bit arithmetic, so no
 * uint128_emu struct, making it safe to force-inline.
 */
static FP10_AVX2_FORCE_INLINE void fp10_mul(fp10 h, const fp10 f, const fp10 g)
{
    int64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int64_t f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];
    int64_t g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3], g4 = g[4];
    int64_t g5 = g[5], g6 = g[6], g7 = g[7], g8 = g[8], g9 = g[9];

    int64_t g1_19 = 19 * g1; /* 1.4*2^29 */
    int64_t g2_19 = 19 * g2; /* 1.4*2^30; still safe in 32 bits */
    int64_t g3_19 = 19 * g3;
    int64_t g4_19 = 19 * g4;
    int64_t g5_19 = 19 * g5;
    int64_t g6_19 = 19 * g6;
    int64_t g7_19 = 19 * g7;
    int64_t g8_19 = 19 * g8;
    int64_t g9_19 = 19 * g9;

    int64_t f1_2 = 2 * f1;
    int64_t f3_2 = 2 * f3;
    int64_t f5_2 = 2 * f5;
    int64_t f7_2 = 2 * f7;
    int64_t f9_2 = 2 * f9;

    int64_t h0 = f0 * g0 + f1_2 * g9_19 + f2 * g8_19 + f3_2 * g7_19 + f4 * g6_19 + f5_2 * g5_19 + f6 * g4_19
                 + f7_2 * g3_19 + f8 * g2_19 + f9_2 * g1_19;
    int64_t h1 = f0 * g1 + f1 * g0 + f2 * g9_19 + f3 * g8_19 + f4 * g7_19 + f5 * g6_19 + f6 * g5_19 + f7 * g4_19
                 + f8 * g3_19 + f9 * g2_19;
    int64_t h2 = f0 * g2 + f1_2 * g1 + f2 * g0 + f3_2 * g9_19 + f4 * g8_19 + f5_2 * g7_19 + f6 * g6_19 + f7_2 * g5_19
                 + f8 * g4_19 + f9_2 * g3_19;
    int64_t h3 = f0 * g3 + f1 * g2 + f2 * g1 + f3 * g0 + f4 * g9_19 + f5 * g8_19 + f6 * g7_19 + f7 * g6_19 + f8 * g5_19
                 + f9 * g4_19;
    int64_t h4 = f0 * g4 + f1_2 * g3 + f2 * g2 + f3_2 * g1 + f4 * g0 + f5_2 * g9_19 + f6 * g8_19 + f7_2 * g7_19
                 + f8 * g6_19 + f9_2 * g5_19;
    int64_t h5 =
        f0 * g5 + f1 * g4 + f2 * g3 + f3 * g2 + f4 * g1 + f5 * g0 + f6 * g9_19 + f7 * g8_19 + f8 * g7_19 + f9 * g6_19;
    int64_t h6 = f0 * g6 + f1_2 * g5 + f2 * g4 + f3_2 * g3 + f4 * g2 + f5_2 * g1 + f6 * g0 + f7_2 * g9_19 + f8 * g8_19
                 + f9_2 * g7_19;
    int64_t h7 =
        f0 * g7 + f1 * g6 + f2 * g5 + f3 * g4 + f4 * g3 + f5 * g2 + f6 * g1 + f7 * g0 + f8 * g9_19 + f9 * g8_19;
    int64_t h8 =
        f0 * g8 + f1_2 * g7 + f2 * g6 + f3_2 * g5 + f4 * g4 + f5_2 * g3 + f6 * g2 + f7_2 * g1 + f8 * g0 + f9_2 * g9_19;
    int64_t h9 = f0 * g9 + f1 * g8 + f2 * g7 + f3 * g6 + f4 * g5 + f5 * g4 + f6 * g3 + f7 * g2 + f8 * g1 + f9 * g0;

    int64_t c;

    c = h0 >> 26;
    h1 += c;
    h0 &= FP10_MASK26;
    c = h4 >> 26;
    h5 += c;
    h4 &= FP10_MASK26;

    c = h1 >> 25;
    h2 += c;
    h1 &= FP10_MASK25;
    c = h5 >> 25;
    h6 += c;
    h5 &= FP10_MASK25;

    c = h2 >> 26;
    h3 += c;
    h2 &= FP10_MASK26;
    c = h6 >> 26;
    h7 += c;
    h6 &= FP10_MASK26;

    c = h3 >> 25;
    h4 += c;
    h3 &= FP10_MASK25;
    c = h7 >> 25;
    h8 += c;
    h7 &= FP10_MASK25;

    c = h4 >> 26;
    h5 += c;
    h4 &= FP10_MASK26;
    c = h8 >> 26;
    h9 += c;
    h8 &= FP10_MASK26;

    c = h9 >> 25;
    h0 += c * 19;
    h9 &= FP10_MASK25;

    c = h0 >> 26;
    h1 += c;
    h0 &= FP10_MASK26;

    h[0] = h0;
    h[1] = h1;
    h[2] = h2;
    h[3] = h3;
    h[4] = h4;
    h[5] = h5;
    h[6] = h6;
    h[7] = h7;
    h[8] = h8;
    h[9] = h9;
}

/**
 * @brief fp10 squaring: h = f^2 (mod 2^255-19).
 */
static FP10_AVX2_FORCE_INLINE void fp10_sq(fp10 h, const fp10 f)
{
    int64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int64_t f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];

    int64_t f0_2 = 2 * f0;
    int64_t f1_2 = 2 * f1;
    int64_t f2_2 = 2 * f2;
    int64_t f3_2 = 2 * f3;
    int64_t f4_2 = 2 * f4;
    int64_t f5_2 = 2 * f5;
    int64_t f6_2 = 2 * f6;
    int64_t f7_2 = 2 * f7;

    int64_t f5_38 = 38 * f5;
    int64_t f6_19 = 19 * f6;
    int64_t f7_38 = 38 * f7;
    int64_t f8_19 = 19 * f8;
    int64_t f9_38 = 38 * f9;

    int64_t h0 = f0 * f0 + f1_2 * f9_38 + f2_2 * f8_19 + f3_2 * f7_38 + f4_2 * f6_19 + f5 * f5_38;
    int64_t h1 = f0_2 * f1 + f2 * f9_38 + f3_2 * f8_19 + f4 * f7_38 + f5_2 * f6_19;
    int64_t h2 = f0_2 * f2 + f1_2 * f1 + f3_2 * f9_38 + f4_2 * f8_19 + f5_2 * f7_38 + f6 * f6_19;
    int64_t h3 = f0_2 * f3 + f1_2 * f2 + f4 * f9_38 + f5_2 * f8_19 + f6 * f7_38;
    int64_t h4 = f0_2 * f4 + f1_2 * f3_2 + f2 * f2 + f5_2 * f9_38 + f6_2 * f8_19 + f7 * f7_38;
    int64_t h5 = f0_2 * f5 + f1_2 * f4 + f2_2 * f3 + f6 * f9_38 + f7_2 * f8_19;
    int64_t h6 = f0_2 * f6 + f1_2 * f5_2 + f2_2 * f4 + f3_2 * f3 + f7_2 * f9_38 + f8 * f8_19;
    int64_t h7 = f0_2 * f7 + f1_2 * f6 + f2_2 * f5 + f3_2 * f4 + f8 * f9_38;
    int64_t h8 = f0_2 * f8 + f1_2 * f7_2 + f2_2 * f6 + f3_2 * f5_2 + f4 * f4 + f9 * f9_38;
    int64_t h9 = f0_2 * f9 + f1_2 * f8 + f2_2 * f7 + f3_2 * f6 + f4_2 * f5;

    int64_t c;

    c = h0 >> 26;
    h1 += c;
    h0 &= FP10_MASK26;
    c = h4 >> 26;
    h5 += c;
    h4 &= FP10_MASK26;

    c = h1 >> 25;
    h2 += c;
    h1 &= FP10_MASK25;
    c = h5 >> 25;
    h6 += c;
    h5 &= FP10_MASK25;

    c = h2 >> 26;
    h3 += c;
    h2 &= FP10_MASK26;
    c = h6 >> 26;
    h7 += c;
    h6 &= FP10_MASK26;

    c = h3 >> 25;
    h4 += c;
    h3 &= FP10_MASK25;
    c = h7 >> 25;
    h8 += c;
    h7 &= FP10_MASK25;

    c = h4 >> 26;
    h5 += c;
    h4 &= FP10_MASK26;
    c = h8 >> 26;
    h9 += c;
    h8 &= FP10_MASK26;

    c = h9 >> 25;
    h0 += c * 19;
    h9 &= FP10_MASK25;

    c = h0 >> 26;
    h1 += c;
    h0 &= FP10_MASK26;

    h[0] = h0;
    h[1] = h1;
    h[2] = h2;
    h[3] = h3;
    h[4] = h4;
    h[5] = h5;
    h[6] = h6;
    h[7] = h7;
    h[8] = h8;
    h[9] = h9;
}

/**
 * @brief fp10 double-squaring: h = 2 * f^2 (mod 2^255-19).
 */
static FP10_AVX2_FORCE_INLINE void fp10_sq2(fp10 h, const fp10 f)
{
    int64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int64_t f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];

    int64_t f0_2 = 2 * f0;
    int64_t f1_2 = 2 * f1;
    int64_t f2_2 = 2 * f2;
    int64_t f3_2 = 2 * f3;
    int64_t f4_2 = 2 * f4;
    int64_t f5_2 = 2 * f5;
    int64_t f6_2 = 2 * f6;
    int64_t f7_2 = 2 * f7;

    int64_t f5_38 = 38 * f5;
    int64_t f6_19 = 19 * f6;
    int64_t f7_38 = 38 * f7;
    int64_t f8_19 = 19 * f8;
    int64_t f9_38 = 38 * f9;

    int64_t h0 = f0 * f0 + f1_2 * f9_38 + f2_2 * f8_19 + f3_2 * f7_38 + f4_2 * f6_19 + f5 * f5_38;
    int64_t h1 = f0_2 * f1 + f2 * f9_38 + f3_2 * f8_19 + f4 * f7_38 + f5_2 * f6_19;
    int64_t h2 = f0_2 * f2 + f1_2 * f1 + f3_2 * f9_38 + f4_2 * f8_19 + f5_2 * f7_38 + f6 * f6_19;
    int64_t h3 = f0_2 * f3 + f1_2 * f2 + f4 * f9_38 + f5_2 * f8_19 + f6 * f7_38;
    int64_t h4 = f0_2 * f4 + f1_2 * f3_2 + f2 * f2 + f5_2 * f9_38 + f6_2 * f8_19 + f7 * f7_38;
    int64_t h5 = f0_2 * f5 + f1_2 * f4 + f2_2 * f3 + f6 * f9_38 + f7_2 * f8_19;
    int64_t h6 = f0_2 * f6 + f1_2 * f5_2 + f2_2 * f4 + f3_2 * f3 + f7_2 * f9_38 + f8 * f8_19;
    int64_t h7 = f0_2 * f7 + f1_2 * f6 + f2_2 * f5 + f3_2 * f4 + f8 * f9_38;
    int64_t h8 = f0_2 * f8 + f1_2 * f7_2 + f2_2 * f6 + f3_2 * f5_2 + f4 * f4 + f9 * f9_38;
    int64_t h9 = f0_2 * f9 + f1_2 * f8 + f2_2 * f7 + f3_2 * f6 + f4_2 * f5;

    h0 += h0;
    h1 += h1;
    h2 += h2;
    h3 += h3;
    h4 += h4;
    h5 += h5;
    h6 += h6;
    h7 += h7;
    h8 += h8;
    h9 += h9;

    int64_t c;

    c = h0 >> 26;
    h1 += c;
    h0 &= FP10_MASK26;
    c = h4 >> 26;
    h5 += c;
    h4 &= FP10_MASK26;

    c = h1 >> 25;
    h2 += c;
    h1 &= FP10_MASK25;
    c = h5 >> 25;
    h6 += c;
    h5 &= FP10_MASK25;

    c = h2 >> 26;
    h3 += c;
    h2 &= FP10_MASK26;
    c = h6 >> 26;
    h7 += c;
    h6 &= FP10_MASK26;

    c = h3 >> 25;
    h4 += c;
    h3 &= FP10_MASK25;
    c = h7 >> 25;
    h8 += c;
    h7 &= FP10_MASK25;

    c = h4 >> 26;
    h5 += c;
    h4 &= FP10_MASK26;
    c = h8 >> 26;
    h9 += c;
    h8 &= FP10_MASK26;

    c = h9 >> 25;
    h0 += c * 19;
    h9 &= FP10_MASK25;

    c = h0 >> 26;
    h1 += c;
    h0 &= FP10_MASK26;

    h[0] = h0;
    h[1] = h1;
    h[2] = h2;
    h[3] = h3;
    h[4] = h4;
    h[5] = h5;
    h[6] = h6;
    h[7] = h7;
    h[8] = h8;
    h[9] = h9;
}

#endif // RANSHAW_X64_AVX2_FP10_AVX2_H
