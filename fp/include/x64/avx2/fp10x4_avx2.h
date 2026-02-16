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
 * @file fp10x4_avx2.h
 * @brief 4-way parallel radix-2^25.5 field element operations using AVX2.
 *
 * This is the field arithmetic layer for the 4-way batch scalarmult operations.
 * Each fp10x4 holds 4 independent field elements packed horizontally into
 * AVX2 registers -- one element per 64-bit lane, 10 registers per fp10x4
 * (one per radix-2^25.5 limb). The representation is the same alternating
 * 26/25-bit unsigned limb layout used by the scalar fp10 in fp10_avx2.h.
 *
 * Multiplication uses _mm256_mul_epu32 (32x32 -> 64 unsigned), which is
 * safe because input limbs are at most 26 bits wide and 19*26 = 30 bits,
 * both fitting comfortably in the low 32 bits of each 64-bit lane. The
 * schoolbook product follows the same formula as fp10_mul in fp10_avx2.h,
 * with pre-multiplied 19*g terms for the wrap-around and pre-doubled
 * odd-indexed f limbs to compensate for the alternating radix.
 *
 * Subtraction uses a 2p bias (different values for limb 0, even limbs, and
 * odd limbs) to keep results non-negative, followed by carry propagation.
 * The carry chain uses unsigned right-shift (_mm256_srli_epi64) since all
 * values are guaranteed positive after the bias addition.
 */

#ifndef RANSHAW_X64_AVX2_FP10X4_AVX2_H
#define RANSHAW_X64_AVX2_FP10X4_AVX2_H

#include "ranshaw_platform.h"
#include "x64/avx2/fp10_avx2.h"

#include <immintrin.h>

#if defined(_MSC_VER)
#define FP10X4_FORCE_INLINE __forceinline
#else
#define FP10X4_FORCE_INLINE inline __attribute__((always_inline))
#endif

/**
 * @brief 4-way parallel field element type: 10 __m256i registers.
 *
 * v[i] holds limb i of 4 independent field elements in the 4 x 64-bit lanes.
 * Even limbs (0,2,4,6,8) are 26-bit, odd limbs (1,3,5,7,9) are 25-bit.
 */
typedef struct
{
    __m256i v[10];
} fp10x4;

// Constants as inline functions rather than aggregate initializers to avoid MSVC
// narrowing warnings (int64_t -> char truncation in __m256i aggregate init).
static inline __m256i fp10x4_mask26(void)
{
    return _mm256_set1_epi64x((1LL << 26) - 1);
}
static inline __m256i fp10x4_mask25(void)
{
    return _mm256_set1_epi64x((1LL << 25) - 1);
}
static inline __m256i fp10x4_c19(void)
{
    return _mm256_set1_epi64x(19);
}
static inline __m256i fp10x4_bias0(void)
{
    return _mm256_set1_epi64x(0x7FFFFDA);
}
static inline __m256i fp10x4_bias_even(void)
{
    return _mm256_set1_epi64x(0x7FFFFFE);
}
static inline __m256i fp10x4_bias_odd(void)
{
    return _mm256_set1_epi64x(0x3FFFFFE);
}

#define FP10X4_MASK26 fp10x4_mask26()
#define FP10X4_MASK25 fp10x4_mask25()
#define FP10X4_19 fp10x4_c19()
#define FP10X4_BIAS0 fp10x4_bias0()
#define FP10X4_BIAS_EVEN fp10x4_bias_even()
#define FP10X4_BIAS_ODD fp10x4_bias_odd()

/**
 * @brief Zero all 4 field elements.
 */
static FP10X4_FORCE_INLINE void fp10x4_0(fp10x4 *h)
{
    const __m256i z = _mm256_setzero_si256();
    for (int i = 0; i < 10; i++)
        h->v[i] = z;
}

/**
 * @brief Set all 4 field elements to 1.
 */
static FP10X4_FORCE_INLINE void fp10x4_1(fp10x4 *h)
{
    const __m256i z = _mm256_setzero_si256();
    h->v[0] = _mm256_set1_epi64x(1);
    for (int i = 1; i < 10; i++)
        h->v[i] = z;
}

/**
 * @brief Copy: h = f.
 */
static FP10X4_FORCE_INLINE void fp10x4_copy(fp10x4 *h, const fp10x4 *f)
{
    for (int i = 0; i < 10; i++)
        h->v[i] = f->v[i];
}

/**
 * @brief Addition: h = f + g (no carry propagation).
 */
static FP10X4_FORCE_INLINE void fp10x4_add(fp10x4 *h, const fp10x4 *f, const fp10x4 *g)
{
    for (int i = 0; i < 10; i++)
        h->v[i] = _mm256_add_epi64(f->v[i], g->v[i]);
}

/**
 * @brief Subtraction: h = f - g with bias + carry propagation.
 *
 * Adds 2*p to avoid underflow, then carry-propagates.
 */
static FP10X4_FORCE_INLINE void fp10x4_sub(fp10x4 *h, const fp10x4 *f, const fp10x4 *g)
{
    // Add 2*p bias and subtract g
    h->v[0] = _mm256_sub_epi64(_mm256_add_epi64(f->v[0], FP10X4_BIAS0), g->v[0]);
    h->v[1] = _mm256_sub_epi64(_mm256_add_epi64(f->v[1], FP10X4_BIAS_ODD), g->v[1]);
    h->v[2] = _mm256_sub_epi64(_mm256_add_epi64(f->v[2], FP10X4_BIAS_EVEN), g->v[2]);
    h->v[3] = _mm256_sub_epi64(_mm256_add_epi64(f->v[3], FP10X4_BIAS_ODD), g->v[3]);
    h->v[4] = _mm256_sub_epi64(_mm256_add_epi64(f->v[4], FP10X4_BIAS_EVEN), g->v[4]);
    h->v[5] = _mm256_sub_epi64(_mm256_add_epi64(f->v[5], FP10X4_BIAS_ODD), g->v[5]);
    h->v[6] = _mm256_sub_epi64(_mm256_add_epi64(f->v[6], FP10X4_BIAS_EVEN), g->v[6]);
    h->v[7] = _mm256_sub_epi64(_mm256_add_epi64(f->v[7], FP10X4_BIAS_ODD), g->v[7]);
    h->v[8] = _mm256_sub_epi64(_mm256_add_epi64(f->v[8], FP10X4_BIAS_EVEN), g->v[8]);
    h->v[9] = _mm256_sub_epi64(_mm256_add_epi64(f->v[9], FP10X4_BIAS_ODD), g->v[9]);

    // Carry propagation -- values are non-negative after bias, use unsigned shift
    __m256i c;
    c = _mm256_srli_epi64(h->v[0], 26);
    h->v[1] = _mm256_add_epi64(h->v[1], c);
    h->v[0] = _mm256_and_si256(h->v[0], FP10X4_MASK26);

    c = _mm256_srli_epi64(h->v[1], 25);
    h->v[2] = _mm256_add_epi64(h->v[2], c);
    h->v[1] = _mm256_and_si256(h->v[1], FP10X4_MASK25);

    c = _mm256_srli_epi64(h->v[2], 26);
    h->v[3] = _mm256_add_epi64(h->v[3], c);
    h->v[2] = _mm256_and_si256(h->v[2], FP10X4_MASK26);

    c = _mm256_srli_epi64(h->v[3], 25);
    h->v[4] = _mm256_add_epi64(h->v[4], c);
    h->v[3] = _mm256_and_si256(h->v[3], FP10X4_MASK25);

    c = _mm256_srli_epi64(h->v[4], 26);
    h->v[5] = _mm256_add_epi64(h->v[5], c);
    h->v[4] = _mm256_and_si256(h->v[4], FP10X4_MASK26);

    c = _mm256_srli_epi64(h->v[5], 25);
    h->v[6] = _mm256_add_epi64(h->v[6], c);
    h->v[5] = _mm256_and_si256(h->v[5], FP10X4_MASK25);

    c = _mm256_srli_epi64(h->v[6], 26);
    h->v[7] = _mm256_add_epi64(h->v[7], c);
    h->v[6] = _mm256_and_si256(h->v[6], FP10X4_MASK26);

    c = _mm256_srli_epi64(h->v[7], 25);
    h->v[8] = _mm256_add_epi64(h->v[8], c);
    h->v[7] = _mm256_and_si256(h->v[7], FP10X4_MASK25);

    c = _mm256_srli_epi64(h->v[8], 26);
    h->v[9] = _mm256_add_epi64(h->v[9], c);
    h->v[8] = _mm256_and_si256(h->v[8], FP10X4_MASK26);

    c = _mm256_srli_epi64(h->v[9], 25);
    h->v[0] = _mm256_add_epi64(h->v[0], _mm256_mul_epu32(c, FP10X4_19));
    h->v[9] = _mm256_and_si256(h->v[9], FP10X4_MASK25);
}

/**
 * @brief Negation: h = -f (mod p).
 */
static FP10X4_FORCE_INLINE void fp10x4_neg(fp10x4 *h, const fp10x4 *f)
{
    fp10x4 zero;
    fp10x4_0(&zero);
    fp10x4_sub(h, &zero, f);
}

/**
 * @brief Carry propagation for unsigned limbs (after mul/sq).
 *
 * Uses unsigned right-shift (srli) since limbs are always positive after
 * multiplication. Interleaves carry chains for better ILP.
 */
static FP10X4_FORCE_INLINE void fp10x4_carry(fp10x4 *h)
{
    __m256i c;

    c = _mm256_srli_epi64(h->v[0], 26);
    h->v[1] = _mm256_add_epi64(h->v[1], c);
    h->v[0] = _mm256_and_si256(h->v[0], FP10X4_MASK26);

    c = _mm256_srli_epi64(h->v[4], 26);
    h->v[5] = _mm256_add_epi64(h->v[5], c);
    h->v[4] = _mm256_and_si256(h->v[4], FP10X4_MASK26);

    c = _mm256_srli_epi64(h->v[1], 25);
    h->v[2] = _mm256_add_epi64(h->v[2], c);
    h->v[1] = _mm256_and_si256(h->v[1], FP10X4_MASK25);

    c = _mm256_srli_epi64(h->v[5], 25);
    h->v[6] = _mm256_add_epi64(h->v[6], c);
    h->v[5] = _mm256_and_si256(h->v[5], FP10X4_MASK25);

    c = _mm256_srli_epi64(h->v[2], 26);
    h->v[3] = _mm256_add_epi64(h->v[3], c);
    h->v[2] = _mm256_and_si256(h->v[2], FP10X4_MASK26);

    c = _mm256_srli_epi64(h->v[6], 26);
    h->v[7] = _mm256_add_epi64(h->v[7], c);
    h->v[6] = _mm256_and_si256(h->v[6], FP10X4_MASK26);

    c = _mm256_srli_epi64(h->v[3], 25);
    h->v[4] = _mm256_add_epi64(h->v[4], c);
    h->v[3] = _mm256_and_si256(h->v[3], FP10X4_MASK25);

    c = _mm256_srli_epi64(h->v[7], 25);
    h->v[8] = _mm256_add_epi64(h->v[8], c);
    h->v[7] = _mm256_and_si256(h->v[7], FP10X4_MASK25);

    c = _mm256_srli_epi64(h->v[4], 26);
    h->v[5] = _mm256_add_epi64(h->v[5], c);
    h->v[4] = _mm256_and_si256(h->v[4], FP10X4_MASK26);

    c = _mm256_srli_epi64(h->v[8], 26);
    h->v[9] = _mm256_add_epi64(h->v[9], c);
    h->v[8] = _mm256_and_si256(h->v[8], FP10X4_MASK26);

    c = _mm256_srli_epi64(h->v[9], 25);
    h->v[0] = _mm256_add_epi64(h->v[0], _mm256_mul_epu32(c, FP10X4_19));
    h->v[9] = _mm256_and_si256(h->v[9], FP10X4_MASK25);

    c = _mm256_srli_epi64(h->v[0], 26);
    h->v[1] = _mm256_add_epi64(h->v[1], c);
    h->v[0] = _mm256_and_si256(h->v[0], FP10X4_MASK26);
}

/**
 * @brief 4-way schoolbook multiplication: h = f * g (mod 2^255-19).
 *
 * Vectorized version of fp10_mul. Uses _mm256_mul_epu32 for 32x32->64
 * unsigned products. Limbs are at most 26 bits, so all products fit in 64 bits.
 */
static FP10X4_FORCE_INLINE void fp10x4_mul(fp10x4 *h, const fp10x4 *f, const fp10x4 *g)
{
    const __m256i f0 = f->v[0], f1 = f->v[1], f2 = f->v[2], f3 = f->v[3], f4 = f->v[4];
    const __m256i f5 = f->v[5], f6 = f->v[6], f7 = f->v[7], f8 = f->v[8], f9 = f->v[9];
    const __m256i g0 = g->v[0], g1 = g->v[1], g2 = g->v[2], g3 = g->v[3], g4 = g->v[4];
    const __m256i g5 = g->v[5], g6 = g->v[6], g7 = g->v[7], g8 = g->v[8], g9 = g->v[9];

    // Pre-multiply g by 19 for wrap-around terms
    const __m256i g1_19 = _mm256_mul_epu32(g1, FP10X4_19);
    const __m256i g2_19 = _mm256_mul_epu32(g2, FP10X4_19);
    const __m256i g3_19 = _mm256_mul_epu32(g3, FP10X4_19);
    const __m256i g4_19 = _mm256_mul_epu32(g4, FP10X4_19);
    const __m256i g5_19 = _mm256_mul_epu32(g5, FP10X4_19);
    const __m256i g6_19 = _mm256_mul_epu32(g6, FP10X4_19);
    const __m256i g7_19 = _mm256_mul_epu32(g7, FP10X4_19);
    const __m256i g8_19 = _mm256_mul_epu32(g8, FP10X4_19);
    const __m256i g9_19 = _mm256_mul_epu32(g9, FP10X4_19);

    // Pre-double odd-indexed f limbs
    const __m256i f1_2 = _mm256_slli_epi64(f1, 1);
    const __m256i f3_2 = _mm256_slli_epi64(f3, 1);
    const __m256i f5_2 = _mm256_slli_epi64(f5, 1);
    const __m256i f7_2 = _mm256_slli_epi64(f7, 1);
    const __m256i f9_2 = _mm256_slli_epi64(f9, 1);

    // Accumulate products for each output limb
    // h0 = f0*g0 + f1_2*g9_19 + f2*g8_19 + f3_2*g7_19 + f4*g6_19
    //     + f5_2*g5_19 + f6*g4_19 + f7_2*g3_19 + f8*g2_19 + f9_2*g1_19
    __m256i h0 = _mm256_mul_epu32(f0, g0);
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f1_2, g9_19));
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f2, g8_19));
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f3_2, g7_19));
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f4, g6_19));
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f5_2, g5_19));
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f6, g4_19));
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f7_2, g3_19));
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f8, g2_19));
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f9_2, g1_19));

    __m256i h1 = _mm256_mul_epu32(f0, g1);
    h1 = _mm256_add_epi64(h1, _mm256_mul_epu32(f1, g0));
    h1 = _mm256_add_epi64(h1, _mm256_mul_epu32(f2, g9_19));
    h1 = _mm256_add_epi64(h1, _mm256_mul_epu32(f3, g8_19));
    h1 = _mm256_add_epi64(h1, _mm256_mul_epu32(f4, g7_19));
    h1 = _mm256_add_epi64(h1, _mm256_mul_epu32(f5, g6_19));
    h1 = _mm256_add_epi64(h1, _mm256_mul_epu32(f6, g5_19));
    h1 = _mm256_add_epi64(h1, _mm256_mul_epu32(f7, g4_19));
    h1 = _mm256_add_epi64(h1, _mm256_mul_epu32(f8, g3_19));
    h1 = _mm256_add_epi64(h1, _mm256_mul_epu32(f9, g2_19));

    __m256i h2 = _mm256_mul_epu32(f0, g2);
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f1_2, g1));
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f2, g0));
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f3_2, g9_19));
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f4, g8_19));
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f5_2, g7_19));
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f6, g6_19));
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f7_2, g5_19));
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f8, g4_19));
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f9_2, g3_19));

    __m256i h3 = _mm256_mul_epu32(f0, g3);
    h3 = _mm256_add_epi64(h3, _mm256_mul_epu32(f1, g2));
    h3 = _mm256_add_epi64(h3, _mm256_mul_epu32(f2, g1));
    h3 = _mm256_add_epi64(h3, _mm256_mul_epu32(f3, g0));
    h3 = _mm256_add_epi64(h3, _mm256_mul_epu32(f4, g9_19));
    h3 = _mm256_add_epi64(h3, _mm256_mul_epu32(f5, g8_19));
    h3 = _mm256_add_epi64(h3, _mm256_mul_epu32(f6, g7_19));
    h3 = _mm256_add_epi64(h3, _mm256_mul_epu32(f7, g6_19));
    h3 = _mm256_add_epi64(h3, _mm256_mul_epu32(f8, g5_19));
    h3 = _mm256_add_epi64(h3, _mm256_mul_epu32(f9, g4_19));

    __m256i h4 = _mm256_mul_epu32(f0, g4);
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f1_2, g3));
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f2, g2));
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f3_2, g1));
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f4, g0));
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f5_2, g9_19));
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f6, g8_19));
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f7_2, g7_19));
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f8, g6_19));
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f9_2, g5_19));

    __m256i h5 = _mm256_mul_epu32(f0, g5);
    h5 = _mm256_add_epi64(h5, _mm256_mul_epu32(f1, g4));
    h5 = _mm256_add_epi64(h5, _mm256_mul_epu32(f2, g3));
    h5 = _mm256_add_epi64(h5, _mm256_mul_epu32(f3, g2));
    h5 = _mm256_add_epi64(h5, _mm256_mul_epu32(f4, g1));
    h5 = _mm256_add_epi64(h5, _mm256_mul_epu32(f5, g0));
    h5 = _mm256_add_epi64(h5, _mm256_mul_epu32(f6, g9_19));
    h5 = _mm256_add_epi64(h5, _mm256_mul_epu32(f7, g8_19));
    h5 = _mm256_add_epi64(h5, _mm256_mul_epu32(f8, g7_19));
    h5 = _mm256_add_epi64(h5, _mm256_mul_epu32(f9, g6_19));

    __m256i h6 = _mm256_mul_epu32(f0, g6);
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f1_2, g5));
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f2, g4));
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f3_2, g3));
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f4, g2));
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f5_2, g1));
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f6, g0));
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f7_2, g9_19));
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f8, g8_19));
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f9_2, g7_19));

    __m256i h7 = _mm256_mul_epu32(f0, g7);
    h7 = _mm256_add_epi64(h7, _mm256_mul_epu32(f1, g6));
    h7 = _mm256_add_epi64(h7, _mm256_mul_epu32(f2, g5));
    h7 = _mm256_add_epi64(h7, _mm256_mul_epu32(f3, g4));
    h7 = _mm256_add_epi64(h7, _mm256_mul_epu32(f4, g3));
    h7 = _mm256_add_epi64(h7, _mm256_mul_epu32(f5, g2));
    h7 = _mm256_add_epi64(h7, _mm256_mul_epu32(f6, g1));
    h7 = _mm256_add_epi64(h7, _mm256_mul_epu32(f7, g0));
    h7 = _mm256_add_epi64(h7, _mm256_mul_epu32(f8, g9_19));
    h7 = _mm256_add_epi64(h7, _mm256_mul_epu32(f9, g8_19));

    __m256i h8 = _mm256_mul_epu32(f0, g8);
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f1_2, g7));
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f2, g6));
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f3_2, g5));
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f4, g4));
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f5_2, g3));
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f6, g2));
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f7_2, g1));
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f8, g0));
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f9_2, g9_19));

    __m256i h9 = _mm256_mul_epu32(f0, g9);
    h9 = _mm256_add_epi64(h9, _mm256_mul_epu32(f1, g8));
    h9 = _mm256_add_epi64(h9, _mm256_mul_epu32(f2, g7));
    h9 = _mm256_add_epi64(h9, _mm256_mul_epu32(f3, g6));
    h9 = _mm256_add_epi64(h9, _mm256_mul_epu32(f4, g5));
    h9 = _mm256_add_epi64(h9, _mm256_mul_epu32(f5, g4));
    h9 = _mm256_add_epi64(h9, _mm256_mul_epu32(f6, g3));
    h9 = _mm256_add_epi64(h9, _mm256_mul_epu32(f7, g2));
    h9 = _mm256_add_epi64(h9, _mm256_mul_epu32(f8, g1));
    h9 = _mm256_add_epi64(h9, _mm256_mul_epu32(f9, g0));

    h->v[0] = h0;
    h->v[1] = h1;
    h->v[2] = h2;
    h->v[3] = h3;
    h->v[4] = h4;
    h->v[5] = h5;
    h->v[6] = h6;
    h->v[7] = h7;
    h->v[8] = h8;
    h->v[9] = h9;

    fp10x4_carry(h);
}

/**
 * @brief 4-way squaring: h = f^2 (mod 2^255-19).
 */
static FP10X4_FORCE_INLINE void fp10x4_sq(fp10x4 *h, const fp10x4 *f)
{
    const __m256i f0 = f->v[0], f1 = f->v[1], f2 = f->v[2], f3 = f->v[3], f4 = f->v[4];
    const __m256i f5 = f->v[5], f6 = f->v[6], f7 = f->v[7], f8 = f->v[8], f9 = f->v[9];

    const __m256i f0_2 = _mm256_slli_epi64(f0, 1);
    const __m256i f1_2 = _mm256_slli_epi64(f1, 1);
    const __m256i f2_2 = _mm256_slli_epi64(f2, 1);
    const __m256i f3_2 = _mm256_slli_epi64(f3, 1);
    const __m256i f4_2 = _mm256_slli_epi64(f4, 1);
    const __m256i f5_2 = _mm256_slli_epi64(f5, 1);
    const __m256i f6_2 = _mm256_slli_epi64(f6, 1);
    const __m256i f7_2 = _mm256_slli_epi64(f7, 1);

    const __m256i v38 = _mm256_set1_epi64x(38);

    const __m256i f5_38 = _mm256_mul_epu32(f5, v38);
    const __m256i f6_19 = _mm256_mul_epu32(f6, FP10X4_19);
    const __m256i f7_38 = _mm256_mul_epu32(f7, v38);
    const __m256i f8_19 = _mm256_mul_epu32(f8, FP10X4_19);
    const __m256i f9_38 = _mm256_mul_epu32(f9, v38);

    __m256i h0 = _mm256_mul_epu32(f0, f0);
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f1_2, f9_38));
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f2_2, f8_19));
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f3_2, f7_38));
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f4_2, f6_19));
    h0 = _mm256_add_epi64(h0, _mm256_mul_epu32(f5, f5_38));

    __m256i h1 = _mm256_mul_epu32(f0_2, f1);
    h1 = _mm256_add_epi64(h1, _mm256_mul_epu32(f2, f9_38));
    h1 = _mm256_add_epi64(h1, _mm256_mul_epu32(f3_2, f8_19));
    h1 = _mm256_add_epi64(h1, _mm256_mul_epu32(f4, f7_38));
    h1 = _mm256_add_epi64(h1, _mm256_mul_epu32(f5_2, f6_19));

    __m256i h2 = _mm256_mul_epu32(f0_2, f2);
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f1_2, f1));
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f3_2, f9_38));
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f4_2, f8_19));
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f5_2, f7_38));
    h2 = _mm256_add_epi64(h2, _mm256_mul_epu32(f6, f6_19));

    __m256i h3 = _mm256_mul_epu32(f0_2, f3);
    h3 = _mm256_add_epi64(h3, _mm256_mul_epu32(f1_2, f2));
    h3 = _mm256_add_epi64(h3, _mm256_mul_epu32(f4, f9_38));
    h3 = _mm256_add_epi64(h3, _mm256_mul_epu32(f5_2, f8_19));
    h3 = _mm256_add_epi64(h3, _mm256_mul_epu32(f6, f7_38));

    __m256i h4 = _mm256_mul_epu32(f0_2, f4);
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f1_2, f3_2));
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f2, f2));
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f5_2, f9_38));
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f6_2, f8_19));
    h4 = _mm256_add_epi64(h4, _mm256_mul_epu32(f7, f7_38));

    __m256i h5 = _mm256_mul_epu32(f0_2, f5);
    h5 = _mm256_add_epi64(h5, _mm256_mul_epu32(f1_2, f4));
    h5 = _mm256_add_epi64(h5, _mm256_mul_epu32(f2_2, f3));
    h5 = _mm256_add_epi64(h5, _mm256_mul_epu32(f6, f9_38));
    h5 = _mm256_add_epi64(h5, _mm256_mul_epu32(f7_2, f8_19));

    __m256i h6 = _mm256_mul_epu32(f0_2, f6);
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f1_2, f5_2));
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f2_2, f4));
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f3_2, f3));
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f7_2, f9_38));
    h6 = _mm256_add_epi64(h6, _mm256_mul_epu32(f8, f8_19));

    __m256i h7 = _mm256_mul_epu32(f0_2, f7);
    h7 = _mm256_add_epi64(h7, _mm256_mul_epu32(f1_2, f6));
    h7 = _mm256_add_epi64(h7, _mm256_mul_epu32(f2_2, f5));
    h7 = _mm256_add_epi64(h7, _mm256_mul_epu32(f3_2, f4));
    h7 = _mm256_add_epi64(h7, _mm256_mul_epu32(f8, f9_38));

    __m256i h8 = _mm256_mul_epu32(f0_2, f8);
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f1_2, f7_2));
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f2_2, f6));
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f3_2, f5_2));
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f4, f4));
    h8 = _mm256_add_epi64(h8, _mm256_mul_epu32(f9, f9_38));

    __m256i h9 = _mm256_mul_epu32(f0_2, f9);
    h9 = _mm256_add_epi64(h9, _mm256_mul_epu32(f1_2, f8));
    h9 = _mm256_add_epi64(h9, _mm256_mul_epu32(f2_2, f7));
    h9 = _mm256_add_epi64(h9, _mm256_mul_epu32(f3_2, f6));
    h9 = _mm256_add_epi64(h9, _mm256_mul_epu32(f4_2, f5));

    h->v[0] = h0;
    h->v[1] = h1;
    h->v[2] = h2;
    h->v[3] = h3;
    h->v[4] = h4;
    h->v[5] = h5;
    h->v[6] = h6;
    h->v[7] = h7;
    h->v[8] = h8;
    h->v[9] = h9;

    fp10x4_carry(h);
}

/**
 * @brief 4-way double-squaring: h = 2 * f^2 (mod 2^255-19).
 */
static FP10X4_FORCE_INLINE void fp10x4_sq2(fp10x4 *h, const fp10x4 *f)
{
    fp10x4_sq(h, f);
    for (int i = 0; i < 10; i++)
        h->v[i] = _mm256_slli_epi64(h->v[i], 1);
    fp10x4_carry(h);
}

/**
 * @brief 4-way conditional move: if mask lane is all-ones, copy src lane.
 *
 * mask should be per-lane all-zeros or all-ones (from cmpeq).
 */
static FP10X4_FORCE_INLINE void fp10x4_cmov(fp10x4 *t, const fp10x4 *u, __m256i mask)
{
    for (int i = 0; i < 10; i++)
        t->v[i] = _mm256_blendv_epi8(t->v[i], u->v[i], mask);
}

/**
 * @brief Pack a single fp10 into one lane of a fp10x4.
 *
 * Sets lane `lane` of each register in `out` to the corresponding limb of `in`.
 * Other lanes are unchanged. lane must be 0..3.
 */
static FP10X4_FORCE_INLINE void fp10x4_insert_lane(fp10x4 *out, const fp10 in, int lane)
{
    alignas(32) int64_t tmp[4];
    for (int i = 0; i < 10; i++)
    {
        _mm256_store_si256((__m256i *)tmp, out->v[i]);
        tmp[lane] = in[i];
        out->v[i] = _mm256_load_si256((const __m256i *)tmp);
    }
}

/**
 * @brief Extract one lane from a fp10x4 into a scalar fp10.
 */
static FP10X4_FORCE_INLINE void fp10x4_extract_lane(fp10 out, const fp10x4 *in, int lane)
{
    alignas(32) int64_t tmp[4];
    for (int i = 0; i < 10; i++)
    {
        _mm256_store_si256((__m256i *)tmp, in->v[i]);
        out[i] = tmp[lane];
    }
}

/**
 * @brief Pack 4 fp10 values into a fp10x4.
 */
static FP10X4_FORCE_INLINE void fp10x4_pack(fp10x4 *out, const fp10 a, const fp10 b, const fp10 c, const fp10 d)
{
    for (int i = 0; i < 10; i++)
        out->v[i] = _mm256_set_epi64x(d[i], c[i], b[i], a[i]);
}

/**
 * @brief Unpack a fp10x4 into 4 fp10 values.
 */
static FP10X4_FORCE_INLINE void fp10x4_unpack(fp10 a, fp10 b, fp10 c, fp10 d, const fp10x4 *in)
{
    alignas(32) int64_t tmp[4];
    for (int i = 0; i < 10; i++)
    {
        _mm256_store_si256((__m256i *)tmp, in->v[i]);
        a[i] = tmp[0];
        b[i] = tmp[1];
        c[i] = tmp[2];
        d[i] = tmp[3];
    }
}

#endif // RANSHAW_X64_AVX2_FP10X4_AVX2_H
