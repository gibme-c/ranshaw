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
 * @file fq10x4_avx2.h
 * @brief 4-way parallel radix-2^25.5 Fq field element operations using AVX2.
 *
 * This is the Fq field arithmetic layer for 4-way batch operations over the
 * Crandall prime q = 2^255 - gamma (gamma ~127 bits). Each fq10x4 holds 4
 * independent field elements packed horizontally into AVX2 registers -- one
 * element per 64-bit lane, 10 registers per fq10x4.
 *
 * KEY DIFFERENCE FROM fp10x4 (Ed25519):
 * The Fp multiplication uses inline x19 folding because 19 is a single limb.
 * For Fq, gamma has 5 limbs in radix-2^25.5, so inline folding is impossible.
 * Instead, the multiplication produces 19 __m256i accumulators (the full
 * 10x10 schoolbook product), then performs Crandall reduction:
 *   1. Carry-propagate all 19 accumulators (positions 0..18)
 *   2. First gamma fold: convolve positions 10..18+ with gamma, fold to 0..13
 *   3. Carry-propagate positions 0..13
 *   4. Second gamma fold: convolve positions 10..14 with gamma, fold to 0..8
 *   5. Final carry-propagate with gamma wrap at limb 9
 *
 * The radix-2^25.5 offset correction applies in both the schoolbook and the
 * gamma folds: when BOTH source position and gamma index are odd, the product
 * must be doubled.
 *
 * Subtraction uses a 2*q bias (different values per limb) to keep results
 * non-negative, followed by carry propagation with gamma fold.
 */

#ifndef RANSHAW_X64_AVX2_FQ10X4_AVX2_H
#define RANSHAW_X64_AVX2_FQ10X4_AVX2_H

#include "portable/fq25.h"
#include "ranshaw_platform.h"
#include "x64/avx2/fq10_avx2.h"

#include <immintrin.h>

#if defined(_MSC_VER)
#define FQ10X4_FORCE_INLINE __forceinline
#else
#define FQ10X4_FORCE_INLINE inline __attribute__((always_inline))
#endif

/**
 * @brief 4-way parallel Fq field element type: 10 __m256i registers.
 *
 * v[i] holds limb i of 4 independent field elements in the 4 x 64-bit lanes.
 * Even limbs (0,2,4,6,8) are 26-bit, odd limbs (1,3,5,7,9) are 25-bit.
 */
typedef struct
{
    __m256i v[10];
} fq10x4;

/* ---- Constants as inline functions (avoids MSVC narrowing warnings) ---- */

static inline __m256i fq10x4_mask26(void)
{
    return _mm256_set1_epi64x((1LL << 26) - 1);
}
static inline __m256i fq10x4_mask25(void)
{
    return _mm256_set1_epi64x((1LL << 25) - 1);
}

/* Gamma limbs broadcast */
static inline __m256i fq10x4_gamma0(void)
{
    return _mm256_set1_epi64x(GAMMA_25[0]);
}
static inline __m256i fq10x4_gamma1(void)
{
    return _mm256_set1_epi64x(GAMMA_25[1]);
}
static inline __m256i fq10x4_gamma2(void)
{
    return _mm256_set1_epi64x(GAMMA_25[2]);
}
static inline __m256i fq10x4_gamma3(void)
{
    return _mm256_set1_epi64x(GAMMA_25[3]);
}
static inline __m256i fq10x4_gamma4(void)
{
    return _mm256_set1_epi64x(GAMMA_25[4]);
}

/* Pre-doubled odd gamma limbs for offset correction in gamma fold */
static inline __m256i fq10x4_gamma1_2(void)
{
    return _mm256_set1_epi64x(2 * (int64_t)GAMMA_25[1]);
}
static inline __m256i fq10x4_gamma3_2(void)
{
    return _mm256_set1_epi64x(2 * (int64_t)GAMMA_25[3]);
}

/*
 * 8*Q_25 bias values for subtraction. In radix-2^25.5:
 *
 * CRITICAL: Fq = 2^255 - gamma, where gamma ~ 2^127. This means the lower
 * limbs of q in radix-2^25.5 are much smaller than their radix capacity.
 * A 2q bias is INSUFFICIENT because 2q's lower even limbs (38.8M, 47.6M)
 * are smaller than the max canonical 26-bit value (67.1M = 2^26-1), and
 * a 4q bias (77.5M, 95.2M) is insufficient for 27-bit limbs that arise
 * from a single fq10x4_add (e.g., 2*V in add-2007-bl). We use 8q bias:
 *   {155073784, 228910320, 190344880, 187554136, 401600256,
 *    268435448, 536870904, 268435448, 536870904, 268435448}
 *
 * 8q[0] = 155M > 134M (max 27-bit), so all limbs safely absorb single-add
 * non-canonical subtrahends. For 28-bit inputs from double-chained adds,
 * callers must normalize with fq10x4_carry_gamma before subtraction.
 *
 * This is the radix-2^25.5 analogue of the 5x51 representation needing
 * 8q bias (EIGHT_Q_51) instead of 4q for the same reason.
 */
static inline __m256i fq10x4_bias0(void)
{
    return _mm256_set1_epi64x(8LL * Q_25[0]);
}
static inline __m256i fq10x4_bias1(void)
{
    return _mm256_set1_epi64x(8LL * Q_25[1]);
}
static inline __m256i fq10x4_bias2(void)
{
    return _mm256_set1_epi64x(8LL * Q_25[2]);
}
static inline __m256i fq10x4_bias3(void)
{
    return _mm256_set1_epi64x(8LL * Q_25[3]);
}
static inline __m256i fq10x4_bias4(void)
{
    return _mm256_set1_epi64x(8LL * Q_25[4]);
}
static inline __m256i fq10x4_bias_odd_upper(void)
{
    return _mm256_set1_epi64x(8LL * Q_25[5]);
}
static inline __m256i fq10x4_bias_even_upper(void)
{
    return _mm256_set1_epi64x(8LL * Q_25[6]);
}

/* Shorthand macros */
#define FQ10X4_MASK26 fq10x4_mask26()
#define FQ10X4_MASK25 fq10x4_mask25()
#define FQ10X4_GAMMA0 fq10x4_gamma0()
#define FQ10X4_GAMMA1 fq10x4_gamma1()
#define FQ10X4_GAMMA2 fq10x4_gamma2()
#define FQ10X4_GAMMA3 fq10x4_gamma3()
#define FQ10X4_GAMMA4 fq10x4_gamma4()
#define FQ10X4_GAMMA1_2 fq10x4_gamma1_2()
#define FQ10X4_GAMMA3_2 fq10x4_gamma3_2()

/**
 * @brief Zero all 4 field elements.
 */
static FQ10X4_FORCE_INLINE void fq10x4_0(fq10x4 *h)
{
    const __m256i z = _mm256_setzero_si256();
    for (int i = 0; i < 10; i++)
        h->v[i] = z;
}

/**
 * @brief Set all 4 field elements to 1.
 */
static FQ10X4_FORCE_INLINE void fq10x4_1(fq10x4 *h)
{
    const __m256i z = _mm256_setzero_si256();
    h->v[0] = _mm256_set1_epi64x(1);
    for (int i = 1; i < 10; i++)
        h->v[i] = z;
}

/**
 * @brief Copy: h = f.
 */
static FQ10X4_FORCE_INLINE void fq10x4_copy(fq10x4 *h, const fq10x4 *f)
{
    for (int i = 0; i < 10; i++)
        h->v[i] = f->v[i];
}

/**
 * @brief Addition: h = f + g (no carry propagation).
 */
static FQ10X4_FORCE_INLINE void fq10x4_add(fq10x4 *h, const fq10x4 *f, const fq10x4 *g)
{
    for (int i = 0; i < 10; i++)
        h->v[i] = _mm256_add_epi64(f->v[i], g->v[i]);
}

/**
 * @brief Carry propagation with gamma fold at limb 9.
 *
 * Carries 0->1->...->9, then folds carry out of limb 9 via gamma
 * multiplication back into limbs 0..4, then re-carries 0..4.
 */
static FQ10X4_FORCE_INLINE void fq10x4_carry_gamma(fq10x4 *h)
{
    const __m256i mask26 = FQ10X4_MASK26;
    const __m256i mask25 = FQ10X4_MASK25;
    __m256i c;

    /* Linear carry chain 0 -> 9 */
    c = _mm256_srli_epi64(h->v[0], 26);
    h->v[1] = _mm256_add_epi64(h->v[1], c);
    h->v[0] = _mm256_and_si256(h->v[0], mask26);

    c = _mm256_srli_epi64(h->v[1], 25);
    h->v[2] = _mm256_add_epi64(h->v[2], c);
    h->v[1] = _mm256_and_si256(h->v[1], mask25);

    c = _mm256_srli_epi64(h->v[2], 26);
    h->v[3] = _mm256_add_epi64(h->v[3], c);
    h->v[2] = _mm256_and_si256(h->v[2], mask26);

    c = _mm256_srli_epi64(h->v[3], 25);
    h->v[4] = _mm256_add_epi64(h->v[4], c);
    h->v[3] = _mm256_and_si256(h->v[3], mask25);

    c = _mm256_srli_epi64(h->v[4], 26);
    h->v[5] = _mm256_add_epi64(h->v[5], c);
    h->v[4] = _mm256_and_si256(h->v[4], mask26);

    c = _mm256_srli_epi64(h->v[5], 25);
    h->v[6] = _mm256_add_epi64(h->v[6], c);
    h->v[5] = _mm256_and_si256(h->v[5], mask25);

    c = _mm256_srli_epi64(h->v[6], 26);
    h->v[7] = _mm256_add_epi64(h->v[7], c);
    h->v[6] = _mm256_and_si256(h->v[6], mask26);

    c = _mm256_srli_epi64(h->v[7], 25);
    h->v[8] = _mm256_add_epi64(h->v[8], c);
    h->v[7] = _mm256_and_si256(h->v[7], mask25);

    c = _mm256_srli_epi64(h->v[8], 26);
    h->v[9] = _mm256_add_epi64(h->v[9], c);
    h->v[8] = _mm256_and_si256(h->v[8], mask26);

    c = _mm256_srli_epi64(h->v[9], 25);
    h->v[9] = _mm256_and_si256(h->v[9], mask25);

    /* Gamma fold: c * gamma[0..GAMMA_25_LIMBS-1] into limbs 0..GAMMA_25_LIMBS-1 */
    for (int j = 0; j < GAMMA_25_LIMBS; j++)
        h->v[j] = _mm256_add_epi64(h->v[j], _mm256_mul_epu32(c, _mm256_set1_epi64x(GAMMA_25[j])));

    /* Re-carry limbs 0..4 */
    c = _mm256_srli_epi64(h->v[0], 26);
    h->v[1] = _mm256_add_epi64(h->v[1], c);
    h->v[0] = _mm256_and_si256(h->v[0], mask26);

    c = _mm256_srli_epi64(h->v[1], 25);
    h->v[2] = _mm256_add_epi64(h->v[2], c);
    h->v[1] = _mm256_and_si256(h->v[1], mask25);

    c = _mm256_srli_epi64(h->v[2], 26);
    h->v[3] = _mm256_add_epi64(h->v[3], c);
    h->v[2] = _mm256_and_si256(h->v[2], mask26);

    c = _mm256_srli_epi64(h->v[3], 25);
    h->v[4] = _mm256_add_epi64(h->v[4], c);
    h->v[3] = _mm256_and_si256(h->v[3], mask25);

    /* Carry limb 4 -> 5: the gamma fold adds c*gamma[4] (up to ~30 bits when c is
       large from mul/sq output), leaving limb 4 far above 26 bits. Without this carry,
       fq10x4_sub's 2q bias (100,400,064 ≈ 27 bits) cannot absorb the non-canonical
       limb 4 as a subtrahend, causing unsigned underflow and garbage carries. */
    c = _mm256_srli_epi64(h->v[4], 26);
    h->v[5] = _mm256_add_epi64(h->v[5], c);
    h->v[4] = _mm256_and_si256(h->v[4], mask26);
}

/**
 * @brief Subtraction: h = f - g with 8*q bias + carry with gamma fold.
 */
static FQ10X4_FORCE_INLINE void fq10x4_sub(fq10x4 *h, const fq10x4 *f, const fq10x4 *g)
{
    /* Add 8*q bias and subtract g */
    h->v[0] = _mm256_sub_epi64(_mm256_add_epi64(f->v[0], fq10x4_bias0()), g->v[0]);
    h->v[1] = _mm256_sub_epi64(_mm256_add_epi64(f->v[1], fq10x4_bias1()), g->v[1]);
    h->v[2] = _mm256_sub_epi64(_mm256_add_epi64(f->v[2], fq10x4_bias2()), g->v[2]);
    h->v[3] = _mm256_sub_epi64(_mm256_add_epi64(f->v[3], fq10x4_bias3()), g->v[3]);
    h->v[4] = _mm256_sub_epi64(_mm256_add_epi64(f->v[4], fq10x4_bias4()), g->v[4]);
    h->v[5] = _mm256_sub_epi64(_mm256_add_epi64(f->v[5], fq10x4_bias_odd_upper()), g->v[5]);
    h->v[6] = _mm256_sub_epi64(_mm256_add_epi64(f->v[6], fq10x4_bias_even_upper()), g->v[6]);
    h->v[7] = _mm256_sub_epi64(_mm256_add_epi64(f->v[7], fq10x4_bias_odd_upper()), g->v[7]);
    h->v[8] = _mm256_sub_epi64(_mm256_add_epi64(f->v[8], fq10x4_bias_even_upper()), g->v[8]);
    h->v[9] = _mm256_sub_epi64(_mm256_add_epi64(f->v[9], fq10x4_bias_odd_upper()), g->v[9]);

    fq10x4_carry_gamma(h);
}

/**
 * @brief Negation: h = -f (mod q).
 */
static FQ10X4_FORCE_INLINE void fq10x4_neg(fq10x4 *h, const fq10x4 *f)
{
    fq10x4 zero;
    fq10x4_0(&zero);
    fq10x4_sub(h, &zero, f);
}

/**
 * @brief 4-way schoolbook multiplication: h = f * g (mod 2^255 - gamma).
 *
 * CRITICAL: Unlike fp10x4_mul, this cannot use inline x19 folding because
 * gamma has 5 limbs. Instead:
 *   1. Full 10x10 schoolbook -> 19 __m256i accumulators
 *   2. Carry-propagate positions 0..18 (linear, no wrap)
 *   3. First gamma fold: convolve positions 10..18+ with gamma
 *   4. Carry-propagate positions 0..13
 *   5. Second gamma fold: convolve positions 10..14 with gamma
 *   6. Final carry with gamma wrap at limb 9
 *
 * The schoolbook uses _mm256_mul_epu32 for 32x32->64 unsigned products.
 * Pre-doubled odd-indexed f limbs handle the radix-2^25.5 offset correction.
 */
static FQ10X4_FORCE_INLINE void fq10x4_mul(fq10x4 *h, const fq10x4 *f, const fq10x4 *g)
{
    const __m256i f0 = f->v[0], f1 = f->v[1], f2 = f->v[2], f3 = f->v[3], f4 = f->v[4];
    const __m256i f5 = f->v[5], f6 = f->v[6], f7 = f->v[7], f8 = f->v[8], f9 = f->v[9];
    const __m256i g0 = g->v[0], g1 = g->v[1], g2 = g->v[2], g3 = g->v[3], g4 = g->v[4];
    const __m256i g5 = g->v[5], g6 = g->v[6], g7 = g->v[7], g8 = g->v[8], g9 = g->v[9];

    /* Pre-double odd-indexed f limbs for radix-2^25.5 offset correction */
    const __m256i f1_2 = _mm256_slli_epi64(f1, 1);
    const __m256i f3_2 = _mm256_slli_epi64(f3, 1);
    const __m256i f5_2 = _mm256_slli_epi64(f5, 1);
    const __m256i f7_2 = _mm256_slli_epi64(f7, 1);
    const __m256i f9_2 = _mm256_slli_epi64(f9, 1);

    /*
     * Full 10x10 schoolbook: 19 output accumulators.
     * NO inline folding -- all products go to their natural position i+j.
     * fi_2 used when both i and j are odd.
     */

    /* t0 = f0*g0 */
    __m256i t0 = _mm256_mul_epu32(f0, g0);

    /* t1 = f0*g1 + f1*g0 */
    __m256i t1 = _mm256_mul_epu32(f0, g1);
    t1 = _mm256_add_epi64(t1, _mm256_mul_epu32(f1, g0));

    /* t2 = f0*g2 + f1_2*g1 + f2*g0 */
    __m256i t2 = _mm256_mul_epu32(f0, g2);
    t2 = _mm256_add_epi64(t2, _mm256_mul_epu32(f1_2, g1));
    t2 = _mm256_add_epi64(t2, _mm256_mul_epu32(f2, g0));

    /* t3 = f0*g3 + f1*g2 + f2*g1 + f3*g0 */
    __m256i t3 = _mm256_mul_epu32(f0, g3);
    t3 = _mm256_add_epi64(t3, _mm256_mul_epu32(f1, g2));
    t3 = _mm256_add_epi64(t3, _mm256_mul_epu32(f2, g1));
    t3 = _mm256_add_epi64(t3, _mm256_mul_epu32(f3, g0));

    /* t4 = f0*g4 + f1_2*g3 + f2*g2 + f3_2*g1 + f4*g0 */
    __m256i t4 = _mm256_mul_epu32(f0, g4);
    t4 = _mm256_add_epi64(t4, _mm256_mul_epu32(f1_2, g3));
    t4 = _mm256_add_epi64(t4, _mm256_mul_epu32(f2, g2));
    t4 = _mm256_add_epi64(t4, _mm256_mul_epu32(f3_2, g1));
    t4 = _mm256_add_epi64(t4, _mm256_mul_epu32(f4, g0));

    /* t5 = f0*g5 + f1*g4 + f2*g3 + f3*g2 + f4*g1 + f5*g0 */
    __m256i t5 = _mm256_mul_epu32(f0, g5);
    t5 = _mm256_add_epi64(t5, _mm256_mul_epu32(f1, g4));
    t5 = _mm256_add_epi64(t5, _mm256_mul_epu32(f2, g3));
    t5 = _mm256_add_epi64(t5, _mm256_mul_epu32(f3, g2));
    t5 = _mm256_add_epi64(t5, _mm256_mul_epu32(f4, g1));
    t5 = _mm256_add_epi64(t5, _mm256_mul_epu32(f5, g0));

    /* t6 = f0*g6 + f1_2*g5 + f2*g4 + f3_2*g3 + f4*g2 + f5_2*g1 + f6*g0 */
    __m256i t6 = _mm256_mul_epu32(f0, g6);
    t6 = _mm256_add_epi64(t6, _mm256_mul_epu32(f1_2, g5));
    t6 = _mm256_add_epi64(t6, _mm256_mul_epu32(f2, g4));
    t6 = _mm256_add_epi64(t6, _mm256_mul_epu32(f3_2, g3));
    t6 = _mm256_add_epi64(t6, _mm256_mul_epu32(f4, g2));
    t6 = _mm256_add_epi64(t6, _mm256_mul_epu32(f5_2, g1));
    t6 = _mm256_add_epi64(t6, _mm256_mul_epu32(f6, g0));

    /* t7 = f0*g7 + f1*g6 + f2*g5 + f3*g4 + f4*g3 + f5*g2 + f6*g1 + f7*g0 */
    __m256i t7 = _mm256_mul_epu32(f0, g7);
    t7 = _mm256_add_epi64(t7, _mm256_mul_epu32(f1, g6));
    t7 = _mm256_add_epi64(t7, _mm256_mul_epu32(f2, g5));
    t7 = _mm256_add_epi64(t7, _mm256_mul_epu32(f3, g4));
    t7 = _mm256_add_epi64(t7, _mm256_mul_epu32(f4, g3));
    t7 = _mm256_add_epi64(t7, _mm256_mul_epu32(f5, g2));
    t7 = _mm256_add_epi64(t7, _mm256_mul_epu32(f6, g1));
    t7 = _mm256_add_epi64(t7, _mm256_mul_epu32(f7, g0));

    /* t8 = f0*g8 + f1_2*g7 + f2*g6 + f3_2*g5 + f4*g4 + f5_2*g3 + f6*g2 + f7_2*g1 + f8*g0 */
    __m256i t8 = _mm256_mul_epu32(f0, g8);
    t8 = _mm256_add_epi64(t8, _mm256_mul_epu32(f1_2, g7));
    t8 = _mm256_add_epi64(t8, _mm256_mul_epu32(f2, g6));
    t8 = _mm256_add_epi64(t8, _mm256_mul_epu32(f3_2, g5));
    t8 = _mm256_add_epi64(t8, _mm256_mul_epu32(f4, g4));
    t8 = _mm256_add_epi64(t8, _mm256_mul_epu32(f5_2, g3));
    t8 = _mm256_add_epi64(t8, _mm256_mul_epu32(f6, g2));
    t8 = _mm256_add_epi64(t8, _mm256_mul_epu32(f7_2, g1));
    t8 = _mm256_add_epi64(t8, _mm256_mul_epu32(f8, g0));

    /* t9 = f0*g9 + f1*g8 + f2*g7 + f3*g6 + f4*g5 + f5*g4 + f6*g3 + f7*g2 + f8*g1 + f9*g0 */
    __m256i t9 = _mm256_mul_epu32(f0, g9);
    t9 = _mm256_add_epi64(t9, _mm256_mul_epu32(f1, g8));
    t9 = _mm256_add_epi64(t9, _mm256_mul_epu32(f2, g7));
    t9 = _mm256_add_epi64(t9, _mm256_mul_epu32(f3, g6));
    t9 = _mm256_add_epi64(t9, _mm256_mul_epu32(f4, g5));
    t9 = _mm256_add_epi64(t9, _mm256_mul_epu32(f5, g4));
    t9 = _mm256_add_epi64(t9, _mm256_mul_epu32(f6, g3));
    t9 = _mm256_add_epi64(t9, _mm256_mul_epu32(f7, g2));
    t9 = _mm256_add_epi64(t9, _mm256_mul_epu32(f8, g1));
    t9 = _mm256_add_epi64(t9, _mm256_mul_epu32(f9, g0));

    /* t10 = f1_2*g9 + f2*g8 + f3_2*g7 + f4*g6 + f5_2*g5 + f6*g4 + f7_2*g3 + f8*g2 + f9_2*g1 */
    __m256i t10 = _mm256_mul_epu32(f1_2, g9);
    t10 = _mm256_add_epi64(t10, _mm256_mul_epu32(f2, g8));
    t10 = _mm256_add_epi64(t10, _mm256_mul_epu32(f3_2, g7));
    t10 = _mm256_add_epi64(t10, _mm256_mul_epu32(f4, g6));
    t10 = _mm256_add_epi64(t10, _mm256_mul_epu32(f5_2, g5));
    t10 = _mm256_add_epi64(t10, _mm256_mul_epu32(f6, g4));
    t10 = _mm256_add_epi64(t10, _mm256_mul_epu32(f7_2, g3));
    t10 = _mm256_add_epi64(t10, _mm256_mul_epu32(f8, g2));
    t10 = _mm256_add_epi64(t10, _mm256_mul_epu32(f9_2, g1));

    /* t11 = f2*g9 + f3*g8 + f4*g7 + f5*g6 + f6*g5 + f7*g4 + f8*g3 + f9*g2 */
    __m256i t11 = _mm256_mul_epu32(f2, g9);
    t11 = _mm256_add_epi64(t11, _mm256_mul_epu32(f3, g8));
    t11 = _mm256_add_epi64(t11, _mm256_mul_epu32(f4, g7));
    t11 = _mm256_add_epi64(t11, _mm256_mul_epu32(f5, g6));
    t11 = _mm256_add_epi64(t11, _mm256_mul_epu32(f6, g5));
    t11 = _mm256_add_epi64(t11, _mm256_mul_epu32(f7, g4));
    t11 = _mm256_add_epi64(t11, _mm256_mul_epu32(f8, g3));
    t11 = _mm256_add_epi64(t11, _mm256_mul_epu32(f9, g2));

    /* t12 = f3_2*g9 + f4*g8 + f5_2*g7 + f6*g6 + f7_2*g5 + f8*g4 + f9_2*g3 */
    __m256i t12 = _mm256_mul_epu32(f3_2, g9);
    t12 = _mm256_add_epi64(t12, _mm256_mul_epu32(f4, g8));
    t12 = _mm256_add_epi64(t12, _mm256_mul_epu32(f5_2, g7));
    t12 = _mm256_add_epi64(t12, _mm256_mul_epu32(f6, g6));
    t12 = _mm256_add_epi64(t12, _mm256_mul_epu32(f7_2, g5));
    t12 = _mm256_add_epi64(t12, _mm256_mul_epu32(f8, g4));
    t12 = _mm256_add_epi64(t12, _mm256_mul_epu32(f9_2, g3));

    /* t13 = f4*g9 + f5*g8 + f6*g7 + f7*g6 + f8*g5 + f9*g4 */
    __m256i t13 = _mm256_mul_epu32(f4, g9);
    t13 = _mm256_add_epi64(t13, _mm256_mul_epu32(f5, g8));
    t13 = _mm256_add_epi64(t13, _mm256_mul_epu32(f6, g7));
    t13 = _mm256_add_epi64(t13, _mm256_mul_epu32(f7, g6));
    t13 = _mm256_add_epi64(t13, _mm256_mul_epu32(f8, g5));
    t13 = _mm256_add_epi64(t13, _mm256_mul_epu32(f9, g4));

    /* t14 = f5_2*g9 + f6*g8 + f7_2*g7 + f8*g6 + f9_2*g5 */
    __m256i t14 = _mm256_mul_epu32(f5_2, g9);
    t14 = _mm256_add_epi64(t14, _mm256_mul_epu32(f6, g8));
    t14 = _mm256_add_epi64(t14, _mm256_mul_epu32(f7_2, g7));
    t14 = _mm256_add_epi64(t14, _mm256_mul_epu32(f8, g6));
    t14 = _mm256_add_epi64(t14, _mm256_mul_epu32(f9_2, g5));

    /* t15 = f6*g9 + f7*g8 + f8*g7 + f9*g6 */
    __m256i t15 = _mm256_mul_epu32(f6, g9);
    t15 = _mm256_add_epi64(t15, _mm256_mul_epu32(f7, g8));
    t15 = _mm256_add_epi64(t15, _mm256_mul_epu32(f8, g7));
    t15 = _mm256_add_epi64(t15, _mm256_mul_epu32(f9, g6));

    /* t16 = f7_2*g9 + f8*g8 + f9_2*g7 */
    __m256i t16 = _mm256_mul_epu32(f7_2, g9);
    t16 = _mm256_add_epi64(t16, _mm256_mul_epu32(f8, g8));
    t16 = _mm256_add_epi64(t16, _mm256_mul_epu32(f9_2, g7));

    /* t17 = f8*g9 + f9*g8 */
    __m256i t17 = _mm256_mul_epu32(f8, g9);
    t17 = _mm256_add_epi64(t17, _mm256_mul_epu32(f9, g8));

    /* t18 = f9_2*g9 */
    __m256i t18 = _mm256_mul_epu32(f9_2, g9);

    /*
     * Crandall reduction: carry-propagate 0..18, gamma fold, repeat.
     */

    const __m256i mask26 = FQ10X4_MASK26;
    const __m256i mask25 = FQ10X4_MASK25;
    __m256i c;

    /* Step 1: Carry-propagate t0..t18 (linear chain, NO wrap) */
    c = _mm256_srli_epi64(t0, 26);
    t1 = _mm256_add_epi64(t1, c);
    t0 = _mm256_and_si256(t0, mask26);

    c = _mm256_srli_epi64(t1, 25);
    t2 = _mm256_add_epi64(t2, c);
    t1 = _mm256_and_si256(t1, mask25);

    c = _mm256_srli_epi64(t2, 26);
    t3 = _mm256_add_epi64(t3, c);
    t2 = _mm256_and_si256(t2, mask26);

    c = _mm256_srli_epi64(t3, 25);
    t4 = _mm256_add_epi64(t4, c);
    t3 = _mm256_and_si256(t3, mask25);

    c = _mm256_srli_epi64(t4, 26);
    t5 = _mm256_add_epi64(t5, c);
    t4 = _mm256_and_si256(t4, mask26);

    c = _mm256_srli_epi64(t5, 25);
    t6 = _mm256_add_epi64(t6, c);
    t5 = _mm256_and_si256(t5, mask25);

    c = _mm256_srli_epi64(t6, 26);
    t7 = _mm256_add_epi64(t7, c);
    t6 = _mm256_and_si256(t6, mask26);

    c = _mm256_srli_epi64(t7, 25);
    t8 = _mm256_add_epi64(t8, c);
    t7 = _mm256_and_si256(t7, mask25);

    c = _mm256_srli_epi64(t8, 26);
    t9 = _mm256_add_epi64(t9, c);
    t8 = _mm256_and_si256(t8, mask26);

    c = _mm256_srli_epi64(t9, 25);
    t10 = _mm256_add_epi64(t10, c);
    t9 = _mm256_and_si256(t9, mask25);

    c = _mm256_srli_epi64(t10, 26);
    t11 = _mm256_add_epi64(t11, c);
    t10 = _mm256_and_si256(t10, mask26);

    c = _mm256_srli_epi64(t11, 25);
    t12 = _mm256_add_epi64(t12, c);
    t11 = _mm256_and_si256(t11, mask25);

    c = _mm256_srli_epi64(t12, 26);
    t13 = _mm256_add_epi64(t13, c);
    t12 = _mm256_and_si256(t12, mask26);

    c = _mm256_srli_epi64(t13, 25);
    t14 = _mm256_add_epi64(t14, c);
    t13 = _mm256_and_si256(t13, mask25);

    c = _mm256_srli_epi64(t14, 26);
    t15 = _mm256_add_epi64(t15, c);
    t14 = _mm256_and_si256(t14, mask26);

    c = _mm256_srli_epi64(t15, 25);
    t16 = _mm256_add_epi64(t16, c);
    t15 = _mm256_and_si256(t15, mask25);

    c = _mm256_srli_epi64(t16, 26);
    t17 = _mm256_add_epi64(t17, c);
    t16 = _mm256_and_si256(t16, mask26);

    c = _mm256_srli_epi64(t17, 25);
    t18 = _mm256_add_epi64(t18, c);
    t17 = _mm256_and_si256(t17, mask25);

    c = _mm256_srli_epi64(t18, 26);
    t18 = _mm256_and_si256(t18, mask26);
    __m256i t19 = c;

    /*
     * Step 2: First gamma fold.
     * Convolve t[10..19] with gamma[0..GAMMA_25_LIMBS-1], add to positions 0..(9+GAMMA_25_LIMBS-1).
     *
     * Offset correction: when both source position (10+k) and gamma index j are odd,
     * the product occupies an even output position and must be doubled.
     */
    const __m256i tv[10] = {t10, t11, t12, t13, t14, t15, t16, t17, t18, t19};
    __m256i hv[10 + GAMMA_25_LIMBS - 1];
    {
        const __m256i low[10] = {t0, t1, t2, t3, t4, t5, t6, t7, t8, t9};
        for (int n = 0; n < 10; n++)
            hv[n] = low[n];
    }
    for (int n = 10; n < 10 + GAMMA_25_LIMBS - 1; n++)
        hv[n] = _mm256_setzero_si256();
    for (int k = 0; k < 10; k++)
    {
        for (int j = 0; j < GAMMA_25_LIMBS; j++)
        {
            /* Offset correction: source position is (10+k), gamma index is j.
               When both (10+k) and j are odd, use 2*gamma[j]. Since 10+k is odd
               iff k is odd, the condition is: k odd AND j odd. */
            const int need_double = (k & 1) & (j & 1);
            const __m256i gval = _mm256_set1_epi64x(need_double ? 2 * (int64_t)GAMMA_25[j] : (int64_t)GAMMA_25[j]);
            hv[k + j] = _mm256_add_epi64(hv[k + j], _mm256_mul_epu32(tv[k], gval));
        }
    }
    __m256i h0 = hv[0], h1 = hv[1], h2 = hv[2], h3 = hv[3], h4 = hv[4];
    __m256i h5 = hv[5], h6 = hv[6], h7 = hv[7], h8 = hv[8], h9 = hv[9];
    __m256i h10 = hv[10], h11 = hv[11], h12 = hv[12], h13 = hv[13];

    /*
     * Step 3: Carry-propagate h0..h9, overflow into h10..h13.
     */
    c = _mm256_srli_epi64(h0, 26);
    h1 = _mm256_add_epi64(h1, c);
    h0 = _mm256_and_si256(h0, mask26);

    c = _mm256_srli_epi64(h1, 25);
    h2 = _mm256_add_epi64(h2, c);
    h1 = _mm256_and_si256(h1, mask25);

    c = _mm256_srli_epi64(h2, 26);
    h3 = _mm256_add_epi64(h3, c);
    h2 = _mm256_and_si256(h2, mask26);

    c = _mm256_srli_epi64(h3, 25);
    h4 = _mm256_add_epi64(h4, c);
    h3 = _mm256_and_si256(h3, mask25);

    c = _mm256_srli_epi64(h4, 26);
    h5 = _mm256_add_epi64(h5, c);
    h4 = _mm256_and_si256(h4, mask26);

    c = _mm256_srli_epi64(h5, 25);
    h6 = _mm256_add_epi64(h6, c);
    h5 = _mm256_and_si256(h5, mask25);

    c = _mm256_srli_epi64(h6, 26);
    h7 = _mm256_add_epi64(h7, c);
    h6 = _mm256_and_si256(h6, mask26);

    c = _mm256_srli_epi64(h7, 25);
    h8 = _mm256_add_epi64(h8, c);
    h7 = _mm256_and_si256(h7, mask25);

    c = _mm256_srli_epi64(h8, 26);
    h9 = _mm256_add_epi64(h9, c);
    h8 = _mm256_and_si256(h8, mask26);

    c = _mm256_srli_epi64(h9, 25);
    h10 = _mm256_add_epi64(h10, c);
    h9 = _mm256_and_si256(h9, mask25);

    /*
     * Step 4: Carry-propagate h10..h13, extract h14.
     */
    c = _mm256_srli_epi64(h10, 26);
    h11 = _mm256_add_epi64(h11, c);
    h10 = _mm256_and_si256(h10, mask26);

    c = _mm256_srli_epi64(h11, 25);
    h12 = _mm256_add_epi64(h12, c);
    h11 = _mm256_and_si256(h11, mask25);

    c = _mm256_srli_epi64(h12, 26);
    h13 = _mm256_add_epi64(h13, c);
    h12 = _mm256_and_si256(h12, mask26);

    c = _mm256_srli_epi64(h13, 25);
    h13 = _mm256_and_si256(h13, mask25);
    __m256i h14 = c;

    /*
     * Step 5: Second gamma fold: h[10..14] * gamma -> positions 0..8.
     *
     * Same offset correction rule as first fold: when both source position
     * (10+k) and gamma index j are odd, double the gamma coefficient.
     */
    {
        const __m256i ov[5] = {h10, h11, h12, h13, h14};
        __m256i *dst[10] = {&h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7, &h8, &h9};
        for (int k = 0; k < 5; k++)
        {
            for (int j = 0; j < GAMMA_25_LIMBS; j++)
            {
                if (k + j >= 10)
                    break;
                const int need_double = (k & 1) & (j & 1);
                const __m256i gval = _mm256_set1_epi64x(need_double ? 2 * (int64_t)GAMMA_25[j] : (int64_t)GAMMA_25[j]);
                *dst[k + j] = _mm256_add_epi64(*dst[k + j], _mm256_mul_epu32(ov[k], gval));
            }
        }
    }

    /*
     * Step 6: Final carry with gamma fold at limb 9.
     */
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
    fq10x4_carry_gamma(h);
}

/**
 * @brief 4-way squaring: h = f^2 (mod 2^255 - gamma).
 *
 * Uses squaring-specific optimizations (cross-terms doubled, diagonal once).
 * Same Crandall reduction as fq10x4_mul.
 */
static FQ10X4_FORCE_INLINE void fq10x4_sq(fq10x4 *h, const fq10x4 *f)
{
    const __m256i f0 = f->v[0], f1 = f->v[1], f2 = f->v[2], f3 = f->v[3], f4 = f->v[4];
    const __m256i f5 = f->v[5], f6 = f->v[6], f7 = f->v[7], f8 = f->v[8], f9 = f->v[9];

    /* Even-index doubled (cross-term 2x) */
    const __m256i f0_2 = _mm256_slli_epi64(f0, 1);
    const __m256i f2_2 = _mm256_slli_epi64(f2, 1);
    const __m256i f4_2 = _mm256_slli_epi64(f4, 1);
    const __m256i f6_2 = _mm256_slli_epi64(f6, 1);
    const __m256i f8_2 = _mm256_slli_epi64(f8, 1);
    /* Odd-index doubled (cross-term 2x AND offset correction 2x) */
    const __m256i f1_2 = _mm256_slli_epi64(f1, 1);
    const __m256i f3_2 = _mm256_slli_epi64(f3, 1);
    const __m256i f5_2 = _mm256_slli_epi64(f5, 1);
    const __m256i f7_2 = _mm256_slli_epi64(f7, 1);
    const __m256i f9_2 = _mm256_slli_epi64(f9, 1);

    /* Full squaring schoolbook: 19 accumulators */

    /* t0 = f0*f0 */
    __m256i t0 = _mm256_mul_epu32(f0, f0);

    /* t1 = 2*f0*f1 */
    __m256i t1 = _mm256_mul_epu32(f0_2, f1);

    /* t2 = 2*f0*f2 + f1_2*f1 */
    __m256i t2 = _mm256_mul_epu32(f0_2, f2);
    t2 = _mm256_add_epi64(t2, _mm256_mul_epu32(f1_2, f1));

    /* t3 = 2*f0*f3 + 2*f1*f2 */
    __m256i t3 = _mm256_mul_epu32(f0_2, f3);
    t3 = _mm256_add_epi64(t3, _mm256_mul_epu32(f1_2, f2));

    /* t4 = 2*f0*f4 + f1_2*f3_2 + f2*f2 */
    __m256i t4 = _mm256_mul_epu32(f0_2, f4);
    t4 = _mm256_add_epi64(t4, _mm256_mul_epu32(f1_2, f3_2));
    t4 = _mm256_add_epi64(t4, _mm256_mul_epu32(f2, f2));

    /* t5 = 2*f0*f5 + 2*f1*f4 + 2*f2*f3 */
    __m256i t5 = _mm256_mul_epu32(f0_2, f5);
    t5 = _mm256_add_epi64(t5, _mm256_mul_epu32(f1_2, f4));
    t5 = _mm256_add_epi64(t5, _mm256_mul_epu32(f2_2, f3));

    /* t6 = 2*f0*f6 + f1_2*f5_2 + 2*f2*f4 + f3_2*f3 */
    __m256i t6 = _mm256_mul_epu32(f0_2, f6);
    t6 = _mm256_add_epi64(t6, _mm256_mul_epu32(f1_2, f5_2));
    t6 = _mm256_add_epi64(t6, _mm256_mul_epu32(f2_2, f4));
    t6 = _mm256_add_epi64(t6, _mm256_mul_epu32(f3_2, f3));

    /* t7 = 2*f0*f7 + 2*f1*f6 + 2*f2*f5 + 2*f3*f4 */
    __m256i t7 = _mm256_mul_epu32(f0_2, f7);
    t7 = _mm256_add_epi64(t7, _mm256_mul_epu32(f1_2, f6));
    t7 = _mm256_add_epi64(t7, _mm256_mul_epu32(f2_2, f5));
    t7 = _mm256_add_epi64(t7, _mm256_mul_epu32(f3_2, f4));

    /* t8 = 2*f0*f8 + f1_2*f7_2 + 2*f2*f6 + f3_2*f5_2 + f4*f4 */
    __m256i t8 = _mm256_mul_epu32(f0_2, f8);
    t8 = _mm256_add_epi64(t8, _mm256_mul_epu32(f1_2, f7_2));
    t8 = _mm256_add_epi64(t8, _mm256_mul_epu32(f2_2, f6));
    t8 = _mm256_add_epi64(t8, _mm256_mul_epu32(f3_2, f5_2));
    t8 = _mm256_add_epi64(t8, _mm256_mul_epu32(f4, f4));

    /* t9 = 2*f0*f9 + 2*f1*f8 + 2*f2*f7 + 2*f3*f6 + 2*f4*f5 */
    __m256i t9 = _mm256_mul_epu32(f0_2, f9);
    t9 = _mm256_add_epi64(t9, _mm256_mul_epu32(f1_2, f8));
    t9 = _mm256_add_epi64(t9, _mm256_mul_epu32(f2_2, f7));
    t9 = _mm256_add_epi64(t9, _mm256_mul_epu32(f3_2, f6));
    t9 = _mm256_add_epi64(t9, _mm256_mul_epu32(f4_2, f5));

    /* t10 = f1_2*f9_2 + 2*f2*f8 + f3_2*f7_2 + 2*f4*f6 + f5_2*f5 */
    __m256i t10 = _mm256_mul_epu32(f1_2, f9_2);
    t10 = _mm256_add_epi64(t10, _mm256_mul_epu32(f2_2, f8));
    t10 = _mm256_add_epi64(t10, _mm256_mul_epu32(f3_2, f7_2));
    t10 = _mm256_add_epi64(t10, _mm256_mul_epu32(f4_2, f6));
    t10 = _mm256_add_epi64(t10, _mm256_mul_epu32(f5_2, f5));

    /* t11 = 2*f2*f9 + 2*f3*f8 + 2*f4*f7 + 2*f5*f6 */
    __m256i t11 = _mm256_mul_epu32(f2_2, f9);
    t11 = _mm256_add_epi64(t11, _mm256_mul_epu32(f3_2, f8));
    t11 = _mm256_add_epi64(t11, _mm256_mul_epu32(f4_2, f7));
    t11 = _mm256_add_epi64(t11, _mm256_mul_epu32(f5_2, f6));

    /* t12 = f3_2*f9_2 + 2*f4*f8 + f5_2*f7_2 + f6*f6 */
    __m256i t12 = _mm256_mul_epu32(f3_2, f9_2);
    t12 = _mm256_add_epi64(t12, _mm256_mul_epu32(f4_2, f8));
    t12 = _mm256_add_epi64(t12, _mm256_mul_epu32(f5_2, f7_2));
    t12 = _mm256_add_epi64(t12, _mm256_mul_epu32(f6, f6));

    /* t13 = 2*f4*f9 + 2*f5*f8 + 2*f6*f7 */
    __m256i t13 = _mm256_mul_epu32(f4_2, f9);
    t13 = _mm256_add_epi64(t13, _mm256_mul_epu32(f5_2, f8));
    t13 = _mm256_add_epi64(t13, _mm256_mul_epu32(f6_2, f7));

    /* t14 = f5_2*f9_2 + 2*f6*f8 + f7_2*f7 */
    __m256i t14 = _mm256_mul_epu32(f5_2, f9_2);
    t14 = _mm256_add_epi64(t14, _mm256_mul_epu32(f6_2, f8));
    t14 = _mm256_add_epi64(t14, _mm256_mul_epu32(f7_2, f7));

    /* t15 = 2*f6*f9 + 2*f7*f8 */
    __m256i t15 = _mm256_mul_epu32(f6_2, f9);
    t15 = _mm256_add_epi64(t15, _mm256_mul_epu32(f7_2, f8));

    /* t16 = f7_2*f9_2 + f8*f8 */
    __m256i t16 = _mm256_mul_epu32(f7_2, f9_2);
    t16 = _mm256_add_epi64(t16, _mm256_mul_epu32(f8, f8));

    /* t17 = 2*f8*f9 */
    __m256i t17 = _mm256_mul_epu32(f8_2, f9);

    /* t18 = f9_2*f9 */
    __m256i t18 = _mm256_mul_epu32(f9_2, f9);

    /*
     * Crandall reduction (same as in fq10x4_mul, reuse the pattern).
     */
    const __m256i mask26 = FQ10X4_MASK26;
    const __m256i mask25 = FQ10X4_MASK25;
    __m256i c;

    /* Carry-propagate t0..t18 */
    c = _mm256_srli_epi64(t0, 26);
    t1 = _mm256_add_epi64(t1, c);
    t0 = _mm256_and_si256(t0, mask26);
    c = _mm256_srli_epi64(t1, 25);
    t2 = _mm256_add_epi64(t2, c);
    t1 = _mm256_and_si256(t1, mask25);
    c = _mm256_srli_epi64(t2, 26);
    t3 = _mm256_add_epi64(t3, c);
    t2 = _mm256_and_si256(t2, mask26);
    c = _mm256_srli_epi64(t3, 25);
    t4 = _mm256_add_epi64(t4, c);
    t3 = _mm256_and_si256(t3, mask25);
    c = _mm256_srli_epi64(t4, 26);
    t5 = _mm256_add_epi64(t5, c);
    t4 = _mm256_and_si256(t4, mask26);
    c = _mm256_srli_epi64(t5, 25);
    t6 = _mm256_add_epi64(t6, c);
    t5 = _mm256_and_si256(t5, mask25);
    c = _mm256_srli_epi64(t6, 26);
    t7 = _mm256_add_epi64(t7, c);
    t6 = _mm256_and_si256(t6, mask26);
    c = _mm256_srli_epi64(t7, 25);
    t8 = _mm256_add_epi64(t8, c);
    t7 = _mm256_and_si256(t7, mask25);
    c = _mm256_srli_epi64(t8, 26);
    t9 = _mm256_add_epi64(t9, c);
    t8 = _mm256_and_si256(t8, mask26);
    c = _mm256_srli_epi64(t9, 25);
    t10 = _mm256_add_epi64(t10, c);
    t9 = _mm256_and_si256(t9, mask25);
    c = _mm256_srli_epi64(t10, 26);
    t11 = _mm256_add_epi64(t11, c);
    t10 = _mm256_and_si256(t10, mask26);
    c = _mm256_srli_epi64(t11, 25);
    t12 = _mm256_add_epi64(t12, c);
    t11 = _mm256_and_si256(t11, mask25);
    c = _mm256_srli_epi64(t12, 26);
    t13 = _mm256_add_epi64(t13, c);
    t12 = _mm256_and_si256(t12, mask26);
    c = _mm256_srli_epi64(t13, 25);
    t14 = _mm256_add_epi64(t14, c);
    t13 = _mm256_and_si256(t13, mask25);
    c = _mm256_srli_epi64(t14, 26);
    t15 = _mm256_add_epi64(t15, c);
    t14 = _mm256_and_si256(t14, mask26);
    c = _mm256_srli_epi64(t15, 25);
    t16 = _mm256_add_epi64(t16, c);
    t15 = _mm256_and_si256(t15, mask25);
    c = _mm256_srli_epi64(t16, 26);
    t17 = _mm256_add_epi64(t17, c);
    t16 = _mm256_and_si256(t16, mask26);
    c = _mm256_srli_epi64(t17, 25);
    t18 = _mm256_add_epi64(t18, c);
    t17 = _mm256_and_si256(t17, mask25);
    c = _mm256_srli_epi64(t18, 26);
    t18 = _mm256_and_si256(t18, mask26);
    __m256i t19 = c;

    /* First gamma fold: convolve t[10..19] with gamma -> positions 0..(9+GAMMA_25_LIMBS-1) */
    __m256i h0, h1, h2, h3, h4, h5, h6, h7, h8, h9;
    __m256i h10, h11, h12, h13;
    {
        const __m256i upper[10] = {t10, t11, t12, t13, t14, t15, t16, t17, t18, t19};
        __m256i hv[10 + GAMMA_25_LIMBS - 1];
        const __m256i low[10] = {t0, t1, t2, t3, t4, t5, t6, t7, t8, t9};
        for (int n = 0; n < 10; n++)
            hv[n] = low[n];
        for (int n = 10; n < 10 + GAMMA_25_LIMBS - 1; n++)
            hv[n] = _mm256_setzero_si256();
        for (int k = 0; k < 10; k++)
            for (int j = 0; j < GAMMA_25_LIMBS; j++)
            {
                const int need_double = (k & 1) & (j & 1);
                const __m256i gval = _mm256_set1_epi64x(need_double ? 2 * (int64_t)GAMMA_25[j] : (int64_t)GAMMA_25[j]);
                hv[k + j] = _mm256_add_epi64(hv[k + j], _mm256_mul_epu32(upper[k], gval));
            }
        h0 = hv[0];
        h1 = hv[1];
        h2 = hv[2];
        h3 = hv[3];
        h4 = hv[4];
        h5 = hv[5];
        h6 = hv[6];
        h7 = hv[7];
        h8 = hv[8];
        h9 = hv[9];
        h10 = hv[10];
        h11 = hv[11];
        h12 = hv[12];
        h13 = hv[13];
    }

    /* Carry h0..h9 -> h10 */
    c = _mm256_srli_epi64(h0, 26);
    h1 = _mm256_add_epi64(h1, c);
    h0 = _mm256_and_si256(h0, mask26);
    c = _mm256_srli_epi64(h1, 25);
    h2 = _mm256_add_epi64(h2, c);
    h1 = _mm256_and_si256(h1, mask25);
    c = _mm256_srli_epi64(h2, 26);
    h3 = _mm256_add_epi64(h3, c);
    h2 = _mm256_and_si256(h2, mask26);
    c = _mm256_srli_epi64(h3, 25);
    h4 = _mm256_add_epi64(h4, c);
    h3 = _mm256_and_si256(h3, mask25);
    c = _mm256_srli_epi64(h4, 26);
    h5 = _mm256_add_epi64(h5, c);
    h4 = _mm256_and_si256(h4, mask26);
    c = _mm256_srli_epi64(h5, 25);
    h6 = _mm256_add_epi64(h6, c);
    h5 = _mm256_and_si256(h5, mask25);
    c = _mm256_srli_epi64(h6, 26);
    h7 = _mm256_add_epi64(h7, c);
    h6 = _mm256_and_si256(h6, mask26);
    c = _mm256_srli_epi64(h7, 25);
    h8 = _mm256_add_epi64(h8, c);
    h7 = _mm256_and_si256(h7, mask25);
    c = _mm256_srli_epi64(h8, 26);
    h9 = _mm256_add_epi64(h9, c);
    h8 = _mm256_and_si256(h8, mask26);
    c = _mm256_srli_epi64(h9, 25);
    h10 = _mm256_add_epi64(h10, c);
    h9 = _mm256_and_si256(h9, mask25);

    /* Carry h10..h13 -> h14 */
    c = _mm256_srli_epi64(h10, 26);
    h11 = _mm256_add_epi64(h11, c);
    h10 = _mm256_and_si256(h10, mask26);
    c = _mm256_srli_epi64(h11, 25);
    h12 = _mm256_add_epi64(h12, c);
    h11 = _mm256_and_si256(h11, mask25);
    c = _mm256_srli_epi64(h12, 26);
    h13 = _mm256_add_epi64(h13, c);
    h12 = _mm256_and_si256(h12, mask26);
    c = _mm256_srli_epi64(h13, 25);
    h13 = _mm256_and_si256(h13, mask25);
    __m256i h14 = c;

    /* Second gamma fold: h[10..14] * gamma -> positions 0..8 */
    {
        const __m256i ov[5] = {h10, h11, h12, h13, h14};
        __m256i *dst[10] = {&h0, &h1, &h2, &h3, &h4, &h5, &h6, &h7, &h8, &h9};
        for (int k = 0; k < 5; k++)
            for (int j = 0; j < GAMMA_25_LIMBS; j++)
            {
                if (k + j >= 10)
                    break;
                const int need_double = (k & 1) & (j & 1);
                const __m256i gval = _mm256_set1_epi64x(need_double ? 2 * (int64_t)GAMMA_25[j] : (int64_t)GAMMA_25[j]);
                *dst[k + j] = _mm256_add_epi64(*dst[k + j], _mm256_mul_epu32(ov[k], gval));
            }
    }

    /* Final carry with gamma fold */
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
    fq10x4_carry_gamma(h);
}

/**
 * @brief 4-way double-squaring: h = 2 * f^2 (mod 2^255 - gamma).
 */
static FQ10X4_FORCE_INLINE void fq10x4_sq2(fq10x4 *h, const fq10x4 *f)
{
    fq10x4_sq(h, f);
    for (int i = 0; i < 10; i++)
        h->v[i] = _mm256_slli_epi64(h->v[i], 1);
    fq10x4_carry_gamma(h);
}

/**
 * @brief 4-way conditional move: if mask lane is all-ones, copy src lane.
 *
 * mask should be per-lane all-zeros or all-ones (from cmpeq).
 */
static FQ10X4_FORCE_INLINE void fq10x4_cmov(fq10x4 *t, const fq10x4 *u, __m256i mask)
{
    for (int i = 0; i < 10; i++)
        t->v[i] = _mm256_blendv_epi8(t->v[i], u->v[i], mask);
}

/**
 * @brief Pack a single fq10 into one lane of a fq10x4.
 *
 * Sets lane `lane` of each register in `out` to the corresponding limb of `in`.
 * Other lanes are unchanged. lane must be 0..3.
 */
static FQ10X4_FORCE_INLINE void fq10x4_insert_lane(fq10x4 *out, const fq10 in, int lane)
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
 * @brief Extract one lane from a fq10x4 into a scalar fq10.
 */
static FQ10X4_FORCE_INLINE void fq10x4_extract_lane(fq10 out, const fq10x4 *in, int lane)
{
    alignas(32) int64_t tmp[4];
    for (int i = 0; i < 10; i++)
    {
        _mm256_store_si256((__m256i *)tmp, in->v[i]);
        out[i] = tmp[lane];
    }
}

/**
 * @brief Pack 4 fq10 values into a fq10x4.
 */
static FQ10X4_FORCE_INLINE void fq10x4_pack(fq10x4 *out, const fq10 a, const fq10 b, const fq10 c, const fq10 d)
{
    for (int i = 0; i < 10; i++)
        out->v[i] = _mm256_set_epi64x(d[i], c[i], b[i], a[i]);
}

/**
 * @brief Unpack a fq10x4 into 4 fq10 values.
 */
static FQ10X4_FORCE_INLINE void fq10x4_unpack(fq10 a, fq10 b, fq10 c, fq10 d, const fq10x4 *in)
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

#endif // RANSHAW_X64_AVX2_FQ10X4_AVX2_H
