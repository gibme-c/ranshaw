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
 * @file fq51x8_ifma.h
 * @brief 8-way parallel radix-2^51 Fq field element operations using AVX-512 IFMA.
 *
 * This is the Fq field arithmetic layer for the 8-way batch scalarmult operations
 * over the Crandall prime q = 2^255 - gamma, where
 * gamma = 85737960593035654572250192257530476641 (~127 bits, 3 radix-2^51 limbs).
 *
 * Each fq51x8 holds 8 independent Fq field elements packed horizontally into
 * AVX-512 registers -- one element per 64-bit lane, 5 registers per fq51x8
 * (one per radix-2^51 limb). The representation mirrors the scalar fq_fe on x64.
 *
 * The critical difference from fp51x8 (mod 2^255-19) is the reduction step.
 * Instead of folding upper limbs with x19, we fold with gamma (3 limbs). This
 * makes the reduction a 4-limb x 3-limb convolution (c[5..8] x GAMMA_51[0..2])
 * rather than a simple scalar multiply. IFMA pairs (madd52lo/madd52hi) compute
 * each product term, with the hi part shifted left by 1 (2^52/2^51 = 2) before
 * adding to the next limb.
 *
 * The gamma fold can produce overflow into limbs 5..7, requiring a second
 * mini-fold. After the second fold, a final carry chain with gamma wrap
 * normalizes all limbs to <=51 bits.
 *
 * All IFMA inputs must have limbs <=52 bits. After schoolbook recombination,
 * limbs can reach ~56 bits, so a linear carry chain normalizes them to <=51
 * bits before the gamma fold.
 */

#ifndef RANSHAW_X64_IFMA_FQ51X8_IFMA_H
#define RANSHAW_X64_IFMA_FQ51X8_IFMA_H

#include "fq_ops.h"
#include "ranshaw_platform.h"
#include "x64/fq51.h"

#include <immintrin.h>

#if defined(_MSC_VER)
#define FQ51X8_FORCE_INLINE __forceinline
#else
#define FQ51X8_FORCE_INLINE inline __attribute__((always_inline))
#endif

/**
 * @brief 8-way parallel Fq field element type: 5 __m512i registers.
 *
 * v[i] holds limb i of 8 independent field elements in the 8 x 64-bit lanes.
 * All limbs are unsigned, radix-2^51, <=51 bits after carry propagation.
 */
typedef struct
{
    __m512i v[5];
} fq51x8;

static inline __m512i fq51x8_mask51(void)
{
    return _mm512_set1_epi64((long long)((1ULL << 51) - 1));
}

#define FQ51X8_MASK51 fq51x8_mask51()

// -- Trivial operations (zero, one, copy) --

static FQ51X8_FORCE_INLINE void fq51x8_0(fq51x8 *h)
{
    const __m512i z = _mm512_setzero_si512();
    for (int i = 0; i < 5; i++)
        h->v[i] = z;
}

static FQ51X8_FORCE_INLINE void fq51x8_1(fq51x8 *h)
{
    const __m512i z = _mm512_setzero_si512();
    h->v[0] = _mm512_set1_epi64(1);
    for (int i = 1; i < 5; i++)
        h->v[i] = z;
}

static FQ51X8_FORCE_INLINE void fq51x8_copy(fq51x8 *h, const fq51x8 *f)
{
    for (int i = 0; i < 5; i++)
        h->v[i] = f->v[i];
}

// -- Addition (no carry propagation) --
// For two <=51-bit inputs, the output is at most 52 bits -- still within
// IFMA's input window. No carry needed.

static FQ51X8_FORCE_INLINE void fq51x8_add(fq51x8 *h, const fq51x8 *f, const fq51x8 *g)
{
    for (int i = 0; i < 5; i++)
        h->v[i] = _mm512_add_epi64(f->v[i], g->v[i]);
}

// -- Carry propagation with gamma fold --
// Standard radix-2^51 carry chain: shift right 51, mask, add to next limb.
// Limb 4 wraps back via gamma fold (since 2^255 = gamma mod q). The carry
// out of limb 4 is multiplied by each of the 3 gamma limbs using IFMA,
// with hi parts shifted by 1 and carried into the next position.
// Two passes on limb 0->1 to absorb the final wrap carry.

static FQ51X8_FORCE_INLINE void fq51x8_carry(fq51x8 *h)
{
    const __m512i mask = FQ51X8_MASK51;
    const __m512i zero = _mm512_setzero_si512();
    __m512i c, t;

    c = _mm512_srli_epi64(h->v[0], 51);
    h->v[1] = _mm512_add_epi64(h->v[1], c);
    h->v[0] = _mm512_and_si512(h->v[0], mask);

    c = _mm512_srli_epi64(h->v[1], 51);
    h->v[2] = _mm512_add_epi64(h->v[2], c);
    h->v[1] = _mm512_and_si512(h->v[1], mask);

    c = _mm512_srli_epi64(h->v[2], 51);
    h->v[3] = _mm512_add_epi64(h->v[3], c);
    h->v[2] = _mm512_and_si512(h->v[2], mask);

    c = _mm512_srli_epi64(h->v[3], 51);
    h->v[4] = _mm512_add_epi64(h->v[4], c);
    h->v[3] = _mm512_and_si512(h->v[3], mask);

    // Carry out of limb 4 -- fold via gamma using IFMA
    c = _mm512_srli_epi64(h->v[4], 51);
    h->v[4] = _mm512_and_si512(h->v[4], mask);

    // c * GAMMA_51[j] -> limb j, hi -> limb j+1
    for (int j = 0; j < GAMMA_51_LIMBS; j++)
    {
        const __m512i gj = _mm512_set1_epi64((long long)GAMMA_51[j]);
        h->v[j] = _mm512_madd52lo_epu64(h->v[j], c, gj);
        t = _mm512_madd52hi_epu64(zero, c, gj);
        h->v[j + 1] = _mm512_add_epi64(h->v[j + 1], _mm512_slli_epi64(t, 1));
    }

    // Re-carry limbs 0..4 to normalize
    c = _mm512_srli_epi64(h->v[0], 51);
    h->v[1] = _mm512_add_epi64(h->v[1], c);
    h->v[0] = _mm512_and_si512(h->v[0], mask);

    c = _mm512_srli_epi64(h->v[1], 51);
    h->v[2] = _mm512_add_epi64(h->v[2], c);
    h->v[1] = _mm512_and_si512(h->v[1], mask);

    c = _mm512_srli_epi64(h->v[2], 51);
    h->v[3] = _mm512_add_epi64(h->v[3], c);
    h->v[2] = _mm512_and_si512(h->v[2], mask);

    c = _mm512_srli_epi64(h->v[3], 51);
    h->v[4] = _mm512_add_epi64(h->v[4], c);
    h->v[3] = _mm512_and_si512(h->v[3], mask);
}

// -- Subtraction with 8q bias + carry --
// To keep limbs non-negative, we add 8q before subtracting. The bias values
// are EIGHT_Q_51[i] = 8 * Q_51[i] for each limb. The carry chain with gamma
// fold then normalizes back to <=51-bit limbs.
//
// Fp uses 4p bias because all p limbs ≈ 2^51, so 4p limbs ≈ 2^53. For Fq,
// the lower limbs of q are much smaller than 2^51 (gamma ≈ 2^127), so
// 4*Q_51[0] ≈ 2^52.77 < 2^53 -- insufficient for 53-bit operands produced
// by chained additions in dbl_8x. We need 8q to ensure all bias limbs exceed
// 2^53. All 8q limbs fit in 54 bits, well within the 64-bit lane.
//
// 8q bias values:
//   limb 0: 8 * 0x6D2727927C79F = 0x369393C93E3CF8
//   limb 1: 8 * 0x596ECAD6B0DD6 = 0x2CB7656B586EB0
//   limb 2: 8 * 0x7FFFFFEFDFDE0 = 0x3FFFFFF7EFEF00
//   limb 3: 8 * 0x7FFFFFFFFFFFF = 0x3FFFFFFFFFFFF8
//   limb 4: 8 * 0x7FFFFFFFFFFFF = 0x3FFFFFFFFFFFF8

static FQ51X8_FORCE_INLINE void fq51x8_sub(fq51x8 *h, const fq51x8 *f, const fq51x8 *g)
{
    const __m512i bias0 = _mm512_set1_epi64((long long)EIGHT_Q_51[0]);
    const __m512i bias1 = _mm512_set1_epi64((long long)EIGHT_Q_51[1]);
    const __m512i bias2 = _mm512_set1_epi64((long long)EIGHT_Q_51[2]);
    const __m512i bias3 = _mm512_set1_epi64((long long)EIGHT_Q_51[3]);
    const __m512i bias4 = _mm512_set1_epi64((long long)EIGHT_Q_51[4]);

    h->v[0] = _mm512_add_epi64(_mm512_sub_epi64(f->v[0], g->v[0]), bias0);
    h->v[1] = _mm512_add_epi64(_mm512_sub_epi64(f->v[1], g->v[1]), bias1);
    h->v[2] = _mm512_add_epi64(_mm512_sub_epi64(f->v[2], g->v[2]), bias2);
    h->v[3] = _mm512_add_epi64(_mm512_sub_epi64(f->v[3], g->v[3]), bias3);
    h->v[4] = _mm512_add_epi64(_mm512_sub_epi64(f->v[4], g->v[4]), bias4);

    fq51x8_carry(h);
}

// -- Negation --

static FQ51X8_FORCE_INLINE void fq51x8_neg(fq51x8 *h, const fq51x8 *f)
{
    fq51x8 zero;
    fq51x8_0(&zero);
    fq51x8_sub(h, &zero, f);
}

// -- Weak normalization --
// Same as fq51x8_carry, used to fix limbs that exceed 52 bits after a
// problematic addition. Only needed at specific points in batch point
// addition/subtraction where limbs may exceed the 52-bit IFMA input window.

static FQ51X8_FORCE_INLINE void fq51x8_normalize_weak(fq51x8 *h)
{
    fq51x8_carry(h);
}

// -- Conditional move (k-mask) --
// AVX-512 k-mask blend: for each of the 8 lanes, if the corresponding bit
// in `mask` is set, take the value from `u`; otherwise keep the value in `t`.

static FQ51X8_FORCE_INLINE void fq51x8_cmov(fq51x8 *t, const fq51x8 *u, __mmask8 mask)
{
    for (int i = 0; i < 5; i++)
        t->v[i] = _mm512_mask_blend_epi64(mask, t->v[i], u->v[i]);
}

// -- Internal: Crandall reduction for schoolbook result --
// Takes 9 recombined limbs c0..c8 (each <=51 bits after linear carry) and
// folds c5..c8 with gamma to produce 5-limb output in h.
// This is shared by mul, sq, and sq2.

static FQ51X8_FORCE_INLINE void fq51x8_crandall_reduce(
    fq51x8 *h,
    __m512i c0,
    __m512i c1,
    __m512i c2,
    __m512i c3,
    __m512i c4,
    __m512i c5,
    __m512i c6,
    __m512i c7,
    __m512i c8)
{
    const __m512i mask = FQ51X8_MASK51;
    const __m512i zero = _mm512_setzero_si512();
    __m512i c, t;

    // Linear carry chain c0..c8 to bring all limbs to <=51 bits.
    // After schoolbook recombination, limbs can be ~56 bits. We need
    // them <=51 bits so they are safe as IFMA inputs for the gamma fold.
    c = _mm512_srli_epi64(c0, 51);
    c1 = _mm512_add_epi64(c1, c);
    c0 = _mm512_and_si512(c0, mask);
    c = _mm512_srli_epi64(c1, 51);
    c2 = _mm512_add_epi64(c2, c);
    c1 = _mm512_and_si512(c1, mask);
    c = _mm512_srli_epi64(c2, 51);
    c3 = _mm512_add_epi64(c3, c);
    c2 = _mm512_and_si512(c2, mask);
    c = _mm512_srli_epi64(c3, 51);
    c4 = _mm512_add_epi64(c4, c);
    c3 = _mm512_and_si512(c3, mask);
    c = _mm512_srli_epi64(c4, 51);
    c5 = _mm512_add_epi64(c5, c);
    c4 = _mm512_and_si512(c4, mask);
    c = _mm512_srli_epi64(c5, 51);
    c6 = _mm512_add_epi64(c6, c);
    c5 = _mm512_and_si512(c5, mask);
    c = _mm512_srli_epi64(c6, 51);
    c7 = _mm512_add_epi64(c7, c);
    c6 = _mm512_and_si512(c6, mask);
    c = _mm512_srli_epi64(c7, 51);
    c8 = _mm512_add_epi64(c8, c);
    c7 = _mm512_and_si512(c7, mask);
    __m512i carry_out = _mm512_srli_epi64(c8, 51);
    c8 = _mm512_and_si512(c8, mask);

    // Now all of c0..c8 are <=51 bits, carry_out is <=~5 bits.
    // Fold c5..c8 and carry_out with gamma using nested loops.
    __m512i r[5 + GAMMA_51_LIMBS];
    r[0] = c0;
    r[1] = c1;
    r[2] = c2;
    r[3] = c3;
    r[4] = c4;
    for (int i = 5; i < 5 + GAMMA_51_LIMBS; i++)
        r[i] = zero;
    {
        const __m512i overflow[5] = {c5, c6, c7, c8, carry_out};
        for (int k = 0; k < 5; k++)
        {
            for (int j = 0; j < GAMMA_51_LIMBS; j++)
            {
                const __m512i gj = _mm512_set1_epi64((long long)GAMMA_51[j]);
                r[k + j] = _mm512_madd52lo_epu64(r[k + j], overflow[k], gj);
                t = _mm512_madd52hi_epu64(zero, overflow[k], gj);
                r[k + j + 1] = _mm512_add_epi64(r[k + j + 1], _mm512_slli_epi64(t, 1));
            }
        }
    }

    // Carry-propagate r[0..4], with overflow into r[5]
    c = _mm512_srli_epi64(r[0], 51);
    r[1] = _mm512_add_epi64(r[1], c);
    r[0] = _mm512_and_si512(r[0], mask);
    c = _mm512_srli_epi64(r[1], 51);
    r[2] = _mm512_add_epi64(r[2], c);
    r[1] = _mm512_and_si512(r[1], mask);
    c = _mm512_srli_epi64(r[2], 51);
    r[3] = _mm512_add_epi64(r[3], c);
    r[2] = _mm512_and_si512(r[2], mask);
    c = _mm512_srli_epi64(r[3], 51);
    r[4] = _mm512_add_epi64(r[4], c);
    r[3] = _mm512_and_si512(r[3], mask);
    c = _mm512_srli_epi64(r[4], 51);
    r[5] = _mm512_add_epi64(r[5], c);
    r[4] = _mm512_and_si512(r[4], mask);

    // Carry-normalize r[5] into r[6] before second fold.
    // After the first fold + carry chain, r[5] can be ~53 bits. IFMA madd52
    // instructions only use the low 52 bits of their operands (a[51:0]),
    // so bits 52+ would be silently dropped, corrupting the result.
    c = _mm512_srli_epi64(r[5], 51);
    r[6] = _mm512_add_epi64(r[6], c);
    r[5] = _mm512_and_si512(r[5], mask);

    // Second mini-fold: r[5] and r[6] are now <=51 bits, fold them back
    {
        const __m512i overflow2[2] = {r[5], r[6]};
        r[5] = zero;
        r[6] = zero;
        for (int k = 0; k < 2; k++)
        {
            for (int j = 0; j < GAMMA_51_LIMBS; j++)
            {
                const __m512i gj = _mm512_set1_epi64((long long)GAMMA_51[j]);
                r[k + j] = _mm512_madd52lo_epu64(r[k + j], overflow2[k], gj);
                t = _mm512_madd52hi_epu64(zero, overflow2[k], gj);
                r[k + j + 1] = _mm512_add_epi64(r[k + j + 1], _mm512_slli_epi64(t, 1));
            }
        }
    }

    // Final carry chain with gamma fold for carry out of limb 4
    c = _mm512_srli_epi64(r[0], 51);
    r[1] = _mm512_add_epi64(r[1], c);
    r[0] = _mm512_and_si512(r[0], mask);
    c = _mm512_srli_epi64(r[1], 51);
    r[2] = _mm512_add_epi64(r[2], c);
    r[1] = _mm512_and_si512(r[1], mask);
    c = _mm512_srli_epi64(r[2], 51);
    r[3] = _mm512_add_epi64(r[3], c);
    r[2] = _mm512_and_si512(r[2], mask);
    c = _mm512_srli_epi64(r[3], 51);
    r[4] = _mm512_add_epi64(r[4], c);
    r[3] = _mm512_and_si512(r[3], mask);

    // Fold final carry out of limb 4 via gamma (carry is tiny here)
    c = _mm512_srli_epi64(r[4], 51);
    r[4] = _mm512_and_si512(r[4], mask);
    for (int j = 0; j < GAMMA_51_LIMBS; j++)
        r[j] = _mm512_madd52lo_epu64(r[j], c, _mm512_set1_epi64((long long)GAMMA_51[j]));

    // One more carry pass on limb 0->1 to absorb the final fold
    c = _mm512_srli_epi64(r[0], 51);
    r[1] = _mm512_add_epi64(r[1], c);
    r[0] = _mm512_and_si512(r[0], mask);

    h->v[0] = r[0];
    h->v[1] = r[1];
    h->v[2] = r[2];
    h->v[3] = r[3];
    h->v[4] = r[4];
}

// -- Schoolbook multiplication using IFMA --
// This is the heart of the 8-way backend. Two IFMA instructions per product
// term (lo + hi halves), 25 product terms for a 5x5 schoolbook, so 50 IFMA
// ops total -- all operating on 8 independent multiplications in parallel.

/**
 * @brief 8-way multiplication: h = f * g (mod 2^255 - gamma).
 *
 * Both inputs must have limbs <=52 bits (true for IFMA outputs <=51, fq51x8_add
 * of two <=51-bit values <=52, fq51x8_sub outputs <=52 after bias).
 *
 * Algorithm: 5x5 schoolbook -> 9-limb lo/hi accumulators via IFMA,
 * recombine lo/hi at radix-2^51 boundary, linear carry to <=51 bits,
 * fold upper limbs with gamma (Crandall reduction), carry-propagate.
 */
static FQ51X8_FORCE_INLINE void fq51x8_mul(fq51x8 *h, const fq51x8 *f, const fq51x8 *g)
{
    const __m512i zero = _mm512_setzero_si512();

    // 9-limb accumulators for lo (bits 0-51) and hi (bits 52-103)
    __m512i lo0 = zero, lo1 = zero, lo2 = zero, lo3 = zero, lo4 = zero;
    __m512i lo5 = zero, lo6 = zero, lo7 = zero, lo8 = zero;
    __m512i hi0 = zero, hi1 = zero, hi2 = zero, hi3 = zero, hi4 = zero;
    __m512i hi5 = zero, hi6 = zero, hi7 = zero, hi8 = zero;

    // 25 products: f[i] * g[j] -> accumulate into lo[i+j], hi[i+j]
    // Round 0: f[0] * g[0..4]
    lo0 = _mm512_madd52lo_epu64(lo0, f->v[0], g->v[0]);
    hi0 = _mm512_madd52hi_epu64(hi0, f->v[0], g->v[0]);
    lo1 = _mm512_madd52lo_epu64(lo1, f->v[0], g->v[1]);
    hi1 = _mm512_madd52hi_epu64(hi1, f->v[0], g->v[1]);
    lo2 = _mm512_madd52lo_epu64(lo2, f->v[0], g->v[2]);
    hi2 = _mm512_madd52hi_epu64(hi2, f->v[0], g->v[2]);
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[0], g->v[3]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[0], g->v[3]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[0], g->v[4]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[0], g->v[4]);

    // Round 1: f[1] * g[0..4]
    lo1 = _mm512_madd52lo_epu64(lo1, f->v[1], g->v[0]);
    hi1 = _mm512_madd52hi_epu64(hi1, f->v[1], g->v[0]);
    lo2 = _mm512_madd52lo_epu64(lo2, f->v[1], g->v[1]);
    hi2 = _mm512_madd52hi_epu64(hi2, f->v[1], g->v[1]);
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[1], g->v[2]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[1], g->v[2]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[1], g->v[3]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[1], g->v[3]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[1], g->v[4]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[1], g->v[4]);

    // Round 2: f[2] * g[0..4]
    lo2 = _mm512_madd52lo_epu64(lo2, f->v[2], g->v[0]);
    hi2 = _mm512_madd52hi_epu64(hi2, f->v[2], g->v[0]);
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[2], g->v[1]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[2], g->v[1]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[2], g->v[2]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[2], g->v[2]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[2], g->v[3]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[2], g->v[3]);
    lo6 = _mm512_madd52lo_epu64(lo6, f->v[2], g->v[4]);
    hi6 = _mm512_madd52hi_epu64(hi6, f->v[2], g->v[4]);

    // Round 3: f[3] * g[0..4]
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[3], g->v[0]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[3], g->v[0]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[3], g->v[1]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[3], g->v[1]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[3], g->v[2]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[3], g->v[2]);
    lo6 = _mm512_madd52lo_epu64(lo6, f->v[3], g->v[3]);
    hi6 = _mm512_madd52hi_epu64(hi6, f->v[3], g->v[3]);
    lo7 = _mm512_madd52lo_epu64(lo7, f->v[3], g->v[4]);
    hi7 = _mm512_madd52hi_epu64(hi7, f->v[3], g->v[4]);

    // Round 4: f[4] * g[0..4]
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[4], g->v[0]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[4], g->v[0]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[4], g->v[1]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[4], g->v[1]);
    lo6 = _mm512_madd52lo_epu64(lo6, f->v[4], g->v[2]);
    hi6 = _mm512_madd52hi_epu64(hi6, f->v[4], g->v[2]);
    lo7 = _mm512_madd52lo_epu64(lo7, f->v[4], g->v[3]);
    hi7 = _mm512_madd52hi_epu64(hi7, f->v[4], g->v[3]);
    lo8 = _mm512_madd52lo_epu64(lo8, f->v[4], g->v[4]);
    hi8 = _mm512_madd52hi_epu64(hi8, f->v[4], g->v[4]);

    // Recombine: IFMA splits at bit 52, our radix is 2^51.
    // c[0] = lo[0]
    // c[k] = lo[k] + hi[k-1] << 1   for k = 1..8
    __m512i c0 = lo0;
    __m512i c1 = _mm512_add_epi64(lo1, _mm512_slli_epi64(hi0, 1));
    __m512i c2 = _mm512_add_epi64(lo2, _mm512_slli_epi64(hi1, 1));
    __m512i c3 = _mm512_add_epi64(lo3, _mm512_slli_epi64(hi2, 1));
    __m512i c4 = _mm512_add_epi64(lo4, _mm512_slli_epi64(hi3, 1));
    __m512i c5 = _mm512_add_epi64(lo5, _mm512_slli_epi64(hi4, 1));
    __m512i c6 = _mm512_add_epi64(lo6, _mm512_slli_epi64(hi5, 1));
    __m512i c7 = _mm512_add_epi64(lo7, _mm512_slli_epi64(hi6, 1));
    __m512i c8 = _mm512_add_epi64(lo8, _mm512_slli_epi64(hi7, 1));

    // hi8 contributes to c9 = hi8<<1, which represents position 9.
    // Position 9 folds as c9 * gamma into positions 4,5,6.
    // hi8 has at most 1 IFMA accumulation, so hi8 <= 52 bits and
    // c9 = hi8<<1 <= 53 bits. Since IFMA uses only bits [51:0] of its
    // inputs, we normalize c9 into a 51-bit part and a carry:
    //   c9_lo = c9 & mask51  (position 9, folds to 4,5,6)
    //   c9_hi = c9 >> 51     (position 10, folds to 5,6,7)
    {
        const __m512i zero_v = _mm512_setzero_si512();
        const __m512i mask = FQ51X8_MASK51;
        __m512i t;

        __m512i c9 = _mm512_slli_epi64(hi8, 1);
        __m512i c9_hi = _mm512_srli_epi64(c9, 51);
        c9 = _mm512_and_si512(c9, mask);

        // c9_lo * gamma -> positions 4..4+GAMMA_51_LIMBS-1 (c9_lo <= 51 bits, safe for IFMA)
        __m512i *cv[9] = {&c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7, &c8};
        for (int j = 0; j < GAMMA_51_LIMBS; j++)
        {
            const __m512i gj = _mm512_set1_epi64((long long)GAMMA_51[j]);
            *cv[4 + j] = _mm512_madd52lo_epu64(*cv[4 + j], c9, gj);
            t = _mm512_madd52hi_epu64(zero_v, c9, gj);
            *cv[4 + j + 1] = _mm512_add_epi64(*cv[4 + j + 1], _mm512_slli_epi64(t, 1));
        }

        // c9_hi * gamma -> positions 5..5+GAMMA_51_LIMBS-1 (c9_hi <= ~2 bits, products tiny)
        for (int j = 0; j < GAMMA_51_LIMBS; j++)
            *cv[5 + j] = _mm512_madd52lo_epu64(*cv[5 + j], c9_hi, _mm512_set1_epi64((long long)GAMMA_51[j]));
    }

    fq51x8_crandall_reduce(h, c0, c1, c2, c3, c4, c5, c6, c7, c8);
}

// -- Squaring --
// Currently implemented as mul(f, f). Could be optimized to exploit symmetry
// (15 unique products instead of 25) but the savings would be modest given
// that IFMA throughput is the bottleneck, not instruction count.

static FQ51X8_FORCE_INLINE void fq51x8_sq(fq51x8 *h, const fq51x8 *f)
{
    fq51x8_mul(h, f, f);
}

// -- Double-squaring: h = 2 * f^2 --
// Used by point doubling for the 2*Z^2 term. Computes all 25 products,
// doubles every accumulator (lo and hi), then proceeds with the same
// recombination and Crandall reduction as mul.

static FQ51X8_FORCE_INLINE void fq51x8_sq2(fq51x8 *h, const fq51x8 *f)
{
    const __m512i zero = _mm512_setzero_si512();

    __m512i lo0 = zero, lo1 = zero, lo2 = zero, lo3 = zero, lo4 = zero;
    __m512i lo5 = zero, lo6 = zero, lo7 = zero, lo8 = zero;
    __m512i hi0 = zero, hi1 = zero, hi2 = zero, hi3 = zero, hi4 = zero;
    __m512i hi5 = zero, hi6 = zero, hi7 = zero, hi8 = zero;

    // Same 25 products as mul, but f=g
    lo0 = _mm512_madd52lo_epu64(lo0, f->v[0], f->v[0]);
    hi0 = _mm512_madd52hi_epu64(hi0, f->v[0], f->v[0]);
    lo1 = _mm512_madd52lo_epu64(lo1, f->v[0], f->v[1]);
    hi1 = _mm512_madd52hi_epu64(hi1, f->v[0], f->v[1]);
    lo2 = _mm512_madd52lo_epu64(lo2, f->v[0], f->v[2]);
    hi2 = _mm512_madd52hi_epu64(hi2, f->v[0], f->v[2]);
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[0], f->v[3]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[0], f->v[3]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[0], f->v[4]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[0], f->v[4]);

    lo1 = _mm512_madd52lo_epu64(lo1, f->v[1], f->v[0]);
    hi1 = _mm512_madd52hi_epu64(hi1, f->v[1], f->v[0]);
    lo2 = _mm512_madd52lo_epu64(lo2, f->v[1], f->v[1]);
    hi2 = _mm512_madd52hi_epu64(hi2, f->v[1], f->v[1]);
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[1], f->v[2]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[1], f->v[2]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[1], f->v[3]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[1], f->v[3]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[1], f->v[4]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[1], f->v[4]);

    lo2 = _mm512_madd52lo_epu64(lo2, f->v[2], f->v[0]);
    hi2 = _mm512_madd52hi_epu64(hi2, f->v[2], f->v[0]);
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[2], f->v[1]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[2], f->v[1]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[2], f->v[2]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[2], f->v[2]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[2], f->v[3]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[2], f->v[3]);
    lo6 = _mm512_madd52lo_epu64(lo6, f->v[2], f->v[4]);
    hi6 = _mm512_madd52hi_epu64(hi6, f->v[2], f->v[4]);

    lo3 = _mm512_madd52lo_epu64(lo3, f->v[3], f->v[0]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[3], f->v[0]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[3], f->v[1]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[3], f->v[1]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[3], f->v[2]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[3], f->v[2]);
    lo6 = _mm512_madd52lo_epu64(lo6, f->v[3], f->v[3]);
    hi6 = _mm512_madd52hi_epu64(hi6, f->v[3], f->v[3]);
    lo7 = _mm512_madd52lo_epu64(lo7, f->v[3], f->v[4]);
    hi7 = _mm512_madd52hi_epu64(hi7, f->v[3], f->v[4]);

    lo4 = _mm512_madd52lo_epu64(lo4, f->v[4], f->v[0]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[4], f->v[0]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[4], f->v[1]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[4], f->v[1]);
    lo6 = _mm512_madd52lo_epu64(lo6, f->v[4], f->v[2]);
    hi6 = _mm512_madd52hi_epu64(hi6, f->v[4], f->v[2]);
    lo7 = _mm512_madd52lo_epu64(lo7, f->v[4], f->v[3]);
    hi7 = _mm512_madd52hi_epu64(hi7, f->v[4], f->v[3]);
    lo8 = _mm512_madd52lo_epu64(lo8, f->v[4], f->v[4]);
    hi8 = _mm512_madd52hi_epu64(hi8, f->v[4], f->v[4]);

    // Double all accumulators (sq2 = 2 * f^2)
    lo0 = _mm512_add_epi64(lo0, lo0);
    hi0 = _mm512_add_epi64(hi0, hi0);
    lo1 = _mm512_add_epi64(lo1, lo1);
    hi1 = _mm512_add_epi64(hi1, hi1);
    lo2 = _mm512_add_epi64(lo2, lo2);
    hi2 = _mm512_add_epi64(hi2, hi2);
    lo3 = _mm512_add_epi64(lo3, lo3);
    hi3 = _mm512_add_epi64(hi3, hi3);
    lo4 = _mm512_add_epi64(lo4, lo4);
    hi4 = _mm512_add_epi64(hi4, hi4);
    lo5 = _mm512_add_epi64(lo5, lo5);
    hi5 = _mm512_add_epi64(hi5, hi5);
    lo6 = _mm512_add_epi64(lo6, lo6);
    hi6 = _mm512_add_epi64(hi6, hi6);
    lo7 = _mm512_add_epi64(lo7, lo7);
    hi7 = _mm512_add_epi64(hi7, hi7);
    lo8 = _mm512_add_epi64(lo8, lo8);
    hi8 = _mm512_add_epi64(hi8, hi8);

    // Same recombination as mul
    __m512i c0 = lo0;
    __m512i c1 = _mm512_add_epi64(lo1, _mm512_slli_epi64(hi0, 1));
    __m512i c2 = _mm512_add_epi64(lo2, _mm512_slli_epi64(hi1, 1));
    __m512i c3 = _mm512_add_epi64(lo3, _mm512_slli_epi64(hi2, 1));
    __m512i c4 = _mm512_add_epi64(lo4, _mm512_slli_epi64(hi3, 1));
    __m512i c5 = _mm512_add_epi64(lo5, _mm512_slli_epi64(hi4, 1));
    __m512i c6 = _mm512_add_epi64(lo6, _mm512_slli_epi64(hi5, 1));
    __m512i c7 = _mm512_add_epi64(lo7, _mm512_slli_epi64(hi6, 1));
    __m512i c8 = _mm512_add_epi64(lo8, _mm512_slli_epi64(hi7, 1));
    // hi8 contributes to c9 = hi8<<1, which represents position 9.
    // Position 9 folds as c9 * gamma into positions 4,5,6.
    // In sq2, hi8 was doubled (line 697), so hi8 <= 53 bits and
    // c9 = hi8<<1 <= 54 bits. Normalize to 51 bits before IFMA.
    {
        const __m512i zero_v = _mm512_setzero_si512();
        const __m512i mask = FQ51X8_MASK51;
        __m512i t;

        __m512i c9 = _mm512_slli_epi64(hi8, 1);
        __m512i c9_hi = _mm512_srli_epi64(c9, 51);
        c9 = _mm512_and_si512(c9, mask);

        // c9_lo * gamma -> positions 4..4+GAMMA_51_LIMBS-1
        __m512i *cv[9] = {&c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7, &c8};
        for (int j = 0; j < GAMMA_51_LIMBS; j++)
        {
            const __m512i gj = _mm512_set1_epi64((long long)GAMMA_51[j]);
            *cv[4 + j] = _mm512_madd52lo_epu64(*cv[4 + j], c9, gj);
            t = _mm512_madd52hi_epu64(zero_v, c9, gj);
            *cv[4 + j + 1] = _mm512_add_epi64(*cv[4 + j + 1], _mm512_slli_epi64(t, 1));
        }

        // c9_hi * gamma -> positions 5..5+GAMMA_51_LIMBS-1
        for (int j = 0; j < GAMMA_51_LIMBS; j++)
            *cv[5 + j] = _mm512_madd52lo_epu64(*cv[5 + j], c9_hi, _mm512_set1_epi64((long long)GAMMA_51[j]));
    }

    fq51x8_crandall_reduce(h, c0, c1, c2, c3, c4, c5, c6, c7, c8);
}

// -- Lane insert / extract --
// These convert between scalar fq_fe (single field element) and one lane of a
// fq51x8. They're only used at batch entry (packing input points) and exit
// (extracting results) -- not in the hot loop.

static FQ51X8_FORCE_INLINE void fq51x8_insert_lane(fq51x8 *out, const fq_fe in, int lane)
{
    /* fq_fe is already 5x51 radix-2^51, same as fq51x8 lane format — direct copy */
    alignas(64) long long tmp[8];
    for (int i = 0; i < 5; i++)
    {
        _mm512_store_si512((__m512i *)tmp, out->v[i]);
        tmp[lane] = (long long)in[i];
        out->v[i] = _mm512_load_si512((const __m512i *)tmp);
    }
}

static FQ51X8_FORCE_INLINE void fq51x8_extract_lane(fq_fe out, const fq51x8 *in, int lane)
{
    /* fq_fe is already 5x51 radix-2^51, same as fq51x8 lane format — direct extract */
    alignas(64) long long tmp[8];
    for (int i = 0; i < 5; i++)
    {
        _mm512_store_si512((__m512i *)tmp, in->v[i]);
        out[i] = (uint64_t)tmp[lane];
    }
}

#endif // RANSHAW_X64_IFMA_FQ51X8_IFMA_H
