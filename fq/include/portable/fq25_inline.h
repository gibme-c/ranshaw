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
 * @file fq25_inline.h
 * @brief Portable (32-bit, radix-2^25.5) implementation of F_q inline arithmetic helpers with Crandall reduction.
 */

#ifndef RANSHAW_PORTABLE_FQ25_INLINE_H
#define RANSHAW_PORTABLE_FQ25_INLINE_H

#include "fq.h"
#include "portable/fq25.h"

#if defined(_MSC_VER)
#ifndef RANSHAW_FORCE_INLINE
#define RANSHAW_FORCE_INLINE __forceinline
#endif
#elif !defined(RANSHAW_FORCE_INLINE)
#define RANSHAW_FORCE_INLINE inline __attribute__((always_inline))
#endif

/*
 * Crandall carry-reduction for q = 2^255 - gamma (32-bit, 10-limb).
 *
 * Takes 10 int64_t accumulators representing a value in radix-2^25.5,
 * carry-propagates, and folds carry from limb 9 via gamma multiplication.
 */
static RANSHAW_FORCE_INLINE void fq25_carry_reduce(
    fq_fe out,
    int64_t h0,
    int64_t h1,
    int64_t h2,
    int64_t h3,
    int64_t h4,
    int64_t h5,
    int64_t h6,
    int64_t h7,
    int64_t h8,
    int64_t h9)
{
    int64_t carry;

    /* First carry pass */
    carry = (h0 + (int64_t)(1 << 25)) >> 26;
    h1 += carry;
    h0 -= carry << 26;
    carry = (h4 + (int64_t)(1 << 25)) >> 26;
    h5 += carry;
    h4 -= carry << 26;
    carry = (h1 + (int64_t)(1 << 24)) >> 25;
    h2 += carry;
    h1 -= carry << 25;
    carry = (h5 + (int64_t)(1 << 24)) >> 25;
    h6 += carry;
    h5 -= carry << 25;
    carry = (h2 + (int64_t)(1 << 25)) >> 26;
    h3 += carry;
    h2 -= carry << 26;
    carry = (h6 + (int64_t)(1 << 25)) >> 26;
    h7 += carry;
    h6 -= carry << 26;
    carry = (h3 + (int64_t)(1 << 24)) >> 25;
    h4 += carry;
    h3 -= carry << 25;
    carry = (h7 + (int64_t)(1 << 24)) >> 25;
    h8 += carry;
    h7 -= carry << 25;
    carry = (h4 + (int64_t)(1 << 25)) >> 26;
    h5 += carry;
    h4 -= carry << 26;
    carry = (h8 + (int64_t)(1 << 25)) >> 26;
    h9 += carry;
    h8 -= carry << 26;

    /* Gamma fold: carry from h9 wraps as carry * gamma */
    carry = (h9 + (int64_t)(1 << 24)) >> 25;
    h9 -= carry << 25;
    h0 += carry * (int64_t)GAMMA_25[0];
    h1 += carry * (int64_t)GAMMA_25[1];
    h2 += carry * (int64_t)GAMMA_25[2];
    h3 += carry * (int64_t)GAMMA_25[3];
    h4 += carry * (int64_t)GAMMA_25[4];

    /* Second carry pass to normalize after gamma fold */
    carry = (h0 + (int64_t)(1 << 25)) >> 26;
    h1 += carry;
    h0 -= carry << 26;
    carry = (h1 + (int64_t)(1 << 24)) >> 25;
    h2 += carry;
    h1 -= carry << 25;
    carry = (h2 + (int64_t)(1 << 25)) >> 26;
    h3 += carry;
    h2 -= carry << 26;
    carry = (h3 + (int64_t)(1 << 24)) >> 25;
    h4 += carry;
    h3 -= carry << 25;
    carry = (h4 + (int64_t)(1 << 25)) >> 26;
    h5 += carry;
    h4 -= carry << 26;
    carry = (h5 + (int64_t)(1 << 24)) >> 25;
    h6 += carry;
    h5 -= carry << 25;
    carry = (h6 + (int64_t)(1 << 25)) >> 26;
    h7 += carry;
    h6 -= carry << 26;
    carry = (h7 + (int64_t)(1 << 24)) >> 25;
    h8 += carry;
    h7 -= carry << 25;
    carry = (h8 + (int64_t)(1 << 25)) >> 26;
    h9 += carry;
    h8 -= carry << 26;
    carry = (h9 + (int64_t)(1 << 24)) >> 25;
    h9 -= carry << 25;

    /* Second gamma fold (carry should be very small, often 0) */
    h0 += carry * (int64_t)GAMMA_25[0];
    h1 += carry * (int64_t)GAMMA_25[1];
    h2 += carry * (int64_t)GAMMA_25[2];
    h3 += carry * (int64_t)GAMMA_25[3];
    h4 += carry * (int64_t)GAMMA_25[4];

    /* Final carry for limbs 0-4 */
    carry = (h0 + (int64_t)(1 << 25)) >> 26;
    h1 += carry;
    h0 -= carry << 26;
    carry = (h1 + (int64_t)(1 << 24)) >> 25;
    h2 += carry;
    h1 -= carry << 25;
    carry = (h2 + (int64_t)(1 << 25)) >> 26;
    h3 += carry;
    h2 -= carry << 26;
    carry = (h3 + (int64_t)(1 << 24)) >> 25;
    h4 += carry;
    h3 -= carry << 25;

    out[0] = (int32_t)h0;
    out[1] = (int32_t)h1;
    out[2] = (int32_t)h2;
    out[3] = (int32_t)h3;
    out[4] = (int32_t)h4;
    out[5] = (int32_t)h5;
    out[6] = (int32_t)h6;
    out[7] = (int32_t)h7;
    out[8] = (int32_t)h8;
    out[9] = (int32_t)h9;
}

/*
 * Crandall reduction for the full schoolbook product.
 *
 * After a 10x10 schoolbook multiply producing 19 int64_t accumulators
 * (with radix-2^25.5 offset correction already applied via fi_2 trick),
 * carry-propagate, extract the upper part (positions 10-18+), multiply
 * by gamma, and fold back into the lower part (positions 0-9).
 *
 * Gamma fold: t[k] for k>=10 represents overflow past 2^255.
 * Since 2^255 = gamma (mod q), we fold: t[k] * gamma added to position (k-10).
 * Gamma spans 5 limbs (positions 0-4), so t[k] * gamma[j] -> position (k-10+j).
 *
 * Radix-2^25.5 offset correction in gamma fold: when BOTH the source
 * position k AND gamma index j are odd, the product's bit offset is 1
 * higher than nominal, requiring doubling. We use pre-doubled gamma
 * limbs (g1_2, g3_2) for odd-position sources.
 */
static RANSHAW_FORCE_INLINE void fq25_reduce_full(fq_fe out, int64_t t[19])
{
    int64_t carry;

    /*
     * Carry-propagate t[0..18] into canonical-width limbs.
     * After this, each t[i] fits in its alternating 26/25-bit width,
     * and t[10..18] represent the overflow past 2^255.
     */
    carry = (t[0] + (int64_t)(1 << 25)) >> 26;
    t[1] += carry;
    t[0] -= carry << 26;
    carry = (t[1] + (int64_t)(1 << 24)) >> 25;
    t[2] += carry;
    t[1] -= carry << 25;
    carry = (t[2] + (int64_t)(1 << 25)) >> 26;
    t[3] += carry;
    t[2] -= carry << 26;
    carry = (t[3] + (int64_t)(1 << 24)) >> 25;
    t[4] += carry;
    t[3] -= carry << 25;
    carry = (t[4] + (int64_t)(1 << 25)) >> 26;
    t[5] += carry;
    t[4] -= carry << 26;
    carry = (t[5] + (int64_t)(1 << 24)) >> 25;
    t[6] += carry;
    t[5] -= carry << 25;
    carry = (t[6] + (int64_t)(1 << 25)) >> 26;
    t[7] += carry;
    t[6] -= carry << 26;
    carry = (t[7] + (int64_t)(1 << 24)) >> 25;
    t[8] += carry;
    t[7] -= carry << 25;
    carry = (t[8] + (int64_t)(1 << 25)) >> 26;
    t[9] += carry;
    t[8] -= carry << 26;
    carry = (t[9] + (int64_t)(1 << 24)) >> 25;
    t[10] += carry;
    t[9] -= carry << 25;
    carry = (t[10] + (int64_t)(1 << 25)) >> 26;
    t[11] += carry;
    t[10] -= carry << 26;
    carry = (t[11] + (int64_t)(1 << 24)) >> 25;
    t[12] += carry;
    t[11] -= carry << 25;
    carry = (t[12] + (int64_t)(1 << 25)) >> 26;
    t[13] += carry;
    t[12] -= carry << 26;
    carry = (t[13] + (int64_t)(1 << 24)) >> 25;
    t[14] += carry;
    t[13] -= carry << 25;
    carry = (t[14] + (int64_t)(1 << 25)) >> 26;
    t[15] += carry;
    t[14] -= carry << 26;
    carry = (t[15] + (int64_t)(1 << 24)) >> 25;
    t[16] += carry;
    t[15] -= carry << 25;
    carry = (t[16] + (int64_t)(1 << 25)) >> 26;
    t[17] += carry;
    t[16] -= carry << 26;
    carry = (t[17] + (int64_t)(1 << 24)) >> 25;
    t[18] += carry;
    t[17] -= carry << 25;
    carry = (t[18] + (int64_t)(1 << 25)) >> 26;
    t[18] -= carry << 26;
    int64_t t19 = carry;

    /*
     * First gamma fold: convolve t[10..19] with gamma, add to t[0..9].
     * Output positions: 0 through 9 + GAMMA_25_LIMBS - 1.
     *
     * Offset correction: when BOTH source position (10+k) AND gamma
     * index j are odd, the product is doubled.
     */
    int64_t h[20] = {0};
    for (int i = 0; i < 10; i++)
        h[i] = t[i];

    {
        int64_t overflow[10] = {t[10], t[11], t[12], t[13], t[14], t[15], t[16], t[17], t[18], t19};
        for (int k = 0; k < 10; k++)
        {
            int src_odd = (10 + k) & 1;
            for (int j = 0; j < GAMMA_25_LIMBS; j++)
            {
                int64_t gval = (src_odd && (j & 1)) ? 2 * (int64_t)GAMMA_25[j] : (int64_t)GAMMA_25[j];
                h[k + j] += overflow[k] * gval;
            }
        }
    }

    /*
     * Carry-propagate h[0..9], propagating overflow into h[10..].
     */
    carry = (h[0] + (int64_t)(1 << 25)) >> 26;
    h[1] += carry;
    h[0] -= carry << 26;
    carry = (h[1] + (int64_t)(1 << 24)) >> 25;
    h[2] += carry;
    h[1] -= carry << 25;
    carry = (h[2] + (int64_t)(1 << 25)) >> 26;
    h[3] += carry;
    h[2] -= carry << 26;
    carry = (h[3] + (int64_t)(1 << 24)) >> 25;
    h[4] += carry;
    h[3] -= carry << 25;
    carry = (h[4] + (int64_t)(1 << 25)) >> 26;
    h[5] += carry;
    h[4] -= carry << 26;
    carry = (h[5] + (int64_t)(1 << 24)) >> 25;
    h[6] += carry;
    h[5] -= carry << 25;
    carry = (h[6] + (int64_t)(1 << 25)) >> 26;
    h[7] += carry;
    h[6] -= carry << 26;
    carry = (h[7] + (int64_t)(1 << 24)) >> 25;
    h[8] += carry;
    h[7] -= carry << 25;
    carry = (h[8] + (int64_t)(1 << 25)) >> 26;
    h[9] += carry;
    h[8] -= carry << 26;
    carry = (h[9] + (int64_t)(1 << 24)) >> 25;
    h[10] += carry;
    h[9] -= carry << 25;

    /*
     * Carry-propagate h[10..9+GAMMA_25_LIMBS-1] to canonical width.
     * Without this, large intermediates can overflow int64_t in the second fold.
     */
    {
        int last = 9 + GAMMA_25_LIMBS - 1;
        for (int i = 10; i < last; i++)
        {
            int width = (i & 1) ? 25 : 26;
            carry = (h[i] + (1LL << (width - 1))) >> width;
            h[i + 1] += carry;
            h[i] -= carry << width;
        }
        int last_width = (last & 1) ? 25 : 26;
        carry = (h[last] + (1LL << (last_width - 1))) >> last_width;
        h[last] -= carry << last_width;
        h[last + 1] = carry;
    }

    /*
     * Second gamma fold: convolve h[10..9+GAMMA_25_LIMBS] with gamma.
     * All values are now canonical width, so products fit in int64_t.
     */
    {
        int nover = GAMMA_25_LIMBS + 1; /* overflow count including carry-out */
        int64_t over2[11]; /* max 10+1 */
        for (int i = 0; i < nover; i++)
        {
            over2[i] = h[10 + i];
            h[10 + i] = 0;
        }
        for (int k = 0; k < nover; k++)
        {
            int src_odd = (10 + k) & 1;
            for (int j = 0; j < GAMMA_25_LIMBS; j++)
            {
                int64_t gval = (src_odd && (j & 1)) ? 2 * (int64_t)GAMMA_25[j] : (int64_t)GAMMA_25[j];
                h[k + j] += over2[k] * gval;
            }
        }
    }

    /* Final carry reduction */
    fq25_carry_reduce(out, h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8], h[9]);
}

/*
 * F_q multiplication (32-bit): full 10x10 schoolbook + Crandall reduction.
 *
 * Uses the fi_2 trick from ed25519: pre-double odd-indexed f limbs and use
 * them directly in the schoolbook for both-odd pairs, instead of post-hoc
 * correction. This integrates the radix-2^25.5 offset correction cleanly.
 */
static RANSHAW_FORCE_INLINE void fq25_mul_inline(fq_fe h, const fq_fe f, const fq_fe g)
{
    int32_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int32_t f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];
    int32_t g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3], g4 = g[4];
    int32_t g5 = g[5], g6 = g[6], g7 = g[7], g8 = g[8], g9 = g[9];

    /* Pre-doubled odd-indexed f limbs for offset correction */
    int32_t f1_2 = 2 * f1, f3_2 = 2 * f3, f5_2 = 2 * f5;
    int32_t f7_2 = 2 * f7, f9_2 = 2 * f9;

    /*
     * Full 10x10 schoolbook with integrated fi_2 trick.
     * For even-sum positions (t[0], t[2], ...): both-odd pairs use fi_2.
     * For odd-sum positions (t[1], t[3], ...): no both-odd pairs exist.
     */
    int64_t t[19];

    t[0] = f0 * (int64_t)g0;
    t[1] = f0 * (int64_t)g1 + f1 * (int64_t)g0;
    t[2] = f0 * (int64_t)g2 + f1_2 * (int64_t)g1 + f2 * (int64_t)g0;
    t[3] = f0 * (int64_t)g3 + f1 * (int64_t)g2 + f2 * (int64_t)g1 + f3 * (int64_t)g0;
    t[4] = f0 * (int64_t)g4 + f1_2 * (int64_t)g3 + f2 * (int64_t)g2 + f3_2 * (int64_t)g1 + f4 * (int64_t)g0;
    t[5] =
        f0 * (int64_t)g5 + f1 * (int64_t)g4 + f2 * (int64_t)g3 + f3 * (int64_t)g2 + f4 * (int64_t)g1 + f5 * (int64_t)g0;
    t[6] = f0 * (int64_t)g6 + f1_2 * (int64_t)g5 + f2 * (int64_t)g4 + f3_2 * (int64_t)g3 + f4 * (int64_t)g2
           + f5_2 * (int64_t)g1 + f6 * (int64_t)g0;
    t[7] = f0 * (int64_t)g7 + f1 * (int64_t)g6 + f2 * (int64_t)g5 + f3 * (int64_t)g4 + f4 * (int64_t)g3
           + f5 * (int64_t)g2 + f6 * (int64_t)g1 + f7 * (int64_t)g0;
    t[8] = f0 * (int64_t)g8 + f1_2 * (int64_t)g7 + f2 * (int64_t)g6 + f3_2 * (int64_t)g5 + f4 * (int64_t)g4
           + f5_2 * (int64_t)g3 + f6 * (int64_t)g2 + f7_2 * (int64_t)g1 + f8 * (int64_t)g0;
    t[9] = f0 * (int64_t)g9 + f1 * (int64_t)g8 + f2 * (int64_t)g7 + f3 * (int64_t)g6 + f4 * (int64_t)g5
           + f5 * (int64_t)g4 + f6 * (int64_t)g3 + f7 * (int64_t)g2 + f8 * (int64_t)g1 + f9 * (int64_t)g0;
    t[10] = f1_2 * (int64_t)g9 + f2 * (int64_t)g8 + f3_2 * (int64_t)g7 + f4 * (int64_t)g6 + f5_2 * (int64_t)g5
            + f6 * (int64_t)g4 + f7_2 * (int64_t)g3 + f8 * (int64_t)g2 + f9_2 * (int64_t)g1;
    t[11] = f2 * (int64_t)g9 + f3 * (int64_t)g8 + f4 * (int64_t)g7 + f5 * (int64_t)g6 + f6 * (int64_t)g5
            + f7 * (int64_t)g4 + f8 * (int64_t)g3 + f9 * (int64_t)g2;
    t[12] = f3_2 * (int64_t)g9 + f4 * (int64_t)g8 + f5_2 * (int64_t)g7 + f6 * (int64_t)g6 + f7_2 * (int64_t)g5
            + f8 * (int64_t)g4 + f9_2 * (int64_t)g3;
    t[13] =
        f4 * (int64_t)g9 + f5 * (int64_t)g8 + f6 * (int64_t)g7 + f7 * (int64_t)g6 + f8 * (int64_t)g5 + f9 * (int64_t)g4;
    t[14] = f5_2 * (int64_t)g9 + f6 * (int64_t)g8 + f7_2 * (int64_t)g7 + f8 * (int64_t)g6 + f9_2 * (int64_t)g5;
    t[15] = f6 * (int64_t)g9 + f7 * (int64_t)g8 + f8 * (int64_t)g7 + f9 * (int64_t)g6;
    t[16] = f7_2 * (int64_t)g9 + f8 * (int64_t)g8 + f9_2 * (int64_t)g7;
    t[17] = f8 * (int64_t)g9 + f9 * (int64_t)g8;
    t[18] = f9_2 * (int64_t)g9;

    fq25_reduce_full(h, t);
}

/*
 * F_q squaring (32-bit): full 10x10 schoolbook (with squaring optimization)
 * + Crandall reduction.
 *
 * Offset correction is integrated: for both-odd cross-terms, use fi_2*fj_2
 * (4x factor = 2x cross-term + 2x offset). For both-odd diagonals, use
 * fi_2*fi (2x offset on the single fi^2 term).
 */
static RANSHAW_FORCE_INLINE void fq25_sq_inline(fq_fe h, const fq_fe f)
{
    int32_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int32_t f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];

    /* Even-index doubled (for standard squaring cross-term 2x) */
    int32_t f0_2 = 2 * f0, f2_2 = 2 * f2, f4_2 = 2 * f4;
    int32_t f6_2 = 2 * f6, f8_2 = 2 * f8;
    /* Odd-index doubled (for cross-term 2x AND offset correction 2x) */
    int32_t f1_2 = 2 * f1, f3_2 = 2 * f3, f5_2 = 2 * f5;
    int32_t f7_2 = 2 * f7, f9_2 = 2 * f9;

    int64_t t[19];

    t[0] = f0 * (int64_t)f0;
    t[1] = f0_2 * (int64_t)f1;
    t[2] = f0_2 * (int64_t)f2 + f1_2 * (int64_t)f1;
    t[3] = f0_2 * (int64_t)f3 + f1_2 * (int64_t)f2;
    t[4] = f0_2 * (int64_t)f4 + f1_2 * (int64_t)f3_2 + f2 * (int64_t)f2;
    t[5] = f0_2 * (int64_t)f5 + f1_2 * (int64_t)f4 + f2_2 * (int64_t)f3;
    t[6] = f0_2 * (int64_t)f6 + f1_2 * (int64_t)f5_2 + f2_2 * (int64_t)f4 + f3_2 * (int64_t)f3;
    t[7] = f0_2 * (int64_t)f7 + f1_2 * (int64_t)f6 + f2_2 * (int64_t)f5 + f3_2 * (int64_t)f4;
    t[8] = f0_2 * (int64_t)f8 + f1_2 * (int64_t)f7_2 + f2_2 * (int64_t)f6 + f3_2 * (int64_t)f5_2 + f4 * (int64_t)f4;
    t[9] = f0_2 * (int64_t)f9 + f1_2 * (int64_t)f8 + f2_2 * (int64_t)f7 + f3_2 * (int64_t)f6 + f4_2 * (int64_t)f5;
    t[10] = f1_2 * (int64_t)f9_2 + f2_2 * (int64_t)f8 + f3_2 * (int64_t)f7_2 + f4_2 * (int64_t)f6 + f5_2 * (int64_t)f5;
    t[11] = f2_2 * (int64_t)f9 + f3_2 * (int64_t)f8 + f4_2 * (int64_t)f7 + f5_2 * (int64_t)f6;
    t[12] = f3_2 * (int64_t)f9_2 + f4_2 * (int64_t)f8 + f5_2 * (int64_t)f7_2 + f6 * (int64_t)f6;
    t[13] = f4_2 * (int64_t)f9 + f5_2 * (int64_t)f8 + f6_2 * (int64_t)f7;
    t[14] = f5_2 * (int64_t)f9_2 + f6_2 * (int64_t)f8 + f7_2 * (int64_t)f7;
    t[15] = f6_2 * (int64_t)f9 + f7_2 * (int64_t)f8;
    t[16] = f7_2 * (int64_t)f9_2 + f8 * (int64_t)f8;
    t[17] = f8_2 * (int64_t)f9;
    t[18] = f9_2 * (int64_t)f9;

    fq25_reduce_full(h, t);
}

#endif // RANSHAW_PORTABLE_FQ25_INLINE_H
