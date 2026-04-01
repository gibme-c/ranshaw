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
 * @file fq10_avx2.h
 * @brief AVX2 radix-2^25.5 Fq field element operations using scalar int64_t.
 *
 * Scalar radix-2^25.5 representation for the Crandall prime q = 2^255 - gamma,
 * where gamma = 239666463199878229209741112730228557729 (~128 bits). This exists
 * for the same reason as fp10_avx2.h: MSVC cannot keep _umul128 results in
 * registers when force-inlining into curve bodies, causing severe register
 * spilling. The radix-2^25.5 representation avoids 128-bit arithmetic entirely.
 *
 * The key difference from fp10 is the carry wrap: instead of multiplying the
 * carry out of limb 9 by 19 (as for p = 2^255 - 19), we multiply by gamma
 * which spans 5 limbs in radix-2^25.5. This gamma fold replaces the simple
 * multiply-by-19 wrap-around used in Fp arithmetic.
 *
 * For multiplication, the inline ×19 folding trick used in fp10 cannot work
 * because gamma has 5 limbs (not 1). Instead, the full 10×10 schoolbook
 * produces 19 accumulators, which are then reduced via two rounds of gamma
 * folding (the Crandall reduction).
 *
 * Provides fq51-to-fq10 and fq10-to-fq51 conversion, basic arithmetic
 * (add, sub, neg, copy, cmov), and multiplication/squaring with Crandall
 * reduction.
 */

#ifndef RANSHAW_X64_AVX2_FQ10_AVX2_H
#define RANSHAW_X64_AVX2_FQ10_AVX2_H

#include "fq_ops.h"
#include "portable/fq25.h"
#include "ranshaw_ct_barrier.h"
#include "ranshaw_platform.h"
#include "x64/fq51.h"

#include <immintrin.h>

#if defined(_MSC_VER)
#define FQ10_AVX2_FORCE_INLINE __forceinline
#else
#define FQ10_AVX2_FORCE_INLINE inline __attribute__((always_inline))
#endif

typedef int64_t fq10[10];

static const int64_t FQ10_MASK26 = (1LL << 26) - 1;
static const int64_t FQ10_MASK25 = (1LL << 25) - 1;

/**
 * @brief Convert fq_fe (5x51 radix-2^51) to fq10 (radix-2^25.5, int64_t[10]).
 *
 * Trivial split: each 51-bit limb splits into 26-bit low + 25-bit high sub-limbs.
 * out[2k] = src[k] & 0x3FFFFFF (26 bits), out[2k+1] = src[k] >> 26 (25 bits).
 */
static FQ10_AVX2_FORCE_INLINE void fq51_to_fq10(fq10 out, const fq_fe src)
{
    out[0] = (int64_t)(src[0] & 0x3FFFFFFULL);
    out[1] = (int64_t)(src[0] >> 26);
    out[2] = (int64_t)(src[1] & 0x3FFFFFFULL);
    out[3] = (int64_t)(src[1] >> 26);
    out[4] = (int64_t)(src[2] & 0x3FFFFFFULL);
    out[5] = (int64_t)(src[2] >> 26);
    out[6] = (int64_t)(src[3] & 0x3FFFFFFULL);
    out[7] = (int64_t)(src[3] >> 26);
    out[8] = (int64_t)(src[4] & 0x3FFFFFFULL);
    out[9] = (int64_t)(src[4] >> 26);
}

/**
 * @brief Convert fq10 (radix-2^25.5, int64_t[10]) to fq_fe (5x51 radix-2^51).
 *
 * Carry-propagate with gamma fold, then merge pairs: out[k] = t[2k] | (t[2k+1] << 26).
 */
static FQ10_AVX2_FORCE_INLINE void fq10_to_fq51(fq_fe out, const fq10 src)
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

    // Carry-propagate to canonical range
    c = t[0] >> 26;
    t[1] += c;
    t[0] &= FQ10_MASK26;
    c = t[1] >> 25;
    t[2] += c;
    t[1] &= FQ10_MASK25;
    c = t[2] >> 26;
    t[3] += c;
    t[2] &= FQ10_MASK26;
    c = t[3] >> 25;
    t[4] += c;
    t[3] &= FQ10_MASK25;
    c = t[4] >> 26;
    t[5] += c;
    t[4] &= FQ10_MASK26;
    c = t[5] >> 25;
    t[6] += c;
    t[5] &= FQ10_MASK25;
    c = t[6] >> 26;
    t[7] += c;
    t[6] &= FQ10_MASK26;
    c = t[7] >> 25;
    t[8] += c;
    t[7] &= FQ10_MASK25;
    c = t[8] >> 26;
    t[9] += c;
    t[8] &= FQ10_MASK26;
    c = t[9] >> 25;
    t[9] &= FQ10_MASK25;

    // Gamma fold: carry out of limb 9 * gamma
    t[0] += c * (int64_t)GAMMA_25[0];
    t[1] += c * (int64_t)GAMMA_25[1];
    t[2] += c * (int64_t)GAMMA_25[2];
    t[3] += c * (int64_t)GAMMA_25[3];
    t[4] += c * (int64_t)GAMMA_25[4];

    // Re-carry all limbs 0..9
    c = t[0] >> 26;
    t[1] += c;
    t[0] &= FQ10_MASK26;
    c = t[1] >> 25;
    t[2] += c;
    t[1] &= FQ10_MASK25;
    c = t[2] >> 26;
    t[3] += c;
    t[2] &= FQ10_MASK26;
    c = t[3] >> 25;
    t[4] += c;
    t[3] &= FQ10_MASK25;
    c = t[4] >> 26;
    t[5] += c;
    t[4] &= FQ10_MASK26;
    c = t[5] >> 25;
    t[6] += c;
    t[5] &= FQ10_MASK25;
    c = t[6] >> 26;
    t[7] += c;
    t[6] &= FQ10_MASK26;
    c = t[7] >> 25;
    t[8] += c;
    t[7] &= FQ10_MASK25;
    c = t[8] >> 26;
    t[9] += c;
    t[8] &= FQ10_MASK26;

    // Merge pairs: out[k] = t[2k] | (t[2k+1] << 26)
    out[0] = (uint64_t)t[0] | ((uint64_t)t[1] << 26);
    out[1] = (uint64_t)t[2] | ((uint64_t)t[3] << 26);
    out[2] = (uint64_t)t[4] | ((uint64_t)t[5] << 26);
    out[3] = (uint64_t)t[6] | ((uint64_t)t[7] << 26);
    out[4] = (uint64_t)t[8] | ((uint64_t)t[9] << 26);
}

/**
 * @brief fq10 addition: h = f + g (no carry propagation).
 */
static FQ10_AVX2_FORCE_INLINE void fq10_add(fq10 h, const fq10 f, const fq10 g)
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
 * @brief fq10 subtraction: h = f - g with 2*q bias + carry with gamma fold.
 *
 * Adds 2*q to avoid underflow, subtracts g, then carry-propagates with
 * gamma fold at limb 9. Uses signed int64_t arithmetic so the 2q multiplier
 * is sufficient regardless of subtrahend size. The 2*Q_25 bias values are:
 *   {14807230, 2302710, 65657946, 14915376, 39685976,
 *    67108862, 134217726, 67108862, 134217726, 67108862}
 */
static FQ10_AVX2_FORCE_INLINE void fq10_sub(fq10 h, const fq10 f, const fq10 g)
{
    h[0] = f[0] + 14807230LL - g[0];
    h[1] = f[1] + 2302710LL - g[1];
    h[2] = f[2] + 65657946LL - g[2];
    h[3] = f[3] + 14915376LL - g[3];
    h[4] = f[4] + 39685976LL - g[4];
    h[5] = f[5] + 67108862LL - g[5];
    h[6] = f[6] + 134217726LL - g[6];
    h[7] = f[7] + 67108862LL - g[7];
    h[8] = f[8] + 134217726LL - g[8];
    h[9] = f[9] + 67108862LL - g[9];

    // Carry propagation with gamma fold
    int64_t c;
    c = h[0] >> 26;
    h[1] += c;
    h[0] &= FQ10_MASK26;
    c = h[1] >> 25;
    h[2] += c;
    h[1] &= FQ10_MASK25;
    c = h[2] >> 26;
    h[3] += c;
    h[2] &= FQ10_MASK26;
    c = h[3] >> 25;
    h[4] += c;
    h[3] &= FQ10_MASK25;
    c = h[4] >> 26;
    h[5] += c;
    h[4] &= FQ10_MASK26;
    c = h[5] >> 25;
    h[6] += c;
    h[5] &= FQ10_MASK25;
    c = h[6] >> 26;
    h[7] += c;
    h[6] &= FQ10_MASK26;
    c = h[7] >> 25;
    h[8] += c;
    h[7] &= FQ10_MASK25;
    c = h[8] >> 26;
    h[9] += c;
    h[8] &= FQ10_MASK26;
    c = h[9] >> 25;
    h[9] &= FQ10_MASK25;

    // Gamma fold
    h[0] += c * (int64_t)GAMMA_25[0];
    h[1] += c * (int64_t)GAMMA_25[1];
    h[2] += c * (int64_t)GAMMA_25[2];
    h[3] += c * (int64_t)GAMMA_25[3];
    h[4] += c * (int64_t)GAMMA_25[4];

    // Re-carry all limbs 0..9 (gamma fold affects limbs 0-4; carry from
    // limb 4 can cascade through 5-9 when GAMMA_25[4] + carry exceeds 26 bits)
    c = h[0] >> 26;
    h[1] += c;
    h[0] &= FQ10_MASK26;
    c = h[1] >> 25;
    h[2] += c;
    h[1] &= FQ10_MASK25;
    c = h[2] >> 26;
    h[3] += c;
    h[2] &= FQ10_MASK26;
    c = h[3] >> 25;
    h[4] += c;
    h[3] &= FQ10_MASK25;
    c = h[4] >> 26;
    h[5] += c;
    h[4] &= FQ10_MASK26;
    c = h[5] >> 25;
    h[6] += c;
    h[5] &= FQ10_MASK25;
    c = h[6] >> 26;
    h[7] += c;
    h[6] &= FQ10_MASK26;
    c = h[7] >> 25;
    h[8] += c;
    h[7] &= FQ10_MASK25;
    c = h[8] >> 26;
    h[9] += c;
    h[8] &= FQ10_MASK26;
}

/**
 * @brief fq10 negation: h = -f (mod q).
 */
static FQ10_AVX2_FORCE_INLINE void fq10_neg(fq10 h, const fq10 f)
{
    fq10 zero = {0};
    fq10_sub(h, zero, f);
}

/**
 * @brief fq10 copy: h = f.
 */
static FQ10_AVX2_FORCE_INLINE void fq10_copy(fq10 h, const fq10 f)
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
 * @brief fq10 conditional move: if b, then t = u (constant-time).
 */
static FQ10_AVX2_FORCE_INLINE void fq10_cmov(fq10 t, const fq10 u, int64_t b)
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
 * @brief Crandall carry-reduction for 10-limb fq10 accumulators.
 *
 * Takes 10 int64_t values, carry-propagates with gamma fold at limb 9.
 * Used after schoolbook products have been reduced to 10 accumulators.
 */
static FQ10_AVX2_FORCE_INLINE void fq10_carry_reduce(
    fq10 out,
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
    carry = (h0 + (1LL << 25)) >> 26;
    h1 += carry;
    h0 -= carry << 26;
    carry = (h4 + (1LL << 25)) >> 26;
    h5 += carry;
    h4 -= carry << 26;
    carry = (h1 + (1LL << 24)) >> 25;
    h2 += carry;
    h1 -= carry << 25;
    carry = (h5 + (1LL << 24)) >> 25;
    h6 += carry;
    h5 -= carry << 25;
    carry = (h2 + (1LL << 25)) >> 26;
    h3 += carry;
    h2 -= carry << 26;
    carry = (h6 + (1LL << 25)) >> 26;
    h7 += carry;
    h6 -= carry << 26;
    carry = (h3 + (1LL << 24)) >> 25;
    h4 += carry;
    h3 -= carry << 25;
    carry = (h7 + (1LL << 24)) >> 25;
    h8 += carry;
    h7 -= carry << 25;
    carry = (h4 + (1LL << 25)) >> 26;
    h5 += carry;
    h4 -= carry << 26;
    carry = (h8 + (1LL << 25)) >> 26;
    h9 += carry;
    h8 -= carry << 26;

    /* Gamma fold: carry from h9 wraps as carry * gamma */
    carry = (h9 + (1LL << 24)) >> 25;
    h9 -= carry << 25;
    h0 += carry * (int64_t)GAMMA_25[0];
    h1 += carry * (int64_t)GAMMA_25[1];
    h2 += carry * (int64_t)GAMMA_25[2];
    h3 += carry * (int64_t)GAMMA_25[3];
    h4 += carry * (int64_t)GAMMA_25[4];

    /* Second carry pass to normalize after gamma fold */
    carry = (h0 + (1LL << 25)) >> 26;
    h1 += carry;
    h0 -= carry << 26;
    carry = (h1 + (1LL << 24)) >> 25;
    h2 += carry;
    h1 -= carry << 25;
    carry = (h2 + (1LL << 25)) >> 26;
    h3 += carry;
    h2 -= carry << 26;
    carry = (h3 + (1LL << 24)) >> 25;
    h4 += carry;
    h3 -= carry << 25;
    carry = (h4 + (1LL << 25)) >> 26;
    h5 += carry;
    h4 -= carry << 26;
    carry = (h5 + (1LL << 24)) >> 25;
    h6 += carry;
    h5 -= carry << 25;
    carry = (h6 + (1LL << 25)) >> 26;
    h7 += carry;
    h6 -= carry << 26;
    carry = (h7 + (1LL << 24)) >> 25;
    h8 += carry;
    h7 -= carry << 25;
    carry = (h8 + (1LL << 25)) >> 26;
    h9 += carry;
    h8 -= carry << 26;
    carry = (h9 + (1LL << 24)) >> 25;
    h9 -= carry << 25;

    /* Second gamma fold (carry should be very small, often 0) */
    h0 += carry * (int64_t)GAMMA_25[0];
    h1 += carry * (int64_t)GAMMA_25[1];
    h2 += carry * (int64_t)GAMMA_25[2];
    h3 += carry * (int64_t)GAMMA_25[3];
    h4 += carry * (int64_t)GAMMA_25[4];

    /* Final carry for limbs 0-4 */
    carry = (h0 + (1LL << 25)) >> 26;
    h1 += carry;
    h0 -= carry << 26;
    carry = (h1 + (1LL << 24)) >> 25;
    h2 += carry;
    h1 -= carry << 25;
    carry = (h2 + (1LL << 25)) >> 26;
    h3 += carry;
    h2 -= carry << 26;
    carry = (h3 + (1LL << 24)) >> 25;
    h4 += carry;
    h3 -= carry << 25;

    out[0] = h0;
    out[1] = h1;
    out[2] = h2;
    out[3] = h3;
    out[4] = h4;
    out[5] = h5;
    out[6] = h6;
    out[7] = h7;
    out[8] = h8;
    out[9] = h9;
}

/**
 * @brief Full Crandall reduction for 19 int64_t accumulators (fq10 version).
 *
 * After a 10x10 schoolbook producing 19 accumulators with radix-2^25.5 offset
 * correction already applied, carry-propagate, extract the upper part
 * (positions 10-18+), convolve with gamma, and fold back into positions 0-9.
 *
 * This is the int64_t version of fq25_reduce_full from portable/fq25_inline.h,
 * outputting int64_t[10] instead of int32_t[10].
 */
static FQ10_AVX2_FORCE_INLINE void fq10_reduce_full(fq10 out, int64_t t[19])
{
    int64_t carry;
    const int64_t g0 = (int64_t)GAMMA_25[0];
    const int64_t g1 = (int64_t)GAMMA_25[1];
    const int64_t g2 = (int64_t)GAMMA_25[2];
    const int64_t g3 = (int64_t)GAMMA_25[3];
    const int64_t g4 = (int64_t)GAMMA_25[4];
    /* Pre-doubled odd gamma limbs for offset correction */
    const int64_t g1_2 = 2 * g1;
    const int64_t g3_2 = 2 * g3;

    /*
     * Carry-propagate t[0..18] into canonical-width limbs.
     */
    carry = (t[0] + (1LL << 25)) >> 26;
    t[1] += carry;
    t[0] -= carry << 26;
    carry = (t[1] + (1LL << 24)) >> 25;
    t[2] += carry;
    t[1] -= carry << 25;
    carry = (t[2] + (1LL << 25)) >> 26;
    t[3] += carry;
    t[2] -= carry << 26;
    carry = (t[3] + (1LL << 24)) >> 25;
    t[4] += carry;
    t[3] -= carry << 25;
    carry = (t[4] + (1LL << 25)) >> 26;
    t[5] += carry;
    t[4] -= carry << 26;
    carry = (t[5] + (1LL << 24)) >> 25;
    t[6] += carry;
    t[5] -= carry << 25;
    carry = (t[6] + (1LL << 25)) >> 26;
    t[7] += carry;
    t[6] -= carry << 26;
    carry = (t[7] + (1LL << 24)) >> 25;
    t[8] += carry;
    t[7] -= carry << 25;
    carry = (t[8] + (1LL << 25)) >> 26;
    t[9] += carry;
    t[8] -= carry << 26;
    carry = (t[9] + (1LL << 24)) >> 25;
    t[10] += carry;
    t[9] -= carry << 25;
    carry = (t[10] + (1LL << 25)) >> 26;
    t[11] += carry;
    t[10] -= carry << 26;
    carry = (t[11] + (1LL << 24)) >> 25;
    t[12] += carry;
    t[11] -= carry << 25;
    carry = (t[12] + (1LL << 25)) >> 26;
    t[13] += carry;
    t[12] -= carry << 26;
    carry = (t[13] + (1LL << 24)) >> 25;
    t[14] += carry;
    t[13] -= carry << 25;
    carry = (t[14] + (1LL << 25)) >> 26;
    t[15] += carry;
    t[14] -= carry << 26;
    carry = (t[15] + (1LL << 24)) >> 25;
    t[16] += carry;
    t[15] -= carry << 25;
    carry = (t[16] + (1LL << 25)) >> 26;
    t[17] += carry;
    t[16] -= carry << 26;
    carry = (t[17] + (1LL << 24)) >> 25;
    t[18] += carry;
    t[17] -= carry << 25;
    carry = (t[18] + (1LL << 25)) >> 26;
    t[18] -= carry << 26;
    int64_t t19 = carry;

    /*
     * First gamma fold: multiply t[10..19] by gamma, add to positions 0..13.
     *
     * Offset correction: when BOTH (k) and (j) are odd, double the term.
     * Even positions (k=10,12,14,16,18): no correction needed.
     * Odd positions (k=11,13,15,17,19): use g1_2, g3_2 for odd gamma indices.
     */
    int64_t h0 = t[0] + t[10] * g0;
    int64_t h1 = t[1] + t[10] * g1 + t[11] * g0;
    int64_t h2 = t[2] + t[10] * g2 + t[11] * g1_2 + t[12] * g0;
    int64_t h3 = t[3] + t[10] * g3 + t[11] * g2 + t[12] * g1 + t[13] * g0;
    int64_t h4 = t[4] + t[10] * g4 + t[11] * g3_2 + t[12] * g2 + t[13] * g1_2 + t[14] * g0;
    int64_t h5 = t[5] + t[11] * g4 + t[12] * g3 + t[13] * g2 + t[14] * g1 + t[15] * g0;
    int64_t h6 = t[6] + t[12] * g4 + t[13] * g3_2 + t[14] * g2 + t[15] * g1_2 + t[16] * g0;
    int64_t h7 = t[7] + t[13] * g4 + t[14] * g3 + t[15] * g2 + t[16] * g1 + t[17] * g0;
    int64_t h8 = t[8] + t[14] * g4 + t[15] * g3_2 + t[16] * g2 + t[17] * g1_2 + t[18] * g0;
    int64_t h9 = t[9] + t[15] * g4 + t[16] * g3 + t[17] * g2 + t[18] * g1 + t19 * g0;
    int64_t h10 = t[16] * g4 + t[17] * g3_2 + t[18] * g2 + t19 * g1_2;
    int64_t h11 = t[17] * g4 + t[18] * g3 + t19 * g2;
    int64_t h12 = t[18] * g4 + t19 * g3_2;
    int64_t h13 = t19 * g4;

    /*
     * Carry-propagate h[0..9], propagating overflow into h[10..13].
     */
    carry = (h0 + (1LL << 25)) >> 26;
    h1 += carry;
    h0 -= carry << 26;
    carry = (h1 + (1LL << 24)) >> 25;
    h2 += carry;
    h1 -= carry << 25;
    carry = (h2 + (1LL << 25)) >> 26;
    h3 += carry;
    h2 -= carry << 26;
    carry = (h3 + (1LL << 24)) >> 25;
    h4 += carry;
    h3 -= carry << 25;
    carry = (h4 + (1LL << 25)) >> 26;
    h5 += carry;
    h4 -= carry << 26;
    carry = (h5 + (1LL << 24)) >> 25;
    h6 += carry;
    h5 -= carry << 25;
    carry = (h6 + (1LL << 25)) >> 26;
    h7 += carry;
    h6 -= carry << 26;
    carry = (h7 + (1LL << 24)) >> 25;
    h8 += carry;
    h7 -= carry << 25;
    carry = (h8 + (1LL << 25)) >> 26;
    h9 += carry;
    h8 -= carry << 26;
    carry = (h9 + (1LL << 24)) >> 25;
    h10 += carry;
    h9 -= carry << 25;

    /*
     * Carry-propagate h[10..13] to canonical width, including carry out
     * of h13 into h14. Without this, h13 can be ~49 bits, causing
     * h13 * gamma[j] to overflow int64_t in the second fold.
     */
    carry = (h10 + (1LL << 25)) >> 26;
    h11 += carry;
    h10 -= carry << 26;
    carry = (h11 + (1LL << 24)) >> 25;
    h12 += carry;
    h11 -= carry << 25;
    carry = (h12 + (1LL << 25)) >> 26;
    h13 += carry;
    h12 -= carry << 26;
    carry = (h13 + (1LL << 24)) >> 25;
    h13 -= carry << 25;
    int64_t h14 = carry;

    /*
     * Second gamma fold: h[10..14] * gamma -> positions 0..8.
     *
     * Offset corrections:
     *   h10 at pos 10 (even): no correction
     *   h11 at pos 11 (odd): double when gamma index j is odd (use g1_2, g3_2)
     *   h12 at pos 12 (even): no correction
     *   h13 at pos 13 (odd): double when gamma index j is odd (use g1_2, g3_2)
     *   h14 at pos 14 (even): no correction
     */
    h0 += h10 * g0;
    h1 += h10 * g1 + h11 * g0;
    h2 += h10 * g2 + h11 * g1_2 + h12 * g0;
    h3 += h10 * g3 + h11 * g2 + h12 * g1 + h13 * g0;
    h4 += h10 * g4 + h11 * g3_2 + h12 * g2 + h13 * g1_2 + h14 * g0;
    h5 += h11 * g4 + h12 * g3 + h13 * g2 + h14 * g1;
    h6 += h12 * g4 + h13 * g3_2 + h14 * g2;
    h7 += h13 * g4 + h14 * g3;
    h8 += h14 * g4;

    /* Final carry reduction */
    fq10_carry_reduce(out, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

/**
 * @brief fq10 schoolbook multiplication: h = f * g (mod 2^255 - gamma).
 *
 * Full 10x10 schoolbook producing 19 accumulators, followed by Crandall
 * reduction (gamma fold). Unlike fp10_mul which uses inline x19 folding,
 * gamma has 5 limbs so we produce all 19 positions and reduce via
 * fq10_reduce_full.
 */
static FQ10_AVX2_FORCE_INLINE void fq10_mul(fq10 h, const fq10 f, const fq10 g)
{
    int64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int64_t f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];
    int64_t g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3], g4 = g[4];
    int64_t g5 = g[5], g6 = g[6], g7 = g[7], g8 = g[8], g9 = g[9];

    /* Pre-doubled odd-indexed f limbs for offset correction */
    int64_t f1_2 = 2 * f1, f3_2 = 2 * f3, f5_2 = 2 * f5;
    int64_t f7_2 = 2 * f7, f9_2 = 2 * f9;

    /* Full 10x10 schoolbook with integrated fi_2 trick */
    int64_t t[19];

    t[0] = f0 * g0;
    t[1] = f0 * g1 + f1 * g0;
    t[2] = f0 * g2 + f1_2 * g1 + f2 * g0;
    t[3] = f0 * g3 + f1 * g2 + f2 * g1 + f3 * g0;
    t[4] = f0 * g4 + f1_2 * g3 + f2 * g2 + f3_2 * g1 + f4 * g0;
    t[5] = f0 * g5 + f1 * g4 + f2 * g3 + f3 * g2 + f4 * g1 + f5 * g0;
    t[6] = f0 * g6 + f1_2 * g5 + f2 * g4 + f3_2 * g3 + f4 * g2 + f5_2 * g1 + f6 * g0;
    t[7] = f0 * g7 + f1 * g6 + f2 * g5 + f3 * g4 + f4 * g3 + f5 * g2 + f6 * g1 + f7 * g0;
    t[8] = f0 * g8 + f1_2 * g7 + f2 * g6 + f3_2 * g5 + f4 * g4 + f5_2 * g3 + f6 * g2 + f7_2 * g1 + f8 * g0;
    t[9] = f0 * g9 + f1 * g8 + f2 * g7 + f3 * g6 + f4 * g5 + f5 * g4 + f6 * g3 + f7 * g2 + f8 * g1 + f9 * g0;
    t[10] = f1_2 * g9 + f2 * g8 + f3_2 * g7 + f4 * g6 + f5_2 * g5 + f6 * g4 + f7_2 * g3 + f8 * g2 + f9_2 * g1;
    t[11] = f2 * g9 + f3 * g8 + f4 * g7 + f5 * g6 + f6 * g5 + f7 * g4 + f8 * g3 + f9 * g2;
    t[12] = f3_2 * g9 + f4 * g8 + f5_2 * g7 + f6 * g6 + f7_2 * g5 + f8 * g4 + f9_2 * g3;
    t[13] = f4 * g9 + f5 * g8 + f6 * g7 + f7 * g6 + f8 * g5 + f9 * g4;
    t[14] = f5_2 * g9 + f6 * g8 + f7_2 * g7 + f8 * g6 + f9_2 * g5;
    t[15] = f6 * g9 + f7 * g8 + f8 * g7 + f9 * g6;
    t[16] = f7_2 * g9 + f8 * g8 + f9_2 * g7;
    t[17] = f8 * g9 + f9 * g8;
    t[18] = f9_2 * g9;

    fq10_reduce_full(h, t);
}

/**
 * @brief fq10 squaring: h = f^2 (mod 2^255 - gamma).
 */
static FQ10_AVX2_FORCE_INLINE void fq10_sq(fq10 h, const fq10 f)
{
    int64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int64_t f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];

    int64_t f0_2 = 2 * f0, f2_2 = 2 * f2, f4_2 = 2 * f4;
    int64_t f6_2 = 2 * f6, f8_2 = 2 * f8;
    int64_t f1_2 = 2 * f1, f3_2 = 2 * f3, f5_2 = 2 * f5;
    int64_t f7_2 = 2 * f7, f9_2 = 2 * f9;

    int64_t t[19];

    t[0] = f0 * f0;
    t[1] = f0_2 * f1;
    t[2] = f0_2 * f2 + f1_2 * f1;
    t[3] = f0_2 * f3 + f1_2 * f2;
    t[4] = f0_2 * f4 + f1_2 * f3_2 + f2 * f2;
    t[5] = f0_2 * f5 + f1_2 * f4 + f2_2 * f3;
    t[6] = f0_2 * f6 + f1_2 * f5_2 + f2_2 * f4 + f3_2 * f3;
    t[7] = f0_2 * f7 + f1_2 * f6 + f2_2 * f5 + f3_2 * f4;
    t[8] = f0_2 * f8 + f1_2 * f7_2 + f2_2 * f6 + f3_2 * f5_2 + f4 * f4;
    t[9] = f0_2 * f9 + f1_2 * f8 + f2_2 * f7 + f3_2 * f6 + f4_2 * f5;
    t[10] = f1_2 * f9_2 + f2_2 * f8 + f3_2 * f7_2 + f4_2 * f6 + f5_2 * f5;
    t[11] = f2_2 * f9 + f3_2 * f8 + f4_2 * f7 + f5_2 * f6;
    t[12] = f3_2 * f9_2 + f4_2 * f8 + f5_2 * f7_2 + f6 * f6;
    t[13] = f4_2 * f9 + f5_2 * f8 + f6_2 * f7;
    t[14] = f5_2 * f9_2 + f6_2 * f8 + f7_2 * f7;
    t[15] = f6_2 * f9 + f7_2 * f8;
    t[16] = f7_2 * f9_2 + f8 * f8;
    t[17] = f8_2 * f9;
    t[18] = f9_2 * f9;

    fq10_reduce_full(h, t);
}

/**
 * @brief fq10 double-squaring: h = 2 * f^2 (mod 2^255 - gamma).
 */
static FQ10_AVX2_FORCE_INLINE void fq10_sq2(fq10 h, const fq10 f)
{
    int64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int64_t f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];

    int64_t f0_2 = 2 * f0, f2_2 = 2 * f2, f4_2 = 2 * f4;
    int64_t f6_2 = 2 * f6, f8_2 = 2 * f8;
    int64_t f1_2 = 2 * f1, f3_2 = 2 * f3, f5_2 = 2 * f5;
    int64_t f7_2 = 2 * f7, f9_2 = 2 * f9;

    int64_t t[19];

    t[0] = f0 * f0;
    t[1] = f0_2 * f1;
    t[2] = f0_2 * f2 + f1_2 * f1;
    t[3] = f0_2 * f3 + f1_2 * f2;
    t[4] = f0_2 * f4 + f1_2 * f3_2 + f2 * f2;
    t[5] = f0_2 * f5 + f1_2 * f4 + f2_2 * f3;
    t[6] = f0_2 * f6 + f1_2 * f5_2 + f2_2 * f4 + f3_2 * f3;
    t[7] = f0_2 * f7 + f1_2 * f6 + f2_2 * f5 + f3_2 * f4;
    t[8] = f0_2 * f8 + f1_2 * f7_2 + f2_2 * f6 + f3_2 * f5_2 + f4 * f4;
    t[9] = f0_2 * f9 + f1_2 * f8 + f2_2 * f7 + f3_2 * f6 + f4_2 * f5;
    t[10] = f1_2 * f9_2 + f2_2 * f8 + f3_2 * f7_2 + f4_2 * f6 + f5_2 * f5;
    t[11] = f2_2 * f9 + f3_2 * f8 + f4_2 * f7 + f5_2 * f6;
    t[12] = f3_2 * f9_2 + f4_2 * f8 + f5_2 * f7_2 + f6 * f6;
    t[13] = f4_2 * f9 + f5_2 * f8 + f6_2 * f7;
    t[14] = f5_2 * f9_2 + f6_2 * f8 + f7_2 * f7;
    t[15] = f6_2 * f9 + f7_2 * f8;
    t[16] = f7_2 * f9_2 + f8 * f8;
    t[17] = f8_2 * f9;
    t[18] = f9_2 * f9;

    /* Double all accumulators before reduction */
    for (int i = 0; i < 19; i++)
        t[i] += t[i];

    fq10_reduce_full(h, t);
}

/**
 * @brief fq10 repeated squaring: h = f^(2^n) (mod 2^255 - gamma).
 */
static FQ10_AVX2_FORCE_INLINE void fq10_sqn(fq10 h, const fq10 f, int n)
{
    fq10_sq(h, f);
    for (int i = 1; i < n; i++)
        fq10_sq(h, h);
}

#endif // RANSHAW_X64_AVX2_FQ10_AVX2_H
