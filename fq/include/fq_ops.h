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
 * @file fq_ops.h
 * @brief Basic F_q arithmetic: add, sub, neg, copy, zero, one, conditional operations.
 *
 * F_q uses 8q bias for subtraction (not 4q like F_p) because gamma ~ 2^126
 * makes lower limbs much smaller than 2^51.
 */

#ifndef RANSHAW_FQ_OPS_H
#define RANSHAW_FQ_OPS_H

#include "fq.h"

#include <cstring>

#if RANSHAW_PLATFORM_64BIT
#include "x64/fq51.h"

/*
 * Addition in radix-2^51: 5 independent limb adds, no carry propagation.
 * Lazy reduction — identical to fp_add. Limbs may exceed 51 bits; mul/sq
 * handles the extra width (column accumulation has >21 bits headroom).
 */
static inline void fq_add(fq_fe h, const fq_fe f, const fq_fe g)
{
    h[0] = f[0] + g[0];
    h[1] = f[1] + g[1];
    h[2] = f[2] + g[2];
    h[3] = f[3] + g[3];
    h[4] = f[4] + g[4];
}

/*
 * Subtraction in radix-2^51: add 8*q bias, subtract, carry chain with
 * gamma fold of top carry. Uses 8q bias (not 2q or 4q) because q's lower
 * limbs are << 2^51 (gamma ≈ 2^127), so 4q limbs < 2^53. The 8q bias
 * ensures all limbs exceed 2^53, handling 53-bit inputs from chained adds.
 */
static inline void fq_sub(fq_fe h, const fq_fe f, const fq_fe g)
{
    uint64_t c;
    h[0] = f[0] + EIGHT_Q_51[0] - g[0];
    c = h[0] >> 51;
    h[0] &= FQ51_MASK;
    h[1] = f[1] + EIGHT_Q_51[1] - g[1] + c;
    c = h[1] >> 51;
    h[1] &= FQ51_MASK;
    h[2] = f[2] + EIGHT_Q_51[2] - g[2] + c;
    c = h[2] >> 51;
    h[2] &= FQ51_MASK;
    h[3] = f[3] + EIGHT_Q_51[3] - g[3] + c;
    c = h[3] >> 51;
    h[3] &= FQ51_MASK;
    h[4] = f[4] + EIGHT_Q_51[4] - g[4] + c;
    c = h[4] >> 51;
    h[4] &= FQ51_MASK;
    /* Gamma fold: carry * 2^255 ≡ carry * gamma (mod q) */
    for (int j = 0; j < GAMMA_51_LIMBS; j++)
        h[j] += c * GAMMA_51[j];
    /* Re-carry limbs touched by gamma fold */
    for (int j = 0; j < GAMMA_51_LIMBS - 1; j++)
    {
        uint64_t cc = h[j] >> 51;
        h[j + 1] += cc;
        h[j] &= FQ51_MASK;
    }
}

static inline void fq_neg(fq_fe h, const fq_fe f)
{
    fq_fe zero;
    std::memset(zero, 0, sizeof(fq_fe));
    fq_sub(h, zero, f);
}

#else
#include "portable/fq25.h"

static inline void fq_add(fq_fe h, const fq_fe f, const fq_fe g)
{
    for (int i = 0; i < 10; i++)
        h[i] = f[i] + g[i];
}

static inline void fq_sub(fq_fe h, const fq_fe f, const fq_fe g)
{
    int64_t d0 = (int64_t)f[0] - g[0];
    int64_t d1 = (int64_t)f[1] - g[1];
    int64_t d2 = (int64_t)f[2] - g[2];
    int64_t d3 = (int64_t)f[3] - g[3];
    int64_t d4 = (int64_t)f[4] - g[4];
    int64_t d5 = (int64_t)f[5] - g[5];
    int64_t d6 = (int64_t)f[6] - g[6];
    int64_t d7 = (int64_t)f[7] - g[7];
    int64_t d8 = (int64_t)f[8] - g[8];
    int64_t d9 = (int64_t)f[9] - g[9];
    int64_t carry;
    carry = d0 >> 26;
    d1 += carry;
    d0 -= carry << 26;
    carry = d1 >> 25;
    d2 += carry;
    d1 -= carry << 25;
    carry = d2 >> 26;
    d3 += carry;
    d2 -= carry << 26;
    carry = d3 >> 25;
    d4 += carry;
    d3 -= carry << 25;
    carry = d4 >> 26;
    d5 += carry;
    d4 -= carry << 26;
    carry = d5 >> 25;
    d6 += carry;
    d5 -= carry << 25;
    carry = d6 >> 26;
    d7 += carry;
    d6 -= carry << 26;
    carry = d7 >> 25;
    d8 += carry;
    d7 -= carry << 25;
    carry = d8 >> 26;
    d9 += carry;
    d8 -= carry << 26;
    carry = d9 >> 25;
    d9 -= carry << 25;
    /* Fold: carry * 2^255 ≡ carry * gamma (mod q) */
    {
        int64_t *dptrs[] = {&d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7, &d8, &d9};
        for (int j = 0; j < GAMMA_25_LIMBS; j++)
            *dptrs[j] += carry * (int64_t)GAMMA_25[j];
    }
    /* Second carry pass */
    carry = d0 >> 26;
    d1 += carry;
    d0 -= carry << 26;
    carry = d1 >> 25;
    d2 += carry;
    d1 -= carry << 25;
    carry = d2 >> 26;
    d3 += carry;
    d2 -= carry << 26;
    carry = d3 >> 25;
    d4 += carry;
    d3 -= carry << 25;
    carry = d4 >> 26;
    d5 += carry;
    d4 -= carry << 26;
    carry = d5 >> 25;
    d6 += carry;
    d5 -= carry << 25;
    carry = d6 >> 26;
    d7 += carry;
    d6 -= carry << 26;
    carry = d7 >> 25;
    d8 += carry;
    d7 -= carry << 25;
    carry = d8 >> 26;
    d9 += carry;
    d8 -= carry << 26;
    carry = d9 >> 25;
    d9 -= carry << 25;
    {
        int64_t *dptrs[] = {&d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7, &d8, &d9};
        for (int j = 0; j < GAMMA_25_LIMBS; j++)
            *dptrs[j] += carry * (int64_t)GAMMA_25[j];
    }
    carry = d0 >> 26;
    d1 += carry;
    d0 -= carry << 26;
    carry = d1 >> 25;
    d2 += carry;
    d1 -= carry << 25;
    h[0] = (int32_t)d0;
    h[1] = (int32_t)d1;
    h[2] = (int32_t)d2;
    h[3] = (int32_t)d3;
    h[4] = (int32_t)d4;
    h[5] = (int32_t)d5;
    h[6] = (int32_t)d6;
    h[7] = (int32_t)d7;
    h[8] = (int32_t)d8;
    h[9] = (int32_t)d9;
}

static inline void fq_neg(fq_fe h, const fq_fe f)
{
    fq_fe zero;
    for (int i = 0; i < 10; i++)
        zero[i] = 0;
    fq_sub(h, zero, f);
}

#endif // RANSHAW_PLATFORM_64BIT

static inline void fq_copy(fq_fe h, const fq_fe f)
{
    std::memcpy(h, f, sizeof(fq_fe));
}

static inline void fq_0(fq_fe h)
{
    std::memset(h, 0, sizeof(fq_fe));
}

static inline void fq_1(fq_fe h)
{
    h[0] = 1;
    h[1] = 0;
    h[2] = 0;
    h[3] = 0;
#if RANSHAW_PLATFORM_64BIT
    h[4] = 0;
#else
    h[4] = 0;
    h[5] = 0;
    h[6] = 0;
    h[7] = 0;
    h[8] = 0;
    h[9] = 0;
#endif
}

#endif // RANSHAW_FQ_OPS_H
