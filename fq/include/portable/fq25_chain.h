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
 * @file fq25_chain.h
 * @brief Portable (32-bit, radix-2^25.5) implementation of F_q addition chains with Crandall reduction.
 */

#ifndef RANSHAW_PORTABLE_FQ25_CHAIN_H
#define RANSHAW_PORTABLE_FQ25_CHAIN_H

#include "portable/fq25_inline.h"

#define fq25_chain_mul fq25_mul_inline
#define fq25_chain_sq fq25_sq_inline

static RANSHAW_FORCE_INLINE void fq25_sq2_inline(fq_fe h, const fq_fe f)
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

    /* Double all accumulators (2 * f^2) */
    for (int i = 0; i < 19; i++)
        t[i] += t[i];

    fq25_reduce_full(h, t);
}

#define fq25_chain_sq2 fq25_sq2_inline

static RANSHAW_FORCE_INLINE void fq25_sqn_inline(fq_fe h, const fq_fe f, int n)
{
    fq25_sq_inline(h, f);
    for (int i = 1; i < n; i++)
        fq25_sq_inline(h, h);
}

#define fq25_chain_sqn fq25_sqn_inline

#endif // RANSHAW_PORTABLE_FQ25_CHAIN_H
