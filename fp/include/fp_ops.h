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
 * @file fp_ops.h
 * @brief Basic F_p arithmetic: add, sub, neg, copy, zero, one, conditional operations.
 */

#ifndef RANSHAW_FP_OPS_H
#define RANSHAW_FP_OPS_H

#include "fp.h"

#include <cstring>

#if RANSHAW_PLATFORM_64BIT
#include "x64/fp51.h"

static inline void fp_add(fp_fe h, const fp_fe f, const fp_fe g)
{
    h[0] = f[0] + g[0];
    h[1] = f[1] + g[1];
    h[2] = f[2] + g[2];
    h[3] = f[3] + g[3];
    h[4] = f[4] + g[4];
}

static inline void fp_sub(fp_fe h, const fp_fe f, const fp_fe g)
{
    uint64_t c;
    h[0] = f[0] + 0x1FFFFFFFFFFFB4ULL - g[0];
    c = h[0] >> 51;
    h[0] &= FP51_MASK;
    h[1] = f[1] + 0x1FFFFFFFFFFFFCULL - g[1] + c;
    c = h[1] >> 51;
    h[1] &= FP51_MASK;
    h[2] = f[2] + 0x1FFFFFFFFFFFFCULL - g[2] + c;
    c = h[2] >> 51;
    h[2] &= FP51_MASK;
    h[3] = f[3] + 0x1FFFFFFFFFFFFCULL - g[3] + c;
    c = h[3] >> 51;
    h[3] &= FP51_MASK;
    h[4] = f[4] + 0x1FFFFFFFFFFFFCULL - g[4] + c;
    c = h[4] >> 51;
    h[4] &= FP51_MASK;
    h[0] += c * 19;
}

static inline void fp_neg(fp_fe h, const fp_fe f)
{
    uint64_t c;
    h[0] = 0xFFFFFFFFFFFDAULL - f[0];
    c = h[0] >> 51;
    h[0] &= FP51_MASK;
    h[1] = 0xFFFFFFFFFFFFEULL - f[1] + c;
    c = h[1] >> 51;
    h[1] &= FP51_MASK;
    h[2] = 0xFFFFFFFFFFFFEULL - f[2] + c;
    c = h[2] >> 51;
    h[2] &= FP51_MASK;
    h[3] = 0xFFFFFFFFFFFFEULL - f[3] + c;
    c = h[3] >> 51;
    h[3] &= FP51_MASK;
    h[4] = 0xFFFFFFFFFFFFEULL - f[4] + c;
    c = h[4] >> 51;
    h[4] &= FP51_MASK;
    h[0] += c * 19;
}

#else

static inline void fp_add(fp_fe h, const fp_fe f, const fp_fe g)
{
    for (int i = 0; i < 10; i++)
        h[i] = f[i] + g[i];
}

static inline void fp_sub(fp_fe h, const fp_fe f, const fp_fe g)
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
    d0 += carry * 19;
    d9 -= carry << 25;
    carry = d0 >> 26;
    d1 += carry;
    d0 -= carry << 26;
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

static inline void fp_neg(fp_fe h, const fp_fe f)
{
    fp_fe zero;
    for (int i = 0; i < 10; i++)
        zero[i] = 0;
    fp_sub(h, zero, f);
}

#endif // RANSHAW_PLATFORM_64BIT

static inline void fp_copy(fp_fe h, const fp_fe f)
{
    std::memcpy(h, f, sizeof(fp_fe));
}

static inline void fp_0(fp_fe h)
{
    std::memset(h, 0, sizeof(fp_fe));
}

static inline void fp_1(fp_fe h)
{
    h[0] = 1;
    h[1] = 0;
    h[2] = 0;
    h[3] = 0;
    h[4] = 0;
#if !RANSHAW_PLATFORM_64BIT
    h[5] = 0;
    h[6] = 0;
    h[7] = 0;
    h[8] = 0;
    h[9] = 0;
#endif
}

#endif // RANSHAW_FP_OPS_H
