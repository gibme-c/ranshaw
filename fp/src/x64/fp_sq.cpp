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

#include "x64/fp_sq.h"

#include "x64/fp51.h"
#include "x64/mul128.h"

void fp_sq_x64(fp_fe h, const fp_fe f)
{
    uint64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];

    uint64_t f0_2 = 2 * f0;
    uint64_t f1_2 = 2 * f1;
    uint64_t f3_2 = 2 * f3;

    uint64_t f1_38 = 38 * f1;
    uint64_t f2_19 = 19 * f2;
    uint64_t f2_38 = 38 * f2;
    uint64_t f3_19 = 19 * f3;
    uint64_t f3_38 = 38 * f3;
    uint64_t f4_19 = 19 * f4;

#if RANSHAW_HAVE_INT128
    ranshaw_uint128 h0 = mul64(f0, f0) + mul64(f1_38, f4) + mul64(f2_19, f3_2);
    ranshaw_uint128 h1 = mul64(f0_2, f1) + mul64(f2_38, f4) + mul64(f3_19, f3);
    ranshaw_uint128 h2 = mul64(f0_2, f2) + mul64(f1, f1) + mul64(f3_38, f4);
    ranshaw_uint128 h3 = mul64(f0_2, f3) + mul64(f1_2, f2) + mul64(f4_19, f4);
    ranshaw_uint128 h4 = mul64(f0_2, f4) + mul64(f1_2, f3) + mul64(f2, f2);

    uint64_t carry;
    carry = (uint64_t)(h0 >> 51);
    h1 += carry;
    h0 &= FP51_MASK;
    carry = (uint64_t)(h1 >> 51);
    h2 += carry;
    h1 &= FP51_MASK;
    carry = (uint64_t)(h2 >> 51);
    h3 += carry;
    h2 &= FP51_MASK;
    carry = (uint64_t)(h3 >> 51);
    h4 += carry;
    h3 &= FP51_MASK;
    carry = (uint64_t)(h4 >> 51);
    h0 += carry * 19;
    h4 &= FP51_MASK;
    carry = (uint64_t)(h0 >> 51);
    h1 += carry;
    h0 &= FP51_MASK;

    h[0] = (uint64_t)h0;
    h[1] = (uint64_t)h1;
    h[2] = (uint64_t)h2;
    h[3] = (uint64_t)h3;
    h[4] = (uint64_t)h4;

#elif RANSHAW_HAVE_UMUL128
    ranshaw_uint128_emu h0 = mul64(f0, f0);
    h0 += mul64(f1_38, f4);
    h0 += mul64(f2_19, f3_2);

    ranshaw_uint128_emu h1 = mul64(f0_2, f1);
    h1 += mul64(f2_38, f4);
    h1 += mul64(f3_19, f3);

    ranshaw_uint128_emu h2 = mul64(f0_2, f2);
    h2 += mul64(f1, f1);
    h2 += mul64(f3_38, f4);

    ranshaw_uint128_emu h3 = mul64(f0_2, f3);
    h3 += mul64(f1_2, f2);
    h3 += mul64(f4_19, f4);

    ranshaw_uint128_emu h4 = mul64(f0_2, f4);
    h4 += mul64(f1_2, f3);
    h4 += mul64(f2, f2);

    uint64_t carry;
    carry = shr128(h0, 51);
    h1 += carry;
    uint64_t r0 = lo128(h0) & FP51_MASK;
    carry = shr128(h1, 51);
    h2 += carry;
    uint64_t r1 = lo128(h1) & FP51_MASK;
    carry = shr128(h2, 51);
    h3 += carry;
    uint64_t r2 = lo128(h2) & FP51_MASK;
    carry = shr128(h3, 51);
    h4 += carry;
    uint64_t r3 = lo128(h3) & FP51_MASK;
    carry = shr128(h4, 51);
    r0 += carry * 19;
    uint64_t r4 = lo128(h4) & FP51_MASK;
    carry = r0 >> 51;
    r1 += carry;
    r0 &= FP51_MASK;

    h[0] = r0;
    h[1] = r1;
    h[2] = r2;
    h[3] = r3;
    h[4] = r4;
#endif
}
