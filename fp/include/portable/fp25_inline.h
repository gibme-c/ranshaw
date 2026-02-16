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
 * @file fp25_inline.h
 * @brief Portable (32-bit, radix-2^25.5) implementation of F_p inline arithmetic helpers.
 */

#ifndef RANSHAW_PORTABLE_FP25_INLINE_H
#define RANSHAW_PORTABLE_FP25_INLINE_H

#include "fp.h"

#if defined(_MSC_VER)
#define RANSHAW_FORCE_INLINE __forceinline
#elif !defined(RANSHAW_FORCE_INLINE)
#define RANSHAW_FORCE_INLINE inline __attribute__((always_inline))
#endif

static RANSHAW_FORCE_INLINE void fp25_mul_inline(fp_fe h, const fp_fe f, const fp_fe g)
{
    int32_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int32_t f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];
    int32_t g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3], g4 = g[4];
    int32_t g5 = g[5], g6 = g[6], g7 = g[7], g8 = g[8], g9 = g[9];
    int64_t g1_19 = 19 * (int64_t)g1, g2_19 = 19 * (int64_t)g2, g3_19 = 19 * (int64_t)g3;
    int64_t g4_19 = 19 * (int64_t)g4, g5_19 = 19 * (int64_t)g5, g6_19 = 19 * (int64_t)g6;
    int64_t g7_19 = 19 * (int64_t)g7, g8_19 = 19 * (int64_t)g8, g9_19 = 19 * (int64_t)g9;
    int64_t f1_2 = 2 * (int64_t)f1, f3_2 = 2 * (int64_t)f3, f5_2 = 2 * (int64_t)f5;
    int64_t f7_2 = 2 * (int64_t)f7, f9_2 = 2 * (int64_t)f9;

    int64_t h0 = (int64_t)f0 * g0 + f1_2 * g9_19 + (int64_t)f2 * g8_19 + f3_2 * g7_19 + (int64_t)f4 * g6_19
                 + f5_2 * g5_19 + (int64_t)f6 * g4_19 + f7_2 * g3_19 + (int64_t)f8 * g2_19 + f9_2 * g1_19;
    int64_t h1 = (int64_t)f0 * g1 + (int64_t)f1 * g0 + (int64_t)f2 * g9_19 + (int64_t)f3 * g8_19 + (int64_t)f4 * g7_19
                 + (int64_t)f5 * g6_19 + (int64_t)f6 * g5_19 + (int64_t)f7 * g4_19 + (int64_t)f8 * g3_19
                 + (int64_t)f9 * g2_19;
    int64_t h2 = (int64_t)f0 * g2 + f1_2 * (int64_t)g1 + (int64_t)f2 * g0 + f3_2 * g9_19 + (int64_t)f4 * g8_19
                 + f5_2 * g7_19 + (int64_t)f6 * g6_19 + f7_2 * g5_19 + (int64_t)f8 * g4_19 + f9_2 * g3_19;
    int64_t h3 = (int64_t)f0 * g3 + (int64_t)f1 * g2 + (int64_t)f2 * g1 + (int64_t)f3 * g0 + (int64_t)f4 * g9_19
                 + (int64_t)f5 * g8_19 + (int64_t)f6 * g7_19 + (int64_t)f7 * g6_19 + (int64_t)f8 * g5_19
                 + (int64_t)f9 * g4_19;
    int64_t h4 = (int64_t)f0 * g4 + f1_2 * (int64_t)g3 + (int64_t)f2 * g2 + f3_2 * (int64_t)g1 + (int64_t)f4 * g0
                 + f5_2 * g9_19 + (int64_t)f6 * g8_19 + f7_2 * g7_19 + (int64_t)f8 * g6_19 + f9_2 * g5_19;
    int64_t h5 = (int64_t)f0 * g5 + (int64_t)f1 * g4 + (int64_t)f2 * g3 + (int64_t)f3 * g2 + (int64_t)f4 * g1
                 + (int64_t)f5 * g0 + (int64_t)f6 * g9_19 + (int64_t)f7 * g8_19 + (int64_t)f8 * g7_19
                 + (int64_t)f9 * g6_19;
    int64_t h6 = (int64_t)f0 * g6 + f1_2 * (int64_t)g5 + (int64_t)f2 * g4 + f3_2 * (int64_t)g3 + (int64_t)f4 * g2
                 + f5_2 * (int64_t)g1 + (int64_t)f6 * g0 + f7_2 * g9_19 + (int64_t)f8 * g8_19 + f9_2 * g7_19;
    int64_t h7 = (int64_t)f0 * g7 + (int64_t)f1 * g6 + (int64_t)f2 * g5 + (int64_t)f3 * g4 + (int64_t)f4 * g3
                 + (int64_t)f5 * g2 + (int64_t)f6 * g1 + (int64_t)f7 * g0 + (int64_t)f8 * g9_19 + (int64_t)f9 * g8_19;
    int64_t h8 = (int64_t)f0 * g8 + f1_2 * (int64_t)g7 + (int64_t)f2 * g6 + f3_2 * (int64_t)g5 + (int64_t)f4 * g4
                 + f5_2 * (int64_t)g3 + (int64_t)f6 * g2 + f7_2 * (int64_t)g1 + (int64_t)f8 * g0 + f9_2 * g9_19;
    int64_t h9 = (int64_t)f0 * g9 + (int64_t)f1 * g8 + (int64_t)f2 * g7 + (int64_t)f3 * g6 + (int64_t)f4 * g5
                 + (int64_t)f5 * g4 + (int64_t)f6 * g3 + (int64_t)f7 * g2 + (int64_t)f8 * g1 + (int64_t)f9 * g0;

    int64_t carry0, carry1, carry2, carry3, carry4, carry5, carry6, carry7, carry8, carry9;

    carry0 = (h0 + (int64_t)(1 << 25)) >> 26;
    h1 += carry0;
    h0 -= carry0 << 26;
    carry4 = (h4 + (int64_t)(1 << 25)) >> 26;
    h5 += carry4;
    h4 -= carry4 << 26;
    carry1 = (h1 + (int64_t)(1 << 24)) >> 25;
    h2 += carry1;
    h1 -= carry1 << 25;
    carry5 = (h5 + (int64_t)(1 << 24)) >> 25;
    h6 += carry5;
    h5 -= carry5 << 25;
    carry2 = (h2 + (int64_t)(1 << 25)) >> 26;
    h3 += carry2;
    h2 -= carry2 << 26;
    carry6 = (h6 + (int64_t)(1 << 25)) >> 26;
    h7 += carry6;
    h6 -= carry6 << 26;
    carry3 = (h3 + (int64_t)(1 << 24)) >> 25;
    h4 += carry3;
    h3 -= carry3 << 25;
    carry7 = (h7 + (int64_t)(1 << 24)) >> 25;
    h8 += carry7;
    h7 -= carry7 << 25;
    carry4 = (h4 + (int64_t)(1 << 25)) >> 26;
    h5 += carry4;
    h4 -= carry4 << 26;
    carry8 = (h8 + (int64_t)(1 << 25)) >> 26;
    h9 += carry8;
    h8 -= carry8 << 26;
    carry9 = (h9 + (int64_t)(1 << 24)) >> 25;
    h0 += carry9 * 19;
    h9 -= carry9 << 25;
    carry0 = (h0 + (int64_t)(1 << 25)) >> 26;
    h1 += carry0;
    h0 -= carry0 << 26;

    h[0] = (int32_t)h0;
    h[1] = (int32_t)h1;
    h[2] = (int32_t)h2;
    h[3] = (int32_t)h3;
    h[4] = (int32_t)h4;
    h[5] = (int32_t)h5;
    h[6] = (int32_t)h6;
    h[7] = (int32_t)h7;
    h[8] = (int32_t)h8;
    h[9] = (int32_t)h9;
}

static RANSHAW_FORCE_INLINE void fp25_sq_inline(fp_fe h, const fp_fe f)
{
    int32_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int32_t f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];
    int64_t f0_2 = 2 * (int64_t)f0, f1_2 = 2 * (int64_t)f1, f2_2 = 2 * (int64_t)f2;
    int64_t f3_2 = 2 * (int64_t)f3, f4_2 = 2 * (int64_t)f4;
    int64_t f5_2 = 2 * (int64_t)f5, f6_2 = 2 * (int64_t)f6, f7_2 = 2 * (int64_t)f7;
    int64_t f5_38 = 38 * (int64_t)f5, f6_19 = 19 * (int64_t)f6;
    int64_t f7_38 = 38 * (int64_t)f7, f8_19 = 19 * (int64_t)f8, f9_38 = 38 * (int64_t)f9;

    int64_t h0 = (int64_t)f0 * f0 + f1_2 * f9_38 + f2_2 * f8_19 + f3_2 * f7_38 + f4_2 * f6_19 + (int64_t)f5 * f5_38;
    int64_t h1 = f0_2 * f1 + (int64_t)f2 * f9_38 + f3_2 * f8_19 + (int64_t)f4 * f7_38 + f5_2 * f6_19;
    int64_t h2 = f0_2 * f2 + f1_2 * f1 + f3_2 * f9_38 + f4_2 * f8_19 + f5_2 * f7_38 + (int64_t)f6 * f6_19;
    int64_t h3 = f0_2 * f3 + f1_2 * f2 + (int64_t)f4 * f9_38 + f5_2 * f8_19 + (int64_t)f6 * f7_38;
    int64_t h4 = f0_2 * f4 + f1_2 * f3_2 + (int64_t)f2 * f2 + f5_2 * f9_38 + f6_2 * f8_19 + (int64_t)f7 * f7_38;
    int64_t h5 = f0_2 * f5 + f1_2 * f4 + f2_2 * f3 + (int64_t)f6 * f9_38 + f7_2 * f8_19;
    int64_t h6 = f0_2 * f6 + f1_2 * f5_2 + f2_2 * f4 + f3_2 * f3 + f7_2 * f9_38 + (int64_t)f8 * f8_19;
    int64_t h7 = f0_2 * f7 + f1_2 * f6 + f2_2 * f5 + f3_2 * f4 + (int64_t)f8 * f9_38;
    int64_t h8 = f0_2 * f8 + f1_2 * f7_2 + f2_2 * f6 + f3_2 * f5_2 + (int64_t)f4 * f4 + (int64_t)f9 * f9_38;
    int64_t h9 = f0_2 * f9 + f1_2 * f8 + f2_2 * f7 + f3_2 * f6 + f4_2 * f5;

    int64_t carry0, carry1, carry2, carry3, carry4, carry5, carry6, carry7, carry8, carry9;

    carry0 = (h0 + (int64_t)(1 << 25)) >> 26;
    h1 += carry0;
    h0 -= carry0 << 26;
    carry4 = (h4 + (int64_t)(1 << 25)) >> 26;
    h5 += carry4;
    h4 -= carry4 << 26;
    carry1 = (h1 + (int64_t)(1 << 24)) >> 25;
    h2 += carry1;
    h1 -= carry1 << 25;
    carry5 = (h5 + (int64_t)(1 << 24)) >> 25;
    h6 += carry5;
    h5 -= carry5 << 25;
    carry2 = (h2 + (int64_t)(1 << 25)) >> 26;
    h3 += carry2;
    h2 -= carry2 << 26;
    carry6 = (h6 + (int64_t)(1 << 25)) >> 26;
    h7 += carry6;
    h6 -= carry6 << 26;
    carry3 = (h3 + (int64_t)(1 << 24)) >> 25;
    h4 += carry3;
    h3 -= carry3 << 25;
    carry7 = (h7 + (int64_t)(1 << 24)) >> 25;
    h8 += carry7;
    h7 -= carry7 << 25;
    carry4 = (h4 + (int64_t)(1 << 25)) >> 26;
    h5 += carry4;
    h4 -= carry4 << 26;
    carry8 = (h8 + (int64_t)(1 << 25)) >> 26;
    h9 += carry8;
    h8 -= carry8 << 26;
    carry9 = (h9 + (int64_t)(1 << 24)) >> 25;
    h0 += carry9 * 19;
    h9 -= carry9 << 25;
    carry0 = (h0 + (int64_t)(1 << 25)) >> 26;
    h1 += carry0;
    h0 -= carry0 << 26;

    h[0] = (int32_t)h0;
    h[1] = (int32_t)h1;
    h[2] = (int32_t)h2;
    h[3] = (int32_t)h3;
    h[4] = (int32_t)h4;
    h[5] = (int32_t)h5;
    h[6] = (int32_t)h6;
    h[7] = (int32_t)h7;
    h[8] = (int32_t)h8;
    h[9] = (int32_t)h9;
}

#endif // RANSHAW_PORTABLE_FP25_INLINE_H
