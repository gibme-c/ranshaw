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
 * @file fq51.h
 * @brief x64 (radix-2^51) implementation of F_q core type and operations with Crandall reduction.
 */

#ifndef RANSHAW_X64_FQ51_H
#define RANSHAW_X64_FQ51_H

#include <cstdint>

static const uint64_t FQ51_MASK = (1ULL << 51) - 1;

/*
 * q = 2^255 - gamma, where gamma = 239666463199878229209741112730228557729
 * gamma is 128 bits, fitting in 3 radix-2^51 limbs.
 *
 * gamma in radix-2^51:
 *   GAMMA_51[0] = 0x7B9BA138F07A1
 *   GAMMA_51[1] = 0x638D19E0B11D2
 *   GAMMA_51[2] = 0x2D13853
 *
 * 2*gamma in radix-2^51 (129 bits, 3 limbs):
 *   TWO_GAMMA_51[0] = 0x77374271E0F42
 *   TWO_GAMMA_51[1] = 0x471A33C1623A5
 *   TWO_GAMMA_51[2] = 0x5A270A7
 */
#define GAMMA_51_LIMBS 3
static const uint64_t GAMMA_51[5] = {0x7B9BA138F07A1ULL, 0x638D19E0B11D2ULL, 0x2D13853ULL, 0, 0};

#define TWO_GAMMA_51_LIMBS 3
static const uint64_t TWO_GAMMA_51[5] = {0x77374271E0F42ULL, 0x471A33C1623A5ULL, 0x5A270A7ULL, 0, 0};

/*
 * 2*gamma in radix-2^64 (4 limbs, full 256-bit width).
 * Used by the 4×64 MULX+ADCX+ADOX multiplication path.
 * 2^256 ≡ 2*gamma (mod q), so the fold multiplies by TWO_GAMMA_64.
 */
#define TWO_GAMMA_64_LIMBS 3
static const uint64_t TWO_GAMMA_64[4] = {0x1D2F7374271E0F42ULL, 0x689C29E38D19E0B1ULL, 0x1ULL, 0};

/*
 * q in radix-2^51:
 *   Q_51[0] = 0x4645EC70F85F
 *   Q_51[1] = 0x1C72E61F4EE2D
 *   Q_51[2] = 0x7FFFFFD2EC7AC
 *   Q_51[3] = 0x7FFFFFFFFFFFF
 *   Q_51[4] = 0x7FFFFFFFFFFFF
 */
static const uint64_t Q_51[5] =
    {0x4645EC70F85FULL, 0x1C72E61F4EE2DULL, 0x7FFFFFD2EC7ACULL, 0x7FFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFULL};

/*
 * 128*q bias in radix-2^51, used in fq_sub to prevent underflow.
 * BIAS_Q_51[i] = 128 * Q_51[i], all fit in 58 bits and all >= 2^53.
 *
 * Fp uses 4p bias (all 4p limbs ≈ 2^53) because p = 2^255 − 19 has all limbs
 * near 2^51. For Fq = 2^255 − gamma (gamma ≈ 2^128), the lower limbs of q are
 * significantly less than 2^51, so even 8q limbs can fall below 2^53. We use
 * 128q to ensure all bias limbs comfortably exceed 2^53, handling up to 53-bit
 * input limbs (two chained lazy additions before subtraction).
 */
static const uint64_t BIAS_Q_51[5] =
    {0x2322F6387C2F80ULL, 0xE39730FA771680ULL, 0x3FFFFFE9763D600ULL, 0x3FFFFFFFFFFFF80ULL, 0x3FFFFFFFFFFFF80ULL};

#endif // RANSHAW_X64_FQ51_H
