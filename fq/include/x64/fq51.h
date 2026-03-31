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
 * q = 2^255 - gamma, where gamma = 85737960593035654572250192257530476641
 * gamma is 127 bits, fitting in 3 radix-2^51 limbs.
 *
 * gamma in radix-2^51:
 *   GAMMA_51[0] = 0x12D8D86D83861
 *   GAMMA_51[1] = 0x269135294F229
 *   GAMMA_51[2] = 0x102021F
 *
 * 2*gamma in radix-2^51 (128 bits, 3 limbs):
 *   TWO_GAMMA_51[0] = 0x25B1B0DB070C2
 *   TWO_GAMMA_51[1] = 0x4D226A529E452
 *   TWO_GAMMA_51[2] = 0x204043E
 */
#define GAMMA_51_LIMBS 3
static const uint64_t GAMMA_51[5] = {0x12D8D86D83861ULL, 0x269135294F229ULL, 0x102021FULL, 0, 0};

#define TWO_GAMMA_51_LIMBS 3
static const uint64_t TWO_GAMMA_51[5] = {0x25B1B0DB070C2ULL, 0x4D226A529E452ULL, 0x204043EULL, 0, 0};

/*
 * 2*gamma in radix-2^64 (4 limbs, full 256-bit width).
 * Used by the 4×64 MULX+ADCX+ADOX multiplication path.
 * 2^256 ≡ 2*gamma (mod q), so the fold multiplies by TWO_GAMMA_64.
 */
#define TWO_GAMMA_64_LIMBS 2
static const uint64_t TWO_GAMMA_64[4] = {0x22925B1B0DB070C2ULL, 0x81010FA69135294FULL, 0, 0};

/*
 * q in radix-2^51:
 *   Q_51[0] = 0x6D2727927C79F
 *   Q_51[1] = 0x596ECAD6B0DD6
 *   Q_51[2] = 0x7FFFFFEFDFDE0
 *   Q_51[3] = 0x7FFFFFFFFFFFF
 *   Q_51[4] = 0x7FFFFFFFFFFFF
 */
static const uint64_t Q_51[5] =
    {0x6D2727927C79FULL, 0x596ECAD6B0DD6ULL, 0x7FFFFFEFDFDE0ULL, 0x7FFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFULL};

/*
 * 8*q in radix-2^51, used as bias in fq_sub to prevent underflow.
 * EIGHT_Q_51[i] = 8 * Q_51[i], all fit in 54 bits and all >= 2^53.
 *
 * Fp uses 4p bias (all 4p limbs ≈ 2^53) because p = 2^255 − 19 has all limbs
 * near 2^51. For Fq = 2^255 − gamma (gamma ≈ 2^127), the lower limbs of q are
 * significantly less than 2^51, so 4q limbs are < 2^53. We need 8q to ensure
 * all bias limbs exceed 2^53, safely handling up to 53-bit input limbs
 * (two chained lazy additions before subtraction).
 */
static const uint64_t EIGHT_Q_51[5] =
    {0x369393C93E3CF8ULL, 0x2CB7656B586EB0ULL, 0x3FFFFFF7EFEF00ULL, 0x3FFFFFFFFFFFF8ULL, 0x3FFFFFFFFFFFF8ULL};

#endif // RANSHAW_X64_FQ51_H
