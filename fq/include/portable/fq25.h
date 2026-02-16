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
 * @file fq25.h
 * @brief Portable (32-bit, radix-2^25.5) implementation of F_q core type and operations with Crandall reduction.
 */

#ifndef RANSHAW_PORTABLE_FQ25_H
#define RANSHAW_PORTABLE_FQ25_H

#include <cstdint>

/*
 * gamma = 85737960593035654572250192257530476641 in unsigned 10-limb
 * radix-2^25.5 form. Only the first 5 limbs are nonzero (gamma is 127 bits).
 *
 * Limb widths alternate 26/25/26/25/26 bits:
 *   limb0 = gamma & 0x3FFFFFF        (bits 0-25)
 *   limb1 = (gamma >> 26) & 0x1FFFFFF (bits 26-50)
 *   limb2 = (gamma >> 51) & 0x3FFFFFF (bits 51-76)
 *   limb3 = (gamma >> 77) & 0x1FFFFFF (bits 77-101)
 *   limb4 = (gamma >> 102) & 0x3FFFFFF (bits 102-126)
 *
 * Derived from GAMMA_51 (radix-2^51 representation):
 *   GAMMA_51[0] = 0x12D8D86D83861 -> limb0, limb1
 *   GAMMA_51[1] = 0x269135294F229 -> limb2, limb3
 *   GAMMA_51[2] = 0x102021F        -> limb4
 */
static const int32_t GAMMA_25[5] = {47724641, 4940641, 43315753, 10110164, 16908831};

/*
 * q = 2^255 - gamma in 10-limb radix-2^25.5 form (unsigned, canonical).
 *
 * q bytes (LE): 9f c7 27 79 72 d2 b6 6e 58 6b 65 b7 2c 78 7f bf
 *               ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff 7f
 *
 * Derived by splitting q into alternating 26/25-bit limbs:
 *   Q_25[i] = (q >> bit_position_i) & mask_i
 */
static const int32_t Q_25[10] =
    {19384223, 28613790, 23793110, 23444267, 50200032, 33554431, 67108863, 33554431, 67108863, 33554431};

/* Masks for alternating 26/25-bit limbs */
static const int32_t FQ25_MASK26 = (1 << 26) - 1;
static const int32_t FQ25_MASK25 = (1 << 25) - 1;

#endif // RANSHAW_PORTABLE_FQ25_H
