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
 * @file fp51.h
 * @brief x64 (radix-2^51) implementation of F_p core type and operations.
 */

#ifndef RANSHAW_X64_FP51_H
#define RANSHAW_X64_FP51_H

#include <cstdint>

static const uint64_t FP51_MASK = (1ULL << 51) - 1;

/*
 * Carry-propagate a field element so every limb is ≤ 51 bits.
 * Needed before feeding the result of consecutive fp_add calls
 * into mul/sq chains that assume bounded limbs.
 */
static inline void fp51_carry(fp_fe h, const fp_fe f)
{
    uint64_t c;
    uint64_t h0 = f[0], h1 = f[1], h2 = f[2], h3 = f[3], h4 = f[4];
    c = h0 >> 51;
    h1 += c;
    h0 &= FP51_MASK;
    c = h1 >> 51;
    h2 += c;
    h1 &= FP51_MASK;
    c = h2 >> 51;
    h3 += c;
    h2 &= FP51_MASK;
    c = h3 >> 51;
    h4 += c;
    h3 &= FP51_MASK;
    c = h4 >> 51;
    h0 += c * 19;
    h4 &= FP51_MASK;
    c = h0 >> 51;
    h1 += c;
    h0 &= FP51_MASK;
    h[0] = h0;
    h[1] = h1;
    h[2] = h2;
    h[3] = h3;
    h[4] = h4;
}

#endif // RANSHAW_X64_FP51_H
