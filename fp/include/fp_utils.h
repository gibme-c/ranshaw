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
 * @file fp_utils.h
 * @brief Utility functions for F_p: is_zero, is_negative (parity check), equality test.
 */

#ifndef RANSHAW_FP_UTILS_H
#define RANSHAW_FP_UTILS_H

#include "ranshaw_ct_barrier.h"
#include "fp.h"
#include "fp_tobytes.h"

/*
 * Returns 1 if h is nonzero (in canonical form), 0 if zero.
 * Branchless: uses ct_barrier + bit-shift to avoid conditional branch on d.
 */
static inline int fp_isnonzero(const fp_fe h)
{
    unsigned char s[32];
    fp_tobytes(s, h);
    unsigned char d = 0;
    for (int i = 0; i < 32; i++)
        d |= s[i];
    uint64_t w = ranshaw_ct_barrier_u64((uint64_t)d);
    return (int)((w | (0 - w)) >> 63);
}

/*
 * Returns the "sign" of h: the least significant bit of the canonical
 * representation. 0 = even (non-negative), 1 = odd (negative).
 */
static inline int fp_isnegative(const fp_fe h)
{
    unsigned char s[32];
    fp_tobytes(s, h);
    return s[0] & 1;
}

#endif // RANSHAW_FP_UTILS_H
