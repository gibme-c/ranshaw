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
 * @file shaw_validate.h
 * @brief On-curve validation for Shaw: verify y^2 = x^3 - 3x + b.
 */

#ifndef RANSHAW_SHAW_VALIDATE_H
#define RANSHAW_SHAW_VALIDATE_H

#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "fq_tobytes.h"
#include "shaw.h"
#include "shaw_constants.h"

/*
 * Check if an affine point is on the Shaw curve: y^2 = x^3 - 3x + b (mod q).
 * Variable-time (validation-only, not secret-dependent).
 * Returns 1 if on curve, 0 if not.
 */
static inline int shaw_is_on_curve(const shaw_affine *p)
{
    fq_fe x2, x3, rhs, lhs, diff;

    /* lhs = y^2 */
    fq_sq(lhs, p->y);

    /* rhs = x^3 - 3x + b */
    fq_sq(x2, p->x);
    fq_mul(x3, x2, p->x);

    fq_fe three_x;
    fq_add(three_x, p->x, p->x);
    fq_add(three_x, three_x, p->x);

    fq_sub(rhs, x3, three_x);
    fq_add(rhs, rhs, SHAW_B);

    /* Check lhs == rhs */
    fq_sub(diff, lhs, rhs);
    unsigned char bytes[32];
    fq_tobytes(bytes, diff);

    unsigned char d = 0;
    for (int i = 0; i < 32; i++)
        d |= bytes[i];

    return d == 0;
}

#endif // RANSHAW_SHAW_VALIDATE_H
