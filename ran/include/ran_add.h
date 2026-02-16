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
 * @file ran_add.h
 * @brief Ran Jacobian point addition with edge-case handling (identity, doubling, inverse).
 */

#ifndef RANSHAW_RAN_ADD_H
#define RANSHAW_RAN_ADD_H

#include "ran_dbl.h"
#include "ran_ops.h"

#if RANSHAW_PLATFORM_64BIT
void ran_add_x64(ran_jacobian *r, const ran_jacobian *p, const ran_jacobian *q);
#else
void ran_add_portable(ran_jacobian *r, const ran_jacobian *p, const ran_jacobian *q);
#endif

static inline void ran_add(ran_jacobian *r, const ran_jacobian *p, const ran_jacobian *q)
{
    /* Identity inputs */
    if (ran_is_identity(p))
    {
        ran_copy(r, q);
        return;
    }
    if (ran_is_identity(q))
    {
        ran_copy(r, p);
        return;
    }

    /* Projective x-coordinate comparison: U1 = X1*Z2^2, U2 = X2*Z1^2 */
    fp_fe z1z1, z2z2, u1, u2, diff;
    fp_sq(z1z1, p->Z);
    fp_sq(z2z2, q->Z);
    fp_mul(u1, p->X, z2z2);
    fp_mul(u2, q->X, z1z1);
    fp_sub(diff, u1, u2);

    if (!fp_isnonzero(diff))
    {
        /* Same x: check y parity via S1 = Y1*Z2^3, S2 = Y2*Z1^3 */
        fp_fe s1, s2, t;
        fp_mul(t, q->Z, z2z2);
        fp_mul(s1, p->Y, t);
        fp_mul(t, p->Z, z1z1);
        fp_mul(s2, q->Y, t);
        fp_sub(diff, s1, s2);
        if (!fp_isnonzero(diff))
            ran_dbl(r, p);
        else
            ran_identity(r);
        return;
    }

#if RANSHAW_PLATFORM_64BIT
    ran_add_x64(r, p, q);
#else
    ran_add_portable(r, p, q);
#endif
}

#endif // RANSHAW_RAN_ADD_H
