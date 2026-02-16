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
 * @file shaw_ops.h
 * @brief Core Shaw point operations: identity, copy, negate, is_identity, affine conversion.
 */

#ifndef RANSHAW_SHAW_OPS_H
#define RANSHAW_SHAW_OPS_H

#include "fq_cmov.h"
#include "fq_invert.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "fq_utils.h"
#include "ranshaw_secure_erase.h"
#include "shaw.h"

/* Set r to the identity (point at infinity): (1:1:0) */
static inline void shaw_identity(shaw_jacobian *r)
{
    fq_1(r->X);
    fq_1(r->Y);
    fq_0(r->Z);
}

/* Copy p to r */
static inline void shaw_copy(shaw_jacobian *r, const shaw_jacobian *p)
{
    fq_copy(r->X, p->X);
    fq_copy(r->Y, p->Y);
    fq_copy(r->Z, p->Z);
}

/* Check if p is the identity (Z == 0) */
static inline int shaw_is_identity(const shaw_jacobian *p)
{
    return !fq_isnonzero(p->Z);
}

/* Negate: (X:Y:Z) -> (X:-Y:Z) */
static inline void shaw_neg(shaw_jacobian *r, const shaw_jacobian *p)
{
    fq_copy(r->X, p->X);
    fq_neg(r->Y, p->Y);
    fq_copy(r->Z, p->Z);
}

/* Constant-time conditional move: r = b ? p : r */
static inline void shaw_cmov(shaw_jacobian *r, const shaw_jacobian *p, unsigned int b)
{
    fq_cmov(r->X, p->X, b);
    fq_cmov(r->Y, p->Y, b);
    fq_cmov(r->Z, p->Z, b);
}

/* Constant-time conditional move for affine points */
static inline void shaw_affine_cmov(shaw_affine *r, const shaw_affine *p, unsigned int b)
{
    fq_cmov(r->x, p->x, b);
    fq_cmov(r->y, p->y, b);
}

/* Constant-time conditional negate: if b, negate Y in place */
static inline void shaw_cneg(shaw_jacobian *r, unsigned int b)
{
    fq_fe neg_y;
    fq_neg(neg_y, r->Y);
    fq_cmov(r->Y, neg_y, b);
    ranshaw_secure_erase(neg_y, sizeof(neg_y));
}

/* Constant-time conditional negate for affine: if b, negate y in place */
static inline void shaw_affine_cneg(shaw_affine *r, unsigned int b)
{
    fq_fe neg_y;
    fq_neg(neg_y, r->y);
    fq_cmov(r->y, neg_y, b);
    ranshaw_secure_erase(neg_y, sizeof(neg_y));
}

/* Convert Jacobian to affine: x = X/Z^2, y = Y/Z^3 */
static inline void shaw_to_affine(shaw_affine *r, const shaw_jacobian *p)
{
    fq_fe z_inv, z_inv2, z_inv3;
    fq_invert(z_inv, p->Z);
    fq_sq(z_inv2, z_inv);
    fq_mul(z_inv3, z_inv2, z_inv);
    fq_mul(r->x, p->X, z_inv2);
    fq_mul(r->y, p->Y, z_inv3);
}

/* Convert affine to Jacobian: (x, y) -> (x:y:1) */
static inline void shaw_from_affine(shaw_jacobian *r, const shaw_affine *p)
{
    fq_copy(r->X, p->x);
    fq_copy(r->Y, p->y);
    fq_1(r->Z);
}

#endif // RANSHAW_SHAW_OPS_H
