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

#include "portable/ran_dbl.h"

#include "fp_ops.h"
#include "portable/fp25_chain.h"

/*
 * Jacobian point doubling with a = -3 optimization.
 * EFD: dbl-2001-b
 * Cost: 3M + 5S
 *
 * delta = Z1^2
 * gamma = Y1^2
 * beta = X1 * gamma
 * alpha = 3 * (X1 - delta) * (X1 + delta)    [a = -3 optimization]
 * X3 = alpha^2 - 8*beta
 * Z3 = (Y1 + Z1)^2 - gamma - delta
 * Y3 = alpha * (4*beta - X3) - 8*gamma^2
 */
void ran_dbl_portable(ran_jacobian *r, const ran_jacobian *p)
{
    fp_fe delta, gamma, beta, alpha;
    fp_fe t0, t1, t2;

    /* delta = Z1^2 */
    fp25_chain_sq(delta, p->Z);

    /* gamma = Y1^2 */
    fp25_chain_sq(gamma, p->Y);

    /* beta = X1 * gamma */
    fp25_chain_mul(beta, p->X, gamma);

    /* alpha = 3 * (X1 - delta) * (X1 + delta) */
    fp_sub(t0, p->X, delta);
    fp_add(t1, p->X, delta);
    fp25_chain_mul(alpha, t0, t1);
    /* alpha = 3 * alpha */
    fp_add(t0, alpha, alpha);
    fp_add(alpha, t0, alpha);

    /* X3 = alpha^2 - 8*beta */
    fp25_chain_sq(r->X, alpha);
    fp_add(t0, beta, beta); /* 2*beta */
    fp_add(t0, t0, t0); /* 4*beta */
    fp_sub(r->X, r->X, t0); /* alpha^2 - 4*beta */
    fp_sub(r->X, r->X, t0); /* alpha^2 - 8*beta */

    /* Z3 = (Y1 + Z1)^2 - gamma - delta */
    fp_add(t1, p->Y, p->Z);
    fp25_chain_sq(t2, t1);
    fp_sub(t2, t2, gamma);
    fp_sub(r->Z, t2, delta);

    /* Y3 = alpha * (4*beta - X3) - 8*gamma^2 */
    fp_sub(t1, t0, r->X); /* 4*beta - X3 */
    fp25_chain_mul(t2, alpha, t1);
    fp25_chain_sq(t0, gamma); /* gamma^2 */
    fp_add(t0, t0, t0); /* 2*gamma^2 */
    fp_add(t0, t0, t0); /* 4*gamma^2 */
    fp_sub(r->Y, t2, t0); /* ... - 4*gamma^2 */
    fp_sub(r->Y, r->Y, t0); /* ... - 8*gamma^2 */
}
