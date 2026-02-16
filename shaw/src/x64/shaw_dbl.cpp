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

#include "x64/shaw_dbl.h"

#include "fq_ops.h"
#include "x64/fq51_chain.h"

/*
 * Jacobian point doubling with a = -3 optimization.
 * Same formula as ran_dbl but over F_q.
 * Cost: 3M + 5S
 */
void shaw_dbl_x64(shaw_jacobian *r, const shaw_jacobian *p)
{
    fq_fe delta, gamma, beta, alpha;
    fq_fe t0, t1, t2;

    /* delta = Z1^2 */
    fq51_chain_sq(delta, p->Z);

    /* gamma = Y1^2 */
    fq51_chain_sq(gamma, p->Y);

    /* beta = X1 * gamma */
    fq51_chain_mul(beta, p->X, gamma);

    /* alpha = 3 * (X1 - delta) * (X1 + delta) */
    fq_sub(t0, p->X, delta);
    fq_add(t1, p->X, delta);
    fq51_chain_mul(alpha, t0, t1);
    fq_add(t0, alpha, alpha);
    fq_add(alpha, t0, alpha);

    /* X3 = alpha^2 - 8*beta */
    fq51_chain_sq(r->X, alpha);
    fq_add(t0, beta, beta); /* 2*beta */
    fq_add(t0, t0, t0); /* 4*beta */
    fq_sub(r->X, r->X, t0); /* alpha^2 - 4*beta */
    fq_sub(r->X, r->X, t0); /* alpha^2 - 8*beta */

    /* Z3 = (Y1 + Z1)^2 - gamma - delta */
    fq_add(t1, p->Y, p->Z);
    fq51_chain_sq(t2, t1);
    fq_sub(t2, t2, gamma);
    fq_sub(r->Z, t2, delta);

    /* Y3 = alpha * (4*beta - X3) - 8*gamma^2 */
    fq_sub(t1, t0, r->X); /* 4*beta - X3 */
    fq51_chain_mul(t2, alpha, t1);
    fq51_chain_sq(t0, gamma); /* gamma^2 */
    fq_add(t0, t0, t0); /* 2*gamma^2 */
    fq_add(t0, t0, t0); /* 4*gamma^2 */
    fq_sub(r->Y, t2, t0); /* ... - 4*gamma^2 */
    fq_sub(r->Y, r->Y, t0); /* ... - 8*gamma^2 */
}
