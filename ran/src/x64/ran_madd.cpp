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

#include "x64/ran_madd.h"

#include "fp_ops.h"
#include "x64/fp51_chain.h"

/*
 * Mixed addition: Jacobian + Affine -> Jacobian
 * EFD: madd-2007-bl
 * Cost: 7M + 4S
 *
 * Does NOT handle: p == identity, q == identity, p == q, p == -q.
 * Caller must handle these cases.
 *
 * Z1Z1 = Z1^2
 * U2 = X2 * Z1Z1
 * S2 = Y2 * Z1 * Z1Z1
 * H = U2 - X1
 * HH = H^2
 * I = 4 * HH
 * J = H * I
 * r = 2 * (S2 - Y1)
 * V = X1 * I
 * X3 = r^2 - J - 2*V
 * Y3 = r * (V - X3) - 2*Y1*J
 * Z3 = (Z1 + H)^2 - Z1Z1 - HH
 */
void ran_madd_x64(ran_jacobian *r, const ran_jacobian *p, const ran_affine *q)
{
    fp_fe Z1Z1, U2, S2, H, HH, I, J, rr, V;
    fp_fe t0, t1;

    /* Z1Z1 = Z1^2 */
    fp51_chain_sq(Z1Z1, p->Z);

    /* U2 = X2 * Z1Z1 */
    fp51_chain_mul(U2, q->x, Z1Z1);

    /* S2 = Y2 * Z1 * Z1Z1 */
    fp51_chain_mul(t0, p->Z, Z1Z1);
    fp51_chain_mul(S2, q->y, t0);

    /* H = U2 - X1 */
    fp_sub(H, U2, p->X);

    /* HH = H^2 */
    fp51_chain_sq(HH, H);

    /* I = 4 * HH */
    fp_add(I, HH, HH);
    fp_add(I, I, I);

    /* J = H * I */
    fp51_chain_mul(J, H, I);

    /* rr = 2 * (S2 - Y1) */
    fp_sub(rr, S2, p->Y);
    fp_add(rr, rr, rr);

    /* V = X1 * I */
    fp51_chain_mul(V, p->X, I);

    /* X3 = rr^2 - J - 2*V */
    fp51_chain_sq(r->X, rr);
    fp_sub(r->X, r->X, J);
    fp_add(t0, V, V);
    fp_sub(r->X, r->X, t0);

    /* Y3 = rr * (V - X3) - 2*Y1*J */
    fp_sub(t0, V, r->X);
    fp51_chain_mul(t1, rr, t0);
    fp51_chain_mul(t0, p->Y, J);
    fp_add(t0, t0, t0);
    fp_sub(r->Y, t1, t0);

    /* Z3 = (Z1 + H)^2 - Z1Z1 - HH */
    fp_add(t0, p->Z, H);
    fp51_chain_sq(t1, t0);
    fp_sub(t1, t1, Z1Z1);
    fp_sub(r->Z, t1, HH);
}
