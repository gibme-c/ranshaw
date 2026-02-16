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

#include "portable/ran_add.h"

#include "fp_ops.h"
#include "portable/fp25_chain.h"

/*
 * General addition: Jacobian + Jacobian -> Jacobian
 * EFD: add-2007-bl
 * Cost: 11M + 5S
 *
 * Raw incomplete formula — does not handle p == q, p == -q, or identity inputs.
 * Edge cases are handled by the inline wrapper in ran_add.h.
 *
 * Z1Z1 = Z1^2, Z2Z2 = Z2^2
 * U1 = X1*Z2Z2, U2 = X2*Z1Z1
 * S1 = Y1*Z2*Z2Z2, S2 = Y2*Z1*Z1Z1
 * H = U2 - U1
 * I = (2*H)^2
 * J = H*I
 * r = 2*(S2 - S1)
 * V = U1*I
 * X3 = r^2 - J - 2*V
 * Y3 = r*(V - X3) - 2*S1*J
 * Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
 */
void ran_add_portable(ran_jacobian *r, const ran_jacobian *p, const ran_jacobian *q)
{
    fp_fe Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, rr, V;
    fp_fe t0, t1;

    /* Z1Z1 = Z1^2, Z2Z2 = Z2^2 */
    fp25_chain_sq(Z1Z1, p->Z);
    fp25_chain_sq(Z2Z2, q->Z);

    /* U1 = X1*Z2Z2, U2 = X2*Z1Z1 */
    fp25_chain_mul(U1, p->X, Z2Z2);
    fp25_chain_mul(U2, q->X, Z1Z1);

    /* S1 = Y1*Z2*Z2Z2, S2 = Y2*Z1*Z1Z1 */
    fp25_chain_mul(t0, q->Z, Z2Z2);
    fp25_chain_mul(S1, p->Y, t0);
    fp25_chain_mul(t0, p->Z, Z1Z1);
    fp25_chain_mul(S2, q->Y, t0);

    /* H = U2 - U1 */
    fp_sub(H, U2, U1);

    /* I = (2*H)^2 */
    fp_add(t0, H, H);
    fp25_chain_sq(I, t0);

    /* J = H*I */
    fp25_chain_mul(J, H, I);

    /* rr = 2*(S2 - S1) */
    fp_sub(rr, S2, S1);
    fp_add(rr, rr, rr);

    /* V = U1*I */
    fp25_chain_mul(V, U1, I);

    /* X3 = rr^2 - J - 2*V */
    fp25_chain_sq(r->X, rr);
    fp_sub(r->X, r->X, J);
    fp_add(t0, V, V);
    fp_sub(r->X, r->X, t0);

    /* Y3 = rr*(V - X3) - 2*S1*J */
    fp_sub(t0, V, r->X);
    fp25_chain_mul(t1, rr, t0);
    fp25_chain_mul(t0, S1, J);
    fp_add(t0, t0, t0);
    fp_sub(r->Y, t1, t0);

    /* Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H */
    fp_add(t0, p->Z, q->Z);
    fp25_chain_sq(t1, t0);
    fp_sub(t1, t1, Z1Z1);
    fp_sub(t1, t1, Z2Z2);
    fp25_chain_mul(r->Z, t1, H);
}
