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

#include "portable/shaw_add.h"

#include "fq_ops.h"
#include "portable/fq25_chain.h"

/*
 * General addition: Jacobian + Jacobian -> Jacobian (over F_q)
 * EFD: add-2007-bl. Cost: 11M + 5S
 *
 * Raw incomplete formula — does not handle p == q, p == -q, or identity inputs.
 * Edge cases are handled by the inline wrapper in shaw_add.h.
 */
void shaw_add_portable(shaw_jacobian *r, const shaw_jacobian *p, const shaw_jacobian *q)
{
    fq_fe Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, rr, V;
    fq_fe t0, t1;

    fq25_chain_sq(Z1Z1, p->Z);
    fq25_chain_sq(Z2Z2, q->Z);

    fq25_chain_mul(U1, p->X, Z2Z2);
    fq25_chain_mul(U2, q->X, Z1Z1);

    fq25_chain_mul(t0, q->Z, Z2Z2);
    fq25_chain_mul(S1, p->Y, t0);
    fq25_chain_mul(t0, p->Z, Z1Z1);
    fq25_chain_mul(S2, q->Y, t0);

    fq_sub(H, U2, U1);

    fq_add(t0, H, H);
    fq25_chain_sq(I, t0);

    fq25_chain_mul(J, H, I);

    fq_sub(rr, S2, S1);
    fq_add(rr, rr, rr);

    fq25_chain_mul(V, U1, I);

    fq25_chain_sq(r->X, rr);
    fq_sub(r->X, r->X, J);
    fq_add(t0, V, V);
    fq_sub(r->X, r->X, t0);

    fq_sub(t0, V, r->X);
    fq25_chain_mul(t1, rr, t0);
    fq25_chain_mul(t0, S1, J);
    fq_add(t0, t0, t0);
    fq_sub(r->Y, t1, t0);

    fq_add(t0, p->Z, q->Z);
    fq25_chain_sq(t1, t0);
    fq_sub(t1, t1, Z1Z1);
    fq_sub(t1, t1, Z2Z2);
    fq25_chain_mul(r->Z, t1, H);
}
