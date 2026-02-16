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

#include "x64/shaw_add.h"

#include "fq_ops.h"
#include "x64/fq51_chain.h"

/*
 * General addition: Jacobian + Jacobian -> Jacobian (over F_q)
 * EFD: add-2007-bl. Cost: 11M + 5S
 *
 * Raw incomplete formula — does not handle p == q, p == -q, or identity inputs.
 * Edge cases are handled by the inline wrapper in shaw_add.h.
 */

#if defined(FQ51_HAVE_ADX_MUL)
/*
 * Pack-once 4×64 variant: pack 6 input fields once at entry, perform all
 * 11M + 5S + 15 add/sub in 4×64 representation, unpack 3 outputs at exit.
 * Saves 32+ pack/unpack conversions per point add.
 */
void shaw_add_x64(shaw_jacobian *r, const shaw_jacobian *p, const shaw_jacobian *q)
{
    uint64_t pX[4], pY[4], pZ[4], qX[4], qY[4], qZ[4];
    fq51_normalize_and_pack(pX, p->X);
    fq51_normalize_and_pack(pY, p->Y);
    fq51_normalize_and_pack(pZ, p->Z);
    fq51_normalize_and_pack(qX, q->X);
    fq51_normalize_and_pack(qY, q->Y);
    fq51_normalize_and_pack(qZ, q->Z);

    uint64_t Z1Z1[4], Z2Z2[4], U1[4], U2[4], S1[4], S2[4];
    uint64_t H[4], I[4], J[4], rr[4], V[4];
    uint64_t t0[4], t1[4];

    fq64_sq(Z1Z1, pZ);
    fq64_sq(Z2Z2, qZ);

    fq64_mul(U1, pX, Z2Z2);
    fq64_mul(U2, qX, Z1Z1);

    fq64_mul(t0, qZ, Z2Z2);
    fq64_mul(S1, pY, t0);
    fq64_mul(t0, pZ, Z1Z1);
    fq64_mul(S2, qY, t0);

    fq64_sub(H, U2, U1);

    fq64_add(t0, H, H);
    fq64_sq(I, t0);

    fq64_mul(J, H, I);

    fq64_sub(rr, S2, S1);
    fq64_add(rr, rr, rr);

    fq64_mul(V, U1, I);

    uint64_t rX[4], rY[4], rZ[4];

    fq64_sq(rX, rr);
    fq64_sub(rX, rX, J);
    fq64_add(t0, V, V);
    fq64_sub(rX, rX, t0);

    fq64_sub(t0, V, rX);
    fq64_mul(t1, rr, t0);
    fq64_mul(t0, S1, J);
    fq64_add(t0, t0, t0);
    fq64_sub(rY, t1, t0);

    fq64_add(t0, pZ, qZ);
    fq64_sq(t1, t0);
    fq64_sub(t1, t1, Z1Z1);
    fq64_sub(t1, t1, Z2Z2);
    fq64_mul(rZ, t1, H);

    /* Unpack 4×64 → 5×51 with post-normalize */
    const uint64_t M = FQ51_MASK;
    uint64_t c;

    fq64_to_fq51(r->X, rX);
    c = r->X[0] >> 51;
    r->X[0] &= M;
    r->X[1] += c;
    c = r->X[1] >> 51;
    r->X[1] &= M;
    r->X[2] += c;
    c = r->X[2] >> 51;
    r->X[2] &= M;
    r->X[3] += c;
    c = r->X[3] >> 51;
    r->X[3] &= M;
    r->X[4] += c;
    c = r->X[4] >> 51;
    r->X[4] &= M;
    r->X[0] += c * GAMMA_51[0];
    r->X[1] += c * GAMMA_51[1];
    r->X[2] += c * GAMMA_51[2];
    c = r->X[0] >> 51;
    r->X[0] &= M;
    r->X[1] += c;
    c = r->X[1] >> 51;
    r->X[1] &= M;
    r->X[2] += c;

    fq64_to_fq51(r->Y, rY);
    c = r->Y[0] >> 51;
    r->Y[0] &= M;
    r->Y[1] += c;
    c = r->Y[1] >> 51;
    r->Y[1] &= M;
    r->Y[2] += c;
    c = r->Y[2] >> 51;
    r->Y[2] &= M;
    r->Y[3] += c;
    c = r->Y[3] >> 51;
    r->Y[3] &= M;
    r->Y[4] += c;
    c = r->Y[4] >> 51;
    r->Y[4] &= M;
    r->Y[0] += c * GAMMA_51[0];
    r->Y[1] += c * GAMMA_51[1];
    r->Y[2] += c * GAMMA_51[2];
    c = r->Y[0] >> 51;
    r->Y[0] &= M;
    r->Y[1] += c;
    c = r->Y[1] >> 51;
    r->Y[1] &= M;
    r->Y[2] += c;

    fq64_to_fq51(r->Z, rZ);
    c = r->Z[0] >> 51;
    r->Z[0] &= M;
    r->Z[1] += c;
    c = r->Z[1] >> 51;
    r->Z[1] &= M;
    r->Z[2] += c;
    c = r->Z[2] >> 51;
    r->Z[2] &= M;
    r->Z[3] += c;
    c = r->Z[3] >> 51;
    r->Z[3] &= M;
    r->Z[4] += c;
    c = r->Z[4] >> 51;
    r->Z[4] &= M;
    r->Z[0] += c * GAMMA_51[0];
    r->Z[1] += c * GAMMA_51[1];
    r->Z[2] += c * GAMMA_51[2];
    c = r->Z[0] >> 51;
    r->Z[0] &= M;
    r->Z[1] += c;
    c = r->Z[1] >> 51;
    r->Z[1] &= M;
    r->Z[2] += c;
}

#else

void shaw_add_x64(shaw_jacobian *r, const shaw_jacobian *p, const shaw_jacobian *q)
{
    fq_fe Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, rr, V;
    fq_fe t0, t1;

    fq51_chain_sq(Z1Z1, p->Z);
    fq51_chain_sq(Z2Z2, q->Z);

    fq51_chain_mul(U1, p->X, Z2Z2);
    fq51_chain_mul(U2, q->X, Z1Z1);

    fq51_chain_mul(t0, q->Z, Z2Z2);
    fq51_chain_mul(S1, p->Y, t0);
    fq51_chain_mul(t0, p->Z, Z1Z1);
    fq51_chain_mul(S2, q->Y, t0);

    fq_sub(H, U2, U1);

    fq_add(t0, H, H);
    fq51_chain_sq(I, t0);

    fq51_chain_mul(J, H, I);

    fq_sub(rr, S2, S1);
    fq_add(rr, rr, rr);

    fq51_chain_mul(V, U1, I);

    fq51_chain_sq(r->X, rr);
    fq_sub(r->X, r->X, J);
    fq_add(t0, V, V);
    fq_sub(r->X, r->X, t0);

    fq_sub(t0, V, r->X);
    fq51_chain_mul(t1, rr, t0);
    fq51_chain_mul(t0, S1, J);
    fq_add(t0, t0, t0);
    fq_sub(r->Y, t1, t0);

    fq_add(t0, p->Z, q->Z);
    fq51_chain_sq(t1, t0);
    fq_sub(t1, t1, Z1Z1);
    fq_sub(t1, t1, Z2Z2);
    fq51_chain_mul(r->Z, t1, H);
}

#endif /* FQ51_HAVE_ADX_MUL */
