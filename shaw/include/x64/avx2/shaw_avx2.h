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
 * @file shaw_avx2.h
 * @brief AVX2 backend for Shaw curve operations.
 */

#ifndef RANSHAW_X64_AVX2_SHAW_AVX2_H
#define RANSHAW_X64_AVX2_SHAW_AVX2_H

#include "shaw.h"
#include "x64/avx2/fq10_avx2.h"
#include "x64/avx2/fq10x4_avx2.h"

// 4-way parallel Jacobian point for Shaw (Fq)
typedef struct
{
    fq10x4 X, Y, Z;
} shaw_jacobian_4x;

/**
 * Set 4-way Jacobian point to the identity (point at infinity).
 * Identity in Jacobian coordinates: (1 : 1 : 0)
 */
static inline void shaw_identity_4x(shaw_jacobian_4x *r)
{
    fq10x4_1(&r->X);
    fq10x4_1(&r->Y);
    fq10x4_0(&r->Z);
}

/**
 * Copy a 4-way Jacobian point.
 */
static inline void shaw_copy_4x(shaw_jacobian_4x *r, const shaw_jacobian_4x *p)
{
    fq10x4_copy(&r->X, &p->X);
    fq10x4_copy(&r->Y, &p->Y);
    fq10x4_copy(&r->Z, &p->Z);
}

/**
 * Negate a 4-way Jacobian point: -(X, Y, Z) = (X, -Y, Z)
 */
static inline void shaw_neg_4x(shaw_jacobian_4x *r, const shaw_jacobian_4x *p)
{
    fq10x4_copy(&r->X, &p->X);
    fq10x4_neg(&r->Y, &p->Y);
    fq10x4_copy(&r->Z, &p->Z);
}

/**
 * Constant-time conditional move: if mask is all-ones, copy u into t.
 */
static inline void shaw_cmov_4x(shaw_jacobian_4x *t, const shaw_jacobian_4x *u, __m256i mask)
{
    fq10x4_cmov(&t->X, &u->X, mask);
    fq10x4_cmov(&t->Y, &u->Y, mask);
    fq10x4_cmov(&t->Z, &u->Z, mask);
}

/**
 * Jacobian point doubling (a = -3 optimization, dbl-2001-b).
 * Cost: 3M + 5S
 *
 * delta = Z1^2
 * gamma = Y1^2
 * beta  = X1 * gamma
 * alpha = 3 * (X1 - delta) * (X1 + delta)
 * X3 = alpha^2 - 8*beta
 * Z3 = (Y1 + Z1)^2 - gamma - delta
 * Y3 = alpha * (4*beta - X3) - 8*gamma^2
 */
static inline void shaw_dbl_4x(shaw_jacobian_4x *r, const shaw_jacobian_4x *p)
{
    fq10x4 delta, gamma, beta, alpha, t0, t1, t2;

    fq10x4_sq(&delta, &p->Z); // delta = Z1^2
    fq10x4_sq(&gamma, &p->Y); // gamma = Y1^2
    fq10x4_mul(&beta, &p->X, &gamma); // beta = X1 * gamma

    // alpha = 3 * (X1 - delta) * (X1 + delta)
    fq10x4_sub(&t0, &p->X, &delta);
    fq10x4_add(&t1, &p->X, &delta);
    fq10x4_mul(&alpha, &t0, &t1);
    fq10x4_add(&t0, &alpha, &alpha);
    fq10x4_add(&alpha, &t0, &alpha); // 3 * (X1 - delta)(X1 + delta)
    fq10x4_carry_gamma(&alpha); // Normalize: limbs can reach 28 bits after 2 chained adds,
                                // causing g*19 pre-products to exceed 32 bits in mul_epu32

    // X3 = alpha^2 - 8*beta
    fq10x4_sq(&r->X, &alpha);
    fq10x4_add(&t0, &beta, &beta); // 2*beta
    fq10x4_add(&t0, &t0, &t0); // 4*beta
    fq10x4_carry_gamma(&t0); // Normalize: 2 chained adds produce 28-bit limbs, but fq10x4_sub's
                             // 2q bias only covers up to 27-bit subtrahends. Without this carry,
                             // (26-bit + 2q_bias) - 28-bit goes negative, wrapping in uint64_t,
                             // and the unsigned carry shift (srli) produces garbage.
    fq10x4_sub(&r->X, &r->X, &t0); // alpha^2 - 4*beta
    fq10x4_sub(&r->X, &r->X, &t0); // alpha^2 - 8*beta

    // Z3 = (Y1 + Z1)^2 - gamma - delta
    fq10x4_add(&t1, &p->Y, &p->Z);
    fq10x4_sq(&t2, &t1);
    fq10x4_sub(&t2, &t2, &gamma);
    fq10x4_sub(&r->Z, &t2, &delta);

    // Y3 = alpha * (4*beta - X3) - 8*gamma^2
    fq10x4_sub(&t1, &t0, &r->X); // 4*beta - X3
    fq10x4_mul(&t2, &alpha, &t1);
    fq10x4_sq(&t0, &gamma); // gamma^2
    fq10x4_add(&t0, &t0, &t0); // 2*gamma^2
    fq10x4_add(&t0, &t0, &t0); // 4*gamma^2
    fq10x4_carry_gamma(&t0); // Normalize: same 28-bit limb issue as 4*beta above
    fq10x4_sub(&r->Y, &t2, &t0); // - 4*gamma^2
    fq10x4_sub(&r->Y, &r->Y, &t0); // - 8*gamma^2
}

/**
 * General Jacobian point addition (add-2007-bl).
 * Cost: 11M + 5S
 *
 * Z1Z1 = Z1^2, Z2Z2 = Z2^2
 * U1 = X1*Z2Z2, U2 = X2*Z1Z1
 * S1 = Y1*Z2*Z2Z2, S2 = Y2*Z1*Z1Z1
 * H = U2 - U1
 * I = (2*H)^2
 * J = H*I
 * rr = 2*(S2 - S1)
 * V = U1*I
 * X3 = rr^2 - J - 2*V
 * Y3 = rr*(V - X3) - 2*S1*J
 * Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
 */
static inline void shaw_add_4x(shaw_jacobian_4x *r, const shaw_jacobian_4x *p, const shaw_jacobian_4x *q)
{
    fq10x4 z1z1, z2z2, u1, u2, s1, s2, h, i, j, rr, v, t0, t1;

    fq10x4_sq(&z1z1, &p->Z); // Z1Z1 = Z1^2
    fq10x4_sq(&z2z2, &q->Z); // Z2Z2 = Z2^2

    fq10x4_mul(&u1, &p->X, &z2z2); // U1 = X1*Z2Z2
    fq10x4_mul(&u2, &q->X, &z1z1); // U2 = X2*Z1Z1

    fq10x4_mul(&t0, &q->Z, &z2z2); // Z2*Z2Z2
    fq10x4_mul(&s1, &p->Y, &t0); // S1 = Y1*Z2*Z2Z2

    fq10x4_mul(&t0, &p->Z, &z1z1); // Z1*Z1Z1
    fq10x4_mul(&s2, &q->Y, &t0); // S2 = Y2*Z1*Z1Z1

    fq10x4_sub(&h, &u2, &u1); // H = U2 - U1

    fq10x4_add(&t0, &h, &h); // 2*H
    fq10x4_sq(&i, &t0); // I = (2*H)^2

    fq10x4_mul(&j, &h, &i); // J = H*I

    fq10x4_sub(&t0, &s2, &s1); // S2 - S1
    fq10x4_add(&rr, &t0, &t0); // rr = 2*(S2 - S1)

    fq10x4_mul(&v, &u1, &i); // V = U1*I

    // X3 = rr^2 - J - 2*V
    fq10x4_sq(&r->X, &rr); // rr^2
    fq10x4_sub(&r->X, &r->X, &j); // rr^2 - J
    fq10x4_add(&t0, &v, &v); // 2*V
    fq10x4_sub(&r->X, &r->X, &t0); // rr^2 - J - 2*V

    // Y3 = rr*(V - X3) - 2*S1*J
    fq10x4_sub(&t0, &v, &r->X); // V - X3
    fq10x4_mul(&t1, &rr, &t0); // rr*(V - X3)
    fq10x4_mul(&t0, &s1, &j); // S1*J
    fq10x4_add(&t0, &t0, &t0); // 2*S1*J
    fq10x4_sub(&r->Y, &t1, &t0); // rr*(V - X3) - 2*S1*J

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    fq10x4_add(&t0, &p->Z, &q->Z); // Z1+Z2
    fq10x4_sq(&t1, &t0); // (Z1+Z2)^2
    fq10x4_sub(&t1, &t1, &z1z1); // - Z1Z1
    fq10x4_sub(&t1, &t1, &z2z2); // - Z2Z2
    fq10x4_mul(&r->Z, &t1, &h); // * H
}

/**
 * Pack four fq51 Jacobian points into a 4-way fq10x4 Jacobian point.
 */
static inline void shaw_pack_4x(
    shaw_jacobian_4x *out,
    const shaw_jacobian *p0,
    const shaw_jacobian *p1,
    const shaw_jacobian *p2,
    const shaw_jacobian *p3)
{
    fq10 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;

    fq51_to_fq10(x0, p0->X);
    fq51_to_fq10(x1, p1->X);
    fq51_to_fq10(x2, p2->X);
    fq51_to_fq10(x3, p3->X);
    fq10x4_pack(&out->X, x0, x1, x2, x3);

    fq51_to_fq10(y0, p0->Y);
    fq51_to_fq10(y1, p1->Y);
    fq51_to_fq10(y2, p2->Y);
    fq51_to_fq10(y3, p3->Y);
    fq10x4_pack(&out->Y, y0, y1, y2, y3);

    fq51_to_fq10(z0, p0->Z);
    fq51_to_fq10(z1, p1->Z);
    fq51_to_fq10(z2, p2->Z);
    fq51_to_fq10(z3, p3->Z);
    fq10x4_pack(&out->Z, z0, z1, z2, z3);
}

/**
 * Unpack a 4-way fq10x4 Jacobian point into four fq51 Jacobian points.
 */
static inline void shaw_unpack_4x(
    shaw_jacobian *p0,
    shaw_jacobian *p1,
    shaw_jacobian *p2,
    shaw_jacobian *p3,
    const shaw_jacobian_4x *in)
{
    fq10 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;

    fq10x4_unpack(x0, x1, x2, x3, &in->X);
    fq10_to_fq51(p0->X, x0);
    fq10_to_fq51(p1->X, x1);
    fq10_to_fq51(p2->X, x2);
    fq10_to_fq51(p3->X, x3);

    fq10x4_unpack(y0, y1, y2, y3, &in->Y);
    fq10_to_fq51(p0->Y, y0);
    fq10_to_fq51(p1->Y, y1);
    fq10_to_fq51(p2->Y, y2);
    fq10_to_fq51(p3->Y, y3);

    fq10x4_unpack(z0, z1, z2, z3, &in->Z);
    fq10_to_fq51(p0->Z, z0);
    fq10_to_fq51(p1->Z, z1);
    fq10_to_fq51(p2->Z, z2);
    fq10_to_fq51(p3->Z, z3);
}

/**
 * Insert a single fq51 Jacobian point into one lane of a 4-way point.
 */
static inline void shaw_insert_lane_4x(shaw_jacobian_4x *out, const shaw_jacobian *p, int lane)
{
    fq10 x, y, z;

    fq51_to_fq10(x, p->X);
    fq51_to_fq10(y, p->Y);
    fq51_to_fq10(z, p->Z);

    fq10x4_insert_lane(&out->X, x, lane);
    fq10x4_insert_lane(&out->Y, y, lane);
    fq10x4_insert_lane(&out->Z, z, lane);
}

/**
 * Extract a single lane from a 4-way point into a fq51 Jacobian point.
 */
static inline void shaw_extract_lane_4x(shaw_jacobian *out, const shaw_jacobian_4x *in, int lane)
{
    fq10 x, y, z;

    fq10x4_extract_lane(x, &in->X, lane);
    fq10x4_extract_lane(y, &in->Y, lane);
    fq10x4_extract_lane(z, &in->Z, lane);

    fq10_to_fq51(out->X, x);
    fq10_to_fq51(out->Y, y);
    fq10_to_fq51(out->Z, z);
}

#endif
