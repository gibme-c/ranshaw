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
 * @file shaw_ifma.h
 * @brief AVX-512 IFMA backend for Shaw curve operations.
 */

#ifndef RANSHAW_X64_IFMA_SHAW_IFMA_H
#define RANSHAW_X64_IFMA_SHAW_IFMA_H

#include "shaw.h"
#include "x64/ifma/fq51x8_ifma.h"

// 8-way parallel Jacobian point for Shaw (Fq)
typedef struct
{
    fq51x8 X, Y, Z;
} shaw_jacobian_8x;

/**
 * Set 8-way Jacobian point to the identity (point at infinity).
 * Identity in Jacobian coordinates: (1 : 1 : 0)
 */
static inline void shaw_identity_8x(shaw_jacobian_8x *r)
{
    fq51x8_1(&r->X);
    fq51x8_1(&r->Y);
    fq51x8_0(&r->Z);
}

/**
 * Copy an 8-way Jacobian point.
 */
static inline void shaw_copy_8x(shaw_jacobian_8x *r, const shaw_jacobian_8x *p)
{
    fq51x8_copy(&r->X, &p->X);
    fq51x8_copy(&r->Y, &p->Y);
    fq51x8_copy(&r->Z, &p->Z);
}

/**
 * Negate an 8-way Jacobian point: -(X, Y, Z) = (X, -Y, Z)
 */
static inline void shaw_neg_8x(shaw_jacobian_8x *r, const shaw_jacobian_8x *p)
{
    fq51x8_copy(&r->X, &p->X);
    fq51x8_neg(&r->Y, &p->Y);
    fq51x8_copy(&r->Z, &p->Z);
}

/**
 * Constant-time conditional move: for each of the 8 lanes, if the corresponding
 * bit in mask is set, copy u into t; otherwise keep t.
 */
static inline void shaw_cmov_8x(shaw_jacobian_8x *t, const shaw_jacobian_8x *u, __mmask8 mask)
{
    fq51x8_cmov(&t->X, &u->X, mask);
    fq51x8_cmov(&t->Y, &u->Y, mask);
    fq51x8_cmov(&t->Z, &u->Z, mask);
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
 *
 * Normalize-weak placement: only before mul/sq inputs that exceed 52 bits.
 * After mul/sq/sub: limbs <= 51 bits (safe for IFMA).
 * After 1 add of reduced inputs: limbs <= 52 bits (safe for IFMA).
 * After 2+ adds without intervening mul/sq/sub: limbs may exceed 52 bits (NEED normalize).
 */
static inline void shaw_dbl_8x(shaw_jacobian_8x *r, const shaw_jacobian_8x *p)
{
    fq51x8 delta, gamma, beta, alpha, t0, t1, t2;

    fq51x8_sq(&delta, &p->Z); // delta = Z1^2, <= 51 bits
    fq51x8_sq(&gamma, &p->Y); // gamma = Y1^2, <= 51 bits
    fq51x8_mul(&beta, &p->X, &gamma); // beta = X1 * gamma, <= 51 bits

    // alpha = 3 * (X1 - delta) * (X1 + delta)
    fq51x8_sub(&t0, &p->X, &delta); // <= 51 bits (sub carries)
    fq51x8_add(&t1, &p->X, &delta); // <= 52 bits (1 add, OK for mul)
    fq51x8_mul(&alpha, &t0, &t1); // (X1-delta)(X1+delta), <= 51 bits
    fq51x8_add(&t0, &alpha, &alpha); // 2 * product, <= 52 bits
    fq51x8_add(&alpha, &t0, &alpha); // 3 * product, <= 53 bits -- NEED normalize
    fq51x8_normalize_weak(&alpha); // <= 51 bits

    // X3 = alpha^2 - 8*beta
    fq51x8_sq(&r->X, &alpha); // alpha^2, <= 51 bits
    fq51x8_add(&t0, &beta, &beta); // 2*beta, <= 52 bits
    fq51x8_add(&t0, &t0, &t0); // 4*beta, <= 53 bits (sub operand, not mul -- OK)
    fq51x8_sub(&r->X, &r->X, &t0); // alpha^2 - 4*beta, <= 51 bits (sub carries)
    fq51x8_sub(&r->X, &r->X, &t0); // alpha^2 - 8*beta, <= 51 bits

    // Z3 = (Y1 + Z1)^2 - gamma - delta
    fq51x8_add(&t1, &p->Y, &p->Z); // <= 52 bits (1 add, OK for sq)
    fq51x8_sq(&t2, &t1); // (Y1+Z1)^2, <= 51 bits
    fq51x8_sub(&t2, &t2, &gamma); // - gamma, <= 51 bits
    fq51x8_sub(&r->Z, &t2, &delta); // - delta, <= 51 bits

    // Y3 = alpha * (4*beta - X3) - 8*gamma^2
    // t0 is still 4*beta (<= 53 bits), used as sub operand -- OK
    fq51x8_sub(&t1, &t0, &r->X); // 4*beta - X3, <= 51 bits (sub carries)
    // t1 <= 51 bits, alpha <= 51 bits (was normalized) -- both OK for mul
    fq51x8_mul(&t2, &alpha, &t1); // alpha * (4*beta - X3), <= 51 bits
    fq51x8_sq(&t0, &gamma); // gamma^2, <= 51 bits
    fq51x8_add(&t0, &t0, &t0); // 2*gamma^2, <= 52 bits
    fq51x8_add(&t0, &t0, &t0); // 4*gamma^2, <= 53 bits (sub operand -- OK)
    fq51x8_sub(&r->Y, &t2, &t0); // - 4*gamma^2, <= 51 bits
    fq51x8_sub(&r->Y, &r->Y, &t0); // - 8*gamma^2, <= 51 bits
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
 *
 * No normalize_weak needed: every mul/sq input is either a mul/sq output (<= 51 bits),
 * a sub output (<= 51 bits), or a single add of reduced inputs (<= 52 bits).
 */
static inline void shaw_add_8x(shaw_jacobian_8x *r, const shaw_jacobian_8x *p, const shaw_jacobian_8x *q)
{
    fq51x8 z1z1, z2z2, u1, u2, s1, s2, h, i, j, rr, v, t0, t1;

    fq51x8_sq(&z1z1, &p->Z); // Z1Z1 = Z1^2
    fq51x8_sq(&z2z2, &q->Z); // Z2Z2 = Z2^2

    fq51x8_mul(&u1, &p->X, &z2z2); // U1 = X1*Z2Z2
    fq51x8_mul(&u2, &q->X, &z1z1); // U2 = X2*Z1Z1

    fq51x8_mul(&t0, &q->Z, &z2z2); // Z2*Z2Z2
    fq51x8_mul(&s1, &p->Y, &t0); // S1 = Y1*Z2*Z2Z2

    fq51x8_mul(&t0, &p->Z, &z1z1); // Z1*Z1Z1
    fq51x8_mul(&s2, &q->Y, &t0); // S2 = Y2*Z1*Z1Z1

    fq51x8_sub(&h, &u2, &u1); // H = U2 - U1

    fq51x8_add(&t0, &h, &h); // 2*H, <= 52 bits (OK for sq)
    fq51x8_sq(&i, &t0); // I = (2*H)^2

    fq51x8_mul(&j, &h, &i); // J = H*I

    fq51x8_sub(&t0, &s2, &s1); // S2 - S1
    fq51x8_add(&rr, &t0, &t0); // rr = 2*(S2 - S1), <= 52 bits (OK for sq/mul)

    fq51x8_mul(&v, &u1, &i); // V = U1*I

    // X3 = rr^2 - J - 2*V
    fq51x8_sq(&r->X, &rr); // rr^2
    fq51x8_sub(&r->X, &r->X, &j); // rr^2 - J
    fq51x8_add(&t0, &v, &v); // 2*V, <= 52 bits (sub operand -- OK)
    fq51x8_sub(&r->X, &r->X, &t0); // rr^2 - J - 2*V

    // Y3 = rr*(V - X3) - 2*S1*J
    fq51x8_sub(&t0, &v, &r->X); // V - X3
    fq51x8_mul(&t1, &rr, &t0); // rr*(V - X3)
    fq51x8_mul(&t0, &s1, &j); // S1*J
    fq51x8_add(&t0, &t0, &t0); // 2*S1*J, <= 52 bits (sub operand -- OK)
    fq51x8_sub(&r->Y, &t1, &t0); // rr*(V - X3) - 2*S1*J

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    fq51x8_add(&t0, &p->Z, &q->Z); // Z1+Z2, <= 52 bits (OK for sq)
    fq51x8_sq(&t1, &t0); // (Z1+Z2)^2
    fq51x8_sub(&t1, &t1, &z1z1); // - Z1Z1
    fq51x8_sub(&t1, &t1, &z2z2); // - Z2Z2
    fq51x8_mul(&r->Z, &t1, &h); // * H
}

/**
 * Pack eight fq51 Jacobian points into an 8-way fq51x8 Jacobian point.
 * No radix conversion needed -- both use radix-2^51.
 */
static inline void shaw_pack_8x(
    shaw_jacobian_8x *out,
    const shaw_jacobian *p0,
    const shaw_jacobian *p1,
    const shaw_jacobian *p2,
    const shaw_jacobian *p3,
    const shaw_jacobian *p4,
    const shaw_jacobian *p5,
    const shaw_jacobian *p6,
    const shaw_jacobian *p7)
{
    fq51x8_insert_lane(&out->X, p0->X, 0);
    fq51x8_insert_lane(&out->X, p1->X, 1);
    fq51x8_insert_lane(&out->X, p2->X, 2);
    fq51x8_insert_lane(&out->X, p3->X, 3);
    fq51x8_insert_lane(&out->X, p4->X, 4);
    fq51x8_insert_lane(&out->X, p5->X, 5);
    fq51x8_insert_lane(&out->X, p6->X, 6);
    fq51x8_insert_lane(&out->X, p7->X, 7);

    fq51x8_insert_lane(&out->Y, p0->Y, 0);
    fq51x8_insert_lane(&out->Y, p1->Y, 1);
    fq51x8_insert_lane(&out->Y, p2->Y, 2);
    fq51x8_insert_lane(&out->Y, p3->Y, 3);
    fq51x8_insert_lane(&out->Y, p4->Y, 4);
    fq51x8_insert_lane(&out->Y, p5->Y, 5);
    fq51x8_insert_lane(&out->Y, p6->Y, 6);
    fq51x8_insert_lane(&out->Y, p7->Y, 7);

    fq51x8_insert_lane(&out->Z, p0->Z, 0);
    fq51x8_insert_lane(&out->Z, p1->Z, 1);
    fq51x8_insert_lane(&out->Z, p2->Z, 2);
    fq51x8_insert_lane(&out->Z, p3->Z, 3);
    fq51x8_insert_lane(&out->Z, p4->Z, 4);
    fq51x8_insert_lane(&out->Z, p5->Z, 5);
    fq51x8_insert_lane(&out->Z, p6->Z, 6);
    fq51x8_insert_lane(&out->Z, p7->Z, 7);
}

/**
 * Unpack an 8-way fq51x8 Jacobian point into eight fq51 Jacobian points.
 * No radix conversion needed -- both use radix-2^51.
 */
static inline void shaw_unpack_8x(
    shaw_jacobian *p0,
    shaw_jacobian *p1,
    shaw_jacobian *p2,
    shaw_jacobian *p3,
    shaw_jacobian *p4,
    shaw_jacobian *p5,
    shaw_jacobian *p6,
    shaw_jacobian *p7,
    const shaw_jacobian_8x *in)
{
    fq51x8_extract_lane(p0->X, &in->X, 0);
    fq51x8_extract_lane(p1->X, &in->X, 1);
    fq51x8_extract_lane(p2->X, &in->X, 2);
    fq51x8_extract_lane(p3->X, &in->X, 3);
    fq51x8_extract_lane(p4->X, &in->X, 4);
    fq51x8_extract_lane(p5->X, &in->X, 5);
    fq51x8_extract_lane(p6->X, &in->X, 6);
    fq51x8_extract_lane(p7->X, &in->X, 7);

    fq51x8_extract_lane(p0->Y, &in->Y, 0);
    fq51x8_extract_lane(p1->Y, &in->Y, 1);
    fq51x8_extract_lane(p2->Y, &in->Y, 2);
    fq51x8_extract_lane(p3->Y, &in->Y, 3);
    fq51x8_extract_lane(p4->Y, &in->Y, 4);
    fq51x8_extract_lane(p5->Y, &in->Y, 5);
    fq51x8_extract_lane(p6->Y, &in->Y, 6);
    fq51x8_extract_lane(p7->Y, &in->Y, 7);

    fq51x8_extract_lane(p0->Z, &in->Z, 0);
    fq51x8_extract_lane(p1->Z, &in->Z, 1);
    fq51x8_extract_lane(p2->Z, &in->Z, 2);
    fq51x8_extract_lane(p3->Z, &in->Z, 3);
    fq51x8_extract_lane(p4->Z, &in->Z, 4);
    fq51x8_extract_lane(p5->Z, &in->Z, 5);
    fq51x8_extract_lane(p6->Z, &in->Z, 6);
    fq51x8_extract_lane(p7->Z, &in->Z, 7);
}

/**
 * Insert a single fq51 Jacobian point into one lane of an 8-way point.
 */
static inline void shaw_insert_lane_8x(shaw_jacobian_8x *out, const shaw_jacobian *p, int lane)
{
    fq51x8_insert_lane(&out->X, p->X, lane);
    fq51x8_insert_lane(&out->Y, p->Y, lane);
    fq51x8_insert_lane(&out->Z, p->Z, lane);
}

/**
 * Extract a single lane from an 8-way point into a fq51 Jacobian point.
 */
static inline void shaw_extract_lane_8x(shaw_jacobian *out, const shaw_jacobian_8x *in, int lane)
{
    fq51x8_extract_lane(out->X, &in->X, lane);
    fq51x8_extract_lane(out->Y, &in->Y, lane);
    fq51x8_extract_lane(out->Z, &in->Z, lane);
}

#endif // RANSHAW_X64_IFMA_SHAW_IFMA_H
