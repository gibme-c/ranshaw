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
 * @file x64/shaw_map_to_curve.cpp
 * @brief Constant-time simplified SWU map-to-curve for Shaw (RFC 9380 section 6.6.2).
 *
 * Shaw: y^2 = x^3 - 3x + b over F_q (q = 2^255 - gamma).
 * A = -3, B = b. Since A != 0 and B != 0, simplified SWU applies directly.
 * Z = -1 (non-square in F_q, g(B/(Z*A)) is square).
 *
 * Since q ≡ 3 (mod 4), fq_sqrt computes z^((q+1)/4) which is the principal
 * square root when z is a QR. To check if gx is a QR, we compute sqrt and
 * verify by squaring.
 *
 * This implementation is fully constant-time as required by RFC 9380 Section 4.
 * All branches on secret-derived data are replaced with cmov selections.
 */

#include "shaw_map_to_curve.h"

#include "fq_cmov.h"
#include "fq_cneg.h"
#include "fq_frombytes.h"
#include "fq_invert.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "fq_sqrt.h"
#include "fq_tobytes.h"
#include "fq_utils.h"
#include "shaw_add.h"
#include "shaw_constants.h"
#include "shaw_ops.h"

/* Z = -1 mod q */
static const fq_fe SSWU_Z =
    {0x04645EC70F85EULL, 0x1C72E61F4EE2DULL, 0x7FFFFFD2EC7ACULL, 0x7FFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFULL};

/* -B/A = b/3 mod q */
static const fq_fe SSWU_NEG_B_OVER_A =
    {0x1576D988C94B0ULL, 0x30416E92A6BF3ULL, 0x60E7CC341F1CDULL, 0x2DE0528CA1516ULL, 0x4C021D4F8D4FEULL};

/* B/(Z*A) = b/((-1)*(-3)) = b/3 mod q */
static const fq_fe SSWU_B_OVER_ZA =
    {0x1576D988C94B0ULL, 0x30416E92A6BF3ULL, 0x60E7CC341F1CDULL, 0x2DE0528CA1516ULL, 0x4C021D4F8D4FEULL};

/* A = -3 mod q */
static const fq_fe SSWU_A =
    {0x04645EC70F85CULL, 0x1C72E61F4EE2DULL, 0x7FFFFFD2EC7ACULL, 0x7FFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFULL};

/*
 * Constant-time equality check via serialization and OR-fold.
 * Returns a clean 0/1 unsigned int suitable for cmov.
 */
static unsigned int fq_ct_equal(const fq_fe a, const fq_fe b)
{
    unsigned char sa[32], sb[32];
    fq_tobytes(sa, a);
    fq_tobytes(sb, b);
    uint32_t d = 0;
    for (int i = 0; i < 32; i++)
        d |= sa[i] ^ sb[i];
    return ((uint32_t)(d - 1u)) >> 31;
}

/*
 * Constant-time simplified SWU (RFC 9380 section 6.6.2)
 *
 * Input: field element u
 * Output: Jacobian point (x:y:1) on Shaw
 */
static void sswu_shaw(shaw_jacobian *r, const fq_fe u)
{
    fq_fe u2, u4, Zu2, Z2u4, denom, tv1;
    fq_fe x1, gx1, x2, gx2;
    fq_fe x, y;

    /* u^2 */
    fq_sq(u2, u);

    /* Z * u^2 */
    fq_mul(Zu2, SSWU_Z, u2);

    /* Z^2 * u^4 */
    fq_sq(u4, u2);
    fq_fe Z2;
    fq_sq(Z2, SSWU_Z);
    fq_mul(Z2u4, Z2, u4);

    /* denom = Z^2*u^4 + Z*u^2 */
    fq_add(denom, Z2u4, Zu2);

    /* CT denom-is-zero flag */
    unsigned char denom_bytes[32];
    fq_tobytes(denom_bytes, denom);
    uint32_t denom_d = 0;
    for (int i = 0; i < 32; i++)
        denom_d |= denom_bytes[i];
    unsigned int denom_z = ((uint32_t)(denom_d - 1u)) >> 31;

    /* Always compute inv(denom) — Fermat inversion gives inv0 semantics: 0^(q-2) = 0 */
    fq_invert(tv1, denom);

    /* x1 = (-B/A) * (1 + tv1) */
    fq_fe one_plus_tv1;
    fq_fe one;
    fq_1(one);
    fq_add(one_plus_tv1, one, tv1);
    fq_mul(x1, SSWU_NEG_B_OVER_A, one_plus_tv1);

    /* Select exceptional case: x1 = B/(Z*A) when denom was zero */
    fq_cmov(x1, SSWU_B_OVER_ZA, denom_z);

    /* gx1 = x1^3 + A*x1 + B */
    fq_fe x1_sq, x1_cu, ax1;
    fq_sq(x1_sq, x1);
    fq_mul(x1_cu, x1_sq, x1);
    fq_mul(ax1, SSWU_A, x1);
    fq_add(gx1, x1_cu, ax1);
    fq_add(gx1, gx1, SHAW_B);

    /* x2 = Z * u^2 * x1 */
    fq_mul(x2, Zu2, x1);

    /* gx2 = x2^3 + A*x2 + B */
    fq_fe x2_sq, x2_cu, ax2;
    fq_sq(x2_sq, x2);
    fq_mul(x2_cu, x2_sq, x2);
    fq_mul(ax2, SSWU_A, x2);
    fq_add(gx2, x2_cu, ax2);
    fq_add(gx2, gx2, SHAW_B);

    /* Always compute sqrt of both gx1 and gx2 */
    fq_fe sqrt_gx1, sqrt_gx2, check;
    fq_sqrt(sqrt_gx1, gx1);
    fq_sqrt(sqrt_gx2, gx2);

    /* Verify gx1 is square by checking sqrt(gx1)^2 == gx1 */
    fq_sq(check, sqrt_gx1);
    unsigned int gx1_is_square = fq_ct_equal(check, gx1);

    /* CT select: if gx1_is_square, use (x1, sqrt_gx1); else (x2, sqrt_gx2) */
    fq_copy(x, x2);
    fq_copy(y, sqrt_gx2);
    fq_cmov(x, x1, gx1_is_square);
    fq_cmov(y, sqrt_gx1, gx1_is_square);

    /* CT sign adjustment: sgn0(u) != sgn0(y) => negate y */
    unsigned int u_sign = (unsigned int)fq_isnegative(u);
    unsigned int y_sign = (unsigned int)fq_isnegative(y);
    fq_cneg(y, y, u_sign ^ y_sign);

    /* Output as Jacobian with Z=1 */
    fq_copy(r->X, x);
    fq_copy(r->Y, y);
    fq_1(r->Z);
}

void shaw_map_to_curve_x64(shaw_jacobian *r, const unsigned char u[32])
{
    fq_fe u_fe;
    fq_frombytes(u_fe, u);
    sswu_shaw(r, u_fe);
}

void shaw_map_to_curve2_x64(shaw_jacobian *r, const unsigned char u0[32], const unsigned char u1[32])
{
    shaw_jacobian p0, p1;
    shaw_map_to_curve_x64(&p0, u0);
    shaw_map_to_curve_x64(&p1, u1);
    shaw_add(r, &p0, &p1);
}
