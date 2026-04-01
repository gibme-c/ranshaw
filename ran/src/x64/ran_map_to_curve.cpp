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
 * @file x64/ran_map_to_curve.cpp
 * @brief Constant-time simplified SWU map-to-curve for Ran (RFC 9380 section 6.6.2).
 *
 * Ran: y^2 = x^3 - 3x + b over F_p (p = 2^255 - 19).
 * A = -3, B = b. Since A != 0 and B != 0, simplified SWU applies directly.
 * Z = 7 (non-square in F_p, g(B/(Z*A)) is square).
 *
 * This implementation is fully constant-time as required by RFC 9380 Section 4.
 * All branches on secret-derived data are replaced with cmov selections.
 */

#include "ran_map_to_curve.h"

#include "fp_cmov.h"
#include "fp_cneg.h"
#include "fp_frombytes.h"
#include "fp_invert.h"
#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "fp_sqrt.h"
#include "fp_tobytes.h"
#include "fp_utils.h"
#include "ran_add.h"
#include "ran_constants.h"
#include "ran_ops.h"

/* Z = -2 mod p */
static const fp_fe SSWU_Z =
    {0x7FFFFFFFFFFEBULL, 0x7FFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFULL};

/* -B/A = b/3 mod p */
static const fp_fe SSWU_NEG_B_OVER_A =
    {0x573A3509B467AULL, 0x5053957D6EC74ULL, 0x21934CE3A1488ULL, 0x4902A9E82C622ULL, 0x15C6A74E3C972ULL};

/* B/(Z*A) = b/((-2)*(-3)) = b/6 mod p */
static const fp_fe SSWU_B_OVER_ZA =
    {0x2B9D1A84DA33DULL, 0x2829CABEB763AULL, 0x10C9A671D0A44ULL, 0x248154F416311ULL, 0x0AE353A71E4B9ULL};

/* A = -3 mod p */
static const fp_fe SSWU_A =
    {0x7FFFFFFFFFFEAULL, 0x7FFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFULL};

/*
 * Constant-time equality check via serialization and OR-fold.
 * Returns a clean 0/1 unsigned int suitable for cmov.
 */
static unsigned int fp_ct_equal(const fp_fe a, const fp_fe b)
{
    unsigned char sa[32], sb[32];
    fp_tobytes(sa, a);
    fp_tobytes(sb, b);
    uint32_t d = 0;
    for (int i = 0; i < 32; i++)
        d |= sa[i] ^ sb[i];
    return ((uint32_t)(d - 1u)) >> 31; /* 1 if equal, 0 if not */
}

/*
 * Constant-time simplified SWU (RFC 9380 section 6.6.2)
 *
 * Input: field element u
 * Output: Jacobian point (x:y:1) on Ran
 *
 * All three original branches eliminated:
 *   Branch 1 (denom==0): Always invert; fp_invert gives inv0 semantics (0→0).
 *                         Compute normal-path x1, then cmov to B/(Z*A) if denom was zero.
 *   Branch 2 (gx1 is square): Always compute sqrt(gx1) AND sqrt(gx2), select via cmov.
 *   Branch 3 (sign adjustment): Replace if/negate with fp_cneg.
 */
static void sswu_ran(ran_jacobian *r, const fp_fe u)
{
    fp_fe u2, u4, Zu2, Z2u4, denom, tv1;
    fp_fe x1, gx1, x2, gx2;
    fp_fe x, y;

    /* u^2 */
    fp_sq(u2, u);

    /* Z * u^2 */
    fp_mul(Zu2, SSWU_Z, u2);

    /* Z^2 * u^4 */
    fp_sq(u4, u2);
    fp_fe Z2;
    fp_sq(Z2, SSWU_Z);
    fp_mul(Z2u4, Z2, u4);

    /* denom = Z^2*u^4 + Z*u^2 */
    fp_add(denom, Z2u4, Zu2);

    /* CT denom-is-zero flag */
    unsigned char denom_bytes[32];
    fp_tobytes(denom_bytes, denom);
    uint32_t denom_d = 0;
    for (int i = 0; i < 32; i++)
        denom_d |= denom_bytes[i];
    unsigned int denom_z = ((uint32_t)(denom_d - 1u)) >> 31; /* 1 if zero, 0 if nonzero */

    /* Always compute inv(denom) — Fermat inversion gives inv0 semantics: 0^(p-2) = 0 */
    fp_invert(tv1, denom);

    /* x1 = (-B/A) * (1 + tv1) — normal path (tv1=0 when denom=0, so x1 = -B/A) */
    fp_fe one_plus_tv1;
    fp_fe one;
    fp_1(one);
    fp_add(one_plus_tv1, one, tv1);
    fp_mul(x1, SSWU_NEG_B_OVER_A, one_plus_tv1);

    /* Select exceptional case: x1 = B/(Z*A) when denom was zero */
    fp_cmov(x1, SSWU_B_OVER_ZA, denom_z);

    /* gx1 = x1^3 + A*x1 + B */
    fp_fe x1_sq, x1_cu, ax1;
    fp_sq(x1_sq, x1);
    fp_mul(x1_cu, x1_sq, x1);
    fp_mul(ax1, SSWU_A, x1);
    fp_add(gx1, x1_cu, ax1);
    fp_add(gx1, gx1, RAN_B);

    /* x2 = Z * u^2 * x1 */
    fp_mul(x2, Zu2, x1);

    /* gx2 = x2^3 + A*x2 + B */
    fp_fe x2_sq, x2_cu, ax2;
    fp_sq(x2_sq, x2);
    fp_mul(x2_cu, x2_sq, x2);
    fp_mul(ax2, SSWU_A, x2);
    fp_add(gx2, x2_cu, ax2);
    fp_add(gx2, gx2, RAN_B);

    /* Always compute sqrt of both gx1 and gx2 */
    fp_fe sqrt_gx1, sqrt_gx2, check;
    fp_sqrt(sqrt_gx1, gx1);
    fp_sqrt(sqrt_gx2, gx2);

    /* Verify gx1 is square by checking sqrt(gx1)^2 == gx1 */
    fp_sq(check, sqrt_gx1);
    unsigned int gx1_is_square = fp_ct_equal(check, gx1);

    /* CT select: if gx1_is_square, use (x1, sqrt_gx1); else (x2, sqrt_gx2) */
    fp_copy(x, x2);
    fp_copy(y, sqrt_gx2);
    fp_cmov(x, x1, gx1_is_square);
    fp_cmov(y, sqrt_gx1, gx1_is_square);

    /* CT sign adjustment: sgn0(u) != sgn0(y) => negate y */
    unsigned int u_sign = (unsigned int)fp_isnegative(u);
    unsigned int y_sign = (unsigned int)fp_isnegative(y);
    fp_cneg(y, y, u_sign ^ y_sign);

    /* Output as Jacobian with Z=1 */
    fp_copy(r->X, x);
    fp_copy(r->Y, y);
    fp_1(r->Z);
}

void ran_map_to_curve_x64(ran_jacobian *r, const unsigned char u[32])
{
    fp_fe u_fe;
    fp_frombytes(u_fe, u);
    sswu_ran(r, u_fe);
}

void ran_map_to_curve2_x64(ran_jacobian *r, const unsigned char u0[32], const unsigned char u1[32])
{
    ran_jacobian p0, p1;
    ran_map_to_curve_x64(&p0, u0);
    ran_map_to_curve_x64(&p1, u1);
    ran_add(r, &p0, &p1);
}
