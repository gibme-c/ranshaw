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

/*
 * AVX2 constant-time scalar multiplication for Shaw (over F_q).
 *
 * Uses radix-2^25.5 (fq10) field arithmetic throughout the main loop
 * to avoid 128-bit integer arithmetic (MSVC _umul128 register spilling).
 * Converts fq51 -> fq10 once at entry, fq10 -> fq51 once at exit.
 *
 * Algorithm: signed 4-bit fixed-window (radix-16), same as shaw_scalarmult_x64
 * but with inline fq10 point doubling (dbl-2001-b, a=-3) and mixed addition
 * (madd-2007-bl).
 */

#include "shaw_scalarmult.h"

#include "fq_invert.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "ranshaw_secure_erase.h"
#include "shaw.h"
#include "shaw_ops.h"
#include "x64/avx2/fq10_avx2.h"
#include "x64/shaw_add.h"
#include "x64/shaw_dbl.h"

#include <vector>

/* ── fq10 affine point type ── */

typedef struct
{
    fq10 x, y;
} shaw_affine_10;

/* ── Scalar recoding ── */

/*
 * Recode scalar into signed 4-bit digits.
 * Input: 256-bit scalar as 32 bytes LE
 * Output: 64 signed digits in [-8, 8], with carry absorbed
 */
static void scalar_recode_signed4(int8_t digits[64], const unsigned char scalar[32])
{
    uint8_t nibbles[64];
    for (int i = 0; i < 32; i++)
    {
        nibbles[2 * i] = scalar[i] & 0x0f;
        nibbles[2 * i + 1] = (scalar[i] >> 4) & 0x0f;
    }

    int carry = 0;
    for (int i = 0; i < 63; i++)
    {
        int val = nibbles[i] + carry;
        carry = (val + 8) >> 4;
        digits[i] = (int8_t)(val - (carry << 4));
    }
    digits[63] = (int8_t)(nibbles[63] + carry);
    ranshaw_secure_erase(nibbles, sizeof(nibbles));
}

/* ── Batch affine conversion (fq51) ── */

/*
 * Batch affine conversion using Montgomery's trick.
 * Converts n Jacobian points to affine using a single inversion.
 */
static void batch_to_affine(shaw_affine *out, const shaw_jacobian *in, size_t n)
{
    if (n == 0)
        return;

    struct fq_fe_s
    {
        fq_fe v;
    };
    std::vector<fq_fe_s> z_vals(n);
    std::vector<fq_fe_s> products(n);

    for (size_t i = 0; i < n; i++)
        fq_copy(z_vals[i].v, in[i].Z);

    fq_copy(products[0].v, z_vals[0].v);
    for (size_t i = 1; i < n; i++)
        fq_mul(products[i].v, products[i - 1].v, z_vals[i].v);

    fq_fe inv;
    fq_invert(inv, products[n - 1].v);

    for (size_t i = n - 1; i > 0; i--)
    {
        fq_fe z_inv;
        fq_mul(z_inv, inv, products[i - 1].v);
        fq_mul(inv, inv, z_vals[i].v);

        fq_fe z_inv2, z_inv3;
        fq_sq(z_inv2, z_inv);
        fq_mul(z_inv3, z_inv2, z_inv);
        fq_mul(out[i].x, in[i].X, z_inv2);
        fq_mul(out[i].y, in[i].Y, z_inv3);
    }

    {
        fq_fe z_inv2, z_inv3;
        fq_sq(z_inv2, inv);
        fq_mul(z_inv3, z_inv2, inv);
        fq_mul(out[0].x, in[0].X, z_inv2);
        fq_mul(out[0].y, in[0].Y, z_inv3);
    }

    ranshaw_secure_erase(&inv, sizeof(inv));
    ranshaw_secure_erase(z_vals.data(), n * sizeof(fq_fe_s));
    ranshaw_secure_erase(products.data(), n * sizeof(fq_fe_s));
}

/* ── Inline fq10 point doubling: dbl-2001-b with a = -3 ── */

/*
 * Jacobian doubling with a = -3 optimization (same formula as shaw_dbl_x64).
 * Cost: 3M + 5S (in fq10 arithmetic)
 *
 * Input:  Jacobian (X1:Y1:Z1) in fq10
 * Output: Jacobian (X3:Y3:Z3) in fq10
 */
static FQ10_AVX2_FORCE_INLINE void shaw_dbl_fq10(fq10 X3, fq10 Y3, fq10 Z3, const fq10 X1, const fq10 Y1, const fq10 Z1)
{
    fq10 delta, gamma, beta, alpha;
    fq10 t0, t1, t2;

    /* delta = Z1^2 */
    fq10_sq(delta, Z1);

    /* gamma = Y1^2 */
    fq10_sq(gamma, Y1);

    /* beta = X1 * gamma */
    fq10_mul(beta, X1, gamma);

    /* alpha = 3 * (X1 - delta) * (X1 + delta) */
    fq10_sub(t0, X1, delta);
    fq10_add(t1, X1, delta);
    fq10_mul(alpha, t0, t1);
    fq10_add(t0, alpha, alpha);
    fq10_add(alpha, t0, alpha);

    /* X3 = alpha^2 - 8*beta */
    fq10_sq(X3, alpha);
    fq10_add(t0, beta, beta); /* 2*beta */
    fq10_add(t0, t0, t0); /* 4*beta */
    fq10_add(t1, t0, t0); /* 8*beta */
    fq10_sub(X3, X3, t1);

    /* Z3 = (Y1 + Z1)^2 - gamma - delta */
    fq10_add(t1, Y1, Z1);
    fq10_sq(t2, t1);
    fq10_sub(t2, t2, gamma);
    fq10_sub(Z3, t2, delta);

    /* Y3 = alpha * (4*beta - X3) - 8*gamma^2 */
    fq10_sub(t1, t0, X3); /* 4*beta - X3 */
    fq10_mul(t2, alpha, t1);
    fq10_sq(t0, gamma); /* gamma^2 */
    fq10_add(t0, t0, t0); /* 2*gamma^2 */
    fq10_add(t0, t0, t0); /* 4*gamma^2 */
    fq10_add(t0, t0, t0); /* 8*gamma^2 */
    fq10_sub(Y3, t2, t0);
}

/* ── Inline fq10 mixed addition: madd-2007-bl ── */

/*
 * Mixed addition: Jacobian + Affine -> Jacobian (same formula as shaw_madd_x64).
 * Cost: 7M + 4S (in fq10 arithmetic)
 *
 * Input:  Jacobian (X1:Y1:Z1) in fq10, Affine (x2, y2) in fq10
 * Output: Jacobian (X3:Y3:Z3) in fq10
 */
static FQ10_AVX2_FORCE_INLINE void
    shaw_madd_fq10(fq10 X3, fq10 Y3, fq10 Z3, const fq10 X1, const fq10 Y1, const fq10 Z1, const fq10 x2, const fq10 y2)
{
    fq10 Z1Z1, U2, S2, H, HH, I, J, rr, V;
    fq10 t0, t1;

    /* Z1Z1 = Z1^2 */
    fq10_sq(Z1Z1, Z1);

    /* U2 = x2 * Z1Z1 */
    fq10_mul(U2, x2, Z1Z1);

    /* S2 = y2 * Z1 * Z1Z1 */
    fq10_mul(t0, Z1, Z1Z1);
    fq10_mul(S2, y2, t0);

    /* H = U2 - X1 */
    fq10_sub(H, U2, X1);

    /* HH = H^2 */
    fq10_sq(HH, H);

    /* I = 4 * HH */
    fq10_add(I, HH, HH);
    fq10_add(I, I, I);

    /* J = H * I */
    fq10_mul(J, H, I);

    /* rr = 2 * (S2 - Y1) */
    fq10_sub(rr, S2, Y1);
    fq10_add(rr, rr, rr);

    /* V = X1 * I */
    fq10_mul(V, X1, I);

    /* X3 = rr^2 - J - 2*V */
    fq10_sq(X3, rr);
    fq10_sub(X3, X3, J);
    fq10_add(t0, V, V);
    fq10_sub(X3, X3, t0);

    /* Y3 = rr * (V - X3) - 2 * Y1 * J */
    fq10_sub(t0, V, X3);
    fq10_mul(t1, rr, t0);
    fq10_mul(t0, Y1, J);
    fq10_add(t0, t0, t0);
    fq10_sub(Y3, t1, t0);

    /* Z3 = (Z1 + H)^2 - Z1Z1 - HH */
    fq10_add(t0, Z1, H);
    fq10_sq(t1, t0);
    fq10_sub(t1, t1, Z1Z1);
    fq10_sub(Z3, t1, HH);
}

/* ── Constant-time table operations in fq10 ── */

static FQ10_AVX2_FORCE_INLINE void shaw_affine_10_cmov(shaw_affine_10 *r, const shaw_affine_10 *p, int64_t b)
{
    fq10_cmov(r->x, p->x, b);
    fq10_cmov(r->y, p->y, b);
}

static FQ10_AVX2_FORCE_INLINE void shaw_affine_10_cneg(shaw_affine_10 *r, unsigned int b)
{
    fq10 neg_y;
    fq10_neg(neg_y, r->y);
    fq10_cmov(r->y, neg_y, (int64_t)b);
}

/* ── Main function ── */

void shaw_scalarmult_avx2(shaw_jacobian *r, const unsigned char scalar[32], const shaw_jacobian *p)
{
    /* Step 1: Precompute table [P, 2P, 3P, 4P, 5P, 6P, 7P, 8P] in Jacobian (fq51) */
    shaw_jacobian table_jac[8];
    shaw_copy(&table_jac[0], p); /* 1P */
    shaw_dbl_x64(&table_jac[1], p); /* 2P */
    shaw_add_x64(&table_jac[2], &table_jac[1], p); /* 3P */
    shaw_dbl_x64(&table_jac[3], &table_jac[1]); /* 4P */
    shaw_add_x64(&table_jac[4], &table_jac[3], p); /* 5P */
    shaw_dbl_x64(&table_jac[5], &table_jac[2]); /* 6P */
    shaw_add_x64(&table_jac[6], &table_jac[5], p); /* 7P */
    shaw_dbl_x64(&table_jac[7], &table_jac[3]); /* 8P */

    /* Step 2: Batch convert to affine (single inversion, fq51) */
    shaw_affine table_affine[8];
    batch_to_affine(table_affine, table_jac, 8);

    /* Step 3: Convert affine table to fq10 */
    shaw_affine_10 table10[8];
    for (int i = 0; i < 8; i++)
    {
        fq51_to_fq10(table10[i].x, table_affine[i].x);
        fq51_to_fq10(table10[i].y, table_affine[i].y);
    }

    /* Step 4: Recode scalar */
    int8_t digits[64];
    scalar_recode_signed4(digits, scalar);

    /* Step 5: Initialize from top digit (branchless abs + sign) */
    int32_t d = (int32_t)digits[63];
    int32_t sign_mask = -(int32_t)((uint32_t)d >> 31);
    unsigned int abs_d = (unsigned int)((d ^ sign_mask) - sign_mask);
    unsigned int neg = (unsigned int)(sign_mask & 1);

    /* CT table lookup for initial value (fq10) */
    shaw_affine_10 selected;
    for (int j = 0; j < 10; j++)
        selected.x[j] = 0;
    for (int j = 0; j < 10; j++)
        selected.y[j] = 0;

    for (unsigned int j = 0; j < 8; j++)
    {
        unsigned int eq = ((abs_d ^ (j + 1)) - 1u) >> 31;
        shaw_affine_10_cmov(&selected, &table10[j], (int64_t)eq);
    }

    /* CT conditional negate */
    shaw_affine_10_cneg(&selected, neg);

    /* Working accumulator in fq10 */
    fq10 rX, rY, rZ;

    /* Always compute both paths, then CT select */
    /* from_table: (x:y:1) */
    fq10 tableX, tableY, tableZ;
    fq10_copy(tableX, selected.x);
    fq10_copy(tableY, selected.y);
    for (int j = 0; j < 10; j++)
        tableZ[j] = 0;
    tableZ[0] = 1;

    /* identity: (1:1:0) */
    fq10 identX, identY, identZ;
    for (int j = 0; j < 10; j++)
        identX[j] = 0;
    identX[0] = 1;
    for (int j = 0; j < 10; j++)
        identY[j] = 0;
    identY[0] = 1;
    for (int j = 0; j < 10; j++)
        identZ[j] = 0;

    /* CT select: identity if abs_d == 0, from_table otherwise */
    unsigned int nonzero = 1u ^ ((abs_d - 1u) >> 31);
    fq10_copy(rX, identX);
    fq10_copy(rY, identY);
    fq10_copy(rZ, identZ);
    fq10_cmov(rX, tableX, (int64_t)nonzero);
    fq10_cmov(rY, tableY, (int64_t)nonzero);
    fq10_cmov(rZ, tableZ, (int64_t)nonzero);

    /* Step 6: Main loop: digits[62] down to digits[0] */
    for (int i = 62; i >= 0; i--)
    {
        /* 4 doublings in fq10 */
        shaw_dbl_fq10(rX, rY, rZ, rX, rY, rZ);
        shaw_dbl_fq10(rX, rY, rZ, rX, rY, rZ);
        shaw_dbl_fq10(rX, rY, rZ, rX, rY, rZ);
        shaw_dbl_fq10(rX, rY, rZ, rX, rY, rZ);

        /* Extract digit (branchless) */
        d = (int32_t)digits[i];
        sign_mask = -(int32_t)((uint32_t)d >> 31);
        abs_d = (unsigned int)((d ^ sign_mask) - sign_mask);
        neg = (unsigned int)(sign_mask & 1);

        /* CT table lookup (fq10) — default to (1, 1) for identity handling */
        for (int j = 0; j < 10; j++)
            selected.x[j] = 0;
        selected.x[0] = 1;
        for (int j = 0; j < 10; j++)
            selected.y[j] = 0;
        selected.y[0] = 1;

        for (unsigned int j = 0; j < 8; j++)
        {
            unsigned int eq = ((abs_d ^ (j + 1)) - 1u) >> 31;
            shaw_affine_10_cmov(&selected, &table10[j], (int64_t)eq);
        }

        /* CT conditional negate */
        shaw_affine_10_cneg(&selected, neg);

        /* Mixed addition if digit != 0 */
        nonzero = 1u ^ ((abs_d - 1u) >> 31);

        /* Check if accumulator is identity (Z==0).
         * Convert Z back to fq51 to use fq_isnonzero. */
        fq_fe z_check;
        fq10_to_fq51(z_check, rZ);
        unsigned int z_nonzero = (unsigned int)fq_isnonzero(z_check);

        /* madd result */
        fq10 tmpX, tmpY, tmpZ;
        shaw_madd_fq10(tmpX, tmpY, tmpZ, rX, rY, rZ, selected.x, selected.y);

        /* from_affine result (for when accumulator is identity) */
        fq10 freshX, freshY, freshZ;
        fq10_copy(freshX, selected.x);
        fq10_copy(freshY, selected.y);
        for (int j = 0; j < 10; j++)
            freshZ[j] = 0;
        freshZ[0] = 1;

        /* If digit nonzero and accumulator is valid (Z!=0): use madd result */
        int64_t use_madd = (int64_t)(nonzero & z_nonzero);
        fq10_cmov(rX, tmpX, use_madd);
        fq10_cmov(rY, tmpY, use_madd);
        fq10_cmov(rZ, tmpZ, use_madd);

        /* If digit nonzero and accumulator is identity (Z==0): use from_affine */
        int64_t use_fresh = (int64_t)(nonzero & (1u - z_nonzero));
        fq10_cmov(rX, freshX, use_fresh);
        fq10_cmov(rY, freshY, use_fresh);
        fq10_cmov(rZ, freshZ, use_fresh);
    }

    /* Step 7: Convert result back to fq51 */
    fq10_to_fq51(r->X, rX);
    fq10_to_fq51(r->Y, rY);
    fq10_to_fq51(r->Z, rZ);

    /* Secure erase */
    ranshaw_secure_erase(table_jac, sizeof(table_jac));
    ranshaw_secure_erase(table_affine, sizeof(table_affine));
    ranshaw_secure_erase(table10, sizeof(table10));
    ranshaw_secure_erase(digits, sizeof(digits));
    ranshaw_secure_erase(&selected, sizeof(selected));
    ranshaw_secure_erase(rX, sizeof(rX));
    ranshaw_secure_erase(rY, sizeof(rY));
    ranshaw_secure_erase(rZ, sizeof(rZ));
}
