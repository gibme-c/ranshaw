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
 * AVX-512 IFMA constant-time scalar multiplication for the Shaw curve using fq10
 * (radix-2^25.5) field arithmetic.
 *
 * The key optimization: fq10 uses only 64-bit multiplies (no 128-bit multiply),
 * which is significantly faster on MSVC where 128-bit multiply emulation via
 * uint128_emu causes massive register spilling when force-inlined.
 *
 * The IFMA TU is compiled with AVX-512 flags which imply AVX2, so we can
 * include AVX2 headers (fq10_avx2.h).
 *
 * Algorithm: signed 4-bit fixed-window (radix-16), identical to x64 baseline.
 *   1. Precompute table [P, 2P, 3P, ..., 8P] using fq51 ops (batch_to_affine
 *      needs fq_invert which is fq51-only)
 *   2. Recode scalar to 64 signed digits in [-8, 8]
 *   3. Main loop (63 down to 0): 4 doublings, CT table lookup, CT conditional
 *      negate, mixed addition — all using inline fq10 point ops
 *   4. Convert result back to fq51, secure erase intermediates
 */

#include "shaw_scalarmult.h"

#include "fq_invert.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "fq_tobytes.h"
#include "fq_utils.h"
#include "ranshaw_secure_erase.h"
#include "shaw.h"
#include "shaw_ops.h"
#include "x64/avx2/fq10_avx2.h"
#include "x64/shaw_add.h"
#include "x64/shaw_dbl.h"

#include <vector>

/* ------------------------------------------------------------------ */
/* fq10 affine point type                                             */
/* ------------------------------------------------------------------ */

typedef struct
{
    fq10 x, y;
} shaw_affine_10;

/* ------------------------------------------------------------------ */
/* fq10 constant-time helpers                                         */
/* ------------------------------------------------------------------ */

static inline void shaw_affine10_cmov(shaw_affine_10 *r, const shaw_affine_10 *p, int64_t b)
{
    fq10_cmov(r->x, p->x, b);
    fq10_cmov(r->y, p->y, b);
}

static inline void shaw_affine10_cneg(shaw_affine_10 *r, int64_t b)
{
    fq10 neg_y;
    fq10_neg(neg_y, r->y);
    fq10_cmov(r->y, neg_y, b);
}

/* ---- fq10 zero / one / isnonzero ---- */

static inline void fq10_set0(fq10 h)
{
    h[0] = h[1] = h[2] = h[3] = h[4] = 0;
    h[5] = h[6] = h[7] = h[8] = h[9] = 0;
}

static inline void fq10_set1(fq10 h)
{
    h[0] = 1;
    h[1] = h[2] = h[3] = h[4] = 0;
    h[5] = h[6] = h[7] = h[8] = h[9] = 0;
}

/*
 * CT check if fq10 element is nonzero (mod q). Returns 1 if nonzero, 0 if zero.
 * Used to detect identity (Z == 0).
 *
 * Cannot simply OR the limbs: fq10_sub(x, x) produces q (a non-canonical
 * representation of 0 with all-nonzero limbs). We must fully reduce through
 * fq_tobytes via fq_isnonzero.
 */
static inline unsigned int fq10_isnonzero_ct(const fq10 f)
{
    fq_fe tmp;
    fq10_to_fq51(tmp, f);
    return (unsigned int)fq_isnonzero(tmp);
}

/* ------------------------------------------------------------------ */
/* fq10 point doubling — dbl-2001-b, a = -3                          */
/* Cost: 3M + 4S (fq10 ops)                                          */
/* ------------------------------------------------------------------ */

static inline void shaw_dbl_fq10(fq10 rX, fq10 rY, fq10 rZ, const fq10 pX, const fq10 pY, const fq10 pZ)
{
    fq10 delta, gamma, beta, alpha, t0, t1, t2;

    fq10_sq(delta, pZ); /* delta = Z1^2 */
    fq10_sq(gamma, pY); /* gamma = Y1^2 */
    fq10_mul(beta, pX, gamma); /* beta  = X1 * gamma */

    fq10_sub(t0, pX, delta);
    fq10_add(t1, pX, delta);
    fq10_mul(alpha, t0, t1);
    fq10_add(t0, alpha, alpha);
    fq10_add(alpha, t0, alpha); /* alpha = 3*(X1-delta)*(X1+delta) */

    fq10_sq(rX, alpha); /* alpha^2 */
    fq10_add(t0, beta, beta); /* 2*beta */
    fq10_add(t0, t0, t0); /* 4*beta */
    fq10_sub(rX, rX, t0); /* alpha^2 - 4*beta */
    fq10_sub(rX, rX, t0); /* alpha^2 - 8*beta = X3 */

    fq10_add(t1, pY, pZ);
    fq10_sq(t2, t1);
    fq10_sub(t2, t2, gamma);
    fq10_sub(rZ, t2, delta); /* Z3 = (Y1+Z1)^2 - gamma - delta */

    fq10_sub(t1, t0, rX); /* 4*beta - X3 */
    fq10_mul(t2, alpha, t1);
    fq10_sq(t0, gamma); /* gamma^2 */
    fq10_add(t0, t0, t0); /* 2*gamma^2 */
    fq10_add(t0, t0, t0); /* 4*gamma^2 */
    fq10_sub(rY, t2, t0); /* - 4*gamma^2 */
    fq10_sub(rY, rY, t0); /* - 8*gamma^2 = Y3 */
}

/* ------------------------------------------------------------------ */
/* fq10 mixed addition — madd-2007-bl                                 */
/* Cost: 7M + 4S (fq10 ops)                                          */
/* ------------------------------------------------------------------ */

static inline void shaw_madd_fq10(
    fq10 rX,
    fq10 rY,
    fq10 rZ,
    const fq10 pX,
    const fq10 pY,
    const fq10 pZ,
    const fq10 qx,
    const fq10 qy)
{
    fq10 Z1Z1, U2, S2, H, HH, I, J, rr, V, t0, t1;

    fq10_sq(Z1Z1, pZ);
    fq10_mul(U2, qx, Z1Z1);
    fq10_mul(t0, pZ, Z1Z1);
    fq10_mul(S2, qy, t0);
    fq10_sub(H, U2, pX);
    fq10_sq(HH, H);
    fq10_add(I, HH, HH);
    fq10_add(I, I, I);
    fq10_mul(J, H, I);
    fq10_sub(rr, S2, pY);
    fq10_add(rr, rr, rr);
    fq10_mul(V, pX, I);

    fq10_sq(rX, rr);
    fq10_sub(rX, rX, J);
    fq10_add(t0, V, V);
    fq10_sub(rX, rX, t0);

    fq10_sub(t0, V, rX);
    fq10_mul(t1, rr, t0);
    fq10_mul(t0, pY, J);
    fq10_add(t0, t0, t0);
    fq10_sub(rY, t1, t0);

    fq10_add(t0, pZ, H);
    fq10_sq(t1, t0);
    fq10_sub(t1, t1, Z1Z1);
    fq10_sub(rZ, t1, HH);
}

/* ------------------------------------------------------------------ */
/* Scalar recoding                                                    */
/* ------------------------------------------------------------------ */

/*
 * Recode scalar into signed 4-bit digits.
 * Input: 256-bit scalar as 32 bytes LE
 * Output: 64 signed digits in [-8, 8], with carry absorbed
 *
 * Each digit d[i] represents bits [4i, 4i+3] with a borrow/carry scheme
 * such that scalar = sum(d[i] * 16^i).
 */
static void scalar_recode_signed4(int8_t digits[64], const unsigned char scalar[32])
{
    /* Extract 4-bit nibbles */
    uint8_t nibbles[64];
    for (int i = 0; i < 32; i++)
    {
        nibbles[2 * i] = scalar[i] & 0x0f;
        nibbles[2 * i + 1] = (scalar[i] >> 4) & 0x0f;
    }

    /* Convert to signed (branchless): carry = (val + 8) >> 4 */
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

/* ------------------------------------------------------------------ */
/* Batch affine conversion (fq51, single inversion)                   */
/* ------------------------------------------------------------------ */

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

    /* Collect Z values */
    for (size_t i = 0; i < n; i++)
        fq_copy(z_vals[i].v, in[i].Z);

    /* Compute cumulative products: products[i] = z[0] * z[1] * ... * z[i] */
    fq_copy(products[0].v, z_vals[0].v);
    for (size_t i = 1; i < n; i++)
        fq_mul(products[i].v, products[i - 1].v, z_vals[i].v);

    /* Invert the cumulative product */
    fq_fe inv;
    fq_invert(inv, products[n - 1].v);

    /* Work backwards to get individual inverses */
    for (size_t i = n - 1; i > 0; i--)
    {
        fq_fe z_inv;
        fq_mul(z_inv, inv, products[i - 1].v); /* z_inv = 1/z[i] */
        fq_mul(inv, inv, z_vals[i].v); /* inv = 1/(z[0]*...*z[i-1]) */

        fq_fe z_inv2, z_inv3;
        fq_sq(z_inv2, z_inv);
        fq_mul(z_inv3, z_inv2, z_inv);
        fq_mul(out[i].x, in[i].X, z_inv2);
        fq_mul(out[i].y, in[i].Y, z_inv3);
    }

    /* First element: inv is now 1/z[0] */
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

/* ------------------------------------------------------------------ */
/* Entry point                                                        */
/* ------------------------------------------------------------------ */

void shaw_scalarmult_ifma(shaw_jacobian *r, const unsigned char scalar[32], const shaw_jacobian *p)
{
    /* Step 1: Precompute table [P, 2P, 3P, 4P, 5P, 6P, 7P, 8P] using fq51 */
    shaw_jacobian table_jac[8];
    shaw_copy(&table_jac[0], p); /* 1P */
    shaw_dbl_x64(&table_jac[1], p); /* 2P */
    shaw_add_x64(&table_jac[2], &table_jac[1], p); /* 3P */
    shaw_dbl_x64(&table_jac[3], &table_jac[1]); /* 4P */
    shaw_add_x64(&table_jac[4], &table_jac[3], p); /* 5P */
    shaw_dbl_x64(&table_jac[5], &table_jac[2]); /* 6P */
    shaw_add_x64(&table_jac[6], &table_jac[5], p); /* 7P */
    shaw_dbl_x64(&table_jac[7], &table_jac[3]); /* 8P */

    /* Step 2: Convert to affine (single inversion) */
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

    /* Step 5: Main loop -- start from the top digit */
    fq10 rX, rY, rZ;

    /* Initialize with the top digit (branchless abs + sign extraction) */
    int32_t d = (int32_t)digits[63];
    int32_t sign_mask = -(int32_t)((uint32_t)d >> 31);
    unsigned int abs_d = (unsigned int)((d ^ sign_mask) - sign_mask);
    unsigned int neg = (unsigned int)(sign_mask & 1);

    /* CT table lookup for initial value */
    shaw_affine_10 selected;
    fq10_set0(selected.x);
    fq10_set0(selected.y);

    for (unsigned int j = 0; j < 8; j++)
    {
        unsigned int eq = ((abs_d ^ (j + 1)) - 1u) >> 31;
        shaw_affine10_cmov(&selected, &table10[j], eq);
    }

    /* CT conditional negate */
    shaw_affine10_cneg(&selected, neg);

    /* Always compute both paths, then CT select */
    /* from_table: (x:y:1) */
    fq10 tableX, tableY, tableZ;
    fq10_copy(tableX, selected.x);
    fq10_copy(tableY, selected.y);
    fq10_set1(tableZ);

    /* identity: (1:1:0) */
    fq10 identX, identY, identZ;
    fq10_set1(identX);
    fq10_set1(identY);
    fq10_set0(identZ);

    /* CT select: identity if abs_d == 0, from_table otherwise */
    unsigned int nonzero = 1u ^ ((abs_d - 1u) >> 31);
    fq10_copy(rX, identX);
    fq10_copy(rY, identY);
    fq10_copy(rZ, identZ);
    fq10_cmov(rX, tableX, (int64_t)nonzero);
    fq10_cmov(rY, tableY, (int64_t)nonzero);
    fq10_cmov(rZ, tableZ, (int64_t)nonzero);

    /* Main loop: digits[62] down to digits[0] */
    for (int i = 62; i >= 0; i--)
    {
        /* 4 doublings */
        shaw_dbl_fq10(rX, rY, rZ, rX, rY, rZ);
        shaw_dbl_fq10(rX, rY, rZ, rX, rY, rZ);
        shaw_dbl_fq10(rX, rY, rZ, rX, rY, rZ);
        shaw_dbl_fq10(rX, rY, rZ, rX, rY, rZ);

        /* Extract digit (branchless) */
        d = (int32_t)digits[i];
        sign_mask = -(int32_t)((uint32_t)d >> 31);
        abs_d = (unsigned int)((d ^ sign_mask) - sign_mask);
        neg = (unsigned int)(sign_mask & 1);

        /* CT table lookup */
        fq10_set1(selected.x);
        fq10_set1(selected.y);
        for (unsigned int j = 0; j < 8; j++)
        {
            unsigned int eq = ((abs_d ^ (j + 1)) - 1u) >> 31;
            shaw_affine10_cmov(&selected, &table10[j], eq);
        }

        /* CT conditional negate */
        shaw_affine10_cneg(&selected, neg);

        /* Mixed addition if digit != 0 */
        nonzero = 1u ^ ((abs_d - 1u) >> 31);

        /* Handle identity accumulator: madd(identity, P) is degenerate.
         * If Z==0 (identity), use from_affine instead. */
        unsigned int z_nonzero = fq10_isnonzero_ct(rZ);

        fq10 tmpX, tmpY, tmpZ;
        shaw_madd_fq10(tmpX, tmpY, tmpZ, rX, rY, rZ, selected.x, selected.y);

        fq10 freshX, freshY, freshZ;
        fq10_copy(freshX, selected.x);
        fq10_copy(freshY, selected.y);
        fq10_set1(freshZ);

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

    /* Step 6: Convert result back to fq51 */
    fq10_to_fq51(r->X, rX);
    fq10_to_fq51(r->Y, rY);
    fq10_to_fq51(r->Z, rZ);

    /* Step 7: Secure erase */
    ranshaw_secure_erase(table_jac, sizeof(table_jac));
    ranshaw_secure_erase(table_affine, sizeof(table_affine));
    ranshaw_secure_erase(table10, sizeof(table10));
    ranshaw_secure_erase(digits, sizeof(digits));
    ranshaw_secure_erase(rX, sizeof(rX));
    ranshaw_secure_erase(rY, sizeof(rY));
    ranshaw_secure_erase(rZ, sizeof(rZ));
}
