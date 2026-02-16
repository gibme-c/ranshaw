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
 * AVX2 constant-time scalar multiplication for the Ran curve using fp10
 * (radix-2^25.5) field arithmetic.
 *
 * The key optimization: fp10 uses only 64-bit multiplies (no 128-bit multiply),
 * which is significantly faster on MSVC where 128-bit multiply emulation via
 * uint128_emu causes massive register spilling when force-inlined.
 *
 * Algorithm: signed 4-bit fixed-window (radix-16), identical to x64 baseline.
 *   1. Precompute table [P, 2P, 3P, ..., 8P] using fp51 ops (batch_to_affine
 *      needs fp_invert which is fp51-only)
 *   2. Recode scalar to 64 signed digits in [-8, 8]
 *   3. Main loop (63 down to 0): 4 doublings, CT table lookup, CT conditional
 *      negate, mixed addition — all using inline fp10 point ops
 *   4. Convert result back to fp51, secure erase intermediates
 */

#include "ran_scalarmult.h"

#include "fp_invert.h"
#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "ran.h"
#include "ran_ops.h"
#include "ranshaw_secure_erase.h"
#include "x64/avx2/fp10_avx2.h"
#include "x64/ran_add.h"
#include "x64/ran_dbl.h"

#include <vector>

/* ------------------------------------------------------------------ */
/* fp10 affine point type                                             */
/* ------------------------------------------------------------------ */

typedef struct
{
    fp10 x, y;
} ran_affine_10;

/* ------------------------------------------------------------------ */
/* fp10 constant-time helpers                                         */
/* ------------------------------------------------------------------ */

static inline void ran_affine10_cmov(ran_affine_10 *r, const ran_affine_10 *p, int64_t b)
{
    fp10_cmov(r->x, p->x, b);
    fp10_cmov(r->y, p->y, b);
}

static inline void ran_affine10_cneg(ran_affine_10 *r, int64_t b)
{
    fp10 neg_y;
    fp10_neg(neg_y, r->y);
    fp10_cmov(r->y, neg_y, b);
}

/* ------------------------------------------------------------------ */
/* fp10 point doubling — dbl-2001-b, a = -3                          */
/* Cost: 3M + 4S (fp10 ops)                                          */
/* ------------------------------------------------------------------ */

static FP10_AVX2_FORCE_INLINE void
    ran_dbl_fp10(fp10 rX, fp10 rY, fp10 rZ, const fp10 pX, const fp10 pY, const fp10 pZ)
{
    fp10 delta, gamma, beta, alpha;
    fp10 t0, t1, t2;

    /* delta = Z1^2 */
    fp10_sq(delta, pZ);

    /* gamma = Y1^2 */
    fp10_sq(gamma, pY);

    /* beta = X1 * gamma */
    fp10_mul(beta, pX, gamma);

    /* alpha = 3 * (X1 - delta) * (X1 + delta) */
    fp10_sub(t0, pX, delta);
    fp10_add(t1, pX, delta);
    fp10_mul(alpha, t0, t1);
    fp10_add(t0, alpha, alpha);
    fp10_add(alpha, t0, alpha);

    /* X3 = alpha^2 - 8*beta */
    fp10_sq(rX, alpha);
    fp10_add(t0, beta, beta); /* 2*beta */
    fp10_add(t0, t0, t0); /* 4*beta */
    fp10_add(t1, t0, t0); /* 8*beta */
    fp10_sub(rX, rX, t1);

    /* Z3 = (Y1 + Z1)^2 - gamma - delta */
    fp10_add(t1, pY, pZ);
    fp10_sq(t2, t1);
    fp10_sub(t2, t2, gamma);
    fp10_sub(rZ, t2, delta);

    /* Y3 = alpha * (4*beta - X3) - 8*gamma^2 */
    fp10_sub(t1, t0, rX); /* 4*beta - X3 */
    fp10_mul(t2, alpha, t1);
    fp10_sq(t0, gamma); /* gamma^2 */
    fp10_add(t0, t0, t0); /* 2*gamma^2 */
    fp10_add(t0, t0, t0); /* 4*gamma^2 */
    fp10_add(t0, t0, t0); /* 8*gamma^2 */
    fp10_sub(rY, t2, t0);
}

/* ------------------------------------------------------------------ */
/* fp10 mixed addition — madd-2007-bl                                 */
/* Cost: 7M + 4S (fp10 ops)                                          */
/* ------------------------------------------------------------------ */

static inline void ran_madd_fp10(
    fp10 rX,
    fp10 rY,
    fp10 rZ,
    const fp10 pX,
    const fp10 pY,
    const fp10 pZ,
    const fp10 qx,
    const fp10 qy)
{
    fp10 Z1Z1, U2, S2, H, HH, I, J, rr, V, t0, t1;

    fp10_sq(Z1Z1, pZ);
    fp10_mul(U2, qx, Z1Z1);
    fp10_mul(t0, pZ, Z1Z1);
    fp10_mul(S2, qy, t0);
    fp10_sub(H, U2, pX);
    fp10_sq(HH, H);
    fp10_add(I, HH, HH);
    fp10_add(I, I, I);
    fp10_mul(J, H, I);
    fp10_sub(rr, S2, pY);
    fp10_add(rr, rr, rr);
    fp10_mul(V, pX, I);

    fp10_sq(rX, rr);
    fp10_sub(rX, rX, J);
    fp10_add(t0, V, V);
    fp10_sub(rX, rX, t0);

    fp10_sub(t0, V, rX);
    fp10_mul(t1, rr, t0);
    fp10_mul(t0, pY, J);
    fp10_add(t0, t0, t0);
    fp10_sub(rY, t1, t0);

    fp10_add(t0, pZ, H);
    fp10_sq(t1, t0);
    fp10_sub(t1, t1, Z1Z1);
    fp10_sub(rZ, t1, HH);
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
/* Batch affine conversion (fp51, single inversion)                   */
/* ------------------------------------------------------------------ */

static void batch_to_affine(ran_affine *out, const ran_jacobian *in, size_t n)
{
    if (n == 0)
        return;

    struct fp_fe_s
    {
        fp_fe v;
    };
    std::vector<fp_fe_s> z_vals(n);
    std::vector<fp_fe_s> products(n);

    /* Collect Z values */
    for (size_t i = 0; i < n; i++)
        fp_copy(z_vals[i].v, in[i].Z);

    /* Compute cumulative products: products[i] = z[0] * z[1] * ... * z[i] */
    fp_copy(products[0].v, z_vals[0].v);
    for (size_t i = 1; i < n; i++)
        fp_mul(products[i].v, products[i - 1].v, z_vals[i].v);

    /* Invert the cumulative product */
    fp_fe inv;
    fp_invert(inv, products[n - 1].v);

    /* Work backwards to get individual inverses */
    for (size_t i = n - 1; i > 0; i--)
    {
        fp_fe z_inv;
        fp_mul(z_inv, inv, products[i - 1].v); /* z_inv = 1/z[i] */
        fp_mul(inv, inv, z_vals[i].v); /* inv = 1/(z[0]*...*z[i-1]) */

        fp_fe z_inv2, z_inv3;
        fp_sq(z_inv2, z_inv);
        fp_mul(z_inv3, z_inv2, z_inv);
        fp_mul(out[i].x, in[i].X, z_inv2);
        fp_mul(out[i].y, in[i].Y, z_inv3);
    }

    /* First element: inv is now 1/z[0] */
    {
        fp_fe z_inv2, z_inv3;
        fp_sq(z_inv2, inv);
        fp_mul(z_inv3, z_inv2, inv);
        fp_mul(out[0].x, in[0].X, z_inv2);
        fp_mul(out[0].y, in[0].Y, z_inv3);
    }

    ranshaw_secure_erase(&inv, sizeof(inv));
    ranshaw_secure_erase(z_vals.data(), n * sizeof(fp_fe_s));
    ranshaw_secure_erase(products.data(), n * sizeof(fp_fe_s));
}

/* ------------------------------------------------------------------ */
/* Entry point                                                        */
/* ------------------------------------------------------------------ */

void ran_scalarmult_avx2(ran_jacobian *r, const unsigned char scalar[32], const ran_jacobian *p)
{
    /* Step 1: Precompute table [P, 2P, 3P, 4P, 5P, 6P, 7P, 8P] using fp51 */
    ran_jacobian table_jac[8];
    ran_copy(&table_jac[0], p); /* 1P */
    ran_dbl_x64(&table_jac[1], p); /* 2P */
    ran_add_x64(&table_jac[2], &table_jac[1], p); /* 3P */
    ran_dbl_x64(&table_jac[3], &table_jac[1]); /* 4P */
    ran_add_x64(&table_jac[4], &table_jac[3], p); /* 5P */
    ran_dbl_x64(&table_jac[5], &table_jac[2]); /* 6P */
    ran_add_x64(&table_jac[6], &table_jac[5], p); /* 7P */
    ran_dbl_x64(&table_jac[7], &table_jac[3]); /* 8P */

    /* Convert to affine (single inversion, all fp51) */
    ran_affine table[8];
    batch_to_affine(table, table_jac, 8);

    /* Convert affine table to fp10 */
    ran_affine_10 table10[8];
    for (int i = 0; i < 8; i++)
    {
        fp51_to_fp10(table10[i].x, table[i].x);
        fp51_to_fp10(table10[i].y, table[i].y);
    }

    /* Step 2: Recode scalar */
    int8_t digits[64];
    scalar_recode_signed4(digits, scalar);

    /* Step 3: Main loop — start from the top digit (branchless abs + sign) */
    int32_t d = (int32_t)digits[63];
    int32_t sign_mask = -(int32_t)((uint32_t)d >> 31);
    unsigned int abs_d = (unsigned int)((d ^ sign_mask) - sign_mask);
    unsigned int neg = (unsigned int)(sign_mask & 1);

    /* CT table lookup for initial value (fp10) */
    ran_affine_10 selected10;
    fp10 zero10 = {0};
    fp10_copy(selected10.x, zero10);
    fp10_copy(selected10.y, zero10);

    for (unsigned int j = 0; j < 8; j++)
    {
        unsigned int eq = ((abs_d ^ (j + 1)) - 1u) >> 31;
        ran_affine10_cmov(&selected10, &table10[j], (int64_t)eq);
    }

    /* CT conditional negate */
    ran_affine10_cneg(&selected10, (int64_t)neg);

    /* Accumulator in fp10 */
    fp10 accX, accY, accZ;

    /* Always compute both paths, then CT select */
    fp10 one10 = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    /* from_table: (x:y:1) */
    fp10 tableX, tableY, tableZ;
    fp10_copy(tableX, selected10.x);
    fp10_copy(tableY, selected10.y);
    fp10_copy(tableZ, one10);

    /* identity: (1:1:0) */
    fp10 identX, identY, identZ;
    fp10_copy(identX, one10);
    fp10_copy(identY, one10);
    fp10_copy(identZ, zero10);

    /* CT select: identity if abs_d == 0, from_table otherwise */
    unsigned int nonzero = 1u ^ ((abs_d - 1u) >> 31);
    fp10_copy(accX, identX);
    fp10_copy(accY, identY);
    fp10_copy(accZ, identZ);
    fp10_cmov(accX, tableX, (int64_t)nonzero);
    fp10_cmov(accY, tableY, (int64_t)nonzero);
    fp10_cmov(accZ, tableZ, (int64_t)nonzero);

    /* Main loop: digits[62] down to digits[0] */
    for (int i = 62; i >= 0; i--)
    {
        /* 4 doublings in fp10 */
        ran_dbl_fp10(accX, accY, accZ, accX, accY, accZ);
        ran_dbl_fp10(accX, accY, accZ, accX, accY, accZ);
        ran_dbl_fp10(accX, accY, accZ, accX, accY, accZ);
        ran_dbl_fp10(accX, accY, accZ, accX, accY, accZ);

        /* Extract digit (branchless) */
        d = (int32_t)digits[i];
        sign_mask = -(int32_t)((uint32_t)d >> 31);
        abs_d = (unsigned int)((d ^ sign_mask) - sign_mask);
        neg = (unsigned int)(sign_mask & 1);

        /* CT table lookup (fp10) */
        fp10_copy(selected10.x, one10);
        fp10_copy(selected10.y, one10);
        for (unsigned int j = 0; j < 8; j++)
        {
            unsigned int eq = ((abs_d ^ (j + 1)) - 1u) >> 31;
            ran_affine10_cmov(&selected10, &table10[j], (int64_t)eq);
        }

        /* CT conditional negate */
        ran_affine10_cneg(&selected10, (int64_t)neg);

        /* Mixed addition if digit != 0 */
        nonzero = 1u ^ ((abs_d - 1u) >> 31);

        /* Handle identity accumulator: madd(identity, P) is degenerate.
         * Check if Z==0 (identity) by converting Z back to fp51 temporarily. */
        fp_fe z_check;
        fp10_to_fp51(z_check, accZ);
        unsigned int z_nonzero = (unsigned int)fp_isnonzero(z_check);

        fp10 tmpX, tmpY, tmpZ;
        ran_madd_fp10(tmpX, tmpY, tmpZ, accX, accY, accZ, selected10.x, selected10.y);

        /* from_affine in fp10 */
        fp10 freshX, freshY, freshZ;
        fp10 one10_f = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        fp10_copy(freshX, selected10.x);
        fp10_copy(freshY, selected10.y);
        fp10_copy(freshZ, one10_f);

        /* If digit nonzero and accumulator is valid (Z!=0): use madd result */
        int64_t use_madd = (int64_t)(nonzero & z_nonzero);
        fp10_cmov(accX, tmpX, use_madd);
        fp10_cmov(accY, tmpY, use_madd);
        fp10_cmov(accZ, tmpZ, use_madd);

        /* If digit nonzero and accumulator is identity (Z==0): use from_affine */
        int64_t use_fresh = (int64_t)(nonzero & (1u - z_nonzero));
        fp10_cmov(accX, freshX, use_fresh);
        fp10_cmov(accY, freshY, use_fresh);
        fp10_cmov(accZ, freshZ, use_fresh);
    }

    /* Convert result back to fp51 */
    fp10_to_fp51(r->X, accX);
    fp10_to_fp51(r->Y, accY);
    fp10_to_fp51(r->Z, accZ);

    /* Secure erase */
    ranshaw_secure_erase(table_jac, sizeof(table_jac));
    ranshaw_secure_erase(table, sizeof(table));
    ranshaw_secure_erase(table10, sizeof(table10));
    ranshaw_secure_erase(digits, sizeof(digits));
    ranshaw_secure_erase(&selected10, sizeof(selected10));
    ranshaw_secure_erase(accX, sizeof(accX));
    ranshaw_secure_erase(accY, sizeof(accY));
    ranshaw_secure_erase(accZ, sizeof(accZ));
}
