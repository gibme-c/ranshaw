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
 * IFMA (AVX-512) constant-time scalar multiplication for Ran.
 *
 * For single-scalar operations there is no benefit to 8-way IFMA parallelism.
 * Instead we use scalar fp10 (radix-2^25.5) field arithmetic -- the same
 * approach as the AVX2 backend. This avoids 128-bit multiply overhead and is
 * genuinely faster than the x64 baseline on MSVC, where _umul128 results
 * returned through uint128_emu structs cause heavy register spilling.
 *
 * The IFMA TU is compiled with -mavx512f -mavx512ifma (GCC/Clang) or
 * /arch:AVX512 (MSVC), which implies AVX2 support, so we can include AVX2
 * headers for fp10 operations.
 *
 * Algorithm: signed 4-bit fixed-window (radix-16).
 *   1. Precompute [P, 2P, ..., 8P] using fp51 Jacobian ops
 *   2. Batch to affine (single inversion)
 *   3. Convert affine table to fp10
 *   4. Recode scalar to 64 signed 4-bit digits
 *   5. Main loop: dbl/madd using inline fp10 ops, CT table lookup
 *   6. Convert result back to fp51
 *   7. Secure erase
 */

#include "ran_scalarmult.h"

#include "fp_invert.h"
#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "fp_utils.h"
#include "ran.h"
#include "ran_ops.h"
#include "ranshaw_secure_erase.h"
#include "x64/avx2/fp10_avx2.h"
#include "x64/ran_add.h"
#include "x64/ran_dbl.h"

#include <vector>

/* ---- Types ---- */

typedef struct
{
    fp10 x, y;
} ran_affine_10;

/* ---- Scalar recoding ---- */

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

/* ---- Batch affine conversion ---- */

/*
 * Batch affine conversion using Montgomery's trick.
 * Converts n Jacobian points to affine using a single inversion.
 */
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
        fp_mul(z_inv, inv, products[i - 1].v); /* z_inv = inv * products[i-1] = 1/z[i] */
        fp_mul(inv, inv, z_vals[i].v); /* inv = inv * z[i] = 1/(z[0]*...*z[i-1]) */

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

/* ---- Inline fp10 point doubling (a=-3, dbl-2001-b) ---- */

/*
 * Point doubling on y^2 = x^3 - 3x + b using Jacobian coordinates.
 * Formula: dbl-2001-b (3M + 4S, exploiting a = -3).
 *
 *   delta = Z^2
 *   gamma = Y^2
 *   beta = X * gamma
 *   alpha = 3 * (X - delta) * (X + delta)
 *   X3 = alpha^2 - 8*beta
 *   Z3 = (Y + Z)^2 - gamma - delta
 *   Y3 = alpha * (4*beta - X3) - 8*gamma^2
 */
static inline void ran_dbl_fp10(fp10 rX, fp10 rY, fp10 rZ, const fp10 pX, const fp10 pY, const fp10 pZ)
{
    fp10 delta, gamma, beta, alpha, t0, t1, t2;

    fp10_sq(delta, pZ); /* delta = Z^2 */
    fp10_sq(gamma, pY); /* gamma = Y^2 */
    fp10_mul(beta, pX, gamma); /* beta = X * gamma */

    fp10_sub(t0, pX, delta); /* t0 = X - delta */
    fp10_add(t1, pX, delta); /* t1 = X + delta */
    fp10_mul(alpha, t0, t1); /* alpha = (X - delta)(X + delta) */
    fp10_add(t0, alpha, alpha); /* t0 = 2 * alpha */
    fp10_add(alpha, t0, alpha); /* alpha = 3 * (X - delta)(X + delta) */

    fp10_sq(rX, alpha); /* rX = alpha^2 */
    fp10_add(t0, beta, beta); /* t0 = 2*beta */
    fp10_add(t0, t0, t0); /* t0 = 4*beta */
    fp10_sub(rX, rX, t0); /* rX = alpha^2 - 4*beta */
    fp10_sub(rX, rX, t0); /* rX = alpha^2 - 8*beta */

    fp10_add(t1, pY, pZ); /* t1 = Y + Z */
    fp10_sq(t2, t1); /* t2 = (Y + Z)^2 */
    fp10_sub(t2, t2, gamma); /* t2 = (Y+Z)^2 - gamma */
    fp10_sub(rZ, t2, delta); /* rZ = (Y+Z)^2 - gamma - delta */

    fp10_sub(t1, t0, rX); /* t1 = 4*beta - X3 */
    fp10_mul(t2, alpha, t1); /* t2 = alpha * (4*beta - X3) */
    fp10_sq(t0, gamma); /* t0 = gamma^2 */
    fp10_add(t0, t0, t0); /* t0 = 2*gamma^2 */
    fp10_add(t0, t0, t0); /* t0 = 4*gamma^2 */
    fp10_sub(rY, t2, t0); /* rY = alpha*(4*beta - X3) - 4*gamma^2 */
    fp10_sub(rY, rY, t0); /* rY = alpha*(4*beta - X3) - 8*gamma^2 */
}

/* ---- Inline fp10 mixed addition (madd-2007-bl) ---- */

/*
 * Mixed addition: Jacobian + affine -> Jacobian.
 * Formula: madd-2007-bl (7M + 4S).
 *
 *   Z1Z1 = Z1^2
 *   U2 = X2 * Z1Z1
 *   S2 = Y2 * Z1 * Z1Z1
 *   H = U2 - X1
 *   HH = H^2
 *   I = 4 * HH
 *   J = H * I
 *   r = 2 * (S2 - Y1)
 *   V = X1 * I
 *   X3 = r^2 - J - 2*V
 *   Y3 = r * (V - X3) - 2*Y1*J
 *   Z3 = (Z1 + H)^2 - Z1Z1 - HH
 */
static inline void
    ran_madd_fp10(fp10 rX, fp10 rY, fp10 rZ, const fp10 pX, const fp10 pY, const fp10 pZ, const fp10 qx, const fp10 qy)
{
    fp10 Z1Z1, U2, S2, H, HH, I, J, rr, V, t0, t1;

    fp10_sq(Z1Z1, pZ); /* Z1Z1 = Z1^2 */
    fp10_mul(U2, qx, Z1Z1); /* U2 = X2 * Z1Z1 */
    fp10_mul(t0, pZ, Z1Z1); /* t0 = Z1 * Z1Z1 = Z1^3 */
    fp10_mul(S2, qy, t0); /* S2 = Y2 * Z1^3 */

    fp10_sub(H, U2, pX); /* H = U2 - X1 */
    fp10_sq(HH, H); /* HH = H^2 */
    fp10_add(I, HH, HH); /* I = 2*HH */
    fp10_add(I, I, I); /* I = 4*HH */
    fp10_mul(J, H, I); /* J = H * I */

    fp10_sub(rr, S2, pY); /* rr = S2 - Y1 */
    fp10_add(rr, rr, rr); /* rr = 2*(S2 - Y1) */

    fp10_mul(V, pX, I); /* V = X1 * I */

    fp10_sq(rX, rr); /* X3 = r^2 */
    fp10_sub(rX, rX, J); /* X3 = r^2 - J */
    fp10_add(t0, V, V); /* t0 = 2*V */
    fp10_sub(rX, rX, t0); /* X3 = r^2 - J - 2*V */

    fp10_sub(t0, V, rX); /* t0 = V - X3 */
    fp10_mul(t1, rr, t0); /* t1 = r * (V - X3) */
    fp10_mul(t0, pY, J); /* t0 = Y1 * J */
    fp10_add(t0, t0, t0); /* t0 = 2 * Y1 * J */
    fp10_sub(rY, t1, t0); /* Y3 = r*(V - X3) - 2*Y1*J */

    fp10_add(t0, pZ, H); /* t0 = Z1 + H */
    fp10_sq(t1, t0); /* t1 = (Z1 + H)^2 */
    fp10_sub(t1, t1, Z1Z1); /* t1 = (Z1+H)^2 - Z1Z1 */
    fp10_sub(rZ, t1, HH); /* Z3 = (Z1+H)^2 - Z1Z1 - HH */
}

/* ---- CT helpers ---- */

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

/* ---- fp10 zero / one / isnonzero ---- */

static inline void fp10_set0(fp10 h)
{
    h[0] = h[1] = h[2] = h[3] = h[4] = 0;
    h[5] = h[6] = h[7] = h[8] = h[9] = 0;
}

static inline void fp10_set1(fp10 h)
{
    h[0] = 1;
    h[1] = h[2] = h[3] = h[4] = 0;
    h[5] = h[6] = h[7] = h[8] = h[9] = 0;
}

/*
 * CT check if fp10 element is nonzero (mod p). Returns 1 if nonzero, 0 if zero.
 * Used to detect identity (Z == 0).
 *
 * Cannot simply OR the limbs: fp10_sub(x, x) produces p (a non-canonical
 * representation of 0 with all-nonzero limbs). We must fully reduce through
 * fp_tobytes via fp_isnonzero.
 */
static inline unsigned int fp10_isnonzero_ct(const fp10 f)
{
    fp_fe tmp;
    fp10_to_fp51(tmp, f);
    return (unsigned int)fp_isnonzero(tmp);
}

/* ---- Main function ---- */

void ran_scalarmult_ifma(ran_jacobian *r, const unsigned char scalar[32], const ran_jacobian *p)
{
    /* Step 1: Precompute table [P, 2P, 3P, 4P, 5P, 6P, 7P, 8P] using fp51 ops */
    ran_jacobian table_jac[8];
    ran_copy(&table_jac[0], p); /* 1P */
    ran_dbl_x64(&table_jac[1], p); /* 2P */
    ran_add_x64(&table_jac[2], &table_jac[1], p); /* 3P */
    ran_dbl_x64(&table_jac[3], &table_jac[1]); /* 4P */
    ran_add_x64(&table_jac[4], &table_jac[3], p); /* 5P */
    ran_dbl_x64(&table_jac[5], &table_jac[2]); /* 6P */
    ran_add_x64(&table_jac[6], &table_jac[5], p); /* 7P */
    ran_dbl_x64(&table_jac[7], &table_jac[3]); /* 8P */

    /* Step 2: Convert to affine (single inversion) */
    ran_affine table_affine[8];
    batch_to_affine(table_affine, table_jac, 8);

    /* Step 3: Convert affine table to fp10 */
    ran_affine_10 table10[8];
    for (int i = 0; i < 8; i++)
    {
        fp51_to_fp10(table10[i].x, table_affine[i].x);
        fp51_to_fp10(table10[i].y, table_affine[i].y);
    }

    /* Step 4: Recode scalar */
    int8_t digits[64];
    scalar_recode_signed4(digits, scalar);

    /* Step 5: Main loop -- start from the top digit */
    fp10 rX, rY, rZ;

    /* Initialize with the top digit (branchless abs + sign extraction) */
    int32_t d = (int32_t)digits[63];
    int32_t sign_mask = -(int32_t)((uint32_t)d >> 31);
    unsigned int abs_d = (unsigned int)((d ^ sign_mask) - sign_mask);
    unsigned int neg = (unsigned int)(sign_mask & 1);

    /* CT table lookup for initial value */
    ran_affine_10 selected;
    fp10_set0(selected.x);
    fp10_set0(selected.y);

    for (unsigned int j = 0; j < 8; j++)
    {
        unsigned int eq = ((abs_d ^ (j + 1)) - 1u) >> 31;
        ran_affine10_cmov(&selected, &table10[j], eq);
    }

    /* CT conditional negate */
    ran_affine10_cneg(&selected, neg);

    /* Always compute both paths, then CT select */
    /* from_table: (x:y:1) */
    fp10 tableX, tableY, tableZ;
    fp10_copy(tableX, selected.x);
    fp10_copy(tableY, selected.y);
    fp10_set1(tableZ);

    /* identity: (1:1:0) */
    fp10 identX, identY, identZ;
    fp10_set1(identX);
    fp10_set1(identY);
    fp10_set0(identZ);

    /* CT select: identity if abs_d == 0, from_table otherwise */
    unsigned int nonzero = 1u ^ ((abs_d - 1u) >> 31);
    fp10_copy(rX, identX);
    fp10_copy(rY, identY);
    fp10_copy(rZ, identZ);
    fp10_cmov(rX, tableX, (int64_t)nonzero);
    fp10_cmov(rY, tableY, (int64_t)nonzero);
    fp10_cmov(rZ, tableZ, (int64_t)nonzero);

    /* Main loop: digits[62] down to digits[0] */
    for (int i = 62; i >= 0; i--)
    {
        /* 4 doublings */
        ran_dbl_fp10(rX, rY, rZ, rX, rY, rZ);
        ran_dbl_fp10(rX, rY, rZ, rX, rY, rZ);
        ran_dbl_fp10(rX, rY, rZ, rX, rY, rZ);
        ran_dbl_fp10(rX, rY, rZ, rX, rY, rZ);

        /* Extract digit (branchless) */
        d = (int32_t)digits[i];
        sign_mask = -(int32_t)((uint32_t)d >> 31);
        abs_d = (unsigned int)((d ^ sign_mask) - sign_mask);
        neg = (unsigned int)(sign_mask & 1);

        /* CT table lookup */
        fp10_set1(selected.x);
        fp10_set1(selected.y);
        for (unsigned int j = 0; j < 8; j++)
        {
            unsigned int eq = ((abs_d ^ (j + 1)) - 1u) >> 31;
            ran_affine10_cmov(&selected, &table10[j], eq);
        }

        /* CT conditional negate */
        ran_affine10_cneg(&selected, neg);

        /* Mixed addition if digit != 0 */
        nonzero = 1u ^ ((abs_d - 1u) >> 31);

        /* Handle identity accumulator: madd(identity, P) is degenerate.
         * If Z==0 (identity), use from_affine instead. */
        unsigned int z_nonzero = fp10_isnonzero_ct(rZ);

        fp10 tmpX, tmpY, tmpZ;
        ran_madd_fp10(tmpX, tmpY, tmpZ, rX, rY, rZ, selected.x, selected.y);

        fp10 freshX, freshY, freshZ;
        fp10_copy(freshX, selected.x);
        fp10_copy(freshY, selected.y);
        fp10_set1(freshZ);

        /* If digit nonzero and accumulator is valid (Z!=0): use madd result */
        int64_t use_madd = (int64_t)(nonzero & z_nonzero);
        fp10_cmov(rX, tmpX, use_madd);
        fp10_cmov(rY, tmpY, use_madd);
        fp10_cmov(rZ, tmpZ, use_madd);

        /* If digit nonzero and accumulator is identity (Z==0): use from_affine */
        int64_t use_fresh = (int64_t)(nonzero & (1u - z_nonzero));
        fp10_cmov(rX, freshX, use_fresh);
        fp10_cmov(rY, freshY, use_fresh);
        fp10_cmov(rZ, freshZ, use_fresh);
    }

    /* Step 6: Convert result back to fp51 */
    fp10_to_fp51(r->X, rX);
    fp10_to_fp51(r->Y, rY);
    fp10_to_fp51(r->Z, rZ);

    /* Step 7: Secure erase */
    ranshaw_secure_erase(table_jac, sizeof(table_jac));
    ranshaw_secure_erase(table_affine, sizeof(table_affine));
    ranshaw_secure_erase(table10, sizeof(table10));
    ranshaw_secure_erase(digits, sizeof(digits));
    ranshaw_secure_erase(rX, sizeof(rX));
    ranshaw_secure_erase(rY, sizeof(rY));
    ranshaw_secure_erase(rZ, sizeof(rZ));
}
