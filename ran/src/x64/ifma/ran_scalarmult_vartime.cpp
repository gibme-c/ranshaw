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
 * IFMA (AVX-512) variable-time scalar multiplication for Ran.
 *
 * For single-scalar operations there is no benefit to 8-way IFMA parallelism.
 * Instead we use scalar fp10 (radix-2^25.5) field arithmetic -- the same
 * approach as the AVX2 backend. This avoids 128-bit multiply overhead and is
 * genuinely faster than the x64 baseline on MSVC.
 *
 * The IFMA TU is compiled with -mavx512f -mavx512ifma (GCC/Clang) or
 * /arch:AVX512 (MSVC), which implies AVX2 support, so we can include AVX2
 * headers for fp10 operations.
 *
 * Algorithm: wNAF with window width w=5.
 *   1. Precompute odd multiples [P, 3P, 5P, ..., 15P] using fp51 Jacobian ops
 *   2. Convert to fp10 Jacobian table
 *   3. wNAF-encode scalar
 *   4. Main loop: dbl/add using inline fp10 ops (general addition, not mixed)
 *   5. Convert result back to fp51
 */

#include "ran_scalarmult_vartime.h"

#include "fp_ops.h"
#include "ran.h"
#include "ran_ops.h"
#include "ranshaw_secure_erase.h"
#include "x64/avx2/fp10_avx2.h"
#include "x64/ran_add.h"
#include "x64/ran_dbl.h"

/* ---- Types ---- */

typedef struct
{
    fp10 X, Y, Z;
} ran_jacobian_10;

/* ---- wNAF encoding ---- */

/*
 * wNAF encoding with window width w=5.
 * Output: naf[257] with values in {-15,-13,...,-1,0,1,...,13,15}
 * Returns the position of the highest nonzero digit + 1.
 */
static int wnaf_encode(int8_t naf[257], const unsigned char scalar[32])
{
    /* Convert scalar to a mutable array of bits */
    uint32_t bits[9] = {0};
    for (int i = 0; i < 32; i++)
    {
        bits[i / 4] |= (uint32_t)scalar[i] << ((i % 4) * 8);
    }

    int pos = 0;
    int highest = 0;

    for (int i = 0; i <= 256; i++)
        naf[i] = 0;

    while (pos <= 256)
    {
        /* Get current bit */
        if (!((bits[pos / 32] >> (pos % 32)) & 1))
        {
            pos++;
            continue;
        }

        /* Extract w bits starting at pos */
        int word_idx = pos / 32;
        int bit_idx = pos % 32;
        int32_t val = (int32_t)((bits[word_idx] >> bit_idx) & 0x1f);
        if (bit_idx > 27 && word_idx + 1 < 9)
            val |= (int32_t)((bits[word_idx + 1] << (32 - bit_idx)) & 0x1f);

        if (val > 16)
            val -= 32;

        naf[pos] = (int8_t)val;
        highest = pos + 1;

        /* Zero out the w bits we just consumed by subtracting val << pos.
         * Must propagate borrows/carries across word boundaries. */
        {
            int wi = pos / 32;
            int bi = pos % 32;
            if (val > 0)
            {
                uint64_t sub = (uint64_t)(uint32_t)val << bi;
                uint32_t borrow = 0;
                for (int k = wi; k < 9 && (sub || borrow); k++)
                {
                    uint64_t lo = (k == wi) ? (sub & 0xffffffffULL) : ((k == wi + 1) ? (sub >> 32) : 0);
                    lo += borrow;
                    borrow = (bits[k] < lo) ? 1 : 0;
                    bits[k] -= (uint32_t)lo;
                }
            }
            else
            {
                uint64_t add = (uint64_t)(uint32_t)(-val) << bi;
                uint32_t carry = 0;
                for (int k = wi; k < 9 && (add || carry); k++)
                {
                    uint64_t lo = (k == wi) ? (add & 0xffffffffULL) : ((k == wi + 1) ? (add >> 32) : 0);
                    uint64_t sum = (uint64_t)bits[k] + lo + carry;
                    bits[k] = (uint32_t)sum;
                    carry = (uint32_t)(sum >> 32);
                }
            }
        }

        pos += 5; /* wNAF guarantees next w-1 digits are 0 */
    }

    ranshaw_secure_erase(bits, sizeof(bits));
    return highest;
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

/* ---- Inline fp10 general addition (add-2007-bl, 11M + 5S) ---- */

/*
 * General addition: Jacobian + Jacobian -> Jacobian.
 * Formula: add-2007-bl (11M + 5S).
 *
 *   Z1Z1 = Z1^2,  Z2Z2 = Z2^2
 *   U1 = X1 * Z2Z2,  U2 = X2 * Z1Z1
 *   S1 = Y1 * Z2 * Z2Z2,  S2 = Y2 * Z1 * Z1Z1
 *   H = U2 - U1
 *   I = (2*H)^2
 *   J = H * I
 *   r = 2 * (S2 - S1)
 *   V = U1 * I
 *   X3 = r^2 - J - 2*V
 *   Y3 = r * (V - X3) - 2*S1*J
 *   Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
 */
static inline void ran_add_fp10(
    fp10 rX,
    fp10 rY,
    fp10 rZ,
    const fp10 pX,
    const fp10 pY,
    const fp10 pZ,
    const fp10 qX,
    const fp10 qY,
    const fp10 qZ)
{
    fp10 Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, rr, V, t0, t1;

    fp10_sq(Z1Z1, pZ); /* Z1Z1 = Z1^2 */
    fp10_sq(Z2Z2, qZ); /* Z2Z2 = Z2^2 */

    fp10_mul(U1, pX, Z2Z2); /* U1 = X1 * Z2Z2 */
    fp10_mul(U2, qX, Z1Z1); /* U2 = X2 * Z1Z1 */

    fp10_mul(t0, qZ, Z2Z2); /* t0 = Z2 * Z2Z2 = Z2^3 */
    fp10_mul(S1, pY, t0); /* S1 = Y1 * Z2^3 */
    fp10_mul(t0, pZ, Z1Z1); /* t0 = Z1 * Z1Z1 = Z1^3 */
    fp10_mul(S2, qY, t0); /* S2 = Y2 * Z1^3 */

    fp10_sub(H, U2, U1); /* H = U2 - U1 */
    fp10_add(t0, H, H); /* t0 = 2*H */
    fp10_sq(I, t0); /* I = (2*H)^2 */
    fp10_mul(J, H, I); /* J = H * I */

    fp10_sub(rr, S2, S1); /* rr = S2 - S1 */
    fp10_add(rr, rr, rr); /* rr = 2*(S2 - S1) */

    fp10_mul(V, U1, I); /* V = U1 * I */

    fp10_sq(rX, rr); /* X3 = r^2 */
    fp10_sub(rX, rX, J); /* X3 = r^2 - J */
    fp10_add(t0, V, V); /* t0 = 2*V */
    fp10_sub(rX, rX, t0); /* X3 = r^2 - J - 2*V */

    fp10_sub(t0, V, rX); /* t0 = V - X3 */
    fp10_mul(t1, rr, t0); /* t1 = r * (V - X3) */
    fp10_mul(t0, S1, J); /* t0 = S1 * J */
    fp10_add(t0, t0, t0); /* t0 = 2 * S1 * J */
    fp10_sub(rY, t1, t0); /* Y3 = r*(V - X3) - 2*S1*J */

    fp10_add(t0, pZ, qZ); /* t0 = Z1 + Z2 */
    fp10_sq(t1, t0); /* t1 = (Z1 + Z2)^2 */
    fp10_sub(t1, t1, Z1Z1); /* t1 = (Z1+Z2)^2 - Z1Z1 */
    fp10_sub(t1, t1, Z2Z2); /* t1 = (Z1+Z2)^2 - Z1Z1 - Z2Z2 */
    fp10_mul(rZ, t1, H); /* Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2) * H */
}

/* ---- fp10 negation helper for Jacobian ---- */

static inline void ran_neg_fp10(ran_jacobian_10 *r, const ran_jacobian_10 *p)
{
    fp10_copy(r->X, p->X);
    fp10_neg(r->Y, p->Y);
    fp10_copy(r->Z, p->Z);
}

/* ---- Jacobian fp51 to fp10 conversion ---- */

static inline void ran_jac_to_fp10(ran_jacobian_10 *out, const ran_jacobian *in)
{
    fp51_to_fp10(out->X, in->X);
    fp51_to_fp10(out->Y, in->Y);
    fp51_to_fp10(out->Z, in->Z);
}

/* ---- Main function ---- */

void ran_scalarmult_vartime_ifma(ran_jacobian *r, const unsigned char scalar[32], const ran_jacobian *p)
{
    /* Step 1: Precompute odd multiples [P, 3P, 5P, 7P, 9P, 11P, 13P, 15P] using fp51 ops */
    ran_jacobian table_jac[8];
    ran_jacobian p2;

    ran_copy(&table_jac[0], p); /* 1P */
    ran_dbl_x64(&p2, p); /* 2P */

    for (int i = 1; i < 8; i++)
        ran_add_x64(&table_jac[i], &table_jac[i - 1], &p2); /* (2i+1)P */

    /* Step 2: Convert to fp10 Jacobian table */
    ran_jacobian_10 table10[8];
    for (int i = 0; i < 8; i++)
        ran_jac_to_fp10(&table10[i], &table_jac[i]);

    /* Step 3: wNAF encode */
    int8_t naf[257];
    int top = wnaf_encode(naf, scalar);

    if (top == 0)
    {
        ranshaw_secure_erase(naf, sizeof(naf));
        ranshaw_secure_erase(table_jac, sizeof(table_jac));
        ranshaw_secure_erase(table10, sizeof(table10));
        ranshaw_secure_erase(&p2, sizeof(p2));
        ran_identity(r);
        return;
    }

    /* Find the highest nonzero digit to start */
    int start = top - 1;
    while (start >= 0 && naf[start] == 0)
        start--;

    if (start < 0)
    {
        ranshaw_secure_erase(naf, sizeof(naf));
        ranshaw_secure_erase(table_jac, sizeof(table_jac));
        ranshaw_secure_erase(table10, sizeof(table10));
        ranshaw_secure_erase(&p2, sizeof(p2));
        ran_identity(r);
        return;
    }

    /* Initialize with the highest nonzero digit's point */
    fp10 rX, rY, rZ;
    int8_t d = naf[start];
    int idx = ((d < 0) ? -d : d) / 2; /* table index: |d|/2 since table stores odd multiples */
    fp10_copy(rX, table10[idx].X);
    fp10_copy(rY, table10[idx].Y);
    fp10_copy(rZ, table10[idx].Z);
    if (d < 0)
    {
        fp10_neg(rY, rY);
    }

    /* Step 4: Main loop */
    for (int i = start - 1; i >= 0; i--)
    {
        ran_dbl_fp10(rX, rY, rZ, rX, rY, rZ);

        if (naf[i] != 0)
        {
            d = naf[i];
            idx = ((d < 0) ? -d : d) / 2;
            if (d > 0)
            {
                ran_add_fp10(rX, rY, rZ, rX, rY, rZ, table10[idx].X, table10[idx].Y, table10[idx].Z);
            }
            else
            {
                ran_jacobian_10 neg_pt;
                ran_neg_fp10(&neg_pt, &table10[idx]);
                ran_add_fp10(rX, rY, rZ, rX, rY, rZ, neg_pt.X, neg_pt.Y, neg_pt.Z);
            }
        }
    }

    /* Step 5: Convert result back to fp51 */
    fp10_to_fp51(r->X, rX);
    fp10_to_fp51(r->Y, rY);
    fp10_to_fp51(r->Z, rZ);

    ranshaw_secure_erase(naf, sizeof(naf));
    ranshaw_secure_erase(table_jac, sizeof(table_jac));
    ranshaw_secure_erase(table10, sizeof(table10));
    ranshaw_secure_erase(&p2, sizeof(p2));
}
