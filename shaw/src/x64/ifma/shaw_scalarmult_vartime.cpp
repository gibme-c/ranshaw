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
 * AVX-512 IFMA variable-time scalar multiplication for the Shaw curve using fq10
 * (radix-2^25.5) field arithmetic.
 *
 * The key optimization: fq10 uses only 64-bit multiplies (no 128-bit multiply),
 * which is significantly faster on MSVC where 128-bit multiply emulation via
 * uint128_emu causes massive register spilling when force-inlined.
 *
 * The IFMA TU is compiled with AVX-512 flags which imply AVX2, so we can
 * include AVX2 headers (fq10_avx2.h).
 *
 * Algorithm: wNAF with window width 5.
 *   1. Precompute odd multiples [P, 3P, 5P, 7P, 9P, 11P, 13P, 15P]
 *      using fq51 ops, then convert to fq10 Jacobian
 *   2. wNAF-encode scalar with w=5 -> digits in [-15, 15], non-adjacent
 *   3. Scan from MSB to LSB: double, if digit != 0 add/sub precomputed point
 *      All curve ops use inline fq10 point ops
 *   4. Convert result back to fq51
 */

#include "shaw_scalarmult_vartime.h"

#include "fq_ops.h"
#include "ranshaw_secure_erase.h"
#include "shaw.h"
#include "shaw_ops.h"
#include "x64/avx2/fq10_avx2.h"
#include "x64/shaw_add.h"
#include "x64/shaw_dbl.h"

/* ------------------------------------------------------------------ */
/* fq10 Jacobian point type                                           */
/* ------------------------------------------------------------------ */

typedef struct
{
    fq10 X, Y, Z;
} shaw_jacobian_10;

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
/* fq10 general addition — add-2007-bl                                */
/* Cost: 11M + 5S (fq10 ops)                                         */
/* ------------------------------------------------------------------ */

static inline void shaw_add_fq10(
    fq10 rX,
    fq10 rY,
    fq10 rZ,
    const fq10 pX,
    const fq10 pY,
    const fq10 pZ,
    const fq10 qX,
    const fq10 qY,
    const fq10 qZ)
{
    fq10 Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, rr, V, t0, t1;

    fq10_sq(Z1Z1, pZ); /* Z1Z1 = Z1^2 */
    fq10_sq(Z2Z2, qZ); /* Z2Z2 = Z2^2 */
    fq10_mul(U1, pX, Z2Z2); /* U1 = X1 * Z2Z2 */
    fq10_mul(U2, qX, Z1Z1); /* U2 = X2 * Z1Z1 */
    fq10_mul(t0, pY, qZ);
    fq10_mul(S1, t0, Z2Z2); /* S1 = Y1 * Z2 * Z2Z2 */
    fq10_mul(t0, qY, pZ);
    fq10_mul(S2, t0, Z1Z1); /* S2 = Y2 * Z1 * Z1Z1 */
    fq10_sub(H, U2, U1); /* H = U2 - U1 */
    fq10_add(t0, H, H);
    fq10_sq(I, t0); /* I = (2*H)^2 */
    fq10_mul(J, H, I); /* J = H * I */
    fq10_sub(rr, S2, S1);
    fq10_add(rr, rr, rr); /* r = 2*(S2 - S1) */
    fq10_mul(V, U1, I); /* V = U1 * I */

    fq10_sq(rX, rr); /* r^2 */
    fq10_sub(rX, rX, J); /* r^2 - J */
    fq10_add(t0, V, V);
    fq10_sub(rX, rX, t0); /* X3 = r^2 - J - 2*V */

    fq10_sub(t0, V, rX);
    fq10_mul(t1, rr, t0); /* r*(V - X3) */
    fq10_mul(t0, S1, J);
    fq10_add(t0, t0, t0); /* 2*S1*J */
    fq10_sub(rY, t1, t0); /* Y3 = r*(V - X3) - 2*S1*J */

    fq10_add(t0, pZ, qZ);
    fq10_sq(t1, t0); /* (Z1+Z2)^2 */
    fq10_sub(t1, t1, Z1Z1);
    fq10_sub(t1, t1, Z2Z2); /* (Z1+Z2)^2 - Z1Z1 - Z2Z2 */
    fq10_mul(rZ, t1, H); /* Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2) * H */
}

/* ------------------------------------------------------------------ */
/* wNAF encoding                                                      */
/* ------------------------------------------------------------------ */

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
        bits[i / 4] |= (uint32_t)scalar[i] << ((i % 4) * 8);

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

/* ------------------------------------------------------------------ */
/* Entry point                                                        */
/* ------------------------------------------------------------------ */

void shaw_scalarmult_vartime_ifma(shaw_jacobian *r, const unsigned char scalar[32], const shaw_jacobian *p)
{
    /* Precompute odd multiples [P, 3P, 5P, 7P, 9P, 11P, 13P, 15P] using fq51 */
    shaw_jacobian table_jac[8];
    shaw_jacobian p2;

    shaw_copy(&table_jac[0], p); /* 1P */
    shaw_dbl_x64(&p2, p); /* 2P */

    for (int i = 1; i < 8; i++)
        shaw_add_x64(&table_jac[i], &table_jac[i - 1], &p2); /* (2i+1)P */

    /* Convert precomputed table to fq10 */
    shaw_jacobian_10 table10[8];
    for (int i = 0; i < 8; i++)
    {
        fq51_to_fq10(table10[i].X, table_jac[i].X);
        fq51_to_fq10(table10[i].Y, table_jac[i].Y);
        fq51_to_fq10(table10[i].Z, table_jac[i].Z);
    }

    /* wNAF encode */
    int8_t naf[257];
    int top = wnaf_encode(naf, scalar);

    if (top == 0)
    {
        ranshaw_secure_erase(naf, sizeof(naf));
        ranshaw_secure_erase(table_jac, sizeof(table_jac));
        ranshaw_secure_erase(table10, sizeof(table10));
        ranshaw_secure_erase(&p2, sizeof(p2));
        shaw_identity(r);
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
        shaw_identity(r);
        return;
    }

    /* Initialize accumulator with the highest nonzero digit's point (fq10) */
    fq10 accX, accY, accZ;
    int8_t d = naf[start];
    int idx = ((d < 0) ? -d : d) / 2; /* table index: |d|/2 since table stores odd multiples */
    fq10_copy(accX, table10[idx].X);
    fq10_copy(accY, table10[idx].Y);
    fq10_copy(accZ, table10[idx].Z);
    if (d < 0)
    {
        fq10_neg(accY, accY);
    }

    /* Main loop */
    for (int i = start - 1; i >= 0; i--)
    {
        shaw_dbl_fq10(accX, accY, accZ, accX, accY, accZ);

        if (naf[i] != 0)
        {
            d = naf[i];
            idx = ((d < 0) ? -d : d) / 2;
            if (d > 0)
            {
                shaw_add_fq10(accX, accY, accZ, accX, accY, accZ, table10[idx].X, table10[idx].Y, table10[idx].Z);
            }
            else
            {
                fq10 neg_y;
                fq10_neg(neg_y, table10[idx].Y);
                shaw_add_fq10(accX, accY, accZ, accX, accY, accZ, table10[idx].X, neg_y, table10[idx].Z);
            }
        }
    }

    /* Convert result back to fq51 */
    fq10_to_fq51(r->X, accX);
    fq10_to_fq51(r->Y, accY);
    fq10_to_fq51(r->Z, accZ);

    ranshaw_secure_erase(naf, sizeof(naf));
    ranshaw_secure_erase(table_jac, sizeof(table_jac));
    ranshaw_secure_erase(table10, sizeof(table10));
    ranshaw_secure_erase(&p2, sizeof(p2));
}
