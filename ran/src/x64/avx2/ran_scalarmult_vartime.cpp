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
 * AVX2 variable-time scalar multiplication for the Ran curve using fp10
 * (radix-2^25.5) field arithmetic.
 *
 * The key optimization: fp10 uses only 64-bit multiplies (no 128-bit multiply),
 * which is significantly faster on MSVC where 128-bit multiply emulation via
 * uint128_emu causes massive register spilling when force-inlined.
 *
 * Algorithm: wNAF with window width 5.
 *   1. Precompute odd multiples [P, 3P, 5P, ..., 15P] using fp51 ops
 *   2. Convert precomputed table to fp10 Jacobian
 *   3. wNAF-encode scalar (w=5) -> digits in [-15, 15], non-adjacent
 *   4. Find highest nonzero digit, initialize accumulator in fp10
 *   5. Main loop (MSB to LSB): fp10 doubling, if digit != 0: fp10 general
 *      addition or subtraction
 *   6. Convert result back to fp51
 */

#include "ran_scalarmult_vartime.h"

#include "fp_ops.h"
#include "ran.h"
#include "ran_ops.h"
#include "ranshaw_secure_erase.h"
#include "x64/avx2/fp10_avx2.h"
#include "x64/ran_add.h"
#include "x64/ran_dbl.h"

/* ------------------------------------------------------------------ */
/* fp10 Jacobian point type                                           */
/* ------------------------------------------------------------------ */

typedef struct
{
    fp10 X, Y, Z;
} ran_jacobian_10;

/* ------------------------------------------------------------------ */
/* fp10 point doubling — dbl-2001-b, a = -3                          */
/* Cost: 3M + 4S (fp10 ops)                                          */
/* ------------------------------------------------------------------ */

static inline void ran_dbl_fp10(fp10 rX, fp10 rY, fp10 rZ, const fp10 pX, const fp10 pY, const fp10 pZ)
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
/* fp10 general addition — add-2007-bl                                */
/* Cost: 11M + 5S (fp10 ops)                                         */
/* ------------------------------------------------------------------ */

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

    fp10_sq(Z1Z1, pZ);
    fp10_sq(Z2Z2, qZ);
    fp10_mul(U1, pX, Z2Z2);
    fp10_mul(U2, qX, Z1Z1);
    fp10_mul(t0, qZ, Z2Z2);
    fp10_mul(S1, pY, t0);
    fp10_mul(t0, pZ, Z1Z1);
    fp10_mul(S2, qY, t0);
    fp10_sub(H, U2, U1);
    fp10_add(t0, H, H);
    fp10_sq(I, t0);
    fp10_mul(J, H, I);
    fp10_sub(rr, S2, S1);
    fp10_add(rr, rr, rr);
    fp10_mul(V, U1, I);

    fp10_sq(rX, rr);
    fp10_sub(rX, rX, J);
    fp10_add(t0, V, V);
    fp10_sub(rX, rX, t0);

    fp10_sub(t0, V, rX);
    fp10_mul(t1, rr, t0);
    fp10_mul(t0, S1, J);
    fp10_add(t0, t0, t0);
    fp10_sub(rY, t1, t0);

    fp10_add(t0, pZ, qZ);
    fp10_sq(t1, t0);
    fp10_sub(t1, t1, Z1Z1);
    fp10_sub(t1, t1, Z2Z2);
    fp10_mul(rZ, t1, H);
}

/* ------------------------------------------------------------------ */
/* fp10 point negation                                                */
/* ------------------------------------------------------------------ */

static inline void ran_neg_fp10(ran_jacobian_10 *r, const ran_jacobian_10 *p)
{
    fp10_copy(r->X, p->X);
    fp10_neg(r->Y, p->Y);
    fp10_copy(r->Z, p->Z);
}

/* ------------------------------------------------------------------ */
/* wNAF encoding with window width w=5                                */
/* ------------------------------------------------------------------ */

/*
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

/* ------------------------------------------------------------------ */
/* Entry point                                                        */
/* ------------------------------------------------------------------ */

void ran_scalarmult_vartime_avx2(ran_jacobian *r, const unsigned char scalar[32], const ran_jacobian *p)
{
    /* Precompute odd multiples [P, 3P, 5P, 7P, 9P, 11P, 13P, 15P] using fp51 */
    ran_jacobian table_jac[8];
    ran_jacobian p2;

    ran_copy(&table_jac[0], p); /* 1P */
    ran_dbl_x64(&p2, p); /* 2P */

    for (int i = 1; i < 8; i++)
        ran_add_x64(&table_jac[i], &table_jac[i - 1], &p2); /* (2i+1)P */

    /* Convert precomputed table to fp10 Jacobian */
    ran_jacobian_10 table10[8];
    for (int i = 0; i < 8; i++)
    {
        fp51_to_fp10(table10[i].X, table_jac[i].X);
        fp51_to_fp10(table10[i].Y, table_jac[i].Y);
        fp51_to_fp10(table10[i].Z, table_jac[i].Z);
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

    /* Initialize accumulator in fp10 with the highest nonzero digit's point */
    fp10 accX, accY, accZ;
    int8_t d = naf[start];
    int idx = ((d < 0) ? -d : d) / 2; /* table index: |d|/2 since table stores odd multiples */
    fp10_copy(accX, table10[idx].X);
    fp10_copy(accY, table10[idx].Y);
    fp10_copy(accZ, table10[idx].Z);
    if (d < 0)
    {
        fp10_neg(accY, accY);
    }

    /* Main loop */
    for (int i = start - 1; i >= 0; i--)
    {
        ran_dbl_fp10(accX, accY, accZ, accX, accY, accZ);

        if (naf[i] != 0)
        {
            d = naf[i];
            idx = ((d < 0) ? -d : d) / 2;
            if (d > 0)
            {
                ran_add_fp10(accX, accY, accZ, accX, accY, accZ, table10[idx].X, table10[idx].Y, table10[idx].Z);
            }
            else
            {
                ran_jacobian_10 neg_pt;
                ran_neg_fp10(&neg_pt, &table10[idx]);
                ran_add_fp10(accX, accY, accZ, accX, accY, accZ, neg_pt.X, neg_pt.Y, neg_pt.Z);
            }
        }
    }

    /* Convert result back to fp51 */
    fp10_to_fp51(r->X, accX);
    fp10_to_fp51(r->Y, accY);
    fp10_to_fp51(r->Z, accZ);

    ranshaw_secure_erase(naf, sizeof(naf));
    ranshaw_secure_erase(table_jac, sizeof(table_jac));
    ranshaw_secure_erase(table10, sizeof(table10));
    ranshaw_secure_erase(&p2, sizeof(p2));
}
