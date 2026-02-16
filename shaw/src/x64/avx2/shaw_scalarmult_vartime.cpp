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
 * AVX2 variable-time scalar multiplication for Shaw (over F_q).
 *
 * Uses radix-2^25.5 (fq10) field arithmetic throughout the main loop
 * to avoid 128-bit integer arithmetic (MSVC _umul128 register spilling).
 * Converts fq51 -> fq10 once at entry, fq10 -> fq51 once at exit.
 *
 * Algorithm: wNAF with window width 5, same as shaw_scalarmult_vartime_x64
 * but with inline fq10 point doubling (dbl-2001-b, a=-3) and general
 * addition (add-2007-bl).
 */

#include "shaw_scalarmult_vartime.h"

#include "fq_ops.h"
#include "ranshaw_secure_erase.h"
#include "shaw.h"
#include "shaw_ops.h"
#include "x64/avx2/fq10_avx2.h"
#include "x64/shaw_add.h"
#include "x64/shaw_dbl.h"

/* ── fq10 Jacobian point type ── */

typedef struct
{
    fq10 X, Y, Z;
} shaw_jacobian_10;

/* ── wNAF encoding ── */

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

/* ── Inline fq10 point doubling: dbl-2001-b with a = -3 ── */

/*
 * Jacobian doubling with a = -3 optimization (same formula as shaw_dbl_x64).
 * Cost: 3M + 5S (in fq10 arithmetic)
 */
static FQ10_AVX2_FORCE_INLINE void shaw_dbl_fq10(shaw_jacobian_10 *r, const shaw_jacobian_10 *p)
{
    fq10 delta, gamma, beta, alpha;
    fq10 t0, t1, t2;

    /* delta = Z1^2 */
    fq10_sq(delta, p->Z);

    /* gamma = Y1^2 */
    fq10_sq(gamma, p->Y);

    /* beta = X1 * gamma */
    fq10_mul(beta, p->X, gamma);

    /* alpha = 3 * (X1 - delta) * (X1 + delta) */
    fq10_sub(t0, p->X, delta);
    fq10_add(t1, p->X, delta);
    fq10_mul(alpha, t0, t1);
    fq10_add(t0, alpha, alpha);
    fq10_add(alpha, t0, alpha);

    /* X3 = alpha^2 - 8*beta */
    fq10_sq(r->X, alpha);
    fq10_add(t0, beta, beta); /* 2*beta */
    fq10_add(t0, t0, t0); /* 4*beta */
    fq10_add(t1, t0, t0); /* 8*beta */
    fq10_sub(r->X, r->X, t1);

    /* Z3 = (Y1 + Z1)^2 - gamma - delta */
    fq10_add(t1, p->Y, p->Z);
    fq10_sq(t2, t1);
    fq10_sub(t2, t2, gamma);
    fq10_sub(r->Z, t2, delta);

    /* Y3 = alpha * (4*beta - X3) - 8*gamma^2 */
    fq10_sub(t1, t0, r->X); /* 4*beta - X3 */
    fq10_mul(t2, alpha, t1);
    fq10_sq(t0, gamma); /* gamma^2 */
    fq10_add(t0, t0, t0); /* 2*gamma^2 */
    fq10_add(t0, t0, t0); /* 4*gamma^2 */
    fq10_add(t0, t0, t0); /* 8*gamma^2 */
    fq10_sub(r->Y, t2, t0);
}

/* ── Inline fq10 general addition: add-2007-bl ── */

/*
 * General addition: Jacobian + Jacobian -> Jacobian (same formula as shaw_add_x64).
 * Cost: 11M + 5S (in fq10 arithmetic)
 */
static FQ10_AVX2_FORCE_INLINE void
    shaw_add_fq10(shaw_jacobian_10 *r, const shaw_jacobian_10 *p, const shaw_jacobian_10 *q)
{
    fq10 Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, rr, V;
    fq10 t0, t1;

    /* Z1Z1 = Z1^2 */
    fq10_sq(Z1Z1, p->Z);

    /* Z2Z2 = Z2^2 */
    fq10_sq(Z2Z2, q->Z);

    /* U1 = X1 * Z2Z2 */
    fq10_mul(U1, p->X, Z2Z2);

    /* U2 = X2 * Z1Z1 */
    fq10_mul(U2, q->X, Z1Z1);

    /* S1 = Y1 * Z2 * Z2Z2 */
    fq10_mul(t0, q->Z, Z2Z2);
    fq10_mul(S1, p->Y, t0);

    /* S2 = Y2 * Z1 * Z1Z1 */
    fq10_mul(t0, p->Z, Z1Z1);
    fq10_mul(S2, q->Y, t0);

    /* H = U2 - U1 */
    fq10_sub(H, U2, U1);

    /* I = (2*H)^2 */
    fq10_add(t0, H, H);
    fq10_sq(I, t0);

    /* J = H * I */
    fq10_mul(J, H, I);

    /* rr = 2 * (S2 - S1) */
    fq10_sub(rr, S2, S1);
    fq10_add(rr, rr, rr);

    /* V = U1 * I */
    fq10_mul(V, U1, I);

    /* X3 = rr^2 - J - 2*V */
    fq10_sq(r->X, rr);
    fq10_sub(r->X, r->X, J);
    fq10_add(t0, V, V);
    fq10_sub(r->X, r->X, t0);

    /* Y3 = rr * (V - X3) - 2 * S1 * J */
    fq10_sub(t0, V, r->X);
    fq10_mul(t1, rr, t0);
    fq10_mul(t0, S1, J);
    fq10_add(t0, t0, t0);
    fq10_sub(r->Y, t1, t0);

    /* Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H */
    fq10_add(t0, p->Z, q->Z);
    fq10_sq(t1, t0);
    fq10_sub(t1, t1, Z1Z1);
    fq10_sub(t1, t1, Z2Z2);
    fq10_mul(r->Z, t1, H);
}

/* ── fq10 point utility functions ── */

static FQ10_AVX2_FORCE_INLINE void shaw_copy_fq10(shaw_jacobian_10 *r, const shaw_jacobian_10 *p)
{
    fq10_copy(r->X, p->X);
    fq10_copy(r->Y, p->Y);
    fq10_copy(r->Z, p->Z);
}

static FQ10_AVX2_FORCE_INLINE void shaw_neg_fq10(shaw_jacobian_10 *r, const shaw_jacobian_10 *p)
{
    fq10_copy(r->X, p->X);
    fq10_neg(r->Y, p->Y);
    fq10_copy(r->Z, p->Z);
}

/* ── Main function ── */

void shaw_scalarmult_vartime_avx2(shaw_jacobian *r, const unsigned char scalar[32], const shaw_jacobian *p)
{
    /* Precompute odd multiples: [P, 3P, 5P, 7P, 9P, 11P, 13P, 15P] in Jacobian (fq51) */
    shaw_jacobian table_jac[8];
    shaw_jacobian p2_jac;

    shaw_copy(&table_jac[0], p); /* 1P */
    shaw_dbl_x64(&p2_jac, p); /* 2P */

    for (int i = 1; i < 8; i++)
        shaw_add_x64(&table_jac[i], &table_jac[i - 1], &p2_jac); /* (2i+1)P */

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
        ranshaw_secure_erase(&p2_jac, sizeof(p2_jac));
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
        ranshaw_secure_erase(&p2_jac, sizeof(p2_jac));
        shaw_identity(r);
        return;
    }

    /* Initialize accumulator in fq10 with the highest nonzero digit's point */
    shaw_jacobian_10 acc;
    int8_t d = naf[start];
    int idx = ((d < 0) ? -d : d) / 2; /* table index: |d|/2 since table stores odd multiples */
    shaw_copy_fq10(&acc, &table10[idx]);
    if (d < 0)
        shaw_neg_fq10(&acc, &acc);

    /* Main loop */
    for (int i = start - 1; i >= 0; i--)
    {
        shaw_dbl_fq10(&acc, &acc);

        if (naf[i] != 0)
        {
            d = naf[i];
            idx = ((d < 0) ? -d : d) / 2;
            if (d > 0)
            {
                shaw_add_fq10(&acc, &acc, &table10[idx]);
            }
            else
            {
                shaw_jacobian_10 neg_pt;
                shaw_neg_fq10(&neg_pt, &table10[idx]);
                shaw_add_fq10(&acc, &acc, &neg_pt);
            }
        }
    }

    /* Convert result back to fq51 */
    fq10_to_fq51(r->X, acc.X);
    fq10_to_fq51(r->Y, acc.Y);
    fq10_to_fq51(r->Z, acc.Z);

    ranshaw_secure_erase(naf, sizeof(naf));
    ranshaw_secure_erase(table_jac, sizeof(table_jac));
    ranshaw_secure_erase(table10, sizeof(table10));
    ranshaw_secure_erase(&p2_jac, sizeof(p2_jac));
}
