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

#include "ran_scalarmult_vartime.h"

#include "fp_ops.h"
#include "fp_utils.h"
#include "ran.h"
#include "ran_add.h"
#include "ran_dbl.h"
#include "ran_madd.h"
#include "ran_ops.h"
#include "ranshaw_secure_erase.h"

/*
 * Variable-time scalar multiplication using wNAF with window width 5.
 *
 * Algorithm:
 *   1. Precompute odd multiples: [P, 3P, 5P, 7P, 9P, 11P, 13P, 15P]
 *   2. wNAF-encode scalar with w=5 → digits in [-15, 15], non-adjacent
 *   3. Scan from MSB to LSB:
 *      - Double
 *      - If digit != 0: add/sub precomputed point
 */

/*
 * wNAF encoding with window width w=5.
 * Output: naf[256] with values in {-15,-13,...,-1,0,1,...,13,15}
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

void ran_scalarmult_vartime_x64(ran_jacobian *r, const unsigned char scalar[32], const ran_jacobian *p)
{
    /* Precompute odd multiples: [P, 3P, 5P, 7P, 9P, 11P, 13P, 15P] */
    ran_jacobian table[8];
    ran_jacobian p2;

    ran_copy(&table[0], p); /* 1P */
    ran_dbl(&p2, p); /* 2P */

    for (int i = 1; i < 8; i++)
        ran_add(&table[i], &table[i - 1], &p2); /* (2i+1)P */

    /* wNAF encode */
    int8_t naf[257];
    int top = wnaf_encode(naf, scalar);

    if (top == 0)
    {
        ranshaw_secure_erase(naf, sizeof(naf));
        ranshaw_secure_erase(table, sizeof(table));
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
        ranshaw_secure_erase(table, sizeof(table));
        ranshaw_secure_erase(&p2, sizeof(p2));
        ran_identity(r);
        return;
    }

    /* Initialize with the highest nonzero digit's point */
    int8_t d = naf[start];
    int idx = ((d < 0) ? -d : d) / 2; /* table index: |d|/2 since table stores odd multiples */
    ran_copy(r, &table[idx]);
    if (d < 0)
        ran_neg(r, r);

    /* Main loop */
    for (int i = start - 1; i >= 0; i--)
    {
        ran_dbl(r, r);

        if (naf[i] != 0)
        {
            d = naf[i];
            idx = ((d < 0) ? -d : d) / 2;
            if (d > 0)
                ran_add(r, r, &table[idx]);
            else
            {
                ran_jacobian neg_pt;
                ran_neg(&neg_pt, &table[idx]);
                ran_add(r, r, &neg_pt);
            }
        }
    }

    ranshaw_secure_erase(naf, sizeof(naf));
    ranshaw_secure_erase(table, sizeof(table));
    ranshaw_secure_erase(&p2, sizeof(p2));
}
