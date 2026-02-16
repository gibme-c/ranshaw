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

#include "x64/fq_tobytes.h"

#include "ranshaw_platform.h"
#include "x64/fq51.h"

/*
 * Canonical reduction mod q = 2^255 - gamma, then serialize to 32 bytes LE.
 *
 * fq_fe is uint64_t[5] in radix-2^51. Follows the same pattern as fp_tobytes_x64:
 * 1. Carry chain with gamma fold to nearly canonicalize
 * 2. "Add gamma and check overflow" trick to fully canonicalize
 * 3. Pack 5×51 limbs into 32 little-endian bytes
 *
 * The "add gamma" trick: if t >= q, then t + gamma >= 2^255, so bit 255 is set
 * after carry propagation. We use the overflow bit to select between t and t-q.
 * Since t + gamma = t - q + 2^255, the lower 255 bits of (t + gamma) equal t - q.
 */
void fq_tobytes_x64(unsigned char *s, const fq_fe h)
{
    uint64_t t[5];
    t[0] = h[0];
    t[1] = h[1];
    t[2] = h[2];
    t[3] = h[3];
    t[4] = h[4];

    /* First carry chain with gamma fold */
    uint64_t carry;
    carry = t[0] >> 51;
    t[1] += carry;
    t[0] &= FQ51_MASK;
    carry = t[1] >> 51;
    t[2] += carry;
    t[1] &= FQ51_MASK;
    carry = t[2] >> 51;
    t[3] += carry;
    t[2] &= FQ51_MASK;
    carry = t[3] >> 51;
    t[4] += carry;
    t[3] &= FQ51_MASK;
    carry = t[4] >> 51;
    t[4] &= FQ51_MASK;
    /* Gamma fold */
    t[0] += carry * GAMMA_51[0];
    t[1] += carry * GAMMA_51[1];
    t[2] += carry * GAMMA_51[2];
    /* Re-carry */
    carry = t[0] >> 51;
    t[1] += carry;
    t[0] &= FQ51_MASK;
    carry = t[1] >> 51;
    t[2] += carry;
    t[1] &= FQ51_MASK;

    /*
     * "Add gamma and check overflow" trick for canonicalization.
     * If t >= q, then t + gamma >= 2^255, carry out of limb 4 is nonzero.
     * The lower 255 bits of (t + gamma) give t - q.
     */
    uint64_t u[5];
    u[0] = t[0] + GAMMA_51[0];
    carry = u[0] >> 51;
    u[0] &= FQ51_MASK;
    u[1] = t[1] + GAMMA_51[1] + carry;
    carry = u[1] >> 51;
    u[1] &= FQ51_MASK;
    u[2] = t[2] + GAMMA_51[2] + carry;
    carry = u[2] >> 51;
    u[2] &= FQ51_MASK;
    u[3] = t[3] + carry;
    carry = u[3] >> 51;
    u[3] &= FQ51_MASK;
    u[4] = t[4] + carry;
    uint64_t q = u[4] >> 51; /* 1 if t >= q, 0 otherwise */

    /* Select: if q == 1, use u (= t - q); else use t */
    uint64_t mask = 0 - q;
    t[0] = (t[0] & ~mask) | (u[0] & mask);
    t[1] = (t[1] & ~mask) | (u[1] & mask);
    t[2] = (t[2] & ~mask) | (u[2] & mask);
    t[3] = (t[3] & ~mask) | (u[3] & mask);
    t[4] = (t[4] & ~mask) | ((u[4] & FQ51_MASK) & mask);

    /* Pack 5×51 limbs into 32 bytes LE (identical to fp_tobytes_x64) */
    s[0] = (unsigned char)(t[0]);
    s[1] = (unsigned char)(t[0] >> 8);
    s[2] = (unsigned char)(t[0] >> 16);
    s[3] = (unsigned char)(t[0] >> 24);
    s[4] = (unsigned char)(t[0] >> 32);
    s[5] = (unsigned char)(t[0] >> 40);
    s[6] = (unsigned char)((t[0] >> 48) | (t[1] << 3));
    s[7] = (unsigned char)(t[1] >> 5);
    s[8] = (unsigned char)(t[1] >> 13);
    s[9] = (unsigned char)(t[1] >> 21);
    s[10] = (unsigned char)(t[1] >> 29);
    s[11] = (unsigned char)(t[1] >> 37);
    s[12] = (unsigned char)((t[1] >> 45) | (t[2] << 6));
    s[13] = (unsigned char)(t[2] >> 2);
    s[14] = (unsigned char)(t[2] >> 10);
    s[15] = (unsigned char)(t[2] >> 18);
    s[16] = (unsigned char)(t[2] >> 26);
    s[17] = (unsigned char)(t[2] >> 34);
    s[18] = (unsigned char)(t[2] >> 42);
    s[19] = (unsigned char)((t[2] >> 50) | (t[3] << 1));
    s[20] = (unsigned char)(t[3] >> 7);
    s[21] = (unsigned char)(t[3] >> 15);
    s[22] = (unsigned char)(t[3] >> 23);
    s[23] = (unsigned char)(t[3] >> 31);
    s[24] = (unsigned char)(t[3] >> 39);
    s[25] = (unsigned char)((t[3] >> 47) | (t[4] << 4));
    s[26] = (unsigned char)(t[4] >> 4);
    s[27] = (unsigned char)(t[4] >> 12);
    s[28] = (unsigned char)(t[4] >> 20);
    s[29] = (unsigned char)(t[4] >> 28);
    s[30] = (unsigned char)(t[4] >> 36);
    s[31] = (unsigned char)(t[4] >> 44);
}
