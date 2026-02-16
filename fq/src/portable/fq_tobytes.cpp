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

#include "portable/fq_tobytes.h"

#include "portable/fq25.h"

/*
 * Canonical reduction mod q = 2^255 - gamma, then serialize to 32 bytes LE.
 *
 * Unlike F_p where canonical reduction adds 19 and checks overflow,
 * for F_q we add gamma and check if the result overflows 2^255.
 *
 * Algorithm:
 *   1. Carry-normalize limbs with gamma-fold for limb 9 overflow
 *   2. Compute trial = h + gamma; if trial >= 2^255, use trial (= h - q)
 *   3. Serialize the canonical representative
 */
void fq_tobytes_portable(unsigned char *s, const fq_fe h)
{
    int32_t h0 = h[0], h1 = h[1], h2 = h[2], h3 = h[3], h4 = h[4];
    int32_t h5 = h[5], h6 = h[6], h7 = h[7], h8 = h[8], h9 = h[9];
    int32_t carry0, carry1, carry2, carry3, carry4;
    int32_t carry5, carry6, carry7, carry8, carry9;

    /*
     * First carry normalization pass with gamma-fold.
     * Carry from limb 9 wraps as carry * gamma to limbs 0-4.
     */
    carry9 = (h9 + (1 << 24)) >> 25;
    h9 -= carry9 << 25;
    h0 += carry9 * GAMMA_25[0];
    h1 += carry9 * GAMMA_25[1];
    h2 += carry9 * GAMMA_25[2];
    h3 += carry9 * GAMMA_25[3];
    h4 += carry9 * GAMMA_25[4];

    carry0 = h0 >> 26;
    h1 += carry0;
    h0 -= carry0 << 26;
    carry1 = h1 >> 25;
    h2 += carry1;
    h1 -= carry1 << 25;
    carry2 = h2 >> 26;
    h3 += carry2;
    h2 -= carry2 << 26;
    carry3 = h3 >> 25;
    h4 += carry3;
    h3 -= carry3 << 25;
    carry4 = h4 >> 26;
    h5 += carry4;
    h4 -= carry4 << 26;
    carry5 = h5 >> 25;
    h6 += carry5;
    h5 -= carry5 << 25;
    carry6 = h6 >> 26;
    h7 += carry6;
    h6 -= carry6 << 26;
    carry7 = h7 >> 25;
    h8 += carry7;
    h7 -= carry7 << 25;
    carry8 = h8 >> 26;
    h9 += carry8;
    h8 -= carry8 << 26;
    carry9 = h9 >> 25;
    h9 -= carry9 << 25;

    /* Second gamma fold if needed */
    h0 += carry9 * GAMMA_25[0];
    h1 += carry9 * GAMMA_25[1];
    h2 += carry9 * GAMMA_25[2];
    h3 += carry9 * GAMMA_25[3];
    h4 += carry9 * GAMMA_25[4];

    carry0 = h0 >> 26;
    h1 += carry0;
    h0 -= carry0 << 26;
    carry1 = h1 >> 25;
    h2 += carry1;
    h1 -= carry1 << 25;
    carry2 = h2 >> 26;
    h3 += carry2;
    h2 -= carry2 << 26;
    carry3 = h3 >> 25;
    h4 += carry3;
    h3 -= carry3 << 25;
    carry4 = h4 >> 26;
    h5 += carry4;
    h4 -= carry4 << 26;
    carry5 = h5 >> 25;
    h6 += carry5;
    h5 -= carry5 << 25;
    carry6 = h6 >> 26;
    h7 += carry6;
    h6 -= carry6 << 26;
    carry7 = h7 >> 25;
    h8 += carry7;
    h7 -= carry7 << 25;
    carry8 = h8 >> 26;
    h9 += carry8;
    h8 -= carry8 << 26;

    /*
     * Canonical reduction: check if value >= q by trial-adding gamma.
     * q = 2^255 - gamma, so value >= q iff value + gamma >= 2^255.
     *
     * We compute the carry chain of (h + gamma) and check if limb 9 overflows.
     */
    int32_t u0 = h0 + GAMMA_25[0];
    carry0 = u0 >> 26;
    u0 &= (1 << 26) - 1;
    int32_t u1 = h1 + GAMMA_25[1] + carry0;
    carry1 = u1 >> 25;
    u1 &= (1 << 25) - 1;
    int32_t u2 = h2 + GAMMA_25[2] + carry1;
    carry2 = u2 >> 26;
    u2 &= (1 << 26) - 1;
    int32_t u3 = h3 + GAMMA_25[3] + carry2;
    carry3 = u3 >> 25;
    u3 &= (1 << 25) - 1;
    int32_t u4 = h4 + GAMMA_25[4] + carry3;
    carry4 = u4 >> 26;
    u4 &= (1 << 26) - 1;
    int32_t u5 = h5 + carry4;
    carry5 = u5 >> 25;
    u5 &= (1 << 25) - 1;
    int32_t u6 = h6 + carry5;
    carry6 = u6 >> 26;
    u6 &= (1 << 26) - 1;
    int32_t u7 = h7 + carry6;
    carry7 = u7 >> 25;
    u7 &= (1 << 25) - 1;
    int32_t u8 = h8 + carry7;
    carry8 = u8 >> 26;
    u8 &= (1 << 26) - 1;
    int32_t u9 = h9 + carry8;
    int32_t overflow = u9 >> 25;
    u9 &= (1 << 25) - 1;

    /* If overflow, use u (= h + gamma = h - q), else use h */
    int32_t mask = 0 - overflow;
    h0 = (h0 & ~mask) | (u0 & mask);
    h1 = (h1 & ~mask) | (u1 & mask);
    h2 = (h2 & ~mask) | (u2 & mask);
    h3 = (h3 & ~mask) | (u3 & mask);
    h4 = (h4 & ~mask) | (u4 & mask);
    h5 = (h5 & ~mask) | (u5 & mask);
    h6 = (h6 & ~mask) | (u6 & mask);
    h7 = (h7 & ~mask) | (u7 & mask);
    h8 = (h8 & ~mask) | (u8 & mask);
    h9 = (h9 & ~mask) | (u9 & mask);

    /* Serialize 10 limbs to 32 bytes little-endian (same bit packing as Fp) */
    s[0] = (unsigned char)(h0 >> 0);
    s[1] = (unsigned char)(h0 >> 8);
    s[2] = (unsigned char)(h0 >> 16);
    s[3] = (unsigned char)((h0 >> 24) | (h1 << 2));
    s[4] = (unsigned char)(h1 >> 6);
    s[5] = (unsigned char)(h1 >> 14);
    s[6] = (unsigned char)((h1 >> 22) | (h2 << 3));
    s[7] = (unsigned char)(h2 >> 5);
    s[8] = (unsigned char)(h2 >> 13);
    s[9] = (unsigned char)((h2 >> 21) | (h3 << 5));
    s[10] = (unsigned char)(h3 >> 3);
    s[11] = (unsigned char)(h3 >> 11);
    s[12] = (unsigned char)((h3 >> 19) | (h4 << 6));
    s[13] = (unsigned char)(h4 >> 2);
    s[14] = (unsigned char)(h4 >> 10);
    s[15] = (unsigned char)(h4 >> 18);
    s[16] = (unsigned char)(h5 >> 0);
    s[17] = (unsigned char)(h5 >> 8);
    s[18] = (unsigned char)(h5 >> 16);
    s[19] = (unsigned char)((h5 >> 24) | (h6 << 1));
    s[20] = (unsigned char)(h6 >> 7);
    s[21] = (unsigned char)(h6 >> 15);
    s[22] = (unsigned char)((h6 >> 23) | (h7 << 3));
    s[23] = (unsigned char)(h7 >> 5);
    s[24] = (unsigned char)(h7 >> 13);
    s[25] = (unsigned char)((h7 >> 21) | (h8 << 4));
    s[26] = (unsigned char)(h8 >> 4);
    s[27] = (unsigned char)(h8 >> 12);
    s[28] = (unsigned char)((h8 >> 20) | (h9 << 6));
    s[29] = (unsigned char)(h9 >> 2);
    s[30] = (unsigned char)(h9 >> 10);
    s[31] = (unsigned char)(h9 >> 18);
}
