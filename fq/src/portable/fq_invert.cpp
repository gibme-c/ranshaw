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

#include "portable/fq_invert.h"

#include "portable/fq25_chain.h"
#include "ranshaw_secure_erase.h"

/*
 * Compute z^(q-2) mod q via optimized addition chain.
 *
 * q-2 = 0x7fffffffffffffffffffffffffffffff4bb1eb0e39730fa771684645ec70f85d
 *
 * Decomposition:
 *   Upper 128 bits = 0x7fffffffffffffffffffffffffffffff = 2^127 - 1 (all ones)
 *   Lower 128 bits = 0x4bb1eb0e39730fa771684645ec70f85d
 *
 * Strategy:
 *   1. Build z^(2^127-1) via addition chain (~126 sq + 9 mul)
 *   2. Use 4-bit fixed window for bottom 128 bits (~128 sq + 29 mul)
 *   3. Precompute z^2..z^15 for the window table (~3 sq + 11 mul)
 *
 * Total: 254 sq + 47 mul (vs ~254 sq + 202 mul for naive bit-scan)
 */

void fq_invert_portable(fq_fe out, const fq_fe z)
{
    /* Precomputed table entries z^2 through z^15 */
    fq_fe zt2, zt3, zt4, zt5, zt6, zt7, zt8, zt9, zt10, zt11, zt12, zt13, zt14, zt15;

    /* Addition chain temporaries */
    fq_fe x31, x10, x25, x50, x100, acc;

    /* ---- Precompute z^2 through z^15 ---- */
    fq25_chain_sq(zt2, z); /* z^2 */
    fq25_chain_mul(zt3, zt2, z); /* z^3 */
    fq25_chain_sq(zt4, zt2); /* z^4 */
    fq25_chain_mul(zt5, zt4, z); /* z^5 */
    fq25_chain_mul(zt6, zt4, zt2); /* z^6 */
    fq25_chain_mul(zt7, zt4, zt3); /* z^7 */
    fq25_chain_sq(zt8, zt4); /* z^8 */
    fq25_chain_mul(zt9, zt8, z); /* z^9 */
    fq25_chain_mul(zt10, zt8, zt2); /* z^10 */
    fq25_chain_mul(zt11, zt8, zt3); /* z^11 */
    fq25_chain_mul(zt12, zt8, zt4); /* z^12 */
    fq25_chain_mul(zt13, zt8, zt5); /* z^13 */
    fq25_chain_mul(zt14, zt8, zt6); /* z^14 */
    fq25_chain_mul(zt15, zt8, zt7); /* z^15 */

    /* ---- Addition chain for z^(2^127-1) ---- */
    /* z^31 = z^(2^5-1) */
    fq25_chain_sq(x31, zt15); /* z^30 */
    fq25_chain_mul(x31, x31, z); /* z^31 */

    /* z^(2^10-1) */
    fq25_chain_sqn(x10, x31, 5);
    fq25_chain_mul(x10, x10, x31);

    /* z^(2^20-1) */
    fq25_chain_sqn(acc, x10, 10);
    fq25_chain_mul(acc, acc, x10);

    /* z^(2^25-1) */
    fq25_chain_sqn(x25, acc, 5);
    fq25_chain_mul(x25, x25, x31);

    /* z^(2^50-1) */
    fq25_chain_sqn(x50, x25, 25);
    fq25_chain_mul(x50, x50, x25);

    /* z^(2^100-1) */
    fq25_chain_sqn(x100, x50, 50);
    fq25_chain_mul(x100, x100, x50);

    /* z^(2^125-1) */
    fq25_chain_sqn(acc, x100, 25);
    fq25_chain_mul(acc, acc, x25);

    /* z^(2^127-1) */
    fq25_chain_sqn(acc, acc, 2);
    fq25_chain_mul(acc, acc, zt3);

    /* ---- 4-bit window scan of bottom 128 bits ---- */
    /* Lower 128 bits of q-2 = 0x4bb1eb0e39730fa771684645ec70f85d */
    /* Nibbles (MSB first): 4,b,b,1,e,b,0,e,3,9,7,3,0,f,a,7,7,1,6,8,4,6,4,5,e,c,7,0,f,8,5,d */

    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt4); /* 4 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt11); /* b = 11 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt11); /* b = 11 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, z); /* 1 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt14); /* e = 14 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt11); /* b = 11 */
    fq25_chain_sqn(acc, acc, 4);
    /* nibble 0: shift only, no multiply */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt14); /* e = 14 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt3); /* 3 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt9); /* 9 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt7); /* 7 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt3); /* 3 */
    fq25_chain_sqn(acc, acc, 4);
    /* nibble 0: shift only, no multiply */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt15); /* f = 15 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt10); /* a = 10 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt7); /* 7 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt7); /* 7 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, z); /* 1 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt6); /* 6 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt8); /* 8 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt4); /* 4 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt6); /* 6 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt4); /* 4 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt5); /* 5 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt14); /* e = 14 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt12); /* c = 12 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt7); /* 7 */
    fq25_chain_sqn(acc, acc, 4);
    /* nibble 0: shift only, no multiply */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt15); /* f = 15 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt8); /* 8 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt5); /* 5 */
    fq25_chain_sqn(acc, acc, 4);
    fq25_chain_mul(acc, acc, zt13); /* d = 13 */

    for (int i = 0; i < 10; i++)
        out[i] = acc[i];

    ranshaw_secure_erase(zt2, sizeof(fq_fe));
    ranshaw_secure_erase(zt3, sizeof(fq_fe));
    ranshaw_secure_erase(zt4, sizeof(fq_fe));
    ranshaw_secure_erase(zt5, sizeof(fq_fe));
    ranshaw_secure_erase(zt6, sizeof(fq_fe));
    ranshaw_secure_erase(zt7, sizeof(fq_fe));
    ranshaw_secure_erase(zt8, sizeof(fq_fe));
    ranshaw_secure_erase(zt9, sizeof(fq_fe));
    ranshaw_secure_erase(zt10, sizeof(fq_fe));
    ranshaw_secure_erase(zt11, sizeof(fq_fe));
    ranshaw_secure_erase(zt12, sizeof(fq_fe));
    ranshaw_secure_erase(zt13, sizeof(fq_fe));
    ranshaw_secure_erase(zt14, sizeof(fq_fe));
    ranshaw_secure_erase(zt15, sizeof(fq_fe));
    ranshaw_secure_erase(x31, sizeof(fq_fe));
    ranshaw_secure_erase(x10, sizeof(fq_fe));
    ranshaw_secure_erase(x25, sizeof(fq_fe));
    ranshaw_secure_erase(x50, sizeof(fq_fe));
    ranshaw_secure_erase(x100, sizeof(fq_fe));
    ranshaw_secure_erase(acc, sizeof(fq_fe));
}
