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

#include "shaw_frombytes.h"

#include "fq_frombytes.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "fq_sqrt.h"
#include "fq_tobytes.h"
#include "fq_utils.h"
#include "shaw_validate.h"

/*
 * Convert a 5-limb radix-2^51 constant (stored as raw uint64_t[5]) to fq_fe
 * via tobytes+frombytes round-trip. This avoids type-punning issues on 32-bit
 * where fq_fe is int32_t[10].
 */
static void fq_from_limbs51(fq_fe r, const uint64_t limbs[5])
{
    unsigned char s[32];
    uint64_t h0 = limbs[0], h1 = limbs[1], h2 = limbs[2], h3 = limbs[3], h4 = limbs[4];

    uint64_t w0 = h0 | (h1 << 51);
    uint64_t w1 = (h1 >> 13) | (h2 << 38);
    uint64_t w2 = (h2 >> 26) | (h3 << 25);
    uint64_t w3 = (h3 >> 39) | (h4 << 12);

    for (int i = 0; i < 8; i++)
        s[i] = (unsigned char)(w0 >> (8 * i));
    for (int i = 0; i < 8; i++)
        s[8 + i] = (unsigned char)(w1 >> (8 * i));
    for (int i = 0; i < 8; i++)
        s[16 + i] = (unsigned char)(w2 >> (8 * i));
    for (int i = 0; i < 8; i++)
        s[24 + i] = (unsigned char)(w3 >> (8 * i));

    fq_frombytes(r, s);
}

/*
 * SHAW_B from shaw_constants.h, stored as raw 5-limb values.
 * Cannot use shaw_constants.h directly since fq_fe is int32_t[10] on 32-bit.
 */
static const uint64_t SHAW_B_LIMBS[5] =
    {0x3C002DD34C5B1ULL, 0x74516598A55ACULL, 0x22B764C970DBBULL, 0x9A0F7A5E3F44ULL, 0x640657EEA7EFBULL};

/*
 * Deserialize 32 bytes to a Shaw Jacobian point.
 * Same algorithm as ran_frombytes but over F_q.
 *
 * Returns 0 on success, -1 on invalid input.
 *
 * SECURITY NOTE: Early returns on validation failure are intentionally
 * variable-time. The input bytes are public (untrusted external data),
 * not secret. Timing side-channels on public data are not exploitable.
 */
int shaw_frombytes_portable(shaw_jacobian *r, const unsigned char s[32])
{
    unsigned int y_parity = (s[31] >> 7) & 1;

    unsigned char x_bytes[32];
    for (int i = 0; i < 32; i++)
        x_bytes[i] = s[i];
    x_bytes[31] &= 0x7f;

    /* Reject non-canonical x */
    fq_fe x;
    fq_frombytes(x, x_bytes);
    unsigned char x_check[32];
    fq_tobytes(x_check, x);

    unsigned char x_diff = 0;
    for (int i = 0; i < 32; i++)
        x_diff |= x_check[i] ^ x_bytes[i];
    if (x_diff != 0)
        return -1;

    /* Load SHAW_B from limbs */
    fq_fe shaw_b;
    fq_from_limbs51(shaw_b, SHAW_B_LIMBS);

    /* Compute rhs = x^3 - 3x + b */
    fq_fe x2, x3, rhs;
    fq_sq(x2, x);
    fq_mul(x3, x2, x);

    fq_fe three_x;
    fq_add(three_x, x, x);
    fq_add(three_x, three_x, x);

    fq_sub(rhs, x3, three_x);
    fq_add(rhs, rhs, shaw_b);

    /* Compute y = sqrt(rhs) -- for q = 3 mod 4, sqrt = rhs^((q+1)/4) */
    fq_fe y;
    fq_sqrt(y, rhs);

    /* Verify: y^2 == rhs */
    fq_fe y2, diff;
    fq_sq(y2, y);
    fq_sub(diff, y2, rhs);
    unsigned char diff_bytes[32];
    fq_tobytes(diff_bytes, diff);
    unsigned char d = 0;
    for (int i = 0; i < 32; i++)
        d |= diff_bytes[i];
    if (d != 0)
        return -1;

    /* Adjust y parity */
    if ((unsigned int)fq_isnegative(y) != y_parity)
        fq_neg(y, y);

    /* Return Jacobian (x, y, 1) */
    fq_copy(r->X, x);
    fq_copy(r->Y, y);
    fq_1(r->Z);

    return 0;
}
