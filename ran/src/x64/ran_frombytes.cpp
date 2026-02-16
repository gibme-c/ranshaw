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

#include "ran_frombytes.h"

#include "fp_frombytes.h"
#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "fp_sqrt.h"
#include "fp_tobytes.h"
#include "fp_utils.h"
#include "ran_constants.h"
#include "ran_validate.h"

/*
 * Deserialize 32 bytes to a Ran Jacobian point.
 *
 * Format: x-coordinate LE with y-parity in bit 255.
 *
 * Algorithm:
 *   1. Extract y-parity from bit 255
 *   2. Mask bit 255, deserialize x
 *   3. Reject non-canonical x (>= p)
 *   4. Compute rhs = x^3 - 3x + b
 *   5. y = sqrt(rhs) — returns -1 if not QR (invalid point)
 *   6. If y parity doesn't match, negate y
 *   7. Return Jacobian (x, y, 1)
 *
 * Returns 0 on success, -1 on invalid input.
 *
 * SECURITY NOTE: Early returns on validation failure are intentionally
 * variable-time. The input bytes are public (untrusted external data),
 * not secret. Timing side-channels on public data are not exploitable.
 */
int ran_frombytes_x64(ran_jacobian *r, const unsigned char s[32])
{
    /* Extract y-parity */
    unsigned int y_parity = (s[31] >> 7) & 1;

    /* Mask off bit 255 and deserialize x */
    unsigned char x_bytes[32];
    for (int i = 0; i < 32; i++)
        x_bytes[i] = s[i];
    x_bytes[31] &= 0x7f;

    /* Reject non-canonical x: deserialize and re-serialize, check equality */
    fp_fe x;
    fp_frombytes(x, x_bytes);
    unsigned char x_check[32];
    fp_tobytes(x_check, x);

    unsigned char diff = 0;
    for (int i = 0; i < 32; i++)
        diff |= x_check[i] ^ x_bytes[i];
    if (diff != 0)
        return -1;

    /* Compute rhs = x^3 - 3x + b */
    fp_fe x2, x3, rhs;
    fp_sq(x2, x);
    fp_mul(x3, x2, x);

    fp_fe three_x;
    fp_add(three_x, x, x);
    fp_add(three_x, three_x, x);

    fp_sub(rhs, x3, three_x);
    fp_add(rhs, rhs, RAN_B);

    /* Compute y = sqrt(rhs) */
    fp_fe y;
    if (fp_sqrt(y, rhs) != 0)
        return -1;

    /* Adjust y parity */
    if ((unsigned int)fp_isnegative(y) != y_parity)
        fp_neg(y, y);

    /* Return Jacobian (x, y, 1) */
    fp_copy(r->X, x);
    fp_copy(r->Y, y);
    fp_1(r->Z);

    return 0;
}
