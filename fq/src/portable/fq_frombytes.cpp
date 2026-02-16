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

#include "portable/fq_frombytes.h"

#include "load_3.h"
#include "load_4.h"
#include "portable/fq25.h"

/*
 * Deserialize 32 bytes (little-endian) into a 10-limb F_q field element.
 *
 * Same unpacking as fp_frombytes (alternating 26/25-bit limbs), but the
 * carry from limb 9 wraps via gamma instead of *19.
 */
void fq_frombytes_portable(fq_fe h, const unsigned char *s)
{
    int64_t h0 = static_cast<int64_t>(load_4(s));
    int64_t h1 = static_cast<int64_t>(load_3(s + 4) << 6);
    int64_t h2 = static_cast<int64_t>(load_3(s + 7) << 5);
    int64_t h3 = static_cast<int64_t>(load_3(s + 10) << 3);
    int64_t h4 = static_cast<int64_t>(load_3(s + 13) << 2);
    int64_t h5 = static_cast<int64_t>(load_4(s + 16));
    int64_t h6 = static_cast<int64_t>(load_3(s + 20) << 7);
    int64_t h7 = static_cast<int64_t>(load_3(s + 23) << 5);
    int64_t h8 = static_cast<int64_t>(load_3(s + 26) << 4);
    int64_t h9 = static_cast<int64_t>((load_3(s + 29) & 8388607) << 2);
    int64_t carry0, carry1, carry2, carry3, carry4;
    int64_t carry5, carry6, carry7, carry8, carry9;

    /* Carry from limb 9 wraps via gamma (not *19 as in Fp) */
    carry9 = (h9 + (int64_t)(1 << 24)) >> 25;
    h9 -= carry9 << 25;
    h0 += carry9 * (int64_t)GAMMA_25[0];
    h1 += carry9 * (int64_t)GAMMA_25[1];
    h2 += carry9 * (int64_t)GAMMA_25[2];
    h3 += carry9 * (int64_t)GAMMA_25[3];
    h4 += carry9 * (int64_t)GAMMA_25[4];

    carry1 = (h1 + (int64_t)(1 << 24)) >> 25;
    h2 += carry1;
    h1 -= carry1 << 25;
    carry3 = (h3 + (int64_t)(1 << 24)) >> 25;
    h4 += carry3;
    h3 -= carry3 << 25;
    carry5 = (h5 + (int64_t)(1 << 24)) >> 25;
    h6 += carry5;
    h5 -= carry5 << 25;
    carry7 = (h7 + (int64_t)(1 << 24)) >> 25;
    h8 += carry7;
    h7 -= carry7 << 25;

    carry0 = (h0 + (int64_t)(1 << 25)) >> 26;
    h1 += carry0;
    h0 -= carry0 << 26;
    carry2 = (h2 + (int64_t)(1 << 25)) >> 26;
    h3 += carry2;
    h2 -= carry2 << 26;
    carry4 = (h4 + (int64_t)(1 << 25)) >> 26;
    h5 += carry4;
    h4 -= carry4 << 26;
    carry6 = (h6 + (int64_t)(1 << 25)) >> 26;
    h7 += carry6;
    h6 -= carry6 << 26;
    carry8 = (h8 + (int64_t)(1 << 25)) >> 26;
    h9 += carry8;
    h8 -= carry8 << 26;

    h[0] = (int32_t)h0;
    h[1] = (int32_t)h1;
    h[2] = (int32_t)h2;
    h[3] = (int32_t)h3;
    h[4] = (int32_t)h4;
    h[5] = (int32_t)h5;
    h[6] = (int32_t)h6;
    h[7] = (int32_t)h7;
    h[8] = (int32_t)h8;
    h[9] = (int32_t)h9;
}
