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

#include "x64/fp_frombytes.h"

#include "x64/fp51.h"

void fp_frombytes_x64(fp_fe h, const unsigned char *s)
{
    uint64_t h0 = ((uint64_t)s[0]) | ((uint64_t)s[1] << 8) | ((uint64_t)s[2] << 16) | ((uint64_t)s[3] << 24)
                  | ((uint64_t)s[4] << 32) | ((uint64_t)s[5] << 40) | ((uint64_t)(s[6] & 0x07) << 48);

    uint64_t h1 = ((uint64_t)(s[6] >> 3)) | ((uint64_t)s[7] << 5) | ((uint64_t)s[8] << 13) | ((uint64_t)s[9] << 21)
                  | ((uint64_t)s[10] << 29) | ((uint64_t)s[11] << 37) | ((uint64_t)(s[12] & 0x3f) << 45);

    uint64_t h2 = ((uint64_t)(s[12] >> 6)) | ((uint64_t)s[13] << 2) | ((uint64_t)s[14] << 10) | ((uint64_t)s[15] << 18)
                  | ((uint64_t)s[16] << 26) | ((uint64_t)s[17] << 34) | ((uint64_t)s[18] << 42)
                  | ((uint64_t)(s[19] & 0x01) << 50);

    uint64_t h3 = ((uint64_t)(s[19] >> 1)) | ((uint64_t)s[20] << 7) | ((uint64_t)s[21] << 15) | ((uint64_t)s[22] << 23)
                  | ((uint64_t)s[23] << 31) | ((uint64_t)s[24] << 39) | ((uint64_t)(s[25] & 0x0f) << 47);

    uint64_t h4 = ((uint64_t)(s[25] >> 4)) | ((uint64_t)s[26] << 4) | ((uint64_t)s[27] << 12) | ((uint64_t)s[28] << 20)
                  | ((uint64_t)s[29] << 28) | ((uint64_t)s[30] << 36) | ((uint64_t)(s[31] & 0x7f) << 44);

    h[0] = h0 & FP51_MASK;
    h[1] = h1 & FP51_MASK;
    h[2] = h2 & FP51_MASK;
    h[3] = h3 & FP51_MASK;
    h[4] = h4 & FP51_MASK;
}
