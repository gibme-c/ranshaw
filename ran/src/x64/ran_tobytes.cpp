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

#include "ran_tobytes.h"

#include "fp_invert.h"
#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "fp_tobytes.h"
#include "fp_utils.h"

/*
 * Serialize a Ran Jacobian point to 32 bytes.
 * Format: x-coordinate little-endian, y-parity in bit 255.
 *
 * For identity point, outputs all zeros.
 */
void ran_tobytes_x64(unsigned char s[32], const ran_jacobian *p)
{
    /* Check for identity */
    if (!fp_isnonzero(p->Z))
    {
        for (int i = 0; i < 32; i++)
            s[i] = 0;
        return;
    }

    fp_fe z_inv, z_inv2, z_inv3, x, y;

    /* x = X/Z^2, y = Y/Z^3 */
    fp_invert(z_inv, p->Z);
    fp_sq(z_inv2, z_inv);
    fp_mul(z_inv3, z_inv2, z_inv);
    fp_mul(x, p->X, z_inv2);
    fp_mul(y, p->Y, z_inv3);

    /* Serialize x-coordinate */
    fp_tobytes(s, x);

    /* Pack y-parity into bit 255 */
    s[31] |= (unsigned char)(fp_isnegative(y) << 7);
}
