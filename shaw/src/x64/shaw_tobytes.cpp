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

#include "shaw_tobytes.h"

#include "fq_invert.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "fq_tobytes.h"
#include "fq_utils.h"

/*
 * Serialize a Shaw Jacobian point to 32 bytes.
 * Format: x-coordinate little-endian, y-parity in bit 255.
 *
 * For identity point, outputs all zeros.
 */
void shaw_tobytes_x64(unsigned char s[32], const shaw_jacobian *p)
{
    if (!fq_isnonzero(p->Z))
    {
        for (int i = 0; i < 32; i++)
            s[i] = 0;
        return;
    }

    fq_fe z_inv, z_inv2, z_inv3, x, y;

    fq_invert(z_inv, p->Z);
    fq_sq(z_inv2, z_inv);
    fq_mul(z_inv3, z_inv2, z_inv);
    fq_mul(x, p->X, z_inv2);
    fq_mul(y, p->Y, z_inv3);

    fq_tobytes(s, x);
    s[31] |= (unsigned char)(fq_isnegative(y) << 7);
}
