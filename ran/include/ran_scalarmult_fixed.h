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

#ifndef RANSHAW_RAN_SCALARMULT_FIXED_H
#define RANSHAW_RAN_SCALARMULT_FIXED_H

/**
 * @file ran_scalarmult_fixed.h
 * @brief Fixed-base constant-time scalar multiplication for Ran (w=5).
 *
 * Precomputes a 16-entry affine table [1P, 2P, ..., 16P] once, then uses
 * signed 5-bit windowed scalar multiplication with 52 windows. Saves ~12
 * mixed additions per scalarmult compared to w=4, and amortizes table
 * computation across multiple calls with the same base point.
 */

#include "ran.h"
#include "ran_add.h"
#include "ran_batch_affine.h"
#include "ran_dbl.h"
#include "ran_madd.h"
#include "ran_ops.h"
#include "ranshaw_secure_erase.h"

/**
 * Precompute fixed-base table: 16 affine points [1P, 2P, ..., 16P].
 *
 * @param table  Output array of 16 affine points
 * @param p      Base point in Jacobian coordinates
 */
static inline void ran_scalarmult_fixed_precompute(ran_affine table[16], const ran_jacobian *p)
{
    ran_jacobian jac[16];
    ran_copy(&jac[0], p); /* 1P */
    ran_dbl(&jac[1], p); /* 2P */
    for (int i = 2; i < 16; i++)
        ran_add(&jac[i], &jac[i - 1], p); /* (i+1)P */

    ran_batch_to_affine(table, jac, 16);
}

/**
 * Recode 256-bit scalar into 52 signed 5-bit digits in [-16, 16].
 * scalar = sum(digits[i] * 32^i) for i = 0..51.
 */
static inline void ran_scalar_recode_signed5(int8_t digits[52], const unsigned char scalar[32])
{
    int carry = 0;
    for (int i = 0; i < 51; i++)
    {
        int bit_offset = 5 * i;
        int byte_idx = bit_offset >> 3;
        int bit_pos = bit_offset & 7;

        /* Read up to 2 bytes to handle cross-byte windows */
        unsigned int word = scalar[byte_idx];
        if (byte_idx + 1 < 32)
            word |= ((unsigned int)scalar[byte_idx + 1]) << 8;

        int val = (int)((word >> bit_pos) & 0x1fu) + carry;
        carry = (val + 16) >> 5;
        digits[i] = (int8_t)(val - (carry << 5));
    }
    /* Last window: bit 255 (1 bit) */
    digits[51] = (int8_t)(((scalar[31] >> 7) & 1) + carry);
}

/**
 * Fixed-base constant-time scalar multiplication using precomputed table.
 *
 * @param r       Output point (Jacobian)
 * @param scalar  256-bit scalar (32 bytes, little-endian)
 * @param table   Precomputed affine table of 16 points [1P..16P]
 */
static inline void ran_scalarmult_fixed(ran_jacobian *r, const unsigned char scalar[32], const ran_affine table[16])
{
    /* Recode scalar to signed 5-bit digits */
    int8_t digits[52];
    ran_scalar_recode_signed5(digits, scalar);

    /* Start from top digit (window 51) */
    int32_t d = (int32_t)digits[51];
    int32_t sign_mask = -(int32_t)((uint32_t)d >> 31);
    unsigned int abs_d = (unsigned int)((d ^ sign_mask) - sign_mask);
    unsigned int neg = (unsigned int)(sign_mask & 1);

    /* CT table lookup */
    ran_affine selected;
    fp_0(selected.x);
    fp_0(selected.y);
    for (unsigned int j = 0; j < 16; j++)
    {
        unsigned int eq = ((abs_d ^ (j + 1)) - 1u) >> 31;
        ran_affine_cmov(&selected, &table[j], eq);
    }
    ran_affine_cneg(&selected, neg);

    /* Initialize accumulator */
    ran_jacobian from_table;
    ran_from_affine(&from_table, &selected);

    ran_jacobian ident;
    ran_identity(&ident);

    unsigned int nonzero = 1u ^ ((abs_d - 1u) >> 31);
    ran_copy(r, &ident);
    ran_cmov(r, &from_table, nonzero);

    /* Main loop: windows 50 down to 0 */
    ran_jacobian tmp, fresh;
    for (int i = 50; i >= 0; i--)
    {
        /* 5 doublings */
        ran_dbl(r, r);
        ran_dbl(r, r);
        ran_dbl(r, r);
        ran_dbl(r, r);
        ran_dbl(r, r);

        /* Extract digit */
        d = (int32_t)digits[i];
        sign_mask = -(int32_t)((uint32_t)d >> 31);
        abs_d = (unsigned int)((d ^ sign_mask) - sign_mask);
        neg = (unsigned int)(sign_mask & 1);

        /* CT table lookup */
        fp_1(selected.x);
        fp_1(selected.y);
        for (unsigned int j = 0; j < 16; j++)
        {
            unsigned int eq = ((abs_d ^ (j + 1)) - 1u) >> 31;
            ran_affine_cmov(&selected, &table[j], eq);
        }
        ran_affine_cneg(&selected, neg);

        /* Mixed addition if digit != 0 */
        nonzero = 1u ^ ((abs_d - 1u) >> 31);
        unsigned int z_nonzero = (unsigned int)fp_isnonzero(r->Z);

        ran_madd(&tmp, r, &selected);
        ran_from_affine(&fresh, &selected);

        ran_cmov(r, &tmp, nonzero & z_nonzero);
        ran_cmov(r, &fresh, nonzero & (1u - z_nonzero));
    }

    /* Secure erase */
    ranshaw_secure_erase(digits, sizeof(digits));
    ranshaw_secure_erase(&selected, sizeof(selected));
    ranshaw_secure_erase(&from_table, sizeof(from_table));
    ranshaw_secure_erase(&ident, sizeof(ident));
    ranshaw_secure_erase(&tmp, sizeof(tmp));
    ranshaw_secure_erase(&fresh, sizeof(fresh));
}

#endif // RANSHAW_RAN_SCALARMULT_FIXED_H
