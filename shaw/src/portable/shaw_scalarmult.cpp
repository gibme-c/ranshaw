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

#include "shaw_scalarmult.h"

#include "fq_invert.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "ranshaw_secure_erase.h"
#include "shaw.h"
#include "shaw_add.h"
#include "shaw_dbl.h"
#include "shaw_madd.h"
#include "shaw_ops.h"

#include <vector>

/*
 * Constant-time scalar multiplication for Shaw (over F_q).
 * Same algorithm as ran_scalarmult but using fq_* field ops.
 */

static void scalar_recode_signed4(int8_t digits[64], const unsigned char scalar[32])
{
    uint8_t nibbles[64];
    for (int i = 0; i < 32; i++)
    {
        nibbles[2 * i] = scalar[i] & 0x0f;
        nibbles[2 * i + 1] = (scalar[i] >> 4) & 0x0f;
    }

    int carry = 0;
    for (int i = 0; i < 63; i++)
    {
        int val = nibbles[i] + carry;
        carry = (val + 8) >> 4;
        digits[i] = (int8_t)(val - (carry << 4));
    }
    digits[63] = (int8_t)(nibbles[63] + carry);
    ranshaw_secure_erase(nibbles, sizeof(nibbles));
}

static void batch_to_affine(shaw_affine *out, const shaw_jacobian *in, size_t n)
{
    if (n == 0)
        return;

    struct fq_fe_s
    {
        fq_fe v;
    };
    std::vector<fq_fe_s> z_vals(n);
    std::vector<fq_fe_s> products(n);

    for (size_t i = 0; i < n; i++)
        fq_copy(z_vals[i].v, in[i].Z);

    fq_copy(products[0].v, z_vals[0].v);
    for (size_t i = 1; i < n; i++)
        fq_mul(products[i].v, products[i - 1].v, z_vals[i].v);

    fq_fe inv;
    fq_invert(inv, products[n - 1].v);

    for (size_t i = n - 1; i > 0; i--)
    {
        fq_fe z_inv;
        fq_mul(z_inv, inv, products[i - 1].v);
        fq_mul(inv, inv, z_vals[i].v);

        fq_fe z_inv2, z_inv3;
        fq_sq(z_inv2, z_inv);
        fq_mul(z_inv3, z_inv2, z_inv);
        fq_mul(out[i].x, in[i].X, z_inv2);
        fq_mul(out[i].y, in[i].Y, z_inv3);
    }

    {
        fq_fe z_inv2, z_inv3;
        fq_sq(z_inv2, inv);
        fq_mul(z_inv3, z_inv2, inv);
        fq_mul(out[0].x, in[0].X, z_inv2);
        fq_mul(out[0].y, in[0].Y, z_inv3);
    }

    ranshaw_secure_erase(&inv, sizeof(inv));
    ranshaw_secure_erase(z_vals.data(), n * sizeof(fq_fe_s));
    ranshaw_secure_erase(products.data(), n * sizeof(fq_fe_s));
}

void shaw_scalarmult_portable(shaw_jacobian *r, const unsigned char scalar[32], const shaw_jacobian *p)
{
    shaw_jacobian table_jac[8];
    shaw_copy(&table_jac[0], p);
    shaw_dbl(&table_jac[1], p);
    shaw_add(&table_jac[2], &table_jac[1], p);
    shaw_dbl(&table_jac[3], &table_jac[1]);
    shaw_add(&table_jac[4], &table_jac[3], p);
    shaw_dbl(&table_jac[5], &table_jac[2]);
    shaw_add(&table_jac[6], &table_jac[5], p);
    shaw_dbl(&table_jac[7], &table_jac[3]);

    shaw_affine table[8];
    batch_to_affine(table, table_jac, 8);

    int8_t digits[64];
    scalar_recode_signed4(digits, scalar);

    int32_t d = (int32_t)digits[63];
    int32_t sign_mask = -(int32_t)((uint32_t)d >> 31);
    unsigned int abs_d = (unsigned int)((d ^ sign_mask) - sign_mask);
    unsigned int neg = (unsigned int)(sign_mask & 1);

    shaw_affine selected;
    fq_0(selected.x);
    fq_0(selected.y);

    for (unsigned int j = 0; j < 8; j++)
    {
        unsigned int eq = ((abs_d ^ (j + 1)) - 1u) >> 31;
        shaw_affine_cmov(&selected, &table[j], eq);
    }

    /* CT conditional negate + select identity or table point */
    shaw_affine_cneg(&selected, neg);

    shaw_jacobian from_table;
    shaw_from_affine(&from_table, &selected);

    shaw_jacobian ident;
    shaw_identity(&ident);

    unsigned int nonzero = 1u ^ ((abs_d - 1u) >> 31);
    shaw_copy(r, &ident);
    shaw_cmov(r, &from_table, nonzero);

    shaw_jacobian tmp, fresh;
    for (int i = 62; i >= 0; i--)
    {
        shaw_dbl(r, r);
        shaw_dbl(r, r);
        shaw_dbl(r, r);
        shaw_dbl(r, r);

        d = (int32_t)digits[i];
        sign_mask = -(int32_t)((uint32_t)d >> 31);
        abs_d = (unsigned int)((d ^ sign_mask) - sign_mask);
        neg = (unsigned int)(sign_mask & 1);

        fq_1(selected.x);
        fq_1(selected.y);
        for (unsigned int j = 0; j < 8; j++)
        {
            unsigned int eq = ((abs_d ^ (j + 1)) - 1u) >> 31;
            shaw_affine_cmov(&selected, &table[j], eq);
        }

        shaw_affine_cneg(&selected, neg);

        nonzero = 1u ^ ((abs_d - 1u) >> 31);
        unsigned int z_nonzero = (unsigned int)fq_isnonzero(r->Z);

        shaw_madd(&tmp, r, &selected);

        shaw_from_affine(&fresh, &selected);

        shaw_cmov(r, &tmp, nonzero & z_nonzero);
        shaw_cmov(r, &fresh, nonzero & (1u - z_nonzero));
    }

    ranshaw_secure_erase(&selected, sizeof(selected));
    ranshaw_secure_erase(&from_table, sizeof(from_table));
    ranshaw_secure_erase(&ident, sizeof(ident));
    ranshaw_secure_erase(&tmp, sizeof(tmp));
    ranshaw_secure_erase(&fresh, sizeof(fresh));
    ranshaw_secure_erase(table_jac, sizeof(table_jac));
    ranshaw_secure_erase(table, sizeof(table));
    ranshaw_secure_erase(digits, sizeof(digits));
}
