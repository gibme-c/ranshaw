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

/**
 * @file x64/ran_msm_vartime.cpp
 * @brief x64 multi-scalar multiplication for Ran: Straus (n<=32) and Pippenger (n>32).
 */

#include "ran_msm_vartime.h"

#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "fp_utils.h"
#include "ran_add.h"
#include "ran_dbl.h"
#include "ran_ops.h"
#include "ranshaw_secure_erase.h"

#include <cstdint>
#include <cstring>
#include <vector>

// ============================================================================
// Safe variable-time addition for Jacobian coordinates
// ============================================================================

/*
 * Variable-time "safe" addition that handles all edge cases:
 * - p == identity: return q
 * - q == identity: return p
 * - p == q: use doubling
 * - p == -q: return identity
 * - otherwise: standard addition
 */
static void ran_add_safe(ran_jacobian *r, const ran_jacobian *p, const ran_jacobian *q)
{
    if (ran_is_identity(p))
    {
        ran_copy(r, q);
        return;
    }
    if (ran_is_identity(q))
    {
        ran_copy(r, p);
        return;
    }

    /* Check if x-coordinates match (projective comparison) */
    fp_fe z1z1, z2z2, u1, u2, diff;
    fp_sq(z1z1, p->Z);
    fp_sq(z2z2, q->Z);
    fp_mul(u1, p->X, z2z2);
    fp_mul(u2, q->X, z1z1);
    fp_sub(diff, u1, u2);

    if (!fp_isnonzero(diff))
    {
        /* Same x: check if same or opposite y */
        fp_fe s1, s2, t;
        fp_mul(t, q->Z, z2z2);
        fp_mul(s1, p->Y, t);
        fp_mul(t, p->Z, z1z1);
        fp_mul(s2, q->Y, t);
        fp_sub(diff, s1, s2);

        if (!fp_isnonzero(diff))
        {
            /* P == Q: double */
            ran_dbl(r, p);
        }
        else
        {
            /* P == -Q: identity */
            ran_identity(r);
        }
        return;
    }

    ran_add(r, p, q);
}

// ============================================================================
// Signed digit encoding (curve-independent)
// ============================================================================

static void encode_signed_w4(int16_t *digits, const unsigned char *scalar)
{
    int carry = 0;
    for (int i = 0; i < 31; i++)
    {
        carry += scalar[i];
        int carry2 = (carry + 8) >> 4;
        digits[2 * i] = static_cast<int16_t>(carry - (carry2 << 4));
        carry = (carry2 + 8) >> 4;
        digits[2 * i + 1] = static_cast<int16_t>(carry2 - (carry << 4));
    }
    carry += scalar[31];
    int carry2 = (carry + 8) >> 4;
    digits[62] = static_cast<int16_t>(carry - (carry2 << 4));
    digits[63] = static_cast<int16_t>(carry2);
}

static int encode_signed_wbit(int16_t *digits, const unsigned char *scalar, int w)
{
    const int half = 1 << (w - 1);
    const int mask = (1 << w) - 1;
    const int num_digits = (256 + w - 1) / w;

    int carry = 0;
    for (int i = 0; i < num_digits; i++)
    {
        int bit_pos = i * w;
        int byte_pos = bit_pos / 8;
        int bit_off = bit_pos % 8;

        int raw = 0;
        if (byte_pos < 32)
            raw = scalar[byte_pos] >> bit_off;
        if (byte_pos + 1 < 32 && bit_off + w > 8)
            raw |= static_cast<int>(scalar[byte_pos + 1]) << (8 - bit_off);
        if (byte_pos + 2 < 32 && bit_off + w > 16)
            raw |= static_cast<int>(scalar[byte_pos + 2]) << (16 - bit_off);

        int val = (raw & mask) + carry;
        carry = val >> w;
        val &= mask;

        if (val >= half)
        {
            val -= (1 << w);
            carry = 1;
        }

        digits[i] = static_cast<int16_t>(val);
    }

    return num_digits;
}

// ============================================================================
// Straus (interleaved) method -- used for n <= 32
// ============================================================================

static void msm_straus(ran_jacobian *result, const unsigned char *scalars, const ran_jacobian *points, size_t n)
{
    std::vector<int16_t> all_digits(n * 64);
    for (size_t i = 0; i < n; i++)
    {
        encode_signed_w4(all_digits.data() + i * 64, scalars + i * 32);
    }

    // Precompute tables: table[i][j] = (j+1) * points[i]
    std::vector<ran_jacobian> tables(n * 8);
    for (size_t i = 0; i < n; i++)
    {
        ran_jacobian *Ti = tables.data() + i * 8;
        ran_copy(&Ti[0], &points[i]); // Ti[0] = 1*P
        ran_dbl(&Ti[1], &points[i]); // Ti[1] = 2*P (use dbl, not add)
        for (int j = 1; j < 7; j++)
        {
            ran_add_safe(&Ti[j + 1], &Ti[j], &points[i]); // Ti[j+1] = (j+2)*P
        }
    }

    // Main loop: process digits from most significant to least
    ran_jacobian acc;
    ran_identity(&acc);
    bool acc_is_identity = true;

    for (int d = 63; d >= 0; d--)
    {
        // 4 doublings
        if (!acc_is_identity)
        {
            ran_dbl(&acc, &acc);
            ran_dbl(&acc, &acc);
            ran_dbl(&acc, &acc);
            ran_dbl(&acc, &acc);
        }

        // Add contributions from each scalar
        for (size_t i = 0; i < n; i++)
        {
            int16_t digit = all_digits[i * 64 + (size_t)d];
            if (digit == 0)
                continue;

            ran_jacobian pt;
            if (digit > 0)
            {
                ran_copy(&pt, &tables[i * 8 + (size_t)(digit - 1)]);
            }
            else
            {
                ran_neg(&pt, &tables[i * 8 + (size_t)((-digit) - 1)]);
            }

            if (acc_is_identity)
            {
                ran_copy(&acc, &pt);
                acc_is_identity = false;
            }
            else
            {
                ran_add_safe(&acc, &acc, &pt);
            }
        }
    }

    // Defense-in-depth: erase digit encodings and precomputed tables
    ranshaw_secure_erase(all_digits.data(), all_digits.size() * sizeof(all_digits[0]));
    ranshaw_secure_erase(tables.data(), tables.size() * sizeof(tables[0]));

    if (acc_is_identity)
        ran_identity(result);
    else
        ran_copy(result, &acc);
}

// ============================================================================
// Pippenger (bucket method) -- used for n > 32
// ============================================================================

static int pippenger_window_size(size_t n)
{
    if (n < 96)
        return 5;
    if (n < 288)
        return 6;
    if (n < 864)
        return 7;
    if (n < 2592)
        return 8;
    if (n < 7776)
        return 9;
    if (n < 23328)
        return 10;
    return 11;
}

static void
    msm_pippenger(ran_jacobian *result, const unsigned char *scalars, const ran_jacobian *points, size_t n)
{
    const int w = pippenger_window_size(n);
    const size_t num_buckets = (size_t)1 << (w - 1);
    const size_t num_windows = (size_t)((256 + w - 1) / w);

    std::vector<int16_t> all_digits(n * num_windows);
    for (size_t i = 0; i < n; i++)
    {
        encode_signed_wbit(all_digits.data() + i * num_windows, scalars + i * 32, w);
    }

    ran_jacobian total;
    ran_identity(&total);
    bool total_is_identity = true;

    for (size_t win = num_windows; win-- > 0;)
    {
        // Horner step: multiply accumulated result by 2^w
        if (!total_is_identity)
        {
            for (int d = 0; d < w; d++)
                ran_dbl(&total, &total);
        }

        // Initialize buckets
        std::vector<ran_jacobian> bucket_points(num_buckets);
        std::vector<bool> bucket_is_identity(num_buckets, true);

        // Distribute points into buckets
        for (size_t i = 0; i < n; i++)
        {
            int16_t digit = all_digits[i * num_windows + win];
            if (digit == 0)
                continue;

            size_t bucket_idx;
            ran_jacobian effective_point;

            if (digit > 0)
            {
                bucket_idx = (size_t)(digit - 1);
                ran_copy(&effective_point, &points[i]);
            }
            else
            {
                bucket_idx = (size_t)((-digit) - 1);
                ran_neg(&effective_point, &points[i]);
            }

            if (bucket_is_identity[bucket_idx])
            {
                ran_copy(&bucket_points[bucket_idx], &effective_point);
                bucket_is_identity[bucket_idx] = false;
            }
            else
            {
                ran_add_safe(&bucket_points[bucket_idx], &bucket_points[bucket_idx], &effective_point);
            }
        }

        // Running-sum combination
        ran_jacobian running;
        bool running_is_identity = true;

        ran_jacobian partial;
        bool partial_is_identity = true;

        for (size_t j = num_buckets; j-- > 0;)
        {
            if (!bucket_is_identity[j])
            {
                if (running_is_identity)
                {
                    ran_copy(&running, &bucket_points[j]);
                    running_is_identity = false;
                }
                else
                {
                    ran_add_safe(&running, &running, &bucket_points[j]);
                }
            }

            if (!running_is_identity)
            {
                if (partial_is_identity)
                {
                    ran_copy(&partial, &running);
                    partial_is_identity = false;
                }
                else
                {
                    ran_add_safe(&partial, &partial, &running);
                }
            }
        }

        // Defense-in-depth: erase bucket points
        ranshaw_secure_erase(bucket_points.data(), bucket_points.size() * sizeof(bucket_points[0]));

        // Add this window's result to total
        if (!partial_is_identity)
        {
            if (total_is_identity)
            {
                ran_copy(&total, &partial);
                total_is_identity = false;
            }
            else
            {
                ran_add_safe(&total, &total, &partial);
            }
        }
    }

    // Defense-in-depth: erase digit encodings
    ranshaw_secure_erase(all_digits.data(), all_digits.size() * sizeof(all_digits[0]));

    if (total_is_identity)
        ran_identity(result);
    else
        ran_copy(result, &total);
}

// ============================================================================
// Public API (x64)
// ============================================================================

static const size_t STRAUS_PIPPENGER_CROSSOVER = 16;

void ran_msm_vartime_x64(
    ran_jacobian *result,
    const unsigned char *scalars,
    const ran_jacobian *points,
    size_t n)
{
    if (n == 0)
    {
        ran_identity(result);
        return;
    }

    if (n <= STRAUS_PIPPENGER_CROSSOVER)
    {
        msm_straus(result, scalars, points, n);
    }
    else
    {
        msm_pippenger(result, scalars, points, n);
    }
}
