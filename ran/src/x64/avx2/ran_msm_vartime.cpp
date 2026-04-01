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
 * @file x64/avx2/ran_msm_vartime.cpp
 * @brief AVX2 multi-scalar multiplication for Ran: Straus (n<=32) with 4-way parallel
 *        fp10x4 point operations, and Pippenger (n>32) with scalar fp51 point operations.
 */

#include "ran_msm_vartime.h"

#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "fp_utils.h"
#include "ran.h"
#include "ran_ops.h"
#include "ranshaw_secure_erase.h"
#include "x64/avx2/fp10_avx2.h"
#include "x64/avx2/fp10x4_avx2.h"
#include "x64/avx2/ran_avx2.h"
#include "x64/ran_add.h"
#include "x64/ran_dbl.h"

#include <cstdint>
#include <cstring>
#include <vector>

// ============================================================================
// Safe variable-time addition for Jacobian coordinates (fp51)
// ============================================================================

/*
 * Variable-time "safe" addition that handles all edge cases:
 * - p == identity: return q
 * - q == identity: return p
 * - p == q: use doubling
 * - p == -q: return identity
 * - otherwise: standard addition
 *
 * Uses the x64 baseline ran_dbl_x64/ran_add_x64 for scalar point ops.
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
            ran_dbl_x64(r, p);
        }
        else
        {
            /* P == -Q: identity */
            ran_identity(r);
        }
        return;
    }

    ran_add_x64(r, p, q);
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
// 4-Way Straus (interleaved) method -- used for n <= 32
// ============================================================================

/*
 * Process groups of 4 scalars using AVX2 4-way parallel Jacobian point ops.
 * Each group of 4 scalars shares one 4-way accumulator, using fp10x4 arithmetic
 * for doubling and addition. Table lookups use per-lane conditional moves.
 */
static void msm_straus_avx2(ran_jacobian *result, const unsigned char *scalars, const ran_jacobian *points, size_t n)
{
    // Encode all scalars into signed 4-bit digits
    std::vector<int16_t> all_digits(n * 64);
    for (size_t i = 0; i < n; i++)
    {
        encode_signed_w4(all_digits.data() + i * 64, scalars + i * 32);
    }

    // Precompute tables: table[i][j] = (j+1) * points[i], in Jacobian (fp51)
    std::vector<ran_jacobian> tables(n * 8);
    for (size_t i = 0; i < n; i++)
    {
        ran_jacobian *Ti = tables.data() + i * 8;
        ran_copy(&Ti[0], &points[i]); // Ti[0] = 1*P
        ran_dbl_x64(&Ti[1], &points[i]); // Ti[1] = 2*P
        for (size_t j = 1; j < 7; j++)
        {
            ran_add_safe(&Ti[j + 1], &Ti[j], &points[i]); // Ti[j+1] = (j+2)*P
        }
    }

    // Process groups of 4 scalars using 4-way parallel ops
    const size_t num_groups = (n + 3) / 4;

    // Build 4-way precomputed tables: tables_4x[g*8+j] = packed table entry j for group g
    std::vector<ran_jacobian_4x> tables_4x(num_groups * 8);
    for (size_t g = 0; g < num_groups; g++)
    {
        for (size_t j = 0; j < 8; j++)
        {
            ran_jacobian id;
            ran_identity(&id);

            const ran_jacobian *p0 = (g * 4 + 0 < n) ? &tables[(g * 4 + 0) * 8 + j] : &id;
            const ran_jacobian *p1 = (g * 4 + 1 < n) ? &tables[(g * 4 + 1) * 8 + j] : &id;
            const ran_jacobian *p2 = (g * 4 + 2 < n) ? &tables[(g * 4 + 2) * 8 + j] : &id;
            const ran_jacobian *p3 = (g * 4 + 3 < n) ? &tables[(g * 4 + 3) * 8 + j] : &id;
            ran_pack_4x(&tables_4x[g * 8 + j], p0, p1, p2, p3);
        }
    }

    // Main loop: process digit positions from MSB to LSB
    // Each group has its own 4-way accumulator with per-lane start tracking.
    // The raw ran_add_4x formula corrupts lanes where either input has Z=0
    // (identity), so we must use cmov to protect those lanes.
    std::vector<ran_jacobian_4x> accum(num_groups);
    std::vector<uint8_t> lane_started(num_groups, 0);

    for (int d = 63; d >= 0; d--)
    {
        // 4 doublings on all started accumulators
        for (size_t g = 0; g < num_groups; g++)
        {
            if (lane_started[g])
            {
                ran_dbl_4x(&accum[g], &accum[g]);
                ran_dbl_4x(&accum[g], &accum[g]);
                ran_dbl_4x(&accum[g], &accum[g]);
                ran_dbl_4x(&accum[g], &accum[g]);
            }
        }

        // Add contributions from each group
        for (size_t g = 0; g < num_groups; g++)
        {
            // Get the 4 digits for this group at digit position d
            int16_t d0 = (g * 4 + 0 < n) ? all_digits[(g * 4 + 0) * 64 + (size_t)d] : 0;
            int16_t d1 = (g * 4 + 1 < n) ? all_digits[(g * 4 + 1) * 64 + (size_t)d] : 0;
            int16_t d2 = (g * 4 + 2 < n) ? all_digits[(g * 4 + 2) * 64 + (size_t)d] : 0;
            int16_t d3 = (g * 4 + 3 < n) ? all_digits[(g * 4 + 3) * 64 + (size_t)d] : 0;

            if (d0 == 0 && d1 == 0 && d2 == 0 && d3 == 0)
                continue;

            // For each lane, compute |digit| and sign
            int16_t digits[4] = {d0, d1, d2, d3};
            unsigned int abs_d[4], neg[4];
            uint8_t nonzero_mask = 0;
            for (size_t k = 0; k < 4; k++)
            {
                abs_d[k] = static_cast<unsigned int>((digits[k] < 0) ? -digits[k] : digits[k]);
                neg[k] = (digits[k] < 0) ? 1u : 0u;
                if (digits[k] != 0)
                    nonzero_mask |= static_cast<uint8_t>(1u << k);
            }

            // Per-lane table selection using conditional moves
            ran_jacobian_4x selected;
            ran_identity_4x(&selected);

            for (size_t j = 0; j < 8; j++)
            {
                // Build per-lane mask: lane k selected if abs_d[k] == j+1
                unsigned int jp1 = static_cast<unsigned int>(j + 1);
                int64_t m0 = -static_cast<int64_t>(abs_d[0] == jp1);
                int64_t m1 = -static_cast<int64_t>(abs_d[1] == jp1);
                int64_t m2 = -static_cast<int64_t>(abs_d[2] == jp1);
                int64_t m3 = -static_cast<int64_t>(abs_d[3] == jp1);
                __m256i mask = _mm256_set_epi64x(m3, m2, m1, m0);
                ran_cmov_4x(&selected, &tables_4x[g * 8 + j], mask);
            }

            // Per-lane conditional negate: negate Y for lanes where digit was negative
            {
                ran_jacobian_4x neg_sel;
                ran_neg_4x(&neg_sel, &selected);
                int64_t nm0 = -static_cast<int64_t>(neg[0]);
                int64_t nm1 = -static_cast<int64_t>(neg[1]);
                int64_t nm2 = -static_cast<int64_t>(neg[2]);
                int64_t nm3 = -static_cast<int64_t>(neg[3]);
                __m256i neg_mask = _mm256_set_epi64x(nm3, nm2, nm1, nm0);
                // Blend Y coordinate: for lanes where neg, use negated Y
                for (size_t k = 0; k < 10; k++)
                    selected.Y.v[k] = _mm256_blendv_epi8(selected.Y.v[k], neg_sel.Y.v[k], neg_mask);
            }

            // Accumulate with per-lane identity protection
            uint8_t first_time = nonzero_mask & ~lane_started[g];
            uint8_t need_add = nonzero_mask & lane_started[g];

            if (need_add)
            {
                ran_jacobian_4x saved;
                ran_copy_4x(&saved, &accum[g]);
                ran_add_4x(&accum[g], &accum[g], &selected);
                // Restore accumulator for lanes where digit was 0
                uint8_t zero_lanes = lane_started[g] & ~nonzero_mask;
                if (zero_lanes)
                {
                    __m256i zmask = _mm256_set_epi64x(
                        (zero_lanes & 8) ? -1LL : 0LL,
                        (zero_lanes & 4) ? -1LL : 0LL,
                        (zero_lanes & 2) ? -1LL : 0LL,
                        (zero_lanes & 1) ? -1LL : 0LL);
                    ran_cmov_4x(&accum[g], &saved, zmask);
                }
            }

            if (first_time)
            {
                __m256i fmask = _mm256_set_epi64x(
                    (first_time & 8) ? -1LL : 0LL,
                    (first_time & 4) ? -1LL : 0LL,
                    (first_time & 2) ? -1LL : 0LL,
                    (first_time & 1) ? -1LL : 0LL);
                ran_cmov_4x(&accum[g], &selected, fmask);
            }

            lane_started[g] |= nonzero_mask;
        }
    }

    // Combine results from all groups: unpack each 4-way accumulator
    // and add the partial results together using scalar fp51 ops
    ran_jacobian total;
    ran_identity(&total);
    bool total_started = false;

    for (size_t g = 0; g < num_groups; g++)
    {
        if (!lane_started[g])
            continue;

        ran_jacobian p0, p1, p2, p3;
        ran_unpack_4x(&p0, &p1, &p2, &p3, &accum[g]);

        ran_jacobian parts[4] = {p0, p1, p2, p3};
        for (size_t k = 0; k < 4 && g * 4 + k < n; k++)
        {
            if (ran_is_identity(&parts[k]))
                continue;
            if (!total_started)
            {
                ran_copy(&total, &parts[k]);
                total_started = true;
            }
            else
            {
                ran_add_safe(&total, &total, &parts[k]);
            }
        }
    }

    // Defense-in-depth: erase digit encodings and precomputed tables
    ranshaw_secure_erase(all_digits.data(), all_digits.size() * sizeof(all_digits[0]));
    ranshaw_secure_erase(tables.data(), tables.size() * sizeof(tables[0]));

    if (total_started)
        ran_copy(result, &total);
    else
        ran_identity(result);
}

// ============================================================================
// Pippenger (bucket method) -- used for n > 32
// ============================================================================

/*
 * Pippenger uses scalar fp51 point operations for bucket accumulation.
 * The bucket-based approach doesn't benefit from 4-way grouping because
 * each point goes into a different bucket, so there's no parallelism to exploit.
 */

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

static void msm_pippenger_avx2(ran_jacobian *result, const unsigned char *scalars, const ran_jacobian *points, size_t n)
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
            for (int dd = 0; dd < w; dd++)
                ran_dbl_x64(&total, &total);
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
// Public API (AVX2)
// ============================================================================

static const size_t STRAUS_PIPPENGER_CROSSOVER = 16;

void ran_msm_vartime_avx2(ran_jacobian *result, const unsigned char *scalars, const ran_jacobian *points, size_t n)
{
    if (n == 0)
    {
        ran_identity(result);
        return;
    }

    if (n <= STRAUS_PIPPENGER_CROSSOVER)
    {
        msm_straus_avx2(result, scalars, points, n);
    }
    else
    {
        msm_pippenger_avx2(result, scalars, points, n);
    }
}
