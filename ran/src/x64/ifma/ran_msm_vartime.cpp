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
 * @file x64/ifma/ran_msm_vartime.cpp
 * @brief AVX-512 IFMA 8-way parallel MSM for Ran: Straus (n<=32) and Pippenger (n>32).
 *
 * Straus uses 8-way parallel fp51x8 point operations (ran_dbl_8x, ran_add_8x)
 * to process 8 independent scalar multiplications simultaneously. Points are packed
 * into ran_jacobian_8x structures, and per-lane table selection uses AVX-512 k-masks.
 *
 * Pippenger falls back to scalar x64 baseline point operations (ran_dbl_x64,
 * ran_add_x64) because the bucket accumulation method does not benefit from
 * lane-level parallelism.
 */

#include "ran_msm_vartime.h"

#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "fp_utils.h"
#include "ran_ops.h"
#include "ranshaw_secure_erase.h"
#include "x64/ifma/fp51x8_ifma.h"
#include "x64/ifma/ran_ifma.h"
#include "x64/ran_add.h"
#include "x64/ran_dbl.h"

#include <cstdint>
#include <cstring>
#include <vector>

// ============================================================================
// Safe variable-time addition for Jacobian coordinates (scalar fp51 ops)
// ============================================================================

/*
 * Variable-time "safe" addition that handles all edge cases:
 * - p == identity: return q
 * - q == identity: return p
 * - p == q: use doubling
 * - p == -q: return identity
 * - otherwise: standard addition
 *
 * Uses x64 baseline scalar ops (not dispatch table) since this file is
 * compiled with AVX-512 flags and we need the x64 implementations directly.
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
// Straus (interleaved) method with 8-way IFMA parallelism -- used for n <= 32
// ============================================================================

/*
 * 8-way parallel Straus MSM. Groups of 8 scalars are processed in parallel
 * using fp51x8 SIMD point operations. Each group of 8 shares a single
 * 8-way accumulator; after all digit positions are processed, the 8 results
 * are unpacked and combined with scalar additions.
 *
 * Precomputation: build scalar (fp51) tables for each point, then pack
 * groups of 8 table entries into ran_jacobian_8x structures.
 *
 * Main loop: for each digit position (63 down to 0):
 *   1. Double the 8-way accumulator 4 times (w=4 window)
 *   2. For each group, build a per-lane k-mask selection from the 8 table
 *      entries, conditionally negate per lane, and add to the accumulator
 *
 * Table selection uses AVX-512 k-mask conditional moves (ran_cmov_8x):
 * for table index j (1..8), a k-mask is built where bit k is set if
 * |digit[k]| == j. This selects the correct table entry per lane without
 * branches.
 */
static void msm_straus_ifma(ran_jacobian *result, const unsigned char *scalars, const ran_jacobian *points, size_t n)
{
    // Encode all scalars into signed w=4 digits
    std::vector<int16_t> all_digits(n * 64);
    for (size_t i = 0; i < n; i++)
    {
        encode_signed_w4(all_digits.data() + i * 64, scalars + i * 32);
    }

    // Precompute scalar tables: table[i][j] = (j+1) * points[i], j=0..7
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

    // Number of groups of 8
    const size_t num_groups = (n + 7) / 8;

    // Pack tables into 8-way format: tables_8x[g*8+j] holds table entry j
    // for group g, with up to 8 lanes populated (identity for padding lanes)
    std::vector<ran_jacobian_8x> tables_8x(num_groups * 8);
    {
        ran_jacobian id;
        ran_identity(&id);

        for (size_t g = 0; g < num_groups; g++)
        {
            for (size_t j = 0; j < 8; j++)
            {
                const ran_jacobian *pts[8];
                for (size_t k = 0; k < 8; k++)
                {
                    pts[k] = (g * 8 + k < n) ? &tables[(g * 8 + k) * 8 + j] : &id;
                }

                ran_pack_8x(&tables_8x[g * 8 + j], pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], pts[6], pts[7]);
            }
        }
    }

    // Per-group 8-way accumulators with per-lane start tracking.
    // lane_started tracks which lanes have received at least one nonzero digit.
    // The raw ran_add_8x formula corrupts lanes where either input has Z=0
    // (identity), so we must use cmov to protect those lanes.
    std::vector<ran_jacobian_8x> accum(num_groups);
    std::vector<__mmask8> lane_started(num_groups, 0);

    // Main loop: process digit positions from most significant to least
    for (int d = 63; d >= 0; d--)
    {
        // 4 doublings per digit position (w=4 window)
        for (size_t g = 0; g < num_groups; g++)
        {
            if (lane_started[g])
            {
                ran_dbl_8x(&accum[g], &accum[g]);
                ran_dbl_8x(&accum[g], &accum[g]);
                ran_dbl_8x(&accum[g], &accum[g]);
                ran_dbl_8x(&accum[g], &accum[g]);
            }
        }

        // Add contributions for each group
        for (size_t g = 0; g < num_groups; g++)
        {
            // Gather the 8 digits for this group at this position
            int16_t digits[8];
            unsigned int abs_d[8];
            unsigned int neg_flag[8];
            __mmask8 nonzero_mask = 0;

            for (size_t k = 0; k < 8; k++)
            {
                digits[k] = (g * 8 + k < n) ? all_digits[(g * 8 + k) * 64 + (size_t)d] : 0;
                abs_d[k] = static_cast<unsigned int>((digits[k] < 0) ? -digits[k] : digits[k]);
                neg_flag[k] = (digits[k] < 0) ? 1u : 0u;
                if (digits[k] != 0)
                    nonzero_mask |= static_cast<__mmask8>(1u << k);
            }

            if (!nonzero_mask)
                continue;

            // Per-lane table selection using k-masks:
            // Start with identity, then for each table index j (1..8), build a
            // mask of lanes whose |digit| == j and conditionally move that table
            // entry into those lanes.
            ran_jacobian_8x selected;
            ran_identity_8x(&selected);

            for (size_t j = 0; j < 8; j++)
            {
                __mmask8 mask = 0;
                unsigned int jp1 = static_cast<unsigned int>(j + 1);
                for (size_t k = 0; k < 8; k++)
                {
                    if (abs_d[k] == jp1)
                        mask |= static_cast<__mmask8>(1u << k);
                }

                if (mask)
                    ran_cmov_8x(&selected, &tables_8x[g * 8 + j], mask);
            }

            // Per-lane conditional negate: for lanes where digit < 0, negate Y
            {
                __mmask8 neg_mask = 0;
                for (size_t k = 0; k < 8; k++)
                {
                    if (neg_flag[k])
                        neg_mask |= static_cast<__mmask8>(1u << k);
                }

                if (neg_mask)
                {
                    ran_jacobian_8x neg_sel;
                    ran_neg_8x(&neg_sel, &selected);
                    fp51x8_cmov(&selected.Y, &neg_sel.Y, neg_mask);
                }
            }

            // Accumulate with per-lane identity protection:
            // - Lanes not yet started: copy selected directly into those lanes
            // - Lanes already started with nonzero digit: normal add
            // - Lanes already started with zero digit: preserve accumulator
            __mmask8 first_time = nonzero_mask & ~lane_started[g];
            __mmask8 need_add = nonzero_mask & lane_started[g];

            if (need_add)
            {
                ran_jacobian_8x saved;
                ran_copy_8x(&saved, &accum[g]);
                ran_add_8x(&accum[g], &accum[g], &selected);
                // Restore accumulator for lanes where digit was 0
                __mmask8 zero_mask = lane_started[g] & ~nonzero_mask;
                if (zero_mask)
                    ran_cmov_8x(&accum[g], &saved, zero_mask);
            }

            // For lanes seeing their first nonzero digit, copy selected directly
            if (first_time)
                ran_cmov_8x(&accum[g], &selected, first_time);

            lane_started[g] |= nonzero_mask;
        }
    }

    // Combine all groups: unpack each 8-way accumulator and sum the individual
    // results with scalar additions
    ran_jacobian total;
    ran_identity(&total);
    bool total_started = false;

    for (size_t g = 0; g < num_groups; g++)
    {
        if (!lane_started[g])
            continue;

        ran_jacobian parts[8];
        ran_unpack_8x(
            &parts[0], &parts[1], &parts[2], &parts[3], &parts[4], &parts[5], &parts[6], &parts[7], &accum[g]);

        for (size_t k = 0; k < 8 && g * 8 + k < n; k++)
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
// Pippenger (bucket method) using scalar x64 ops -- used for n > 32
// ============================================================================

/*
 * Pippenger's bucket method does not benefit from 8-way lane parallelism
 * because bucket accumulation involves irregular scatter-gather patterns
 * (each point goes to a different bucket based on its digit). Instead, we
 * use the x64 baseline scalar point operations which are already efficient
 * for this access pattern.
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

static void msm_pippenger_ifma(ran_jacobian *result, const unsigned char *scalars, const ran_jacobian *points, size_t n)
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
// Public API (IFMA)
// ============================================================================

static const size_t STRAUS_PIPPENGER_CROSSOVER = 16;

void ran_msm_vartime_ifma(ran_jacobian *result, const unsigned char *scalars, const ran_jacobian *points, size_t n)
{
    if (n == 0)
    {
        ran_identity(result);
        return;
    }

    if (n <= STRAUS_PIPPENGER_CROSSOVER)
    {
        msm_straus_ifma(result, scalars, points, n);
    }
    else
    {
        msm_pippenger_ifma(result, scalars, points, n);
    }
}
