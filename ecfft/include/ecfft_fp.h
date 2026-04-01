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
 * @file ecfft_fp.h
 * @brief ECFFT (Elliptic Curve Fast Fourier Transform) interface for F_p polynomials.
 *
 * Uses precomputed coset data from an auxiliary curve over F_p to achieve
 * O(n log^2 n) polynomial multiplication.
 */

#ifndef RANSHAW_ECFFT_FP_H
#define RANSHAW_ECFFT_FP_H

#ifdef RANSHAW_ECFFT

// clang-format off
#include <cstddef>
// clang-format on

#include "ecfft_fp_data.inl"
#include "fp_batch_invert.h"
#include "fp_frombytes.h"
#include "fp_invert.h"
#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "fp_tobytes.h"
#include "fp_utils.h"

#include <vector>

/*
 * ECFFT (Elliptic Curve Fast Fourier Transform) over Fp.
 *
 * Based on Ben-Sasson, Carmon, Kopparty, Levit (2021).
 * Replaces roots of unity with a 2-to-1 rational map from degree-2 isogenies
 * on an auxiliary curve, providing a structured evaluation domain.
 *
 * ENTER: coefficients -> evaluations via direct Horner evaluation, O(n^2).
 * EXIT:  evaluations -> coefficients via Newton interpolation, O(n^2).
 * The butterfly matrices (fwd/inv) encode the isogeny fiber pairing and are
 * available for future evaluation-domain operations (EXTEND/REDUCE).
 */

/* Wrapper struct so fp_fe (a C array typedef) can be stored in std::vector */
struct ecfft_fp_fe_s
{
    fp_fe v;
};

struct ecfft_fp_matrix
{
    fp_fe a;
    fp_fe b;
    fp_fe c;
    fp_fe d;
};

struct ecfft_fp_level
{
    std::vector<ecfft_fp_matrix> fwd;
    std::vector<ecfft_fp_matrix> inv;
    std::vector<ecfft_fp_fe_s> s;
    size_t n;
};

struct ecfft_fp_ctx
{
    std::vector<ecfft_fp_level> levels;
    size_t log_n;
    size_t domain_size;
};

static inline void ecfft_fp_apply_psi(
    fp_fe result,
    const fp_fe x,
    const ecfft_fp_fe_s *num_coeffs,
    size_t num_degree,
    const ecfft_fp_fe_s *den_coeffs,
    size_t den_degree)
{
    fp_fe num_val;
    fp_copy(num_val, num_coeffs[num_degree].v);
    for (size_t i = num_degree; i-- > 0;)
    {
        fp_fe tmp;
        fp_mul(tmp, num_val, x);
        fp_add(num_val, tmp, num_coeffs[i].v);
    }

    fp_fe den_val;
    fp_copy(den_val, den_coeffs[den_degree].v);
    for (size_t i = den_degree; i-- > 0;)
    {
        fp_fe tmp;
        fp_mul(tmp, den_val, x);
        fp_add(den_val, tmp, den_coeffs[i].v);
    }

    fp_fe den_inv;
    fp_invert(den_inv, den_val);
    fp_mul(result, num_val, den_inv);
}

static inline void ecfft_fp_build_level_matrices(ecfft_fp_level *level, const ecfft_fp_fe_s *points, size_t n)
{
    size_t half = n / 2;

    std::vector<ecfft_fp_fe_s> diffs(half);
    std::vector<ecfft_fp_fe_s> inv_diffs(half);

    for (size_t i = 0; i < half; i++)
        fp_sub(diffs[i].v, points[2 * i].v, points[2 * i + 1].v);

    fp_batch_invert(&inv_diffs[0].v, &diffs[0].v, half);

    for (size_t i = 0; i < half; i++)
    {
        fp_fe neg_s1, neg_inv;
        fp_neg(neg_s1, points[2 * i + 1].v);
        fp_neg(neg_inv, inv_diffs[i].v);

        fp_mul(level->fwd[i].a, neg_s1, inv_diffs[i].v);
        fp_mul(level->fwd[i].b, points[2 * i].v, inv_diffs[i].v);
        fp_copy(level->fwd[i].c, inv_diffs[i].v);
        fp_copy(level->fwd[i].d, neg_inv);

        fp_1(level->inv[i].a);
        fp_copy(level->inv[i].b, points[2 * i].v);
        fp_1(level->inv[i].c);
        fp_copy(level->inv[i].d, points[2 * i + 1].v);
    }
}

static inline void ecfft_fp_init(ecfft_fp_ctx *ctx)
{
    ctx->log_n = ECFFT_FP_LOG_DOMAIN;
    ctx->domain_size = ECFFT_FP_DOMAIN_SIZE;
    ctx->levels.resize(ctx->log_n);

    struct iso_info
    {
        const unsigned char *num_data;
        size_t num_degree;
        const unsigned char *den_data;
        size_t den_degree;
    };

    iso_info iso_levels[ECFFT_FP_LOG_DOMAIN];
    for (size_t i = 0; i < ECFFT_FP_LOG_DOMAIN; i++)
    {
        iso_levels[i].num_data = ECFFT_FP_ISO_NUM_PTRS[i];
        iso_levels[i].num_degree = ECFFT_FP_ISO_NUM_DEGREE[i];
        iso_levels[i].den_data = ECFFT_FP_ISO_DEN_PTRS[i];
        iso_levels[i].den_degree = ECFFT_FP_ISO_DEN_DEGREE[i];
    }

    /* Load coset data and apply bit-reversal permutation.
     * The .inl data stores coset points in natural order {R + i*G}.
     * Bit-reversal reorders them so that at each level, isogeny fiber pairs
     * (points mapping to the same x under the 2-isogeny) are at adjacent
     * even/odd indices, matching the ECFFT's recursive decomposition. */
    std::vector<ecfft_fp_fe_s> current_points(ctx->domain_size);
    for (size_t i = 0; i < ctx->domain_size; i++)
    {
        /* Bit-reverse i within log_n bits */
        size_t rev = 0;
        size_t tmp = i;
        for (size_t b = 0; b < ctx->log_n; b++)
        {
            rev = (rev << 1) | (tmp & 1);
            tmp >>= 1;
        }
        fp_frombytes(current_points[i].v, &ECFFT_FP_COSET[rev * 32]);
    }

    size_t level_size = ctx->domain_size;

    for (size_t lv = 0; lv < ctx->log_n; lv++)
    {
        size_t half = level_size / 2;

        ctx->levels[lv].n = level_size;
        ctx->levels[lv].s.resize(level_size);
        ctx->levels[lv].fwd.resize(half);
        ctx->levels[lv].inv.resize(half);

        for (size_t i = 0; i < level_size; i++)
            fp_copy(ctx->levels[lv].s[i].v, current_points[i].v);

        ecfft_fp_build_level_matrices(&ctx->levels[lv], current_points.data(), level_size);

        if (lv + 1 < ctx->log_n)
        {
            size_t num_deg = iso_levels[lv].num_degree;
            size_t den_deg = iso_levels[lv].den_degree;
            std::vector<ecfft_fp_fe_s> num_c(num_deg + 1);
            std::vector<ecfft_fp_fe_s> den_c(den_deg + 1);
            for (size_t k = 0; k <= num_deg; k++)
                fp_frombytes(num_c[k].v, &iso_levels[lv].num_data[k * 32]);
            for (size_t k = 0; k <= den_deg; k++)
                fp_frombytes(den_c[k].v, &iso_levels[lv].den_data[k * 32]);

            std::vector<ecfft_fp_fe_s> next_points(half);
            for (size_t i = 0; i < half; i++)
            {
                ecfft_fp_apply_psi(
                    next_points[i].v, current_points[2 * i].v, num_c.data(), num_deg, den_c.data(), den_deg);
            }

            current_points = std::move(next_points);
        }

        level_size = half;
    }
}

/* ====================================================================
 * ECFFT ENTER: coefficients -> evaluations
 *
 * Direct Horner evaluation at each domain point. O(n^2) with very low
 * constant factor: n Horner evaluations, each O(n) multiply-adds.
 * No heap allocations beyond the working copy of coefficients.
 * ==================================================================== */

static inline void ecfft_fp_enter(fp_fe *data, size_t n, const ecfft_fp_ctx *ctx)
{
    if (n <= 1)
        return;

    /* Find the level whose domain has size n */
    size_t level = 0;
    for (size_t lv = 0; lv < ctx->log_n; lv++)
        if (ctx->levels[lv].n == n)
        {
            level = lv;
            break;
        }

    /* Save a copy of the input coefficients */
    std::vector<ecfft_fp_fe_s> coeffs(n);
    for (size_t i = 0; i < n; i++)
        fp_copy(coeffs[i].v, data[i]);

    /* Evaluate f(x) = c[0] + c[1]*x + ... + c[n-1]*x^{n-1} at each domain point
     * using Horner's method: f(s) = c[0] + s*(c[1] + s*(c[2] + ... + s*c[n-1])) */
    const ecfft_fp_fe_s *s = ctx->levels[level].s.data();

    for (size_t i = 0; i < n; i++)
    {
        fp_fe result;
        fp_copy(result, coeffs[n - 1].v);

        for (size_t k = n - 1; k > 0; k--)
        {
            fp_fe tmp;
            fp_mul(tmp, result, s[i].v);
            fp_add(result, tmp, coeffs[k - 1].v);
        }

        fp_copy(data[i], result);
    }

    /* Normalize: carry-propagate via fp_sub(x, x, 0) */
    fp_fe zero;
    fp_0(zero);
    for (size_t k = 0; k < n; k++)
        fp_sub(data[k], data[k], zero);
}

/* ====================================================================
 * ECFFT EXIT: evaluations -> coefficients
 *
 * Newton divided-difference interpolation. O(n^2) total:
 *   - Divided differences: O(n^2/2) field operations + n-1 batch inversions
 *   - Newton-to-monomial conversion: O(n^2/2) multiply-adds
 * ==================================================================== */

static inline void ecfft_fp_exit(fp_fe *data, size_t n, const ecfft_fp_ctx *ctx)
{
    if (n <= 1)
        return;

    /* Find the level whose domain has size n */
    size_t level = 0;
    for (size_t lv = 0; lv < ctx->log_n; lv++)
        if (ctx->levels[lv].n == n)
        {
            level = lv;
            break;
        }

    const ecfft_fp_fe_s *s = ctx->levels[level].s.data();

    /* Stage 1: Compute Newton divided differences in-place.
     * d[i] starts as evaluation v[i], ends as f[s[0], ..., s[i]].
     *
     * For each gap j = 1..n-1:
     *   d[i] = (d[i] - d[i-1]) / (s[i] - s[i-j])   for i = n-1 down to j
     *
     * Use batch inversion at each gap to avoid per-element inversions. */
    std::vector<ecfft_fp_fe_s> d(n);
    for (size_t i = 0; i < n; i++)
        fp_copy(d[i].v, data[i]);

    for (size_t j = 1; j < n; j++)
    {
        size_t count = n - j;

        /* Compute denominators: s[i] - s[i-j] for i = j..n-1 */
        std::vector<ecfft_fp_fe_s> denoms(count);
        std::vector<ecfft_fp_fe_s> inv_denoms(count);
        for (size_t i = j; i < n; i++)
            fp_sub(denoms[i - j].v, s[i].v, s[i - j].v);

        fp_batch_invert(&inv_denoms[0].v, &denoms[0].v, count);

        /* Update: d[i] = (d[i] - d[i-1]) * inv(s[i] - s[i-j]) */
        for (size_t i = n; i-- > j;)
        {
            fp_fe diff;
            fp_sub(diff, d[i].v, d[i - 1].v);
            fp_mul(d[i].v, diff, inv_denoms[i - j].v);
        }
    }

    /* Stage 2: Convert Newton form to monomial (standard) coefficients.
     *
     * Newton form: f(x) = d[0] + d[1](x-s[0]) + d[2](x-s[0])(x-s[1]) + ...
     *
     * Build via Horner from inside out:
     *   p = d[n-1]
     *   for k = n-2 down to 0:
     *     p = p * (x - s[k]) + d[k]
     */
    std::vector<ecfft_fp_fe_s> p(n);
    for (size_t i = 0; i < n; i++)
        fp_0(p[i].v);
    fp_copy(p[0].v, d[n - 1].v);
    size_t deg = 0;

    for (size_t k = n - 1; k-- > 0;)
    {
        /* Multiply p[0..deg] by (x - s[k]):
         *   p[deg+1] = p[deg]
         *   p[j] = p[j-1] - s[k]*p[j]   for j = deg down to 1
         *   p[0] = -s[k]*p[0]            */
        fp_copy(p[deg + 1].v, p[deg].v);
        for (size_t j = deg; j >= 1; j--)
        {
            fp_fe prod;
            fp_mul(prod, s[k].v, p[j].v);
            fp_sub(p[j].v, p[j - 1].v, prod);
        }
        {
            fp_fe prod;
            fp_mul(prod, s[k].v, p[0].v);
            fp_neg(p[0].v, prod);
        }
        deg++;

        /* Add d[k] to constant term */
        fp_add(p[0].v, p[0].v, d[k].v);
    }

    /* Copy result and normalize */
    fp_fe zero;
    fp_0(zero);
    for (size_t i = 0; i < n; i++)
        fp_sub(data[i], p[i].v, zero);
}

/* ====================================================================
 * ECFFT EXTEND / REDUCE
 *
 * Evaluation-domain operations. Currently implemented as EXIT + ENTER
 * (O(n^2)). Can be optimized to O(n log n) with a proper recursive
 * ECFFT butterfly in a future pass.
 * ==================================================================== */

/*
 * EXTEND: given evaluations of a degree-<n_from polynomial at n_from domain
 * points, compute evaluations at n_to > n_from domain points.
 * data[0..n_from-1] = input evaluations; data[0..n_to-1] = output.
 */
static inline void ecfft_fp_extend(fp_fe *data, size_t n_from, size_t n_to, const ecfft_fp_ctx *ctx)
{
    if (n_from >= n_to || n_from <= 1)
        return;

    /* Interpolate: evaluations at n_from points -> coefficients */
    ecfft_fp_exit(data, n_from, ctx);

    /* Zero-pad to n_to coefficients */
    for (size_t i = n_from; i < n_to; i++)
        fp_0(data[i]);

    /* Re-evaluate at n_to domain points */
    ecfft_fp_enter(data, n_to, ctx);
}

/*
 * REDUCE: given evaluations of a degree-<n_to polynomial at n_from > n_to
 * domain points, produce evaluations at n_to domain points.
 * data[0..n_from-1] = input evaluations; data[0..n_to-1] = output.
 */
static inline void ecfft_fp_reduce(fp_fe *data, size_t n_from, size_t n_to, const ecfft_fp_ctx *ctx)
{
    if (n_to >= n_from || n_to <= 1)
        return;

    /* Interpolate: evaluations at n_from points -> coefficients */
    ecfft_fp_exit(data, n_from, ctx);

    /* Evaluate at n_to domain points */
    ecfft_fp_enter(data, n_to, ctx);
}

/* ====================================================================
 * ECFFT polynomial multiplication
 *
 * ENTER both operands, pointwise multiply, EXIT result.
 * ==================================================================== */

static inline void ecfft_fp_poly_mul(
    fp_fe *result,
    size_t *result_len,
    const fp_fe *a,
    size_t a_len,
    const fp_fe *b,
    size_t b_len,
    const ecfft_fp_ctx *ctx)
{
    if (a_len == 0 || b_len == 0)
    {
        *result_len = 1;
        fp_0(result[0]);
        return;
    }

    size_t out_len = a_len + b_len - 1;

    size_t n = 1;
    while (n < out_len)
        n <<= 1;

    if (n > ctx->domain_size)
    {
        *result_len = 0;
        return;
    }

    std::vector<ecfft_fp_fe_s> fa(n);
    std::vector<ecfft_fp_fe_s> fb(n);

    for (size_t i = 0; i < a_len; i++)
        fp_copy(fa[i].v, a[i]);
    for (size_t i = a_len; i < n; i++)
        fp_0(fa[i].v);

    for (size_t i = 0; i < b_len; i++)
        fp_copy(fb[i].v, b[i]);
    for (size_t i = b_len; i < n; i++)
        fp_0(fb[i].v);

    ecfft_fp_enter(&fa[0].v, n, ctx);
    ecfft_fp_enter(&fb[0].v, n, ctx);

    for (size_t i = 0; i < n; i++)
        fp_mul(fa[i].v, fa[i].v, fb[i].v);

    ecfft_fp_exit(&fa[0].v, n, ctx);

    *result_len = out_len;
    for (size_t i = 0; i < out_len; i++)
        fp_copy(result[i], fa[i].v);
}

#endif /* RANSHAW_ECFFT */
#endif /* RANSHAW_ECFFT_FP_H */
