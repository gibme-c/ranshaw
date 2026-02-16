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
 * @file ecfft_fq.h
 * @brief ECFFT (Elliptic Curve Fast Fourier Transform) interface for F_q polynomials.
 *
 * Uses precomputed coset data from an auxiliary curve over F_q to achieve
 * O(n log^2 n) polynomial multiplication.
 */

#ifndef RANSHAW_ECFFT_FQ_H
#define RANSHAW_ECFFT_FQ_H

#ifdef RANSHAW_ECFFT

// clang-format off
#include <cstddef>
// clang-format on

#include "ecfft_fq_data.inl"
#include "fq_batch_invert.h"
#include "fq_frombytes.h"
#include "fq_invert.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "fq_tobytes.h"
#include "fq_utils.h"

#include <vector>

/*
 * ECFFT (Elliptic Curve Fast Fourier Transform) over Fq.
 *
 * Mirror of ecfft_fp.h for the Fq field. See ecfft_fp.h for documentation.
 */

struct ecfft_fq_fe_s
{
    fq_fe v;
};

struct ecfft_fq_matrix
{
    fq_fe a;
    fq_fe b;
    fq_fe c;
    fq_fe d;
};

struct ecfft_fq_level
{
    std::vector<ecfft_fq_matrix> fwd;
    std::vector<ecfft_fq_matrix> inv;
    std::vector<ecfft_fq_fe_s> s;
    size_t n;
};

struct ecfft_fq_ctx
{
    std::vector<ecfft_fq_level> levels;
    size_t log_n;
    size_t domain_size;
};

static inline void ecfft_fq_apply_psi(
    fq_fe result,
    const fq_fe x,
    const ecfft_fq_fe_s *num_coeffs,
    size_t num_degree,
    const ecfft_fq_fe_s *den_coeffs,
    size_t den_degree)
{
    fq_fe num_val;
    fq_copy(num_val, num_coeffs[num_degree].v);
    for (size_t i = num_degree; i-- > 0;)
    {
        fq_fe tmp;
        fq_mul(tmp, num_val, x);
        fq_add(num_val, tmp, num_coeffs[i].v);
    }

    fq_fe den_val;
    fq_copy(den_val, den_coeffs[den_degree].v);
    for (size_t i = den_degree; i-- > 0;)
    {
        fq_fe tmp;
        fq_mul(tmp, den_val, x);
        fq_add(den_val, tmp, den_coeffs[i].v);
    }

    fq_fe den_inv;
    fq_invert(den_inv, den_val);
    fq_mul(result, num_val, den_inv);
}

static inline void ecfft_fq_build_level_matrices(ecfft_fq_level *level, const ecfft_fq_fe_s *points, size_t n)
{
    size_t half = n / 2;

    std::vector<ecfft_fq_fe_s> diffs(half);
    std::vector<ecfft_fq_fe_s> inv_diffs(half);

    for (size_t i = 0; i < half; i++)
        fq_sub(diffs[i].v, points[2 * i].v, points[2 * i + 1].v);

    fq_batch_invert(&inv_diffs[0].v, &diffs[0].v, half);

    for (size_t i = 0; i < half; i++)
    {
        fq_fe neg_s1, neg_inv;
        fq_neg(neg_s1, points[2 * i + 1].v);
        fq_neg(neg_inv, inv_diffs[i].v);

        fq_mul(level->fwd[i].a, neg_s1, inv_diffs[i].v);
        fq_mul(level->fwd[i].b, points[2 * i].v, inv_diffs[i].v);
        fq_copy(level->fwd[i].c, inv_diffs[i].v);
        fq_copy(level->fwd[i].d, neg_inv);

        fq_1(level->inv[i].a);
        fq_copy(level->inv[i].b, points[2 * i].v);
        fq_1(level->inv[i].c);
        fq_copy(level->inv[i].d, points[2 * i + 1].v);
    }
}

static inline void ecfft_fq_init(ecfft_fq_ctx *ctx)
{
    ctx->log_n = ECFFT_FQ_LOG_DOMAIN;
    ctx->domain_size = ECFFT_FQ_DOMAIN_SIZE;
    ctx->levels.resize(ctx->log_n);

    struct iso_info
    {
        const unsigned char *num_data;
        size_t num_degree;
        const unsigned char *den_data;
        size_t den_degree;
    };

    iso_info iso_levels[ECFFT_FQ_LOG_DOMAIN] = {
        {ECFFT_FQ_ISO_NUM_0, ECFFT_FQ_ISO_NUM_DEGREE[0], ECFFT_FQ_ISO_DEN_0, ECFFT_FQ_ISO_DEN_DEGREE[0]},
        {ECFFT_FQ_ISO_NUM_1, ECFFT_FQ_ISO_NUM_DEGREE[1], ECFFT_FQ_ISO_DEN_1, ECFFT_FQ_ISO_DEN_DEGREE[1]},
        {ECFFT_FQ_ISO_NUM_2, ECFFT_FQ_ISO_NUM_DEGREE[2], ECFFT_FQ_ISO_DEN_2, ECFFT_FQ_ISO_DEN_DEGREE[2]},
        {ECFFT_FQ_ISO_NUM_3, ECFFT_FQ_ISO_NUM_DEGREE[3], ECFFT_FQ_ISO_DEN_3, ECFFT_FQ_ISO_DEN_DEGREE[3]},
        {ECFFT_FQ_ISO_NUM_4, ECFFT_FQ_ISO_NUM_DEGREE[4], ECFFT_FQ_ISO_DEN_4, ECFFT_FQ_ISO_DEN_DEGREE[4]},
        {ECFFT_FQ_ISO_NUM_5, ECFFT_FQ_ISO_NUM_DEGREE[5], ECFFT_FQ_ISO_DEN_5, ECFFT_FQ_ISO_DEN_DEGREE[5]},
        {ECFFT_FQ_ISO_NUM_6, ECFFT_FQ_ISO_NUM_DEGREE[6], ECFFT_FQ_ISO_DEN_6, ECFFT_FQ_ISO_DEN_DEGREE[6]},
        {ECFFT_FQ_ISO_NUM_7, ECFFT_FQ_ISO_NUM_DEGREE[7], ECFFT_FQ_ISO_DEN_7, ECFFT_FQ_ISO_DEN_DEGREE[7]},
        {ECFFT_FQ_ISO_NUM_8, ECFFT_FQ_ISO_NUM_DEGREE[8], ECFFT_FQ_ISO_DEN_8, ECFFT_FQ_ISO_DEN_DEGREE[8]},
        {ECFFT_FQ_ISO_NUM_9, ECFFT_FQ_ISO_NUM_DEGREE[9], ECFFT_FQ_ISO_DEN_9, ECFFT_FQ_ISO_DEN_DEGREE[9]},
        {ECFFT_FQ_ISO_NUM_10, ECFFT_FQ_ISO_NUM_DEGREE[10], ECFFT_FQ_ISO_DEN_10, ECFFT_FQ_ISO_DEN_DEGREE[10]},
        {ECFFT_FQ_ISO_NUM_11, ECFFT_FQ_ISO_NUM_DEGREE[11], ECFFT_FQ_ISO_DEN_11, ECFFT_FQ_ISO_DEN_DEGREE[11]},
        {ECFFT_FQ_ISO_NUM_12, ECFFT_FQ_ISO_NUM_DEGREE[12], ECFFT_FQ_ISO_DEN_12, ECFFT_FQ_ISO_DEN_DEGREE[12]},
        {ECFFT_FQ_ISO_NUM_13, ECFFT_FQ_ISO_NUM_DEGREE[13], ECFFT_FQ_ISO_DEN_13, ECFFT_FQ_ISO_DEN_DEGREE[13]},
        {ECFFT_FQ_ISO_NUM_14, ECFFT_FQ_ISO_NUM_DEGREE[14], ECFFT_FQ_ISO_DEN_14, ECFFT_FQ_ISO_DEN_DEGREE[14]},
        {ECFFT_FQ_ISO_NUM_15, ECFFT_FQ_ISO_NUM_DEGREE[15], ECFFT_FQ_ISO_DEN_15, ECFFT_FQ_ISO_DEN_DEGREE[15]},
    };

    std::vector<ecfft_fq_fe_s> current_points(ctx->domain_size);
    for (size_t i = 0; i < ctx->domain_size; i++)
    {
        size_t rev = 0;
        size_t tmp = i;
        for (size_t b = 0; b < ctx->log_n; b++)
        {
            rev = (rev << 1) | (tmp & 1);
            tmp >>= 1;
        }
        fq_frombytes(current_points[i].v, &ECFFT_FQ_COSET[rev * 32]);
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
            fq_copy(ctx->levels[lv].s[i].v, current_points[i].v);

        ecfft_fq_build_level_matrices(&ctx->levels[lv], current_points.data(), level_size);

        if (lv + 1 < ctx->log_n)
        {
            size_t num_deg = iso_levels[lv].num_degree;
            size_t den_deg = iso_levels[lv].den_degree;
            std::vector<ecfft_fq_fe_s> num_c(num_deg + 1);
            std::vector<ecfft_fq_fe_s> den_c(den_deg + 1);
            for (size_t k = 0; k <= num_deg; k++)
                fq_frombytes(num_c[k].v, &iso_levels[lv].num_data[k * 32]);
            for (size_t k = 0; k <= den_deg; k++)
                fq_frombytes(den_c[k].v, &iso_levels[lv].den_data[k * 32]);

            std::vector<ecfft_fq_fe_s> next_points(half);
            for (size_t i = 0; i < half; i++)
            {
                ecfft_fq_apply_psi(
                    next_points[i].v, current_points[2 * i].v, num_c.data(), num_deg, den_c.data(), den_deg);
            }

            current_points = std::move(next_points);
        }

        level_size = half;
    }
}

/* ====================================================================
 * ECFFT ENTER: coefficients -> evaluations
 * ==================================================================== */

static inline void ecfft_fq_enter(fq_fe *data, size_t n, const ecfft_fq_ctx *ctx)
{
    if (n <= 1)
        return;

    size_t level = 0;
    for (size_t lv = 0; lv < ctx->log_n; lv++)
        if (ctx->levels[lv].n == n)
        {
            level = lv;
            break;
        }

    std::vector<ecfft_fq_fe_s> coeffs(n);
    for (size_t i = 0; i < n; i++)
        fq_copy(coeffs[i].v, data[i]);

    const ecfft_fq_fe_s *s = ctx->levels[level].s.data();

    for (size_t i = 0; i < n; i++)
    {
        fq_fe result;
        fq_copy(result, coeffs[n - 1].v);

        for (size_t k = n - 1; k > 0; k--)
        {
            fq_fe tmp;
            fq_mul(tmp, result, s[i].v);
            fq_add(result, tmp, coeffs[k - 1].v);
        }

        fq_copy(data[i], result);
    }

    fq_fe zero;
    fq_0(zero);
    for (size_t k = 0; k < n; k++)
        fq_sub(data[k], data[k], zero);
}

/* ====================================================================
 * ECFFT EXIT: evaluations -> coefficients
 * ==================================================================== */

static inline void ecfft_fq_exit(fq_fe *data, size_t n, const ecfft_fq_ctx *ctx)
{
    if (n <= 1)
        return;

    size_t level = 0;
    for (size_t lv = 0; lv < ctx->log_n; lv++)
        if (ctx->levels[lv].n == n)
        {
            level = lv;
            break;
        }

    const ecfft_fq_fe_s *s = ctx->levels[level].s.data();

    /* Stage 1: Newton divided differences */
    std::vector<ecfft_fq_fe_s> d(n);
    for (size_t i = 0; i < n; i++)
        fq_copy(d[i].v, data[i]);

    for (size_t j = 1; j < n; j++)
    {
        size_t count = n - j;

        std::vector<ecfft_fq_fe_s> denoms(count);
        std::vector<ecfft_fq_fe_s> inv_denoms(count);
        for (size_t i = j; i < n; i++)
            fq_sub(denoms[i - j].v, s[i].v, s[i - j].v);

        fq_batch_invert(&inv_denoms[0].v, &denoms[0].v, count);

        for (size_t i = n; i-- > j;)
        {
            fq_fe diff;
            fq_sub(diff, d[i].v, d[i - 1].v);
            fq_mul(d[i].v, diff, inv_denoms[i - j].v);
        }
    }

    /* Stage 2: Newton form to monomial coefficients */
    std::vector<ecfft_fq_fe_s> p(n);
    for (size_t i = 0; i < n; i++)
        fq_0(p[i].v);
    fq_copy(p[0].v, d[n - 1].v);
    size_t deg = 0;

    for (size_t k = n - 1; k-- > 0;)
    {
        fq_copy(p[deg + 1].v, p[deg].v);
        for (size_t j = deg; j >= 1; j--)
        {
            fq_fe prod;
            fq_mul(prod, s[k].v, p[j].v);
            fq_sub(p[j].v, p[j - 1].v, prod);
        }
        {
            fq_fe prod;
            fq_mul(prod, s[k].v, p[0].v);
            fq_neg(p[0].v, prod);
        }
        deg++;

        fq_add(p[0].v, p[0].v, d[k].v);
    }

    fq_fe zero;
    fq_0(zero);
    for (size_t i = 0; i < n; i++)
        fq_sub(data[i], p[i].v, zero);
}

/* ====================================================================
 * ECFFT EXTEND / REDUCE
 * ==================================================================== */

static inline void ecfft_fq_extend(fq_fe *data, size_t n_from, size_t n_to, const ecfft_fq_ctx *ctx)
{
    if (n_from >= n_to || n_from <= 1)
        return;

    ecfft_fq_exit(data, n_from, ctx);

    for (size_t i = n_from; i < n_to; i++)
        fq_0(data[i]);

    ecfft_fq_enter(data, n_to, ctx);
}

static inline void ecfft_fq_reduce(fq_fe *data, size_t n_from, size_t n_to, const ecfft_fq_ctx *ctx)
{
    if (n_to >= n_from || n_to <= 1)
        return;

    ecfft_fq_exit(data, n_from, ctx);
    ecfft_fq_enter(data, n_to, ctx);
}

/* ====================================================================
 * ECFFT polynomial multiplication
 * ==================================================================== */

static inline void ecfft_fq_poly_mul(
    fq_fe *result,
    size_t *result_len,
    const fq_fe *a,
    size_t a_len,
    const fq_fe *b,
    size_t b_len,
    const ecfft_fq_ctx *ctx)
{
    if (a_len == 0 || b_len == 0)
    {
        *result_len = 1;
        fq_0(result[0]);
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

    std::vector<ecfft_fq_fe_s> fa(n);
    std::vector<ecfft_fq_fe_s> fb(n);

    for (size_t i = 0; i < a_len; i++)
        fq_copy(fa[i].v, a[i]);
    for (size_t i = a_len; i < n; i++)
        fq_0(fa[i].v);

    for (size_t i = 0; i < b_len; i++)
        fq_copy(fb[i].v, b[i]);
    for (size_t i = b_len; i < n; i++)
        fq_0(fb[i].v);

    ecfft_fq_enter(&fa[0].v, n, ctx);
    ecfft_fq_enter(&fb[0].v, n, ctx);

    for (size_t i = 0; i < n; i++)
        fq_mul(fa[i].v, fa[i].v, fb[i].v);

    ecfft_fq_exit(&fa[0].v, n, ctx);

    *result_len = out_len;
    for (size_t i = 0; i < out_len; i++)
        fq_copy(result[i], fa[i].v);
}

#endif /* RANSHAW_ECFFT */
#endif /* RANSHAW_ECFFT_FQ_H */
