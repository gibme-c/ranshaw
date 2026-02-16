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

// poly.cpp — Polynomial arithmetic over F_p and F_q.
// Schoolbook (deg < 32), Karatsuba (32 <= deg < 1024), ECFFT (deg >= 1024).
// Includes evaluation (Horner), from_roots, divmod (long division), and Lagrange interpolation.

#include "poly.h"

#ifdef RANSHAW_ECFFT
#include "ecfft_fp.h"
#include "ecfft_fq.h"

/* Threshold: use ECFFT above this many coefficients (both operands).
 * ECFFT ENTER/EXIT are O(n^2), so ECFFT poly_mul only wins over Karatsuba
 * at very large sizes where the evaluation-domain approach amortizes. */
static const size_t ECFFT_THRESHOLD = 1024;

/* Defined in ecfft.cpp */
const ecfft_fp_ctx *ecfft_fp_global_ctx();
const ecfft_fq_ctx *ecfft_fq_global_ctx();
#endif

#include "fp_frombytes.h"
#include "fp_invert.h"
#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "fp_tobytes.h"
#include "fp_utils.h"
#include "fq_frombytes.h"
#include "fq_invert.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "fq_tobytes.h"
#include "fq_utils.h"

/* Karatsuba threshold: use schoolbook below this many coefficients */
static const size_t KARATSUBA_THRESHOLD = 32;

/* ---- Helpers: copy fp_fe in/out of storage ---- */

static inline void fp_fe_store(fp_fe_storage *dst, const fp_fe src)
{
    std::memcpy(dst->v, src, sizeof(fp_fe));
}

static inline void fp_fe_load(fp_fe dst, const fp_fe_storage *src)
{
    std::memcpy(dst, src->v, sizeof(fp_fe));
}

static inline void fq_fe_store(fq_fe_storage *dst, const fq_fe src)
{
    std::memcpy(dst->v, src, sizeof(fq_fe));
}

static inline void fq_fe_load(fq_fe dst, const fq_fe_storage *src)
{
    std::memcpy(dst, src->v, sizeof(fq_fe));
}

/* ---- Helpers: check if an fp_fe is zero (via byte comparison) ---- */

static int fp_fe_is_zero(const fp_fe f)
{
    unsigned char s[32];
    fp_tobytes(s, f);
    unsigned char d = 0;
    for (int i = 0; i < 32; i++)
        d |= s[i];
    return d == 0;
}

static int fq_fe_is_zero(const fq_fe f)
{
    unsigned char s[32];
    fq_tobytes(s, f);
    unsigned char d = 0;
    for (int i = 0; i < 32; i++)
        d |= s[i];
    return d == 0;
}

/* ---- Strip trailing zero coefficients ---- */

static void fp_poly_strip(fp_poly *p)
{
    while (p->coeffs.size() > 1)
    {
        fp_fe tmp;
        fp_fe_load(tmp, &p->coeffs.back());
        if (!fp_fe_is_zero(tmp))
            break;
        p->coeffs.pop_back();
    }
}

static void fq_poly_strip(fq_poly *p)
{
    while (p->coeffs.size() > 1)
    {
        fq_fe tmp;
        fq_fe_load(tmp, &p->coeffs.back());
        if (!fq_fe_is_zero(tmp))
            break;
        p->coeffs.pop_back();
    }
}

/* ---- Normalize polynomial coefficients ---- */

/*
 * fp_add on 64-bit does not carry-propagate, so after schoolbook accumulation
 * limb values can grow large. fp_sub uses unsigned bias addition which can
 * overflow on non-canonical inputs. Force carry-propagation via fp_sub(x, x, 0)
 * which works on both 64-bit (carries through bias) and 32-bit (signed carries).
 */
static void fp_poly_normalize(fp_poly *p)
{
    fp_fe zero;
    fp_0(zero);
    for (size_t i = 0; i < p->coeffs.size(); i++)
    {
        fp_fe tmp;
        fp_fe_load(tmp, &p->coeffs[i]);
        fp_sub(tmp, tmp, zero);
        fp_fe_store(&p->coeffs[i], tmp);
    }
}

/*
 * Normalize F_q polynomial coefficients to canonical (centered-carry) limb form.
 *
 * fq_sub(x, x, 0) leaves non-canonical limbs on the portable backend because
 * the Crandall gamma fold only partially propagates carries.  Downstream
 * consumers (Karatsuba, ECFFT) are representation-sensitive, so we round-trip
 * through tobytes/frombytes which always produces the centered-carry form
 * that fq_frombytes guarantees.
 */
static void fq_poly_normalize(fq_poly *p)
{
    for (size_t i = 0; i < p->coeffs.size(); i++)
    {
        unsigned char buf[32];
        fq_fe tmp;
        fq_fe_load(tmp, &p->coeffs[i]);
        fq_tobytes(buf, tmp);
        fq_frombytes(p->coeffs[i].v, buf);
    }
}

/* ---- Helpers: polynomial add/sub (used by Karatsuba) ---- */

static void fp_poly_add(fp_poly *r, const fp_poly *a, const fp_poly *b)
{
    size_t na = a->coeffs.size();
    size_t nb = b->coeffs.size();
    size_t nr = (na > nb) ? na : nb;
    r->coeffs.resize(nr);
    for (size_t i = 0; i < nr; i++)
    {
        fp_fe ai_val, bi_val;
        if (i < na)
            fp_fe_load(ai_val, &a->coeffs[i]);
        else
            fp_0(ai_val);
        if (i < nb)
            fp_fe_load(bi_val, &b->coeffs[i]);
        else
            fp_0(bi_val);
        fp_add(r->coeffs[i].v, ai_val, bi_val);
    }
    fp_poly_normalize(r);
    fp_poly_strip(r);
}

static void fp_poly_sub(fp_poly *r, const fp_poly *a, const fp_poly *b)
{
    size_t na = a->coeffs.size();
    size_t nb = b->coeffs.size();
    size_t nr = (na > nb) ? na : nb;
    r->coeffs.resize(nr);
    for (size_t i = 0; i < nr; i++)
    {
        fp_fe ai_val, bi_val;
        if (i < na)
            fp_fe_load(ai_val, &a->coeffs[i]);
        else
            fp_0(ai_val);
        if (i < nb)
            fp_fe_load(bi_val, &b->coeffs[i]);
        else
            fp_0(bi_val);
        fp_sub(r->coeffs[i].v, ai_val, bi_val);
    }
    fp_poly_strip(r);
}

static void fq_poly_add(fq_poly *r, const fq_poly *a, const fq_poly *b)
{
    size_t na = a->coeffs.size();
    size_t nb = b->coeffs.size();
    size_t nr = (na > nb) ? na : nb;
    r->coeffs.resize(nr);
    for (size_t i = 0; i < nr; i++)
    {
        fq_fe ai_val, bi_val;
        if (i < na)
            fq_fe_load(ai_val, &a->coeffs[i]);
        else
            fq_0(ai_val);
        if (i < nb)
            fq_fe_load(bi_val, &b->coeffs[i]);
        else
            fq_0(bi_val);
        fq_add(r->coeffs[i].v, ai_val, bi_val);
    }
    fq_poly_strip(r);
}

static void fq_poly_sub(fq_poly *r, const fq_poly *a, const fq_poly *b)
{
    size_t na = a->coeffs.size();
    size_t nb = b->coeffs.size();
    size_t nr = (na > nb) ? na : nb;
    r->coeffs.resize(nr);
    for (size_t i = 0; i < nr; i++)
    {
        fq_fe ai_val, bi_val;
        if (i < na)
            fq_fe_load(ai_val, &a->coeffs[i]);
        else
            fq_0(ai_val);
        if (i < nb)
            fq_fe_load(bi_val, &b->coeffs[i]);
        else
            fq_0(bi_val);
        fq_sub(r->coeffs[i].v, ai_val, bi_val);
    }
    fq_poly_strip(r);
}

/* ---- Schoolbook multiplication (internal) ---- */

static void fp_poly_mul_schoolbook(fp_poly *r, const fp_poly *a, const fp_poly *b)
{
    size_t na = a->coeffs.size();
    size_t nb = b->coeffs.size();

    if (na == 0 || nb == 0)
    {
        r->coeffs.resize(1);
        fp_0(r->coeffs[0].v);
        return;
    }

    size_t nr = na + nb - 1;
    r->coeffs.resize(nr);

    for (size_t k = 0; k < nr; k++)
    {
        fp_0(r->coeffs[k].v);
    }

    for (size_t i = 0; i < na; i++)
    {
        fp_fe ai;
        fp_fe_load(ai, &a->coeffs[i]);
        for (size_t j = 0; j < nb; j++)
        {
            fp_fe bj, prod, sum;
            fp_fe_load(bj, &b->coeffs[j]);
            fp_mul(prod, ai, bj);
            fp_fe_load(sum, &r->coeffs[i + j]);
            fp_add(sum, sum, prod);
            fp_fe_store(&r->coeffs[i + j], sum);
        }
    }

    fp_poly_normalize(r);
    fp_poly_strip(r);
}

static void fq_poly_mul_schoolbook(fq_poly *r, const fq_poly *a, const fq_poly *b)
{
    size_t na = a->coeffs.size();
    size_t nb = b->coeffs.size();

    if (na == 0 || nb == 0)
    {
        r->coeffs.resize(1);
        fq_0(r->coeffs[0].v);
        return;
    }

    size_t nr = na + nb - 1;
    r->coeffs.resize(nr);

    for (size_t k = 0; k < nr; k++)
    {
        fq_0(r->coeffs[k].v);
    }

    for (size_t i = 0; i < na; i++)
    {
        fq_fe ai;
        fq_fe_load(ai, &a->coeffs[i]);
        for (size_t j = 0; j < nb; j++)
        {
            fq_fe bj, prod, sum;
            fq_fe_load(bj, &b->coeffs[j]);
            fq_mul(prod, ai, bj);
            fq_fe_load(sum, &r->coeffs[i + j]);
            fq_add(sum, sum, prod);
            fq_fe_store(&r->coeffs[i + j], sum);
        }
    }

    fq_poly_normalize(r);
    fq_poly_strip(r);
}

/* ---- Helpers: extract sub-polynomial (slice) ---- */

static void fp_poly_slice(fp_poly *r, const fp_poly *p, size_t start, size_t len)
{
    size_t n = p->coeffs.size();
    if (start >= n || len == 0)
    {
        r->coeffs.resize(1);
        fp_0(r->coeffs[0].v);
        return;
    }
    size_t actual = (start + len > n) ? (n - start) : len;
    r->coeffs.resize(actual);
    for (size_t i = 0; i < actual; i++)
    {
        std::memcpy(r->coeffs[i].v, p->coeffs[start + i].v, sizeof(fp_fe));
    }
    fp_poly_strip(r);
}

static void fq_poly_slice(fq_poly *r, const fq_poly *p, size_t start, size_t len)
{
    size_t n = p->coeffs.size();
    if (start >= n || len == 0)
    {
        r->coeffs.resize(1);
        fq_0(r->coeffs[0].v);
        return;
    }
    size_t actual = (start + len > n) ? (n - start) : len;
    r->coeffs.resize(actual);
    for (size_t i = 0; i < actual; i++)
    {
        std::memcpy(r->coeffs[i].v, p->coeffs[start + i].v, sizeof(fq_fe));
    }
    fq_poly_strip(r);
}

/* ---- Helpers: shift polynomial by m positions (multiply by x^m) ---- */

static void fp_poly_shift(fp_poly *r, const fp_poly *p, size_t m)
{
    if (m == 0)
    {
        *r = *p;
        return;
    }
    size_t n = p->coeffs.size();
    r->coeffs.resize(n + m);
    for (size_t i = 0; i < m; i++)
        fp_0(r->coeffs[i].v);
    for (size_t i = 0; i < n; i++)
        std::memcpy(r->coeffs[i + m].v, p->coeffs[i].v, sizeof(fp_fe));
}

static void fq_poly_shift(fq_poly *r, const fq_poly *p, size_t m)
{
    if (m == 0)
    {
        *r = *p;
        return;
    }
    size_t n = p->coeffs.size();
    r->coeffs.resize(n + m);
    for (size_t i = 0; i < m; i++)
        fq_0(r->coeffs[i].v);
    for (size_t i = 0; i < n; i++)
        std::memcpy(r->coeffs[i + m].v, p->coeffs[i].v, sizeof(fq_fe));
}

/* ================================================================
 * F_p polynomial operations
 * ================================================================ */

/*
 * Karatsuba polynomial multiplication (recursive).
 *
 * Given A, B, split at midpoint m:
 *   A = A_lo + x^m * A_hi
 *   B = B_lo + x^m * B_hi
 *   z0 = A_lo * B_lo
 *   z2 = A_hi * B_hi
 *   z1 = (A_lo + A_hi) * (B_lo + B_hi) - z0 - z2
 *   result = z0 + x^m * z1 + x^(2m) * z2
 */
static void fp_poly_mul_karatsuba(fp_poly *r, const fp_poly *a, const fp_poly *b)
{
    size_t na = a->coeffs.size();
    size_t nb = b->coeffs.size();

    /* Base case: fall through to schoolbook */
    if (na < KARATSUBA_THRESHOLD || nb < KARATSUBA_THRESHOLD)
    {
        fp_poly_mul_schoolbook(r, a, b);
        return;
    }

    size_t m = ((na > nb) ? na : nb) / 2;

    fp_poly a_lo, a_hi, b_lo, b_hi;
    fp_poly_slice(&a_lo, a, 0, m);
    fp_poly_slice(&a_hi, a, m, na - m);
    fp_poly_slice(&b_lo, b, 0, m);
    fp_poly_slice(&b_hi, b, m, nb - m);

    /* z0 = a_lo * b_lo */
    fp_poly z0;
    fp_poly_mul(&z0, &a_lo, &b_lo);

    /* z2 = a_hi * b_hi */
    fp_poly z2;
    fp_poly_mul(&z2, &a_hi, &b_hi);

    /* z1 = (a_lo + a_hi) * (b_lo + b_hi) - z0 - z2 */
    fp_poly a_sum, b_sum, z1_raw, z1_tmp, z1;
    fp_poly_add(&a_sum, &a_lo, &a_hi);
    fp_poly_add(&b_sum, &b_lo, &b_hi);
    fp_poly_mul(&z1_raw, &a_sum, &b_sum);
    fp_poly_sub(&z1_tmp, &z1_raw, &z0);
    fp_poly_sub(&z1, &z1_tmp, &z2);

    /* result = z0 + x^m * z1 + x^(2m) * z2 */
    fp_poly z1_shifted, z2_shifted, tmp;
    fp_poly_shift(&z1_shifted, &z1, m);
    fp_poly_shift(&z2_shifted, &z2, 2 * m);
    fp_poly_add(&tmp, &z0, &z1_shifted);
    fp_poly_add(r, &tmp, &z2_shifted);
}

void fp_poly_mul(fp_poly *r, const fp_poly *a, const fp_poly *b)
{
    size_t na = a->coeffs.size();
    size_t nb = b->coeffs.size();

    if (na == 0 || nb == 0)
    {
        r->coeffs.resize(1);
        fp_0(r->coeffs[0].v);
        return;
    }

#ifdef RANSHAW_ECFFT
    const ecfft_fp_ctx *ectx = ecfft_fp_global_ctx();
    if (ectx && na >= ECFFT_THRESHOLD && nb >= ECFFT_THRESHOLD)
    {
        size_t out_len = na + nb - 1;
        size_t n_padded = 1;
        while (n_padded < out_len)
            n_padded <<= 1;

        if (n_padded <= ectx->domain_size)
        {
            std::vector<fp_fe_storage> fa(na), fb(nb), fr(n_padded);

            /* Canonicalize coefficients via tobytes/frombytes to ensure the
             * centered-carry limb form that the ECFFT domain points use.
             * Without this, the O(n^2) Horner evaluation in ecfft_fp_enter
             * can accumulate representation-dependent rounding differences. */
            for (size_t i = 0; i < na; i++)
            {
                unsigned char buf[32];
                fp_fe tmp;
                fp_fe_load(tmp, &a->coeffs[i]);
                fp_tobytes(buf, tmp);
                fp_frombytes(fa[i].v, buf);
            }
            for (size_t i = 0; i < nb; i++)
            {
                unsigned char buf[32];
                fp_fe tmp;
                fp_fe_load(tmp, &b->coeffs[i]);
                fp_tobytes(buf, tmp);
                fp_frombytes(fb[i].v, buf);
            }

            size_t result_len = 0;
            ecfft_fp_poly_mul(&fr[0].v, &result_len, &fa[0].v, na, &fb[0].v, nb, ectx);

            if (result_len > 0)
            {
                r->coeffs.resize(result_len);
                for (size_t i = 0; i < result_len; i++)
                    fp_fe_store(&r->coeffs[i], fr[i].v);
                fp_poly_strip(r);
                return;
            }
        }
    }
#endif

    if (na >= KARATSUBA_THRESHOLD && nb >= KARATSUBA_THRESHOLD)
    {
        fp_poly_mul_karatsuba(r, a, b);
    }
    else
    {
        fp_poly_mul_schoolbook(r, a, b);
    }
}

void fp_poly_eval(fp_fe result, const fp_poly *p, const fp_fe x)
{
    size_t n = p->coeffs.size();
    if (n == 0)
    {
        fp_0(result);
        return;
    }

    /* Horner's method: start from highest coefficient */
    fp_fe_load(result, &p->coeffs[n - 1]);
    for (size_t i = n - 1; i > 0; i--)
    {
        fp_fe tmp, ci;
        fp_mul(tmp, result, x);
        fp_fe_load(ci, &p->coeffs[i - 1]);
        fp_add(result, tmp, ci);
    }
}

void fp_poly_from_roots(fp_poly *r, const fp_fe *roots, size_t n)
{
    if (n == 0)
    {
        /* Return the constant polynomial 1 */
        r->coeffs.resize(1);
        fp_1(r->coeffs[0].v);
        return;
    }

    /* Start with (x - roots[0]) = [-roots[0], 1] */
    r->coeffs.resize(2);
    fp_neg(r->coeffs[0].v, roots[0]);
    fp_1(r->coeffs[1].v);

    /* Multiply by (x - roots[i]) for i = 1..n-1 */
    for (size_t i = 1; i < n; i++)
    {
        fp_poly linear;
        linear.coeffs.resize(2);
        fp_neg(linear.coeffs[0].v, roots[i]);
        fp_1(linear.coeffs[1].v);

        fp_poly tmp;
        fp_poly_mul(&tmp, r, &linear);
        *r = tmp;
    }
}

void fp_poly_divmod(fp_poly *q, fp_poly *rem, const fp_poly *a, const fp_poly *b)
{
    size_t na = a->coeffs.size();
    size_t nb = b->coeffs.size();

    /* Copy a into remainder */
    *rem = *a;
    fp_poly_strip(rem);
    na = rem->coeffs.size();

    /* Strip b to get true degree */
    fp_poly bstrip = *b;
    fp_poly_strip(&bstrip);
    nb = bstrip.coeffs.size();

    /* Check for zero divisor (after strip: single zero coefficient) */
    {
        fp_fe lead_check;
        fp_fe_load(lead_check, &bstrip.coeffs[nb - 1]);
        if (!fp_isnonzero(lead_check))
        {
            q->coeffs.resize(1);
            fp_0(q->coeffs[0].v);
            return;
        }
    }

    /* If deg(a) < deg(b), quotient is 0 and remainder is a */
    if (na < nb)
    {
        q->coeffs.resize(1);
        fp_0(q->coeffs[0].v);
        return;
    }

    size_t nq = na - nb + 1;
    q->coeffs.resize(nq);
    for (size_t i = 0; i < nq; i++)
    {
        fp_0(q->coeffs[i].v);
    }

    /* Invert the leading coefficient of b */
    fp_fe b_lead, b_lead_inv;
    fp_fe_load(b_lead, &bstrip.coeffs[nb - 1]);
    fp_invert(b_lead_inv, b_lead);

    /* Long division: work from highest degree down */
    for (size_t i = na; i >= nb; i--)
    {
        fp_fe rem_lead, coeff;
        fp_fe_load(rem_lead, &rem->coeffs[i - 1]);
        fp_mul(coeff, rem_lead, b_lead_inv);
        fp_fe_store(&q->coeffs[i - nb], coeff);

        /* Subtract coeff * b * x^(i - nb) from remainder */
        for (size_t j = 0; j < nb; j++)
        {
            fp_fe bj, prod, rval, diff;
            fp_fe_load(bj, &bstrip.coeffs[j]);
            fp_mul(prod, coeff, bj);
            fp_fe_load(rval, &rem->coeffs[i - nb + j]);
            fp_sub(diff, rval, prod);
            fp_fe_store(&rem->coeffs[i - nb + j], diff);
        }
    }

    /* Trim remainder to degree < deg(b) and strip */
    rem->coeffs.resize(nb - 1 > 0 ? nb - 1 : 1);
    fp_poly_strip(rem);
    fp_poly_strip(q);
}

void fp_poly_interpolate(fp_poly *out, const fp_fe *xs, const fp_fe *ys, size_t n)
{
    if (n == 0)
    {
        out->coeffs.resize(1);
        fp_0(out->coeffs[0].v);
        return;
    }

    if (n == 1)
    {
        out->coeffs.resize(1);
        fp_copy(out->coeffs[0].v, ys[0]);
        return;
    }

    /* Build vanishing polynomial v(x) = prod(x - x_i) */
    fp_poly v;
    fp_poly_from_roots(&v, xs, n);

    /* Initialize output to zero polynomial of degree n-1 */
    out->coeffs.resize(n);
    for (size_t k = 0; k < n; k++)
    {
        fp_0(out->coeffs[k].v);
    }

    /* For each point, compute Lagrange basis L_i(x) and accumulate */
    for (size_t i = 0; i < n; i++)
    {
        /* L_num_i(x) = v(x) / (x - x_i) */
        fp_poly lin;
        lin.coeffs.resize(2);
        fp_neg(lin.coeffs[0].v, xs[i]);
        fp_1(lin.coeffs[1].v);

        fp_poly L_num, remainder;
        fp_poly_divmod(&L_num, &remainder, &v, &lin);

        /* w_i = prod_{j!=i}(x_i - x_j) */
        fp_fe wi;
        fp_1(wi);
        for (size_t j = 0; j < n; j++)
        {
            if (j == i)
                continue;
            fp_fe diff, tmp;
            fp_sub(diff, xs[i], xs[j]);
            fp_mul(tmp, wi, diff);
            fp_copy(wi, tmp);
        }

        fp_fe wi_inv;
        fp_invert(wi_inv, wi);

        /* scale = y_i / w_i */
        fp_fe scale;
        fp_mul(scale, ys[i], wi_inv);

        /* Accumulate: out += scale * L_num(x) */
        for (size_t k = 0; k < L_num.coeffs.size() && k < n; k++)
        {
            fp_fe lk, prod, cur;
            fp_fe_load(lk, &L_num.coeffs[k]);
            fp_mul(prod, scale, lk);
            fp_fe_load(cur, &out->coeffs[k]);
            fp_add(cur, cur, prod);
            fp_fe_store(&out->coeffs[k], cur);
        }
    }

    fp_poly_normalize(out);
    fp_poly_strip(out);
}

/* ================================================================
 * F_q polynomial operations
 * ================================================================ */

static void fq_poly_mul_karatsuba(fq_poly *r, const fq_poly *a, const fq_poly *b)
{
    size_t na = a->coeffs.size();
    size_t nb = b->coeffs.size();

    if (na < KARATSUBA_THRESHOLD || nb < KARATSUBA_THRESHOLD)
    {
        fq_poly_mul_schoolbook(r, a, b);
        return;
    }

    size_t m = ((na > nb) ? na : nb) / 2;

    fq_poly a_lo, a_hi, b_lo, b_hi;
    fq_poly_slice(&a_lo, a, 0, m);
    fq_poly_slice(&a_hi, a, m, na - m);
    fq_poly_slice(&b_lo, b, 0, m);
    fq_poly_slice(&b_hi, b, m, nb - m);

    fq_poly z0;
    fq_poly_mul(&z0, &a_lo, &b_lo);

    fq_poly z2;
    fq_poly_mul(&z2, &a_hi, &b_hi);

    fq_poly a_sum, b_sum, z1_raw, z1_tmp, z1;
    fq_poly_add(&a_sum, &a_lo, &a_hi);
    fq_poly_add(&b_sum, &b_lo, &b_hi);
    fq_poly_mul(&z1_raw, &a_sum, &b_sum);
    fq_poly_sub(&z1_tmp, &z1_raw, &z0);
    fq_poly_sub(&z1, &z1_tmp, &z2);

    fq_poly z1_shifted, z2_shifted, tmp;
    fq_poly_shift(&z1_shifted, &z1, m);
    fq_poly_shift(&z2_shifted, &z2, 2 * m);
    fq_poly_add(&tmp, &z0, &z1_shifted);
    fq_poly_add(r, &tmp, &z2_shifted);
}

void fq_poly_mul(fq_poly *r, const fq_poly *a, const fq_poly *b)
{
    size_t na = a->coeffs.size();
    size_t nb = b->coeffs.size();

    if (na == 0 || nb == 0)
    {
        r->coeffs.resize(1);
        fq_0(r->coeffs[0].v);
        return;
    }

#ifdef RANSHAW_ECFFT
    const ecfft_fq_ctx *ectx = ecfft_fq_global_ctx();
    if (ectx && na >= ECFFT_THRESHOLD && nb >= ECFFT_THRESHOLD)
    {
        size_t out_len = na + nb - 1;
        size_t n_padded = 1;
        while (n_padded < out_len)
            n_padded <<= 1;

        if (n_padded <= ectx->domain_size)
        {
            std::vector<fq_fe_storage> fa(na), fb(nb), fr(n_padded);

            /* Canonicalize coefficients via tobytes/frombytes to ensure the
             * centered-carry limb form that the ECFFT domain points use.
             * Without this, the O(n^2) Horner evaluation in ecfft_fq_enter
             * can accumulate representation-dependent rounding differences. */
            for (size_t i = 0; i < na; i++)
            {
                unsigned char buf[32];
                fq_fe tmp;
                fq_fe_load(tmp, &a->coeffs[i]);
                fq_tobytes(buf, tmp);
                fq_frombytes(fa[i].v, buf);
            }
            for (size_t i = 0; i < nb; i++)
            {
                unsigned char buf[32];
                fq_fe tmp;
                fq_fe_load(tmp, &b->coeffs[i]);
                fq_tobytes(buf, tmp);
                fq_frombytes(fb[i].v, buf);
            }

            size_t result_len = 0;
            ecfft_fq_poly_mul(&fr[0].v, &result_len, &fa[0].v, na, &fb[0].v, nb, ectx);

            if (result_len > 0)
            {
                r->coeffs.resize(result_len);
                for (size_t i = 0; i < result_len; i++)
                    fq_fe_store(&r->coeffs[i], fr[i].v);
                fq_poly_strip(r);
                return;
            }
        }
    }
#endif

    if (na >= KARATSUBA_THRESHOLD && nb >= KARATSUBA_THRESHOLD)
    {
        fq_poly_mul_karatsuba(r, a, b);
    }
    else
    {
        fq_poly_mul_schoolbook(r, a, b);
    }
}

void fq_poly_eval(fq_fe result, const fq_poly *p, const fq_fe x)
{
    size_t n = p->coeffs.size();
    if (n == 0)
    {
        fq_0(result);
        return;
    }

    fq_fe_load(result, &p->coeffs[n - 1]);
    for (size_t i = n - 1; i > 0; i--)
    {
        fq_fe tmp, ci;
        fq_mul(tmp, result, x);
        fq_fe_load(ci, &p->coeffs[i - 1]);
        fq_add(result, tmp, ci);
    }
}

void fq_poly_from_roots(fq_poly *r, const fq_fe *roots, size_t n)
{
    if (n == 0)
    {
        r->coeffs.resize(1);
        fq_1(r->coeffs[0].v);
        return;
    }

    r->coeffs.resize(2);
    fq_neg(r->coeffs[0].v, roots[0]);
    fq_1(r->coeffs[1].v);

    for (size_t i = 1; i < n; i++)
    {
        fq_poly linear;
        linear.coeffs.resize(2);
        fq_neg(linear.coeffs[0].v, roots[i]);
        fq_1(linear.coeffs[1].v);

        fq_poly tmp;
        fq_poly_mul(&tmp, r, &linear);
        *r = tmp;
    }
}

void fq_poly_divmod(fq_poly *q, fq_poly *rem, const fq_poly *a, const fq_poly *b)
{
    size_t na = a->coeffs.size();
    size_t nb = b->coeffs.size();

    *rem = *a;
    fq_poly_strip(rem);
    na = rem->coeffs.size();

    fq_poly bstrip = *b;
    fq_poly_strip(&bstrip);
    nb = bstrip.coeffs.size();

    /* Check for zero divisor (after strip: single zero coefficient) */
    {
        fq_fe lead_check;
        fq_fe_load(lead_check, &bstrip.coeffs[nb - 1]);
        if (!fq_isnonzero(lead_check))
        {
            q->coeffs.resize(1);
            fq_0(q->coeffs[0].v);
            return;
        }
    }

    if (na < nb)
    {
        q->coeffs.resize(1);
        fq_0(q->coeffs[0].v);
        return;
    }

    size_t nq = na - nb + 1;
    q->coeffs.resize(nq);
    for (size_t i = 0; i < nq; i++)
    {
        fq_0(q->coeffs[i].v);
    }

    fq_fe b_lead, b_lead_inv;
    fq_fe_load(b_lead, &bstrip.coeffs[nb - 1]);
    fq_invert(b_lead_inv, b_lead);

    for (size_t i = na; i >= nb; i--)
    {
        fq_fe rem_lead, coeff;
        fq_fe_load(rem_lead, &rem->coeffs[i - 1]);
        fq_mul(coeff, rem_lead, b_lead_inv);
        fq_fe_store(&q->coeffs[i - nb], coeff);

        for (size_t j = 0; j < nb; j++)
        {
            fq_fe bj_val, prod, rval, diff;
            fq_fe_load(bj_val, &bstrip.coeffs[j]);
            fq_mul(prod, coeff, bj_val);
            fq_fe_load(rval, &rem->coeffs[i - nb + j]);
            fq_sub(diff, rval, prod);
            fq_fe_store(&rem->coeffs[i - nb + j], diff);
        }
    }

    rem->coeffs.resize(nb - 1 > 0 ? nb - 1 : 1);
    fq_poly_strip(rem);
    fq_poly_strip(q);
}

void fq_poly_interpolate(fq_poly *out, const fq_fe *xs, const fq_fe *ys, size_t n)
{
    if (n == 0)
    {
        out->coeffs.resize(1);
        fq_0(out->coeffs[0].v);
        return;
    }

    if (n == 1)
    {
        out->coeffs.resize(1);
        fq_copy(out->coeffs[0].v, ys[0]);
        return;
    }

    /* Build vanishing polynomial v(x) = prod(x - x_i) */
    fq_poly v;
    fq_poly_from_roots(&v, xs, n);

    /* Initialize output to zero polynomial of degree n-1 */
    out->coeffs.resize(n);
    for (size_t k = 0; k < n; k++)
    {
        fq_0(out->coeffs[k].v);
    }

    for (size_t i = 0; i < n; i++)
    {
        fq_poly lin;
        lin.coeffs.resize(2);
        fq_neg(lin.coeffs[0].v, xs[i]);
        fq_1(lin.coeffs[1].v);

        fq_poly L_num, remainder;
        fq_poly_divmod(&L_num, &remainder, &v, &lin);

        fq_fe wi;
        fq_1(wi);
        for (size_t j = 0; j < n; j++)
        {
            if (j == i)
                continue;
            fq_fe diff, tmp;
            fq_sub(diff, xs[i], xs[j]);
            fq_mul(tmp, wi, diff);
            fq_copy(wi, tmp);
        }

        fq_fe wi_inv;
        fq_invert(wi_inv, wi);

        fq_fe scale;
        fq_mul(scale, ys[i], wi_inv);

        for (size_t k = 0; k < L_num.coeffs.size() && k < n; k++)
        {
            fq_fe lk, prod, cur;
            fq_fe_load(lk, &L_num.coeffs[k]);
            fq_mul(prod, scale, lk);
            fq_fe_load(cur, &out->coeffs[k]);
            fq_add(cur, cur, prod);
            fq_fe_store(&out->coeffs[k], cur);
        }
    }

    fq_poly_normalize(out);
    fq_poly_strip(out);
}
