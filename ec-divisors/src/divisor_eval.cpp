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

// divisor_eval.cpp — Evaluation-domain divisor operations using SoA layout.
// Precomputes curve evaluations and barycentric weights over an integer domain {0..N-1}.
// Supports SIMD-dispatched element-wise operations (AVX2/IFMA) for divisor multiplication.
// Tree-reduce merges per-point divisors via the curve-equation multiplication formula:
//   result.a = a1*a2 + curve(x)*b1*b2,  result.b = a1*b2 + a2*b1

#include "divisor_eval.h"

#include "divisor_eval_internal.h"

#if RANSHAW_SIMD
#include "ranshaw_cpuid.h"
#endif

#include "fp_batch_invert.h"
#include "fp_frombytes.h"
#include "fp_invert.h"
#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "fp_tobytes.h"
#include "fp_utils.h"
#include "fq_batch_invert.h"
#include "fq_frombytes.h"
#include "fq_invert.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "fq_tobytes.h"
#include "fq_utils.h"
#include "ran_add.h"
#include "ran_batch_affine.h"
#include "ran_constants.h"
#include "ran_dbl.h"
#include "ran_ops.h"
#include "ranshaw_secure_erase.h"
#include "shaw_add.h"
#include "shaw_batch_affine.h"
#include "shaw_constants.h"
#include "shaw_dbl.h"
#include "shaw_ops.h"

#include <cstring>
#include <mutex>

/* Safe addition: handles identity, P==P (doubles), P==-P (returns identity).
 * The raw ran_add/shaw_add formulas produce garbage for these cases. */
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
    fp_fe z1z1, z2z2, u1, u2, diff;
    fp_sq(z1z1, p->Z);
    fp_sq(z2z2, q->Z);
    fp_mul(u1, p->X, z2z2);
    fp_mul(u2, q->X, z1z1);
    fp_sub(diff, u1, u2);
    if (!fp_isnonzero(diff))
    {
        fp_fe s1, s2, t;
        fp_mul(t, q->Z, z2z2);
        fp_mul(s1, p->Y, t);
        fp_mul(t, p->Z, z1z1);
        fp_mul(s2, q->Y, t);
        fp_sub(diff, s1, s2);
        if (!fp_isnonzero(diff))
            ran_dbl(r, p);
        else
            ran_identity(r);
        return;
    }
    ran_add(r, p, q);
}

static void shaw_add_safe(shaw_jacobian *r, const shaw_jacobian *p, const shaw_jacobian *q)
{
    if (shaw_is_identity(p))
    {
        shaw_copy(r, q);
        return;
    }
    if (shaw_is_identity(q))
    {
        shaw_copy(r, p);
        return;
    }
    fq_fe z1z1, z2z2, u1, u2, diff;
    fq_sq(z1z1, p->Z);
    fq_sq(z2z2, q->Z);
    fq_mul(u1, p->X, z2z2);
    fq_mul(u2, q->X, z1z1);
    fq_sub(diff, u1, u2);
    if (!fq_isnonzero(diff))
    {
        fq_fe s1, s2, t;
        fq_mul(t, q->Z, z2z2);
        fq_mul(s1, p->Y, t);
        fq_mul(t, p->Z, z1z1);
        fq_mul(s2, q->Y, t);
        fq_sub(diff, s1, s2);
        if (!fq_isnonzero(diff))
            shaw_dbl(r, p);
        else
            shaw_identity(r);
        return;
    }
    shaw_add(r, p, q);
}
#include <vector>

static const size_t N = EVAL_DOMAIN_SIZE;

/* ================================================================
 * Precomputed tables
 * ================================================================ */

static fp_evals FP_CURVE_EVALS; /* curve(i) = i^3 - 3i + b */
static fp_evals FP_BARY_WEIGHTS; /* barycentric weights */
static fq_evals FQ_CURVE_EVALS;
static fq_evals FQ_BARY_WEIGHTS;

static std::once_flag fp_init_flag;
static std::once_flag fq_init_flag;

/* ================================================================
 * Scalar fallback implementations (match SIMD function signatures)
 * ================================================================ */

static void fp_evals_mul_scalar(fp_evals *r, const fp_evals *a, const fp_evals *b)
{
    for (size_t i = 0; i < N; i++)
    {
        fp_fe fa, fb, fr;
        fp_evals_get(fa, a, i);
        fp_evals_get(fb, b, i);
        fp_mul(fr, fa, fb);
        fp_evals_set(r, i, fr);
    }
    r->degree = a->degree + b->degree;
}

static void fq_evals_mul_scalar(fq_evals *r, const fq_evals *a, const fq_evals *b)
{
    for (size_t i = 0; i < N; i++)
    {
        fq_fe fa, fb, fr;
        fq_evals_get(fa, a, i);
        fq_evals_get(fb, b, i);
        fq_mul(fr, fa, fb);
        fq_evals_set(r, i, fr);
    }
    r->degree = a->degree + b->degree;
}

static void fp_evals_add_scalar(fp_evals *r, const fp_evals *a, const fp_evals *b)
{
    for (size_t i = 0; i < N; i++)
    {
        fp_fe fa, fb, fr;
        fp_evals_get(fa, a, i);
        fp_evals_get(fb, b, i);
        fp_add(fr, fa, fb);
        fp_evals_set(r, i, fr);
    }
    r->degree = (a->degree > b->degree) ? a->degree : b->degree;
}

static void fp_evals_sub_scalar(fp_evals *r, const fp_evals *a, const fp_evals *b)
{
    for (size_t i = 0; i < N; i++)
    {
        fp_fe fa, fb, fr;
        fp_evals_get(fa, a, i);
        fp_evals_get(fb, b, i);
        fp_sub(fr, fa, fb);
        fp_evals_set(r, i, fr);
    }
    r->degree = (a->degree > b->degree) ? a->degree : b->degree;
}

static void fq_evals_add_scalar(fq_evals *r, const fq_evals *a, const fq_evals *b)
{
    for (size_t i = 0; i < N; i++)
    {
        fq_fe fa, fb, fr;
        fq_evals_get(fa, a, i);
        fq_evals_get(fb, b, i);
        fq_add(fr, fa, fb);
        fq_evals_set(r, i, fr);
    }
    r->degree = (a->degree > b->degree) ? a->degree : b->degree;
}

static void fq_evals_sub_scalar(fq_evals *r, const fq_evals *a, const fq_evals *b)
{
    for (size_t i = 0; i < N; i++)
    {
        fq_fe fa, fb, fr;
        fq_evals_get(fa, a, i);
        fq_evals_get(fb, b, i);
        fq_sub(fr, fa, fb);
        fq_evals_set(r, i, fr);
    }
    r->degree = (a->degree > b->degree) ? a->degree : b->degree;
}

static void ran_eval_divisor_mul_scalar(
    ran_eval_divisor *r,
    const ran_eval_divisor *d1,
    const ran_eval_divisor *d2,
    const fp_evals *curve_evals)
{
    for (size_t i = 0; i < N; i++)
    {
        fp_fe va1, va2, vb1, vb2, vc;
        fp_evals_get(va1, &d1->a, i);
        fp_evals_get(va2, &d2->a, i);
        fp_evals_get(vb1, &d1->b, i);
        fp_evals_get(vb2, &d2->b, i);
        fp_evals_get(vc, curve_evals, i);

        fp_fe a1a2, b1b2, cb1b2, ra;
        fp_mul(a1a2, va1, va2);
        fp_mul(b1b2, vb1, vb2);
        fp_mul(cb1b2, vc, b1b2);
        fp_add(ra, a1a2, cb1b2);
        fp_evals_set(&r->a, i, ra);

        fp_fe t1, t2, t3, rb;
        fp_add(t1, va1, vb1);
        fp_add(t2, va2, vb2);
        fp_mul(t3, t1, t2);
        fp_sub(t3, t3, a1a2);
        fp_sub(rb, t3, b1b2);
        fp_evals_set(&r->b, i, rb);
    }
}

static void shaw_eval_divisor_mul_scalar(
    shaw_eval_divisor *r,
    const shaw_eval_divisor *d1,
    const shaw_eval_divisor *d2,
    const fq_evals *curve_evals)
{
    for (size_t i = 0; i < N; i++)
    {
        fq_fe va1, va2, vb1, vb2, vc;
        fq_evals_get(va1, &d1->a, i);
        fq_evals_get(va2, &d2->a, i);
        fq_evals_get(vb1, &d1->b, i);
        fq_evals_get(vb2, &d2->b, i);
        fq_evals_get(vc, curve_evals, i);

        fq_fe a1a2, b1b2, cb1b2, ra;
        fq_mul(a1a2, va1, va2);
        fq_mul(b1b2, vb1, vb2);
        fq_mul(cb1b2, vc, b1b2);
        fq_add(ra, a1a2, cb1b2);
        fq_evals_set(&r->a, i, ra);

        fq_fe t1, t2, t3, rb;
        fq_add(t1, va1, vb1);
        fq_add(t2, va2, vb2);
        fq_mul(t3, t1, t2);
        fq_sub(t3, t3, a1a2);
        fq_sub(rb, t3, b1b2);
        fq_evals_set(&r->b, i, rb);
    }
}

/* ================================================================
 * Dispatch table (initialized to scalar defaults)
 * ================================================================ */

static fp_evals_mul_fn_t fp_evals_mul_impl = fp_evals_mul_scalar;
static fq_evals_mul_fn_t fq_evals_mul_impl = fq_evals_mul_scalar;
static fp_evals_add_fn_t fp_evals_add_impl = fp_evals_add_scalar;
static fq_evals_add_fn_t fq_evals_add_impl = fq_evals_add_scalar;
static fp_evals_sub_fn_t fp_evals_sub_impl = fp_evals_sub_scalar;
static fq_evals_sub_fn_t fq_evals_sub_impl = fq_evals_sub_scalar;
static ran_eval_divisor_mul_fn_t ran_eval_divisor_mul_impl = ran_eval_divisor_mul_scalar;
static shaw_eval_divisor_mul_fn_t shaw_eval_divisor_mul_impl = shaw_eval_divisor_mul_scalar;

/* Helper: create field element from small integer */
static void fp_from_small(fp_fe h, int64_t val)
{
    unsigned char buf[32] = {};
    if (val >= 0)
    {
        for (int i = 0; i < 8 && val > 0; i++)
        {
            buf[i] = (unsigned char)(val & 0xff);
            val >>= 8;
        }
    }
    else
    {
        /* Negative: compute p + val (val is negative, |val| < p) */
        /* p = 2^255 - 19, load -val then negate in field */
        int64_t pos = -val;
        for (int i = 0; i < 8 && pos > 0; i++)
        {
            buf[i] = (unsigned char)(pos & 0xff);
            pos >>= 8;
        }
    }
    fp_frombytes(h, buf);
    if (val < 0)
    {
        fp_neg(h, h);
    }
}

static void fq_from_small(fq_fe h, int64_t val)
{
    unsigned char buf[32] = {};
    if (val >= 0)
    {
        for (int i = 0; i < 8 && val > 0; i++)
        {
            buf[i] = (unsigned char)(val & 0xff);
            val >>= 8;
        }
    }
    else
    {
        int64_t pos = -val;
        for (int i = 0; i < 8 && pos > 0; i++)
        {
            buf[i] = (unsigned char)(pos & 0xff);
            pos >>= 8;
        }
    }
    fq_frombytes(h, buf);
    if (val < 0)
    {
        fq_neg(h, h);
    }
}

/*
 * Compute barycentric weights for integer domain {0, 1, ..., N-1}.
 * w_j = (-1)^(N-1-j) / (j! * (N-1-j)!)
 *
 * We compute factorials iteratively and batch-invert.
 */
static void compute_fp_bary_weights()
{
    /* fact[i] = i! mod p */
    std::vector<fp_fe_storage> fact(N);
    fp_1(fact[0].v);
    for (size_t i = 1; i < N; i++)
    {
        fp_fe small_i;
        fp_from_small(small_i, (int64_t)i);
        fp_mul(fact[i].v, fact[i - 1].v, small_i);
    }

    /* denom[j] = j! * (N-1-j)! */
    std::vector<fp_fe_storage> denom(N);
    for (size_t j = 0; j < N; j++)
    {
        fp_mul(denom[j].v, fact[j].v, fact[N - 1 - j].v);
    }

    /* Batch invert */
    std::vector<fp_fe_storage> inv_denom(N);
    fp_batch_invert(&inv_denom[0].v, &denom[0].v, N);

    /* Apply sign: (-1)^(N-1-j) */
    for (size_t j = 0; j < N; j++)
    {
        fp_fe w;
        if ((N - 1 - j) & 1)
            fp_neg(w, inv_denom[j].v);
        else
            fp_copy(w, inv_denom[j].v);
        fp_evals_set(&FP_BARY_WEIGHTS, j, w);
    }
}

static void compute_fq_bary_weights()
{
    std::vector<fq_fe_storage> fact(N);
    fq_1(fact[0].v);
    for (size_t i = 1; i < N; i++)
    {
        fq_fe small_i;
        fq_from_small(small_i, (int64_t)i);
        fq_mul(fact[i].v, fact[i - 1].v, small_i);
    }

    std::vector<fq_fe_storage> denom(N);
    for (size_t j = 0; j < N; j++)
    {
        fq_mul(denom[j].v, fact[j].v, fact[N - 1 - j].v);
    }

    std::vector<fq_fe_storage> inv_denom(N);
    fq_batch_invert(&inv_denom[0].v, &denom[0].v, N);

    for (size_t j = 0; j < N; j++)
    {
        fq_fe w;
        if ((N - 1 - j) & 1)
            fq_neg(w, inv_denom[j].v);
        else
            fq_copy(w, inv_denom[j].v);
        fq_evals_set(&FQ_BARY_WEIGHTS, j, w);
    }
}

static void fp_init_impl()
{
    /* Compute curve evals: curve(i) = i^3 - 3*i + RAN_B */
    fp_fe three;
    fp_from_small(three, 3);
    for (size_t i = 0; i < N; i++)
    {
        fp_fe xi, xi2, xi3, t1, t2, cv;
        fp_from_small(xi, (int64_t)i);
        fp_sq(xi2, xi);
        fp_mul(xi3, xi2, xi);
        fp_mul(t1, three, xi); /* 3*i */
        fp_sub(t2, xi3, t1); /* i^3 - 3*i */
        fp_add(cv, t2, RAN_B); /* + b */
        fp_evals_set(&FP_CURVE_EVALS, i, cv);
    }
    compute_fp_bary_weights();

    /* Select SIMD backend for Fp eval-domain ops */
#if RANSHAW_SIMD
    uint32_t fp_features = ranshaw_cpu_features();
#ifndef RANSHAW_NO_AVX512
    if (fp_features & RANSHAW_CPU_AVX512IFMA)
    {
        fp_evals_mul_impl = fp_evals_mul_ifma;
        fp_evals_add_impl = fp_evals_add_ifma;
        fp_evals_sub_impl = fp_evals_sub_ifma;
        ran_eval_divisor_mul_impl = ran_eval_divisor_mul_ifma;
    }
    else
#endif
#ifndef RANSHAW_NO_AVX2
        if (fp_features & RANSHAW_CPU_AVX2)
    {
        fp_evals_mul_impl = fp_evals_mul_avx2;
        fp_evals_add_impl = fp_evals_add_avx2;
        fp_evals_sub_impl = fp_evals_sub_avx2;
        ran_eval_divisor_mul_impl = ran_eval_divisor_mul_avx2;
    }
#endif
    (void)fp_features;
#endif /* RANSHAW_SIMD */
}

static void fq_init_impl()
{
    fq_fe three;
    fq_from_small(three, 3);
    for (size_t i = 0; i < N; i++)
    {
        fq_fe xi, xi2, xi3, t1, t2, cv;
        fq_from_small(xi, (int64_t)i);
        fq_sq(xi2, xi);
        fq_mul(xi3, xi2, xi);
        fq_mul(t1, three, xi);
        fq_sub(t2, xi3, t1);
        fq_add(cv, t2, SHAW_B);
        fq_evals_set(&FQ_CURVE_EVALS, i, cv);
    }
    compute_fq_bary_weights();

    /* Select SIMD backend for Fq eval-domain ops */
#if RANSHAW_SIMD
    uint32_t fq_features = ranshaw_cpu_features();
#ifndef RANSHAW_NO_AVX512
    if (fq_features & RANSHAW_CPU_AVX512IFMA)
    {
        fq_evals_mul_impl = fq_evals_mul_ifma;
        fq_evals_add_impl = fq_evals_add_ifma;
        fq_evals_sub_impl = fq_evals_sub_ifma;
        shaw_eval_divisor_mul_impl = shaw_eval_divisor_mul_ifma;
    }
    else
#endif
#ifndef RANSHAW_NO_AVX2
        if (fq_features & RANSHAW_CPU_AVX2)
    {
        fq_evals_mul_impl = fq_evals_mul_avx2;
        fq_evals_add_impl = fq_evals_add_avx2;
        fq_evals_sub_impl = fq_evals_sub_avx2;
        shaw_eval_divisor_mul_impl = shaw_eval_divisor_mul_avx2;
    }
#endif
    (void)fq_features;
#endif /* RANSHAW_SIMD */
}

void ran_eval_divisor_init()
{
    std::call_once(fp_init_flag, fp_init_impl);
}

void shaw_eval_divisor_init()
{
    std::call_once(fq_init_flag, fq_init_impl);
}

/* ================================================================
 * F_p eval-domain polynomial operations
 * ================================================================ */

void fp_evals_mul(fp_evals *r, const fp_evals *a, const fp_evals *b)
{
    fp_evals_mul_impl(r, a, b);
}

void fp_evals_add(fp_evals *r, const fp_evals *a, const fp_evals *b)
{
    fp_evals_add_impl(r, a, b);
}

void fp_evals_sub(fp_evals *r, const fp_evals *a, const fp_evals *b)
{
    fp_evals_sub_impl(r, a, b);
}

void fp_evals_from_constant(fp_evals *r, const fp_fe c)
{
    for (size_t i = 0; i < N; i++)
        fp_evals_set(r, i, c);
    r->degree = 0;
}

void fp_evals_from_linear(fp_evals *r, const fp_fe c)
{
    /* f(x) = x - c, so f(i) = i - c */
    for (size_t i = 0; i < N; i++)
    {
        fp_fe xi, val;
        fp_from_small(xi, (int64_t)i);
        fp_sub(val, xi, c);
        fp_evals_set(r, i, val);
    }
    r->degree = 1;
}

/*
 * Barycentric evaluation: given evals f(0)..f(N-1), compute f(c)
 * where c is NOT a domain point.
 *
 * f(c) = L(c) * sum_{j=0}^{N-1} w_j * f(j) / (c - j)
 * where L(c) = prod_{j=0}^{N-1} (c - j)
 */
static void fp_bary_eval(fp_fe result, const fp_evals *ev, const fp_fe c)
{
    ran_eval_divisor_init();

    /* Compute (c - j) for all j */
    fp_fe_storage diffs[EVAL_DOMAIN_SIZE];
    for (size_t j = 0; j < N; j++)
    {
        fp_fe xj;
        fp_from_small(xj, (int64_t)j);
        fp_sub(diffs[j].v, c, xj);
    }

    /* Batch invert the differences */
    fp_fe_storage inv_diffs[EVAL_DOMAIN_SIZE];
    fp_batch_invert(&inv_diffs[0].v, &diffs[0].v, N);

    /* L(c) = product of all (c - j) */
    fp_fe L;
    fp_copy(L, diffs[0].v);
    for (size_t j = 1; j < N; j++)
        fp_mul(L, L, diffs[j].v);

    /* sum = sum of w_j * f(j) / (c - j) */
    fp_fe sum;
    fp_0(sum);
    for (size_t j = 0; j < N; j++)
    {
        fp_fe wj, fj, term;
        fp_evals_get(wj, &FP_BARY_WEIGHTS, j);
        fp_evals_get(fj, ev, j);
        fp_mul(term, wj, fj);
        fp_mul(term, term, inv_diffs[j].v);
        fp_add(sum, sum, term);
    }

    fp_mul(result, L, sum);
}

void fp_evals_to_poly(fp_poly *out, const fp_evals *ev)
{
    ran_eval_divisor_init();

    size_t deg = ev->degree;
    size_t n = deg + 1; /* number of coefficients */

    /* Use the first n domain points for interpolation */
    std::vector<fp_fe_storage> xs(n), ys(n);
    for (size_t i = 0; i < n; i++)
    {
        fp_from_small(xs[i].v, (int64_t)i);
        fp_evals_get(ys[i].v, ev, i);
    }

    fp_poly_interpolate(out, reinterpret_cast<const fp_fe *>(xs.data()), reinterpret_cast<const fp_fe *>(ys.data()), n);
}

void fp_evals_div_linear(fp_evals *q, const fp_evals *f, const fp_fe c)
{
    ran_eval_divisor_init();

    /* f(c) via barycentric */
    fp_fe fc;
    fp_bary_eval(fc, f, c);

    /* q(j) = (f(j) - f(c)) / (j - c) with batch inversion */
    fp_fe_storage diffs[EVAL_DOMAIN_SIZE];
    fp_fe_storage nums[EVAL_DOMAIN_SIZE];
    for (size_t j = 0; j < N; j++)
    {
        fp_fe xj, fj;
        fp_from_small(xj, (int64_t)j);
        fp_sub(diffs[j].v, xj, c);
        fp_evals_get(fj, f, j);
        fp_sub(nums[j].v, fj, fc);
    }

    fp_fe_storage inv_diffs[EVAL_DOMAIN_SIZE];
    fp_batch_invert(&inv_diffs[0].v, &diffs[0].v, N);

    for (size_t j = 0; j < N; j++)
    {
        fp_fe qj;
        fp_mul(qj, nums[j].v, inv_diffs[j].v);
        fp_evals_set(q, j, qj);
    }

    q->degree = (f->degree > 0) ? f->degree - 1 : 0;
}

/* ================================================================
 * F_q eval-domain polynomial operations
 * ================================================================ */

void fq_evals_mul(fq_evals *r, const fq_evals *a, const fq_evals *b)
{
    fq_evals_mul_impl(r, a, b);
}

void fq_evals_add(fq_evals *r, const fq_evals *a, const fq_evals *b)
{
    fq_evals_add_impl(r, a, b);
}

void fq_evals_sub(fq_evals *r, const fq_evals *a, const fq_evals *b)
{
    fq_evals_sub_impl(r, a, b);
}

void fq_evals_from_constant(fq_evals *r, const fq_fe c)
{
    for (size_t i = 0; i < N; i++)
        fq_evals_set(r, i, c);
    r->degree = 0;
}

void fq_evals_from_linear(fq_evals *r, const fq_fe c)
{
    for (size_t i = 0; i < N; i++)
    {
        fq_fe xi, val;
        fq_from_small(xi, (int64_t)i);
        fq_sub(val, xi, c);
        fq_evals_set(r, i, val);
    }
    r->degree = 1;
}

static void fq_bary_eval(fq_fe result, const fq_evals *ev, const fq_fe c)
{
    shaw_eval_divisor_init();

    fq_fe_storage diffs[EVAL_DOMAIN_SIZE];
    for (size_t j = 0; j < N; j++)
    {
        fq_fe xj;
        fq_from_small(xj, (int64_t)j);
        fq_sub(diffs[j].v, c, xj);
    }

    fq_fe_storage inv_diffs[EVAL_DOMAIN_SIZE];
    fq_batch_invert(&inv_diffs[0].v, &diffs[0].v, N);

    fq_fe L;
    fq_copy(L, diffs[0].v);
    for (size_t j = 1; j < N; j++)
        fq_mul(L, L, diffs[j].v);

    fq_fe sum;
    fq_0(sum);
    for (size_t j = 0; j < N; j++)
    {
        fq_fe wj, fj, term;
        fq_evals_get(wj, &FQ_BARY_WEIGHTS, j);
        fq_evals_get(fj, ev, j);
        fq_mul(term, wj, fj);
        fq_mul(term, term, inv_diffs[j].v);
        fq_add(sum, sum, term);
    }

    fq_mul(result, L, sum);
}

void fq_evals_to_poly(fq_poly *out, const fq_evals *ev)
{
    shaw_eval_divisor_init();

    size_t deg = ev->degree;
    size_t n = deg + 1;

    std::vector<fq_fe_storage> xs(n), ys(n);
    for (size_t i = 0; i < n; i++)
    {
        fq_from_small(xs[i].v, (int64_t)i);
        fq_evals_get(ys[i].v, ev, i);
    }

    fq_poly_interpolate(out, reinterpret_cast<const fq_fe *>(xs.data()), reinterpret_cast<const fq_fe *>(ys.data()), n);
}

void fq_evals_div_linear(fq_evals *q, const fq_evals *f, const fq_fe c)
{
    shaw_eval_divisor_init();

    fq_fe fc;
    fq_bary_eval(fc, f, c);

    fq_fe_storage diffs[EVAL_DOMAIN_SIZE];
    fq_fe_storage nums[EVAL_DOMAIN_SIZE];
    for (size_t j = 0; j < N; j++)
    {
        fq_fe xj, fj;
        fq_from_small(xj, (int64_t)j);
        fq_sub(diffs[j].v, xj, c);
        fq_evals_get(fj, f, j);
        fq_sub(nums[j].v, fj, fc);
    }

    fq_fe_storage inv_diffs[EVAL_DOMAIN_SIZE];
    fq_batch_invert(&inv_diffs[0].v, &diffs[0].v, N);

    for (size_t j = 0; j < N; j++)
    {
        fq_fe qj;
        fq_mul(qj, nums[j].v, inv_diffs[j].v);
        fq_evals_set(q, j, qj);
    }

    q->degree = (f->degree > 0) ? f->degree - 1 : 0;
}

/* ================================================================
 * Ran eval-domain divisor operations
 * ================================================================ */

/*
 * Divisor multiplication using the curve equation y^2 = x^3 - 3x + b.
 *
 * Given D1 = a1(x) - y*b1(x) and D2 = a2(x) - y*b2(x):
 *   result.a[i] = a1[i]*a2[i] + curve[i]*b1[i]*b2[i]
 *   result.b[i] = a1[i]*b2[i] + a2[i]*b1[i]
 */
void ran_eval_divisor_mul(ran_eval_divisor *r, const ran_eval_divisor *d1, const ran_eval_divisor *d2)
{
    ran_eval_divisor_init();

    ran_eval_divisor_mul_impl(r, d1, d2, &FP_CURVE_EVALS);

    r->a.degree = d1->a.degree + d2->a.degree;
    r->b.degree = d1->b.degree + d2->b.degree;
    /* Adjust: a degree could be max(deg(a1)+deg(a2), 3+deg(b1)+deg(b2)) */
    size_t ab_deg = 3 + d1->b.degree + d2->b.degree;
    if (ab_deg > r->a.degree)
        r->a.degree = ab_deg;
    /* b degree is max(deg(a1)+deg(b2), deg(a2)+deg(b1)) */
    size_t b_deg1 = d1->a.degree + d2->b.degree;
    size_t b_deg2 = d2->a.degree + d1->b.degree;
    r->b.degree = (b_deg1 > b_deg2) ? b_deg1 : b_deg2;
}

/*
 * Create eval-domain divisor for a single affine point P = (px, py).
 *
 * From Lagrange interpolation (matching ran_compute_divisor for n=1):
 *   b(x) = py (constant), a(x) = py^2 (constant)
 *
 * D(px, py) = py^2 - py * py = 0. Correct.
 * The product formula a1*a2 + curve*b1*b2 then builds the combined witness
 * that vanishes at all input points when multiplied together.
 */
void ran_eval_divisor_from_point(ran_eval_divisor *d, const ran_affine *point)
{
    fp_fe pysq;
    fp_sq(pysq, point->y);
    fp_evals_from_constant(&d->a, pysq);
    fp_evals_from_constant(&d->b, point->y);
}

void ran_eval_divisor_merge(
    ran_eval_divisor *r,
    const ran_eval_divisor *d1,
    const ran_eval_divisor *d2,
    const ran_affine * /* sum1 */,
    const ran_affine * /* sum2 */,
    const ran_affine * /* sum_total */)
{
    /*
     * Merge two divisors via curve-equation multiplication.
     *
     * The product D1*D2 in the function field F(C)[y]/(y^2 - curve(x)):
     *   result.a = a1*a2 + curve*b1*b2
     *   result.b = a1*b2 + a2*b1
     *
     * This directly produces a valid divisor witness that vanishes at the
     * union of points from both inputs. The curve-equation substitution
     * (y^2 -> curve(x)) handles the algebraic reduction implicitly.
     *
     * Degrees grow at each merge level (a degree roughly doubles + 3 from
     * curve term), but for a tree of depth ~log2(n) starting from degree-0
     * leaves, the root has degree O(n) which fits in our N=256 domain.
     *
     * The sum points are accepted for API compatibility with potential
     * future optimizations.
     */
    ran_eval_divisor_mul(r, d1, d2);
}

void ran_eval_divisor_to_divisor(ran_divisor *out, const ran_eval_divisor *ed)
{
    fp_evals_to_poly(&out->a, &ed->a);
    fp_evals_to_poly(&out->b, &ed->b);
}

void ran_eval_divisor_tree_reduce(
    ran_eval_divisor *out,
    ran_eval_divisor *divisors,
    ran_affine *points,
    size_t n)
{
    if (n == 0)
        return;
    if (n == 1)
    {
        *out = divisors[0];
        return;
    }

    /* Pairwise merge in a tree */
    std::vector<ran_eval_divisor> current(divisors, divisors + n);
    std::vector<ran_affine> sums(points, points + n);

    while (current.size() > 1)
    {
        size_t m = current.size();
        size_t pairs = m / 2;
        std::vector<ran_eval_divisor> next(pairs + (m & 1));
        std::vector<ran_affine> next_sums(pairs + (m & 1));

        /* Compute pairwise EC sums */
        for (size_t i = 0; i < pairs; i++)
        {
            ran_jacobian j1, j2, jsum;
            ran_from_affine(&j1, &sums[2 * i]);
            ran_from_affine(&j2, &sums[2 * i + 1]);
            ran_add_safe(&jsum, &j1, &j2);
            ran_to_affine(&next_sums[i], &jsum);

            ran_eval_divisor_merge(
                &next[i], &current[2 * i], &current[2 * i + 1], &sums[2 * i], &sums[2 * i + 1], &next_sums[i]);
        }

        /* Handle odd element */
        if (m & 1)
        {
            next[pairs] = current[m - 1];
            next_sums[pairs] = sums[m - 1];
        }

        current = std::move(next);
        sums = std::move(next_sums);
    }

    *out = current[0];
}

void ran_scalar_mul_divisor(ran_divisor *d, const unsigned char *scalar, const ran_affine *point)
{
    ran_eval_divisor_init();

    /* Collect all intermediate points from the double-and-add ladder */
    ran_jacobian P;
    ran_from_affine(&P, point);

    /* Constant-time highest-bit scan: no early exit, no secret-dependent branch */
    int highest_bit = -1;
    for (int i = 0; i < 256; i++)
    {
        int bit = (scalar[i / 8] >> (i % 8)) & 1;
        highest_bit = highest_bit ^ ((highest_bit ^ i) & (-bit));
    }

    if (highest_bit < 0)
    {
        /* scalar is zero: empty divisor */
        d->a.coeffs.resize(1);
        fp_0(d->a.coeffs[0].v);
        d->b.coeffs.resize(1);
        fp_0(d->b.coeffs[0].v);
        ranshaw_secure_erase(&P, sizeof(P));
        return;
    }

    /* Collect points: P appears once for each set bit in the scalar.
     * Always scan all 256 bits (the output divisor degree reveals
     * the Hamming weight regardless, so the if-branch is not a leak). */
    std::vector<ran_affine> add_points;
    ran_affine pt_affine;
    ran_to_affine(&pt_affine, &P);

    for (int i = 255; i >= 0; i--)
    {
        if ((scalar[i / 8] >> (i % 8)) & 1)
        {
            add_points.push_back(pt_affine);
        }
    }

    size_t n = add_points.size();
    if (n == 1)
    {
        ran_compute_divisor(d, add_points.data(), 1);
        ranshaw_secure_erase(&P, sizeof(P));
        ranshaw_secure_erase(&pt_affine, sizeof(pt_affine));
        ranshaw_secure_erase(add_points.data(), n * sizeof(ran_affine));
        return;
    }

    /* Create eval-domain divisors for each point */
    std::vector<ran_eval_divisor> divs(n);
    for (size_t i = 0; i < n; i++)
        ran_eval_divisor_from_point(&divs[i], &add_points[i]);

    /* Tree reduce */
    ran_eval_divisor result;
    ran_eval_divisor_tree_reduce(&result, divs.data(), add_points.data(), n);

    /* Convert to coefficient domain */
    ran_eval_divisor_to_divisor(d, &result);

    /* Erase scalar-derived intermediates */
    ranshaw_secure_erase(&P, sizeof(P));
    ranshaw_secure_erase(&pt_affine, sizeof(pt_affine));
    ranshaw_secure_erase(add_points.data(), n * sizeof(ran_affine));
}

/* ================================================================
 * Shaw eval-domain divisor operations
 * ================================================================ */

void shaw_eval_divisor_mul(shaw_eval_divisor *r, const shaw_eval_divisor *d1, const shaw_eval_divisor *d2)
{
    shaw_eval_divisor_init();

    shaw_eval_divisor_mul_impl(r, d1, d2, &FQ_CURVE_EVALS);

    r->a.degree = d1->a.degree + d2->a.degree;
    r->b.degree = d1->b.degree + d2->b.degree;
    size_t ab_deg = 3 + d1->b.degree + d2->b.degree;
    if (ab_deg > r->a.degree)
        r->a.degree = ab_deg;
    size_t b_deg1 = d1->a.degree + d2->b.degree;
    size_t b_deg2 = d2->a.degree + d1->b.degree;
    r->b.degree = (b_deg1 > b_deg2) ? b_deg1 : b_deg2;
}

void shaw_eval_divisor_from_point(shaw_eval_divisor *d, const shaw_affine *point)
{
    fq_fe pysq;
    fq_sq(pysq, point->y);
    fq_evals_from_constant(&d->a, pysq);
    fq_evals_from_constant(&d->b, point->y);
}

void shaw_eval_divisor_merge(
    shaw_eval_divisor *r,
    const shaw_eval_divisor *d1,
    const shaw_eval_divisor *d2,
    const shaw_affine * /* sum1 */,
    const shaw_affine * /* sum2 */,
    const shaw_affine * /* sum_total */)
{
    shaw_eval_divisor_mul(r, d1, d2);
}

void shaw_eval_divisor_to_divisor(shaw_divisor *out, const shaw_eval_divisor *ed)
{
    fq_evals_to_poly(&out->a, &ed->a);
    fq_evals_to_poly(&out->b, &ed->b);
}

void shaw_eval_divisor_tree_reduce(
    shaw_eval_divisor *out,
    shaw_eval_divisor *divisors,
    shaw_affine *points,
    size_t n)
{
    if (n == 0)
        return;
    if (n == 1)
    {
        *out = divisors[0];
        return;
    }

    std::vector<shaw_eval_divisor> current(divisors, divisors + n);
    std::vector<shaw_affine> sums(points, points + n);

    while (current.size() > 1)
    {
        size_t m = current.size();
        size_t pairs = m / 2;
        std::vector<shaw_eval_divisor> next(pairs + (m & 1));
        std::vector<shaw_affine> next_sums(pairs + (m & 1));

        for (size_t i = 0; i < pairs; i++)
        {
            shaw_jacobian j1, j2, jsum;
            shaw_from_affine(&j1, &sums[2 * i]);
            shaw_from_affine(&j2, &sums[2 * i + 1]);
            shaw_add_safe(&jsum, &j1, &j2);
            shaw_to_affine(&next_sums[i], &jsum);

            shaw_eval_divisor_merge(
                &next[i], &current[2 * i], &current[2 * i + 1], &sums[2 * i], &sums[2 * i + 1], &next_sums[i]);
        }

        if (m & 1)
        {
            next[pairs] = current[m - 1];
            next_sums[pairs] = sums[m - 1];
        }

        current = std::move(next);
        sums = std::move(next_sums);
    }

    *out = current[0];
}

void shaw_scalar_mul_divisor(shaw_divisor *d, const unsigned char *scalar, const shaw_affine *point)
{
    shaw_eval_divisor_init();

    shaw_jacobian P;
    shaw_from_affine(&P, point);

    /* Constant-time highest-bit scan: no early exit, no secret-dependent branch */
    int highest_bit = -1;
    for (int i = 0; i < 256; i++)
    {
        int bit = (scalar[i / 8] >> (i % 8)) & 1;
        highest_bit = highest_bit ^ ((highest_bit ^ i) & (-bit));
    }

    if (highest_bit < 0)
    {
        d->a.coeffs.resize(1);
        fq_0(d->a.coeffs[0].v);
        d->b.coeffs.resize(1);
        fq_0(d->b.coeffs[0].v);
        ranshaw_secure_erase(&P, sizeof(P));
        return;
    }

    /* Collect points: P appears once for each set bit in the scalar.
     * Always scan all 256 bits (the output divisor degree reveals
     * the Hamming weight regardless, so the if-branch is not a leak). */
    std::vector<shaw_affine> add_points;
    shaw_affine pt_affine;
    shaw_to_affine(&pt_affine, &P);

    for (int i = 255; i >= 0; i--)
    {
        if ((scalar[i / 8] >> (i % 8)) & 1)
        {
            add_points.push_back(pt_affine);
        }
    }

    size_t n = add_points.size();
    if (n == 1)
    {
        shaw_compute_divisor(d, add_points.data(), 1);
        ranshaw_secure_erase(&P, sizeof(P));
        ranshaw_secure_erase(&pt_affine, sizeof(pt_affine));
        ranshaw_secure_erase(add_points.data(), n * sizeof(shaw_affine));
        return;
    }

    std::vector<shaw_eval_divisor> divs(n);
    for (size_t i = 0; i < n; i++)
        shaw_eval_divisor_from_point(&divs[i], &add_points[i]);

    shaw_eval_divisor result;
    shaw_eval_divisor_tree_reduce(&result, divs.data(), add_points.data(), n);

    shaw_eval_divisor_to_divisor(d, &result);

    /* Erase scalar-derived intermediates */
    ranshaw_secure_erase(&P, sizeof(P));
    ranshaw_secure_erase(&pt_affine, sizeof(pt_affine));
    ranshaw_secure_erase(add_points.data(), n * sizeof(shaw_affine));
}
