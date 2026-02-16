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

/*
 * AVX2 4-way eval-domain divisor operations with SoA aligned load/store.
 *
 * SoA layout: limbs[j][256] — all limb-j values are contiguous.
 * Each limb array is 64-byte aligned, so _mm256_load_si256 loads 4 consecutive
 * limb-j values in a single ~1-cycle instruction (vs ~6-cycle gather).
 *
 * For add/sub: real 4-way SIMD (no scalar fallback needed).
 * For mul: scalar fp_mul/fq_mul per element (no _mm256_mullo_epi64 in pure AVX2).
 *
 * 256 elements / 4 lanes = 64 iterations for SIMD loops.
 */

#include "divisor_eval_internal.h"

#include <immintrin.h>

static const size_t N = EVAL_DOMAIN_SIZE;

/* ================================================================
 * Lightweight fp51x4 / fq51x4 for SoA load/store and add/sub
 * ================================================================ */

typedef struct
{
    __m256i v[5];
} fp51x4;

typedef struct
{
    __m256i v[5];
} fq51x4;

/* Load 4 elements from SoA fp_evals at offset i into fp51x4 */
static inline void fp51x4_load_soa(fp51x4 *out, const fp_evals *ev, size_t i)
{
    for (int j = 0; j < 5; j++)
        out->v[j] = _mm256_load_si256((__m256i *)&ev->limbs[j][i]);
}

/* Store fp51x4 into SoA fp_evals at offset i */
static inline void fp51x4_store_soa(fp_evals *ev, size_t i, const fp51x4 *in)
{
    for (int j = 0; j < 5; j++)
        _mm256_store_si256((__m256i *)&ev->limbs[j][i], in->v[j]);
}

/* Load 4 elements from SoA fq_evals at offset i into fq51x4 */
static inline void fq51x4_load_soa(fq51x4 *out, const fq_evals *ev, size_t i)
{
    for (int j = 0; j < 5; j++)
        out->v[j] = _mm256_load_si256((__m256i *)&ev->limbs[j][i]);
}

/* Store fq51x4 into SoA fq_evals at offset i */
static inline void fq51x4_store_soa(fq_evals *ev, size_t i, const fq51x4 *in)
{
    for (int j = 0; j < 5; j++)
        _mm256_store_si256((__m256i *)&ev->limbs[j][i], in->v[j]);
}

/* ---- Fp 4-way arithmetic (radix-2^51, p = 2^255 - 19) ---- */

static inline __m256i mask51(void)
{
    return _mm256_set1_epi64x(0x7ffffffffffffLL);
}

/* Lazy add: no carry */
static inline void fp51x4_add(fp51x4 *h, const fp51x4 *f, const fp51x4 *g)
{
    for (int i = 0; i < 5; i++)
        h->v[i] = _mm256_add_epi64(f->v[i], g->v[i]);
}

/* Sub with 4p bias + carry */
static inline void fp51x4_sub(fp51x4 *h, const fp51x4 *f, const fp51x4 *g)
{
    const __m256i bias0 = _mm256_set1_epi64x(0x1fffffffffffb4LL);
    const __m256i bias1 = _mm256_set1_epi64x(0x1ffffffffffffcLL);
    h->v[0] = _mm256_add_epi64(_mm256_sub_epi64(f->v[0], g->v[0]), bias0);
    for (int i = 1; i < 5; i++)
        h->v[i] = _mm256_add_epi64(_mm256_sub_epi64(f->v[i], g->v[i]), bias1);

    __m256i c;
    c = _mm256_srli_epi64(h->v[0], 51);
    h->v[1] = _mm256_add_epi64(h->v[1], c);
    h->v[0] = _mm256_and_si256(h->v[0], mask51());
    c = _mm256_srli_epi64(h->v[1], 51);
    h->v[2] = _mm256_add_epi64(h->v[2], c);
    h->v[1] = _mm256_and_si256(h->v[1], mask51());
    c = _mm256_srli_epi64(h->v[2], 51);
    h->v[3] = _mm256_add_epi64(h->v[3], c);
    h->v[2] = _mm256_and_si256(h->v[2], mask51());
    c = _mm256_srli_epi64(h->v[3], 51);
    h->v[4] = _mm256_add_epi64(h->v[4], c);
    h->v[3] = _mm256_and_si256(h->v[3], mask51());
    c = _mm256_srli_epi64(h->v[4], 51);
    /* c*19 = (c<<4) + (c<<1) + c (avoids _mm256_mullo_epi64 which needs AVX-512VL) */
    __m256i c19 = _mm256_add_epi64(_mm256_add_epi64(_mm256_slli_epi64(c, 4), _mm256_slli_epi64(c, 1)), c);
    h->v[0] = _mm256_add_epi64(h->v[0], c19);
    h->v[4] = _mm256_and_si256(h->v[4], mask51());
}

/* Lazy fq add: no carry (same as fp) */
static inline void fq51x4_add(fq51x4 *h, const fq51x4 *f, const fq51x4 *g)
{
    for (int i = 0; i < 5; i++)
        h->v[i] = _mm256_add_epi64(f->v[i], g->v[i]);
}

#include "fp_mul.h"
#include "fp_ops.h"
#include "fq_mul.h"
#include "fq_ops.h"

/* ================================================================
 * Fp pointwise add/sub (AVX2 4-way SIMD)
 * ================================================================ */

void fp_evals_add_avx2(fp_evals *r, const fp_evals *a, const fp_evals *b)
{
    for (size_t i = 0; i < N; i += 4)
    {
        fp51x4 va, vb, vr;
        fp51x4_load_soa(&va, a, i);
        fp51x4_load_soa(&vb, b, i);
        fp51x4_add(&vr, &va, &vb);
        fp51x4_store_soa(r, i, &vr);
    }
    r->degree = (a->degree > b->degree) ? a->degree : b->degree;
}

void fp_evals_sub_avx2(fp_evals *r, const fp_evals *a, const fp_evals *b)
{
    for (size_t i = 0; i < N; i += 4)
    {
        fp51x4 va, vb, vr;
        fp51x4_load_soa(&va, a, i);
        fp51x4_load_soa(&vb, b, i);
        fp51x4_sub(&vr, &va, &vb);
        fp51x4_store_soa(r, i, &vr);
    }
    r->degree = (a->degree > b->degree) ? a->degree : b->degree;
}

/* ================================================================
 * Fq pointwise add/sub (AVX2 4-way SIMD)
 * ================================================================ */

void fq_evals_add_avx2(fq_evals *r, const fq_evals *a, const fq_evals *b)
{
    for (size_t i = 0; i < N; i += 4)
    {
        fq51x4 va, vb, vr;
        fq51x4_load_soa(&va, a, i);
        fq51x4_load_soa(&vb, b, i);
        fq51x4_add(&vr, &va, &vb);
        fq51x4_store_soa(r, i, &vr);
    }
    r->degree = (a->degree > b->degree) ? a->degree : b->degree;
}

void fq_evals_sub_avx2(fq_evals *r, const fq_evals *a, const fq_evals *b)
{
    /* Fq sub needs Crandall gamma fold — use scalar (no _mm256_mullo_epi64 in AVX2) */
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

/* ================================================================
 * Fp/Fq pointwise multiply (AVX2: scalar — no _mm256_mullo_epi64)
 * ================================================================ */

void fp_evals_mul_avx2(fp_evals *r, const fp_evals *a, const fp_evals *b)
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

void fq_evals_mul_avx2(fq_evals *r, const fq_evals *a, const fq_evals *b)
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

/* ================================================================
 * Ran eval-domain divisor multiplication (AVX2: scalar mul, accessor pattern)
 * ================================================================ */

void ran_eval_divisor_mul_avx2(
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

/* ================================================================
 * Shaw eval-domain divisor multiplication (AVX2: scalar mul, accessor pattern)
 * ================================================================ */

void shaw_eval_divisor_mul_avx2(
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
