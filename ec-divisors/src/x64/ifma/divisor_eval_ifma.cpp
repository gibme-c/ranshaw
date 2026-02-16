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
 * IFMA 8-way eval-domain divisor operations with SoA aligned load/store.
 *
 * SoA layout: limbs[j][256] — all limb-j values are contiguous.
 * Each limb array is 64-byte aligned, so _mm512_load_si512 loads 8 consecutive
 * limb-j values in a single ~1-cycle instruction (vs ~10-cycle gather).
 *
 * 256 elements / 8 lanes = 32 iterations per loop.
 */

#include "divisor_eval_internal.h"
#include "x64/ifma/fp51x8_ifma.h"
#include "x64/ifma/fq51x8_ifma.h"

static const size_t N = EVAL_DOMAIN_SIZE;

/* Load 8 elements from SoA fp_evals at offset i into fp51x8 */
static inline void fp51x8_load_soa(fp51x8 *out, const fp_evals *ev, size_t i)
{
    for (int j = 0; j < 5; j++)
        out->v[j] = _mm512_load_si512((__m512i *)&ev->limbs[j][i]);
}

/* Store fp51x8 into SoA fp_evals at offset i */
static inline void fp51x8_store_soa(fp_evals *ev, size_t i, const fp51x8 *in)
{
    for (int j = 0; j < 5; j++)
        _mm512_store_si512((__m512i *)&ev->limbs[j][i], in->v[j]);
}

/* Load 8 elements from SoA fq_evals at offset i into fq51x8 */
static inline void fq51x8_load_soa(fq51x8 *out, const fq_evals *ev, size_t i)
{
    for (int j = 0; j < 5; j++)
        out->v[j] = _mm512_load_si512((__m512i *)&ev->limbs[j][i]);
}

/* Store fq51x8 into SoA fq_evals at offset i */
static inline void fq51x8_store_soa(fq_evals *ev, size_t i, const fq51x8 *in)
{
    for (int j = 0; j < 5; j++)
        _mm512_store_si512((__m512i *)&ev->limbs[j][i], in->v[j]);
}

/* ================================================================
 * Fp pointwise multiply (IFMA 8-way)
 * ================================================================ */

void fp_evals_mul_ifma(fp_evals *r, const fp_evals *a, const fp_evals *b)
{
    for (size_t i = 0; i < N; i += 8)
    {
        fp51x8 va, vb, vr;
        fp51x8_load_soa(&va, a, i);
        fp51x8_load_soa(&vb, b, i);
        fp51x8_mul(&vr, &va, &vb);
        fp51x8_store_soa(r, i, &vr);
    }
    r->degree = a->degree + b->degree;
}

/* ================================================================
 * Fq pointwise multiply (IFMA 8-way)
 * ================================================================ */

void fq_evals_mul_ifma(fq_evals *r, const fq_evals *a, const fq_evals *b)
{
    for (size_t i = 0; i < N; i += 8)
    {
        fq51x8 va, vb, vr;
        fq51x8_load_soa(&va, a, i);
        fq51x8_load_soa(&vb, b, i);
        fq51x8_mul(&vr, &va, &vb);
        fq51x8_store_soa(r, i, &vr);
    }
    r->degree = a->degree + b->degree;
}

/* ================================================================
 * Fp/Fq pointwise add (IFMA 8-way)
 * ================================================================ */

void fp_evals_add_ifma(fp_evals *r, const fp_evals *a, const fp_evals *b)
{
    for (size_t i = 0; i < N; i += 8)
    {
        fp51x8 va, vb, vr;
        fp51x8_load_soa(&va, a, i);
        fp51x8_load_soa(&vb, b, i);
        fp51x8_add(&vr, &va, &vb);
        fp51x8_store_soa(r, i, &vr);
    }
    r->degree = (a->degree > b->degree) ? a->degree : b->degree;
}

void fq_evals_add_ifma(fq_evals *r, const fq_evals *a, const fq_evals *b)
{
    for (size_t i = 0; i < N; i += 8)
    {
        fq51x8 va, vb, vr;
        fq51x8_load_soa(&va, a, i);
        fq51x8_load_soa(&vb, b, i);
        fq51x8_add(&vr, &va, &vb);
        fq51x8_store_soa(r, i, &vr);
    }
    r->degree = (a->degree > b->degree) ? a->degree : b->degree;
}

/* ================================================================
 * Fp/Fq pointwise sub (IFMA 8-way)
 * ================================================================ */

void fp_evals_sub_ifma(fp_evals *r, const fp_evals *a, const fp_evals *b)
{
    for (size_t i = 0; i < N; i += 8)
    {
        fp51x8 va, vb, vr;
        fp51x8_load_soa(&va, a, i);
        fp51x8_load_soa(&vb, b, i);
        fp51x8_sub(&vr, &va, &vb);
        fp51x8_store_soa(r, i, &vr);
    }
    r->degree = (a->degree > b->degree) ? a->degree : b->degree;
}

void fq_evals_sub_ifma(fq_evals *r, const fq_evals *a, const fq_evals *b)
{
    for (size_t i = 0; i < N; i += 8)
    {
        fq51x8 va, vb, vr;
        fq51x8_load_soa(&va, a, i);
        fq51x8_load_soa(&vb, b, i);
        fq51x8_sub(&vr, &va, &vb);
        fq51x8_store_soa(r, i, &vr);
    }
    r->degree = (a->degree > b->degree) ? a->degree : b->degree;
}

/* ================================================================
 * Ran eval-domain divisor multiplication (IFMA 8-way)
 *
 * Per 8 elements:
 *   5 aligned loads × 5 (a1, a2, b1, b2, curve)
 *   4 muls + 3 adds + 2 subs
 *   2 × 5 aligned stores (ra, rb)
 * ================================================================ */

void ran_eval_divisor_mul_ifma(
    ran_eval_divisor *r,
    const ran_eval_divisor *d1,
    const ran_eval_divisor *d2,
    const fp_evals *curve_evals)
{
    for (size_t i = 0; i < N; i += 8)
    {
        fp51x8 a1, a2, b1, b2, curve;
        fp51x8_load_soa(&a1, &d1->a, i);
        fp51x8_load_soa(&a2, &d2->a, i);
        fp51x8_load_soa(&b1, &d1->b, i);
        fp51x8_load_soa(&b2, &d2->b, i);
        fp51x8_load_soa(&curve, curve_evals, i);

        fp51x8 a1a2;
        fp51x8_mul(&a1a2, &a1, &a2);

        fp51x8 b1b2;
        fp51x8_mul(&b1b2, &b1, &b2);

        fp51x8 cb1b2;
        fp51x8_mul(&cb1b2, &curve, &b1b2);

        fp51x8 ra;
        fp51x8_add(&ra, &a1a2, &cb1b2);

        fp51x8 t1, t2, t3, rb;
        fp51x8_add(&t1, &a1, &b1);
        fp51x8_add(&t2, &a2, &b2);
        fp51x8_mul(&t3, &t1, &t2);
        fp51x8_sub(&t3, &t3, &a1a2);
        fp51x8_sub(&rb, &t3, &b1b2);

        fp51x8_store_soa(&r->a, i, &ra);
        fp51x8_store_soa(&r->b, i, &rb);
    }
}

/* ================================================================
 * Shaw eval-domain divisor multiplication (IFMA 8-way)
 * ================================================================ */

void shaw_eval_divisor_mul_ifma(
    shaw_eval_divisor *r,
    const shaw_eval_divisor *d1,
    const shaw_eval_divisor *d2,
    const fq_evals *curve_evals)
{
    for (size_t i = 0; i < N; i += 8)
    {
        fq51x8 a1, a2, b1, b2, curve;
        fq51x8_load_soa(&a1, &d1->a, i);
        fq51x8_load_soa(&a2, &d2->a, i);
        fq51x8_load_soa(&b1, &d1->b, i);
        fq51x8_load_soa(&b2, &d2->b, i);
        fq51x8_load_soa(&curve, curve_evals, i);

        fq51x8 a1a2;
        fq51x8_mul(&a1a2, &a1, &a2);

        fq51x8 b1b2;
        fq51x8_mul(&b1b2, &b1, &b2);

        fq51x8 cb1b2;
        fq51x8_mul(&cb1b2, &curve, &b1b2);

        fq51x8 ra;
        fq51x8_add(&ra, &a1a2, &cb1b2);

        fq51x8 t1, t2, t3, rb;
        fq51x8_add(&t1, &a1, &b1);
        fq51x8_add(&t2, &a2, &b2);
        fq51x8_mul(&t3, &t1, &t2);
        fq51x8_sub(&t3, &t3, &a1a2);
        fq51x8_sub(&rb, &t3, &b1b2);

        fq51x8_store_soa(&r->a, i, &ra);
        fq51x8_store_soa(&r->b, i, &rb);
    }
}
