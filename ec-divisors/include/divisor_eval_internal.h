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
 * @file divisor_eval_internal.h
 * @brief Internal helpers for divisor evaluation: polynomial evaluation kernels
 *        and coefficient access.
 */

#ifndef RANSHAW_DIVISOR_EVAL_INTERNAL_H
#define RANSHAW_DIVISOR_EVAL_INTERNAL_H

#include "divisor_eval.h"

/* ================================================================
 * SIMD-accelerated eval-domain divisor operations.
 *
 * Internal dispatch table — NOT part of the public 6-slot dispatch.
 * Initialized during ran_eval_divisor_init() / shaw_eval_divisor_init().
 *
 * curve_evals parameters are needed because the precomputed curve
 * evaluation tables are file-static in divisor_eval.cpp.
 * ================================================================ */

/* Function pointer types */
typedef void (*fp_evals_mul_fn_t)(fp_evals *r, const fp_evals *a, const fp_evals *b);
typedef void (*fq_evals_mul_fn_t)(fq_evals *r, const fq_evals *a, const fq_evals *b);
typedef void (*fp_evals_add_fn_t)(fp_evals *r, const fp_evals *a, const fp_evals *b);
typedef void (*fq_evals_add_fn_t)(fq_evals *r, const fq_evals *a, const fq_evals *b);
typedef void (*fp_evals_sub_fn_t)(fp_evals *r, const fp_evals *a, const fp_evals *b);
typedef void (*fq_evals_sub_fn_t)(fq_evals *r, const fq_evals *a, const fq_evals *b);
typedef void (*ran_eval_divisor_mul_fn_t)(
    ran_eval_divisor *r,
    const ran_eval_divisor *d1,
    const ran_eval_divisor *d2,
    const fp_evals *curve_evals);
typedef void (*shaw_eval_divisor_mul_fn_t)(
    shaw_eval_divisor *r,
    const shaw_eval_divisor *d1,
    const shaw_eval_divisor *d2,
    const fq_evals *curve_evals);

#if RANSHAW_SIMD

/* ---- AVX-512 IFMA 8-way (256/8 = 32 iterations) ---- */

#ifndef RANSHAW_NO_AVX512

void fp_evals_mul_ifma(fp_evals *r, const fp_evals *a, const fp_evals *b);
void fq_evals_mul_ifma(fq_evals *r, const fq_evals *a, const fq_evals *b);
void fp_evals_add_ifma(fp_evals *r, const fp_evals *a, const fp_evals *b);
void fq_evals_add_ifma(fq_evals *r, const fq_evals *a, const fq_evals *b);
void fp_evals_sub_ifma(fp_evals *r, const fp_evals *a, const fp_evals *b);
void fq_evals_sub_ifma(fq_evals *r, const fq_evals *a, const fq_evals *b);
void ran_eval_divisor_mul_ifma(
    ran_eval_divisor *r,
    const ran_eval_divisor *d1,
    const ran_eval_divisor *d2,
    const fp_evals *curve_evals);
void shaw_eval_divisor_mul_ifma(
    shaw_eval_divisor *r,
    const shaw_eval_divisor *d1,
    const shaw_eval_divisor *d2,
    const fq_evals *curve_evals);

#endif /* !RANSHAW_NO_AVX512 */

/* ---- AVX2 4-way (256/4 = 64 iterations) ---- */

#ifndef RANSHAW_NO_AVX2

void fp_evals_mul_avx2(fp_evals *r, const fp_evals *a, const fp_evals *b);
void fq_evals_mul_avx2(fq_evals *r, const fq_evals *a, const fq_evals *b);
void fp_evals_add_avx2(fp_evals *r, const fp_evals *a, const fp_evals *b);
void fq_evals_add_avx2(fq_evals *r, const fq_evals *a, const fq_evals *b);
void fp_evals_sub_avx2(fp_evals *r, const fp_evals *a, const fp_evals *b);
void fq_evals_sub_avx2(fq_evals *r, const fq_evals *a, const fq_evals *b);
void ran_eval_divisor_mul_avx2(
    ran_eval_divisor *r,
    const ran_eval_divisor *d1,
    const ran_eval_divisor *d2,
    const fp_evals *curve_evals);
void shaw_eval_divisor_mul_avx2(
    shaw_eval_divisor *r,
    const shaw_eval_divisor *d1,
    const shaw_eval_divisor *d2,
    const fq_evals *curve_evals);

#endif /* !RANSHAW_NO_AVX2 */

#endif /* RANSHAW_SIMD */

#endif /* RANSHAW_DIVISOR_EVAL_INTERNAL_H */
