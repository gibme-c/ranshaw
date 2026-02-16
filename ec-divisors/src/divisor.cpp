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

// divisor.cpp — EC-divisor witness computation via Lagrange interpolation.
// For a set of affine points {(x_i, y_i)}, builds D(x,y) = a(x) - y*b(x)
// where b interpolates the y-coordinates and a interpolates y^2 values.

#include "divisor.h"

#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "poly.h"

/* ================================================================
 * Ran (F_p) divisor operations
 * ================================================================ */

/*
 * Compute divisor witness D(x,y) = a(x) - y*b(x) for a set of affine points.
 *
 * Construction via Lagrange interpolation:
 *   b(x) interpolates the y-coordinates through the x-coordinates
 *   a(x) interpolates the y^2 values through the x-coordinates
 *
 * Then D(x_i, y_i) = a(x_i) - y_i * b(x_i) = y_i^2 - y_i * y_i = 0.
 */
void ran_compute_divisor(ran_divisor *d, const ran_affine *points, size_t n)
{
    if (n == 0)
    {
        /* Degenerate: return zero divisor */
        d->a.coeffs.resize(1);
        fp_0(d->a.coeffs[0].v);
        d->b.coeffs.resize(1);
        fp_0(d->b.coeffs[0].v);
        return;
    }

    /* Build flat arrays of x-coordinates, y-coordinates, and y^2 values */
    std::vector<fp_fe_storage> xs_store(n), ys_store(n), ysq_store(n);
    for (size_t i = 0; i < n; i++)
    {
        std::memcpy(xs_store[i].v, points[i].x, sizeof(fp_fe));
        std::memcpy(ys_store[i].v, points[i].y, sizeof(fp_fe));
        fp_sq(ysq_store[i].v, points[i].y);
    }

    const fp_fe *xs = reinterpret_cast<const fp_fe *>(xs_store.data());
    const fp_fe *ys = reinterpret_cast<const fp_fe *>(ys_store.data());
    const fp_fe *ysq = reinterpret_cast<const fp_fe *>(ysq_store.data());

    /* b(x) interpolates y-coordinates, a(x) interpolates y^2 values */
    fp_poly_interpolate(&d->b, xs, ys, n);
    fp_poly_interpolate(&d->a, xs, ysq, n);
}

void ran_evaluate_divisor(fp_fe result, const ran_divisor *d, const fp_fe x, const fp_fe y)
{
    fp_fe ax, bx, ybx;
    fp_poly_eval(ax, &d->a, x);
    fp_poly_eval(bx, &d->b, x);
    fp_mul(ybx, y, bx);
    fp_sub(result, ax, ybx);
}

/* ================================================================
 * Shaw (F_q) divisor operations
 * ================================================================ */

void shaw_compute_divisor(shaw_divisor *d, const shaw_affine *points, size_t n)
{
    if (n == 0)
    {
        d->a.coeffs.resize(1);
        fq_0(d->a.coeffs[0].v);
        d->b.coeffs.resize(1);
        fq_0(d->b.coeffs[0].v);
        return;
    }

    std::vector<fq_fe_storage> xs_store(n), ys_store(n), ysq_store(n);
    for (size_t i = 0; i < n; i++)
    {
        std::memcpy(xs_store[i].v, points[i].x, sizeof(fq_fe));
        std::memcpy(ys_store[i].v, points[i].y, sizeof(fq_fe));
        fq_sq(ysq_store[i].v, points[i].y);
    }

    const fq_fe *xs = reinterpret_cast<const fq_fe *>(xs_store.data());
    const fq_fe *ys = reinterpret_cast<const fq_fe *>(ys_store.data());
    const fq_fe *ysq = reinterpret_cast<const fq_fe *>(ysq_store.data());

    fq_poly_interpolate(&d->b, xs, ys, n);
    fq_poly_interpolate(&d->a, xs, ysq, n);
}

void shaw_evaluate_divisor(fq_fe result, const shaw_divisor *d, const fq_fe x, const fq_fe y)
{
    fq_fe ax, bx, ybx;
    fq_poly_eval(ax, &d->a, x);
    fq_poly_eval(bx, &d->b, x);
    fq_mul(ybx, y, bx);
    fq_sub(result, ax, ybx);
}
