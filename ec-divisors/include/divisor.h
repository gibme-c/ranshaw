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
 * @file divisor.h
 * @brief EC-divisor computation: build the rational function f(x,y) = a(x) + y*b(x)
 *        whose zeros are a given set of curve points.
 */

#ifndef RANSHAW_DIVISOR_H
#define RANSHAW_DIVISOR_H

#include "ran.h"
#include "poly.h"
#include "shaw.h"

/**
 * EC-Divisor witness: D(x,y) = a(x) - y * b(x)
 * Represents a divisor as two polynomials over the base field.
 */
struct ran_divisor
{
    fp_poly a; /* a(x) polynomial */
    fp_poly b; /* b(x) polynomial */
};

struct shaw_divisor
{
    fq_poly a;
    fq_poly b;
};

/**
 * Compute a divisor witness for a set of affine points.
 * The divisor D = a(x) - y*b(x) vanishes at exactly these points.
 *
 * @param d      Output divisor
 * @param points Array of affine points
 * @param n      Number of points (must be >= 1)
 */
void ran_compute_divisor(ran_divisor *d, const ran_affine *points, size_t n);

void shaw_compute_divisor(shaw_divisor *d, const shaw_affine *points, size_t n);

/**
 * Evaluate a divisor at a point (x, y).
 * result = a(x) - y * b(x)
 */
void ran_evaluate_divisor(fp_fe result, const ran_divisor *d, const fp_fe x, const fp_fe y);

void shaw_evaluate_divisor(fq_fe result, const shaw_divisor *d, const fq_fe x, const fq_fe y);

#endif // RANSHAW_DIVISOR_H
