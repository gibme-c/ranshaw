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

#ifndef RANSHAW_SHAW_BATCH_AFFINE_H
#define RANSHAW_SHAW_BATCH_AFFINE_H

/**
 * @file shaw_batch_affine.h
 * @brief Batch Jacobian-to-affine conversion for Shaw using Montgomery's trick.
 *
 * Converts n Jacobian points to affine using 1 inversion + 3(n-1) multiplications,
 * instead of n separate inversions.
 */

#include "fq_batch_invert.h"
#include "fq_cmov.h"
#include "fq_mul.h"
#include "fq_sq.h"
#include "shaw.h"
#include "shaw_ops.h"

#include <vector>

/**
 * Convert n Jacobian points to affine using Montgomery's trick.
 * Identity points (Z==0) are mapped to (0,0).
 *
 * @param out   Output array of n affine points
 * @param points Input array of n Jacobian points
 * @param n     Number of points
 */
static inline void shaw_batch_to_affine(shaw_affine *out, const shaw_jacobian *points, size_t n)
{
    if (n == 0)
        return;
    if (n == 1)
    {
        /* CT: always compute affine, then cmov to (0,0) if identity */
        shaw_to_affine(&out[0], &points[0]);
        unsigned int is_ident = 1u - (unsigned int)fq_isnonzero(points[0].Z);
        fq_fe zero;
        fq_0(zero);
        fq_cmov(out[0].x, zero, is_ident);
        fq_cmov(out[0].y, zero, is_ident);
        return;
    }

    /* Extract Z coordinates and batch-invert */
    struct fq_fe_s
    {
        fq_fe v;
    };
    std::vector<fq_fe_s> zs(n);
    std::vector<fq_fe_s> zinvs(n);
    for (size_t i = 0; i < n; i++)
        fq_copy(zs[i].v, points[i].Z);

    fq_batch_invert(&zinvs[0].v, &zs[0].v, n);

    /* Convert each point using its Z inverse (CT: always compute, cmov to zero if identity) */
    fq_fe zero;
    fq_0(zero);
    for (size_t i = 0; i < n; i++)
    {
        fq_fe zi2, zi3;
        fq_sq(zi2, zinvs[i].v);
        fq_mul(zi3, zi2, zinvs[i].v);
        fq_mul(out[i].x, points[i].X, zi2);
        fq_mul(out[i].y, points[i].Y, zi3);

        unsigned int is_ident = 1u - (unsigned int)fq_isnonzero(points[i].Z);
        fq_cmov(out[i].x, zero, is_ident);
        fq_cmov(out[i].y, zero, is_ident);
    }
}

#endif // RANSHAW_SHAW_BATCH_AFFINE_H
