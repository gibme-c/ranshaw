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

#ifndef RANSHAW_FQ_BATCH_INVERT_H
#define RANSHAW_FQ_BATCH_INVERT_H

/**
 * @file fq_batch_invert.h
 * @brief Batch field inversion for F_q using Montgomery's trick.
 *
 * Inverts n field elements using 1 inversion + 3(n-1) multiplications,
 * instead of n separate inversions. Zero elements are mapped to zero.
 */

#include "fq_invert.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_utils.h"

#include <vector>

/**
 * Batch-invert n F_q elements using Montgomery's trick.
 *
 * For each in[i], writes in[i]^{-1} to out[i].
 * Zero elements produce zero output (not undefined).
 * out and in may alias (in-place inversion is supported).
 *
 * SECURITY NOTE: The isnonzero() branches are intentionally variable-time.
 * This function operates on public geometric data (affine coordinates for
 * batch affine conversion), not secret scalars or secret-derived values.
 * Timing side-channels on public data are not exploitable.
 *
 * @param out  Output array of n inverted elements
 * @param in   Input array of n elements
 * @param n    Number of elements
 */
static inline void fq_batch_invert(fq_fe *out, const fq_fe *in, size_t n)
{
    if (n == 0)
        return;
    if (n == 1)
    {
        if (fq_isnonzero(in[0]))
            fq_invert(out[0], in[0]);
        else
            fq_0(out[0]);
        return;
    }

    /* Forward pass: cumulative products */
    struct fq_fe_s
    {
        fq_fe v;
    };
    std::vector<fq_fe_s> acc(n);
    fq_copy(acc[0].v, in[0]);
    for (size_t i = 1; i < n; i++)
    {
        if (fq_isnonzero(in[i]))
            fq_mul(acc[i].v, acc[i - 1].v, in[i]);
        else
            fq_copy(acc[i].v, acc[i - 1].v);
    }

    /* Single inversion */
    fq_fe inv;
    fq_invert(inv, acc[n - 1].v);

    /* Backward pass: recover individual inverses */
    for (size_t i = n - 1; i > 0; i--)
    {
        if (!fq_isnonzero(in[i]))
        {
            fq_0(out[i]);
        }
        else
        {
            fq_fe tmp;
            fq_copy(tmp, in[i]); /* save before out[i] overwrites (aliasing) */
            fq_mul(out[i], inv, acc[i - 1].v);
            fq_mul(inv, inv, tmp);
        }
    }

    /* First element */
    if (fq_isnonzero(in[0]))
        fq_copy(out[0], inv);
    else
        fq_0(out[0]);
}

#endif // RANSHAW_FQ_BATCH_INVERT_H
