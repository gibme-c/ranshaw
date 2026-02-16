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

#ifndef RANSHAW_SHAW_PEDERSEN_H
#define RANSHAW_SHAW_PEDERSEN_H

/**
 * @file shaw_pedersen.h
 * @brief Pedersen vector commitment for Shaw.
 *
 * Computes C = r*H + sum(a_i * G_i) using a single MSM call with n+1 pairs.
 */

#include "ranshaw_secure_erase.h"
#include "shaw.h"
#include "shaw_msm_vartime.h"
#include "shaw_ops.h"

#include <cstring>
#include <vector>

/**
 * Compute a Pedersen vector commitment: C = blinding*H + sum(values[i]*generators[i]).
 *
 * @param result     Output: the commitment point (Jacobian)
 * @param blinding   32-byte scalar (blinding factor r)
 * @param H          Blinding generator point (Jacobian)
 * @param values     Array of n 32-byte scalars
 * @param generators Array of n generator points (Jacobian)
 * @param n          Number of value/generator pairs
 */
static inline void shaw_pedersen_commit(
    shaw_jacobian *result,
    const unsigned char *blinding,
    const shaw_jacobian *H,
    const unsigned char *values,
    const shaw_jacobian *generators,
    size_t n)
{
    /* Guard against overflow in 32 * (n + 1) */
    if (n > SIZE_MAX / 32 - 1)
    {
        shaw_identity(result);
        return;
    }

    /* Build combined arrays: [blinding, values[0..n-1]] and [H, generators[0..n-1]] */
    std::vector<unsigned char> all_scalars(32 * (n + 1));
    std::vector<shaw_jacobian> all_points(n + 1);

    std::memcpy(all_scalars.data(), blinding, 32);
    shaw_copy(&all_points[0], H);

    if (n > 0)
    {
        std::memcpy(all_scalars.data() + 32, values, 32 * n);
        for (size_t i = 0; i < n; i++)
            shaw_copy(&all_points[i + 1], &generators[i]);
    }

    shaw_msm_vartime(result, all_scalars.data(), all_points.data(), n + 1);
    ranshaw_secure_erase(all_scalars.data(), all_scalars.size());
}

#endif // RANSHAW_SHAW_PEDERSEN_H
