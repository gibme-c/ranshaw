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

#ifndef RANSHAW_SHAW_MSM_VARTIME_H
#define RANSHAW_SHAW_MSM_VARTIME_H

/**
 * @file shaw_msm_vartime.h
 * @brief Variable-time multi-scalar multiplication for Shaw.
 *
 * Computes Q = s_0*P_0 + s_1*P_1 + ... + s_{n-1}*P_{n-1}.
 * Uses Straus (interleaved) for n <= 32, Pippenger (bucket) for n > 32.
 * Variable-time only: all MSM use cases involve public data.
 */

#include "shaw.h"

#include <cstddef>

#if RANSHAW_SIMD
#include "ranshaw_dispatch.h"
static inline void
    shaw_msm_vartime(shaw_jacobian *result, const unsigned char *scalars, const shaw_jacobian *points, size_t n)
{
    ranshaw_get_dispatch().shaw_msm_vartime(result, scalars, points, n);
}
#elif RANSHAW_PLATFORM_64BIT
void shaw_msm_vartime_x64(shaw_jacobian *result, const unsigned char *scalars, const shaw_jacobian *points, size_t n);
static inline void
    shaw_msm_vartime(shaw_jacobian *result, const unsigned char *scalars, const shaw_jacobian *points, size_t n)
{
    shaw_msm_vartime_x64(result, scalars, points, n);
}
#else
void shaw_msm_vartime_portable(
    shaw_jacobian *result,
    const unsigned char *scalars,
    const shaw_jacobian *points,
    size_t n);
static inline void
    shaw_msm_vartime(shaw_jacobian *result, const unsigned char *scalars, const shaw_jacobian *points, size_t n)
{
    shaw_msm_vartime_portable(result, scalars, points, n);
}
#endif

#endif // RANSHAW_SHAW_MSM_VARTIME_H
