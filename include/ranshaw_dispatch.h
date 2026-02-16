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
 * @file ranshaw_dispatch.h
 * @brief Runtime dispatch table for SIMD-accelerated curve operations.
 *
 * Manages a 6-slot function pointer table: {ran,shaw} x {scalarmult, scalarmult_vartime,
 * msm_vartime}. ranshaw_init() populates slots based on CPUID (IFMA > AVX2 > x64 baseline).
 * ranshaw_autotune() benchmarks all available backends and picks the fastest per-slot.
 * On non-SIMD platforms, init/autotune are no-ops (only one backend exists).
 */

#ifndef RANSHAW_DISPATCH_H
#define RANSHAW_DISPATCH_H

#include "ranshaw_platform.h"

#if RANSHAW_SIMD

#include "ran.h"
#include "shaw.h"

#include <cstddef>

struct ranshaw_dispatch_table
{
    void (*ran_scalarmult)(ran_jacobian *, const unsigned char[32], const ran_jacobian *);
    void (*ran_scalarmult_vartime)(ran_jacobian *, const unsigned char[32], const ran_jacobian *);
    void (*ran_msm_vartime)(ran_jacobian *, const unsigned char *, const ran_jacobian *, size_t);
    void (*shaw_scalarmult)(shaw_jacobian *, const unsigned char[32], const shaw_jacobian *);
    void (*shaw_scalarmult_vartime)(shaw_jacobian *, const unsigned char[32], const shaw_jacobian *);
    void (*shaw_msm_vartime)(shaw_jacobian *, const unsigned char *, const shaw_jacobian *, size_t);
};

const ranshaw_dispatch_table &ranshaw_get_dispatch();

void ranshaw_init(void);

void ranshaw_autotune(void);

#else

static inline void ranshaw_init(void) {}
static inline void ranshaw_autotune(void) {}

#endif // RANSHAW_SIMD

#endif // RANSHAW_DISPATCH_H
