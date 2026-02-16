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
 * @file ranshaw_cpuid.h
 * @brief Runtime CPU feature detection for SIMD backend selection.
 *
 * Queries CPUID on x86-64 for AVX2, AVX-512F, and AVX-512 IFMA support.
 * On non-x86 platforms, all feature queries return false (baseline backend only).
 */

#ifndef RANSHAW_CPUID_H
#define RANSHAW_CPUID_H

#include "ranshaw_platform.h"

#include <cstdint>

enum ranshaw_cpu_flag : uint32_t
{
    RANSHAW_CPU_AVX2 = 1 << 0,
    RANSHAW_CPU_AVX512F = 1 << 1,
    RANSHAW_CPU_AVX512IFMA = 1 << 2,
};

#if RANSHAW_PLATFORM_X64

uint32_t ranshaw_cpu_features();

static inline bool ranshaw_has_avx2()
{
    return (ranshaw_cpu_features() & RANSHAW_CPU_AVX2) != 0;
}

static inline bool ranshaw_has_avx512f()
{
    return (ranshaw_cpu_features() & RANSHAW_CPU_AVX512F) != 0;
}

static inline bool ranshaw_has_avx512ifma()
{
    return (ranshaw_cpu_features() & RANSHAW_CPU_AVX512IFMA) != 0;
}

#else

static inline uint32_t ranshaw_cpu_features()
{
    return 0;
}

static inline bool ranshaw_has_avx2()
{
    return false;
}

static inline bool ranshaw_has_avx512f()
{
    return false;
}

static inline bool ranshaw_has_avx512ifma()
{
    return false;
}

#endif // RANSHAW_PLATFORM_X64

#endif // RANSHAW_CPUID_H
