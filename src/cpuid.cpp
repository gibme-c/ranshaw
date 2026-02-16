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

#include "ranshaw_cpuid.h"

#if RANSHAW_PLATFORM_X64

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

// XGETBV via inline assembly for GCC/Clang (the intrinsic requires -mxsave target)
#if !defined(_MSC_VER)
static inline uint64_t ranshaw_xgetbv(uint32_t index)
{
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return (static_cast<uint64_t>(edx) << 32) | eax;
}
#endif

static uint32_t ranshaw_detect_cpu_features()
{
    uint32_t flags = 0;

#if defined(_MSC_VER)
    int regs[4]; // EAX, EBX, ECX, EDX

    // CPUID leaf 1: check OSXSAVE (ECX bit 27)
    __cpuid(regs, 1);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;

    if (!osxsave)
    {
        return 0;
    }

    // XGETBV(0): check OS has enabled XMM (bit 1) and YMM (bit 2) state saving
    const uint64_t xcr0 = _xgetbv(0);
    const bool ymm_enabled = (xcr0 & 0x06) == 0x06; // bits 1 and 2

    if (!ymm_enabled)
    {
        return 0;
    }

    // CPUID leaf 7, subleaf 0: extended feature flags in EBX
    __cpuidex(regs, 7, 0);
    const uint32_t ebx7 = static_cast<uint32_t>(regs[1]);

    // AVX2: EBX bit 5
    if (ebx7 & (1 << 5))
    {
        flags |= RANSHAW_CPU_AVX2;
    }

    // AVX-512 requires OS support for OPMASK (bit 5), ZMM_Hi256 (bit 6), Hi16_ZMM (bit 7)
    const bool zmm_enabled = (xcr0 & 0xE0) == 0xE0; // bits 5, 6, 7

    if (zmm_enabled)
    {
        // AVX-512F: EBX bit 16
        if (ebx7 & (1 << 16))
        {
            flags |= RANSHAW_CPU_AVX512F;
        }

        // AVX-512 IFMA: EBX bit 21
        if (ebx7 & (1 << 21))
        {
            flags |= RANSHAW_CPU_AVX512IFMA;
        }
    }

#else // GCC / Clang
    unsigned int eax, ebx, ecx, edx;

    // CPUID leaf 1: check OSXSAVE (ECX bit 27)
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
    {
        return 0;
    }

    const bool osxsave = (ecx & (1 << 27)) != 0;

    if (!osxsave)
    {
        return 0;
    }

    // XGETBV(0): check OS has enabled XMM (bit 1) and YMM (bit 2) state saving
    const uint64_t xcr0 = ranshaw_xgetbv(0);
    const bool ymm_enabled = (xcr0 & 0x06) == 0x06;

    if (!ymm_enabled)
    {
        return 0;
    }

    // CPUID leaf 7, subleaf 0: extended feature flags in EBX
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
    {
        return 0;
    }

    // AVX2: EBX bit 5
    if (ebx & (1 << 5))
    {
        flags |= RANSHAW_CPU_AVX2;
    }

    // AVX-512 requires OS support for OPMASK (bit 5), ZMM_Hi256 (bit 6), Hi16_ZMM (bit 7)
    const bool zmm_enabled = (xcr0 & 0xE0) == 0xE0;

    if (zmm_enabled)
    {
        // AVX-512F: EBX bit 16
        if (ebx & (1 << 16))
        {
            flags |= RANSHAW_CPU_AVX512F;
        }

        // AVX-512 IFMA: EBX bit 21
        if (ebx & (1 << 21))
        {
            flags |= RANSHAW_CPU_AVX512IFMA;
        }
    }

#endif

    return flags;
}

uint32_t ranshaw_cpu_features()
{
    static const uint32_t cached = ranshaw_detect_cpu_features();

    return cached;
}

#endif // RANSHAW_PLATFORM_X64
