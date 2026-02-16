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
 * @file ranshaw_ct_barrier.h
 * @brief Constant-time barriers to prevent compiler optimization of secret-dependent operations.
 *
 * On GCC/Clang, uses inline asm to make the value opaque to the optimizer.
 * On MSVC, uses a volatile round-trip. Both prevent the compiler from deducing
 * that a conditional move or XOR-blend can be replaced by a branch.
 */

#ifndef RANSHAW_CT_BARRIER_H
#define RANSHAW_CT_BARRIER_H

#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)

static inline uint32_t ranshaw_ct_barrier_u32(uint32_t x)
{
    __asm__ __volatile__("" : "+r"(x));
    return x;
}

static inline uint64_t ranshaw_ct_barrier_u64(uint64_t x)
{
    __asm__ __volatile__("" : "+r"(x));
    return x;
}

#else

static inline uint32_t ranshaw_ct_barrier_u32(uint32_t x)
{
    volatile uint32_t v = x;
    return v;
}

static inline uint64_t ranshaw_ct_barrier_u64(uint64_t x)
{
    volatile uint64_t v = x;
    return v;
}

#endif

#endif // RANSHAW_CT_BARRIER_H
