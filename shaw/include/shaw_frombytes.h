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
 * @file shaw_frombytes.h
 * @brief Compressed point deserialization for Shaw with on-curve validation.
 *
 * Rejects off-curve points (weak twist security is ~99 bits for Shaw).
 */

#ifndef RANSHAW_SHAW_FROMBYTES_H
#define RANSHAW_SHAW_FROMBYTES_H

#include "shaw.h"

#if RANSHAW_PLATFORM_64BIT
int shaw_frombytes_x64(shaw_jacobian *r, const unsigned char s[32]);
static inline int shaw_frombytes(shaw_jacobian *r, const unsigned char s[32])
{
    return shaw_frombytes_x64(r, s);
}
#else
int shaw_frombytes_portable(shaw_jacobian *r, const unsigned char s[32]);
static inline int shaw_frombytes(shaw_jacobian *r, const unsigned char s[32])
{
    return shaw_frombytes_portable(r, s);
}
#endif

#endif // RANSHAW_SHAW_FROMBYTES_H
