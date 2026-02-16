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

#ifndef RANSHAW_SHAW_MAP_TO_CURVE_H
#define RANSHAW_SHAW_MAP_TO_CURVE_H

/**
 * @file shaw_map_to_curve.h
 * @brief Simplified SWU map-to-curve for Shaw (RFC 9380 section 6.6.2).
 *
 * Maps a field element u (as 32-byte LE) to a point on Shaw.
 * The two-input variant maps u0, u1 independently and adds the results.
 */

#include "shaw.h"

#if RANSHAW_PLATFORM_64BIT
void shaw_map_to_curve_x64(shaw_jacobian *r, const unsigned char u[32]);
void shaw_map_to_curve2_x64(shaw_jacobian *r, const unsigned char u0[32], const unsigned char u1[32]);
static inline void shaw_map_to_curve(shaw_jacobian *r, const unsigned char u[32])
{
    shaw_map_to_curve_x64(r, u);
}
static inline void shaw_map_to_curve2(shaw_jacobian *r, const unsigned char u0[32], const unsigned char u1[32])
{
    shaw_map_to_curve2_x64(r, u0, u1);
}
#else
void shaw_map_to_curve_portable(shaw_jacobian *r, const unsigned char u[32]);
void shaw_map_to_curve2_portable(shaw_jacobian *r, const unsigned char u0[32], const unsigned char u1[32]);
static inline void shaw_map_to_curve(shaw_jacobian *r, const unsigned char u[32])
{
    shaw_map_to_curve_portable(r, u);
}
static inline void shaw_map_to_curve2(shaw_jacobian *r, const unsigned char u0[32], const unsigned char u1[32])
{
    shaw_map_to_curve2_portable(r, u0, u1);
}
#endif

#endif // RANSHAW_SHAW_MAP_TO_CURVE_H
