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
 * @file ran_constants.h
 * @brief Curve constants for Ran: generator (Gx, Gy), curve parameter b, and helper constants.
 */

#ifndef RANSHAW_RAN_CONSTANTS_H
#define RANSHAW_RAN_CONSTANTS_H

#include "fp.h"

/*
 * Ran curve: y^2 = x^3 - 3x + b over F_p (p = 2^255 - 19)
 *
 * b = 29548680719914169098707364166668229174524292605831040732610367861908860687214
 */
#if RANSHAW_PLATFORM_64BIT
static const fp_fe RAN_B =
    {0x5AE9F1D1D36EULL, 0x70FAC0784C55EULL, 0x64B9E6AAE3D99ULL, 0x5B07FDB885266ULL, 0x4153F5EAB5C57ULL};
static const fp_fe RAN_GX =
    {0x5D3F5ABD6F63BULL, 0x165E13B3F711FULL, 0x1088D71C48144ULL, 0x1A1D9FE8F33BDULL, 0x44CD9962FA942ULL};
static const fp_fe RAN_GY =
    {0x544CE79A2C13FULL, 0x500264E0125DEULL, 0x34BB9E4E31B90ULL, 0x54E100CE840E7ULL, 0xBEAB1C489555ULL};
#else
static const fp_fe RAN_B =
    {30528366, 1489532, 59032926, 29616897, 44973465, 26404762, 59265638, 23863286, 44784727, 17125335};
static const fp_fe RAN_GX =
    {64419387, 24444266, 54489375, 5863502, 29655364, 4334428, 42939325, 6846079, 36677954, 18036325};
static const fp_fe RAN_GY =
    {27443519, 22098846, 33629662, 20973971, 14883728, 13823609, 15220967, 22250499, 4756821, 3123911};
#endif

/*
 * Ran group order = q = 2^255 - gamma
 * = 57896044618658097711785492504343953926395325869620403790519050891226336262239
 *
 * Stored as 32 bytes little-endian for scalar operations.
 */
static const unsigned char RAN_ORDER[32] = {0x5f, 0xf8, 0x70, 0xec, 0x45, 0x46, 0x68, 0x71, 0xa7, 0x0f, 0x73,
                                            0x39, 0x0e, 0xeb, 0xb1, 0x4b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};

#endif // RANSHAW_RAN_CONSTANTS_H
