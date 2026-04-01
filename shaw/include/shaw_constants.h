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
 * @file shaw_constants.h
 * @brief Curve constants for Shaw: generator (Gx, Gy), curve parameter b, and helper constants.
 */

#ifndef RANSHAW_SHAW_CONSTANTS_H
#define RANSHAW_SHAW_CONSTANTS_H

#include "fq.h"

/*
 * Shaw curve: y^2 = x^3 - 3x + b over F_q (q = 2^255 - gamma)
 *
 * b = 45242492826800543065362849020183379778794895326127509704283577990932728169905
 */
#if RANSHAW_PLATFORM_64BIT
static const fq_fe SHAW_B =
    {0x3C002DD34C5B1ULL, 0x74516598A55ACULL, 0x22B764C970DBBULL, 0x9A0F7A5E3F44ULL, 0x640657EEA7EFBULL};
static const fq_fe SHAW_GX =
    {0x265FA80B76BF4ULL, 0x5EF981E4A3826ULL, 0x2FF1D29A49025ULL, 0x1F78363857FA9ULL, 0x3A8B78295E7C3ULL};
static const fq_fe SHAW_GY =
    {0x3532F445198CBULL, 0x335E9F7FBB724ULL, 0x23B2BEC1DD961ULL, 0x5BA0F2034FE76ULL, 0x47D87BC8873BULL};
#else
static const fq_fe SHAW_B =
    {20235697, 15728823, 25843116, 30492054, 9899451, 9100691, 39731012, 2524126, 48922363, 26220895};
static const fq_fe SHAW_GX =
    {12020724, 10059424, 38418470, 24897031, 27562021, 12568394, 59080617, 8249560, 43378627, 15347168};
static const fq_fe SHAW_GY =
    {5347531, 13945809, 66828068, 13466237, 1956193, 9358075, 3473014, 24019912, 63473467, 1177118};
#endif

/*
 * Shaw group order = p = 2^255 - 19
 * = 57896044618658097711785492504343953926634992332820282019728792003956564819949
 *
 * Stored as 32 bytes little-endian for scalar operations.
 */
static const unsigned char SHAW_ORDER[32] = {0xed, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                             0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                             0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};

#endif // RANSHAW_SHAW_CONSTANTS_H
