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
 * b = 15789920373731020205926570676277057129217619222203920395806844808978996083412
 */
#if RANSHAW_PLATFORM_64BIT
static const fp_fe RAN_B =
    {0x49ee1edd73ad4ULL, 0x7082277e6a456ULL, 0x2edecec10fdbcULL, 0x5c5f4a53b59fULL, 0x22e8c739b0ea7ULL};
static const fp_fe RAN_GX = {0x3ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL};
static const fp_fe RAN_GY =
    {0x3e639e3183ef4ULL, 0x3b8b0d4bb9a48ULL, 0x817c1d6400efULL, 0x10e5ec93341a8ULL, 0x537b74d97ac07ULL};
#else
static const fp_fe RAN_B =
    {30882516, 19380347, 65446998, 29493405, 1113532, 12286779, 39040415, 1513426, 60493479, 9151260};
static const fp_fe RAN_GX = {3, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static const fp_fe RAN_GY =
    {51920628, 16354936, 12294728, 15608885, 23331055, 2121479, 20136360, 4429746, 26717191, 21884371};
#endif

/*
 * Ran group order = q = 2^255 - gamma
 * = 57896044618658097711785492504343953926549254372227246365156541811699034343327
 *
 * Stored as 32 bytes little-endian for scalar operations.
 */
static const unsigned char RAN_ORDER[32] = {0x9f, 0xc7, 0x27, 0x79, 0x72, 0xd2, 0xb6, 0x6e, 0x58, 0x6b, 0x65,
                                               0xb7, 0x2c, 0x78, 0x7f, 0xbf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                               0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};

#endif // RANSHAW_RAN_CONSTANTS_H
