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
 * @file ranshaw_wei25519.h
 * @brief Wei25519 bridge: ingest a Weierstrass-form x-coordinate as an F_p element.
 *
 * The caller's ed25519 library handles the Ed25519 -> Wei25519 coordinate transform.
 * This function validates the raw 32-byte x-coordinate as a canonical F_p element
 * (which is simultaneously a Shaw scalar, due to the cycle property).
 */

#ifndef RANSHAW_WEI25519_H
#define RANSHAW_WEI25519_H

#include "fp.h"
#include "fp_frombytes.h"
#include "fp_tobytes.h"

/*
 * Wei25519 bridge: accept a raw 32-byte x-coordinate and validate it
 * as an F_p element. The caller's ed25519 library handles the
 * Ed25519 -> Wei25519 coordinate transform externally.
 *
 * This function just ingests the raw x-coordinate bytes as an F_p
 * field element (which is also a Shaw scalar).
 *
 * Returns 0 on success, -1 if x >= p (non-canonical).
 */
static inline int ranshaw_wei25519_to_fp(fp_fe out, const unsigned char x_bytes[32])
{
    /* Check bit 255 is clear (any valid field element has bit 255 = 0) */
    if (x_bytes[31] & 0x80)
        return -1;

    /* Deserialize */
    fp_frombytes(out, x_bytes);

    /* Reject non-canonical: re-serialize and compare */
    unsigned char check[32];
    fp_tobytes(check, out);

    unsigned char diff = 0;
    for (int i = 0; i < 32; i++)
        diff |= check[i] ^ x_bytes[i];

    if (diff != 0)
        return -1;

    return 0;
}

#endif // RANSHAW_WEI25519_H
