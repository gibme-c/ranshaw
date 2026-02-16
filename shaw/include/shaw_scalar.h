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
 * @file shaw_scalar.h
 * @brief Shaw scalar operations: arithmetic mod p (the Shaw group order / Ran base field prime, p = 2^255-19).
 */

#ifndef RANSHAW_SHAW_SCALAR_H
#define RANSHAW_SHAW_SCALAR_H

#include "fp_frombytes.h"
#include "fp_invert.h"
#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "fp_tobytes.h"
#include "fp_utils.h"

/*
 * Shaw scalar arithmetic.
 *
 * Due to the curve cycle property, Shaw scalars live in F_p (the Ran
 * base field). All operations are thin wrappers around fp_* functions.
 */

static inline void shaw_scalar_add(fp_fe out, const fp_fe a, const fp_fe b)
{
    fp_add(out, a, b);
}

static inline void shaw_scalar_sub(fp_fe out, const fp_fe a, const fp_fe b)
{
    fp_sub(out, a, b);
}

static inline void shaw_scalar_mul(fp_fe out, const fp_fe a, const fp_fe b)
{
    fp_mul(out, a, b);
}

static inline void shaw_scalar_neg(fp_fe out, const fp_fe a)
{
    fp_neg(out, a);
}

static inline void shaw_scalar_invert(fp_fe out, const fp_fe a)
{
    fp_invert(out, a);
}

static inline void shaw_scalar_from_bytes(fp_fe out, const unsigned char b[32])
{
    fp_frombytes(out, b);
}

static inline void shaw_scalar_to_bytes(unsigned char b[32], const fp_fe a)
{
    fp_tobytes(b, a);
}

static inline int shaw_scalar_is_zero(const fp_fe a)
{
    return !fp_isnonzero(a);
}

static inline void shaw_scalar_one(fp_fe out)
{
    fp_1(out);
}

static inline void shaw_scalar_zero(fp_fe out)
{
    fp_0(out);
}

/*
 * Reduce a 64-byte wide value mod p (for Fiat-Shamir challenge derivation).
 *
 * Splits 64 bytes into lo[32] and hi[32], then computes:
 *   out = lo + hi * 2^256 (mod p)
 *
 * Since p = 2^255 - 19, we have 2^256 mod p = 2*19 = 38.
 *
 * Note: fp_frombytes strips bit 255 (used for y-parity in point encoding).
 * For wide reduction, bit 255 of each half carries value, so we add back:
 *   lo_bit255 * (2^255 mod p) = lo_bit255 * 19
 *   hi_bit255 * (2^511 mod p) = hi_bit255 * 19 * 38 = hi_bit255 * 722
 */
static inline void shaw_scalar_reduce_wide(fp_fe out, const unsigned char wide[64])
{
    fp_fe lo, hi, hi_shifted;
    fp_frombytes(lo, wide);
    fp_frombytes(hi, wide + 32);

#if RANSHAW_PLATFORM_64BIT
    static const fp_fe TWO_TO_256_MOD_P = {38, 0, 0, 0, 0};
#else
    static const fp_fe TWO_TO_256_MOD_P = {38, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif

    fp_mul(hi_shifted, hi, TWO_TO_256_MOD_P);
    fp_add(out, lo, hi_shifted);

    /* Correct for bit 255 stripped by fp_frombytes from each half. */
    unsigned int lo_b = (wide[31] >> 7) & 1u;
    unsigned int hi_b = (wide[63] >> 7) & 1u;
    unsigned int corr = lo_b * 19u + hi_b * 722u;

    if (corr)
    {
        unsigned char corr_bytes[32] = {0};
        corr_bytes[0] = static_cast<unsigned char>(corr & 0xffu);
        corr_bytes[1] = static_cast<unsigned char>((corr >> 8) & 0xffu);
        fp_fe correction;
        fp_frombytes(correction, corr_bytes);
        fp_add(out, out, correction);
    }
}

/*
 * Fused multiply-add: out = a * b + c (mod p).
 * Used in Bulletproofs inner-product argument and Fiat-Shamir challenges.
 */
static inline void shaw_scalar_muladd(fp_fe out, const fp_fe a, const fp_fe b, const fp_fe c)
{
    fp_fe tmp;
    fp_mul(tmp, a, b);
    fp_add(out, tmp, c);
}

/*
 * Scalar squaring: out = a^2 (mod p).
 */
static inline void shaw_scalar_sq(fp_fe out, const fp_fe a)
{
    fp_sq(out, a);
}

#endif // RANSHAW_SHAW_SCALAR_H
