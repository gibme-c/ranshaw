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
 * @file ran_scalar.h
 * @brief Ran scalar operations: arithmetic mod q (the Ran group order / Shaw base field prime).
 */

#ifndef RANSHAW_RAN_SCALAR_H
#define RANSHAW_RAN_SCALAR_H

#include "fq_frombytes.h"
#include "fq_invert.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "fq_tobytes.h"
#include "fq_utils.h"

/*
 * Ran scalar arithmetic.
 *
 * Due to the curve cycle property, Ran scalars live in F_q (the Shaw
 * base field). All operations are thin wrappers around fq_* functions.
 */

static inline void ran_scalar_add(fq_fe out, const fq_fe a, const fq_fe b)
{
    fq_add(out, a, b);
}

static inline void ran_scalar_sub(fq_fe out, const fq_fe a, const fq_fe b)
{
    fq_sub(out, a, b);
}

static inline void ran_scalar_mul(fq_fe out, const fq_fe a, const fq_fe b)
{
    fq_mul(out, a, b);
}

static inline void ran_scalar_neg(fq_fe out, const fq_fe a)
{
    fq_neg(out, a);
}

static inline void ran_scalar_invert(fq_fe out, const fq_fe a)
{
    fq_invert(out, a);
}

static inline void ran_scalar_from_bytes(fq_fe out, const unsigned char b[32])
{
    fq_frombytes(out, b);
}

static inline void ran_scalar_to_bytes(unsigned char b[32], const fq_fe a)
{
    fq_tobytes(b, a);
}

static inline int ran_scalar_is_zero(const fq_fe a)
{
    return !fq_isnonzero(a);
}

static inline void ran_scalar_one(fq_fe out)
{
    fq_1(out);
}

static inline void ran_scalar_zero(fq_fe out)
{
    fq_0(out);
}

/*
 * Reduce a 64-byte wide value mod q (for Fiat-Shamir challenge derivation).
 *
 * Splits 64 bytes into lo[32] and hi[32], then computes:
 *   out = lo + hi * 2^256 (mod q)
 *
 * Since q = 2^255 - gamma, we have 2^256 mod q = 2*gamma.
 *
 * Note: fq_frombytes strips bit 255 (used for y-parity in point encoding).
 * For wide reduction, bit 255 of each half carries value, so we add back:
 *   lo_bit255 * (2^255 mod q) = lo_bit255 * gamma
 *   hi_bit255 * (2^511 mod q) = hi_bit255 * gamma * 2*gamma = hi_bit255 * 2*gamma^2
 */
static inline void ran_scalar_reduce_wide(fq_fe out, const unsigned char wide[64])
{
    fq_fe lo, hi, hi_shifted;
    fq_frombytes(lo, wide);
    fq_frombytes(hi, wide + 32);

    /*
     * 2^256 mod q = 2*gamma.
     * TWO_GAMMA_51 / (2*GAMMA_25) are the radix representations of 2*gamma.
     */
#if RANSHAW_PLATFORM_64BIT
    static const fq_fe TWO_TO_256_MOD_Q = {TWO_GAMMA_51[0], TWO_GAMMA_51[1], TWO_GAMMA_51[2], 0, 0};
    static const fq_fe GAMMA_FE = {GAMMA_51[0], GAMMA_51[1], GAMMA_51[2], 0, 0};
#else
    static const fq_fe TWO_TO_256_MOD_Q = {
        2 * GAMMA_25[0], 2 * GAMMA_25[1], 2 * GAMMA_25[2], 2 * GAMMA_25[3], 2 * GAMMA_25[4], 0, 0, 0, 0, 0};
    static const fq_fe GAMMA_FE = {GAMMA_25[0], GAMMA_25[1], GAMMA_25[2], GAMMA_25[3], GAMMA_25[4], 0, 0, 0, 0, 0};
#endif

    fq_mul(hi_shifted, hi, TWO_TO_256_MOD_Q);
    fq_add(out, lo, hi_shifted);

    /* Correct for bit 255 stripped by fq_frombytes from each half. */
    uint64_t lo_b = (wide[31] >> 7) & 1;
    uint64_t hi_b = (wide[63] >> 7) & 1;

    if (lo_b)
    {
        /* Add gamma (= 2^255 mod q) */
        fq_add(out, out, GAMMA_FE);
    }

    if (hi_b)
    {
        /* Add 2*gamma^2 (= 2^511 mod q = gamma * 2*gamma mod q) */
        fq_fe gamma_sq;
        fq_mul(gamma_sq, GAMMA_FE, TWO_TO_256_MOD_Q);
        fq_add(out, out, gamma_sq);
    }
}

/*
 * Fused multiply-add: out = a * b + c (mod q).
 * Used in Bulletproofs inner-product argument and Fiat-Shamir challenges.
 */
static inline void ran_scalar_muladd(fq_fe out, const fq_fe a, const fq_fe b, const fq_fe c)
{
    fq_fe tmp;
    fq_mul(tmp, a, b);
    fq_add(out, tmp, c);
}

/*
 * Scalar squaring: out = a^2 (mod q).
 */
static inline void ran_scalar_sq(fq_fe out, const fq_fe a)
{
    fq_sq(out, a);
}

#endif // RANSHAW_RAN_SCALAR_H
