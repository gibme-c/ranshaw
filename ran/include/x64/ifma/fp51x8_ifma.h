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
 * @file fp51x8_ifma.h
 * @brief 8-way parallel radix-2^51 Fp field element operations using AVX-512 IFMA.
 *
 * This is the field arithmetic layer for the 8-way batch scalarmult operations.
 * Each fp51x8 holds 8 independent field elements packed horizontally into
 * AVX-512 registers -- one element per 64-bit lane, 5 registers per fp51x8
 * (one per radix-2^51 limb). The representation is the same as the scalar fp_fe
 * on x64, just 8 copies side by side.
 *
 * Multiplication uses vpmadd52lo/vpmadd52hi (AVX-512 IFMA instructions) for
 * hardware 52-bit fused multiply-accumulate. A 5x5 schoolbook product gives
 * 25 IFMA pairs into 9 lo/hi accumulator limbs. The IFMA split point is at
 * bit 52, but our radix is 2^51, so recombination shifts the hi values left
 * by 1 (2^52 / 2^51 = 2) before adding to the next limb. The upper 5 limbs
 * are then folded back with the factor 19 (since 2^255 = 19 mod p), and a
 * carry chain brings everything down to <=51 bits per limb.
 *
 * The x19 reduction uses shift-and-add (19x = 16x + 2x + x) rather than
 * _mm512_mullo_epi64, which requires AVX-512DQ and isn't in the IFMA compile
 * flags. This avoids adding a target feature dependency for one operation.
 *
 * All IFMA inputs must have limbs <=52 bits -- the instructions silently
 * truncate anything above that. Mul/sq outputs are carry-propagated to <=51
 * bits. Addition doesn't carry (output <=52 bits for two <=51-bit inputs),
 * and subtraction uses a 4p bias with carry propagation.
 *
 * Register budget is tight: fp51x8_mul needs f[5] + g[5] + lo[9] + hi[9] = 28
 * of the 32 available ZMM registers. Everything is force-inlined so the
 * compiler can schedule across the full register file.
 */

#ifndef RANSHAW_X64_IFMA_FP51X8_IFMA_H
#define RANSHAW_X64_IFMA_FP51X8_IFMA_H

#include "fp_ops.h"
#include "ranshaw_platform.h"
#include "x64/fp51.h"

#include <immintrin.h>

#if defined(_MSC_VER)
#define FP51X8_FORCE_INLINE __forceinline
#else
#define FP51X8_FORCE_INLINE inline __attribute__((always_inline))
#endif

/**
 * @brief 8-way parallel Fp field element type: 5 __m512i registers.
 *
 * v[i] holds limb i of 8 independent field elements in the 8 x 64-bit lanes.
 * All limbs are unsigned, radix-2^51, <=51 bits after carry propagation.
 */
typedef struct
{
    __m512i v[5];
} fp51x8;

static inline __m512i fp51x8_mask51(void)
{
    return _mm512_set1_epi64((long long)((1ULL << 51) - 1));
}

#define FP51X8_MASK51 fp51x8_mask51()

// Multiply a vector by 19 via shift-and-add: 19x = 16x + 2x + x.
// We avoid _mm512_mullo_epi64 because it needs AVX-512DQ, which isn't in our
// compile flags (-mavx512f -mavx512ifma). The shift-and-add is 3 ops instead
// of 1, but it's only used in the carry-wrap path, not the hot inner products.
#define FP51X8_MUL19(x) _mm512_add_epi64(_mm512_add_epi64(_mm512_slli_epi64((x), 4), _mm512_slli_epi64((x), 1)), (x))

// -- Trivial operations (zero, one, copy) --

static FP51X8_FORCE_INLINE void fp51x8_0(fp51x8 *h)
{
    const __m512i z = _mm512_setzero_si512();
    for (int i = 0; i < 5; i++)
        h->v[i] = z;
}

static FP51X8_FORCE_INLINE void fp51x8_1(fp51x8 *h)
{
    const __m512i z = _mm512_setzero_si512();
    h->v[0] = _mm512_set1_epi64(1);
    for (int i = 1; i < 5; i++)
        h->v[i] = z;
}

static FP51X8_FORCE_INLINE void fp51x8_copy(fp51x8 *h, const fp51x8 *f)
{
    for (int i = 0; i < 5; i++)
        h->v[i] = f->v[i];
}

// -- Addition (no carry propagation) --
// For two <=51-bit inputs, the output is at most 52 bits -- still within
// IFMA's input window. No carry needed.

static FP51X8_FORCE_INLINE void fp51x8_add(fp51x8 *h, const fp51x8 *f, const fp51x8 *g)
{
    for (int i = 0; i < 5; i++)
        h->v[i] = _mm512_add_epi64(f->v[i], g->v[i]);
}

// -- Carry propagation --
// Standard radix-2^51 carry chain: shift right 51, mask, add to next limb.
// Limb 4 wraps back to limb 0 with x19 (mod 2^255 - 19). Two passes on
// limb 0->1 to absorb the final wrap carry.

static FP51X8_FORCE_INLINE void fp51x8_carry(fp51x8 *h)
{
    const __m512i mask = FP51X8_MASK51;
    __m512i c;

    c = _mm512_srli_epi64(h->v[0], 51);
    h->v[1] = _mm512_add_epi64(h->v[1], c);
    h->v[0] = _mm512_and_si512(h->v[0], mask);

    c = _mm512_srli_epi64(h->v[1], 51);
    h->v[2] = _mm512_add_epi64(h->v[2], c);
    h->v[1] = _mm512_and_si512(h->v[1], mask);

    c = _mm512_srli_epi64(h->v[2], 51);
    h->v[3] = _mm512_add_epi64(h->v[3], c);
    h->v[2] = _mm512_and_si512(h->v[2], mask);

    c = _mm512_srli_epi64(h->v[3], 51);
    h->v[4] = _mm512_add_epi64(h->v[4], c);
    h->v[3] = _mm512_and_si512(h->v[3], mask);

    c = _mm512_srli_epi64(h->v[4], 51);
    h->v[0] = _mm512_add_epi64(h->v[0], FP51X8_MUL19(c));
    h->v[4] = _mm512_and_si512(h->v[4], mask);

    c = _mm512_srli_epi64(h->v[0], 51);
    h->v[1] = _mm512_add_epi64(h->v[1], c);
    h->v[0] = _mm512_and_si512(h->v[0], mask);
}

// -- Subtraction with 4p bias + carry --
// To keep limbs non-negative, we add 4p before subtracting. The bias values
// (0x1FFFFFFFFFFFB4 for limb 0, 0x1FFFFFFFFFFFFC for limbs 1-4) are the
// same as the scalar fp_sub. The carry chain then normalizes back to <=51-bit
// limbs. Output limb 0 can be up to 52 bits (51 + carry*19), but that's still
// within IFMA's 52-bit input window.

static FP51X8_FORCE_INLINE void fp51x8_sub(fp51x8 *h, const fp51x8 *f, const fp51x8 *g)
{
    // 4p bias values (same as scalar fp_sub):
    // limb 0: 4 * (2^51 - 19) = 0x1FFFFFFFFFFFB4
    // limbs 1-4: 4 * (2^51 - 1) = 0x1FFFFFFFFFFFFC
    const __m512i bias0 = _mm512_set1_epi64(0x1FFFFFFFFFFFB4LL);
    const __m512i bias1 = _mm512_set1_epi64(0x1FFFFFFFFFFFFCLL);

    h->v[0] = _mm512_add_epi64(_mm512_sub_epi64(f->v[0], g->v[0]), bias0);
    h->v[1] = _mm512_add_epi64(_mm512_sub_epi64(f->v[1], g->v[1]), bias1);
    h->v[2] = _mm512_add_epi64(_mm512_sub_epi64(f->v[2], g->v[2]), bias1);
    h->v[3] = _mm512_add_epi64(_mm512_sub_epi64(f->v[3], g->v[3]), bias1);
    h->v[4] = _mm512_add_epi64(_mm512_sub_epi64(f->v[4], g->v[4]), bias1);

    fp51x8_carry(h);
}

// -- Negation --

static FP51X8_FORCE_INLINE void fp51x8_neg(fp51x8 *h, const fp51x8 *f)
{
    fp51x8 zero;
    fp51x8_0(&zero);
    fp51x8_sub(h, &zero, f);
}

// -- Weak normalization --
// Same as fp51x8_carry, used to fix limbs that exceed 52 bits after a
// problematic addition (e.g. adding a <=52-bit value to a <=51-bit value
// can reach 53 bits). Only needed at specific points in batch point
// addition/subtraction -- see batch IFMA headers for placement details.

static FP51X8_FORCE_INLINE void fp51x8_normalize_weak(fp51x8 *h)
{
    const __m512i mask = FP51X8_MASK51;
    __m512i c;

    c = _mm512_srli_epi64(h->v[0], 51);
    h->v[1] = _mm512_add_epi64(h->v[1], c);
    h->v[0] = _mm512_and_si512(h->v[0], mask);

    c = _mm512_srli_epi64(h->v[1], 51);
    h->v[2] = _mm512_add_epi64(h->v[2], c);
    h->v[1] = _mm512_and_si512(h->v[1], mask);

    c = _mm512_srli_epi64(h->v[2], 51);
    h->v[3] = _mm512_add_epi64(h->v[3], c);
    h->v[2] = _mm512_and_si512(h->v[2], mask);

    c = _mm512_srli_epi64(h->v[3], 51);
    h->v[4] = _mm512_add_epi64(h->v[4], c);
    h->v[3] = _mm512_and_si512(h->v[3], mask);

    c = _mm512_srli_epi64(h->v[4], 51);
    h->v[0] = _mm512_add_epi64(h->v[0], FP51X8_MUL19(c));
    h->v[4] = _mm512_and_si512(h->v[4], mask);

    c = _mm512_srli_epi64(h->v[0], 51);
    h->v[1] = _mm512_add_epi64(h->v[1], c);
    h->v[0] = _mm512_and_si512(h->v[0], mask);
}

// -- Conditional move (k-mask) --
// AVX-512 k-mask blend: for each of the 8 lanes, if the corresponding bit
// in `mask` is set, take the value from `u`; otherwise keep the value in `t`.
// This is the batch equivalent of fp_cmov, used for constant-time table
// selection where each lane independently selects from a different table entry.

static FP51X8_FORCE_INLINE void fp51x8_cmov(fp51x8 *t, const fp51x8 *u, __mmask8 mask)
{
    for (int i = 0; i < 5; i++)
        t->v[i] = _mm512_mask_blend_epi64(mask, t->v[i], u->v[i]);
}

// -- Schoolbook multiplication using IFMA --
// This is the heart of the 8-way backend. Two IFMA instructions per product
// term (lo + hi halves), 25 product terms for a 5x5 schoolbook, so 50 IFMA
// ops total -- all operating on 8 independent multiplications in parallel.

/**
 * @brief 8-way multiplication: h = f * g (mod 2^255-19).
 *
 * Both inputs must have limbs <=52 bits (true for IFMA outputs <=51, fp51x8_add
 * of two <=51-bit values <=52, fp51x8_sub outputs <=52 after bias).
 *
 * Algorithm: 5x5 schoolbook -> 9-limb lo/hi accumulators via IFMA,
 * recombine lo/hi at radix-2^51 boundary, fold upper limbs with x19,
 * carry-propagate.
 */
static FP51X8_FORCE_INLINE void fp51x8_mul(fp51x8 *h, const fp51x8 *f, const fp51x8 *g)
{
    const __m512i zero = _mm512_setzero_si512();

    // 9-limb accumulators for lo (bits 0-51) and hi (bits 52-103)
    __m512i lo0 = zero, lo1 = zero, lo2 = zero, lo3 = zero, lo4 = zero;
    __m512i lo5 = zero, lo6 = zero, lo7 = zero, lo8 = zero;
    __m512i hi0 = zero, hi1 = zero, hi2 = zero, hi3 = zero, hi4 = zero;
    __m512i hi5 = zero, hi6 = zero, hi7 = zero, hi8 = zero;

    // 25 products: f[i] * g[j] -> accumulate into lo[i+j], hi[i+j]
    // Round 0: f[0] * g[0..4]
    lo0 = _mm512_madd52lo_epu64(lo0, f->v[0], g->v[0]);
    hi0 = _mm512_madd52hi_epu64(hi0, f->v[0], g->v[0]);
    lo1 = _mm512_madd52lo_epu64(lo1, f->v[0], g->v[1]);
    hi1 = _mm512_madd52hi_epu64(hi1, f->v[0], g->v[1]);
    lo2 = _mm512_madd52lo_epu64(lo2, f->v[0], g->v[2]);
    hi2 = _mm512_madd52hi_epu64(hi2, f->v[0], g->v[2]);
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[0], g->v[3]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[0], g->v[3]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[0], g->v[4]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[0], g->v[4]);

    // Round 1: f[1] * g[0..4]
    lo1 = _mm512_madd52lo_epu64(lo1, f->v[1], g->v[0]);
    hi1 = _mm512_madd52hi_epu64(hi1, f->v[1], g->v[0]);
    lo2 = _mm512_madd52lo_epu64(lo2, f->v[1], g->v[1]);
    hi2 = _mm512_madd52hi_epu64(hi2, f->v[1], g->v[1]);
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[1], g->v[2]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[1], g->v[2]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[1], g->v[3]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[1], g->v[3]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[1], g->v[4]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[1], g->v[4]);

    // Round 2: f[2] * g[0..4]
    lo2 = _mm512_madd52lo_epu64(lo2, f->v[2], g->v[0]);
    hi2 = _mm512_madd52hi_epu64(hi2, f->v[2], g->v[0]);
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[2], g->v[1]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[2], g->v[1]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[2], g->v[2]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[2], g->v[2]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[2], g->v[3]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[2], g->v[3]);
    lo6 = _mm512_madd52lo_epu64(lo6, f->v[2], g->v[4]);
    hi6 = _mm512_madd52hi_epu64(hi6, f->v[2], g->v[4]);

    // Round 3: f[3] * g[0..4]
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[3], g->v[0]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[3], g->v[0]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[3], g->v[1]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[3], g->v[1]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[3], g->v[2]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[3], g->v[2]);
    lo6 = _mm512_madd52lo_epu64(lo6, f->v[3], g->v[3]);
    hi6 = _mm512_madd52hi_epu64(hi6, f->v[3], g->v[3]);
    lo7 = _mm512_madd52lo_epu64(lo7, f->v[3], g->v[4]);
    hi7 = _mm512_madd52hi_epu64(hi7, f->v[3], g->v[4]);

    // Round 4: f[4] * g[0..4]
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[4], g->v[0]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[4], g->v[0]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[4], g->v[1]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[4], g->v[1]);
    lo6 = _mm512_madd52lo_epu64(lo6, f->v[4], g->v[2]);
    hi6 = _mm512_madd52hi_epu64(hi6, f->v[4], g->v[2]);
    lo7 = _mm512_madd52lo_epu64(lo7, f->v[4], g->v[3]);
    hi7 = _mm512_madd52hi_epu64(hi7, f->v[4], g->v[3]);
    lo8 = _mm512_madd52lo_epu64(lo8, f->v[4], g->v[4]);
    hi8 = _mm512_madd52hi_epu64(hi8, f->v[4], g->v[4]);

    // Recombine: IFMA splits at bit 52, our radix is 2^51.
    // Actual schoolbook limb k value = lo[k] + hi[k] * 2^52.
    // In radix-2^51: c[k] = lo[k], carry from hi into limb k+1 is hi[k] * 2 (since 2^52 / 2^51 = 2).
    // So: c[0] = lo[0]
    //     c[k] = lo[k] + hi[k-1] << 1   for k = 1..8
    //     (c[9] = hi[8] << 1, but we fold c[5..8] with x19 first)

    // Fold upper limbs: r[i] = c[i] + 19 * c[i+5] for i = 0..4
    // c[5] = lo[5] + (hi[4] << 1)
    // c[6] = lo[6] + (hi[5] << 1)
    // c[7] = lo[7] + (hi[6] << 1)
    // c[8] = lo[8] + (hi[7] << 1)
    // c[9] = hi[8] << 1 (only contributes to folded limb 4)

    // Compute c[5..8] incorporating the hi carry-in
    __m512i c5 = _mm512_add_epi64(lo5, _mm512_slli_epi64(hi4, 1));
    __m512i c6 = _mm512_add_epi64(lo6, _mm512_slli_epi64(hi5, 1));
    __m512i c7 = _mm512_add_epi64(lo7, _mm512_slli_epi64(hi6, 1));
    __m512i c8 = _mm512_add_epi64(lo8, _mm512_slli_epi64(hi7, 1));

// x19 via shift-and-add: 19*x = (x<<4) + (x<<1) + x
// This avoids IFMA (which would truncate to 52 bits) and _mm512_mullo_epi64
// (which is slow on some uarchs). Values fit in 64-bit: max c[k] ~ 2^55,
// so 19*c[k] ~ 2^59.3, well within 64-bit range.
#define MUL19_VEC(x) _mm512_add_epi64(_mm512_add_epi64(_mm512_slli_epi64((x), 4), _mm512_slli_epi64((x), 1)), (x))

    __m512i r0 = _mm512_add_epi64(lo0, MUL19_VEC(c5));
    __m512i r1 = _mm512_add_epi64(_mm512_add_epi64(lo1, _mm512_slli_epi64(hi0, 1)), MUL19_VEC(c6));
    __m512i r2 = _mm512_add_epi64(_mm512_add_epi64(lo2, _mm512_slli_epi64(hi1, 1)), MUL19_VEC(c7));
    __m512i r3 = _mm512_add_epi64(_mm512_add_epi64(lo3, _mm512_slli_epi64(hi2, 1)), MUL19_VEC(c8));
    __m512i r4 =
        _mm512_add_epi64(_mm512_add_epi64(lo4, _mm512_slli_epi64(hi3, 1)), MUL19_VEC(_mm512_slli_epi64(hi8, 1)));

#undef MUL19_VEC

    // Carry-propagate to <=51-bit output
    const __m512i mask = FP51X8_MASK51;
    __m512i c;

    c = _mm512_srli_epi64(r0, 51);
    r1 = _mm512_add_epi64(r1, c);
    r0 = _mm512_and_si512(r0, mask);
    c = _mm512_srli_epi64(r1, 51);
    r2 = _mm512_add_epi64(r2, c);
    r1 = _mm512_and_si512(r1, mask);
    c = _mm512_srli_epi64(r2, 51);
    r3 = _mm512_add_epi64(r3, c);
    r2 = _mm512_and_si512(r2, mask);
    c = _mm512_srli_epi64(r3, 51);
    r4 = _mm512_add_epi64(r4, c);
    r3 = _mm512_and_si512(r3, mask);
    c = _mm512_srli_epi64(r4, 51);
    r0 = _mm512_add_epi64(r0, _mm512_add_epi64(_mm512_add_epi64(_mm512_slli_epi64(c, 4), _mm512_slli_epi64(c, 1)), c));
    r4 = _mm512_and_si512(r4, mask);
    c = _mm512_srli_epi64(r0, 51);
    r1 = _mm512_add_epi64(r1, c);
    r0 = _mm512_and_si512(r0, mask);

    h->v[0] = r0;
    h->v[1] = r1;
    h->v[2] = r2;
    h->v[3] = r3;
    h->v[4] = r4;
}

// -- Squaring --
// Currently implemented as mul(f, f). Could be optimized to exploit symmetry
// (15 unique products instead of 25) but the savings would be modest given
// that IFMA throughput is the bottleneck, not instruction count.

static FP51X8_FORCE_INLINE void fp51x8_sq(fp51x8 *h, const fp51x8 *f)
{
    fp51x8_mul(h, f, f);
}

// -- Double-squaring: h = 2 * f^2 --
// Used by point doubling for the 2*Z^2 term. Computes all 25 products,
// doubles every accumulator (lo and hi), then proceeds with the same
// recombination and carry chain as mul.

static FP51X8_FORCE_INLINE void fp51x8_sq2(fp51x8 *h, const fp51x8 *f)
{
    const __m512i zero = _mm512_setzero_si512();

    __m512i lo0 = zero, lo1 = zero, lo2 = zero, lo3 = zero, lo4 = zero;
    __m512i lo5 = zero, lo6 = zero, lo7 = zero, lo8 = zero;
    __m512i hi0 = zero, hi1 = zero, hi2 = zero, hi3 = zero, hi4 = zero;
    __m512i hi5 = zero, hi6 = zero, hi7 = zero, hi8 = zero;

    // Same 25 products as mul, but f=g
    lo0 = _mm512_madd52lo_epu64(lo0, f->v[0], f->v[0]);
    hi0 = _mm512_madd52hi_epu64(hi0, f->v[0], f->v[0]);
    lo1 = _mm512_madd52lo_epu64(lo1, f->v[0], f->v[1]);
    hi1 = _mm512_madd52hi_epu64(hi1, f->v[0], f->v[1]);
    lo2 = _mm512_madd52lo_epu64(lo2, f->v[0], f->v[2]);
    hi2 = _mm512_madd52hi_epu64(hi2, f->v[0], f->v[2]);
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[0], f->v[3]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[0], f->v[3]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[0], f->v[4]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[0], f->v[4]);

    lo1 = _mm512_madd52lo_epu64(lo1, f->v[1], f->v[0]);
    hi1 = _mm512_madd52hi_epu64(hi1, f->v[1], f->v[0]);
    lo2 = _mm512_madd52lo_epu64(lo2, f->v[1], f->v[1]);
    hi2 = _mm512_madd52hi_epu64(hi2, f->v[1], f->v[1]);
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[1], f->v[2]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[1], f->v[2]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[1], f->v[3]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[1], f->v[3]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[1], f->v[4]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[1], f->v[4]);

    lo2 = _mm512_madd52lo_epu64(lo2, f->v[2], f->v[0]);
    hi2 = _mm512_madd52hi_epu64(hi2, f->v[2], f->v[0]);
    lo3 = _mm512_madd52lo_epu64(lo3, f->v[2], f->v[1]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[2], f->v[1]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[2], f->v[2]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[2], f->v[2]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[2], f->v[3]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[2], f->v[3]);
    lo6 = _mm512_madd52lo_epu64(lo6, f->v[2], f->v[4]);
    hi6 = _mm512_madd52hi_epu64(hi6, f->v[2], f->v[4]);

    lo3 = _mm512_madd52lo_epu64(lo3, f->v[3], f->v[0]);
    hi3 = _mm512_madd52hi_epu64(hi3, f->v[3], f->v[0]);
    lo4 = _mm512_madd52lo_epu64(lo4, f->v[3], f->v[1]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[3], f->v[1]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[3], f->v[2]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[3], f->v[2]);
    lo6 = _mm512_madd52lo_epu64(lo6, f->v[3], f->v[3]);
    hi6 = _mm512_madd52hi_epu64(hi6, f->v[3], f->v[3]);
    lo7 = _mm512_madd52lo_epu64(lo7, f->v[3], f->v[4]);
    hi7 = _mm512_madd52hi_epu64(hi7, f->v[3], f->v[4]);

    lo4 = _mm512_madd52lo_epu64(lo4, f->v[4], f->v[0]);
    hi4 = _mm512_madd52hi_epu64(hi4, f->v[4], f->v[0]);
    lo5 = _mm512_madd52lo_epu64(lo5, f->v[4], f->v[1]);
    hi5 = _mm512_madd52hi_epu64(hi5, f->v[4], f->v[1]);
    lo6 = _mm512_madd52lo_epu64(lo6, f->v[4], f->v[2]);
    hi6 = _mm512_madd52hi_epu64(hi6, f->v[4], f->v[2]);
    lo7 = _mm512_madd52lo_epu64(lo7, f->v[4], f->v[3]);
    hi7 = _mm512_madd52hi_epu64(hi7, f->v[4], f->v[3]);
    lo8 = _mm512_madd52lo_epu64(lo8, f->v[4], f->v[4]);
    hi8 = _mm512_madd52hi_epu64(hi8, f->v[4], f->v[4]);

    // Double all accumulators (sq2 = 2 * f^2)
    lo0 = _mm512_add_epi64(lo0, lo0);
    hi0 = _mm512_add_epi64(hi0, hi0);
    lo1 = _mm512_add_epi64(lo1, lo1);
    hi1 = _mm512_add_epi64(hi1, hi1);
    lo2 = _mm512_add_epi64(lo2, lo2);
    hi2 = _mm512_add_epi64(hi2, hi2);
    lo3 = _mm512_add_epi64(lo3, lo3);
    hi3 = _mm512_add_epi64(hi3, hi3);
    lo4 = _mm512_add_epi64(lo4, lo4);
    hi4 = _mm512_add_epi64(hi4, hi4);
    lo5 = _mm512_add_epi64(lo5, lo5);
    hi5 = _mm512_add_epi64(hi5, hi5);
    lo6 = _mm512_add_epi64(lo6, lo6);
    hi6 = _mm512_add_epi64(hi6, hi6);
    lo7 = _mm512_add_epi64(lo7, lo7);
    hi7 = _mm512_add_epi64(hi7, hi7);
    lo8 = _mm512_add_epi64(lo8, lo8);
    hi8 = _mm512_add_epi64(hi8, hi8);

    // Same recombination and fold as mul
    __m512i c5 = _mm512_add_epi64(lo5, _mm512_slli_epi64(hi4, 1));
    __m512i c6 = _mm512_add_epi64(lo6, _mm512_slli_epi64(hi5, 1));
    __m512i c7 = _mm512_add_epi64(lo7, _mm512_slli_epi64(hi6, 1));
    __m512i c8 = _mm512_add_epi64(lo8, _mm512_slli_epi64(hi7, 1));

#define MUL19_VEC(x) _mm512_add_epi64(_mm512_add_epi64(_mm512_slli_epi64((x), 4), _mm512_slli_epi64((x), 1)), (x))

    __m512i r0 = _mm512_add_epi64(lo0, MUL19_VEC(c5));
    __m512i r1 = _mm512_add_epi64(_mm512_add_epi64(lo1, _mm512_slli_epi64(hi0, 1)), MUL19_VEC(c6));
    __m512i r2 = _mm512_add_epi64(_mm512_add_epi64(lo2, _mm512_slli_epi64(hi1, 1)), MUL19_VEC(c7));
    __m512i r3 = _mm512_add_epi64(_mm512_add_epi64(lo3, _mm512_slli_epi64(hi2, 1)), MUL19_VEC(c8));
    __m512i r4 =
        _mm512_add_epi64(_mm512_add_epi64(lo4, _mm512_slli_epi64(hi3, 1)), MUL19_VEC(_mm512_slli_epi64(hi8, 1)));

#undef MUL19_VEC

    const __m512i mask = FP51X8_MASK51;
    __m512i c;

    c = _mm512_srli_epi64(r0, 51);
    r1 = _mm512_add_epi64(r1, c);
    r0 = _mm512_and_si512(r0, mask);
    c = _mm512_srli_epi64(r1, 51);
    r2 = _mm512_add_epi64(r2, c);
    r1 = _mm512_and_si512(r1, mask);
    c = _mm512_srli_epi64(r2, 51);
    r3 = _mm512_add_epi64(r3, c);
    r2 = _mm512_and_si512(r2, mask);
    c = _mm512_srli_epi64(r3, 51);
    r4 = _mm512_add_epi64(r4, c);
    r3 = _mm512_and_si512(r3, mask);
    c = _mm512_srli_epi64(r4, 51);
    r0 = _mm512_add_epi64(r0, _mm512_add_epi64(_mm512_add_epi64(_mm512_slli_epi64(c, 4), _mm512_slli_epi64(c, 1)), c));
    r4 = _mm512_and_si512(r4, mask);
    c = _mm512_srli_epi64(r0, 51);
    r1 = _mm512_add_epi64(r1, c);
    r0 = _mm512_and_si512(r0, mask);

    h->v[0] = r0;
    h->v[1] = r1;
    h->v[2] = r2;
    h->v[3] = r3;
    h->v[4] = r4;
}

// -- Lane insert / extract --
// These convert between scalar fp_fe (single field element) and one lane of an
// fp51x8. They're only used at batch entry (packing input points) and exit
// (extracting results) -- not in the hot loop. The store-modify-reload
// pattern isn't efficient, but it doesn't matter for a handful of calls.

static FP51X8_FORCE_INLINE void fp51x8_insert_lane(fp51x8 *out, const fp_fe in, int lane)
{
    alignas(64) long long tmp[8];
    for (int i = 0; i < 5; i++)
    {
        _mm512_store_si512((__m512i *)tmp, out->v[i]);
        tmp[lane] = (long long)in[i];
        out->v[i] = _mm512_load_si512((const __m512i *)tmp);
    }
}

static FP51X8_FORCE_INLINE void fp51x8_extract_lane(fp_fe out, const fp51x8 *in, int lane)
{
    alignas(64) long long tmp[8];
    for (int i = 0; i < 5; i++)
    {
        _mm512_store_si512((__m512i *)tmp, in->v[i]);
        out[i] = (uint64_t)tmp[lane];
    }
}

#endif // RANSHAW_X64_IFMA_FP51X8_IFMA_H
