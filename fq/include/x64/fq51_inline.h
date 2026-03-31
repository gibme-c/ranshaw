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
 * @file fq51_inline.h
 * @brief x64 (radix-2^51) implementation of F_q inline arithmetic helpers with Crandall reduction.
 */

#ifndef RANSHAW_X64_FQ51_INLINE_H
#define RANSHAW_X64_FQ51_INLINE_H

#include "fq.h"
#include "ranshaw_platform.h"
#include "x64/fq51.h"
#include "x64/mul128.h"

#if !defined(RANSHAW_FORCE_INLINE)
#if defined(_MSC_VER)
#define RANSHAW_FORCE_INLINE __forceinline
#else
#define RANSHAW_FORCE_INLINE inline __attribute__((always_inline))
#endif
#endif

/*
 * 4×64 helpers for squaring chains (used by fq51_chain.h).
 *
 * For squaring chains (inversion, sqrt), pack once → N squarings in 4×64
 * → unpack once. The 4×64 path uses 2^256 ≡ 2*gamma (mod q) with
 * TWO_GAMMA_64 (2 or 3 limbs) for cheaper Crandall folds.
 *
 * Individual mul/sq calls use the 5×51 path below (no pack/unpack overhead).
 */

#if defined(__GNUC__) && defined(__BMI2__) && RANSHAW_HAVE_INT128

static RANSHAW_FORCE_INLINE void fq51_normalize_and_pack(uint64_t r[4], const fq_fe f)
{
    const uint64_t M = FQ51_MASK;
    uint64_t f0, f1, f2, f3, f4, c;

    /* First carry chain */
    c = f[0] >> 51;
    f0 = f[0] & M;
    f1 = f[1] + c;
    c = f1 >> 51;
    f1 &= M;
    f2 = f[2] + c;
    c = f2 >> 51;
    f2 &= M;
    f3 = f[3] + c;
    c = f3 >> 51;
    f3 &= M;
    f4 = f[4] + c;
    c = f4 >> 51;
    f4 &= M;

    /* Gamma fold of carry */
    {
        uint64_t *fs[] = {&f0, &f1, &f2, &f3, &f4};
        for (int j = 0; j < GAMMA_51_LIMBS; j++)
            *fs[j] += c * GAMMA_51[j];
    }

    /* Second carry chain */
    c = f0 >> 51;
    f0 &= M;
    f1 += c;
    c = f1 >> 51;
    f1 &= M;
    f2 += c;
    c = f2 >> 51;
    f2 &= M;
    f3 += c;
    c = f3 >> 51;
    f3 &= M;
    f4 += c;
    c = f4 >> 51;
    f4 &= M;

    /* Second gamma fold (c is now very small, typically 0 or 1) */
    {
        uint64_t *fs[] = {&f0, &f1, &f2, &f3, &f4};
        for (int j = 0; j < GAMMA_51_LIMBS; j++)
            *fs[j] += c * GAMMA_51[j];
    }

    /* Final carry chain to ensure all limbs ≤ 51 bits */
    c = f0 >> 51;
    f0 &= M;
    f1 += c;
    c = f1 >> 51;
    f1 &= M;
    f2 += c;
    c = f2 >> 51;
    f2 &= M;
    f3 += c;
    c = f3 >> 51;
    f3 &= M;
    f4 += c; /* f4 ≤ 52 bits; f4<<12 ≤ 64 bits */

    /* Pack 5×51 → 4×64 */
    r[0] = f0 | (f1 << 51);
    r[1] = (f1 >> 13) | (f2 << 38);
    r[2] = (f2 >> 26) | (f3 << 25);
    r[3] = (f3 >> 39) | (f4 << 12);
}

static RANSHAW_FORCE_INLINE void fq64_to_fq51(fq_fe h, const uint64_t r[4])
{
    const uint64_t M = FQ51_MASK;
    h[0] = r[0] & M;
    h[1] = ((r[0] >> 51) | (r[1] << 13)) & M;
    h[2] = ((r[1] >> 38) | (r[2] << 26)) & M;
    h[3] = ((r[2] >> 25) | (r[3] << 39)) & M;
    h[4] = r[3] >> 12;
}

/*
 * 4×64 addition with Crandall correction: h = f + g (mod 2^256 with correction).
 * If sum overflows 256 bits, add 2*gamma (since 2^256 ≡ 2*gamma mod q).
 * Result < 2^256 (proven: inputs < 2q < 2^256, sum < 4q, after correction < 2^256).
 */
#if defined(__ADX__) && TWO_GAMMA_64_LIMBS <= 3
#if TWO_GAMMA_64_LIMBS <= 2
static RANSHAW_FORCE_INLINE void fq64_add(uint64_t h[4], const uint64_t f[4], const uint64_t g[4])
{
    uint64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3];
    uint64_t g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3];
    __asm__ __volatile__("movq %[f0], %%r8\n\t"
                         "addq %[g0], %%r8\n\t"
                         "movq %[f1], %%r9\n\t"
                         "adcq %[g1], %%r9\n\t"
                         "movq %[f2], %%r10\n\t"
                         "adcq %[g2], %%r10\n\t"
                         "movq %[f3], %%r11\n\t"
                         "adcq %[g3], %%r11\n\t"
                         /* If carry: add TWO_GAMMA_64 */
                         "movl $0, %%eax\n\t"
                         "adcq $0, %%rax\n\t"
                         "negq %%rax\n\t"
                         "movq %[G0], %%rcx\n\t"
                         "andq %%rax, %%rcx\n\t"
                         "movq %[G1], %%rdx\n\t"
                         "andq %%rax, %%rdx\n\t"
                         "addq %%rcx, %%r8\n\t"
                         "adcq %%rdx, %%r9\n\t"
                         "adcq $0, %%r10\n\t"
                         "adcq $0, %%r11\n\t"
                         /* Second correction (rare, but CT) */
                         "movl $0, %%eax\n\t"
                         "adcq $0, %%rax\n\t"
                         "negq %%rax\n\t"
                         "movq %[G0], %%rcx\n\t"
                         "andq %%rax, %%rcx\n\t"
                         "movq %[G1], %%rdx\n\t"
                         "andq %%rax, %%rdx\n\t"
                         "addq %%rcx, %%r8\n\t"
                         "adcq %%rdx, %%r9\n\t"
                         "adcq $0, %%r10\n\t"
                         "adcq $0, %%r11\n\t"
                         "movq %%r8, %[h0]\n\t"
                         "movq %%r9, %[h1]\n\t"
                         "movq %%r10, %[h2]\n\t"
                         "movq %%r11, %[h3]\n\t"
                         : [h0] "=m"(h[0]), [h1] "=m"(h[1]), [h2] "=m"(h[2]), [h3] "=m"(h[3])
                         : [f0] "m"(f0),
                           [f1] "m"(f1),
                           [f2] "m"(f2),
                           [f3] "m"(f3),
                           [g0] "m"(g0),
                           [g1] "m"(g1),
                           [g2] "m"(g2),
                           [g3] "m"(g3),
                           [G0] "m"(TWO_GAMMA_64[0]),
                           [G1] "m"(TWO_GAMMA_64[1])
                         : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "cc", "memory");
}
#else /* TWO_GAMMA_64_LIMBS == 3 */
static RANSHAW_FORCE_INLINE void fq64_add(uint64_t h[4], const uint64_t f[4], const uint64_t g[4])
{
    uint64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3];
    uint64_t g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3];
    __asm__ __volatile__("movq %[f0], %%r8\n\t"
                         "addq %[g0], %%r8\n\t"
                         "movq %[f1], %%r9\n\t"
                         "adcq %[g1], %%r9\n\t"
                         "movq %[f2], %%r10\n\t"
                         "adcq %[g2], %%r10\n\t"
                         "movq %[f3], %%r11\n\t"
                         "adcq %[g3], %%r11\n\t"
                         /* If carry: add TWO_GAMMA_64 (3 limbs) */
                         "movl $0, %%eax\n\t"
                         "adcq $0, %%rax\n\t"
                         "negq %%rax\n\t"
                         "movq %[G0], %%rcx\n\t"
                         "andq %%rax, %%rcx\n\t"
                         "movq %[G1], %%rdx\n\t"
                         "andq %%rax, %%rdx\n\t"
                         "movq %[G2], %%rdi\n\t"
                         "andq %%rax, %%rdi\n\t"
                         "addq %%rcx, %%r8\n\t"
                         "adcq %%rdx, %%r9\n\t"
                         "adcq %%rdi, %%r10\n\t"
                         "adcq $0, %%r11\n\t"
                         /* Second correction (rare, but CT) */
                         "movl $0, %%eax\n\t"
                         "adcq $0, %%rax\n\t"
                         "negq %%rax\n\t"
                         "movq %[G0], %%rcx\n\t"
                         "andq %%rax, %%rcx\n\t"
                         "movq %[G1], %%rdx\n\t"
                         "andq %%rax, %%rdx\n\t"
                         "movq %[G2], %%rdi\n\t"
                         "andq %%rax, %%rdi\n\t"
                         "addq %%rcx, %%r8\n\t"
                         "adcq %%rdx, %%r9\n\t"
                         "adcq %%rdi, %%r10\n\t"
                         "adcq $0, %%r11\n\t"
                         "movq %%r8, %[h0]\n\t"
                         "movq %%r9, %[h1]\n\t"
                         "movq %%r10, %[h2]\n\t"
                         "movq %%r11, %[h3]\n\t"
                         : [h0] "=m"(h[0]), [h1] "=m"(h[1]), [h2] "=m"(h[2]), [h3] "=m"(h[3])
                         : [f0] "m"(f0),
                           [f1] "m"(f1),
                           [f2] "m"(f2),
                           [f3] "m"(f3),
                           [g0] "m"(g0),
                           [g1] "m"(g1),
                           [g2] "m"(g2),
                           [g3] "m"(g3),
                           [G0] "m"(TWO_GAMMA_64[0]),
                           [G1] "m"(TWO_GAMMA_64[1]),
                           [G2] "m"(TWO_GAMMA_64[2])
                         : "rax", "rcx", "rdx", "rdi", "r8", "r9", "r10", "r11", "cc", "memory");
}
#endif /* TWO_GAMMA_64_LIMBS */
#else
static RANSHAW_FORCE_INLINE void fq64_add(uint64_t h[4], const uint64_t f[4], const uint64_t g[4])
{
    __uint128_t acc;
    uint64_t carry;
    acc = (__uint128_t)f[0] + g[0];
    h[0] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)f[1] + g[1] + carry;
    h[1] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)f[2] + g[2] + carry;
    h[2] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)f[3] + g[3] + carry;
    h[3] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    uint64_t mask = -(uint64_t)carry;
    for (int j = 0; j < TWO_GAMMA_64_LIMBS; j++)
    {
        acc = (__uint128_t)h[j] + (TWO_GAMMA_64[j] & mask) + carry;
        h[j] = (uint64_t)acc;
        carry = (uint64_t)(acc >> 64);
    }
    for (int j = TWO_GAMMA_64_LIMBS; j < 4; j++)
    {
        h[j] += carry;
        carry = (h[j] < carry) ? 1 : 0;
    }
}
#endif

/*
 * 4×64 subtraction with Crandall correction: h = f - g (mod 2^256 with correction).
 * If sub borrows, subtract 2*gamma (undo the 2^256 wrap: need to remove the
 * excess 2^256 - q ≈ 2*gamma that wrapping implicitly added).
 */
#if defined(__ADX__) && TWO_GAMMA_64_LIMBS <= 3
#if TWO_GAMMA_64_LIMBS <= 2
static RANSHAW_FORCE_INLINE void fq64_sub(uint64_t h[4], const uint64_t f[4], const uint64_t g[4])
{
    uint64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3];
    uint64_t g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3];
    __asm__ __volatile__("movq %[f0], %%r8\n\t"
                         "subq %[g0], %%r8\n\t"
                         "movq %[f1], %%r9\n\t"
                         "sbbq %[g1], %%r9\n\t"
                         "movq %[f2], %%r10\n\t"
                         "sbbq %[g2], %%r10\n\t"
                         "movq %[f3], %%r11\n\t"
                         "sbbq %[g3], %%r11\n\t"
                         /* If borrow: subtract TWO_GAMMA_64 */
                         "movl $0, %%eax\n\t"
                         "adcq $0, %%rax\n\t"
                         "negq %%rax\n\t"
                         "movq %[G0], %%rcx\n\t"
                         "andq %%rax, %%rcx\n\t"
                         "movq %[G1], %%rdx\n\t"
                         "andq %%rax, %%rdx\n\t"
                         "subq %%rcx, %%r8\n\t"
                         "sbbq %%rdx, %%r9\n\t"
                         "sbbq $0, %%r10\n\t"
                         "sbbq $0, %%r11\n\t"
                         /* Second correction (rare, but CT) */
                         "movl $0, %%eax\n\t"
                         "adcq $0, %%rax\n\t"
                         "negq %%rax\n\t"
                         "movq %[G0], %%rcx\n\t"
                         "andq %%rax, %%rcx\n\t"
                         "movq %[G1], %%rdx\n\t"
                         "andq %%rax, %%rdx\n\t"
                         "subq %%rcx, %%r8\n\t"
                         "sbbq %%rdx, %%r9\n\t"
                         "sbbq $0, %%r10\n\t"
                         "sbbq $0, %%r11\n\t"
                         "movq %%r8, %[h0]\n\t"
                         "movq %%r9, %[h1]\n\t"
                         "movq %%r10, %[h2]\n\t"
                         "movq %%r11, %[h3]\n\t"
                         : [h0] "=m"(h[0]), [h1] "=m"(h[1]), [h2] "=m"(h[2]), [h3] "=m"(h[3])
                         : [f0] "m"(f0),
                           [f1] "m"(f1),
                           [f2] "m"(f2),
                           [f3] "m"(f3),
                           [g0] "m"(g0),
                           [g1] "m"(g1),
                           [g2] "m"(g2),
                           [g3] "m"(g3),
                           [G0] "m"(TWO_GAMMA_64[0]),
                           [G1] "m"(TWO_GAMMA_64[1])
                         : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "cc", "memory");
}
#else /* TWO_GAMMA_64_LIMBS == 3 */
static RANSHAW_FORCE_INLINE void fq64_sub(uint64_t h[4], const uint64_t f[4], const uint64_t g[4])
{
    uint64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3];
    uint64_t g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3];
    __asm__ __volatile__("movq %[f0], %%r8\n\t"
                         "subq %[g0], %%r8\n\t"
                         "movq %[f1], %%r9\n\t"
                         "sbbq %[g1], %%r9\n\t"
                         "movq %[f2], %%r10\n\t"
                         "sbbq %[g2], %%r10\n\t"
                         "movq %[f3], %%r11\n\t"
                         "sbbq %[g3], %%r11\n\t"
                         /* If borrow: subtract TWO_GAMMA_64 (3 limbs) */
                         "movl $0, %%eax\n\t"
                         "adcq $0, %%rax\n\t"
                         "negq %%rax\n\t"
                         "movq %[G0], %%rcx\n\t"
                         "andq %%rax, %%rcx\n\t"
                         "movq %[G1], %%rdx\n\t"
                         "andq %%rax, %%rdx\n\t"
                         "movq %[G2], %%rdi\n\t"
                         "andq %%rax, %%rdi\n\t"
                         "subq %%rcx, %%r8\n\t"
                         "sbbq %%rdx, %%r9\n\t"
                         "sbbq %%rdi, %%r10\n\t"
                         "sbbq $0, %%r11\n\t"
                         /* Second correction (rare, but CT) */
                         "movl $0, %%eax\n\t"
                         "adcq $0, %%rax\n\t"
                         "negq %%rax\n\t"
                         "movq %[G0], %%rcx\n\t"
                         "andq %%rax, %%rcx\n\t"
                         "movq %[G1], %%rdx\n\t"
                         "andq %%rax, %%rdx\n\t"
                         "movq %[G2], %%rdi\n\t"
                         "andq %%rax, %%rdi\n\t"
                         "subq %%rcx, %%r8\n\t"
                         "sbbq %%rdx, %%r9\n\t"
                         "sbbq %%rdi, %%r10\n\t"
                         "sbbq $0, %%r11\n\t"
                         "movq %%r8, %[h0]\n\t"
                         "movq %%r9, %[h1]\n\t"
                         "movq %%r10, %[h2]\n\t"
                         "movq %%r11, %[h3]\n\t"
                         : [h0] "=m"(h[0]), [h1] "=m"(h[1]), [h2] "=m"(h[2]), [h3] "=m"(h[3])
                         : [f0] "m"(f0),
                           [f1] "m"(f1),
                           [f2] "m"(f2),
                           [f3] "m"(f3),
                           [g0] "m"(g0),
                           [g1] "m"(g1),
                           [g2] "m"(g2),
                           [g3] "m"(g3),
                           [G0] "m"(TWO_GAMMA_64[0]),
                           [G1] "m"(TWO_GAMMA_64[1]),
                           [G2] "m"(TWO_GAMMA_64[2])
                         : "rax", "rcx", "rdx", "rdi", "r8", "r9", "r10", "r11", "cc", "memory");
}
#endif /* TWO_GAMMA_64_LIMBS */
#else
static RANSHAW_FORCE_INLINE void fq64_sub(uint64_t h[4], const uint64_t f[4], const uint64_t g[4])
{
    __uint128_t acc;
    uint64_t borrow;
    acc = (__uint128_t)f[0] - g[0];
    h[0] = (uint64_t)acc;
    borrow = (uint64_t)(acc >> 64) & 1;
    acc = (__uint128_t)f[1] - g[1] - borrow;
    h[1] = (uint64_t)acc;
    borrow = (uint64_t)(acc >> 64) & 1;
    acc = (__uint128_t)f[2] - g[2] - borrow;
    h[2] = (uint64_t)acc;
    borrow = (uint64_t)(acc >> 64) & 1;
    acc = (__uint128_t)f[3] - g[3] - borrow;
    h[3] = (uint64_t)acc;
    borrow = (uint64_t)(acc >> 64) & 1;
    uint64_t mask = -(uint64_t)borrow;
    for (int j = 0; j < TWO_GAMMA_64_LIMBS; j++)
    {
        acc = (__uint128_t)h[j] - (TWO_GAMMA_64[j] & mask) - borrow;
        h[j] = (uint64_t)acc;
        borrow = (uint64_t)(acc >> 64) & 1;
    }
    for (int j = TWO_GAMMA_64_LIMBS; j < 4; j++)
    {
        uint64_t prev = h[j];
        h[j] -= borrow;
        borrow = (h[j] > prev) ? 1 : 0;
    }
}
#endif

/*
 * C fallback for 4×64 multiply: a[0..3] × b[0..3] → r[0..3] (mod q).
 */
static RANSHAW_FORCE_INLINE void fq64_mul_c(uint64_t r[4], const uint64_t a[4], const uint64_t b[4])
{
    uint64_t w[8] = {0};
    __uint128_t acc;
    uint64_t carry;

    /* 4×4 schoolbook (untouched — not gamma-dependent) */
    acc = (__uint128_t)a[0] * b[0];
    w[0] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[0] * b[1] + carry;
    w[1] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[0] * b[2] + carry;
    w[2] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[0] * b[3] + carry;
    w[3] = (uint64_t)acc;
    w[4] = (uint64_t)(acc >> 64);

    acc = (__uint128_t)a[1] * b[0] + w[1];
    w[1] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[1] * b[1] + w[2] + carry;
    w[2] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[1] * b[2] + w[3] + carry;
    w[3] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[1] * b[3] + w[4] + carry;
    w[4] = (uint64_t)acc;
    w[5] = (uint64_t)(acc >> 64);

    acc = (__uint128_t)a[2] * b[0] + w[2];
    w[2] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[2] * b[1] + w[3] + carry;
    w[3] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[2] * b[2] + w[4] + carry;
    w[4] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[2] * b[3] + w[5] + carry;
    w[5] = (uint64_t)acc;
    w[6] = (uint64_t)(acc >> 64);

    acc = (__uint128_t)a[3] * b[0] + w[3];
    w[3] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[3] * b[1] + w[4] + carry;
    w[4] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[3] * b[2] + w[5] + carry;
    w[5] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[3] * b[3] + w[6] + carry;
    w[6] = (uint64_t)acc;
    w[7] = (uint64_t)(acc >> 64);

    /*
     * Iterative Crandall fold: fold w[4..7] into w[0..3] by convolving
     * with TWO_GAMMA_64[0..TWO_GAMMA_64_LIMBS-1]. Repeat until overflow
     * is absorbed. 3 fold iterations + 1 CT correction suffices.
     */
    for (int fold = 0; fold < 3; fold++)
    {
        /* Save and zero overflow */
        uint64_t over[4];
        for (int i = 0; i < 4; i++)
        {
            over[i] = w[4 + i];
            w[4 + i] = 0;
        }

        /* Convolve each overflow word with TWO_GAMMA_64 */
        for (int i = 0; i < 4; i++)
        {
            carry = 0;
            for (int j = 0; j < TWO_GAMMA_64_LIMBS; j++)
            {
                acc = (__uint128_t)over[i] * TWO_GAMMA_64[j] + w[i + j] + carry;
                w[i + j] = (uint64_t)acc;
                carry = (uint64_t)(acc >> 64);
            }
            /* Propagate carry through remaining positions (CT: no early exit) */
            for (int k = i + TWO_GAMMA_64_LIMBS; k < 8; k++)
            {
                w[k] += carry;
                carry = (w[k] < carry) ? 1 : 0;
            }
        }
    }

    /* Final CT correction: if w[4] is nonzero (0 or 1), add TWO_GAMMA_64 */
    {
        uint64_t mask = -(uint64_t)(w[4] & 1);
        carry = 0;
        for (int j = 0; j < TWO_GAMMA_64_LIMBS; j++)
        {
            acc = (__uint128_t)w[j] + (TWO_GAMMA_64[j] & mask) + carry;
            w[j] = (uint64_t)acc;
            carry = (uint64_t)(acc >> 64);
        }
        for (int j = TWO_GAMMA_64_LIMBS; j < 4; j++)
        {
            w[j] += carry;
            carry = (w[j] < carry) ? 1 : 0;
        }
    }

    r[0] = w[0];
    r[1] = w[1];
    r[2] = w[2];
    r[3] = w[3];
}

/*
 * 4×64 Crandall squaring: a[0..3]² → r[0..3] (mod q).
 */
#if defined(__ADX__) && TWO_GAMMA_64_LIMBS <= 3
#if TWO_GAMMA_64_LIMBS <= 2
/*
 * MULX+ADCX+ADOX version (2-limb TWO_GAMMA_64): uses dual carry chains for the schoolbook,
 * plain ADD/ADC for the Crandall fold. Requires BMI2 (MULX) + ADX (ADCX/ADOX).
 */
static RANSHAW_FORCE_INLINE void fq64_sq(uint64_t r[4], const uint64_t a[4])
{
    /* Copy inputs to locals to handle r==a aliasing in chain code */
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];

    __asm__ __volatile__(
        /* ===== 4×4 SCHOOLBOOK (16 MULX) ===== */
        /* Row 0: a[0] × a[0..3] → w[0..4] = r8..r12 */
        "movq %[a0], %%rdx\n\t"
        "mulxq %[a0], %%r8, %%r9\n\t"
        "mulxq %[a1], %%rax, %%r10\n\t"
        "addq %%rax, %%r9\n\t"
        "mulxq %[a2], %%rax, %%r11\n\t"
        "adcq %%rax, %%r10\n\t"
        "mulxq %[a3], %%rax, %%r12\n\t"
        "adcq %%rax, %%r11\n\t"
        "adcq $0, %%r12\n\t"
        /* Row 1: a[1] × a[0..3], ADCX+ADOX into w[1..5] */
        "movq %[a1], %%rdx\n\t"
        "xorl %%esi, %%esi\n\t"
        "mulxq %[a0], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r9\n\t"
        "adoxq %%rcx, %%r10\n\t"
        "mulxq %[a1], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "mulxq %[a2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rcx, %%r12\n\t"
        "mulxq %[a3], %%rax, %%r13\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rsi, %%r13\n\t"
        "adcxq %%rsi, %%r13\n\t"
        /* Row 2: a[2] × a[0..3] into w[2..6] */
        "movq %[a2], %%rdx\n\t"
        "xorl %%esi, %%esi\n\t"
        "mulxq %[a0], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "mulxq %[a1], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rcx, %%r12\n\t"
        "mulxq %[a2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rcx, %%r13\n\t"
        "mulxq %[a3], %%rax, %%r14\n\t"
        "adcxq %%rax, %%r13\n\t"
        "adoxq %%rsi, %%r14\n\t"
        "adcxq %%rsi, %%r14\n\t"
        /* Row 3: a[3] × a[0..3] into w[3..7] */
        "movq %[a3], %%rdx\n\t"
        "xorl %%esi, %%esi\n\t"
        "mulxq %[a0], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rcx, %%r12\n\t"
        "mulxq %[a1], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rcx, %%r13\n\t"
        "mulxq %[a2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r13\n\t"
        "adoxq %%rcx, %%r14\n\t"
        "mulxq %[a3], %%rax, %%r15\n\t"
        "adcxq %%rax, %%r14\n\t"
        "adoxq %%rsi, %%r15\n\t"
        "adcxq %%rsi, %%r15\n\t"

        /* w[0..7] = r8..r15 */

        /* ===== FIRST CRANDALL FOLD: w[4..7] × [G0,G1] ===== */

        /* Fold w4 (r12) → positions [0,1,2,3], w4=0 */
        "movq %%r12, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq %%rbx, %%r10\n\t"
        "adcq $0, %%r11\n\t"

        /* Fold w5 (r13) → positions [1,2,3,4] */
        "movq %%r13, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rbx, %%r11\n\t"
        "movl $0, %%r12d\n\t"
        "adcq $0, %%r12\n\t"

        /* Fold w6 (r14) → positions [2,3,4] */
        "movq %%r14, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r11\n\t"
        "adcq %%rbx, %%r12\n\t"
        "movl $0, %%r13d\n\t"
        "adcq $0, %%r13\n\t"

        /* Fold w7 (r15) → positions [3,4,5] */
        "movq %%r15, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r11\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r12\n\t"
        "adcq %%rbx, %%r13\n\t"

        /* After 1st fold: w[0..3]=r8..r11, w[4]=r12, w[5]=r13 */

        /* ===== SECOND CRANDALL FOLD: w[4..5] × [G0,G1] ===== */

        /* Fold w4 (r12) → positions [0,1,2,3], w4=0 */
        "movq %%r12, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq %%rbx, %%r10\n\t"
        "adcq $0, %%r11\n\t"

        /* Fold w5 (r13) → positions [1,2,3,4] */
        "movq %%r13, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rbx, %%r11\n\t"
        "movl $0, %%r12d\n\t"
        "adcq $0, %%r12\n\t"

        /* After 2nd fold: w[0..3]=r8..r11, w[4]=r12 (small) */

        /* ===== THIRD + FOURTH FOLD (CT conditional adds) ===== */
        /* w4 is 0 or 1: add w4 * [G0,G1] via mask */
        "negq %%r12\n\t"
        "movq %[G0], %%rax\n\t"
        "andq %%r12, %%rax\n\t"
        "movq %[G1], %%rcx\n\t"
        "andq %%r12, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rcx, %%r9\n\t"
        "adcq $0, %%r10\n\t"
        "adcq $0, %%r11\n\t"
        /* Fourth fold: carry from above */
        "movl $0, %%r12d\n\t"
        "adcq $0, %%r12\n\t"
        "negq %%r12\n\t"
        "movq %[G0], %%rax\n\t"
        "andq %%r12, %%rax\n\t"
        "movq %[G1], %%rcx\n\t"
        "andq %%r12, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rcx, %%r9\n\t"
        "adcq $0, %%r10\n\t"
        "adcq $0, %%r11\n\t"

        /* ===== STORE ===== */
        "movq %%r8, %[r0]\n\t"
        "movq %%r9, %[r1]\n\t"
        "movq %%r10, %[r2]\n\t"
        "movq %%r11, %[r3]\n\t"

        : [r0] "=m"(r[0]), [r1] "=m"(r[1]), [r2] "=m"(r[2]), [r3] "=m"(r[3])
        : [a0] "m"(a0), [a1] "m"(a1), [a2] "m"(a2), [a3] "m"(a3), [G0] "m"(TWO_GAMMA_64[0]), [G1] "m"(TWO_GAMMA_64[1])
        : "rax", "rbx", "rcx", "rdx", "rsi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "cc", "memory");
}

/*
 * 4×64 Crandall multiply: a[0..3] × b[0..3] → r[0..3] (mod q).
 * Same structure as fq64_sq but with two different inputs.
 */
static RANSHAW_FORCE_INLINE void fq64_mul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4])
{
    /* Copy inputs to locals to handle r==a aliasing */
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    uint64_t b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];

    __asm__ __volatile__(
        /* ===== 4×4 SCHOOLBOOK (16 MULX) ===== */
        "movq %[a0], %%rdx\n\t"
        "mulxq %[b0], %%r8, %%r9\n\t"
        "mulxq %[b1], %%rax, %%r10\n\t"
        "addq %%rax, %%r9\n\t"
        "mulxq %[b2], %%rax, %%r11\n\t"
        "adcq %%rax, %%r10\n\t"
        "mulxq %[b3], %%rax, %%r12\n\t"
        "adcq %%rax, %%r11\n\t"
        "adcq $0, %%r12\n\t"
        "movq %[a1], %%rdx\n\t"
        "xorl %%esi, %%esi\n\t"
        "mulxq %[b0], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r9\n\t"
        "adoxq %%rcx, %%r10\n\t"
        "mulxq %[b1], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "mulxq %[b2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rcx, %%r12\n\t"
        "mulxq %[b3], %%rax, %%r13\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rsi, %%r13\n\t"
        "adcxq %%rsi, %%r13\n\t"
        "movq %[a2], %%rdx\n\t"
        "xorl %%esi, %%esi\n\t"
        "mulxq %[b0], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "mulxq %[b1], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rcx, %%r12\n\t"
        "mulxq %[b2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rcx, %%r13\n\t"
        "mulxq %[b3], %%rax, %%r14\n\t"
        "adcxq %%rax, %%r13\n\t"
        "adoxq %%rsi, %%r14\n\t"
        "adcxq %%rsi, %%r14\n\t"
        "movq %[a3], %%rdx\n\t"
        "xorl %%esi, %%esi\n\t"
        "mulxq %[b0], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rcx, %%r12\n\t"
        "mulxq %[b1], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rcx, %%r13\n\t"
        "mulxq %[b2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r13\n\t"
        "adoxq %%rcx, %%r14\n\t"
        "mulxq %[b3], %%rax, %%r15\n\t"
        "adcxq %%rax, %%r14\n\t"
        "adoxq %%rsi, %%r15\n\t"
        "adcxq %%rsi, %%r15\n\t"
        /* ===== FIRST CRANDALL FOLD ===== */
        "movq %%r12, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq %%rbx, %%r10\n\t"
        "adcq $0, %%r11\n\t"
        "movq %%r13, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rbx, %%r11\n\t"
        "movl $0, %%r12d\n\t"
        "adcq $0, %%r12\n\t"
        "movq %%r14, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r11\n\t"
        "adcq %%rbx, %%r12\n\t"
        "movl $0, %%r13d\n\t"
        "adcq $0, %%r13\n\t"
        "movq %%r15, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r11\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r12\n\t"
        "adcq %%rbx, %%r13\n\t"
        /* ===== SECOND CRANDALL FOLD ===== */
        "movq %%r12, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq %%rbx, %%r10\n\t"
        "adcq $0, %%r11\n\t"
        "movq %%r13, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rbx, %%r11\n\t"
        "movl $0, %%r12d\n\t"
        "adcq $0, %%r12\n\t"
        /* ===== THIRD + FOURTH FOLD ===== */
        "negq %%r12\n\t"
        "movq %[G0], %%rax\n\t"
        "andq %%r12, %%rax\n\t"
        "movq %[G1], %%rcx\n\t"
        "andq %%r12, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rcx, %%r9\n\t"
        "adcq $0, %%r10\n\t"
        "adcq $0, %%r11\n\t"
        "movl $0, %%r12d\n\t"
        "adcq $0, %%r12\n\t"
        "negq %%r12\n\t"
        "movq %[G0], %%rax\n\t"
        "andq %%r12, %%rax\n\t"
        "movq %[G1], %%rcx\n\t"
        "andq %%r12, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rcx, %%r9\n\t"
        "adcq $0, %%r10\n\t"
        "adcq $0, %%r11\n\t"
        /* ===== STORE ===== */
        "movq %%r8, %[r0]\n\t"
        "movq %%r9, %[r1]\n\t"
        "movq %%r10, %[r2]\n\t"
        "movq %%r11, %[r3]\n\t"
        : [r0] "=m"(r[0]), [r1] "=m"(r[1]), [r2] "=m"(r[2]), [r3] "=m"(r[3])
        : [a0] "m"(a0),
          [a1] "m"(a1),
          [a2] "m"(a2),
          [a3] "m"(a3),
          [b0] "m"(b0),
          [b1] "m"(b1),
          [b2] "m"(b2),
          [b3] "m"(b3),
          [G0] "m"(TWO_GAMMA_64[0]),
          [G1] "m"(TWO_GAMMA_64[1])
        : "rax", "rbx", "rcx", "rdx", "rsi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "cc", "memory");
}

#else /* TWO_GAMMA_64_LIMBS == 3 */

/*
 * MULX+ADCX+ADOX version (3-limb TWO_GAMMA_64): identical schoolbook,
 * 3-MULX fold steps with extra fold iteration. Requires BMI2 + ADX.
 */
static RANSHAW_FORCE_INLINE void fq64_sq(uint64_t r[4], const uint64_t a[4])
{
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];

    __asm__ __volatile__(
        /* ===== 4×4 SCHOOLBOOK (16 MULX) — identical to 2-limb ===== */
        /* Row 0: a[0] × a[0..3] → w[0..4] = r8..r12 */
        "movq %[a0], %%rdx\n\t"
        "mulxq %[a0], %%r8, %%r9\n\t"
        "mulxq %[a1], %%rax, %%r10\n\t"
        "addq %%rax, %%r9\n\t"
        "mulxq %[a2], %%rax, %%r11\n\t"
        "adcq %%rax, %%r10\n\t"
        "mulxq %[a3], %%rax, %%r12\n\t"
        "adcq %%rax, %%r11\n\t"
        "adcq $0, %%r12\n\t"
        /* Row 1: a[1] × a[0..3], ADCX+ADOX into w[1..5] */
        "movq %[a1], %%rdx\n\t"
        "xorl %%esi, %%esi\n\t"
        "mulxq %[a0], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r9\n\t"
        "adoxq %%rcx, %%r10\n\t"
        "mulxq %[a1], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "mulxq %[a2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rcx, %%r12\n\t"
        "mulxq %[a3], %%rax, %%r13\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rsi, %%r13\n\t"
        "adcxq %%rsi, %%r13\n\t"
        /* Row 2: a[2] × a[0..3] into w[2..6] */
        "movq %[a2], %%rdx\n\t"
        "xorl %%esi, %%esi\n\t"
        "mulxq %[a0], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "mulxq %[a1], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rcx, %%r12\n\t"
        "mulxq %[a2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rcx, %%r13\n\t"
        "mulxq %[a3], %%rax, %%r14\n\t"
        "adcxq %%rax, %%r13\n\t"
        "adoxq %%rsi, %%r14\n\t"
        "adcxq %%rsi, %%r14\n\t"
        /* Row 3: a[3] × a[0..3] into w[3..7] */
        "movq %[a3], %%rdx\n\t"
        "xorl %%esi, %%esi\n\t"
        "mulxq %[a0], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rcx, %%r12\n\t"
        "mulxq %[a1], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rcx, %%r13\n\t"
        "mulxq %[a2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r13\n\t"
        "adoxq %%rcx, %%r14\n\t"
        "mulxq %[a3], %%rax, %%r15\n\t"
        "adcxq %%rax, %%r14\n\t"
        "adoxq %%rsi, %%r15\n\t"
        "adcxq %%rsi, %%r15\n\t"

        /* w[0..7] = r8..r15 */

        /* ===== FIRST CRANDALL FOLD: w[4..7] × [G0,G1,G2] ===== */

        /* Fold w4 (r12) → positions [0,1,2,3] */
        "movq %%r12, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"

        /* Fold w5 (r13) → positions [1,2,3,4] */
        "movq %%r13, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r11\n\t"
        "movl $0, %%r12d\n\t"
        "adcq %%rcx, %%r12\n\t"

        /* Fold w6 (r14) → positions [2,3,4,5] */
        "movq %%r14, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r11\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r12\n\t"
        "movl $0, %%r13d\n\t"
        "adcq %%rcx, %%r13\n\t"

        /* Fold w7 (r15) → positions [3,4,5,6] */
        "movq %%r15, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r11\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r12\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r13\n\t"
        "movl $0, %%r14d\n\t"
        "adcq %%rcx, %%r14\n\t"

        /* After 1st fold: w[0..3]=r8..r11, overflow: r12, r13, r14 */

        /* ===== SECOND CRANDALL FOLD: overflow[4..6] × [G0,G1,G2] ===== */

        /* Fold r12 → positions [0,1,2,3] */
        "movq %%r12, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"

        /* Fold r13 → positions [1,2,3,4] */
        "movq %%r13, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r11\n\t"
        "movl $0, %%r12d\n\t"
        "adcq %%rcx, %%r12\n\t"

        /* Fold r14 → positions [2,3,4,5] */
        "movq %%r14, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r11\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r12\n\t"
        "movl $0, %%r13d\n\t"
        "adcq %%rcx, %%r13\n\t"

        /* After 2nd fold: w[0..3]=r8..r11, overflow: r12, r13 */

        /* ===== THIRD CRANDALL FOLD: overflow[4..5] × [G0,G1,G2] ===== */

        /* Fold r12 → positions [0,1,2,3] */
        "movq %%r12, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"

        /* Fold r13 → positions [1,2,3,4] */
        "movq %%r13, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r11\n\t"
        "movl $0, %%r12d\n\t"
        "adcq %%rcx, %%r12\n\t"

        /* After 3rd fold: w[0..3]=r8..r11, overflow: r12 (small) */

        /* ===== FOURTH + FIFTH FOLD (CT conditional adds) ===== */
        /* r12 is 0 or small: add r12 * [G0,G1,G2] via mask */
        "negq %%r12\n\t"
        "movq %[G0], %%rax\n\t"
        "andq %%r12, %%rax\n\t"
        "movq %[G1], %%rcx\n\t"
        "andq %%r12, %%rcx\n\t"
        "movq %[G2], %%rdi\n\t"
        "andq %%r12, %%rdi\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rcx, %%r9\n\t"
        "adcq %%rdi, %%r10\n\t"
        "adcq $0, %%r11\n\t"
        /* Fifth fold: carry from above */
        "movl $0, %%r12d\n\t"
        "adcq $0, %%r12\n\t"
        "negq %%r12\n\t"
        "movq %[G0], %%rax\n\t"
        "andq %%r12, %%rax\n\t"
        "movq %[G1], %%rcx\n\t"
        "andq %%r12, %%rcx\n\t"
        "movq %[G2], %%rdi\n\t"
        "andq %%r12, %%rdi\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rcx, %%r9\n\t"
        "adcq %%rdi, %%r10\n\t"
        "adcq $0, %%r11\n\t"

        /* ===== STORE ===== */
        "movq %%r8, %[r0]\n\t"
        "movq %%r9, %[r1]\n\t"
        "movq %%r10, %[r2]\n\t"
        "movq %%r11, %[r3]\n\t"

        : [r0] "=m"(r[0]), [r1] "=m"(r[1]), [r2] "=m"(r[2]), [r3] "=m"(r[3])
        : [a0] "m"(a0),
          [a1] "m"(a1),
          [a2] "m"(a2),
          [a3] "m"(a3),
          [G0] "m"(TWO_GAMMA_64[0]),
          [G1] "m"(TWO_GAMMA_64[1]),
          [G2] "m"(TWO_GAMMA_64[2])
        : "rax",
          "rbx",
          "rcx",
          "rdx",
          "rsi",
          "rdi",
          "r8",
          "r9",
          "r10",
          "r11",
          "r12",
          "r13",
          "r14",
          "r15",
          "cc",
          "memory");
}

/*
 * 4×64 Crandall multiply (3-limb TWO_GAMMA_64): a[0..3] × b[0..3] → r[0..3] (mod q).
 */
static RANSHAW_FORCE_INLINE void fq64_mul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4])
{
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    uint64_t b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];

    __asm__ __volatile__(
        /* ===== 4×4 SCHOOLBOOK (16 MULX) — identical to 2-limb ===== */
        "movq %[a0], %%rdx\n\t"
        "mulxq %[b0], %%r8, %%r9\n\t"
        "mulxq %[b1], %%rax, %%r10\n\t"
        "addq %%rax, %%r9\n\t"
        "mulxq %[b2], %%rax, %%r11\n\t"
        "adcq %%rax, %%r10\n\t"
        "mulxq %[b3], %%rax, %%r12\n\t"
        "adcq %%rax, %%r11\n\t"
        "adcq $0, %%r12\n\t"
        "movq %[a1], %%rdx\n\t"
        "xorl %%esi, %%esi\n\t"
        "mulxq %[b0], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r9\n\t"
        "adoxq %%rcx, %%r10\n\t"
        "mulxq %[b1], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "mulxq %[b2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rcx, %%r12\n\t"
        "mulxq %[b3], %%rax, %%r13\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rsi, %%r13\n\t"
        "adcxq %%rsi, %%r13\n\t"
        "movq %[a2], %%rdx\n\t"
        "xorl %%esi, %%esi\n\t"
        "mulxq %[b0], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "mulxq %[b1], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rcx, %%r12\n\t"
        "mulxq %[b2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rcx, %%r13\n\t"
        "mulxq %[b3], %%rax, %%r14\n\t"
        "adcxq %%rax, %%r13\n\t"
        "adoxq %%rsi, %%r14\n\t"
        "adcxq %%rsi, %%r14\n\t"
        "movq %[a3], %%rdx\n\t"
        "xorl %%esi, %%esi\n\t"
        "mulxq %[b0], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r11\n\t"
        "adoxq %%rcx, %%r12\n\t"
        "mulxq %[b1], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r12\n\t"
        "adoxq %%rcx, %%r13\n\t"
        "mulxq %[b2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r13\n\t"
        "adoxq %%rcx, %%r14\n\t"
        "mulxq %[b3], %%rax, %%r15\n\t"
        "adcxq %%rax, %%r14\n\t"
        "adoxq %%rsi, %%r15\n\t"
        "adcxq %%rsi, %%r15\n\t"

        /* ===== FIRST CRANDALL FOLD: w[4..7] × [G0,G1,G2] ===== */

        /* Fold w4 (r12) → positions [0,1,2,3] */
        "movq %%r12, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"

        /* Fold w5 (r13) → positions [1,2,3,4] */
        "movq %%r13, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r11\n\t"
        "movl $0, %%r12d\n\t"
        "adcq %%rcx, %%r12\n\t"

        /* Fold w6 (r14) → positions [2,3,4,5] */
        "movq %%r14, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r11\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r12\n\t"
        "movl $0, %%r13d\n\t"
        "adcq %%rcx, %%r13\n\t"

        /* Fold w7 (r15) → positions [3,4,5,6] */
        "movq %%r15, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r11\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r12\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r13\n\t"
        "movl $0, %%r14d\n\t"
        "adcq %%rcx, %%r14\n\t"

        /* After 1st fold: w[0..3]=r8..r11, overflow: r12, r13, r14 */

        /* ===== SECOND CRANDALL FOLD: overflow[4..6] × [G0,G1,G2] ===== */

        /* Fold r12 → positions [0,1,2,3] */
        "movq %%r12, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"

        /* Fold r13 → positions [1,2,3,4] */
        "movq %%r13, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r11\n\t"
        "movl $0, %%r12d\n\t"
        "adcq %%rcx, %%r12\n\t"

        /* Fold r14 → positions [2,3,4,5] */
        "movq %%r14, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r11\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r12\n\t"
        "movl $0, %%r13d\n\t"
        "adcq %%rcx, %%r13\n\t"

        /* After 2nd fold: w[0..3]=r8..r11, overflow: r12, r13 */

        /* ===== THIRD CRANDALL FOLD: overflow[4..5] × [G0,G1,G2] ===== */

        /* Fold r12 → positions [0,1,2,3] */
        "movq %%r12, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"

        /* Fold r13 → positions [1,2,3,4] */
        "movq %%r13, %%rdx\n\t"
        "mulxq %[G0], %%rax, %%rcx\n\t"
        "addq %%rax, %%r9\n\t"
        "adcq $0, %%rcx\n\t"
        "mulxq %[G1], %%rax, %%rbx\n\t"
        "addq %%rcx, %%rax\n\t"
        "adcq $0, %%rbx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq $0, %%rbx\n\t"
        "mulxq %[G2], %%rax, %%rcx\n\t"
        "addq %%rbx, %%rax\n\t"
        "adcq $0, %%rcx\n\t"
        "addq %%rax, %%r11\n\t"
        "movl $0, %%r12d\n\t"
        "adcq %%rcx, %%r12\n\t"

        /* After 3rd fold: w[0..3]=r8..r11, overflow: r12 (small) */

        /* ===== FOURTH + FIFTH FOLD (CT conditional adds) ===== */
        "negq %%r12\n\t"
        "movq %[G0], %%rax\n\t"
        "andq %%r12, %%rax\n\t"
        "movq %[G1], %%rcx\n\t"
        "andq %%r12, %%rcx\n\t"
        "movq %[G2], %%rdi\n\t"
        "andq %%r12, %%rdi\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rcx, %%r9\n\t"
        "adcq %%rdi, %%r10\n\t"
        "adcq $0, %%r11\n\t"
        "movl $0, %%r12d\n\t"
        "adcq $0, %%r12\n\t"
        "negq %%r12\n\t"
        "movq %[G0], %%rax\n\t"
        "andq %%r12, %%rax\n\t"
        "movq %[G1], %%rcx\n\t"
        "andq %%r12, %%rcx\n\t"
        "movq %[G2], %%rdi\n\t"
        "andq %%r12, %%rdi\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rcx, %%r9\n\t"
        "adcq %%rdi, %%r10\n\t"
        "adcq $0, %%r11\n\t"

        /* ===== STORE ===== */
        "movq %%r8, %[r0]\n\t"
        "movq %%r9, %[r1]\n\t"
        "movq %%r10, %[r2]\n\t"
        "movq %%r11, %[r3]\n\t"

        : [r0] "=m"(r[0]), [r1] "=m"(r[1]), [r2] "=m"(r[2]), [r3] "=m"(r[3])
        : [a0] "m"(a0),
          [a1] "m"(a1),
          [a2] "m"(a2),
          [a3] "m"(a3),
          [b0] "m"(b0),
          [b1] "m"(b1),
          [b2] "m"(b2),
          [b3] "m"(b3),
          [G0] "m"(TWO_GAMMA_64[0]),
          [G1] "m"(TWO_GAMMA_64[1]),
          [G2] "m"(TWO_GAMMA_64[2])
        : "rax",
          "rbx",
          "rcx",
          "rdx",
          "rsi",
          "rdi",
          "r8",
          "r9",
          "r10",
          "r11",
          "r12",
          "r13",
          "r14",
          "r15",
          "cc",
          "memory");
}

#endif /* TWO_GAMMA_64_LIMBS */

#else
/*
 * C fallback: uses __uint128_t row-by-row accumulation.
 */
static RANSHAW_FORCE_INLINE void fq64_sq(uint64_t r[4], const uint64_t a[4])
{
    uint64_t w[8] = {0};
    __uint128_t acc;
    uint64_t carry;

    /* Row 0 */
    acc = (__uint128_t)a[0] * a[0];
    w[0] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[0] * a[1] + carry;
    w[1] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[0] * a[2] + carry;
    w[2] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[0] * a[3] + carry;
    w[3] = (uint64_t)acc;
    w[4] = (uint64_t)(acc >> 64);

    /* Row 1 */
    acc = (__uint128_t)a[1] * a[0] + w[1];
    w[1] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[1] * a[1] + w[2] + carry;
    w[2] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[1] * a[2] + w[3] + carry;
    w[3] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[1] * a[3] + w[4] + carry;
    w[4] = (uint64_t)acc;
    w[5] = (uint64_t)(acc >> 64);

    /* Row 2 */
    acc = (__uint128_t)a[2] * a[0] + w[2];
    w[2] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[2] * a[1] + w[3] + carry;
    w[3] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[2] * a[2] + w[4] + carry;
    w[4] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[2] * a[3] + w[5] + carry;
    w[5] = (uint64_t)acc;
    w[6] = (uint64_t)(acc >> 64);

    /* Row 3 */
    acc = (__uint128_t)a[3] * a[0] + w[3];
    w[3] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[3] * a[1] + w[4] + carry;
    w[4] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[3] * a[2] + w[5] + carry;
    w[5] = (uint64_t)acc;
    carry = (uint64_t)(acc >> 64);
    acc = (__uint128_t)a[3] * a[3] + w[6] + carry;
    w[6] = (uint64_t)acc;
    w[7] = (uint64_t)(acc >> 64);

    /* Iterative Crandall fold (same as fq64_mul_c) */
    for (int fold = 0; fold < 3; fold++)
    {
        uint64_t over[4];
        for (int i = 0; i < 4; i++)
        {
            over[i] = w[4 + i];
            w[4 + i] = 0;
        }
        for (int i = 0; i < 4; i++)
        {
            carry = 0;
            for (int j = 0; j < TWO_GAMMA_64_LIMBS; j++)
            {
                acc = (__uint128_t)over[i] * TWO_GAMMA_64[j] + w[i + j] + carry;
                w[i + j] = (uint64_t)acc;
                carry = (uint64_t)(acc >> 64);
            }
            /* CT: no early exit */
            for (int k = i + TWO_GAMMA_64_LIMBS; k < 8; k++)
            {
                w[k] += carry;
                carry = (w[k] < carry) ? 1 : 0;
            }
        }
    }

    /* Final CT correction */
    {
        uint64_t mask = -(uint64_t)(w[4] & 1);
        carry = 0;
        for (int j = 0; j < TWO_GAMMA_64_LIMBS; j++)
        {
            acc = (__uint128_t)w[j] + (TWO_GAMMA_64[j] & mask) + carry;
            w[j] = (uint64_t)acc;
            carry = (uint64_t)(acc >> 64);
        }
        for (int j = TWO_GAMMA_64_LIMBS; j < 4; j++)
        {
            w[j] += carry;
            carry = (w[j] < carry) ? 1 : 0;
        }
    }

    r[0] = w[0];
    r[1] = w[1];
    r[2] = w[2];
    r[3] = w[3];
}

#endif /* __ADX__ */

#endif /* __GNUC__ && __BMI2__ && RANSHAW_HAVE_INT128 (4×64 chain helpers) */

/*
 * 5×51 radix-2^51 multiplication and squaring.
 *
 * When BMI2+ADX are available: pack 5×51 → 4×64, MULX schoolbook + Crandall fold,
 * unpack 4×64 → 5×51. Otherwise: 5×5 column-accumulation schoolbook + 3-stage fold.
 */

#if defined(__GNUC__) && defined(__BMI2__) && RANSHAW_HAVE_INT128
#define FQ51_HAVE_ADX_MUL 1

static RANSHAW_FORCE_INLINE void fq51_mul_inline(fq_fe h, const fq_fe f, const fq_fe g)
{
    uint64_t a[4], b[4], out[4];
    fq51_normalize_and_pack(a, f);
    fq51_normalize_and_pack(b, g);
#if defined(__ADX__)
    fq64_mul(out, a, b);
#else
    fq64_mul_c(out, a, b);
#endif
    fq64_to_fq51(h, out);

    /* Post-normalize: carry chain + gamma fold to match 5×51 limb profile */
    const uint64_t M = FQ51_MASK;
    uint64_t c = h[0] >> 51;
    h[0] &= M;
    h[1] += c;
    c = h[1] >> 51;
    h[1] &= M;
    h[2] += c;
    c = h[2] >> 51;
    h[2] &= M;
    h[3] += c;
    c = h[3] >> 51;
    h[3] &= M;
    h[4] += c;
    c = h[4] >> 51;
    h[4] &= M;
    h[0] += c * GAMMA_51[0];
    h[1] += c * GAMMA_51[1];
    h[2] += c * GAMMA_51[2];
    c = h[0] >> 51;
    h[0] &= M;
    h[1] += c;
    c = h[1] >> 51;
    h[1] &= M;
    h[2] += c;
}

static RANSHAW_FORCE_INLINE void fq51_sq_inline(fq_fe h, const fq_fe f)
{
    uint64_t a[4], out[4];
    fq51_normalize_and_pack(a, f);
    fq64_sq(out, a);
    fq64_to_fq51(h, out);

    /* Post-normalize: carry chain + gamma fold to match 5×51 limb profile */
    const uint64_t M = FQ51_MASK;
    uint64_t c = h[0] >> 51;
    h[0] &= M;
    h[1] += c;
    c = h[1] >> 51;
    h[1] &= M;
    h[2] += c;
    c = h[2] >> 51;
    h[2] &= M;
    h[3] += c;
    c = h[3] >> 51;
    h[3] &= M;
    h[4] += c;
    c = h[4] >> 51;
    h[4] &= M;
    for (int j = 0; j < GAMMA_51_LIMBS; j++)
        h[j] += c * GAMMA_51[j];
    for (int j = 0; j < GAMMA_51_LIMBS - 1; j++)
    {
        c = h[j] >> 51;
        h[j] &= M;
        h[j + 1] += c;
    }
}

#endif

#if RANSHAW_HAVE_INT128

#if !defined(FQ51_HAVE_ADX_MUL)
static RANSHAW_FORCE_INLINE void fq51_mul_inline(fq_fe h, const fq_fe f, const fq_fe g)
{
    uint64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    uint64_t g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3], g4 = g[4];

    /* === 5×5 column-accumulation schoolbook (25 mul64) === */
    ranshaw_uint128 t0 = mul64(f0, g0);
    ranshaw_uint128 t1 = mul64(f0, g1) + mul64(f1, g0);
    ranshaw_uint128 t2 = mul64(f0, g2) + mul64(f1, g1) + mul64(f2, g0);
    ranshaw_uint128 t3 = mul64(f0, g3) + mul64(f1, g2) + mul64(f2, g1) + mul64(f3, g0);
    ranshaw_uint128 t4 = mul64(f0, g4) + mul64(f1, g3) + mul64(f2, g2) + mul64(f3, g1) + mul64(f4, g0);
    ranshaw_uint128 t5 = mul64(f1, g4) + mul64(f2, g3) + mul64(f3, g2) + mul64(f4, g1);
    ranshaw_uint128 t6 = mul64(f2, g4) + mul64(f3, g3) + mul64(f4, g2);
    ranshaw_uint128 t7 = mul64(f3, g4) + mul64(f4, g3);
    ranshaw_uint128 t8 = mul64(f4, g4);

    /* === Carry chain: extract 51-bit limbs from t0..t8 === */
    uint64_t r0, r1, r2, r3, r4, r5, r6, r7, r8, c;

    r0 = (uint64_t)t0 & FQ51_MASK;
    c = (uint64_t)(t0 >> 51);
    t1 += c;
    r1 = (uint64_t)t1 & FQ51_MASK;
    c = (uint64_t)(t1 >> 51);
    t2 += c;
    r2 = (uint64_t)t2 & FQ51_MASK;
    c = (uint64_t)(t2 >> 51);
    t3 += c;
    r3 = (uint64_t)t3 & FQ51_MASK;
    c = (uint64_t)(t3 >> 51);
    t4 += c;
    r4 = (uint64_t)t4 & FQ51_MASK;
    c = (uint64_t)(t4 >> 51);
    t5 += c;
    r5 = (uint64_t)t5 & FQ51_MASK;
    c = (uint64_t)(t5 >> 51);
    t6 += c;
    r6 = (uint64_t)t6 & FQ51_MASK;
    c = (uint64_t)(t6 >> 51);
    t7 += c;
    r7 = (uint64_t)t7 & FQ51_MASK;
    c = (uint64_t)(t7 >> 51);
    t8 += c;
    r8 = (uint64_t)t8 & FQ51_MASK;
    uint64_t c9 = (uint64_t)(t8 >> 51);

    /*
     * === First Crandall fold ===
     * [r5,r6,r7,r8,c9] × GAMMA_51[0..GAMMA_51_LIMBS-1] → positions 0..(4+GAMMA_51_LIMBS-1)
     * 2^(51*k) for k≥5: 2^(51*k) = 2^(51*(k-5)) * 2^255 ≡ 2^(51*(k-5)) * gamma
     */
    ranshaw_uint128 p[4 + GAMMA_51_LIMBS] = {0};
    {
        const uint64_t overflow[5] = {r5, r6, r7, r8, c9};
        for (int k = 0; k < 5; k++)
            for (int j = 0; j < GAMMA_51_LIMBS; j++)
                p[k + j] += mul64(overflow[k], GAMMA_51[j]);
    }

    /* Add fold result to low limbs */
    p[0] += r0;
    p[1] += r1;
    p[2] += r2;
    p[3] += r3;
    p[4] += r4;

    /* Carry chain on p[0..4+GAMMA_51_LIMBS-1] */
    r0 = (uint64_t)p[0] & FQ51_MASK;
    c = (uint64_t)(p[0] >> 51);
    p[1] += c;
    r1 = (uint64_t)p[1] & FQ51_MASK;
    c = (uint64_t)(p[1] >> 51);
    p[2] += c;
    r2 = (uint64_t)p[2] & FQ51_MASK;
    c = (uint64_t)(p[2] >> 51);
    p[3] += c;
    r3 = (uint64_t)p[3] & FQ51_MASK;
    c = (uint64_t)(p[3] >> 51);
    p[4] += c;
    r4 = (uint64_t)p[4] & FQ51_MASK;
    c = (uint64_t)(p[4] >> 51);
    p[5] += c;
    r5 = (uint64_t)p[5] & FQ51_MASK;
    c = (uint64_t)(p[5] >> 51);
    p[6] += c;
    r6 = (uint64_t)p[6] & FQ51_MASK;
    uint64_t c7 = (uint64_t)(p[6] >> 51);

    /*
     * === Second Crandall fold ===
     * [r5,r6,c7] × GAMMA_51[0..GAMMA_51_LIMBS-1] → positions 0..4
     */
    ranshaw_uint128 q[4 + GAMMA_51_LIMBS] = {0};
    {
        const uint64_t overflow2[3] = {r5, r6, c7};
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < GAMMA_51_LIMBS; j++)
                q[k + j] += mul64(overflow2[k], GAMMA_51[j]);
    }

    q[0] += r0;
    q[1] += r1;
    q[2] += r2;
    q[3] += r3;
    q[4] += r4;

    /* Final carry chain with gamma fold of top carry */
    r0 = (uint64_t)q[0] & FQ51_MASK;
    c = (uint64_t)(q[0] >> 51);
    q[1] += c;
    r1 = (uint64_t)q[1] & FQ51_MASK;
    c = (uint64_t)(q[1] >> 51);
    q[2] += c;
    r2 = (uint64_t)q[2] & FQ51_MASK;
    c = (uint64_t)(q[2] >> 51);
    q[3] += c;
    r3 = (uint64_t)q[3] & FQ51_MASK;
    c = (uint64_t)(q[3] >> 51);
    q[4] += c;
    r4 = (uint64_t)q[4] & FQ51_MASK;
    c = (uint64_t)(q[4] >> 51);

    /* Tiny third fold: c is very small (0 or 1 typically) */
    h[0] = r0;
    h[1] = r1;
    h[2] = r2;
    h[3] = r3;
    h[4] = r4;
    for (int j = 0; j < GAMMA_51_LIMBS; j++)
        h[j] += c * GAMMA_51[j];
    c = h[0] >> 51;
    h[0] &= FQ51_MASK;
    h[1] += c;
    c = h[1] >> 51;
    h[1] &= FQ51_MASK;
    h[2] += c;
}
#endif /* !FQ51_HAVE_ADX_MUL */

#if !defined(FQ51_HAVE_ADX_MUL)
static RANSHAW_FORCE_INLINE void fq51_sq_inline(fq_fe h, const fq_fe f)
{
    uint64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];

    uint64_t f0_2 = 2 * f0;
    uint64_t f1_2 = 2 * f1;
    uint64_t f2_2 = 2 * f2;
    uint64_t f3_2 = 2 * f3;

    /* === Squaring: 15 mul64 (5 diagonal + 10 doubled cross) === */
    ranshaw_uint128 t0 = mul64(f0, f0);
    ranshaw_uint128 t1 = mul64(f0_2, f1);
    ranshaw_uint128 t2 = mul64(f0_2, f2) + mul64(f1, f1);
    ranshaw_uint128 t3 = mul64(f0_2, f3) + mul64(f1_2, f2);
    ranshaw_uint128 t4 = mul64(f0_2, f4) + mul64(f1_2, f3) + mul64(f2, f2);
    ranshaw_uint128 t5 = mul64(f1_2, f4) + mul64(f2_2, f3);
    ranshaw_uint128 t6 = mul64(f2_2, f4) + mul64(f3, f3);
    ranshaw_uint128 t7 = mul64(f3_2, f4);
    ranshaw_uint128 t8 = mul64(f4, f4);

    /* === Carry chain: extract 51-bit limbs from t0..t8 === */
    uint64_t r0, r1, r2, r3, r4, r5, r6, r7, r8, c;

    r0 = (uint64_t)t0 & FQ51_MASK;
    c = (uint64_t)(t0 >> 51);
    t1 += c;
    r1 = (uint64_t)t1 & FQ51_MASK;
    c = (uint64_t)(t1 >> 51);
    t2 += c;
    r2 = (uint64_t)t2 & FQ51_MASK;
    c = (uint64_t)(t2 >> 51);
    t3 += c;
    r3 = (uint64_t)t3 & FQ51_MASK;
    c = (uint64_t)(t3 >> 51);
    t4 += c;
    r4 = (uint64_t)t4 & FQ51_MASK;
    c = (uint64_t)(t4 >> 51);
    t5 += c;
    r5 = (uint64_t)t5 & FQ51_MASK;
    c = (uint64_t)(t5 >> 51);
    t6 += c;
    r6 = (uint64_t)t6 & FQ51_MASK;
    c = (uint64_t)(t6 >> 51);
    t7 += c;
    r7 = (uint64_t)t7 & FQ51_MASK;
    c = (uint64_t)(t7 >> 51);
    t8 += c;
    r8 = (uint64_t)t8 & FQ51_MASK;
    uint64_t c9 = (uint64_t)(t8 >> 51);

    /* === First Crandall fold === */
    ranshaw_uint128 p[4 + GAMMA_51_LIMBS] = {0};
    {
        const uint64_t overflow[5] = {r5, r6, r7, r8, c9};
        for (int k = 0; k < 5; k++)
            for (int j = 0; j < GAMMA_51_LIMBS; j++)
                p[k + j] += mul64(overflow[k], GAMMA_51[j]);
    }

    p[0] += r0;
    p[1] += r1;
    p[2] += r2;
    p[3] += r3;
    p[4] += r4;

    r0 = (uint64_t)p[0] & FQ51_MASK;
    c = (uint64_t)(p[0] >> 51);
    p[1] += c;
    r1 = (uint64_t)p[1] & FQ51_MASK;
    c = (uint64_t)(p[1] >> 51);
    p[2] += c;
    r2 = (uint64_t)p[2] & FQ51_MASK;
    c = (uint64_t)(p[2] >> 51);
    p[3] += c;
    r3 = (uint64_t)p[3] & FQ51_MASK;
    c = (uint64_t)(p[3] >> 51);
    p[4] += c;
    r4 = (uint64_t)p[4] & FQ51_MASK;
    c = (uint64_t)(p[4] >> 51);
    p[5] += c;
    r5 = (uint64_t)p[5] & FQ51_MASK;
    c = (uint64_t)(p[5] >> 51);
    p[6] += c;
    r6 = (uint64_t)p[6] & FQ51_MASK;
    uint64_t c7 = (uint64_t)(p[6] >> 51);

    /* === Second Crandall fold === */
    ranshaw_uint128 q[4 + GAMMA_51_LIMBS] = {0};
    {
        const uint64_t overflow2[3] = {r5, r6, c7};
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < GAMMA_51_LIMBS; j++)
                q[k + j] += mul64(overflow2[k], GAMMA_51[j]);
    }

    q[0] += r0;
    q[1] += r1;
    q[2] += r2;
    q[3] += r3;
    q[4] += r4;

    r0 = (uint64_t)q[0] & FQ51_MASK;
    c = (uint64_t)(q[0] >> 51);
    q[1] += c;
    r1 = (uint64_t)q[1] & FQ51_MASK;
    c = (uint64_t)(q[1] >> 51);
    q[2] += c;
    r2 = (uint64_t)q[2] & FQ51_MASK;
    c = (uint64_t)(q[2] >> 51);
    q[3] += c;
    r3 = (uint64_t)q[3] & FQ51_MASK;
    c = (uint64_t)(q[3] >> 51);
    q[4] += c;
    r4 = (uint64_t)q[4] & FQ51_MASK;
    c = (uint64_t)(q[4] >> 51);

    h[0] = r0;
    h[1] = r1;
    h[2] = r2;
    h[3] = r3;
    h[4] = r4;
    for (int j = 0; j < GAMMA_51_LIMBS; j++)
        h[j] += c * GAMMA_51[j];
    c = h[0] >> 51;
    h[0] &= FQ51_MASK;
    h[1] += c;
    c = h[1] >> 51;
    h[1] &= FQ51_MASK;
    h[2] += c;
}
#endif /* !FQ51_HAVE_ADX_MUL */

#elif RANSHAW_HAVE_UMUL128

static RANSHAW_FORCE_INLINE void fq51_mul_inline(fq_fe h, const fq_fe f, const fq_fe g)
{
    uint64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    uint64_t g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3], g4 = g[4];

    /* === 5×5 column-accumulation schoolbook (25 mul64) === */
    ranshaw_uint128_emu t0 = mul64(f0, g0);
    ranshaw_uint128_emu t1 = mul64(f0, g1);
    t1 += mul64(f1, g0);
    ranshaw_uint128_emu t2 = mul64(f0, g2);
    t2 += mul64(f1, g1);
    t2 += mul64(f2, g0);
    ranshaw_uint128_emu t3 = mul64(f0, g3);
    t3 += mul64(f1, g2);
    t3 += mul64(f2, g1);
    t3 += mul64(f3, g0);
    ranshaw_uint128_emu t4 = mul64(f0, g4);
    t4 += mul64(f1, g3);
    t4 += mul64(f2, g2);
    t4 += mul64(f3, g1);
    t4 += mul64(f4, g0);
    ranshaw_uint128_emu t5 = mul64(f1, g4);
    t5 += mul64(f2, g3);
    t5 += mul64(f3, g2);
    t5 += mul64(f4, g1);
    ranshaw_uint128_emu t6 = mul64(f2, g4);
    t6 += mul64(f3, g3);
    t6 += mul64(f4, g2);
    ranshaw_uint128_emu t7 = mul64(f3, g4);
    t7 += mul64(f4, g3);
    ranshaw_uint128_emu t8 = mul64(f4, g4);

    /* === Carry chain: extract 51-bit limbs === */
    uint64_t r0, r1, r2, r3, r4, r5, r6, r7, r8, c;

    r0 = lo128(t0) & FQ51_MASK;
    c = shr128(t0, 51);
    t1 += c;
    r1 = lo128(t1) & FQ51_MASK;
    c = shr128(t1, 51);
    t2 += c;
    r2 = lo128(t2) & FQ51_MASK;
    c = shr128(t2, 51);
    t3 += c;
    r3 = lo128(t3) & FQ51_MASK;
    c = shr128(t3, 51);
    t4 += c;
    r4 = lo128(t4) & FQ51_MASK;
    c = shr128(t4, 51);
    t5 += c;
    r5 = lo128(t5) & FQ51_MASK;
    c = shr128(t5, 51);
    t6 += c;
    r6 = lo128(t6) & FQ51_MASK;
    c = shr128(t6, 51);
    t7 += c;
    r7 = lo128(t7) & FQ51_MASK;
    c = shr128(t7, 51);
    t8 += c;
    r8 = lo128(t8) & FQ51_MASK;
    uint64_t c9 = shr128(t8, 51);

    /* === First Crandall fold === */
    ranshaw_uint128_emu p[4 + GAMMA_51_LIMBS] = {};
    {
        const uint64_t overflow[5] = {r5, r6, r7, r8, c9};
        for (int k = 0; k < 5; k++)
            for (int j = 0; j < GAMMA_51_LIMBS; j++)
                p[k + j] += mul64(overflow[k], GAMMA_51[j]);
    }

    p[0] += r0;
    p[1] += r1;
    p[2] += r2;
    p[3] += r3;
    p[4] += r4;

    r0 = lo128(p[0]) & FQ51_MASK;
    c = shr128(p[0], 51);
    p[1] += c;
    r1 = lo128(p[1]) & FQ51_MASK;
    c = shr128(p[1], 51);
    p[2] += c;
    r2 = lo128(p[2]) & FQ51_MASK;
    c = shr128(p[2], 51);
    p[3] += c;
    r3 = lo128(p[3]) & FQ51_MASK;
    c = shr128(p[3], 51);
    p[4] += c;
    r4 = lo128(p[4]) & FQ51_MASK;
    c = shr128(p[4], 51);
    p[5] += c;
    r5 = lo128(p[5]) & FQ51_MASK;
    c = shr128(p[5], 51);
    p[6] += c;
    r6 = lo128(p[6]) & FQ51_MASK;
    uint64_t c7 = shr128(p[6], 51);

    /* === Second Crandall fold === */
    ranshaw_uint128_emu q[4 + GAMMA_51_LIMBS] = {};
    {
        const uint64_t overflow2[3] = {r5, r6, c7};
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < GAMMA_51_LIMBS; j++)
                q[k + j] += mul64(overflow2[k], GAMMA_51[j]);
    }

    q[0] += r0;
    q[1] += r1;
    q[2] += r2;
    q[3] += r3;
    q[4] += r4;

    r0 = lo128(q[0]) & FQ51_MASK;
    c = shr128(q[0], 51);
    q[1] += c;
    r1 = lo128(q[1]) & FQ51_MASK;
    c = shr128(q[1], 51);
    q[2] += c;
    r2 = lo128(q[2]) & FQ51_MASK;
    c = shr128(q[2], 51);
    q[3] += c;
    r3 = lo128(q[3]) & FQ51_MASK;
    c = shr128(q[3], 51);
    q[4] += c;
    r4 = lo128(q[4]) & FQ51_MASK;
    c = shr128(q[4], 51);

    h[0] = r0;
    h[1] = r1;
    h[2] = r2;
    h[3] = r3;
    h[4] = r4;
    for (int j = 0; j < GAMMA_51_LIMBS; j++)
        h[j] += c * GAMMA_51[j];
    c = h[0] >> 51;
    h[0] &= FQ51_MASK;
    h[1] += c;
    c = h[1] >> 51;
    h[1] &= FQ51_MASK;
    h[2] += c;
}

static RANSHAW_FORCE_INLINE void fq51_sq_inline(fq_fe h, const fq_fe f)
{
    uint64_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];

    uint64_t f0_2 = 2 * f0;
    uint64_t f1_2 = 2 * f1;
    uint64_t f2_2 = 2 * f2;
    uint64_t f3_2 = 2 * f3;

    /* === Squaring: 15 mul64 === */
    ranshaw_uint128_emu t0 = mul64(f0, f0);
    ranshaw_uint128_emu t1 = mul64(f0_2, f1);
    ranshaw_uint128_emu t2 = mul64(f0_2, f2);
    t2 += mul64(f1, f1);
    ranshaw_uint128_emu t3 = mul64(f0_2, f3);
    t3 += mul64(f1_2, f2);
    ranshaw_uint128_emu t4 = mul64(f0_2, f4);
    t4 += mul64(f1_2, f3);
    t4 += mul64(f2, f2);
    ranshaw_uint128_emu t5 = mul64(f1_2, f4);
    t5 += mul64(f2_2, f3);
    ranshaw_uint128_emu t6 = mul64(f2_2, f4);
    t6 += mul64(f3, f3);
    ranshaw_uint128_emu t7 = mul64(f3_2, f4);
    ranshaw_uint128_emu t8 = mul64(f4, f4);

    /* === Carry chain === */
    uint64_t r0, r1, r2, r3, r4, r5, r6, r7, r8, c;

    r0 = lo128(t0) & FQ51_MASK;
    c = shr128(t0, 51);
    t1 += c;
    r1 = lo128(t1) & FQ51_MASK;
    c = shr128(t1, 51);
    t2 += c;
    r2 = lo128(t2) & FQ51_MASK;
    c = shr128(t2, 51);
    t3 += c;
    r3 = lo128(t3) & FQ51_MASK;
    c = shr128(t3, 51);
    t4 += c;
    r4 = lo128(t4) & FQ51_MASK;
    c = shr128(t4, 51);
    t5 += c;
    r5 = lo128(t5) & FQ51_MASK;
    c = shr128(t5, 51);
    t6 += c;
    r6 = lo128(t6) & FQ51_MASK;
    c = shr128(t6, 51);
    t7 += c;
    r7 = lo128(t7) & FQ51_MASK;
    c = shr128(t7, 51);
    t8 += c;
    r8 = lo128(t8) & FQ51_MASK;
    uint64_t c9 = shr128(t8, 51);

    /* === First Crandall fold === */
    ranshaw_uint128_emu p[4 + GAMMA_51_LIMBS] = {};
    {
        const uint64_t overflow[5] = {r5, r6, r7, r8, c9};
        for (int k = 0; k < 5; k++)
            for (int j = 0; j < GAMMA_51_LIMBS; j++)
                p[k + j] += mul64(overflow[k], GAMMA_51[j]);
    }

    p[0] += r0;
    p[1] += r1;
    p[2] += r2;
    p[3] += r3;
    p[4] += r4;

    r0 = lo128(p[0]) & FQ51_MASK;
    c = shr128(p[0], 51);
    p[1] += c;
    r1 = lo128(p[1]) & FQ51_MASK;
    c = shr128(p[1], 51);
    p[2] += c;
    r2 = lo128(p[2]) & FQ51_MASK;
    c = shr128(p[2], 51);
    p[3] += c;
    r3 = lo128(p[3]) & FQ51_MASK;
    c = shr128(p[3], 51);
    p[4] += c;
    r4 = lo128(p[4]) & FQ51_MASK;
    c = shr128(p[4], 51);
    p[5] += c;
    r5 = lo128(p[5]) & FQ51_MASK;
    c = shr128(p[5], 51);
    p[6] += c;
    r6 = lo128(p[6]) & FQ51_MASK;
    uint64_t c7 = shr128(p[6], 51);

    /* === Second Crandall fold === */
    ranshaw_uint128_emu q[4 + GAMMA_51_LIMBS] = {};
    {
        const uint64_t overflow2[3] = {r5, r6, c7};
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < GAMMA_51_LIMBS; j++)
                q[k + j] += mul64(overflow2[k], GAMMA_51[j]);
    }

    q[0] += r0;
    q[1] += r1;
    q[2] += r2;
    q[3] += r3;
    q[4] += r4;

    r0 = lo128(q[0]) & FQ51_MASK;
    c = shr128(q[0], 51);
    q[1] += c;
    r1 = lo128(q[1]) & FQ51_MASK;
    c = shr128(q[1], 51);
    q[2] += c;
    r2 = lo128(q[2]) & FQ51_MASK;
    c = shr128(q[2], 51);
    q[3] += c;
    r3 = lo128(q[3]) & FQ51_MASK;
    c = shr128(q[3], 51);
    q[4] += c;
    r4 = lo128(q[4]) & FQ51_MASK;
    c = shr128(q[4], 51);

    h[0] = r0;
    h[1] = r1;
    h[2] = r2;
    h[3] = r3;
    h[4] = r4;
    for (int j = 0; j < GAMMA_51_LIMBS; j++)
        h[j] += c * GAMMA_51[j];
    c = h[0] >> 51;
    h[0] &= FQ51_MASK;
    h[1] += c;
    c = h[1] >> 51;
    h[1] &= FQ51_MASK;
    h[2] += c;
}

#endif /* RANSHAW_HAVE_INT128 / RANSHAW_HAVE_UMUL128 */

#endif // RANSHAW_X64_FQ51_INLINE_H
