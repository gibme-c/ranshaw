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
 * @file fq_divsteps.h
 * @brief x64 (radix-2^51) implementation of F_q divsteps inversion with Crandall reduction.
 */

#ifndef RANSHAW_X64_FQ_DIVSTEPS_H
#define RANSHAW_X64_FQ_DIVSTEPS_H

/*
 * Bernstein-Yang safegcd/divsteps modular inversion for F_q.
 *
 * q = 2^255 - gamma (Crandall prime, gamma ~ 2^127)
 *
 * Based on "Fast constant-time gcd computation and modular inversion"
 * (Bernstein & Yang, 2019). Adapted from libsecp256k1's modinv64 implementation.
 *
 * Representation: 5 x int64_t limbs in radix-2^62 ("signed62").
 * Each limb nominally in [0, 2^62), but intermediate values may be signed.
 *
 * Constant-time: fixed iteration count, no secret-dependent branches or memory access.
 */

#include "ranshaw_platform.h"
#include "x64/fq51.h"

#include <cstdint>

/* ------------------------------------------------------------------ */
/* Signed 128-bit arithmetic helpers                                   */
/* ------------------------------------------------------------------ */

#if RANSHAW_HAVE_INT128

typedef __int128 fq_int128_t;

static inline fq_int128_t fq_smul(int64_t a, int64_t b)
{
    return (fq_int128_t)a * b;
}

static inline fq_int128_t fq_from_i64(int64_t a)
{
    return (fq_int128_t)a;
}

static inline fq_int128_t fq_add128(fq_int128_t a, fq_int128_t b)
{
    return a + b;
}

static inline int64_t fq_lo64(fq_int128_t x)
{
    return (int64_t)(uint64_t)x;
}

static inline int64_t fq_rsh62(fq_int128_t x)
{
    return (int64_t)(x >> 62);
}

#elif RANSHAW_HAVE_UMUL128

struct fq_int128_t
{
    uint64_t lo;
    int64_t hi;
};

static inline fq_int128_t fq_smul(int64_t a, int64_t b)
{
    fq_int128_t r;
    r.lo = (uint64_t)a * (uint64_t)b;
    r.hi = __mulh(a, b);
    return r;
}

static inline fq_int128_t fq_from_i64(int64_t a)
{
    fq_int128_t r;
    r.lo = (uint64_t)a;
    r.hi = a >> 63;
    return r;
}

static inline fq_int128_t fq_add128(fq_int128_t a, fq_int128_t b)
{
    fq_int128_t r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (int64_t)(r.lo < a.lo);
    return r;
}

static inline int64_t fq_lo64(fq_int128_t x)
{
    return (int64_t)x.lo;
}

static inline int64_t fq_rsh62(fq_int128_t x)
{
    return (int64_t)((x.lo >> 62) | ((uint64_t)x.hi << 2));
}

#endif

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

struct fq_signed62
{
    int64_t v[5]; /* 5 x 62-bit signed limbs, radix 2^62 */
};

struct fq_trans2x2
{
    int64_t u, v, q, r; /* 2x2 transition matrix, entries bounded by 2^62 */
};

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

static const uint64_t FQ_M62 = (uint64_t(1) << 62) - 1;

/*
 * q = 2^255 - gamma in signed62 representation.
 *
 * q as 4 x uint64_t (LE):
 *   w[0] = 0x6EB6D2727927C79F
 *   w[1] = 0xBF7F782CB7656B58
 *   w[2] = 0xFFFFFFFFFFFFFFFF
 *   w[3] = 0x7FFFFFFFFFFFFFFF
 *
 * Extracted into 62-bit limbs:
 *   s62[0] = w[0] & M62
 *   s62[1] = ((w[0] >> 62) | (w[1] << 2)) & M62
 *   s62[2] = ((w[1] >> 60) | (w[2] << 4)) & M62
 *   s62[3] = ((w[2] >> 58) | (w[3] << 6)) & M62
 *   s62[4] = w[3] >> 56
 */
static const fq_signed62 FQ_MODULUS_S62 = {{
    (int64_t)0x2EB6D2727927C79FLL,
    (int64_t)0x3DFDE0B2DD95AD61LL,
    (int64_t)0x3FFFFFFFFFFFFFFBLL,
    (int64_t)0x3FFFFFFFFFFFFFFFLL,
    (int64_t)0x7FLL,
}};

/*
 * -q[0]^{-1} mod 2^62, for the modular update step.
 * Computed via Newton's method (Hensel lifting).
 */
static inline constexpr uint64_t fq_compute_modinv64(uint64_t x)
{
    uint64_t inv = 1;
    inv *= 2 - x * inv;
    inv *= 2 - x * inv;
    inv *= 2 - x * inv;
    inv *= 2 - x * inv;
    inv *= 2 - x * inv;
    inv *= 2 - x * inv;
    return inv;
}

static const int64_t FQ_NEG_QINV62 = (int64_t)((0 - fq_compute_modinv64((uint64_t)FQ_MODULUS_S62.v[0])) & FQ_M62);

/* ------------------------------------------------------------------ */
/* Inner loop: 62 divsteps on low bits                                 */
/* ------------------------------------------------------------------ */

/*
 * Perform 62 iterations of the Bernstein-Yang divstep on the low bits
 * of f and g. Returns the new delta and fills in the 2x2 transition
 * matrix t such that:
 *   [new_f]         [t.u  t.v] [old_f]
 *   [new_g] * 2^62 = [t.q  t.r] [old_g]
 *
 * All operations are constant-time.
 */
static inline int64_t fq_divsteps_62(int64_t delta, uint64_t f0, uint64_t g0, fq_trans2x2 *t)
{
    int64_t u = 1, v = 0, q = 0, r = 1;
    uint64_t f = f0, g = g0;

    for (int i = 0; i < 62; i++)
    {
        /* cond = -1 if (delta > 0 AND g is odd), else 0 */
        int64_t cpos = ~((delta - 1) >> 63); /* -1 if delta > 0, 0 otherwise */
        int64_t codd = -(int64_t)(g & 1); /* -1 if g odd, 0 if even */
        int64_t cond = cpos & codd;

        /* Conditional swap f <-> g */
        uint64_t xfg = (f ^ g) & (uint64_t)cond;
        f ^= xfg;
        g ^= xfg;

        /* Conditional swap matrix rows */
        int64_t xu = (u ^ q) & cond;
        u ^= xu;
        q ^= xu;
        int64_t xv = (v ^ r) & cond;
        v ^= xv;
        r ^= xv;

        /* Conditional negate delta, g, q, r */
        delta = (delta ^ cond) - cond;
        g = (g ^ (uint64_t)cond) - (uint64_t)cond;
        q = (q ^ cond) - cond;
        r = (r ^ cond) - cond;

        /* delta += 1 */
        delta++;

        /* If g is odd: g += f, q += u, r += v */
        int64_t c2 = -(int64_t)(g & 1);
        g += f & (uint64_t)c2;
        q += u & c2;
        r += v & c2;

        /* g >>= 1 (arithmetic); double f's row to compensate */
        g >>= 1;
        u <<= 1;
        v <<= 1;
    }

    t->u = u;
    t->v = v;
    t->q = q;
    t->r = r;
    return delta;
}

/* ------------------------------------------------------------------ */
/* Outer loop: apply transition matrix to full-width f,g               */
/* ------------------------------------------------------------------ */

static inline void fq_update_fg(fq_signed62 *f, fq_signed62 *g, const fq_trans2x2 *t)
{
    const int64_t u = t->u, v = t->v, q = t->q, r = t->r;

    /* Limb 0: low 62 bits are zero (division by 2^62 is exact), extract carry only */
    fq_int128_t af = fq_add128(fq_smul(u, f->v[0]), fq_smul(v, g->v[0]));
    fq_int128_t ag = fq_add128(fq_smul(q, f->v[0]), fq_smul(r, g->v[0]));
    int64_t cf = fq_rsh62(af);
    int64_t cg = fq_rsh62(ag);

    /* Limbs 1-4 of numerator become limbs 0-3 of result (shifted down by 2^62) */
    int64_t fi[5], gi[5];
    for (int i = 1; i < 5; i++)
    {
        af = fq_add128(fq_from_i64(cf), fq_add128(fq_smul(u, f->v[i]), fq_smul(v, g->v[i])));
        ag = fq_add128(fq_from_i64(cg), fq_add128(fq_smul(q, f->v[i]), fq_smul(r, g->v[i])));
        fi[i - 1] = fq_lo64(af) & (int64_t)FQ_M62;
        gi[i - 1] = fq_lo64(ag) & (int64_t)FQ_M62;
        cf = fq_rsh62(af);
        cg = fq_rsh62(ag);
    }
    fi[4] = cf;
    gi[4] = cg;

    for (int i = 0; i < 5; i++)
    {
        f->v[i] = fi[i];
        g->v[i] = gi[i];
    }
}

/* ------------------------------------------------------------------ */
/* Outer loop: apply transition matrix to d,e (mod q)                  */
/* ------------------------------------------------------------------ */

/*
 * Update d,e: new_d = (u*d + v*e + cd*q) / 2^62 (mod q)
 * where cd is chosen to make the numerator divisible by 2^62.
 */
static inline void fq_update_de(fq_signed62 *d, fq_signed62 *e, const fq_trans2x2 *t)
{
    const int64_t u = t->u, v = t->v, q = t->q, r = t->r;
    int64_t di[5], ei[5];

    /* Compute cd, ce to ensure divisibility by 2^62 */
    /* Only the low 62 bits of u*d[0]+v*e[0] matter (higher limbs contribute multiples of 2^62) */
    uint64_t md = (uint64_t)u * (uint64_t)d->v[0] + (uint64_t)v * (uint64_t)e->v[0];
    uint64_t me = (uint64_t)q * (uint64_t)d->v[0] + (uint64_t)r * (uint64_t)e->v[0];

    /* cd ≡ -(u*d + v*e) * q[0]^{-1} (mod 2^62) */
    int64_t cd = (int64_t)((md * (uint64_t)FQ_NEG_QINV62) & FQ_M62);
    int64_t ce = (int64_t)((me * (uint64_t)FQ_NEG_QINV62) & FQ_M62);

    /* Keep cd, ce in [-2^62, 2^62) by sign-extending from 62 bits */
    cd = (cd << 2) >> 2;
    ce = (ce << 2) >> 2;

    /* Compute (u*d + v*e + cd*q) / 2^62, limb by limb.
     * Limb 0 of the numerator is zero by construction (that's the point of cd/ce).
     * We extract only the carry from limb 0, then limbs 1-4 become result 0-3. */

    /* Limb 0: extract carry only (low 62 bits are zero by construction) */
    fq_int128_t ad = fq_add128(fq_add128(fq_smul(u, d->v[0]), fq_smul(v, e->v[0])), fq_smul(cd, FQ_MODULUS_S62.v[0]));
    fq_int128_t ae = fq_add128(fq_add128(fq_smul(q, d->v[0]), fq_smul(r, e->v[0])), fq_smul(ce, FQ_MODULUS_S62.v[0]));
    int64_t cf = fq_rsh62(ad);
    int64_t cg = fq_rsh62(ae);

    /* Limbs 1-4 of numerator become result limbs 0-3 */
    for (int i = 1; i < 5; i++)
    {
        ad = fq_add128(
            fq_from_i64(cf),
            fq_add128(fq_add128(fq_smul(u, d->v[i]), fq_smul(v, e->v[i])), fq_smul(cd, FQ_MODULUS_S62.v[i])));
        ae = fq_add128(
            fq_from_i64(cg),
            fq_add128(fq_add128(fq_smul(q, d->v[i]), fq_smul(r, e->v[i])), fq_smul(ce, FQ_MODULUS_S62.v[i])));
        di[i - 1] = fq_lo64(ad) & (int64_t)FQ_M62;
        ei[i - 1] = fq_lo64(ae) & (int64_t)FQ_M62;
        cf = fq_rsh62(ad);
        cg = fq_rsh62(ae);
    }
    di[4] = cf;
    ei[4] = cg;

    for (int i = 0; i < 5; i++)
    {
        d->v[i] = di[i];
        e->v[i] = ei[i];
    }
}

/* ------------------------------------------------------------------ */
/* Normalization: reduce d to [0, q) and convert to fq_fe              */
/* ------------------------------------------------------------------ */

/*
 * After all divstep iterations, f = ±1 and g = 0.
 * d contains the modular inverse (negated if f = -1).
 * This function normalizes d to [0, q) and packs into radix-2^51 fq_fe.
 */
static inline void fq_divsteps_normalize(uint64_t out[5], fq_signed62 *d, const fq_signed62 *f)
{
    /* Determine sign of f. After convergence f = ±1.
     * The sign is in the highest limb (or we can check limb 0 since f = ±1). */
    int64_t f_neg = f->v[4] >> 63; /* -1 if f < 0, 0 if f > 0 */

    /* Conditionally negate d if f < 0 */
    for (int i = 0; i < 5; i++)
        d->v[i] = (d->v[i] ^ f_neg) - f_neg;

    /* Carry-normalize d so all limbs are in [0, 2^62) */
    int64_t carry = 0;
    for (int i = 0; i < 4; i++)
    {
        d->v[i] += carry;
        carry = d->v[i] >> 62;
        d->v[i] -= carry << 62;
    }
    d->v[4] += carry;

    /* If d < 0, add q to make it positive */
    int64_t neg_mask = d->v[4] >> 63;
    carry = 0;
    for (int i = 0; i < 5; i++)
    {
        d->v[i] += FQ_MODULUS_S62.v[i] & neg_mask;
        carry = d->v[i] >> 62;
        if (i < 4)
        {
            d->v[i] -= carry << 62;
            d->v[i + 1] += carry;
        }
    }

    /* If d >= q, subtract q (at most once needed) */
    /* Check if d >= q by computing d - q and checking sign */
    int64_t tmp[5];
    int64_t borrow = 0;
    for (int i = 0; i < 5; i++)
    {
        tmp[i] = d->v[i] - FQ_MODULUS_S62.v[i] - borrow;
        borrow = (tmp[i] >> 63) & 1;
        if (i < 4)
            tmp[i] &= (int64_t)FQ_M62;
    }
    /* If no borrow (tmp[4] >= 0), d >= q, use tmp; else keep d */
    int64_t ge_mask = ~(tmp[4] >> 63); /* -1 if d >= q, 0 otherwise */
    for (int i = 0; i < 5; i++)
        d->v[i] = (d->v[i] & ~ge_mask) | (tmp[i] & ge_mask);

    /* Convert signed62 [0, q) to 4 x uint64_t intermediary, then to 5 x 51-bit (fq_fe) */
    uint64_t w0 = (uint64_t)d->v[0] | ((uint64_t)d->v[1] << 62);
    uint64_t w1 = ((uint64_t)d->v[1] >> 2) | ((uint64_t)d->v[2] << 60);
    uint64_t w2 = ((uint64_t)d->v[2] >> 4) | ((uint64_t)d->v[3] << 58);
    uint64_t w3 = ((uint64_t)d->v[3] >> 6) | ((uint64_t)d->v[4] << 56);

    /* Pack 4x64 → 5x51 */
    out[0] = w0 & ((1ULL << 51) - 1);
    out[1] = ((w0 >> 51) | (w1 << 13)) & ((1ULL << 51) - 1);
    out[2] = ((w1 >> 38) | (w2 << 26)) & ((1ULL << 51) - 1);
    out[3] = ((w2 >> 25) | (w3 << 39)) & ((1ULL << 51) - 1);
    out[4] = w3 >> 12;
}

/* ------------------------------------------------------------------ */
/* Conversion: fq_fe (radix-2^51) -> signed62                         */
/* ------------------------------------------------------------------ */

static inline void fq_fe_to_signed62(fq_signed62 *s, const uint64_t fe[5])
{
    /*
     * Normalize to canonical 51-bit limbs first.
     * Lazy add can produce limbs > 51 bits; shifting non-canonical limbs
     * (e.g. fe[1] << 51) would overflow uint64_t and corrupt the value.
     */
    uint64_t h0 = fe[0], h1 = fe[1], h2 = fe[2], h3 = fe[3], h4 = fe[4];
    uint64_t c;
    c = h0 >> 51;
    h1 += c;
    h0 &= FQ51_MASK;
    c = h1 >> 51;
    h2 += c;
    h1 &= FQ51_MASK;
    c = h2 >> 51;
    h3 += c;
    h2 &= FQ51_MASK;
    c = h3 >> 51;
    h4 += c;
    h3 &= FQ51_MASK;
    c = h4 >> 51;
    h4 &= FQ51_MASK;
    /* Gamma fold: carry from limb 4 wraps as carry * gamma */
    h0 += c * GAMMA_51[0];
    h1 += c * GAMMA_51[1];
    h2 += c * GAMMA_51[2];
    c = h0 >> 51;
    h1 += c;
    h0 &= FQ51_MASK;
    c = h1 >> 51;
    h2 += c;
    h1 &= FQ51_MASK;

    /* Reconstruct 4x64 from 5x51 */
    uint64_t w0 = h0 | (h1 << 51);
    uint64_t w1 = (h1 >> 13) | (h2 << 38);
    uint64_t w2 = (h2 >> 26) | (h3 << 25);
    uint64_t w3 = (h3 >> 39) | (h4 << 12);

    /* Extract 62-bit limbs */
    s->v[0] = (int64_t)(w0 & FQ_M62);
    s->v[1] = (int64_t)(((w0 >> 62) | (w1 << 2)) & FQ_M62);
    s->v[2] = (int64_t)(((w1 >> 60) | (w2 << 4)) & FQ_M62);
    s->v[3] = (int64_t)(((w2 >> 58) | (w3 << 6)) & FQ_M62);
    s->v[4] = (int64_t)(w3 >> 56);
}

#endif // RANSHAW_X64_FQ_DIVSTEPS_H
