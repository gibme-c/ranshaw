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
 * @file gen_ecfft_data.cpp
 * @brief Generate ECFFT precomputed data (.inl files) for Ran/Shaw curves.
 *
 * Native C++ replacement for ecfft_params.sage. Given a known b value for the
 * auxiliary curve y^2 = x^3 + ax + b (a configurable, default -3), this tool:
 *   1. Verifies the curve has full 2-torsion (cubic x^3+ax+b splits over F).
 *   2. Computes v2(#E) via 2-descent halving chains (no point counting needed).
 *   3. Finds a generator G of the 2^k subgroup (k = max halving chain + 1).
 *   4. Builds the degree-2 isogeny chain using Vélu's formulas.
 *   5. Generates the evaluation domain coset {R + i*G}.
 *   6. Outputs the .inl data file to stdout.
 *
 * This tool exists because Sage's SEA point counting over 255-bit primes is
 * extremely slow under WSL (~hours), while the 2-descent approach used here
 * runs in seconds. The Sage script (ecfft_params.sage) serves as a reference
 * implementation for cross-validation.
 *
 * References:
 *   [BCKL23]  Ben-Sasson, Carmon, Kopparty, Levit. "Elliptic Curve Fast
 *             Fourier Transform (ECFFT) Part I." https://arxiv.org/abs/2107.08473
 *   [Velu71]  Jacques Vélu. "Isogénies entre courbes elliptiques."
 *             Comptes Rendus Acad. Sci. Paris 273, pp. 238-241 (1971).
 *   [Cass91]  J.W.S. Cassels. "Lectures on Elliptic Curves." London Math Soc
 *             Student Texts 24 (1991). — 2-descent and halving chains.
 *   [ST92]    Silverman, Tate. "Rational Points on Elliptic Curves." Springer
 *             (1992). — Group structure of E[2^n].
 *
 * Mathematical background:
 *
 *   ECFFT evaluation domain (§3 of [BCKL23]):
 *     The ECFFT requires an auxiliary curve E/F with #E(F) divisible by a large
 *     power of 2, say 2^k. The evaluation domain is the set of x-coordinates
 *     of a coset S = {R + i*G : i = 0..2^k-1} where G generates the cyclic
 *     2^k subgroup and R is offset from the 2-primary part.
 *
 *   Degree-2 isogeny chain (§3.2 of [BCKL23]):
 *     At each level, a degree-2 isogeny phi: E_i -> E_{i+1} with kernel <T>
 *     (where T has order 2) maps the domain to half its size. Points P and P+T
 *     map to the same x-coordinate under phi, providing the "butterfly" pairing.
 *     The x-coordinate rational map psi(x) replaces the twiddle factor of FFT.
 *
 *   2-Sylow subgroup structure ([ST92] §IV):
 *     For E/GF(p), E(GF(p)) ≅ Z/n1 × Z/n2 with n1 | n2. The 2-Sylow subgroup
 *     is Z/2^a × Z/2^b where a <= b. Full 2-torsion (cubic x^3+ax+b splits)
 *     guarantees a >= 1. The halving chains from the three 2-torsion points
 *     determine the exponents: if the chain lengths are c0, c1, c2, then
 *     a = min(ci) + 1 and b = max(ci) + 1, and v2(#E) = a + b.
 *     The ECFFT domain size is 2^b (the larger cyclic factor).
 *
 *   Vélu's 2-isogeny formulas ([Velu71]):
 *     For E: y^2 = x^3 + ax + b with kernel point T = (x0, 0):
 *       gx = 3*x0^2 + a
 *       x-map:  psi(x) = x + gx/(x - x0) = (x^2 - x0*x + gx) / (x - x0)
 *       y-map:  psi_y(x,y) = y * ((x - x0)^2 - gx) / (x - x0)^2
 *       Codomain: a' = a - 5*gx,  b' = b - 7*x0*gx
 *     The x-map numerator has degree 2 and denominator has degree 1.
 *
 *   Coset ordering convention:
 *     The .inl data stores the coset in natural order: position i contains the
 *     x-coordinate of R + i*G. The ECFFT init functions (ecfft_fp_init,
 *     ecfft_fq_init) apply bit-reversal permutation when loading. This ensures
 *     that at each ECFFT level, isogeny fiber pairs (points differing by the
 *     kernel point T, which map to the same x under phi) occupy adjacent
 *     even/odd indices — analogous to bit-reversal in Cooley-Tukey FFT.
 *
 * Usage:
 *   ranshaw-gen-ecfft fp --known-b 0x<hex> [--a N]
 *   ranshaw-gen-ecfft fq --known-b 0x<hex> [--a N]
 *
 * Output goes to stdout (.inl content), progress/diagnostics to stderr.
 */

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

// Field arithmetic (Fp)
#include "fp.h"
#include "fp_frombytes.h"
#include "fp_invert.h"
#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_sq.h"
#include "fp_sqrt.h"
#include "fp_tobytes.h"
#include "fp_utils.h"

// Field arithmetic (Fq)
#include "fq.h"
#include "fq_frombytes.h"
#include "fq_invert.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "fq_sqrt.h"
#include "fq_tobytes.h"
#include "fq_utils.h"

// ============================================================================
// PRNG (xoshiro256** by Blackman & Vigna, 2018)
// Used only for random point generation (finding offset point R).
// Not cryptographic — deterministic from seed for reproducibility.
// ============================================================================

struct Prng
{
    uint64_t s[4];

    static uint64_t rotl(uint64_t x, int k)
    {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t next()
    {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    void seed(uint64_t v)
    {
        for (int i = 0; i < 4; i++)
        {
            v += 0x9e3779b97f4a7c15ULL;
            uint64_t z = v;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            z = z ^ (z >> 31);
            s[i] = z;
        }
    }

    void random_bytes(unsigned char out[32])
    {
        for (int i = 0; i < 4; i++)
        {
            uint64_t v = next();
            std::memcpy(out + i * 8, &v, 8);
        }
        out[31] &= 0x7f;
    }
};

// ============================================================================
// Field ops vtable — generic dispatch for Fp or Fq arithmetic
//
// This vtable lets us write algorithms once that work over either field.
// The gen tool must operate on the ECFFT auxiliary curve, which lives over
// whichever field the user specifies (fp or fq).
// ============================================================================

using fe_t = uint64_t[5];

struct FieldOps
{
    void (*mul)(fe_t, const fe_t, const fe_t);
    void (*sq)(fe_t, const fe_t);
    void (*add)(fe_t, const fe_t, const fe_t);
    void (*sub)(fe_t, const fe_t, const fe_t);
    void (*copy)(fe_t, const fe_t);
    void (*zero)(fe_t);
    void (*one)(fe_t);
    int (*isnonzero)(const fe_t);
    void (*frombytes)(fe_t, const unsigned char *);
    void (*tobytes)(unsigned char *, const fe_t);
    void (*neg)(fe_t, const fe_t);
    void (*invert)(fe_t, const fe_t);
    int (*sqrt_qr)(fe_t out, const fe_t z);
};

static int fp_sqrt_qr(fe_t out, const fe_t z)
{
    return (fp_sqrt(out, z) == 0) ? 1 : 0;
}

static int fq_sqrt_qr(fe_t out, const fe_t z)
{
    fq_sqrt(out, z);
    fe_t check;
    fq_sq(check, out);
    unsigned char z_bytes[32], check_bytes[32];
    fq_tobytes(z_bytes, z);
    fq_tobytes(check_bytes, check);
    return (std::memcmp(z_bytes, check_bytes, 32) == 0) ? 1 : 0;
}

static const FieldOps FQ_OPS = {
    fq_mul,
    fq_sq,
    fq_add,
    fq_sub,
    fq_copy,
    fq_0,
    fq_1,
    fq_isnonzero,
    fq_frombytes,
    fq_tobytes,
    fq_neg,
    fq_invert,
    fq_sqrt_qr};

static const FieldOps FP_OPS = {
    fp_mul,
    fp_sq,
    fp_add,
    fp_sub,
    fp_copy,
    fp_0,
    fp_1,
    fp_isnonzero,
    fp_frombytes,
    fp_tobytes,
    fp_neg,
    fp_invert,
    fp_sqrt_qr};

// ============================================================================
// Polynomial arithmetic mod cubic — for 2-torsion detection and root finding
//
// We work in the quotient ring GF(p)[x] / (x^3 + ax + b).
// The key operation is computing x^p mod (x^3+ax+b): if this equals x,
// the cubic splits completely over GF(p), meaning E has full 2-torsion
// (all three 2-torsion points are rational). See [Cass91] §8.
//
// Root extraction uses the Frobenius endomorphism:
//   gcd(x^p - x, f(x)) = product of linear factors of f over GF(p)
//   gcd(x^{(p-1)/2} - 1, f(x)) = product of (x - r) where r is a QR
// This is a standard probabilistic polynomial factoring technique
// (Cantor-Zassenhaus, 1981), using Legendre symbol splitting.
// ============================================================================

static void polymod3_sq(fe_t r[3], const fe_t f[3], const fe_t neg_a, const fe_t neg_b, const FieldOps *ops)
{
    fe_t d0, d1, d2, d3, d4, t1, t2;
    ops->sq(d0, f[0]);
    ops->mul(t1, f[0], f[1]);
    ops->add(d1, t1, t1);
    ops->mul(t1, f[0], f[2]);
    ops->add(d2, t1, t1);
    ops->sq(t2, f[1]);
    ops->add(d2, d2, t2);
    ops->mul(t1, f[1], f[2]);
    ops->add(d3, t1, t1);
    ops->sq(d4, f[2]);
    ops->mul(t1, d4, neg_a);
    ops->add(d2, d2, t1);
    ops->mul(t1, d4, neg_b);
    ops->add(d1, d1, t1);
    ops->mul(t1, d3, neg_a);
    ops->add(d1, d1, t1);
    ops->mul(t1, d3, neg_b);
    ops->add(d0, d0, t1);
    ops->copy(r[0], d0);
    ops->copy(r[1], d1);
    ops->copy(r[2], d2);
}

static void polymod3_mulx(fe_t r[3], const fe_t f[3], const fe_t neg_a, const fe_t neg_b, const FieldOps *ops)
{
    fe_t new0, new1, t;
    ops->mul(new0, f[2], neg_b);
    ops->mul(t, f[2], neg_a);
    ops->add(new1, f[0], t);
    ops->copy(r[2], f[1]);
    ops->copy(r[1], new1);
    ops->copy(r[0], new0);
}

static void
    polymod3_powx(fe_t result[3], const int *bits, int msb, const fe_t neg_a, const fe_t neg_b, const FieldOps *ops)
{
    ops->one(result[0]);
    ops->zero(result[1]);
    ops->zero(result[2]);

    for (int i = msb; i >= 0; i--)
    {
        fe_t tmp[3];
        polymod3_sq(tmp, result, neg_a, neg_b, ops);
        ops->copy(result[0], tmp[0]);
        ops->copy(result[1], tmp[1]);
        ops->copy(result[2], tmp[2]);

        if (bits[i])
            polymod3_mulx(result, result, neg_a, neg_b, ops);
    }
}

static int check_full_2torsion(const fe_t a, const fe_t b, const int *q_bits, int q_msb, const FieldOps *ops)
{
    fe_t neg_a, neg_b;
    ops->neg(neg_a, a);
    ops->neg(neg_b, b);

    fe_t xq[3];
    polymod3_powx(xq, q_bits, q_msb, neg_a, neg_b, ops);

    fe_t one_fe;
    ops->one(one_fe);
    ops->sub(xq[1], xq[1], one_fe);

    if (!ops->isnonzero(xq[0]) && !ops->isnonzero(xq[1]) && !ops->isnonzero(xq[2]))
        return 1;

    return 0;
}

// ============================================================================
// Polynomial GCD and root extraction
//
// Euclidean GCD algorithm for univariate polynomials over GF(p).
// Used to factor the 2-torsion cubic (for root extraction) and the halving
// quartic (for finding half-points in the 2-descent).
// ============================================================================

static int poly_degree(const fe_t *p, int max_deg, const FieldOps *ops)
{
    for (int i = max_deg; i >= 0; i--)
    {
        if (ops->isnonzero(p[i]))
            return i;
    }
    return -1;
}

static int poly_gcd(fe_t *a, int da, fe_t *b, int db, fe_t *g, const FieldOps *ops)
{
    if (da < db)
    {
        fe_t *tmp_p = a;
        a = b;
        b = tmp_p;
        int tmp_d = da;
        da = db;
        db = tmp_d;
    }

    while (db >= 0)
    {
        int deg_a = poly_degree(a, da, ops);
        int deg_b = poly_degree(b, db, ops);

        if (deg_b < 0)
            break;

        while (deg_a >= deg_b)
        {
            fe_t inv_lc_b, scale;
            ops->invert(inv_lc_b, b[deg_b]);
            ops->mul(scale, a[deg_a], inv_lc_b);

            int shift = deg_a - deg_b;
            for (int i = 0; i <= deg_b; i++)
            {
                fe_t t;
                ops->mul(t, scale, b[i]);
                ops->sub(a[i + shift], a[i + shift], t);
            }

            deg_a = poly_degree(a, da, ops);
        }

        fe_t *tmp_p = a;
        a = b;
        b = tmp_p;
        int tmp_d = da;
        da = db;
        db = tmp_d;
        da = poly_degree(a, da, ops);
        db = poly_degree(b, db, ops);
    }

    int deg = poly_degree(a, da, ops);
    if (deg < 0)
    {
        ops->zero(g[0]);
        return -1;
    }

    fe_t inv_lc;
    ops->invert(inv_lc, a[deg]);
    for (int i = 0; i <= deg; i++)
        ops->mul(g[i], a[i], inv_lc);

    return deg;
}

static int compute_pm1_half_bits(int pm1_half_bits[255], const int *prime_bits)
{
    int pm1_bits[256];
    int borrow = 1;
    for (int i = 0; i < 255; i++)
    {
        int val = prime_bits[i] - borrow;
        if (val < 0)
        {
            val += 2;
            borrow = 1;
        }
        else
            borrow = 0;
        pm1_bits[i] = val;
    }
    pm1_bits[255] = 0;

    for (int i = 0; i < 255; i++)
        pm1_half_bits[i] = pm1_bits[i + 1];

    int pm1_half_msb = 253;
    while (pm1_half_msb > 0 && pm1_half_bits[pm1_half_msb] == 0)
        pm1_half_msb--;

    return pm1_half_msb;
}

static int
    find_root_of_gcd(fe_t root, const fe_t *g, int deg, const int *prime_bits, int prime_msb, const FieldOps *ops);

static int find_cubic_roots(
    fe_t roots[3],
    const fe_t a,
    const fe_t b,
    const int *prime_bits,
    int prime_msb,
    const FieldOps *ops)
{
    fe_t neg_a, neg_b;
    ops->neg(neg_a, a);
    ops->neg(neg_b, b);

    fe_t xp[3];
    polymod3_powx(xp, prime_bits, prime_msb, neg_a, neg_b, ops);

    fe_t one_fe;
    ops->one(one_fe);
    ops->sub(xp[1], xp[1], one_fe);

    int pm1_half_bits[255];
    int pm1_half_msb = compute_pm1_half_bits(pm1_half_bits, prime_bits);

    fe_t xph[3];
    polymod3_powx(xph, pm1_half_bits, pm1_half_msb, neg_a, neg_b, ops);

    ops->sub(xph[0], xph[0], one_fe);

    fe_t aa[4], bb[3];
    ops->copy(aa[0], b);
    ops->copy(aa[1], a);
    ops->zero(aa[2]);
    ops->one(aa[3]);

    ops->copy(bb[0], xph[0]);
    ops->copy(bb[1], xph[1]);
    ops->copy(bb[2], xph[2]);

    fe_t g_poly[4];
    int deg = poly_gcd(aa, 3, bb, 2, g_poly, ops);

    // Helper lambda-like to extract remaining roots from one known root
    auto extract_remaining = [&](fe_t r0) -> int
    {
        ops->copy(roots[0], r0);
        fe_t r_sq;
        ops->sq(r_sq, roots[0]);
        fe_t disc, s;
        fe_t three_r_sq, four_a_val;
        ops->add(three_r_sq, r_sq, r_sq);
        ops->add(three_r_sq, three_r_sq, r_sq);
        ops->add(four_a_val, a, a);
        ops->add(four_a_val, four_a_val, four_a_val);
        ops->neg(disc, three_r_sq);
        ops->sub(disc, disc, four_a_val);
        if (!ops->sqrt_qr(s, disc))
            return 0;
        fe_t neg_r, inv2, two;
        ops->neg(neg_r, roots[0]);
        ops->one(two);
        ops->add(two, two, two);
        ops->invert(inv2, two);
        ops->add(roots[1], neg_r, s);
        ops->mul(roots[1], roots[1], inv2);
        ops->sub(roots[2], neg_r, s);
        ops->mul(roots[2], roots[2], inv2);
        return 1;
    };

    auto extract_from_quadratic = [&](const fe_t *gp) -> int
    {
        fe_t disc, s, inv2, two, neg_g1;
        ops->sq(disc, gp[1]);
        fe_t four_g0;
        ops->add(four_g0, gp[0], gp[0]);
        ops->add(four_g0, four_g0, four_g0);
        ops->sub(disc, disc, four_g0);
        if (!ops->sqrt_qr(s, disc))
            return 0;
        ops->neg(neg_g1, gp[1]);
        ops->one(two);
        ops->add(two, two, two);
        ops->invert(inv2, two);
        ops->add(roots[0], neg_g1, s);
        ops->mul(roots[0], roots[0], inv2);
        ops->sub(roots[1], neg_g1, s);
        ops->mul(roots[1], roots[1], inv2);
        ops->add(roots[2], roots[0], roots[1]);
        ops->neg(roots[2], roots[2]);
        return 1;
    };

    if (deg == 1)
    {
        fe_t r0;
        ops->neg(r0, g_poly[0]);
        return extract_remaining(r0);
    }
    else if (deg == 2)
    {
        return extract_from_quadratic(g_poly);
    }
    else if (deg == 0)
    {
        // Try x^((p-1)/2) + 1
        polymod3_powx(xph, pm1_half_bits, pm1_half_msb, neg_a, neg_b, ops);
        ops->add(xph[0], xph[0], one_fe);

        ops->copy(aa[0], b);
        ops->copy(aa[1], a);
        ops->zero(aa[2]);
        ops->one(aa[3]);
        ops->copy(bb[0], xph[0]);
        ops->copy(bb[1], xph[1]);
        ops->copy(bb[2], xph[2]);

        deg = poly_gcd(aa, 3, bb, 2, g_poly, ops);

        if (deg == 1)
        {
            fe_t r0;
            ops->neg(r0, g_poly[0]);
            return extract_remaining(r0);
        }
        else if (deg == 2)
        {
            return extract_from_quadratic(g_poly);
        }
    }

    // Fallback: shift by constant
    for (int c_val = 1; c_val <= 10; c_val++)
    {
        fe_t base[3];
        ops->zero(base[0]);
        ops->one(base[1]);
        ops->zero(base[2]);
        fe_t c_fe;
        ops->zero(c_fe);
        for (int k = 0; k < c_val; k++)
        {
            fe_t one2;
            ops->one(one2);
            ops->add(c_fe, c_fe, one2);
        }
        ops->add(base[0], base[0], c_fe);

        fe_t result[3];
        ops->one(result[0]);
        ops->zero(result[1]);
        ops->zero(result[2]);

        for (int i = pm1_half_msb; i >= 0; i--)
        {
            // Square mod cubic x^3 + ax + b (no x^2 term)
            fe_t f0, f1, f2;
            ops->copy(f0, result[0]);
            ops->copy(f1, result[1]);
            ops->copy(f2, result[2]);
            fe_t d0, d1, d2, d3, d4, t;
            ops->sq(d0, f0);
            ops->mul(t, f0, f1);
            ops->add(d1, t, t);
            ops->mul(t, f0, f2);
            ops->add(d2, t, t);
            fe_t t2;
            ops->sq(t2, f1);
            ops->add(d2, d2, t2);
            ops->mul(t, f1, f2);
            ops->add(d3, t, t);
            ops->sq(d4, f2);
            ops->mul(t, d4, neg_a);
            ops->add(d2, d2, t);
            ops->mul(t, d4, neg_b);
            ops->add(d1, d1, t);
            ops->mul(t, d3, neg_a);
            ops->add(d1, d1, t);
            ops->mul(t, d3, neg_b);
            ops->add(d0, d0, t);
            ops->copy(result[0], d0);
            ops->copy(result[1], d1);
            ops->copy(result[2], d2);

            if (pm1_half_bits[i])
            {
                // Multiply by base mod cubic
                ops->copy(f0, result[0]);
                ops->copy(f1, result[1]);
                ops->copy(f2, result[2]);
                ops->mul(d0, f0, base[0]);
                ops->mul(t, f0, base[1]);
                ops->mul(d1, f1, base[0]);
                ops->add(d1, d1, t);
                ops->mul(t, f0, base[2]);
                ops->mul(t2, f1, base[1]);
                ops->add(d2, t, t2);
                ops->mul(t, f2, base[0]);
                ops->add(d2, d2, t);
                ops->mul(t, f1, base[2]);
                ops->mul(t2, f2, base[1]);
                ops->add(d3, t, t2);
                ops->mul(d4, f2, base[2]);
                ops->mul(t, d4, neg_a);
                ops->add(d2, d2, t);
                ops->mul(t, d4, neg_b);
                ops->add(d1, d1, t);
                ops->mul(t, d3, neg_a);
                ops->add(d1, d1, t);
                ops->mul(t, d3, neg_b);
                ops->add(d0, d0, t);
                ops->copy(result[0], d0);
                ops->copy(result[1], d1);
                ops->copy(result[2], d2);
            }
        }

        ops->sub(result[0], result[0], one_fe);

        ops->copy(aa[0], b);
        ops->copy(aa[1], a);
        ops->zero(aa[2]);
        ops->one(aa[3]);
        ops->copy(bb[0], result[0]);
        ops->copy(bb[1], result[1]);
        ops->copy(bb[2], result[2]);

        deg = poly_gcd(aa, 3, bb, 2, g_poly, ops);

        if (deg == 1)
        {
            fe_t r0;
            ops->neg(r0, g_poly[0]);
            return extract_remaining(r0);
        }
        else if (deg == 2)
        {
            return extract_from_quadratic(g_poly);
        }
    }

    return 0;
}

// find_root_of_gcd for degree-1 or degree-2 polynomials
static int find_root_of_gcd(
    fe_t root,
    const fe_t *g,
    int deg,
    const int * /*prime_bits*/,
    int /*prime_msb*/,
    const FieldOps *ops)
{
    if (deg == 1)
    {
        ops->neg(root, g[0]);
        return 1;
    }

    if (deg == 2)
    {
        fe_t disc, g1_sq, four_g0, s, inv2, neg_g1;
        ops->sq(g1_sq, g[1]);
        ops->add(four_g0, g[0], g[0]);
        ops->add(four_g0, four_g0, four_g0);
        ops->sub(disc, g1_sq, four_g0);

        if (!ops->sqrt_qr(s, disc))
            return 0;

        ops->neg(neg_g1, g[1]);
        ops->add(root, neg_g1, s);
        fe_t two;
        ops->one(two);
        ops->add(two, two, two);
        ops->invert(inv2, two);
        ops->mul(root, root, inv2);
        return 1;
    }

    return 0;
}

// ============================================================================
// Polynomial arithmetic mod quartic — for halving chain computation
//
// The halving equation for a short Weierstrass curve y^2 = x^3 + Ax + B
// asks: given P = (xP, yP), find Q = (u, v) such that 2Q = P.
// Clearing denominators yields a monic quartic in u:
//   u^4 - 4*xP*u^3 - 2*A*u^2 - (8*B + 4*A*xP)*u + (A^2 - 4*B*xP) = 0
// (See derivation in halving_chain() below.)
//
// Finding roots of this quartic over GF(p) uses Frobenius-based splitting:
//   gcd(x^p - x, quartic) factors into the GF(p)-rational roots
//   gcd(x^{(p-1)/2} ± 1, quartic) further splits via Legendre symbol
// If those fail, shifted elements (x+c)^{(p-1)/2} provide random splitting.
// ============================================================================

static void poly4_reduce(fe_t d[7], const fe_t q[4], const FieldOps *ops)
{
    fe_t t;
    ops->mul(t, d[6], q[3]);
    ops->sub(d[5], d[5], t);
    ops->mul(t, d[6], q[2]);
    ops->sub(d[4], d[4], t);
    ops->mul(t, d[6], q[1]);
    ops->sub(d[3], d[3], t);
    ops->mul(t, d[6], q[0]);
    ops->sub(d[2], d[2], t);
    ops->mul(t, d[5], q[3]);
    ops->sub(d[4], d[4], t);
    ops->mul(t, d[5], q[2]);
    ops->sub(d[3], d[3], t);
    ops->mul(t, d[5], q[1]);
    ops->sub(d[2], d[2], t);
    ops->mul(t, d[5], q[0]);
    ops->sub(d[1], d[1], t);
    ops->mul(t, d[4], q[3]);
    ops->sub(d[3], d[3], t);
    ops->mul(t, d[4], q[2]);
    ops->sub(d[2], d[2], t);
    ops->mul(t, d[4], q[1]);
    ops->sub(d[1], d[1], t);
    ops->mul(t, d[4], q[0]);
    ops->sub(d[0], d[0], t);
}

static void poly4_sq(fe_t r[4], const fe_t f[4], const fe_t q[4], const FieldOps *ops)
{
    fe_t d[7];
    fe_t t;
    ops->sq(d[0], f[0]);
    ops->mul(t, f[0], f[1]);
    ops->add(d[1], t, t);
    ops->sq(d[2], f[1]);
    ops->mul(t, f[0], f[2]);
    ops->add(t, t, t);
    ops->add(d[2], d[2], t);
    ops->mul(d[3], f[1], f[2]);
    ops->add(d[3], d[3], d[3]);
    ops->mul(t, f[0], f[3]);
    ops->add(t, t, t);
    ops->add(d[3], d[3], t);
    ops->sq(d[4], f[2]);
    ops->mul(t, f[1], f[3]);
    ops->add(t, t, t);
    ops->add(d[4], d[4], t);
    ops->mul(t, f[2], f[3]);
    ops->add(d[5], t, t);
    ops->sq(d[6], f[3]);
    poly4_reduce(d, q, ops);
    ops->copy(r[0], d[0]);
    ops->copy(r[1], d[1]);
    ops->copy(r[2], d[2]);
    ops->copy(r[3], d[3]);
}

static void poly4_mulx(fe_t r[4], const fe_t f[4], const fe_t q[4], const FieldOps *ops)
{
    fe_t t, new0, new1, new2, new3;
    ops->mul(t, f[3], q[0]);
    ops->neg(new0, t);
    ops->mul(t, f[3], q[1]);
    ops->sub(new1, f[0], t);
    ops->mul(t, f[3], q[2]);
    ops->sub(new2, f[1], t);
    ops->mul(t, f[3], q[3]);
    ops->sub(new3, f[2], t);
    ops->copy(r[0], new0);
    ops->copy(r[1], new1);
    ops->copy(r[2], new2);
    ops->copy(r[3], new3);
}

static void poly4_mul(fe_t r[4], const fe_t f[4], const fe_t g[4], const fe_t q[4], const FieldOps *ops)
{
    fe_t d[7];
    fe_t t;
    ops->mul(d[0], f[0], g[0]);
    ops->mul(d[1], f[0], g[1]);
    ops->mul(t, f[1], g[0]);
    ops->add(d[1], d[1], t);
    ops->mul(d[2], f[0], g[2]);
    ops->mul(t, f[1], g[1]);
    ops->add(d[2], d[2], t);
    ops->mul(t, f[2], g[0]);
    ops->add(d[2], d[2], t);
    ops->mul(d[3], f[0], g[3]);
    ops->mul(t, f[1], g[2]);
    ops->add(d[3], d[3], t);
    ops->mul(t, f[2], g[1]);
    ops->add(d[3], d[3], t);
    ops->mul(t, f[3], g[0]);
    ops->add(d[3], d[3], t);
    ops->mul(d[4], f[1], g[3]);
    ops->mul(t, f[2], g[2]);
    ops->add(d[4], d[4], t);
    ops->mul(t, f[3], g[1]);
    ops->add(d[4], d[4], t);
    ops->mul(d[5], f[2], g[3]);
    ops->mul(t, f[3], g[2]);
    ops->add(d[5], d[5], t);
    ops->mul(d[6], f[3], g[3]);
    poly4_reduce(d, q, ops);
    ops->copy(r[0], d[0]);
    ops->copy(r[1], d[1]);
    ops->copy(r[2], d[2]);
    ops->copy(r[3], d[3]);
}

static void poly4_powx_p(fe_t result[4], const int *bits, int msb, const fe_t q[4], const FieldOps *ops)
{
    ops->one(result[0]);
    ops->zero(result[1]);
    ops->zero(result[2]);
    ops->zero(result[3]);

    for (int i = msb; i >= 0; i--)
    {
        fe_t tmp[4];
        poly4_sq(tmp, result, q, ops);
        ops->copy(result[0], tmp[0]);
        ops->copy(result[1], tmp[1]);
        ops->copy(result[2], tmp[2]);
        ops->copy(result[3], tmp[3]);

        if (bits[i])
            poly4_mulx(result, result, q, ops);
    }
}

static void
    poly4_pow(fe_t result[4], const fe_t base[4], const int *bits, int msb, const fe_t q[4], const FieldOps *ops)
{
    ops->one(result[0]);
    ops->zero(result[1]);
    ops->zero(result[2]);
    ops->zero(result[3]);

    for (int i = msb; i >= 0; i--)
    {
        fe_t tmp[4];
        poly4_sq(tmp, result, q, ops);
        ops->copy(result[0], tmp[0]);
        ops->copy(result[1], tmp[1]);
        ops->copy(result[2], tmp[2]);
        ops->copy(result[3], tmp[3]);

        if (bits[i])
        {
            poly4_mul(tmp, result, base, q, ops);
            ops->copy(result[0], tmp[0]);
            ops->copy(result[1], tmp[1]);
            ops->copy(result[2], tmp[2]);
            ops->copy(result[3], tmp[3]);
        }
    }
}

static int try_extract_root_from_factor(
    fe_t root,
    const fe_t *g,
    int deg,
    const int *prime_bits,
    int prime_msb,
    const FieldOps *ops)
{
    if (deg >= 1 && deg <= 3)
        return find_root_of_gcd(root, g, deg, prime_bits, prime_msb, ops);
    return 0;
}

static int find_one_root(fe_t root, const fe_t quartic[4], const FieldOps *ops, const int *prime_bits, int prime_msb)
{
    fe_t xp[4];
    poly4_powx_p(xp, prime_bits, prime_msb, quartic, ops);

    fe_t one_fe;
    ops->one(one_fe);
    ops->sub(xp[1], xp[1], one_fe);

    int xp_deg = poly_degree(xp, 3, ops);

    if (xp_deg >= 0)
    {
        fe_t a_poly[5], b_poly[4];
        ops->copy(a_poly[0], quartic[0]);
        ops->copy(a_poly[1], quartic[1]);
        ops->copy(a_poly[2], quartic[2]);
        ops->copy(a_poly[3], quartic[3]);
        ops->one(a_poly[4]);

        ops->copy(b_poly[0], xp[0]);
        ops->copy(b_poly[1], xp[1]);
        ops->copy(b_poly[2], xp[2]);
        ops->copy(b_poly[3], xp[3]);

        fe_t g[5];
        int deg = poly_gcd(a_poly, 4, b_poly, 3, g, ops);

        if (deg >= 1 && deg <= 3)
            return find_root_of_gcd(root, g, deg, prime_bits, prime_msb, ops);

        if (deg <= 0)
            return 0;
    }

    int pm1_half_bits[255];
    int pm1_half_msb = compute_pm1_half_bits(pm1_half_bits, prime_bits);

    fe_t xph[4];
    poly4_powx_p(xph, pm1_half_bits, pm1_half_msb, quartic, ops);

    {
        fe_t a_poly[5], b_poly[4];
        ops->copy(a_poly[0], quartic[0]);
        ops->copy(a_poly[1], quartic[1]);
        ops->copy(a_poly[2], quartic[2]);
        ops->copy(a_poly[3], quartic[3]);
        ops->one(a_poly[4]);

        ops->copy(b_poly[0], xph[0]);
        ops->sub(b_poly[0], b_poly[0], one_fe);
        ops->copy(b_poly[1], xph[1]);
        ops->copy(b_poly[2], xph[2]);
        ops->copy(b_poly[3], xph[3]);

        fe_t g[5];
        int deg = poly_gcd(a_poly, 4, b_poly, 3, g, ops);

        if (try_extract_root_from_factor(root, g, deg, prime_bits, prime_msb, ops))
            return 1;
    }

    {
        fe_t a_poly[5], b_poly[4];
        ops->copy(a_poly[0], quartic[0]);
        ops->copy(a_poly[1], quartic[1]);
        ops->copy(a_poly[2], quartic[2]);
        ops->copy(a_poly[3], quartic[3]);
        ops->one(a_poly[4]);

        ops->copy(b_poly[0], xph[0]);
        ops->add(b_poly[0], b_poly[0], one_fe);
        ops->copy(b_poly[1], xph[1]);
        ops->copy(b_poly[2], xph[2]);
        ops->copy(b_poly[3], xph[3]);

        fe_t g[5];
        int deg = poly_gcd(a_poly, 4, b_poly, 3, g, ops);

        if (try_extract_root_from_factor(root, g, deg, prime_bits, prime_msb, ops))
            return 1;
    }

    for (int c_val = 1; c_val <= 10; c_val++)
    {
        fe_t base[4];
        ops->zero(base[0]);
        ops->one(base[1]);
        ops->zero(base[2]);
        ops->zero(base[3]);
        for (int k = 0; k < c_val; k++)
            ops->add(base[0], base[0], one_fe);

        fe_t result[4];
        poly4_pow(result, base, pm1_half_bits, pm1_half_msb, quartic, ops);

        fe_t a_poly[5], b_poly[4];
        ops->copy(a_poly[0], quartic[0]);
        ops->copy(a_poly[1], quartic[1]);
        ops->copy(a_poly[2], quartic[2]);
        ops->copy(a_poly[3], quartic[3]);
        ops->one(a_poly[4]);

        ops->sub(result[0], result[0], one_fe);
        ops->copy(b_poly[0], result[0]);
        ops->copy(b_poly[1], result[1]);
        ops->copy(b_poly[2], result[2]);
        ops->copy(b_poly[3], result[3]);

        fe_t g[5];
        int deg = poly_gcd(a_poly, 4, b_poly, 3, g, ops);

        if (try_extract_root_from_factor(root, g, deg, prime_bits, prime_msb, ops))
            return 1;
    }

    return 0;
}

// ============================================================================
// Halving chain computation — 2-descent for v2(#E)
//
// The 2-descent determines the structure of the 2-Sylow subgroup E[2^inf]
// without computing #E. Starting from a 2-torsion point (e_i, 0), we
// iteratively try to "halve" it: find Q such that 2Q = P.
//
// Level 2 -> 3 ([Cass91] §8):
//   The 2-torsion point (e_i, 0) is halvable iff D_i = (e_i-e_j)(e_i-e_k)
//   is a quadratic residue, where e_j, e_k are the other two roots.
//   If so, a half-point has x = e_i + sqrt(D_i).
//
// Deeper levels:
//   For point P = (xP, yP), a half-point Q = (u, v) satisfies 2Q = P.
//   From the doubling formula x(2Q) = ((3u^2+A)/(2v))^2 - 2u = xP,
//   clearing denominators (v^2 = u^3 + Au + B) gives the halving quartic:
//     u^4 - 4*xP*u^3 - 2*A*u^2 - (8B + 4A*xP)*u + (A^2 - 4B*xP) = 0
//   If this quartic has a root u in GF(p), then v = sqrt(u^3+Au+B) gives
//   a half-point, and the chain continues.
//
// The chain length for each 2-torsion root determines the 2-Sylow structure:
//   E[2^inf] ≅ Z/2^(min_chain+1) × Z/2^(max_chain+1)
// and v2(#E) = min_chain + max_chain + 2.
//
// The ECFFT uses the larger cyclic factor: levels = max_chain + 1.
// ============================================================================

static int halving_chain(
    const fe_t e_i,
    const fe_t e_j,
    const fe_t e_k,
    const fe_t A,
    const fe_t B,
    const FieldOps *ops,
    const int *prime_bits,
    int prime_msb,
    int max_depth)
{
    fe_t diff_j, diff_k, D_i, sqrt_D;
    ops->sub(diff_j, e_i, e_j);
    ops->sub(diff_k, e_i, e_k);
    ops->mul(D_i, diff_j, diff_k);

    if (!ops->sqrt_qr(sqrt_D, D_i))
        return 0;

    fe_t xP, yP;
    ops->add(xP, e_i, sqrt_D);

    fe_t x2, x3, ax, y2;
    ops->sq(x2, xP);
    ops->mul(x3, x2, xP);
    ops->mul(ax, A, xP);
    ops->add(y2, x3, ax);
    ops->add(y2, y2, B);

    fe_t yP_tmp;
    if (!ops->sqrt_qr(yP_tmp, y2))
        return 0;
    ops->copy(yP, yP_tmp);

    int chain = 1;

    for (int depth = 1; depth < max_depth; depth++)
    {
        fe_t quartic[4];

        fe_t A2, four_BxP;
        ops->sq(A2, A);
        ops->mul(four_BxP, B, xP);
        ops->add(four_BxP, four_BxP, four_BxP);
        ops->add(four_BxP, four_BxP, four_BxP);
        ops->sub(quartic[0], A2, four_BxP);

        fe_t AxP, two_B, c1_inner;
        ops->mul(AxP, A, xP);
        ops->add(two_B, B, B);
        ops->add(c1_inner, two_B, AxP);
        ops->add(c1_inner, c1_inner, c1_inner);
        ops->add(c1_inner, c1_inner, c1_inner);
        ops->neg(quartic[1], c1_inner);

        ops->add(quartic[2], A, A);
        ops->neg(quartic[2], quartic[2]);

        ops->add(quartic[3], xP, xP);
        ops->add(quartic[3], quartic[3], quartic[3]);
        ops->neg(quartic[3], quartic[3]);

        fe_t u;
        if (!find_one_root(u, quartic, ops, prime_bits, prime_msb))
            break;

        fe_t u2_new, u3_new, au_new, v2_new;
        ops->sq(u2_new, u);
        ops->mul(u3_new, u2_new, u);
        ops->mul(au_new, A, u);
        ops->add(v2_new, u3_new, au_new);
        ops->add(v2_new, v2_new, B);

        fe_t v;
        if (!ops->sqrt_qr(v, v2_new))
            break;

        fe_t three_u2, numer, two_v, lambda, lambda_sq, x_double;
        ops->add(three_u2, u2_new, u2_new);
        ops->add(three_u2, three_u2, u2_new);
        ops->add(numer, three_u2, A);
        ops->add(two_v, v, v);
        ops->invert(lambda, two_v);
        ops->mul(lambda, lambda, numer);
        ops->sq(lambda_sq, lambda);
        ops->add(x_double, u, u);
        ops->sub(x_double, lambda_sq, x_double);

        unsigned char xd_bytes[32], xp_bytes[32];
        ops->tobytes(xd_bytes, x_double);
        ops->tobytes(xp_bytes, xP);

        if (std::memcmp(xd_bytes, xp_bytes, 32) != 0)
            break;

        chain++;
        ops->copy(xP, u);
        ops->copy(yP, v);
    }

    return chain;
}

// Compute v2(#E) and return individual chain lengths.
//
// v2 = min(chains) + max(chains) + 2.
//
// The 2-Sylow subgroup E[2^inf](GF(p)) ≅ Z/2^a × Z/2^b where a <= b.
// The three halving chains from the 2-torsion roots satisfy:
//   a = min(c_i) + 1,  b = max(c_i) + 1
// and exactly one chain has length a-1, one has length b-1, and one
// has length a-1 (when a < b) or all three are equal (when a = b).
// See [ST92] §IV.4 for the group structure theorem.
//
// The ECFFT uses the larger cyclic factor: levels = b = max_chain + 1.
static int compute_v2(
    const fe_t A,
    const fe_t B,
    const fe_t roots[3],
    const FieldOps *ops,
    const int *prime_bits,
    int prime_msb,
    int chains_out[3])
{
    int max_depth = 30;

    chains_out[0] = halving_chain(roots[0], roots[1], roots[2], A, B, ops, prime_bits, prime_msb, max_depth);
    chains_out[1] = halving_chain(roots[1], roots[0], roots[2], A, B, ops, prime_bits, prime_msb, max_depth);
    chains_out[2] = halving_chain(roots[2], roots[0], roots[1], A, B, ops, prime_bits, prime_msb, max_depth);

    int mn = chains_out[0], mx = chains_out[0];
    for (int i = 1; i < 3; i++)
    {
        if (chains_out[i] < mn)
            mn = chains_out[i];
        if (chains_out[i] > mx)
            mx = chains_out[i];
    }

    return mn + mx + 2;
}

// ============================================================================
// Prime bytes — little-endian encoding of the field primes
//
// p = 2^255 - 19          (Ed25519 / Ran base field)
// q = 2^255 - gamma       (Shaw base field, Crandall prime)
//   where gamma = g0 + g1*2^51 + g2*2^102 (radix-2^51 limbs from fq51.h)
// ============================================================================

static void get_q_bytes(unsigned char q_bytes[32])
{
    uint64_t g0 = 0x12D8D86D83861ULL;
    uint64_t g1 = 0x269135294F229ULL;
    uint64_t g2 = 0x102021FULL;

    unsigned char gamma[32];
    std::memset(gamma, 0, 32);
    for (int i = 0; i < 8; i++)
        gamma[i] = (unsigned char)(g0 >> (8 * i));

    {
        uint64_t lo = g1 << 3;
        uint64_t hi = g1 >> 61;
        for (int i = 0; i < 8; i++)
        {
            uint16_t sum = (uint16_t)gamma[6 + i] + (uint16_t)((unsigned char)(lo >> (8 * i)));
            gamma[6 + i] = (unsigned char)sum;
            int j = 7 + i;
            while (sum > 255 && j < 32)
            {
                sum = (uint16_t)gamma[j] + (uint16_t)(sum >> 8);
                gamma[j] = (unsigned char)sum;
                j++;
                if (sum <= 255)
                    break;
            }
        }
        for (int i = 0; i < 8 && (14 + i) < 32; i++)
        {
            uint16_t sum = (uint16_t)gamma[14 + i] + (uint16_t)((unsigned char)(hi >> (8 * i)));
            gamma[14 + i] = (unsigned char)sum;
            int j = 15 + i;
            while (sum > 255 && j < 32)
            {
                sum = (uint16_t)gamma[j] + (uint16_t)(sum >> 8);
                gamma[j] = (unsigned char)sum;
                j++;
                if (sum <= 255)
                    break;
            }
        }
    }
    {
        uint64_t lo = g2 << 6;
        uint64_t hi = g2 >> 58;
        for (int i = 0; i < 8 && (12 + i) < 32; i++)
        {
            uint16_t sum = (uint16_t)gamma[12 + i] + (uint16_t)((unsigned char)(lo >> (8 * i)));
            gamma[12 + i] = (unsigned char)sum;
            int j = 13 + i;
            while (sum > 255 && j < 32)
            {
                sum = (uint16_t)gamma[j] + (uint16_t)(sum >> 8);
                gamma[j] = (unsigned char)sum;
                j++;
                if (sum <= 255)
                    break;
            }
        }
        for (int i = 0; i < 8 && (20 + i) < 32; i++)
        {
            uint16_t sum = (uint16_t)gamma[20 + i] + (uint16_t)((unsigned char)(hi >> (8 * i)));
            gamma[20 + i] = (unsigned char)sum;
            int j = 21 + i;
            while (sum > 255 && j < 32)
            {
                sum = (uint16_t)gamma[j] + (uint16_t)(sum >> 8);
                gamma[j] = (unsigned char)sum;
                j++;
                if (sum <= 255)
                    break;
            }
        }
    }

    int borrow = 0;
    for (int i = 0; i < 32; i++)
    {
        int top_byte = (i == 31) ? 0x80 : 0;
        int diff = top_byte - gamma[i] - borrow;
        if (diff < 0)
        {
            diff += 256;
            borrow = 1;
        }
        else
        {
            borrow = 0;
        }
        q_bytes[i] = (unsigned char)diff;
    }
}

static void get_p_bytes(unsigned char p_bytes[32])
{
    std::memset(p_bytes, 0, 32);
    p_bytes[0] = 0xed;
    for (int i = 1; i < 31; i++)
        p_bytes[i] = 0xff;
    p_bytes[31] = 0x7f;
}

// ============================================================================
// Hex formatting
// ============================================================================

static void hex_string(char *out, size_t out_size, const unsigned char *bytes, int len)
{
    size_t pos = 0;
    for (int i = len - 1; i >= 0; i--)
    {
        int n = snprintf(out + pos, out_size - pos, "%02x", bytes[i]);
        if (n > 0)
            pos += (size_t)n;
    }
}

// ============================================================================
// Jacobian point arithmetic (generic, using FieldOps vtable)
//
// Standalone implementation for the ECFFT auxiliary curve y^2 = x^3 + ax + b
// where a = -3 but b varies at each isogeny level. Cannot reuse the library's
// ran_dbl() / shaw_add() because those hardcode the Ran/Shaw curve
// constants.
//
// Jacobian coordinates: affine (x, y) ↔ Jacobian (X : Y : Z) with
//   x = X/Z^2,  y = Y/Z^3.  Identity: Z = 0.
//
// Doubling uses the a = -3 optimization ([CohenFrey] §13.2.1.c) when a = -3:
//   M = 3*(X - Z^2)*(X + Z^2)  instead of  M = 3*X^2 + a*Z^4
//   Cost: 3M + 5S (vs 4M + 4S for general a).
// For general a, uses M = 3*X^2 + a*Z^4 (4M + 4S).
//
// Addition uses standard Jacobian formulas ([CohenFrey] §13.2.1.a):
//   Cost: 11M + 5S.
// ============================================================================

// Helper: convert a small integer to a field element
static void fe_from_int(fe_t out, int val, const FieldOps *ops)
{
    if (val == 0)
    {
        ops->zero(out);
        return;
    }
    fe_t one_fe;
    ops->one(one_fe);
    ops->zero(out);
    int abs_val = val < 0 ? -val : val;
    for (int i = 0; i < abs_val; i++)
        ops->add(out, out, one_fe);
    if (val < 0)
        ops->neg(out, out);
}

struct JacobianPoint
{
    fe_t X, Y, Z;
};

static void jac_identity(JacobianPoint &P, const FieldOps *ops)
{
    ops->zero(P.X);
    ops->one(P.Y);
    ops->zero(P.Z);
}

static int jac_is_identity(const JacobianPoint &P, const FieldOps *ops)
{
    return !ops->isnonzero(P.Z);
}

// Double: general a formula M = 3*X^2 + a*Z^4 (4M + 4S)
static void jac_dbl(JacobianPoint &R, const JacobianPoint &P, const fe_t a_fe, const FieldOps *ops)
{
    if (jac_is_identity(P, ops))
    {
        jac_identity(R, ops);
        return;
    }

    fe_t Z2, M, S, X3, Y3, Z3, t1;

    // M = 3*X^2 + a*Z^4
    ops->sq(Z2, P.Z);
    fe_t X2, Z4, aZ4;
    ops->sq(X2, P.X);
    ops->add(M, X2, X2);
    ops->add(M, M, X2); // M = 3*X^2
    ops->sq(Z4, Z2);
    ops->mul(aZ4, a_fe, Z4);
    ops->add(M, M, aZ4); // M = 3*X^2 + a*Z^4

    // S = 4*X*Y^2
    fe_t Y2;
    ops->sq(Y2, P.Y);
    ops->mul(S, P.X, Y2);
    ops->add(S, S, S);
    ops->add(S, S, S); // S = 4*X*Y^2

    // X3 = M^2 - 2*S
    ops->sq(X3, M);
    ops->sub(X3, X3, S);
    ops->sub(X3, X3, S);

    // Y3 = M*(S - X3) - 8*Y^4
    fe_t Y4;
    ops->sq(Y4, Y2);
    ops->sub(t1, S, X3);
    ops->mul(Y3, M, t1);
    ops->add(Y4, Y4, Y4);
    ops->add(Y4, Y4, Y4);
    ops->add(Y4, Y4, Y4); // 8*Y^4
    ops->sub(Y3, Y3, Y4);

    // Z3 = 2*Y*Z
    ops->mul(Z3, P.Y, P.Z);
    ops->add(Z3, Z3, Z3);

    ops->copy(R.X, X3);
    ops->copy(R.Y, Y3);
    ops->copy(R.Z, Z3);
}

// General Jacobian addition (11M + 5S)
static void
    jac_add(JacobianPoint &R, const JacobianPoint &P, const JacobianPoint &Q, const fe_t a_fe, const FieldOps *ops)
{
    if (jac_is_identity(P, ops))
    {
        ops->copy(R.X, Q.X);
        ops->copy(R.Y, Q.Y);
        ops->copy(R.Z, Q.Z);
        return;
    }
    if (jac_is_identity(Q, ops))
    {
        ops->copy(R.X, P.X);
        ops->copy(R.Y, P.Y);
        ops->copy(R.Z, P.Z);
        return;
    }

    fe_t Z1sq, Z2sq, U1, U2, Z1cu, Z2cu, S1, S2;
    ops->sq(Z1sq, P.Z);
    ops->sq(Z2sq, Q.Z);
    ops->mul(U1, P.X, Z2sq);
    ops->mul(U2, Q.X, Z1sq);
    ops->mul(Z1cu, Z1sq, P.Z);
    ops->mul(Z2cu, Z2sq, Q.Z);
    ops->mul(S1, P.Y, Z2cu);
    ops->mul(S2, Q.Y, Z1cu);

    fe_t H, r_val;
    ops->sub(H, U2, U1);
    ops->sub(r_val, S2, S1);

    // Check if points are equal (H == 0)
    if (!ops->isnonzero(H))
    {
        if (!ops->isnonzero(r_val))
        {
            // P == Q, use doubling
            jac_dbl(R, P, a_fe, ops);
            return;
        }
        // P == -Q, result is identity
        jac_identity(R, ops);
        return;
    }

    fe_t H2, H3, t1;
    ops->sq(H2, H);
    ops->mul(H3, H2, H);

    fe_t U1H2;
    ops->mul(U1H2, U1, H2);

    fe_t X3, Y3, Z3;
    // X3 = r^2 - H^3 - 2*U1*H^2
    ops->sq(X3, r_val);
    ops->sub(X3, X3, H3);
    ops->sub(X3, X3, U1H2);
    ops->sub(X3, X3, U1H2);

    // Y3 = r*(U1*H^2 - X3) - S1*H^3
    ops->sub(t1, U1H2, X3);
    ops->mul(Y3, r_val, t1);
    ops->mul(t1, S1, H3);
    ops->sub(Y3, Y3, t1);

    // Z3 = H * Z1 * Z2
    ops->mul(Z3, H, P.Z);
    ops->mul(Z3, Z3, Q.Z);

    ops->copy(R.X, X3);
    ops->copy(R.Y, Y3);
    ops->copy(R.Z, Z3);
}

// Convert to affine: x = X/Z^2, y = Y/Z^3
static void jac_to_affine(fe_t x_out, fe_t y_out, const JacobianPoint &P, const FieldOps *ops)
{
    fe_t Z_inv, Z_inv2, Z_inv3;
    ops->invert(Z_inv, P.Z);
    ops->sq(Z_inv2, Z_inv);
    ops->mul(Z_inv3, Z_inv2, Z_inv);
    ops->mul(x_out, P.X, Z_inv2);
    ops->mul(y_out, P.Y, Z_inv3);
}

// ============================================================================
// Vélu's degree-2 isogeny formulas [Velu71]
//
// For E: y^2 = x^3 + ax + b with a 2-torsion kernel point T = (x0, 0):
//
//   gx = 3*x0^2 + a                    (derivative of curve equation at x0)
//
//   x-map:  psi(x) = x + gx/(x - x0)
//                   = (x^2 - x0*x + gx) / (x - x0)
//
//   y-map:  psi_y(x,y) = y * ((x - x0)^2 - gx) / (x - x0)^2
//
//   Codomain curve:  a' = a - 5*gx
//                    b' = b - 7*x0*gx
//
// The x-map is stored as:
//   num[0] = gx,   num[1] = -x0,  num[2] = 1     (degree 2)
//   den[0] = -x0,  den[1] = 1                     (degree 1)
//
// Note: num[1] = den[0] = -x0 for ALL levels (not -2*x0).
// This can be verified by expanding psi(x) = x + gx/(x-x0):
//   = (x*(x-x0) + gx) / (x-x0)
//   = (x^2 - x0*x + gx) / (x - x0)
// The coefficient of x in the numerator is -x0, not -2*x0.
// ============================================================================

struct IsogenyData
{
    // Velu x-map: psi(x) = x + gx/(x - x0) = (x^2 - x0*x + gx) / (x - x0)
    // num[0] = gx = 3*x0^2 + a, num[1] = -x0, num[2] = 1
    // den[0] = -x0, den[1] = 1
    unsigned char num[3][32]; // coefficients of numerator, little-endian
    unsigned char den[2][32]; // coefficients of denominator, little-endian
};

// Apply Velu 2-isogeny to a point (x, y) on y^2 = x^3 + ax + b with kernel (x0, 0).
// Returns new curve parameters (a', b') and maps generator G through.
static void velu_2isogeny(
    fe_t a_out,
    fe_t b_out, // codomain curve params
    IsogenyData &iso, // isogeny rational map coefficients
    JacobianPoint &G_out, // image of G under isogeny
    const fe_t x0, // kernel point x-coordinate
    const fe_t a_in,
    const fe_t b_in, // domain curve params
    const JacobianPoint &G_in, // generator to map through
    const FieldOps *ops)
{
    // gx = 3*x0^2 + a
    fe_t x0_sq, gx, three_x0_sq;
    ops->sq(x0_sq, x0);
    ops->add(three_x0_sq, x0_sq, x0_sq);
    ops->add(three_x0_sq, three_x0_sq, x0_sq);
    ops->add(gx, three_x0_sq, a_in);

    // Codomain: a' = a - 5*gx, b' = b - 7*x0*gx
    fe_t five_gx, seven_x0_gx;
    ops->add(five_gx, gx, gx); // 2*gx
    ops->add(five_gx, five_gx, five_gx); // 4*gx
    ops->add(five_gx, five_gx, gx); // 5*gx
    ops->sub(a_out, a_in, five_gx);

    fe_t x0_gx;
    ops->mul(x0_gx, x0, gx);
    ops->add(seven_x0_gx, x0_gx, x0_gx); // 2
    ops->add(seven_x0_gx, seven_x0_gx, seven_x0_gx); // 4
    ops->add(seven_x0_gx, seven_x0_gx, x0_gx); // 5
    ops->add(seven_x0_gx, seven_x0_gx, x0_gx); // 6
    ops->add(seven_x0_gx, seven_x0_gx, x0_gx); // 7
    ops->sub(b_out, b_in, seven_x0_gx);

    // Store isogeny coefficients
    // Velu x-map: psi(x) = x + gx/(x - x0) = (x^2 - x0*x + gx) / (x - x0)
    // num: coeff[0] = gx = 3*x0^2 + a, coeff[1] = -x0, coeff[2] = 1
    // den: coeff[0] = -x0, coeff[1] = 1
    fe_t neg_x0;
    ops->neg(neg_x0, x0);

    fe_t one_fe;
    ops->one(one_fe);

    ops->tobytes(iso.num[0], gx); // 3*x0^2 + a
    ops->tobytes(iso.num[1], neg_x0); // -x0
    ops->tobytes(iso.num[2], one_fe); // 1

    ops->tobytes(iso.den[0], neg_x0); // -x0
    ops->tobytes(iso.den[1], one_fe); // 1

    // Map G through the isogeny
    if (jac_is_identity(G_in, ops))
    {
        jac_identity(G_out, ops);
        return;
    }

    // Convert G to affine for the x-map
    fe_t gx_aff, gy_aff;
    jac_to_affine(gx_aff, gy_aff, G_in, ops);

    // x-map: psi_x = x + gx/(x - x0) = (x^2 - x0*x + gx) / (x - x0)
    fe_t diff, numer_x, gx_sq, x_new;
    ops->sub(diff, gx_aff, x0);

    // Check if G.x == x0 (G is in the kernel, maps to identity)
    if (!ops->isnonzero(diff))
    {
        jac_identity(G_out, ops);
        return;
    }

    ops->sq(gx_sq, gx_aff);
    ops->mul(numer_x, neg_x0, gx_aff);
    ops->add(numer_x, gx_sq, numer_x);
    ops->add(numer_x, numer_x, gx); // gx = 3*x0^2 + a

    fe_t diff_inv;
    ops->invert(diff_inv, diff);
    ops->mul(x_new, numer_x, diff_inv);

    // y-map: psi_y = y * ((x - x0)^2 - gx) / (x - x0)^2
    fe_t diff_sq, y_numer, y_new;
    ops->sq(diff_sq, diff);
    ops->sub(y_numer, diff_sq, gx);
    fe_t diff_sq_inv;
    ops->invert(diff_sq_inv, diff_sq);
    ops->mul(y_new, gy_aff, y_numer);
    ops->mul(y_new, y_new, diff_sq_inv);

    // Store as Jacobian (Z = 1)
    ops->copy(G_out.X, x_new);
    ops->copy(G_out.Y, y_new);
    ops->one(G_out.Z);
}

// ============================================================================
// Generator finding via halving chains
//
// To build G of order 2^v2, we start from a 2-torsion point (e_i, 0) and
// repeatedly halve: find Q such that 2Q = P. After k halvings, Q has order
// 2^(k+1). We need v2-1 successful halvings to reach order 2^v2.
//
// This is the reverse of the descent used to compute v2: we build UP the
// chain rather than walking DOWN to count its length.
//
// Why not use random points + cofactor multiplication?
// We don't know #E (that's why we used 2-descent instead of point counting).
// Without #E, we can't compute the cofactor. The halving approach needs only
// the 2-torsion roots (which we already have from the v2 computation).
// ============================================================================
static int build_generator_from_halving(
    JacobianPoint &G,
    int v2,
    const fe_t roots[3],
    const fe_t A,
    const fe_t B,
    const FieldOps *ops,
    const int *prime_bits,
    int prime_msb)
{
    // We need v2-1 halvings starting from a 2-torsion point.
    // Try each root as starting point.
    for (int ri = 0; ri < 3; ri++)
    {
        int rj = (ri + 1) % 3;
        int rk = (ri + 2) % 3;

        // Level 2->3: D_i = (e_i - e_j)(e_i - e_k); halvable iff D_i is QR
        fe_t diff_j, diff_k, D_i, sqrt_D;
        ops->sub(diff_j, roots[ri], roots[rj]);
        ops->sub(diff_k, roots[ri], roots[rk]);
        ops->mul(D_i, diff_j, diff_k);

        if (!ops->sqrt_qr(sqrt_D, D_i))
            continue;

        // Half-point: x_half = e_i + sqrt(D_i), y = sqrt(x^3 + Ax + B)
        fe_t xP, yP;
        ops->add(xP, roots[ri], sqrt_D);

        fe_t x2, x3, ax, y2;
        ops->sq(x2, xP);
        ops->mul(x3, x2, xP);
        ops->mul(ax, A, xP);
        ops->add(y2, x3, ax);
        ops->add(y2, y2, B);

        fe_t yP_tmp;
        if (!ops->sqrt_qr(yP_tmp, y2))
            continue;
        ops->copy(yP, yP_tmp);

        // We now have a point (xP, yP) of order 4 (since 2*(xP,yP) = (e_i,0) has order 2).
        // We need v2-2 more halvings to get order 2^v2.
        int chain_len = 1; // currently at order 2^2 = 4

        for (int depth = 1; depth < v2 - 1; depth++)
        {
            // Build halving quartic for (xP, yP) on y^2 = x^3 + Ax + B
            fe_t quartic[4];

            fe_t A2, four_BxP;
            ops->sq(A2, A);
            ops->mul(four_BxP, B, xP);
            ops->add(four_BxP, four_BxP, four_BxP);
            ops->add(four_BxP, four_BxP, four_BxP);
            ops->sub(quartic[0], A2, four_BxP);

            fe_t AxP, two_B, c1_inner;
            ops->mul(AxP, A, xP);
            ops->add(two_B, B, B);
            ops->add(c1_inner, two_B, AxP);
            ops->add(c1_inner, c1_inner, c1_inner);
            ops->add(c1_inner, c1_inner, c1_inner);
            ops->neg(quartic[1], c1_inner);

            ops->add(quartic[2], A, A);
            ops->neg(quartic[2], quartic[2]);

            ops->add(quartic[3], xP, xP);
            ops->add(quartic[3], quartic[3], quartic[3]);
            ops->neg(quartic[3], quartic[3]);

            fe_t u;
            if (!find_one_root(u, quartic, ops, prime_bits, prime_msb))
                break;

            // Compute v from u: v^2 = u^3 + Au + B
            fe_t u2_new, u3_new, au_new, v2_new;
            ops->sq(u2_new, u);
            ops->mul(u3_new, u2_new, u);
            ops->mul(au_new, A, u);
            ops->add(v2_new, u3_new, au_new);
            ops->add(v2_new, v2_new, B);

            fe_t v;
            if (!ops->sqrt_qr(v, v2_new))
                break;

            // Verify: 2*(u,v) should give (xP, yP)
            fe_t three_u2, numer, two_v, lambda, lambda_sq, x_double;
            ops->add(three_u2, u2_new, u2_new);
            ops->add(three_u2, three_u2, u2_new);
            ops->add(numer, three_u2, A);
            ops->add(two_v, v, v);
            ops->invert(lambda, two_v);
            ops->mul(lambda, lambda, numer);
            ops->sq(lambda_sq, lambda);
            ops->add(x_double, u, u);
            ops->sub(x_double, lambda_sq, x_double);

            unsigned char xd_bytes[32], xp_bytes[32];
            ops->tobytes(xd_bytes, x_double);
            ops->tobytes(xp_bytes, xP);

            if (std::memcmp(xd_bytes, xp_bytes, 32) != 0)
            {
                // Wrong sign for v, try negating
                ops->neg(v, v);

                ops->add(two_v, v, v);
                ops->invert(lambda, two_v);
                ops->mul(lambda, lambda, numer);
                ops->sq(lambda_sq, lambda);
                ops->add(x_double, u, u);
                ops->sub(x_double, lambda_sq, x_double);

                ops->tobytes(xd_bytes, x_double);
                if (std::memcmp(xd_bytes, xp_bytes, 32) != 0)
                    break; // Neither sign works
            }

            chain_len++;
            ops->copy(xP, u);
            ops->copy(yP, v);
        }

        if (chain_len >= v2 - 1)
        {
            // (xP, yP) has order 2^v2
            ops->copy(G.X, xP);
            ops->copy(G.Y, yP);
            ops->one(G.Z);
            return 1;
        }
    }

    return 0;
}

// ============================================================================
// Coset generation
//
// The ECFFT evaluation domain is a coset of the 2^v2 subgroup: the set
// S = {R + i*G : i = 0, ..., 2^v2 - 1} where G generates the 2^v2 subgroup
// and R is an offset point NOT in that subgroup.
//
// The requirement that R ∉ <G> (more precisely, that 2^v2 * R ≠ O) ensures
// that the coset is disjoint from the subgroup and that all 2^v2 elements
// of S have distinct x-coordinates. If R were in <G>, the coset would
// collapse to <G> itself, and symmetric pairs ±P would share x-coordinates.
//
// The offset point R is found by random sampling: generate random x,
// compute y = sqrt(x^3 + ax + b), check that 2^v2 * (x,y) ≠ O.
// ============================================================================

// Find a point R not in the 2-primary subgroup (its 2-power component < 2^v2).
static int find_offset_point(JacobianPoint &R, int v2, const fe_t a, const fe_t b, const FieldOps *ops, Prng &rng)
{
    for (int attempt = 0; attempt < 10000; attempt++)
    {
        unsigned char x_bytes[32];
        rng.random_bytes(x_bytes);

        fe_t x, x2, x3, ax_val, y2_val, y;
        ops->frombytes(x, x_bytes);
        ops->sq(x2, x);
        ops->mul(x3, x2, x);
        ops->mul(ax_val, a, x);
        ops->add(y2_val, x3, ax_val);
        ops->add(y2_val, y2_val, b);

        if (!ops->sqrt_qr(y, y2_val))
            continue;

        ops->copy(R.X, x);
        ops->copy(R.Y, y);
        ops->one(R.Z);

        // Check that 2^v2 * R != O (R is not in the 2^v2 subgroup)
        JacobianPoint test;
        ops->copy(test.X, R.X);
        ops->copy(test.Y, R.Y);
        ops->copy(test.Z, R.Z);
        for (int k = 0; k < v2; k++)
        {
            JacobianPoint tmp;
            jac_dbl(tmp, test, a, ops);
            ops->copy(test.X, tmp.X);
            ops->copy(test.Y, tmp.Y);
            ops->copy(test.Z, tmp.Z);
        }

        if (!jac_is_identity(test, ops))
            return 1; // R is not in 2^v2 subgroup
    }

    return 0;
}

// Generate coset: {R + i*G : i = 0..2^v2-1}, return x-coordinates in affine.
// Output is in natural order. The ECFFT init functions apply bit-reversal
// permutation when loading this data so that even/odd pairs match isogeny fibers.
static void generate_coset(
    std::vector<unsigned char> &coset_bytes,
    const JacobianPoint &R,
    const JacobianPoint &G,
    int v2,
    const fe_t a_fe,
    const FieldOps *ops)
{
    int domain_size = 1 << v2;

    coset_bytes.resize((size_t)domain_size * 32);

    JacobianPoint current;
    ops->copy(current.X, R.X);
    ops->copy(current.Y, R.Y);
    ops->copy(current.Z, R.Z);

    for (int i = 0; i < domain_size; i++)
    {
        if (jac_is_identity(current, ops))
        {
            std::memset(&coset_bytes[(size_t)i * 32], 0, 32);
        }
        else
        {
            fe_t x_aff, y_aff;
            jac_to_affine(x_aff, y_aff, current, ops);
            ops->tobytes(&coset_bytes[(size_t)i * 32], x_aff);
        }

        JacobianPoint next;
        jac_add(next, current, G, a_fe, ops);
        ops->copy(current.X, next.X);
        ops->copy(current.Y, next.Y);
        ops->copy(current.Z, next.Z);

        if ((i + 1) % 1024 == 0)
            fprintf(stderr, "  Coset: %d / %d points\n", i + 1, domain_size);
    }
}

// ============================================================================
// Output formatting
// ============================================================================

static void print_bytes_row(const unsigned char *data, int count)
{
    for (int i = 0; i < count; i++)
    {
        if (i > 0)
            printf(", ");
        printf("0x%02x", data[i]);
    }
}

static void print_inl(
    const char *field_upper,
    const char *field_lower,
    int v2,
    int domain_size,
    int a_int,
    const std::vector<unsigned char> &coset_bytes,
    const std::vector<IsogenyData> &isogenies,
    uint64_t seed,
    const char *field_prime_hex,
    const char *b_hex,
    const char *order_hex)
{
    printf("// Auto-generated by ranshaw-gen-ecfft — DO NOT EDIT\n");
    printf("// ECFFT precomputed data for F_%s\n", field_lower);
    printf("// Field prime: 0x%s\n", field_prime_hex);
    printf("// Curve parameter a: %d\n", a_int);
    printf("// Curve parameter b: 0x%s\n", b_hex);
    if (order_hex)
        printf("// Group order: 0x%s\n", order_hex);
    printf("// Domain size: %d, Levels: %d\n", domain_size, v2);
    printf("// Seed: %" PRIu64 "\n", seed);
    printf("\n");
    printf("static const size_t ECFFT_%s_DOMAIN_SIZE = %d;\n", field_upper, domain_size);
    printf("static const size_t ECFFT_%s_LOG_DOMAIN = %d;\n", field_upper, v2);
    printf("\n");

    // Coset
    printf("static const unsigned char ECFFT_%s_COSET[%d * 32] = {\n", field_upper, domain_size);
    for (int i = 0; i < domain_size; i++)
    {
        printf("    ");
        print_bytes_row(&coset_bytes[(size_t)i * 32], 32);
        if (i < domain_size - 1)
            printf(",\n");
        else
            printf("\n");
    }
    printf("};\n");

    // Isogenies
    for (int level = 0; level < v2; level++)
    {
        printf("\n// Level %d: num degree 2, den degree 1\n", level);

        printf("static const unsigned char ECFFT_%s_ISO_NUM_%d[3 * 32] = {\n", field_upper, level);
        for (int c = 0; c < 3; c++)
        {
            printf("    ");
            print_bytes_row(isogenies[level].num[c], 32);
            if (c < 2)
                printf(",\n");
            else
                printf("\n");
        }
        printf("};\n");

        printf("static const unsigned char ECFFT_%s_ISO_DEN_%d[2 * 32] = {\n", field_upper, level);
        for (int c = 0; c < 2; c++)
        {
            printf("    ");
            print_bytes_row(isogenies[level].den[c], 32);
            if (c < 1)
                printf(",\n");
            else
                printf("\n");
        }
        printf("};\n");
    }

    // Degree arrays
    printf("\nstatic const size_t ECFFT_%s_ISO_NUM_DEGREE[%d] = {\n    ", field_upper, v2);
    for (int i = 0; i < v2; i++)
    {
        if (i > 0)
            printf(", ");
        printf("2");
    }
    printf("\n};\n");

    printf("\nstatic const size_t ECFFT_%s_ISO_DEN_DEGREE[%d] = {\n    ", field_upper, v2);
    for (int i = 0; i < v2; i++)
    {
        if (i > 0)
            printf(", ");
        printf("1");
    }
    printf("\n};\n");
}

// ============================================================================
// Hex parsing
// ============================================================================

static int hex_nibble(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

static int parse_hex_bytes(unsigned char out[32], const char *hex)
{
    // Skip 0x prefix
    if (hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X'))
        hex += 2;

    size_t len = strlen(hex);
    if (len > 64)
        return 0;

    // Pad to 64 chars
    char padded[65];
    std::memset(padded, '0', 64);
    padded[64] = '\0';
    std::memcpy(padded + 64 - len, hex, len);

    // Convert big-endian hex to little-endian bytes
    for (int i = 0; i < 32; i++)
    {
        int hi = hex_nibble(padded[62 - 2 * i]);
        int lo = hex_nibble(padded[63 - 2 * i]);
        if (hi < 0 || lo < 0)
            return 0;
        out[i] = (unsigned char)((hi << 4) | lo);
    }

    return 1;
}

// ============================================================================
// 256-bit integer helpers (little-endian byte arrays)
// ============================================================================

// Count trailing zero bits in a 256-bit LE integer.
static int count_trailing_zeros_256(const unsigned char bytes[32])
{
    for (int i = 0; i < 32; i++)
    {
        if (bytes[i] != 0)
        {
            unsigned char b = bytes[i];
            int tz = 0;
            while ((b & 1) == 0)
            {
                b >>= 1;
                tz++;
            }
            return i * 8 + tz;
        }
    }
    return 256;
}

// Right-shift a 256-bit LE integer by `shift` bits (0 <= shift <= 255).
static void right_shift_256(unsigned char out[32], const unsigned char in[32], int shift)
{
    int byte_shift = shift / 8;
    int bit_shift = shift % 8;
    std::memset(out, 0, 32);
    for (int i = 0; i < 32 - byte_shift; i++)
    {
        unsigned int lo = in[i + byte_shift];
        unsigned int hi = (i + byte_shift + 1 < 32) ? in[i + byte_shift + 1] : 0;
        out[i] = (unsigned char)((lo >> bit_shift) | (hi << (8 - bit_shift)));
    }
    if (bit_shift == 0)
    {
        for (int i = 0; i < 32 - byte_shift; i++)
            out[i] = in[i + byte_shift];
    }
}

// Return position of highest set bit + 1 (i.e. bit length).
static int bit_length_256(const unsigned char bytes[32])
{
    for (int i = 31; i >= 0; i--)
    {
        if (bytes[i] != 0)
        {
            unsigned char b = bytes[i];
            int bl = 0;
            while (b)
            {
                bl++;
                b >>= 1;
            }
            return i * 8 + bl;
        }
    }
    return 0;
}

// Get bit at position `pos` in a 256-bit LE integer.
static int get_bit_256(const unsigned char bytes[32], int pos)
{
    if (pos < 0 || pos >= 256)
        return 0;
    return (bytes[pos / 8] >> (pos % 8)) & 1;
}

// Check if a 256-bit LE integer is all zeros.
static int is_zero_256(const unsigned char bytes[32])
{
    for (int i = 0; i < 32; i++)
        if (bytes[i] != 0)
            return 0;
    return 1;
}

// ============================================================================
// Scalar multiplication (variable-time, for offline gen tool use only)
// ============================================================================

// Left-to-right double-and-add for 256-bit LE scalar.
static void jac_scalar_mul(
    JacobianPoint &R,
    const JacobianPoint &P,
    const unsigned char scalar[32],
    const fe_t a_fe,
    const FieldOps *ops)
{
    jac_identity(R, ops);

    int bl = bit_length_256(scalar);
    if (bl == 0)
        return;

    for (int i = bl - 1; i >= 0; i--)
    {
        JacobianPoint tmp;
        jac_dbl(tmp, R, a_fe, ops);
        ops->copy(R.X, tmp.X);
        ops->copy(R.Y, tmp.Y);
        ops->copy(R.Z, tmp.Z);

        if (get_bit_256(scalar, i))
        {
            jac_add(tmp, R, P, a_fe, ops);
            ops->copy(R.X, tmp.X);
            ops->copy(R.Y, tmp.Y);
            ops->copy(R.Z, tmp.Z);
        }
    }
}

// ============================================================================
// Generator finding via known group order (--known-order path)
//
// When #E is known, we can find a generator of the 2^v2 subgroup by:
//   1. Compute cofactor = #E >> v2 (odd part).
//   2. Sample random points P, compute G = cofactor * P.
//   3. Repeatedly double G to find the order of G (a power of 2).
//   4. Keep the best (highest order) generator across attempts.
//
// This handles non-cyclic 2-Sylow: if E[2^inf] = Z/2^a × Z/2^b (a <= b),
// a random point's 2-component has order 2^b with probability ~1/2 (unless
// a = b, in which case almost all points work).
// ============================================================================

static int build_generator_from_order(
    JacobianPoint &G,
    int &levels_out,
    const unsigned char order_bytes[32],
    int v2_total,
    const fe_t a_fe,
    const fe_t b_fe,
    const FieldOps *ops,
    Prng &rng)
{
    // cofactor = order >> v2_total
    unsigned char cofactor[32];
    right_shift_256(cofactor, order_bytes, v2_total);

    if (is_zero_256(cofactor))
    {
        fprintf(stderr, "ERROR: cofactor is zero (order has no odd part?)\n");
        return 0;
    }

    int best_order_exp = 0;
    JacobianPoint best_G;
    jac_identity(best_G, ops);

    for (int attempt = 0; attempt < 100; attempt++)
    {
        // Sample random point on y^2 = x^3 + ax + b
        unsigned char x_bytes[32];
        rng.random_bytes(x_bytes);

        fe_t x, x2, x3, ax_val, y2_val, y;
        ops->frombytes(x, x_bytes);
        ops->sq(x2, x);
        ops->mul(x3, x2, x);
        ops->mul(ax_val, a_fe, x);
        ops->add(y2_val, x3, ax_val);
        ops->add(y2_val, y2_val, b_fe);

        if (!ops->sqrt_qr(y, y2_val))
            continue;

        JacobianPoint P;
        ops->copy(P.X, x);
        ops->copy(P.Y, y);
        ops->one(P.Z);

        // G_candidate = cofactor * P
        JacobianPoint G_candidate;
        jac_scalar_mul(G_candidate, P, cofactor, a_fe, ops);

        if (jac_is_identity(G_candidate, ops))
            continue;

        // Measure order: double until identity
        int order_exp = 0;
        JacobianPoint test;
        ops->copy(test.X, G_candidate.X);
        ops->copy(test.Y, G_candidate.Y);
        ops->copy(test.Z, G_candidate.Z);

        for (int k = 0; k < v2_total; k++)
        {
            if (jac_is_identity(test, ops))
                break;
            order_exp++;
            JacobianPoint tmp;
            jac_dbl(tmp, test, a_fe, ops);
            ops->copy(test.X, tmp.X);
            ops->copy(test.Y, tmp.Y);
            ops->copy(test.Z, tmp.Z);
        }

        if (order_exp > best_order_exp)
        {
            best_order_exp = order_exp;
            ops->copy(best_G.X, G_candidate.X);
            ops->copy(best_G.Y, G_candidate.Y);
            ops->copy(best_G.Z, G_candidate.Z);

            fprintf(stderr, "  Attempt %d: found element of order 2^%d\n", attempt, order_exp);

            if (order_exp == v2_total)
                break; // Got maximal order
        }
    }

    if (best_order_exp < 2)
    {
        fprintf(stderr, "ERROR: Could not find generator of sufficient order\n");
        return 0;
    }

    ops->copy(G.X, best_G.X);
    ops->copy(G.Y, best_G.Y);
    ops->copy(G.Z, best_G.Z);
    levels_out = best_order_exp;
    return 1;
}

// ============================================================================
// Main
// ============================================================================

static void usage()
{
    fprintf(stderr, "Usage: ranshaw-gen-ecfft <fp|fq> --known-b 0x<hex> [options]\n\n");
    fprintf(stderr, "Generates ECFFT precomputed data (.inl file) to stdout.\n");
    fprintf(stderr, "Progress and status are printed to stderr.\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --known-b 0x<hex>      The b coefficient (required)\n");
    fprintf(stderr, "  --a N                  Curve parameter a (small integer, default: -3)\n");
    fprintf(stderr, "  --seed N               PRNG seed for deterministic output (decimal or 0x hex)\n");
    fprintf(stderr, "  --known-order 0x<hex>  Group order #E (bypasses 2-descent, enables curves\n");
    fprintf(stderr, "                         without full 2-torsion)\n\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  ranshaw-gen-ecfft fp --known-b 0x1234...\n");
    fprintf(stderr, "  ranshaw-gen-ecfft fq --known-b 0x43d2...\n");
    fprintf(stderr, "  ranshaw-gen-ecfft fp --a 1 --known-b 0x0d63 --seed 12345\n");
    fprintf(stderr, "  ranshaw-gen-ecfft fp --a 1 --known-b 0x0d63 --known-order 0x<hex> --seed 42\n");
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        usage();
        return 1;
    }

    const char *field = argv[1];
    const char *known_b_hex = nullptr;
    const char *known_order_hex = nullptr;
    int a_int = -3;
    uint64_t seed_value = 0;
    int seed_specified = 0;

    for (int i = 2; i < argc; i++)
    {
        if (strcmp(argv[i], "--known-b") == 0 && i + 1 < argc)
            known_b_hex = argv[++i];
        else if (strcmp(argv[i], "--a") == 0 && i + 1 < argc)
            a_int = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        {
            seed_value = strtoull(argv[++i], nullptr, 0);
            seed_specified = 1;
        }
        else if (strcmp(argv[i], "--known-order") == 0 && i + 1 < argc)
            known_order_hex = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            usage();
            return 0;
        }
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage();
            return 1;
        }
    }

    if (strcmp(field, "fp") != 0 && strcmp(field, "fq") != 0)
    {
        fprintf(stderr, "Unknown field: %s (use fp or fq)\n", field);
        return 1;
    }

    if (!known_b_hex)
    {
        fprintf(stderr, "Missing --known-b argument\n");
        usage();
        return 1;
    }

    unsigned char b_bytes[32];
    if (!parse_hex_bytes(b_bytes, known_b_hex))
    {
        fprintf(stderr, "Invalid hex value for --known-b: %s\n", known_b_hex);
        return 1;
    }

    unsigned char order_bytes[32];
    if (known_order_hex)
    {
        if (!parse_hex_bytes(order_bytes, known_order_hex))
        {
            fprintf(stderr, "Invalid hex value for --known-order: %s\n", known_order_hex);
            return 1;
        }
    }

    if (!seed_specified)
        seed_value = (uint64_t)time(nullptr);

    int is_fq = (strcmp(field, "fq") == 0);
    const FieldOps *ops = is_fq ? &FQ_OPS : &FP_OPS;
    const char *field_upper = is_fq ? "FQ" : "FP";
    const char *field_lower = is_fq ? "fq" : "fp";

    // Get field prime
    unsigned char field_bytes[32];
    if (is_fq)
        get_q_bytes(field_bytes);
    else
        get_p_bytes(field_bytes);

    int bits[255];
    for (int i = 0; i < 255; i++)
        bits[i] = (field_bytes[i / 8] >> (i % 8)) & 1;
    int msb = 254;
    while (msb > 0 && bits[msb] == 0)
        msb--;

    fprintf(stderr, "ECFFT Data Generator\n");
    fprintf(stderr, "====================\n\n");
    fprintf(stderr, "Field: %s\n", field);

    char b_hex[65];
    hex_string(b_hex, sizeof(b_hex), b_bytes, 32);
    fprintf(stderr, "b = 0x%s\n", b_hex);

    // Set up curve: y^2 = x^3 + ax + b
    fe_t a, b_fe;
    fe_from_int(a, a_int, ops);
    fprintf(stderr, "a = %d\n", a_int);

    ops->frombytes(b_fe, b_bytes);

    // Seed PRNG
    Prng rng;
    rng.seed(seed_value);
    fprintf(stderr, "PRNG seed: %" PRIu64 "\n", seed_value);

    int levels;
    JacobianPoint G;

    if (known_order_hex)
    {
        // --known-order path: bypass 2-descent, use cofactor multiplication
        int v2_total = count_trailing_zeros_256(order_bytes);
        fprintf(stderr, "\nUsing --known-order (v2 = %d)\n", v2_total);

        char order_hex_str[65];
        hex_string(order_hex_str, sizeof(order_hex_str), order_bytes, 32);
        fprintf(stderr, "  Order = 0x%s\n", order_hex_str);

        if (v2_total < 2)
        {
            fprintf(stderr, "ERROR: v2(#E) = %d, need at least 2 for ECFFT\n", v2_total);
            return 1;
        }

        fprintf(stderr, "\nFinding generator of maximal 2-power order via cofactor multiplication...\n");

        if (!build_generator_from_order(G, levels, order_bytes, v2_total, a, b_fe, ops, rng))
        {
            fprintf(stderr, "ERROR: Failed to find generator!\n");
            return 1;
        }

        fprintf(stderr, "  Generator order: 2^%d\n", levels);
    }
    else
    {
        // Existing 2-descent path
        // Step 1: Verify full 2-torsion
        fprintf(stderr, "\nStep 1: Checking full 2-torsion...\n");
        if (!check_full_2torsion(a, b_fe, bits, msb, ops))
        {
            fprintf(stderr, "ERROR: Curve does not have full 2-torsion!\n");
            fprintf(stderr, "Hint: use --known-order to bypass 2-descent for curves without full 2-torsion.\n");
            return 1;
        }
        fprintf(stderr, "  Full 2-torsion confirmed.\n");

        // Step 2: Find roots and compute v2
        fprintf(stderr, "\nStep 2: Finding 2-torsion roots and computing v2...\n");
        fe_t roots[3];
        if (!find_cubic_roots(roots, a, b_fe, bits, msb, ops))
        {
            fprintf(stderr, "ERROR: Failed to find cubic roots!\n");
            return 1;
        }

        for (int i = 0; i < 3; i++)
        {
            unsigned char r_bytes[32];
            ops->tobytes(r_bytes, roots[i]);
            char r_hex[65];
            hex_string(r_hex, sizeof(r_hex), r_bytes, 32);
            fprintf(stderr, "  Root %d: 0x%s\n", i, r_hex);
        }

        int chains[3];
        int v2 = compute_v2(a, b_fe, roots, ops, bits, msb, chains);
        fprintf(stderr, "  Halving chains: [%d, %d, %d]\n", chains[0], chains[1], chains[2]);
        fprintf(stderr, "  v2(#E) = %d\n", v2);

        // The 2-Sylow subgroup is Z/2^(min_chain+1) × Z/2^(max_chain+1).
        // The ECFFT uses the larger cyclic factor.
        int max_chain = chains[0];
        int max_chain_idx = 0;
        for (int i = 1; i < 3; i++)
        {
            if (chains[i] > max_chain)
            {
                max_chain = chains[i];
                max_chain_idx = i;
            }
        }

        levels = max_chain + 1; // exponent of the larger cyclic factor
        fprintf(stderr, "  ECFFT levels = %d (from root %d, chain depth %d)\n", levels, max_chain_idx, max_chain);

        // Step 3: Find generator G of order 2^levels via halving chain
        fprintf(stderr, "\nStep 3: Finding generator of order 2^%d via halving chain...\n", levels);

        if (!build_generator_from_halving(G, levels, roots, a, b_fe, ops, bits, msb))
        {
            fprintf(stderr, "ERROR: Failed to find generator of order 2^%d!\n", levels);
            return 1;
        }
    }

    int domain_size = 1 << levels;
    fprintf(stderr, "  Domain size = %d\n", domain_size);

    if (levels < 2)
    {
        fprintf(stderr, "ERROR: ECFFT levels too small (need at least 2)\n");
        return 1;
    }

    {
        fe_t gx, gy;
        jac_to_affine(gx, gy, G, ops);
        unsigned char gx_bytes[32];
        ops->tobytes(gx_bytes, gx);
        char gx_hex[65];
        hex_string(gx_hex, sizeof(gx_hex), gx_bytes, 32);
        fprintf(stderr, "  Generator G.x = 0x%s\n", gx_hex);
    }

    // Step 4: Build isogeny chain
    fprintf(stderr, "\nStep 4: Building isogeny chain (%d levels)...\n", levels);
    std::vector<IsogenyData> isogenies(levels);

    fe_t cur_a, cur_b;
    ops->copy(cur_a, a);
    ops->copy(cur_b, b_fe);

    JacobianPoint cur_G;
    ops->copy(cur_G.X, G.X);
    ops->copy(cur_G.Y, G.Y);
    ops->copy(cur_G.Z, G.Z);

    for (int level = 0; level < levels; level++)
    {
        // Kernel point K = 2^(levels - level - 1) * cur_G, which has order 2
        JacobianPoint K;
        ops->copy(K.X, cur_G.X);
        ops->copy(K.Y, cur_G.Y);
        ops->copy(K.Z, cur_G.Z);

        for (int k = 0; k < levels - level - 1; k++)
        {
            JacobianPoint tmp;
            jac_dbl(tmp, K, cur_a, ops);
            ops->copy(K.X, tmp.X);
            ops->copy(K.Y, tmp.Y);
            ops->copy(K.Z, tmp.Z);
        }

        // K is order 2, so K = (x0, 0) in affine
        fe_t x0, y0;
        jac_to_affine(x0, y0, K, ops);

        unsigned char x0_bytes[32];
        ops->tobytes(x0_bytes, x0);
        char x0_hex[65];
        hex_string(x0_hex, sizeof(x0_hex), x0_bytes, 32);
        fprintf(stderr, "  Level %d: kernel x0 = 0x%s\n", level, x0_hex);

        // Apply Velu isogeny
        fe_t new_a, new_b;
        JacobianPoint new_G;
        velu_2isogeny(new_a, new_b, isogenies[level], new_G, x0, cur_a, cur_b, cur_G, ops);

        ops->copy(cur_a, new_a);
        ops->copy(cur_b, new_b);
        ops->copy(cur_G.X, new_G.X);
        ops->copy(cur_G.Y, new_G.Y);
        ops->copy(cur_G.Z, new_G.Z);
    }
    fprintf(stderr, "  Isogeny chain complete.\n");

    // Step 5: Generate coset
    fprintf(stderr, "\nStep 5: Generating coset (%d points)...\n", domain_size);

    JacobianPoint R;
    if (!find_offset_point(R, levels, a, b_fe, ops, rng))
    {
        fprintf(stderr, "ERROR: Failed to find offset point!\n");
        return 1;
    }

    {
        fe_t rx, ry;
        jac_to_affine(rx, ry, R, ops);
        unsigned char rx_bytes[32];
        ops->tobytes(rx_bytes, rx);
        char rx_hex[65];
        hex_string(rx_hex, sizeof(rx_hex), rx_bytes, 32);
        fprintf(stderr, "  Offset R.x = 0x%s\n", rx_hex);
    }

    std::vector<unsigned char> coset_bytes;
    generate_coset(coset_bytes, R, G, levels, a, ops);
    fprintf(stderr, "  Coset generation complete.\n");

    // Step 6: Output .inl
    fprintf(stderr, "\nStep 6: Writing .inl to stdout...\n");

    // Compute field prime hex string for the .inl header
    char field_hex[65];
    hex_string(field_hex, sizeof(field_hex), field_bytes, 32);

    // Compute order hex string (only available when --known-order was used)
    char order_hex_buf[65];
    const char *order_hex_for_header = nullptr;
    if (known_order_hex)
    {
        hex_string(order_hex_buf, sizeof(order_hex_buf), order_bytes, 32);
        order_hex_for_header = order_hex_buf;
    }

    print_inl(
        field_upper,
        field_lower,
        levels,
        domain_size,
        a_int,
        coset_bytes,
        isogenies,
        seed_value,
        field_hex,
        b_hex,
        order_hex_for_header);
    fprintf(stderr, "Done.\n");

    return 0;
}
