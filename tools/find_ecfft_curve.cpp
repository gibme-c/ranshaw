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
 * @file find_ecfft_curve.cpp
 * @brief Fast auxiliary curve search for ECFFT over GF(q) and GF(p).
 *
 * Searches for elliptic curves E: y^2 = x^3 + ax + b (a configurable, default
 * -3) whose group order #E(GF(p)) has high 2-adic valuation v2(#E), suitable
 * as ECFFT evaluation domains per [BCKL23]. Computes v2 natively via 2-descent
 * (halving chains) — no SageMath or point counting needed.
 *
 * References:
 *   [BCKL23]  Ben-Sasson, Carmon, Kopparty, Levit. "Elliptic Curve Fast
 *             Fourier Transform (ECFFT) Part I." https://arxiv.org/abs/2107.08473
 *   [Cass91]  J.W.S. Cassels. "Lectures on Elliptic Curves." London Math Soc
 *             Student Texts 24 (1991). — 2-descent theory.
 *   [ST92]    Silverman, Tate. "Rational Points on Elliptic Curves." Springer.
 *   [CZ81]    Cantor, Zassenhaus. "A new algorithm for factoring polynomials
 *             over finite fields." Math. Comp. 36 (1981). — Polynomial splitting.
 *
 * Strategy:
 *   1. Generate random b (a configurable via --a), check discriminant 4a^3 + 27b^2 != 0.
 *   2. Check if x^3+ax+b splits completely over GF(p) (full 2-torsion, v2 >= 2).
 *      This is tested by computing x^p mod (x^3+ax+b) and checking if it equals x.
 *   3. Extract 2-torsion roots via Frobenius/Legendre splitting (Cantor-Zassenhaus).
 *   4. Compute halving chains to determine exact v2(#E) via 2-descent.
 *   5. Filter by --min-v2 threshold.
 *
 * The probability of finding v2 >= k among full-2-torsion curves is roughly
 * 1/2^(k-2), so finding v2 >= 12 typically requires ~65K full-2-torsion
 * candidates (~400K total random curves, since ~1/6 have full 2-torsion).
 *
 * Why a = -3 (default)?
 *   The Ran/Shaw curves use short Weierstrass form y^2 = x^3 - 3x + b.
 *   Fixing a = -3 lets us reuse the a = -3 optimized doubling formula
 *   (M = 3*(X-Z^2)*(X+Z^2), saving one multiplication per doubling) and
 *   ensures the auxiliary ECFFT curve is in the same family. Other values
 *   of a (e.g. a = 1) are also supported via --a for broader curve search.
 *
 * Usage:
 *   ranshaw-find-ecfft [--field fp|fq] [--a N] [--trials N] [--cpus auto|N]
 *                          [--min-v2 N]
 */

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>
#include <vector>

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

// ============================================================================
// PRNG (xoshiro256** by Blackman & Vigna, 2018) — per-thread instance
// Used only for random curve coefficient generation. Not cryptographic.
// Seeded deterministically per-thread for reproducibility.
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
// Shared state
// ============================================================================

static std::atomic<uint64_t> g_trials_done {0};
static std::atomic<int> g_found {0};
static std::atomic<int> g_best_levels {0};
static std::mutex g_print_mutex;
static std::atomic<bool> g_stop {false};
static FILE *g_outfile = nullptr;

struct Candidate
{
    unsigned char b[32];
    int v2; // 2-adic valuation of #E (= a + b where 2-Sylow ≅ Z/2^a × Z/2^b)
    int levels; // ECFFT domain exponent (= b = max_chain + 1, the larger cyclic factor)
};

static std::vector<Candidate> g_candidates;

// ============================================================================
// Field ops vtable — generic dispatch for Fp or Fq arithmetic
//
// Avoids C++ template issues with array typedefs (fe_t = uint64_t[5]).
// Each function pointer dispatches to the appropriate field implementation.
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
    int (*sqrt_qr)(fe_t out, const fe_t z); // returns 1 if QR, 0 if not
};

// Fp sqrt wrapper: fp_sqrt returns 0 on success (QR), -1 on failure
static int fp_sqrt_qr(fe_t out, const fe_t z)
{
    return (fp_sqrt(out, z) == 0) ? 1 : 0;
}

// Fq sqrt wrapper: fq_sqrt always computes z^((q+1)/4), must verify by squaring
static int fq_sqrt_qr(fe_t out, const fe_t z)
{
    fq_sqrt(out, z);
    // Verify: out^2 == z?
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
// Polynomial arithmetic mod cubic: GF(p)[x] / (x^3 + ax + b)
//
// For checking full 2-torsion: the 2-torsion points of E: y^2 = x^3+ax+b
// are (r, 0) where r is a root of x^3+ax+b. Full 2-torsion means all three
// roots are in GF(p), i.e., x^3+ax+b splits completely over GF(p).
//
// Test: compute x^p mod (x^3+ax+b) via repeated squaring in the quotient
// ring. If x^p ≡ x, all roots are in GF(p) (since x^p = x for all x ∈ GF(p)
// is the defining property of the Frobenius endomorphism).
//
// Root extraction: gcd(x^{(p-1)/2} - 1, x^3+ax+b) gives the product of
// (x - r) where r is a quadratic residue among the roots. This splits the
// cubic into degree-1 or degree-2 factors from which roots can be read off.
// See Cantor-Zassenhaus [CZ81] and [Cass91] §8.
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
// Polynomial arithmetic mod quartic: GF(p)[x] / q(x)
// where q(x) = x^4 + c3*x^3 + c2*x^2 + c1*x + c0  (monic degree 4)
// Elements are degree < 4: f[0] + f[1]*x + f[2]*x^2 + f[3]*x^3
//
// Used to find roots of the halving quartic (see halving_chain below).
// The quartic arises from the doubling formula: given P = (xP, yP),
// the x-coordinates u of points Q with 2Q = P satisfy a monic quartic.
// Finding roots of this quartic over GF(p) tells us whether P is halvable.
// ============================================================================

// Reduce degree-6 polynomial (7 coefficients d[0..6]) mod monic quartic q[0..3]
// x^4 = -c3*x^3 - c2*x^2 - c1*x - c0
static void poly4_reduce(fe_t d[7], const fe_t q[4], const FieldOps *ops)
{
    // Reduce d[6]*x^6: x^6 = x^2 * x^4 = x^2 * (-c3*x^3 - c2*x^2 - c1*x - c0)
    // But easier: reduce from top down. x^6 mod quartic can cascade.
    // Actually, reduce one degree at a time from top:
    // d[6]*x^6: multiply d[6] by the reduction of x^6, but that's complex.
    // Instead: iteratively reduce x^6, then x^5, then x^4.
    fe_t t;

    // Reduce x^6 coefficient: x^6 = x^2 * x^4 = x^2 * (-q[3]*x^3 - q[2]*x^2 - q[1]*x - q[0])
    // = -q[3]*x^5 - q[2]*x^4 - q[1]*x^3 - q[0]*x^2
    // So d[6]*x^6 adds: d[6]*(-q[3]) to x^5, d[6]*(-q[2]) to x^4, d[6]*(-q[1]) to x^3, d[6]*(-q[0]) to x^2
    ops->mul(t, d[6], q[3]);
    ops->sub(d[5], d[5], t);
    ops->mul(t, d[6], q[2]);
    ops->sub(d[4], d[4], t);
    ops->mul(t, d[6], q[1]);
    ops->sub(d[3], d[3], t);
    ops->mul(t, d[6], q[0]);
    ops->sub(d[2], d[2], t);

    // Reduce x^5: x^5 = x * x^4 = -q[3]*x^4 - q[2]*x^3 - q[1]*x^2 - q[0]*x
    ops->mul(t, d[5], q[3]);
    ops->sub(d[4], d[4], t);
    ops->mul(t, d[5], q[2]);
    ops->sub(d[3], d[3], t);
    ops->mul(t, d[5], q[1]);
    ops->sub(d[2], d[2], t);
    ops->mul(t, d[5], q[0]);
    ops->sub(d[1], d[1], t);

    // Reduce x^4: x^4 = -q[3]*x^3 - q[2]*x^2 - q[1]*x - q[0]
    ops->mul(t, d[4], q[3]);
    ops->sub(d[3], d[3], t);
    ops->mul(t, d[4], q[2]);
    ops->sub(d[2], d[2], t);
    ops->mul(t, d[4], q[1]);
    ops->sub(d[1], d[1], t);
    ops->mul(t, d[4], q[0]);
    ops->sub(d[0], d[0], t);
}

// Square a degree-3 polynomial mod quartic
static void poly4_sq(fe_t r[4], const fe_t f[4], const fe_t q[4], const FieldOps *ops)
{
    fe_t d[7];
    fe_t t;

    // d[0] = f0^2
    ops->sq(d[0], f[0]);
    // d[1] = 2*f0*f1
    ops->mul(t, f[0], f[1]);
    ops->add(d[1], t, t);
    // d[2] = f1^2 + 2*f0*f2
    ops->sq(d[2], f[1]);
    ops->mul(t, f[0], f[2]);
    ops->add(t, t, t);
    ops->add(d[2], d[2], t);
    // d[3] = 2*f1*f2 + 2*f0*f3
    ops->mul(d[3], f[1], f[2]);
    ops->add(d[3], d[3], d[3]);
    ops->mul(t, f[0], f[3]);
    ops->add(t, t, t);
    ops->add(d[3], d[3], t);
    // d[4] = f2^2 + 2*f1*f3
    ops->sq(d[4], f[2]);
    ops->mul(t, f[1], f[3]);
    ops->add(t, t, t);
    ops->add(d[4], d[4], t);
    // d[5] = 2*f2*f3
    ops->mul(t, f[2], f[3]);
    ops->add(d[5], t, t);
    // d[6] = f3^2
    ops->sq(d[6], f[3]);

    poly4_reduce(d, q, ops);

    ops->copy(r[0], d[0]);
    ops->copy(r[1], d[1]);
    ops->copy(r[2], d[2]);
    ops->copy(r[3], d[3]);
}

// Multiply by x mod quartic: shift up, reduce x^4
static void poly4_mulx(fe_t r[4], const fe_t f[4], const fe_t q[4], const FieldOps *ops)
{
    // f[3]*x^4 + f[2]*x^3 + f[1]*x^2 + f[0]*x
    // x^4 = -q[3]*x^3 - q[2]*x^2 - q[1]*x - q[0]
    // So: (f[2] - f[3]*q[3])*x^3 + (f[1] - f[3]*q[2])*x^2
    //   + (f[0] - f[3]*q[1])*x + (-f[3]*q[0])
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

// Multiply two degree-3 polynomials mod quartic
static void poly4_mul(fe_t r[4], const fe_t f[4], const fe_t g[4], const fe_t q[4], const FieldOps *ops)
{
    fe_t d[7];
    fe_t t;

    // Schoolbook: d[k] = sum_{i+j=k} f[i]*g[j]
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

// Compute x^p mod quartic via square-and-multiply
static void poly4_powx_p(fe_t result[4], const int *bits, int msb, const fe_t q[4], const FieldOps *ops)
{
    // Start with 1
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

// Compute base^exp mod quartic via square-and-multiply (general base)
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

// ============================================================================
// Compute (p-1)/2 in binary from prime bits
//
// Used for Legendre symbol / Euler criterion computations:
//   a^{(p-1)/2} ≡ 1  if a is a QR mod p
//   a^{(p-1)/2} ≡ -1 if a is a QNR mod p
// ============================================================================

// Given the binary representation of a prime p (bits[0..254], little-endian),
// compute (p-1)/2 in binary. Returns the MSB index.
static int compute_pm1_half_bits(int pm1_half_bits[255], const int *prime_bits)
{
    // Compute p-1 bits (subtract 1 from bit 0)
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

    // Shift right by 1
    for (int i = 0; i < 255; i++)
        pm1_half_bits[i] = pm1_bits[i + 1];

    int pm1_half_msb = 253;
    while (pm1_half_msb > 0 && pm1_half_bits[pm1_half_msb] == 0)
        pm1_half_msb--;

    return pm1_half_msb;
}

// ============================================================================
// Polynomial GCD — Euclidean algorithm for univariate polynomials over GF(p)
//
// Standard algorithm: repeatedly divide the larger polynomial by the smaller
// until the remainder is zero. The last nonzero remainder is the GCD.
// Used throughout for Frobenius-based polynomial splitting.
// ============================================================================

// Return the degree of polynomial p[0..max_deg], where p[i] is coefficient of x^i
// Returns -1 for zero polynomial
static int poly_degree(const fe_t *p, int max_deg, const FieldOps *ops)
{
    for (int i = max_deg; i >= 0; i--)
    {
        if (ops->isnonzero(p[i]))
            return i;
    }
    return -1;
}

// Compute gcd of two polynomials a (degree <= da) and b (degree <= db)
// Result stored in g with degree returned. g must have space for max(da,db)+1 elements.
// Uses Euclidean algorithm. Modifies a and b in place.
static int poly_gcd(fe_t *a, int da, fe_t *b, int db, fe_t *g, const FieldOps *ops)
{
    // Make a the higher-degree one
    if (da < db)
    {
        fe_t *tmp_p = a;
        a = b;
        b = tmp_p;
        int tmp_d = da;
        da = db;
        db = tmp_d;
    }

    // Euclidean algorithm
    while (db >= 0)
    {
        // a = a mod b
        int deg_a = poly_degree(a, da, ops);
        int deg_b = poly_degree(b, db, ops);

        if (deg_b < 0)
            break;

        while (deg_a >= deg_b)
        {
            // Subtract (lc_a / lc_b) * x^(deg_a - deg_b) * b from a
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

        // Swap a, b
        fe_t *tmp_p = a;
        a = b;
        b = tmp_p;
        int tmp_d = da;
        da = db;
        db = tmp_d;
        da = poly_degree(a, da, ops);
        db = poly_degree(b, db, ops);
    }

    // a is the gcd
    int deg = poly_degree(a, da, ops);
    if (deg < 0)
    {
        ops->zero(g[0]);
        return -1;
    }

    // Make monic
    fe_t inv_lc;
    ops->invert(inv_lc, a[deg]);
    for (int i = 0; i <= deg; i++)
        ops->mul(g[i], a[i], inv_lc);

    return deg;
}

// Find one root of a polynomial of degree 1..4.
// For degree 1: root = -p[0]/p[1] (already monic from GCD, so p[1]=1)
// For degree 2: use quadratic formula (need sqrt)
// For degree 3+: use Frobenius splitting (x^p mod poly) to factor further
// Returns 1 if a root was found and stored in root, 0 otherwise.
static int
    find_root_of_gcd(fe_t root, const fe_t *g, int deg, const int *prime_bits, int prime_msb, const FieldOps *ops)
{
    if (deg == 1)
    {
        // g(x) = x + g[0] (monic), root = -g[0]
        ops->neg(root, g[0]);
        return 1;
    }

    if (deg == 2)
    {
        // g(x) = x^2 + g[1]*x + g[0] (monic)
        // Roots: (-g[1] +/- sqrt(g[1]^2 - 4*g[0])) / 2
        fe_t disc, g1_sq, four_g0, s, inv2, neg_g1;
        ops->sq(g1_sq, g[1]);
        ops->add(four_g0, g[0], g[0]);
        ops->add(four_g0, four_g0, four_g0);
        ops->sub(disc, g1_sq, four_g0);

        if (!ops->sqrt_qr(s, disc))
            return 0;

        ops->neg(neg_g1, g[1]);
        ops->add(root, neg_g1, s);
        // Divide by 2: multiply by inverse of 2
        fe_t two;
        ops->one(two);
        ops->add(two, two, two);
        ops->invert(inv2, two);
        ops->mul(root, root, inv2);
        return 1;
    }

    // For degree 3 or 4: split via x^p mod g(x) to get a factor
    // This is rare in practice but handle it for correctness
    if (deg == 3)
    {
        // Compute x^p mod g(x) using polymod3
        // g(x) = x^3 + g[2]*x^2 + g[1]*x + g[0] (monic)
        // In polymod3, the modulus is x^3 + ax + b (no x^2 term).
        // Our polynomial has an x^2 term, so we need to use poly4 approach
        // adapted for degree 3. Actually, let's just use the quartic code
        // with the quartic being our cubic padded.
        // Simpler: treat as quartic x^4 + 0*x^3 + g[2]*x^2 + g[1]*x + g[0]
        // but that changes the polynomial. Instead, do it manually.

        // Use substitution: let x = t - g[2]/3 to eliminate x^2 term.
        // Or just use the same GCD-based approach recursively.
        // For simplicity, use random splitting: compute gcd(g(x), (x+r)^p - (x+r))
        // for a random r. But we don't have good random here.

        // Actually, the simplest approach: compute x^p mod g(x) directly.
        // We need polymod for degree-3 monic with x^2 coefficient.
        // Let's write a general version.

        // General square mod monic cubic g(x) = x^3 + g2*x^2 + g1*x + g0:
        // x^3 = -g2*x^2 - g1*x - g0
        fe_t xp[3]; // result mod cubic
        ops->one(xp[0]);
        ops->zero(xp[1]);
        ops->zero(xp[2]);

        for (int i = prime_msb; i >= 0; i--)
        {
            // Square xp mod cubic
            fe_t f0 = {0}, f1 = {0}, f2 = {0};
            ops->copy(f0, xp[0]);
            ops->copy(f1, xp[1]);
            ops->copy(f2, xp[2]);

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

            // Reduce x^4: d4 * (-g2*x^2 - g1*x - g0)
            ops->mul(t, d4, g[2]);
            ops->sub(d2, d2, t);
            ops->mul(t, d4, g[1]);
            ops->sub(d1, d1, t);
            ops->mul(t, d4, g[0]);
            ops->sub(d0, d0, t);
            // Reduce x^3: d3 * (-g2*x^2 - g1*x - g0)
            ops->mul(t, d3, g[2]);
            ops->sub(d2, d2, t);
            ops->mul(t, d3, g[1]);
            ops->sub(d1, d1, t);
            ops->mul(t, d3, g[0]);
            ops->sub(d0, d0, t);

            ops->copy(xp[0], d0);
            ops->copy(xp[1], d1);
            ops->copy(xp[2], d2);

            if (prime_bits[i])
            {
                // Multiply by x: xp = xp * x mod cubic
                // f2*x^3 + f1*x^2 + f0*x
                // = f2*(-g2*x^2 - g1*x - g0) + f1*x^2 + f0*x
                fe_t n0, n1, n2;
                ops->mul(n0, xp[2], g[0]);
                ops->neg(n0, n0);
                ops->mul(t, xp[2], g[1]);
                ops->sub(n1, xp[0], t);
                ops->mul(t, xp[2], g[2]);
                ops->sub(n2, xp[1], t);
                ops->copy(xp[0], n0);
                ops->copy(xp[1], n1);
                ops->copy(xp[2], n2);
            }
        }

        // h(x) = x^p - x mod cubic
        fe_t one_fe;
        ops->one(one_fe);
        ops->sub(xp[1], xp[1], one_fe);

        // gcd(h, g)
        // h has degree <= 2, g has degree 3
        // Copy both for GCD (it modifies in place)
        fe_t aa[4], bb[3];
        ops->copy(aa[0], g[0]);
        ops->copy(aa[1], g[1]);
        ops->copy(aa[2], g[2]);
        ops->one(aa[3]); // monic
        ops->copy(bb[0], xp[0]);
        ops->copy(bb[1], xp[1]);
        ops->copy(bb[2], xp[2]);

        fe_t gg[4];
        int gd = poly_gcd(aa, 3, bb, 2, gg, ops);

        if (gd >= 1 && gd < 3)
            return find_root_of_gcd(root, gg, gd, prime_bits, prime_msb, ops);

        return 0;
    }

    // deg == 4: shouldn't happen since we take gcd with x^p-x first
    return 0;
}

// Helper: try to extract a root from a degree-1..3 factor via find_root_of_gcd.
// If the factor has degree 4 (= the quartic itself) or 0, returns 0.
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

// Find one root of monic quartic q(x) = x^4 + q[3]*x^3 + q[2]*x^2 + q[1]*x + q[0]
// Returns 1 if root found, 0 if no roots in GF(p).
static int find_one_root(fe_t root, const fe_t quartic[4], const FieldOps *ops, const int *prime_bits, int prime_msb)
{
    // Compute h(x) = x^p mod quartic
    fe_t xp[4];
    poly4_powx_p(xp, prime_bits, prime_msb, quartic, ops);

    // h(x) - x
    fe_t one_fe;
    ops->one(one_fe);
    ops->sub(xp[1], xp[1], one_fe);

    // Check if x^p - x ≡ 0 mod quartic (all 4 roots are in GF(p))
    int xp_deg = poly_degree(xp, 3, ops);

    if (xp_deg >= 0)
    {
        // Partial splitting: gcd(x^p - x, quartic) gives a non-trivial factor
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

        // deg == 4: fall through to Legendre splitting below
    }

    // All 4 roots are in GF(p): x^p ≡ x mod quartic.
    // Use Legendre splitting: gcd(x^((p-1)/2) - 1, quartic) gives a
    // non-trivial factor (the roots that are QRs).
    int pm1_half_bits[255];
    int pm1_half_msb = compute_pm1_half_bits(pm1_half_bits, prime_bits);

    // Compute x^((p-1)/2) mod quartic
    fe_t xph[4];
    poly4_powx_p(xph, pm1_half_bits, pm1_half_msb, quartic, ops);

    // Try gcd(x^((p-1)/2) - 1, quartic)
    {
        fe_t a_poly[5], b_poly[4];
        ops->copy(a_poly[0], quartic[0]);
        ops->copy(a_poly[1], quartic[1]);
        ops->copy(a_poly[2], quartic[2]);
        ops->copy(a_poly[3], quartic[3]);
        ops->one(a_poly[4]);

        ops->copy(b_poly[0], xph[0]);
        ops->sub(b_poly[0], b_poly[0], one_fe); // -1
        ops->copy(b_poly[1], xph[1]);
        ops->copy(b_poly[2], xph[2]);
        ops->copy(b_poly[3], xph[3]);

        fe_t g[5];
        int deg = poly_gcd(a_poly, 4, b_poly, 3, g, ops);

        if (try_extract_root_from_factor(root, g, deg, prime_bits, prime_msb, ops))
            return 1;
    }

    // Try gcd(x^((p-1)/2) + 1, quartic)
    {
        fe_t a_poly[5], b_poly[4];
        ops->copy(a_poly[0], quartic[0]);
        ops->copy(a_poly[1], quartic[1]);
        ops->copy(a_poly[2], quartic[2]);
        ops->copy(a_poly[3], quartic[3]);
        ops->one(a_poly[4]);

        ops->copy(b_poly[0], xph[0]);
        ops->add(b_poly[0], b_poly[0], one_fe); // +1
        ops->copy(b_poly[1], xph[1]);
        ops->copy(b_poly[2], xph[2]);
        ops->copy(b_poly[3], xph[3]);

        fe_t g[5];
        int deg = poly_gcd(a_poly, 4, b_poly, 3, g, ops);

        if (try_extract_root_from_factor(root, g, deg, prime_bits, prime_msb, ops))
            return 1;
    }

    // Fallback: shift by constant c, compute gcd((x+c)^((p-1)/2) - 1, quartic)
    for (int c_val = 1; c_val <= 10; c_val++)
    {
        // base = x + c
        fe_t base[4];
        ops->zero(base[0]);
        ops->one(base[1]);
        ops->zero(base[2]);
        ops->zero(base[3]);
        for (int k = 0; k < c_val; k++)
            ops->add(base[0], base[0], one_fe);

        // Compute (x+c)^((p-1)/2) mod quartic
        fe_t result[4];
        poly4_pow(result, base, pm1_half_bits, pm1_half_msb, quartic, ops);

        // gcd(result - 1, quartic)
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
// Root extraction from cubic x^3 + ax + b (assumes it splits completely)
//
// Strategy (Cantor-Zassenhaus [CZ81]):
//   1. Compute h(x) = x^{(p-1)/2} mod (x^3+ax+b).
//   2. gcd(h(x) - 1, x^3+ax+b) = product of (x - r_i) where r_i is a QR.
//      This splits the cubic into a degree-1 and degree-2 factor (or 0 and 3).
//   3. If the split is trivial (all roots are QRs or all QNRs), try h(x) + 1,
//      or use a shifted element (x+c)^{(p-1)/2} for random c = 1, 2, ...
//   4. Extract roots from the factors using the quadratic formula.
//
// For a depressed cubic x^3 + ax + b (no x^2 term), the sum of roots is 0,
// so given any two roots, the third is their negated sum.
// ============================================================================

// Find all 3 roots of x^3 + ax + b over GF(p).
// Assumes the cubic splits completely. Uses gcd(x^p - x, cubic) approach.
// Returns 1 on success, 0 on failure.
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

    // Compute x^p mod (x^3 + a*x + b)
    fe_t xp[3];
    polymod3_powx(xp, prime_bits, prime_msb, neg_a, neg_b, ops);

    // h(x) = x^p - x
    fe_t one_fe;
    ops->one(one_fe);
    ops->sub(xp[1], xp[1], one_fe);

    // Since cubic splits completely, gcd(x^p - x, cubic) = cubic itself.
    // Use Legendre splitting: gcd(x^((p-1)/2) - 1, cubic) to get a
    // degree-1 or degree-2 factor.

    int pm1_half_bits[255];
    int pm1_half_msb = compute_pm1_half_bits(pm1_half_bits, prime_bits);

    // Compute x^((p-1)/2) mod cubic
    fe_t xph[3];
    polymod3_powx(xph, pm1_half_bits, pm1_half_msb, neg_a, neg_b, ops);

    // g1(x) = x^((p-1)/2) - 1 mod cubic
    ops->sub(xph[0], xph[0], one_fe);

    // gcd(g1, cubic)
    fe_t aa[4], bb[3];
    ops->copy(aa[0], b);
    ops->copy(aa[1], a);
    ops->zero(aa[2]);
    ops->one(aa[3]);

    ops->copy(bb[0], xph[0]);
    ops->copy(bb[1], xph[1]);
    ops->copy(bb[2], xph[2]);

    fe_t g[4];
    int deg = poly_gcd(aa, 3, bb, 2, g, ops);

    if (deg == 1)
    {
        // Found one root: -g[0]
        ops->neg(roots[0], g[0]);

        // Divide cubic by (x - roots[0]) to get quadratic
        // x^3 + ax + b = (x - r)(x^2 + r*x + (r^2 + a))
        fe_t r_sq;
        ops->sq(r_sq, roots[0]);
        fe_t quad_c1; // coefficient of x in quadratic
        ops->copy(quad_c1, roots[0]);
        fe_t quad_c0; // constant term
        ops->add(quad_c0, r_sq, a);

        // Roots of x^2 + quad_c1*x + quad_c0:
        // disc = quad_c1^2 - 4*quad_c0 = r^2 - 4*(r^2 + a) = -3*r^2 - 4*a
        fe_t disc, s, three_r_sq, four_a;
        ops->add(three_r_sq, r_sq, r_sq);
        ops->add(three_r_sq, three_r_sq, r_sq);
        ops->add(four_a, a, a);
        ops->add(four_a, four_a, four_a);
        ops->neg(disc, three_r_sq);
        ops->sub(disc, disc, four_a);

        if (!ops->sqrt_qr(s, disc))
            return 0; // shouldn't happen if cubic splits

        fe_t neg_c1, inv2, two;
        ops->neg(neg_c1, quad_c1);
        ops->one(two);
        ops->add(two, two, two);
        ops->invert(inv2, two);

        ops->add(roots[1], neg_c1, s);
        ops->mul(roots[1], roots[1], inv2);

        ops->sub(roots[2], neg_c1, s);
        ops->mul(roots[2], roots[2], inv2);

        return 1;
    }
    else if (deg == 2)
    {
        // g(x) = x^2 + g[1]*x + g[0] has 2 roots (the QRs)
        // The third root is from the linear cofactor
        fe_t disc, s, inv2, two, neg_g1;
        ops->sq(disc, g[1]);
        fe_t four_g0;
        ops->add(four_g0, g[0], g[0]);
        ops->add(four_g0, four_g0, four_g0);
        ops->sub(disc, disc, four_g0);

        if (!ops->sqrt_qr(s, disc))
            return 0;

        ops->neg(neg_g1, g[1]);
        ops->one(two);
        ops->add(two, two, two);
        ops->invert(inv2, two);

        ops->add(roots[0], neg_g1, s);
        ops->mul(roots[0], roots[0], inv2);

        ops->sub(roots[1], neg_g1, s);
        ops->mul(roots[1], roots[1], inv2);

        // Third root: sum of all roots = 0 (no x^2 term in cubic), so r2 = -r0 - r1
        ops->add(roots[2], roots[0], roots[1]);
        ops->neg(roots[2], roots[2]);

        return 1;
    }
    else if (deg == 0)
    {
        // gcd is 1 — x^((p-1)/2) - 1 shares no roots with cubic
        // All roots are QNRs? That means x^((p-1)/2) = -1 for all roots.
        // Try x^((p-1)/2) + 1 instead.
        // Recompute xph
        polymod3_powx(xph, pm1_half_bits, pm1_half_msb, neg_a, neg_b, ops);
        ops->add(xph[0], xph[0], one_fe);

        ops->copy(aa[0], b);
        ops->copy(aa[1], a);
        ops->zero(aa[2]);
        ops->one(aa[3]);

        ops->copy(bb[0], xph[0]);
        ops->copy(bb[1], xph[1]);
        ops->copy(bb[2], xph[2]);

        deg = poly_gcd(aa, 3, bb, 2, g, ops);

        if (deg == 1)
        {
            ops->neg(roots[0], g[0]);
            fe_t r_sq2;
            ops->sq(r_sq2, roots[0]);

            fe_t disc2, s2, neg_c12, inv22, two2;
            fe_t trs2, fa2;
            ops->add(trs2, r_sq2, r_sq2);
            ops->add(trs2, trs2, r_sq2);
            ops->add(fa2, a, a);
            ops->add(fa2, fa2, fa2);
            ops->neg(disc2, trs2);
            ops->sub(disc2, disc2, fa2);

            if (!ops->sqrt_qr(s2, disc2))
                return 0;

            ops->neg(neg_c12, roots[0]);
            ops->one(two2);
            ops->add(two2, two2, two2);
            ops->invert(inv22, two2);

            ops->add(roots[1], neg_c12, s2);
            ops->mul(roots[1], roots[1], inv22);

            ops->sub(roots[2], neg_c12, s2);
            ops->mul(roots[2], roots[2], inv22);

            return 1;
        }
        else if (deg == 2)
        {
            fe_t disc2, s2, inv22, two2, neg_g12;
            ops->sq(disc2, g[1]);
            fe_t four_g02;
            ops->add(four_g02, g[0], g[0]);
            ops->add(four_g02, four_g02, four_g02);
            ops->sub(disc2, disc2, four_g02);

            if (!ops->sqrt_qr(s2, disc2))
                return 0;

            ops->neg(neg_g12, g[1]);
            ops->one(two2);
            ops->add(two2, two2, two2);
            ops->invert(inv22, two2);

            ops->add(roots[0], neg_g12, s2);
            ops->mul(roots[0], roots[0], inv22);

            ops->sub(roots[1], neg_g12, s2);
            ops->mul(roots[1], roots[1], inv22);

            ops->add(roots[2], roots[0], roots[1]);
            ops->neg(roots[2], roots[2]);

            return 1;
        }

        // deg == 3: all roots satisfy x^((p-1)/2) = -1. This means all roots
        // are QNRs, which is possible. Need random element splitting.
        // Fall through to brute approach.
    }

    // Fallback: If we get here (deg == 3 for both tries), the cubic has 3 roots
    // but our splitting failed. This is extremely unlikely. Try shifting: compute
    // gcd((x+c)^((p-1)/2) - 1, cubic) for c = 1, 2, ...
    for (int c_val = 1; c_val <= 10; c_val++)
    {
        // Compute (x+c)^((p-1)/2) mod cubic
        // Start with (x + c), then raise to power
        fe_t base[3];
        ops->zero(base[0]);
        ops->one(base[1]);
        ops->zero(base[2]);

        // Add c to constant term
        fe_t c_fe;
        ops->zero(c_fe);
        for (int k = 0; k < c_val; k++)
        {
            fe_t one2;
            ops->one(one2);
            ops->add(c_fe, c_fe, one2);
        }
        ops->add(base[0], base[0], c_fe);

        // Now raise base to (p-1)/2 mod cubic using repeated squaring
        fe_t result[3];
        ops->one(result[0]);
        ops->zero(result[1]);
        ops->zero(result[2]);

        for (int i = pm1_half_msb; i >= 0; i--)
        {
            fe_t tmp[3];
            // Square
            {
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
                // Reduce mod cubic: x^3 = -g[2]*x^2 - g[1]*x - g[0]
                // But our cubic is x^3 + a*x + b (no x^2 term)
                ops->mul(t, d4, neg_a);
                ops->add(d2, d2, t);
                ops->mul(t, d4, neg_b);
                ops->add(d1, d1, t);
                ops->mul(t, d3, neg_a);
                ops->add(d1, d1, t);
                ops->mul(t, d3, neg_b);
                ops->add(d0, d0, t);
                ops->copy(tmp[0], d0);
                ops->copy(tmp[1], d1);
                ops->copy(tmp[2], d2);
            }
            ops->copy(result[0], tmp[0]);
            ops->copy(result[1], tmp[1]);
            ops->copy(result[2], tmp[2]);

            if (pm1_half_bits[i])
            {
                // Multiply by base
                fe_t f[3];
                ops->copy(f[0], result[0]);
                ops->copy(f[1], result[1]);
                ops->copy(f[2], result[2]);
                // schoolbook f * base mod cubic
                fe_t d0, d1, d2, d3, d4, t;
                ops->mul(d0, f[0], base[0]);
                ops->mul(t, f[0], base[1]);
                ops->mul(d1, f[1], base[0]);
                ops->add(d1, d1, t);
                ops->mul(t, f[0], base[2]);
                fe_t t2;
                ops->mul(t2, f[1], base[1]);
                ops->add(d2, t, t2);
                ops->mul(t, f[2], base[0]);
                ops->add(d2, d2, t);
                ops->mul(t, f[1], base[2]);
                ops->mul(t2, f[2], base[1]);
                ops->add(d3, t, t2);
                ops->mul(d4, f[2], base[2]);
                // reduce
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

        // result = (x+c)^((p-1)/2) mod cubic. Subtract 1.
        ops->sub(result[0], result[0], one_fe);

        // gcd(result, cubic)
        ops->copy(aa[0], b);
        ops->copy(aa[1], a);
        ops->zero(aa[2]);
        ops->one(aa[3]);
        ops->copy(bb[0], result[0]);
        ops->copy(bb[1], result[1]);
        ops->copy(bb[2], result[2]);

        deg = poly_gcd(aa, 3, bb, 2, g, ops);

        if (deg == 1)
        {
            ops->neg(roots[0], g[0]);

            fe_t r_sq;
            ops->sq(r_sq, roots[0]);

            fe_t disc, s, neg_r, inv2, two;
            fe_t three_r_sq2, four_a2;
            ops->add(three_r_sq2, r_sq, r_sq);
            ops->add(three_r_sq2, three_r_sq2, r_sq);
            ops->add(four_a2, a, a);
            ops->add(four_a2, four_a2, four_a2);
            ops->neg(disc, three_r_sq2);
            ops->sub(disc, disc, four_a2);

            if (!ops->sqrt_qr(s, disc))
                return 0;

            ops->neg(neg_r, roots[0]);
            ops->one(two);
            ops->add(two, two, two);
            ops->invert(inv2, two);

            ops->add(roots[1], neg_r, s);
            ops->mul(roots[1], roots[1], inv2);

            ops->sub(roots[2], neg_r, s);
            ops->mul(roots[2], roots[2], inv2);

            return 1;
        }
        else if (deg == 2)
        {
            fe_t disc2, s2, inv22, two2, neg_g12;
            ops->sq(disc2, g[1]);
            fe_t four_g02;
            ops->add(four_g02, g[0], g[0]);
            ops->add(four_g02, four_g02, four_g02);
            ops->sub(disc2, disc2, four_g02);

            if (!ops->sqrt_qr(s2, disc2))
                return 0;

            ops->neg(neg_g12, g[1]);
            ops->one(two2);
            ops->add(two2, two2, two2);
            ops->invert(inv22, two2);

            ops->add(roots[0], neg_g12, s2);
            ops->mul(roots[0], roots[0], inv22);

            ops->sub(roots[1], neg_g12, s2);
            ops->mul(roots[1], roots[1], inv22);

            ops->add(roots[2], roots[0], roots[1]);
            ops->neg(roots[2], roots[2]);

            return 1;
        }
    }

    return 0; // shouldn't reach here
}

// ============================================================================
// Hex formatting (needed by worker, must be declared before it)
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
// Halving chain computation (2-descent)
//
// The 2-descent determines the structure of E[2^∞](GF(p)) = E(GF(p))[2^∞],
// the 2-primary part of the group, without computing #E.
//
// For a curve with full 2-torsion (x^3+ax+b splits), E[2^∞] ≅ Z/2^a × Z/2^b.
// Each 2-torsion point (e_i, 0) anchors a "halving chain": we iteratively
// find Q with 2Q = P. The chain length c_i is the number of successful halvings.
//
// The halving equation (derived from the doubling formula for short Weierstrass
// y^2 = x^3 + Ax + B) is a monic quartic in u:
//   u^4 - 4*xP*u^3 - 2*A*u^2 - (8*B + 4*A*xP)*u + (A^2 - 4*B*xP) = 0
//
// Derivation: if 2Q = P with Q=(u,v), P=(xP,yP), then
//   xP = lambda^2 - 2u   where  lambda = (3u^2 + A)/(2v)
// Substituting v^2 = u^3 + Au + B and clearing denominators:
//   xP * 4(u^3+Au+B) = (3u^2+A)^2 - 8u(u^3+Au+B)
// Expanding and collecting: u^4 - 4xP*u^3 - 2A*u^2 - (8B+4AxP)*u + (A^2-4BxP) = 0
//
// The first halving (level 2→3) has a simpler criterion [Cass91]:
//   (e_i, 0) is halvable iff D_i = (e_i-e_j)*(e_i-e_k) is a QR in GF(p).
// If so, the half-point has x = e_i + sqrt(D_i).
//
// The three chain lengths determine the 2-Sylow structure:
//   a = min(c_i) + 1,  b = max(c_i) + 1,  v2(#E) = a + b
// ============================================================================

// Compute the halving chain length for 2-torsion point (e_i, 0) on E: y^2 = x^3 + Ax + B.
// Returns the number of successful halvings (0 means not halvable at level 2->3).
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
    // Level 2->3: D_i = (e_i - e_j)(e_i - e_k); halvable iff D_i is QR
    fe_t diff_j, diff_k, D_i, sqrt_D;
    ops->sub(diff_j, e_i, e_j);
    ops->sub(diff_k, e_i, e_k);
    ops->mul(D_i, diff_j, diff_k);

    if (!ops->sqrt_qr(sqrt_D, D_i))
        return 0;

    // Half-point: x_half = e_i + sqrt(D_i), y = sqrt(x_half^3 + A*x_half + B)
    fe_t xP, yP;
    ops->add(xP, e_i, sqrt_D);

    // y^2 = x^3 + Ax + B
    fe_t x2, x3, ax, y2;
    ops->sq(x2, xP);
    ops->mul(x3, x2, xP);
    ops->mul(ax, A, xP);
    ops->add(y2, x3, ax);
    ops->add(y2, y2, B);

    fe_t yP_tmp;
    if (!ops->sqrt_qr(yP_tmp, y2))
        return 0; // shouldn't happen
    ops->copy(yP, yP_tmp);

    int chain = 1;

    // Deeper halvings: for point (xP, yP), build quartic and check for roots
    for (int depth = 1; depth < max_depth; depth++)
    {
        // The halving equation for P = (xP, yP) on y^2 = x^3 + Ax + B:
        // If Q = (u, v) satisfies 2Q = P, then u satisfies:
        //   u^4 - 4*xP*u^2 + (-2*A)*u^2 + (-4*A*xP - 8*B)*u + (A^2 - 4*B*xP) = 0
        //
        // Wait, let me be more careful. The doubling formula for short Weierstrass
        // y^2 = x^3 + ax + b gives:
        //   x(2Q) = lambda^2 - 2*xQ  where lambda = (3*xQ^2 + a) / (2*yQ)
        //
        // So if 2Q = P = (xP, yP), and Q = (u, v), then:
        //   xP = ((3u^2 + A)/(2v))^2 - 2u
        //   yP = ((3u^2 + A)/(2v)) * (u - xP) - v
        //
        // From the x-coordinate equation, clearing denominators (v^2 = u^3 + Au + B):
        //   xP = (3u^2 + A)^2 / (4(u^3 + Au + B)) - 2u
        //   xP * 4(u^3 + Au + B) = (3u^2 + A)^2 - 8u(u^3 + Au + B)
        //   4xP(u^3 + Au + B) = 9u^4 + 6Au^2 + A^2 - 8u^4 - 8Au^2 - 8Bu
        //   4xP*u^3 + 4A*xP*u + 4B*xP = u^4 - 2Au^2 + A^2 - 8Bu
        //   u^4 - 4xP*u^3 - 2A*u^2 - (8B + 4A*xP)*u + (A^2 - 4B*xP) = 0
        //
        // This is our monic quartic in u.
        fe_t quartic[4]; // coefficients of x^0, x^1, x^2, x^3

        // c0 = A^2 - 4*B*xP
        fe_t A2, four_BxP;
        ops->sq(A2, A);
        ops->mul(four_BxP, B, xP);
        ops->add(four_BxP, four_BxP, four_BxP);
        ops->add(four_BxP, four_BxP, four_BxP);
        ops->sub(quartic[0], A2, four_BxP);

        // c1 = -(8B + 4A*xP) = -4*(2B + A*xP)
        fe_t AxP, two_B, c1_inner;
        ops->mul(AxP, A, xP);
        ops->add(two_B, B, B);
        ops->add(c1_inner, two_B, AxP);
        ops->add(c1_inner, c1_inner, c1_inner); // *2
        ops->add(c1_inner, c1_inner, c1_inner); // *4
        ops->neg(quartic[1], c1_inner);

        // c2 = -2A
        ops->add(quartic[2], A, A);
        ops->neg(quartic[2], quartic[2]);

        // c3 = -4*xP
        ops->add(quartic[3], xP, xP);
        ops->add(quartic[3], quartic[3], quartic[3]);
        ops->neg(quartic[3], quartic[3]);

        fe_t u;
        if (!find_one_root(u, quartic, ops, prime_bits, prime_msb))
            break;

        // Compute v from u: v^2 = u^3 + Au + B, then check sign via yP
        fe_t u2_new, u3_new, au_new, v2;
        ops->sq(u2_new, u);
        ops->mul(u3_new, u2_new, u);
        ops->mul(au_new, A, u);
        ops->add(v2, u3_new, au_new);
        ops->add(v2, v2, B);

        fe_t v;
        if (!ops->sqrt_qr(v, v2))
            break; // shouldn't happen

        // Verify correct sign: yP should equal lambda*(u - xP) - v
        // where lambda = (3u^2 + A)/(2v).
        // If sign is wrong, negate v.
        // Check: compute 2*(u,v) and see if x-coordinate matches xP
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

        // Compare x_double with xP
        unsigned char xd_bytes[32], xp_bytes[32];
        ops->tobytes(xd_bytes, x_double);
        ops->tobytes(xp_bytes, xP);

        if (std::memcmp(xd_bytes, xp_bytes, 32) != 0)
        {
            // Something went wrong — root is valid but doubling check failed.
            // This could be a computation error. Stop the chain.
            break;
        }

        chain++;

        // Next iteration: P = (u, v)
        ops->copy(xP, u);
        ops->copy(yP, v);
    }

    return chain;
}

// Compute v2(#E) for E: y^2 = x^3 + Ax + B with known 2-torsion roots.
//
// The 2-Sylow subgroup E[2^∞](GF(p)) ≅ Z/2^a × Z/2^b where:
//   a = min(chain_lengths) + 1
//   b = max(chain_lengths) + 1
// and v2(#E) = a + b = min + max + 2.
//
// See [ST92] §IV.4 for the group structure theorem.
//
// Returns v2(#E). Also sets *levels_out = max(chains) + 1 = the ECFFT domain
// exponent (the larger cyclic factor of the 2-Sylow subgroup).
static int compute_v2(
    const fe_t A,
    const fe_t B,
    const fe_t roots[3],
    const FieldOps *ops,
    const int *prime_bits,
    int prime_msb,
    int *levels_out)
{
    int chains[3];
    int max_depth = 30; // way more than we'll ever see

    chains[0] = halving_chain(roots[0], roots[1], roots[2], A, B, ops, prime_bits, prime_msb, max_depth);
    chains[1] = halving_chain(roots[1], roots[0], roots[2], A, B, ops, prime_bits, prime_msb, max_depth);
    chains[2] = halving_chain(roots[2], roots[0], roots[1], A, B, ops, prime_bits, prime_msb, max_depth);

    int mn = chains[0], mx = chains[0];
    for (int i = 1; i < 3; i++)
    {
        if (chains[i] < mn)
            mn = chains[i];
        if (chains[i] > mx)
            mx = chains[i];
    }

    if (levels_out)
        *levels_out = mx + 1;

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
    uint64_t g0 = 0x7B9BA138F07A1ULL;
    uint64_t g1 = 0x638D19E0B11D2ULL;
    uint64_t g2 = 0x2D13853ULL;

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
// Status thread
// ============================================================================

static void status_thread_fn(uint64_t total_trials, int num_threads, std::chrono::steady_clock::time_point start_time)
{
    while (!g_stop.load(std::memory_order_relaxed))
    {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (g_stop.load(std::memory_order_relaxed))
            break;

        uint64_t done = g_trials_done.load(std::memory_order_relaxed);
        int found = g_found.load(std::memory_order_relaxed);
        int best = g_best_levels.load(std::memory_order_relaxed);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double rate = (elapsed > 0.0) ? done / elapsed : 0.0;
        double pct = (total_trials > 0) ? 100.0 * done / total_trials : 0.0;

        std::lock_guard<std::mutex> lock(g_print_mutex);
        fprintf(
            stderr,
            "  [%5.1f%%] %" PRIu64 " / %" PRIu64 " trials, %d hits, best levels=%d, %.0f curves/sec (%d threads)\n",
            pct,
            done,
            total_trials,
            found,
            best,
            rate,
            num_threads);
    }
}

// ============================================================================
// Helper: convert a small integer to a field element
// ============================================================================

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

// ============================================================================
// Generic worker
// ============================================================================

/*
 * Scalar field-element exponentiation: base^exp mod p.
 * exp given as little-endian bit array bits[0..msb], MSB-first square-and-multiply.
 */
static void fe_pow(fe_t result, const fe_t base, const int *bits, int msb, const FieldOps *ops)
{
    ops->one(result);
    for (int i = msb; i >= 0; i--)
    {
        fe_t tmp;
        ops->sq(tmp, result);
        ops->copy(result, tmp);
        if (bits[i])
            ops->mul(result, result, base);
    }
}

/*
 * Euler criterion: returns 1 if z is a quadratic residue mod p, 0 otherwise.
 * Computes z^((p-1)/2) and checks if the result equals 1.
 */
static int fe_is_qr(const fe_t z, const int *pm1_half_bits, int pm1_half_msb, const FieldOps *ops)
{
    fe_t result;
    fe_pow(result, z, pm1_half_bits, pm1_half_msb, ops);
    fe_t one;
    ops->one(one);
    fe_t diff;
    ops->sub(diff, result, one);
    return !ops->isnonzero(diff);
}

static void worker(
    int thread_id,
    uint64_t trials_start,
    uint64_t trials_count,
    const int *field_bits,
    int field_msb,
    const int *pm1_half_bits,
    int pm1_half_msb,
    int min_levels,
    const FieldOps *ops,
    const char *field_name,
    int a_int,
    uint64_t user_seed)
{
    Prng rng;
    uint64_t base_seed = (uint64_t)thread_id * 0x9E3779B97F4A7C15ULL + trials_start;
    rng.seed(user_seed ? (user_seed ^ base_seed) : base_seed);

    fe_t a;
    fe_from_int(a, a_int, ops);

    for (uint64_t trial = 0; trial < trials_count; trial++)
    {
        if (g_stop.load(std::memory_order_relaxed))
            break;

        unsigned char b_bytes[32];
        rng.random_bytes(b_bytes);

        fe_t b;
        ops->frombytes(b, b_bytes);

        // Check discriminant: 4a^3 + 27b^2 != 0
        // a = -3: 4*(-27) + 27*b^2 = -108 + 27*b^2
        fe_t b2, disc;
        ops->sq(b2, b);

        // 27*b^2
        fe_t b2x3, b2x9, b2_27;
        ops->add(b2x3, b2, b2);
        ops->add(b2x3, b2x3, b2);
        ops->add(b2x9, b2x3, b2x3);
        ops->add(b2x9, b2x9, b2x3);
        ops->add(b2_27, b2x9, b2x9);
        ops->add(b2_27, b2_27, b2x9);

        // 4*a^3 = 4*(-27) = -108
        fe_t a2, a3, four_a3;
        ops->sq(a2, a);
        ops->mul(a3, a2, a);
        ops->add(four_a3, a3, a3);
        ops->add(four_a3, four_a3, four_a3);

        ops->add(disc, four_a3, b2_27);
        if (!ops->isnonzero(disc))
        {
            g_trials_done.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        /*
         * Discriminant QR pre-filter (Euler criterion).
         *
         * For x^3 + ax + b to split completely over GF(p), its discriminant
         * Delta = -4a^3 - 27b^2 must be a quadratic residue. If Delta is NOT
         * a QR, the cubic cannot split → no full 2-torsion. Skip early.
         *
         * Cost: ~254 sq + ~127 mul (scalar exponentiation) vs ~2800 field ops
         * for the full polynomial Frobenius check. Filters ~2/3 of candidates.
         *
         * Note: disc here is 4a^3 + 27b^2 (positive form). The actual curve
         * discriminant is -(4a^3 + 27b^2). Since -1 may or may not be QR,
         * we check disc itself: if disc is not QR AND -1 is QR, then -disc
         * is not QR either. If -1 is not QR, then -disc IS QR when disc is
         * not. So we must check -disc (the actual discriminant) for QR.
         */
        fe_t neg_disc;
        ops->neg(neg_disc, disc);
        if (!fe_is_qr(neg_disc, pm1_half_bits, pm1_half_msb, ops))
        {
            g_trials_done.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        int full_2t = check_full_2torsion(a, b, field_bits, field_msb, ops);

        if (full_2t)
        {
            // Extract roots and compute v2 + levels
            fe_t roots[3];
            if (find_cubic_roots(roots, a, b, field_bits, field_msb, ops))
            {
                int levels;
                int v2 = compute_v2(a, b, roots, ops, field_bits, field_msb, &levels);

                // Update best levels atomically
                int prev_best = g_best_levels.load(std::memory_order_relaxed);
                while (levels > prev_best)
                {
                    if (g_best_levels.compare_exchange_weak(prev_best, levels, std::memory_order_relaxed))
                        break;
                }

                if (levels >= min_levels)
                {
                    g_found.fetch_add(1, std::memory_order_relaxed);

                    Candidate c;
                    ops->tobytes(c.b, b);
                    c.v2 = v2;
                    c.levels = levels;

                    char b_hex[65];
                    hex_string(b_hex, sizeof(b_hex), c.b, 32);

                    std::lock_guard<std::mutex> lock(g_print_mutex);
                    g_candidates.push_back(c);
                    fprintf(
                        stderr,
                        "  *** HIT: field=%s a=%d b=0x%s levels=%d (v2=%d, domain=%d) ***\n",
                        field_name,
                        a_int,
                        b_hex,
                        levels,
                        v2,
                        1 << levels);
                    if (g_outfile)
                    {
                        fprintf(
                            g_outfile,
                            "field=%s a=%d b=0x%s levels=%d v2=%d domain=%d\n",
                            field_name,
                            a_int,
                            b_hex,
                            levels,
                            v2,
                            1 << levels);
                        fflush(g_outfile);
                    }
                }
            }
        }

        g_trials_done.fetch_add(1, std::memory_order_relaxed);
    }
}

// ============================================================================
// Search
// ============================================================================

static int
    search_field(const char *field, uint64_t max_trials, int min_levels, int num_threads, int a_int, uint64_t user_seed)
{
    unsigned char field_bytes[32];
    int bits[255];
    int msb;

    int is_fq = (strcmp(field, "fq") == 0);

    if (is_fq)
    {
        get_q_bytes(field_bytes);
        fprintf(stderr, "Searching for ECFFT curves over GF(q) with a=%d\n", a_int);
    }
    else
    {
        get_p_bytes(field_bytes);
        fprintf(stderr, "Searching for ECFFT curves over GF(p) [p = 2^255 - 19] with a=%d\n", a_int);
    }

    fprintf(stderr, "Prime (hex, BE) = ");
    for (int i = 31; i >= 0; i--)
        fprintf(stderr, "%02x", field_bytes[i]);
    fprintf(stderr, "\n");

    for (int i = 0; i < 255; i++)
        bits[i] = (field_bytes[i / 8] >> (i % 8)) & 1;
    msb = 254;
    while (msb > 0 && bits[msb] == 0)
        msb--;

    if (user_seed)
        fprintf(
            stderr,
            "Trials: %" PRIu64 ", min levels: %d (domain >= %d), threads: %d, seed: %" PRIu64 "\n",
            max_trials,
            min_levels,
            1 << min_levels,
            num_threads,
            user_seed);
    else
        fprintf(
            stderr,
            "Trials: %" PRIu64 ", min levels: %d (domain >= %d), threads: %d, seed: deterministic\n",
            max_trials,
            min_levels,
            1 << min_levels,
            num_threads);
    fprintf(stderr, "2-descent halving chains for native computation.\n");
    fprintf(stderr, "Discriminant QR pre-filter enabled (Euler criterion).\n\n");

    // Write output file header
    if (g_outfile)
    {
        fprintf(g_outfile, "# ECFFT Curve Search Results\n");
        fprintf(g_outfile, "# field=%s a=%d\n", field, a_int);
        fprintf(g_outfile, "# trials=%" PRIu64 " min_levels=%d threads=%d\n", max_trials, min_levels, num_threads);
        if (user_seed)
            fprintf(g_outfile, "# seed=%" PRIu64 "\n", user_seed);
        else
            fprintf(g_outfile, "# seed=deterministic\n");
        fprintf(g_outfile, "#\n");
        fflush(g_outfile);
    }

    // Precompute (p-1)/2 bits for Euler criterion QR check
    int pm1_half_bits[255];
    int pm1_half_msb = compute_pm1_half_bits(pm1_half_bits, bits);

    g_trials_done.store(0);
    g_found.store(0);
    g_best_levels.store(0);
    g_stop.store(false);
    g_candidates.clear();
    g_candidates.reserve(1024);

    auto start_time = std::chrono::steady_clock::now();
    std::thread status_thread(status_thread_fn, max_trials, num_threads, start_time);

    const FieldOps *ops = is_fq ? &FQ_OPS : &FP_OPS;

    std::vector<std::thread> workers;
    uint64_t per_thread = max_trials / num_threads;
    uint64_t remainder = max_trials % num_threads;
    uint64_t offset = 0;
    for (int t = 0; t < num_threads; t++)
    {
        uint64_t count = per_thread + (t < remainder ? 1 : 0);
        workers.emplace_back(
            worker, t, offset, count, bits, msb, pm1_half_bits, pm1_half_msb, min_levels, ops, field, a_int, user_seed);
        offset += count;
    }
    for (auto &w : workers)
        w.join();

    g_stop.store(true);
    status_thread.join();

    auto end_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    uint64_t done = g_trials_done.load();
    int found = (int)g_candidates.size();

    fprintf(
        stderr,
        "\nDone: %d hits (levels >= %d) from %" PRIu64 " trials in %.1f sec (%.0f curves/sec)\n",
        found,
        min_levels,
        done,
        elapsed,
        (double)done / elapsed);
    fprintf(stderr, "Best levels: %d (domain %d)\n", g_best_levels.load(), 1 << g_best_levels.load());

    return found;
}

// ============================================================================
// CLI
// ============================================================================

static void usage()
{
    printf("Usage: ranshaw-find-ecfft [options]\n\n");
    printf("Options:\n");
    printf("  --field fp|fq      Field to search over (default: fq)\n");
    printf("  --a N              Curve parameter a (small integer, default: -3)\n");
    printf("  --trials N|max     Number of random curves to try (default: 100000, max = unlimited)\n");
    printf("  --min-levels N     Minimum ECFFT levels to report (default: 12)\n");
    printf("  --cpus auto|N      Number of threads (default: 1, auto = all cores)\n");
    printf("  --seed N|auto      PRNG seed (default: deterministic, auto = system clock)\n");
    printf("  -o <path>          Write hits to output file (incremental, flushed per hit)\n");
    printf("  --help             Show this help\n\n");
    printf("Algorithm:\n");
    printf("  For each random b, tests y^2 = x^3 + ax + b for full 2-torsion,\n");
    printf("  then computes the 2-Sylow structure via 2-descent (halving chains).\n");
    printf("  No SageMath or point counting needed.\n\n");
    printf("  'levels' is the ECFFT domain exponent: the larger cyclic factor of the\n");
    printf("  2-Sylow subgroup Z/2^a x Z/2^b. The domain size is 2^levels.\n");
    printf("  Note: levels < v2(#E) for full-2-torsion curves (v2 = a + b, levels = b).\n\n");
    printf("  For ECFFT, levels >= 12 (domain 4096) is a practical minimum.\n");
    printf("  The probability of levels >= k is roughly 1/2^(k-1) among full-2-torsion\n");
    printf("  curves, so ~500K trials should yield levels >= 13.\n\n");
    printf("Output:\n");
    printf("  Hits to stdout: field=<f> a=<N> b=<hex> levels=<N> v2=<N> domain=<N>\n");
    printf("  Progress to stderr.\n");
}

int main(int argc, char **argv)
{
    const char *field = "fq";
    const char *outpath = nullptr;
    uint64_t trials = 100000;
    int min_levels = 12;
    int num_threads = 1;
    int a_int = -3;
    uint64_t user_seed = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--field") == 0 && i + 1 < argc)
            field = argv[++i];
        else if (strcmp(argv[i], "--a") == 0 && i + 1 < argc)
            a_int = atoi(argv[++i]);
        else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc)
        {
            i++;
            if (strcmp(argv[i], "max") == 0)
                trials = UINT64_MAX;
            else
                trials = strtoull(argv[i], nullptr, 10);
        }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        {
            i++;
            if (strcmp(argv[i], "auto") == 0)
                user_seed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
            else
                user_seed = strtoull(argv[i], nullptr, 10);
        }
        else if (strcmp(argv[i], "--min-levels") == 0 && i + 1 < argc)
            min_levels = atoi(argv[++i]);
        else if (strcmp(argv[i], "--min-v2") == 0 && i + 1 < argc)
        {
            // Backwards compatibility: treat --min-v2 as --min-levels
            min_levels = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--cpus") == 0 && i + 1 < argc)
        {
            i++;
            if (strcmp(argv[i], "auto") == 0)
            {
                num_threads = (int)std::thread::hardware_concurrency();
                if (num_threads < 1)
                    num_threads = 1;
            }
            else
            {
                num_threads = atoi(argv[i]);
                if (num_threads < 1)
                    num_threads = 1;
            }
        }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
            outpath = argv[++i];
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

    if (strcmp(field, "fq") != 0 && strcmp(field, "fp") != 0)
    {
        fprintf(stderr, "Unknown field: %s (use fp or fq)\n", field);
        return 1;
    }

    if (outpath)
    {
        g_outfile = fopen(outpath, "w");
        if (!g_outfile)
        {
            fprintf(stderr, "Error: cannot open output file: %s\n", outpath);
            return 1;
        }
        fprintf(stderr, "Output file: %s\n", outpath);
    }

    fprintf(stderr, "ECFFT Curve Search (2-Descent)\n");
    fprintf(stderr, "==============================\n\n");

    int found = search_field(field, trials, min_levels, num_threads, a_int, user_seed);

    // Print results to stdout
    for (size_t i = 0; i < g_candidates.size(); i++)
    {
        char b_hex[65];
        hex_string(b_hex, sizeof(b_hex), g_candidates[i].b, 32);
        printf(
            "field=%s a=%d b=0x%s levels=%d v2=%d domain=%d\n",
            field,
            a_int,
            b_hex,
            g_candidates[i].levels,
            g_candidates[i].v2,
            1 << g_candidates[i].levels);
    }

    if (found == 0)
        fprintf(stderr, "No curves with levels >= %d found. Try more trials.\n", min_levels);

    if (g_outfile)
        fclose(g_outfile);

    return 0;
}
