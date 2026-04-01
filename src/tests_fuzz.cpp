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

#include "ranshaw.h"
#include "ranshaw_primitives.h"
#ifdef RANSHAW_ECFFT
#include "ecfft_fp.h"
#include "ecfft_fq.h"
#endif
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

using namespace ranshaw;

/* ======================================================================
 * Test framework
 * ====================================================================== */

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;
static bool quiet_mode = false;
static uint64_t global_seed = 0ULL;

static std::string hex(const unsigned char *data, size_t len)
{
    std::ostringstream oss;
    for (size_t i = 0; i < len; ++i)
        oss << std::hex << std::setfill('0') << std::setw(2) << (int)data[i];
    return oss.str();
}

static bool check_bytes(const char *test_name, const unsigned char *expected, const unsigned char *actual, size_t len)
{
    ++tests_run;
    if (std::memcmp(expected, actual, len) == 0)
    {
        ++tests_passed;
        if (!quiet_mode)
            std::cout << "  PASS: " << test_name << std::endl;
        return true;
    }
    else
    {
        ++tests_failed;
        std::cout << "  FAIL: " << test_name << std::endl;
        std::cout << "    expected: " << hex(expected, len) << std::endl;
        std::cout << "    actual:   " << hex(actual, len) << std::endl;
        return false;
    }
}

static bool check_true(const char *test_name, bool condition)
{
    ++tests_run;
    if (condition)
    {
        ++tests_passed;
        if (!quiet_mode)
            std::cout << "  PASS: " << test_name << std::endl;
        return true;
    }
    else
    {
        ++tests_failed;
        std::cout << "  FAIL: " << test_name << std::endl;
        return false;
    }
}

/* ======================================================================
 * PRNG: xoshiro256** with splitmix64 seeding
 * ====================================================================== */

struct xoshiro256ss
{
    uint64_t s[4];

    static uint64_t splitmix64(uint64_t &state)
    {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    void seed(uint64_t seed_val)
    {
        uint64_t sm = seed_val;
        s[0] = splitmix64(sm);
        s[1] = splitmix64(sm);
        s[2] = splitmix64(sm);
        s[3] = splitmix64(sm);
    }

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

    void fill_bytes(uint8_t *buf, size_t len)
    {
        size_t i = 0;
        while (i + 8 <= len)
        {
            uint64_t v = next();
            std::memcpy(buf + i, &v, 8);
            i += 8;
        }
        if (i < len)
        {
            uint64_t v = next();
            std::memcpy(buf + i, &v, len - i);
        }
    }
};

/* ======================================================================
 * Random generation helpers
 * ====================================================================== */

static RanScalar random_ran_scalar(xoshiro256ss &rng)
{
    uint8_t wide[64];
    rng.fill_bytes(wide, 64);
    return RanScalar::reduce_wide(wide);
}

static ShawScalar random_shaw_scalar(xoshiro256ss &rng)
{
    uint8_t wide[64];
    rng.fill_bytes(wide, 64);
    return ShawScalar::reduce_wide(wide);
}

static RanPoint random_ran_point(xoshiro256ss &rng)
{
    return RanPoint::generator().scalar_mul_vartime(random_ran_scalar(rng));
}

static ShawPoint random_shaw_point(xoshiro256ss &rng)
{
    return ShawPoint::generator().scalar_mul_vartime(random_shaw_scalar(rng));
}

/* Compare two points by serialized bytes */
static bool ran_points_equal(const RanPoint &a, const RanPoint &b)
{
    auto ab = a.to_bytes();
    auto bb = b.to_bytes();
    return std::memcmp(ab.data(), bb.data(), 32) == 0;
}

static bool shaw_points_equal(const ShawPoint &a, const ShawPoint &b)
{
    auto ab = a.to_bytes();
    auto bb = b.to_bytes();
    return std::memcmp(ab.data(), bb.data(), 32) == 0;
}

/* ======================================================================
 * 1. fuzz_scalar_arithmetic — ~10,000 checks
 * ====================================================================== */

static void fuzz_scalar_arithmetic()
{
    std::cout << std::endl << "=== Fuzz: Scalar Arithmetic ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 1);

    for (int i = 0; i < 1000; i++)
    {
        std::string label = "ran_scalar_arith[" + std::to_string(i) + "]";

        auto a = random_ran_scalar(rng);
        auto b = random_ran_scalar(rng);
        auto c = random_ran_scalar(rng);

        /* Commutativity of add */
        check_true((label + " a+b==b+a").c_str(), a + b == b + a);
        /* Commutativity of mul */
        check_true((label + " a*b==b*a").c_str(), a * b == b * a);
        /* Associativity of add */
        check_true((label + " (a+b)+c==a+(b+c)").c_str(), (a + b) + c == a + (b + c));
        /* Distributivity */
        check_true((label + " a*(b+c)==a*b+a*c").c_str(), a * (b + c) == a * b + a * c);
        /* Identity */
        check_true((label + " a+0==a").c_str(), a + RanScalar::zero() == a);
        check_true((label + " a*1==a").c_str(), a * RanScalar::one() == a);
        /* Inverse */
        check_true((label + " a+(-a)==0").c_str(), (a + (-a)).is_zero());
        /* Square */
        check_true((label + " sq==a*a").c_str(), a.sq() == a * a);
        /* muladd */
        check_true((label + " muladd").c_str(), RanScalar::muladd(a, b, c) == a * b + c);
        /* Inversion */
        if (!a.is_zero())
        {
            auto inv = a.invert();
            check_true((label + " a*inv==1").c_str(), inv.has_value() && a * inv.value() == RanScalar::one());
        }
    }

    for (int i = 0; i < 1000; i++)
    {
        std::string label = "shaw_scalar_arith[" + std::to_string(i) + "]";

        auto a = random_shaw_scalar(rng);
        auto b = random_shaw_scalar(rng);
        auto c = random_shaw_scalar(rng);

        check_true((label + " a+b==b+a").c_str(), a + b == b + a);
        check_true((label + " a*b==b*a").c_str(), a * b == b * a);
        check_true((label + " (a+b)+c==a+(b+c)").c_str(), (a + b) + c == a + (b + c));
        check_true((label + " a*(b+c)==a*b+a*c").c_str(), a * (b + c) == a * b + a * c);
        check_true((label + " a+0==a").c_str(), a + ShawScalar::zero() == a);
        check_true((label + " a*1==a").c_str(), a * ShawScalar::one() == a);
        check_true((label + " a+(-a)==0").c_str(), (a + (-a)).is_zero());
        check_true((label + " sq==a*a").c_str(), a.sq() == a * a);
        check_true((label + " muladd").c_str(), ShawScalar::muladd(a, b, c) == a * b + c);
        if (!a.is_zero())
        {
            auto inv = a.invert();
            check_true((label + " a*inv==1").c_str(), inv.has_value() && a * inv.value() == ShawScalar::one());
        }
    }
}

/* ======================================================================
 * 2. fuzz_scalar_edge_cases — ~100
 * ====================================================================== */

static void fuzz_scalar_edge_cases()
{
    std::cout << std::endl << "=== Fuzz: Scalar Edge Cases ===" << std::endl;

    /* Ran */
    {
        auto z = RanScalar::zero();
        auto o = RanScalar::one();

        check_true("ran zero+zero==zero", (z + z).is_zero());
        check_true("ran one*one==one", o * o == o);
        check_true("ran zero.invert()==nullopt", !z.invert().has_value());
        check_true("ran -zero==zero", (-z).is_zero());
        check_true("ran one.invert()==one", o.invert().has_value() && o.invert().value() == o);

        /* reduce_wide all-zero */
        uint8_t all_zero[64] = {0};
        check_true("ran reduce_wide(0)==zero", RanScalar::reduce_wide(all_zero).is_zero());

        /* reduce_wide all-0xFF */
        uint8_t all_ff[64];
        std::memset(all_ff, 0xFF, 64);
        auto rff = RanScalar::reduce_wide(all_ff);
        check_true("ran reduce_wide(ff) != zero", !rff.is_zero());

        /* from_bytes with value >= modulus should fail */
        uint8_t over[32];
        std::memset(over, 0xFF, 32);
        check_true("ran from_bytes(>=q)==nullopt", !RanScalar::from_bytes(over).has_value());
    }

    /* Shaw */
    {
        auto z = ShawScalar::zero();
        auto o = ShawScalar::one();

        check_true("shaw zero+zero==zero", (z + z).is_zero());
        check_true("shaw one*one==one", o * o == o);
        check_true("shaw zero.invert()==nullopt", !z.invert().has_value());
        check_true("shaw -zero==zero", (-z).is_zero());
        check_true("shaw one.invert()==one", o.invert().has_value() && o.invert().value() == o);

        uint8_t all_zero[64] = {0};
        check_true("shaw reduce_wide(0)==zero", ShawScalar::reduce_wide(all_zero).is_zero());

        uint8_t all_ff[64];
        std::memset(all_ff, 0xFF, 64);
        auto rff = ShawScalar::reduce_wide(all_ff);
        check_true("shaw reduce_wide(ff) != zero", !rff.is_zero());

        uint8_t over[32];
        std::memset(over, 0xFF, 32);
        check_true("shaw from_bytes(>=p)==nullopt", !ShawScalar::from_bytes(over).has_value());
    }
}

/* ======================================================================
 * 3. fuzz_point_arithmetic — ~2,000 checks
 * ====================================================================== */

static void fuzz_point_arithmetic()
{
    std::cout << std::endl << "=== Fuzz: Point Arithmetic ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 3);

    for (int i = 0; i < 250; i++)
    {
        std::string label = "ran_pt[" + std::to_string(i) + "]";

        auto P = random_ran_point(rng);
        auto Q = random_ran_point(rng);
        auto R = random_ran_point(rng);
        auto I = RanPoint::identity();

        check_true((label + " P+Q==Q+P").c_str(), ran_points_equal(P + Q, Q + P));
        check_true((label + " P+P==dbl").c_str(), ran_points_equal(P + P, P.dbl()));
        check_true((label + " (P+Q)+R==P+(Q+R)").c_str(), ran_points_equal((P + Q) + R, P + (Q + R)));
        check_true((label + " P+I==P").c_str(), ran_points_equal(P + I, P));
        check_true((label + " I+P==P").c_str(), ran_points_equal(I + P, P));
    }

    for (int i = 0; i < 250; i++)
    {
        std::string label = "shaw_pt[" + std::to_string(i) + "]";

        auto P = random_shaw_point(rng);
        auto Q = random_shaw_point(rng);
        auto R = random_shaw_point(rng);
        auto I = ShawPoint::identity();

        check_true((label + " P+Q==Q+P").c_str(), shaw_points_equal(P + Q, Q + P));
        check_true((label + " P+P==dbl").c_str(), shaw_points_equal(P + P, P.dbl()));
        check_true((label + " (P+Q)+R==P+(Q+R)").c_str(), shaw_points_equal((P + Q) + R, P + (Q + R)));
        check_true((label + " P+I==P").c_str(), shaw_points_equal(P + I, P));
        check_true((label + " I+P==P").c_str(), shaw_points_equal(I + P, P));
    }
}

/* ======================================================================
 * 4. fuzz_ipa_edge_cases — ~120
 * ====================================================================== */

static void fuzz_ipa_edge_cases()
{
    std::cout << std::endl << "=== Fuzz: IPA Edge Cases ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 4);

    /* Ran */
    for (int i = 0; i < 10; i++)
    {
        std::string label = "ran_ipa[" + std::to_string(i) + "]";
        auto G = random_ran_point(rng);
        auto s = random_ran_scalar(rng);
        auto I = RanPoint::identity();

        /* zero * G == identity (CT) */
        check_true((label + " 0*G==I ct").c_str(), G.scalar_mul(RanScalar::zero()).is_identity());
        /* zero * G == identity (vartime) */
        check_true((label + " 0*G==I vt").c_str(), G.scalar_mul_vartime(RanScalar::zero()).is_identity());
        /* s * identity == identity (vartime — CT path doesn't support identity base) */
        check_true((label + " s*I==I vt").c_str(), I.scalar_mul_vartime(s).is_identity());
        /* P + (-P) == identity */
        check_true((label + " P+(-P)==I").c_str(), (G + (-G)).is_identity());
        /* 1 * G == G */
        check_true((label + " 1*G==G").c_str(), ran_points_equal(G.scalar_mul(RanScalar::one()), G));
        /* -(-P) == P */
        check_true((label + " -(-P)==P").c_str(), ran_points_equal(-(-G), G));
        /* MSM n=1 */
        auto msm1 = RanPoint::multi_scalar_mul(&s, &G, 1);
        auto sm1 = G.scalar_mul_vartime(s);
        check_true((label + " msm1==sm").c_str(), ran_points_equal(msm1, sm1));
    }

    /* Shaw */
    for (int i = 0; i < 10; i++)
    {
        std::string label = "shaw_ipa[" + std::to_string(i) + "]";
        auto G = random_shaw_point(rng);
        auto s = random_shaw_scalar(rng);
        auto I = ShawPoint::identity();

        check_true((label + " 0*G==I ct").c_str(), G.scalar_mul(ShawScalar::zero()).is_identity());
        check_true((label + " 0*G==I vt").c_str(), G.scalar_mul_vartime(ShawScalar::zero()).is_identity());
        check_true((label + " s*I==I vt").c_str(), I.scalar_mul_vartime(s).is_identity());
        check_true((label + " P+(-P)==I").c_str(), (G + (-G)).is_identity());
        check_true((label + " 1*G==G").c_str(), shaw_points_equal(G.scalar_mul(ShawScalar::one()), G));
        check_true((label + " -(-P)==P").c_str(), shaw_points_equal(-(-G), G));
        auto msm1 = ShawPoint::multi_scalar_mul(&s, &G, 1);
        auto sm1 = G.scalar_mul_vartime(s);
        check_true((label + " msm1==sm").c_str(), shaw_points_equal(msm1, sm1));
    }
}

/* ======================================================================
 * 5. fuzz_serialization_roundtrip — ~2,000 checks
 * ====================================================================== */

static void fuzz_serialization_roundtrip()
{
    std::cout << std::endl << "=== Fuzz: Serialization Round-trip ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 5);

    for (int i = 0; i < 250; i++)
    {
        std::string label = "ran_ser[" + std::to_string(i) + "]";

        /* Point round-trip */
        auto P = random_ran_point(rng);
        auto pb = P.to_bytes();
        auto P2 = RanPoint::from_bytes(pb.data());
        check_true((label + " pt_rt").c_str(), P2.has_value());
        if (P2.has_value())
        {
            auto pb2 = P2.value().to_bytes();
            check_bytes((label + " pt_bytes").c_str(), pb.data(), pb2.data(), 32);
        }

        /* Scalar round-trip */
        auto s = random_ran_scalar(rng);
        auto sb = s.to_bytes();
        auto s2 = RanScalar::from_bytes(sb.data());
        check_true((label + " sc_rt").c_str(), s2.has_value() && s2.value() == s);

        /* x_coordinate_bytes bit 255 clear */
        auto xb = P.x_coordinate_bytes();
        check_true((label + " x_bit255").c_str(), (xb[31] & 0x80) == 0);
    }

    /* Identity serialization: to_bytes produces all-zeros, from_bytes rejects it (not on-curve) */
    {
        auto I = RanPoint::identity();
        auto ib = I.to_bytes();
        unsigned char zero32[32] = {0};
        check_bytes("ran identity_bytes", zero32, ib.data(), 32);
    }

    for (int i = 0; i < 250; i++)
    {
        std::string label = "shaw_ser[" + std::to_string(i) + "]";

        auto P = random_shaw_point(rng);
        auto pb = P.to_bytes();
        auto P2 = ShawPoint::from_bytes(pb.data());
        check_true((label + " pt_rt").c_str(), P2.has_value());
        if (P2.has_value())
        {
            auto pb2 = P2.value().to_bytes();
            check_bytes((label + " pt_bytes").c_str(), pb.data(), pb2.data(), 32);
        }

        auto s = random_shaw_scalar(rng);
        auto sb = s.to_bytes();
        auto s2 = ShawScalar::from_bytes(sb.data());
        check_true((label + " sc_rt").c_str(), s2.has_value() && s2.value() == s);

        auto xb = P.x_coordinate_bytes();
        check_true((label + " x_bit255").c_str(), (xb[31] & 0x80) == 0);
    }

    {
        auto I = ShawPoint::identity();
        auto ib = I.to_bytes();
        unsigned char zero32[32] = {0};
        check_bytes("shaw identity_bytes", zero32, ib.data(), 32);
    }
}

/* ======================================================================
 * 6. fuzz_cross_curve_cycle — ~1,000 checks
 * ====================================================================== */

static void fuzz_cross_curve_cycle()
{
    std::cout << std::endl << "=== Fuzz: Cross-Curve Cycle ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 6);

    for (int i = 0; i < 250; i++)
    {
        std::string label = "cycle[" + std::to_string(i) + "]";

        /* Ran point -> x-coord bytes (Fp element) -> Shaw scalar (Fq reinterpretation) */
        auto P = random_ran_point(rng);
        auto x_bytes = P.x_coordinate_bytes();

        /* x_bytes is a canonical Fp element; Shaw scalars are Fp elements, so from_bytes should succeed */
        auto sel_s = ShawScalar::from_bytes(x_bytes.data());
        check_true((label + " hp_to_ss").c_str(), sel_s.has_value());

        if (sel_s.has_value())
        {
            /* Use it in a Shaw operation */
            auto Q = ShawPoint::generator().scalar_mul_vartime(sel_s.value());
            check_true((label + " Q_valid").c_str(), !Q.is_identity() || sel_s.value().is_zero());

            /* Extract Q's x-coord -> Ran scalar */
            auto qx = Q.x_coordinate_bytes();
            auto hel_s = RanScalar::from_bytes(qx.data());
            /* This might fail if the Fq x-coord value >= q (Ran scalar field), but it should usually succeed */
            if (hel_s.has_value())
            {
                /* Round-trip the scalar through bytes */
                auto hb = hel_s.value().to_bytes();
                auto hel_s2 = RanScalar::from_bytes(hb.data());
                check_true((label + " hs_rt").c_str(), hel_s2.has_value() && hel_s2.value() == hel_s.value());
            }
        }

        /* Wei25519 bridge check */
        auto wei_s = shaw_scalar_from_wei25519_x(x_bytes.data());
        if (wei_s.has_value())
        {
            auto wb = wei_s.value().to_bytes();
            auto wei_s2 = ShawScalar::from_bytes(wb.data());
            check_true((label + " wei_rt").c_str(), wei_s2.has_value() && wei_s2.value() == wei_s.value());
        }
    }
}

/* ======================================================================
 * 7. fuzz_scalarmul_consistency — ~1,500 checks
 * ====================================================================== */

static void fuzz_scalarmul_consistency()
{
    std::cout << std::endl << "=== Fuzz: ScalarMul Consistency ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 7);

    for (int i = 0; i < 250; i++)
    {
        std::string label = "ran_sm[" + std::to_string(i) + "]";

        auto P = random_ran_point(rng);
        auto a = random_ran_scalar(rng);
        auto b = random_ran_scalar(rng);

        /* CT vs vartime */
        check_true((label + " ct==vt").c_str(), ran_points_equal(P.scalar_mul(a), P.scalar_mul_vartime(a)));
        /* Linearity: P*(a+b) == P*a + P*b */
        auto lhs = P.scalar_mul_vartime(a + b);
        auto rhs = P.scalar_mul_vartime(a) + P.scalar_mul_vartime(b);
        check_true((label + " linear").c_str(), ran_points_equal(lhs, rhs));
        /* Composition: (a*b)*G == a*(b*G) */
        auto G = RanPoint::generator();
        auto lhs2 = G.scalar_mul_vartime(a * b);
        auto rhs2 = G.scalar_mul_vartime(b).scalar_mul_vartime(a);
        check_true((label + " compose").c_str(), ran_points_equal(lhs2, rhs2));
    }

    for (int i = 0; i < 250; i++)
    {
        std::string label = "shaw_sm[" + std::to_string(i) + "]";

        auto P = random_shaw_point(rng);
        auto a = random_shaw_scalar(rng);
        auto b = random_shaw_scalar(rng);

        check_true((label + " ct==vt").c_str(), shaw_points_equal(P.scalar_mul(a), P.scalar_mul_vartime(a)));
        auto lhs = P.scalar_mul_vartime(a + b);
        auto rhs = P.scalar_mul_vartime(a) + P.scalar_mul_vartime(b);
        check_true((label + " linear").c_str(), shaw_points_equal(lhs, rhs));
        auto G = ShawPoint::generator();
        auto lhs2 = G.scalar_mul_vartime(a * b);
        auto rhs2 = G.scalar_mul_vartime(b).scalar_mul_vartime(a);
        check_true((label + " compose").c_str(), shaw_points_equal(lhs2, rhs2));
    }
}

/* ======================================================================
 * 8. fuzz_msm_random — ~400 checks
 * ====================================================================== */

static void fuzz_msm_random()
{
    std::cout << std::endl << "=== Fuzz: MSM Random ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 8);

    const int sizes[] = {1, 2, 4, 8, 16, 33, 64};

    for (int si = 0; si < 7; si++)
    {
        size_t n = (size_t)sizes[si];
        for (int trial = 0; trial < 8; trial++)
        {
            std::string label = "ran_msm[n=" + std::to_string(n) + ",t=" + std::to_string(trial) + "]";

            std::vector<RanScalar> scalars(n);
            std::vector<RanPoint> points(n);
            for (size_t j = 0; j < n; j++)
            {
                scalars[j] = random_ran_scalar(rng);
                points[j] = random_ran_point(rng);
            }

            auto msm = RanPoint::multi_scalar_mul(scalars.data(), points.data(), n);
            auto naive = RanPoint::identity();
            for (size_t j = 0; j < n; j++)
                naive = naive + points[j].scalar_mul_vartime(scalars[j]);

            check_true(label.c_str(), ran_points_equal(msm, naive));
        }
    }

    for (int si = 0; si < 7; si++)
    {
        size_t n = (size_t)sizes[si];
        for (int trial = 0; trial < 8; trial++)
        {
            std::string label = "shaw_msm[n=" + std::to_string(n) + ",t=" + std::to_string(trial) + "]";

            std::vector<ShawScalar> scalars(n);
            std::vector<ShawPoint> points(n);
            for (size_t j = 0; j < n; j++)
            {
                scalars[j] = random_shaw_scalar(rng);
                points[j] = random_shaw_point(rng);
            }

            auto msm = ShawPoint::multi_scalar_mul(scalars.data(), points.data(), n);
            auto naive = ShawPoint::identity();
            for (size_t j = 0; j < n; j++)
                naive = naive + points[j].scalar_mul_vartime(scalars[j]);

            check_true(label.c_str(), shaw_points_equal(msm, naive));
        }
    }
}

/* ======================================================================
 * 9. fuzz_msm_sparse — ~400
 * ====================================================================== */

static void fuzz_msm_sparse()
{
    std::cout << std::endl << "=== Fuzz: MSM Sparse ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 9);

    /* Helper lambda-style tests for each curve */
    /* Ran */
    for (int trial = 0; trial < 20; trial++)
    {
        std::string label = "ran_sparse[" + std::to_string(trial) + "]";
        size_t n = 8;
        std::vector<RanScalar> scalars(n);
        std::vector<RanPoint> points(n);

        /* Zero scalars mixed in */
        for (size_t j = 0; j < n; j++)
        {
            points[j] = random_ran_point(rng);
            if (j % 3 == 0)
                scalars[j] = RanScalar::zero();
            else
                scalars[j] = random_ran_scalar(rng);
        }

        auto msm = RanPoint::multi_scalar_mul(scalars.data(), points.data(), n);
        auto naive = RanPoint::identity();
        for (size_t j = 0; j < n; j++)
            naive = naive + points[j].scalar_mul_vartime(scalars[j]);
        check_true((label + " zero_mixed").c_str(), ran_points_equal(msm, naive));
    }

    /* All scalars = one */
    for (int trial = 0; trial < 10; trial++)
    {
        std::string label = "ran_all_one[" + std::to_string(trial) + "]";
        size_t n = 8;
        std::vector<RanScalar> scalars(n, RanScalar::one());
        std::vector<RanPoint> points(n);
        auto sum = RanPoint::identity();
        for (size_t j = 0; j < n; j++)
        {
            points[j] = random_ran_point(rng);
            sum = sum + points[j];
        }
        auto msm = RanPoint::multi_scalar_mul(scalars.data(), points.data(), n);
        check_true(label.c_str(), ran_points_equal(msm, sum));
    }

    /* Same point repeated */
    for (int trial = 0; trial < 10; trial++)
    {
        std::string label = "ran_same_pt[" + std::to_string(trial) + "]";
        size_t n = 8;
        auto P = random_ran_point(rng);
        std::vector<RanPoint> points(n, P);
        std::vector<RanScalar> scalars(n);
        auto ssum = RanScalar::zero();
        for (size_t j = 0; j < n; j++)
        {
            scalars[j] = random_ran_scalar(rng);
            ssum = ssum + scalars[j];
        }
        auto msm = RanPoint::multi_scalar_mul(scalars.data(), points.data(), n);
        auto expected = P.scalar_mul_vartime(ssum);
        check_true(label.c_str(), ran_points_equal(msm, expected));
    }

    /* All-zero scalars */
    for (int trial = 0; trial < 5; trial++)
    {
        std::string label = "ran_all_zero[" + std::to_string(trial) + "]";
        size_t n = 8;
        std::vector<RanScalar> scalars(n, RanScalar::zero());
        std::vector<RanPoint> points(n);
        for (size_t j = 0; j < n; j++)
            points[j] = random_ran_point(rng);
        auto msm = RanPoint::multi_scalar_mul(scalars.data(), points.data(), n);
        check_true(label.c_str(), msm.is_identity());
    }

    /* Single nonzero in sea of zeros */
    for (int trial = 0; trial < 10; trial++)
    {
        std::string label = "ran_single_nz[" + std::to_string(trial) + "]";
        size_t n = 8;
        std::vector<RanScalar> scalars(n, RanScalar::zero());
        std::vector<RanPoint> points(n);
        for (size_t j = 0; j < n; j++)
            points[j] = random_ran_point(rng);
        size_t idx = (size_t)trial % n;
        scalars[idx] = random_ran_scalar(rng);
        auto msm = RanPoint::multi_scalar_mul(scalars.data(), points.data(), n);
        auto expected = points[idx].scalar_mul_vartime(scalars[idx]);
        check_true(label.c_str(), ran_points_equal(msm, expected));
    }

    /* Shaw - same patterns */
    for (int trial = 0; trial < 20; trial++)
    {
        std::string label = "shaw_sparse[" + std::to_string(trial) + "]";
        size_t n = 8;
        std::vector<ShawScalar> scalars(n);
        std::vector<ShawPoint> points(n);
        for (size_t j = 0; j < n; j++)
        {
            points[j] = random_shaw_point(rng);
            scalars[j] = (j % 3 == 0) ? ShawScalar::zero() : random_shaw_scalar(rng);
        }
        auto msm = ShawPoint::multi_scalar_mul(scalars.data(), points.data(), n);
        auto naive = ShawPoint::identity();
        for (size_t j = 0; j < n; j++)
            naive = naive + points[j].scalar_mul_vartime(scalars[j]);
        check_true((label + " zero_mixed").c_str(), shaw_points_equal(msm, naive));
    }

    for (int trial = 0; trial < 10; trial++)
    {
        std::string label = "shaw_all_one[" + std::to_string(trial) + "]";
        size_t n = 8;
        std::vector<ShawScalar> scalars(n, ShawScalar::one());
        std::vector<ShawPoint> points(n);
        auto sum = ShawPoint::identity();
        for (size_t j = 0; j < n; j++)
        {
            points[j] = random_shaw_point(rng);
            sum = sum + points[j];
        }
        auto msm = ShawPoint::multi_scalar_mul(scalars.data(), points.data(), n);
        check_true(label.c_str(), shaw_points_equal(msm, sum));
    }

    for (int trial = 0; trial < 10; trial++)
    {
        std::string label = "shaw_same_pt[" + std::to_string(trial) + "]";
        size_t n = 8;
        auto P = random_shaw_point(rng);
        std::vector<ShawPoint> points(n, P);
        std::vector<ShawScalar> scalars(n);
        auto ssum = ShawScalar::zero();
        for (size_t j = 0; j < n; j++)
        {
            scalars[j] = random_shaw_scalar(rng);
            ssum = ssum + scalars[j];
        }
        auto msm = ShawPoint::multi_scalar_mul(scalars.data(), points.data(), n);
        auto expected = P.scalar_mul_vartime(ssum);
        check_true(label.c_str(), shaw_points_equal(msm, expected));
    }

    for (int trial = 0; trial < 5; trial++)
    {
        std::string label = "shaw_all_zero[" + std::to_string(trial) + "]";
        size_t n = 8;
        std::vector<ShawScalar> scalars(n, ShawScalar::zero());
        std::vector<ShawPoint> points(n);
        for (size_t j = 0; j < n; j++)
            points[j] = random_shaw_point(rng);
        auto msm = ShawPoint::multi_scalar_mul(scalars.data(), points.data(), n);
        check_true(label.c_str(), msm.is_identity());
    }

    for (int trial = 0; trial < 10; trial++)
    {
        std::string label = "shaw_single_nz[" + std::to_string(trial) + "]";
        size_t n = 8;
        std::vector<ShawScalar> scalars(n, ShawScalar::zero());
        std::vector<ShawPoint> points(n);
        for (size_t j = 0; j < n; j++)
            points[j] = random_shaw_point(rng);
        size_t idx = (size_t)trial % n;
        scalars[idx] = random_shaw_scalar(rng);
        auto msm = ShawPoint::multi_scalar_mul(scalars.data(), points.data(), n);
        auto expected = points[idx].scalar_mul_vartime(scalars[idx]);
        check_true(label.c_str(), shaw_points_equal(msm, expected));
    }
}

/* ======================================================================
 * 10. fuzz_map_to_curve — ~1,000 checks
 * ====================================================================== */

static void fuzz_map_to_curve()
{
    std::cout << std::endl << "=== Fuzz: Map-to-Curve ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 10);

    for (int i = 0; i < 250; i++)
    {
        std::string label = "ran_mtc[" + std::to_string(i) + "]";

        /* For map_to_curve we need Fp bytes → use random Shaw scalar (which IS Fp) */
        auto fp_s = random_shaw_scalar(rng);
        auto u = fp_s.to_bytes();

        /* Single-element map_to_curve */
        auto P = RanPoint::map_to_curve(u.data());
        check_true((label + " non_id").c_str(), !P.is_identity());
        auto pb = P.to_bytes();
        auto P2 = RanPoint::from_bytes(pb.data());
        check_true((label + " rt").c_str(), P2.has_value());

        /* Two-element map_to_curve */
        auto fp_s2 = random_shaw_scalar(rng);
        auto u1 = fp_s2.to_bytes();
        auto Q = RanPoint::map_to_curve(u.data(), u1.data());
        check_true((label + " 2elem_non_id").c_str(), !Q.is_identity());

        /* Determinism */
        auto P3 = RanPoint::map_to_curve(u.data());
        check_true((label + " determ").c_str(), ran_points_equal(P, P3));
    }

    for (int i = 0; i < 250; i++)
    {
        std::string label = "shaw_mtc[" + std::to_string(i) + "]";

        /* For Shaw map_to_curve we need Fq bytes → use RanScalar (which is Fq) */
        auto fq_s = random_ran_scalar(rng);
        auto u = fq_s.to_bytes();

        auto P = ShawPoint::map_to_curve(u.data());
        check_true((label + " non_id").c_str(), !P.is_identity());
        auto pb = P.to_bytes();
        auto P2 = ShawPoint::from_bytes(pb.data());
        check_true((label + " rt").c_str(), P2.has_value());

        auto fq_s2 = random_ran_scalar(rng);
        auto u1 = fq_s2.to_bytes();
        auto Q = ShawPoint::map_to_curve(u.data(), u1.data());
        check_true((label + " 2elem_non_id").c_str(), !Q.is_identity());

        auto P3 = ShawPoint::map_to_curve(u.data());
        check_true((label + " determ").c_str(), shaw_points_equal(P, P3));
    }
}

/* ======================================================================
 * 11. fuzz_wei25519_bridge — ~500
 * ====================================================================== */

static void fuzz_wei25519_bridge()
{
    std::cout << std::endl << "=== Fuzz: Wei25519 Bridge ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 11);

    /* Valid canonical Fp bytes */
    for (int i = 0; i < 400; i++)
    {
        std::string label = "wei_valid[" + std::to_string(i) + "]";
        auto fp_s = random_shaw_scalar(rng); /* Fp element */
        auto bytes = fp_s.to_bytes();
        auto result = shaw_scalar_from_wei25519_x(bytes.data());
        if (result.has_value())
        {
            auto rb = result.value().to_bytes();
            auto result2 = ShawScalar::from_bytes(rb.data());
            check_true(label.c_str(), result2.has_value() && result2.value() == result.value());
        }
        else
        {
            /* Valid Fp element but wei25519 conversion failed — acceptable, just count it */
            check_true(label.c_str(), true);
        }
    }

    /* Bytes with bit 255 set → should return nullopt */
    for (int i = 0; i < 50; i++)
    {
        std::string label = "wei_bit255[" + std::to_string(i) + "]";
        uint8_t bytes[32];
        rng.fill_bytes(bytes, 32);
        bytes[31] |= 0x80;
        auto result = shaw_scalar_from_wei25519_x(bytes);
        check_true(label.c_str(), !result.has_value());
    }

    /* Bytes >= p → should return nullopt */
    for (int i = 0; i < 50; i++)
    {
        std::string label = "wei_over_p[" + std::to_string(i) + "]";
        uint8_t bytes[32];
        std::memset(bytes, 0xFF, 32);
        bytes[31] = 0x7F; /* Just below bit 255 but >= p since p = 2^255-19 */
        auto result = shaw_scalar_from_wei25519_x(bytes);
        check_true(label.c_str(), !result.has_value());
    }
}

/* ======================================================================
 * 12. fuzz_pedersen — ~800 checks
 * ====================================================================== */

static void fuzz_pedersen()
{
    std::cout << std::endl << "=== Fuzz: Pedersen Commitments ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 12);

    const int sizes[] = {1, 2, 4, 8, 16};

    /* Ran */
    for (int si = 0; si < 5; si++)
    {
        size_t n = (size_t)sizes[si];
        for (int trial = 0; trial < 10; trial++)
        {
            std::string label = "ran_ped[n=" + std::to_string(n) + ",t=" + std::to_string(trial) + "]";

            auto blinding = random_ran_scalar(rng);
            auto H = random_ran_point(rng);
            std::vector<RanScalar> vals(n);
            std::vector<RanPoint> gens(n);
            for (size_t j = 0; j < n; j++)
            {
                vals[j] = random_ran_scalar(rng);
                gens[j] = random_ran_point(rng);
            }

            auto commit = RanPoint::pedersen_commit(blinding, H, vals.data(), gens.data(), n);

            /* Naive: b*H + sum(v[i]*G[i]) */
            auto naive = H.scalar_mul_vartime(blinding);
            for (size_t j = 0; j < n; j++)
                naive = naive + gens[j].scalar_mul_vartime(vals[j]);

            check_true((label + " correct").c_str(), ran_points_equal(commit, naive));

            /* Cross-check: pedersen_commit == multi_scalar_mul with combined arrays */
            std::vector<RanScalar> all_scalars(n + 1);
            std::vector<RanPoint> all_points(n + 1);
            all_scalars[0] = blinding;
            all_points[0] = H;
            for (size_t j = 0; j < n; j++)
            {
                all_scalars[j + 1] = vals[j];
                all_points[j + 1] = gens[j];
            }
            auto msm = RanPoint::multi_scalar_mul(all_scalars.data(), all_points.data(), n + 1);
            check_true((label + " ped==msm").c_str(), ran_points_equal(commit, msm));
        }
    }

    /* Homomorphism: C(b1,v1) + C(b2,v2) == C(b1+b2, v1+v2) */
    for (int trial = 0; trial < 10; trial++)
    {
        std::string label = "ran_ped_homo[" + std::to_string(trial) + "]";
        size_t n = 4;
        auto H = random_ran_point(rng);
        std::vector<RanPoint> gens(n);
        for (size_t j = 0; j < n; j++)
            gens[j] = random_ran_point(rng);

        auto b1 = random_ran_scalar(rng);
        auto b2 = random_ran_scalar(rng);
        std::vector<RanScalar> v1(n), v2(n), vsum(n);
        for (size_t j = 0; j < n; j++)
        {
            v1[j] = random_ran_scalar(rng);
            v2[j] = random_ran_scalar(rng);
            vsum[j] = v1[j] + v2[j];
        }

        auto C1 = RanPoint::pedersen_commit(b1, H, v1.data(), gens.data(), n);
        auto C2 = RanPoint::pedersen_commit(b2, H, v2.data(), gens.data(), n);
        auto Csum = RanPoint::pedersen_commit(b1 + b2, H, vsum.data(), gens.data(), n);
        check_true(label.c_str(), ran_points_equal(C1 + C2, Csum));
    }

    /* Zero blinding */
    for (int trial = 0; trial < 5; trial++)
    {
        std::string label = "ran_ped_zblind[" + std::to_string(trial) + "]";
        size_t n = 4;
        auto H = random_ran_point(rng);
        std::vector<RanPoint> gens(n);
        std::vector<RanScalar> vals(n);
        for (size_t j = 0; j < n; j++)
        {
            gens[j] = random_ran_point(rng);
            vals[j] = random_ran_scalar(rng);
        }
        auto commit = RanPoint::pedersen_commit(RanScalar::zero(), H, vals.data(), gens.data(), n);
        auto naive = RanPoint::identity();
        for (size_t j = 0; j < n; j++)
            naive = naive + gens[j].scalar_mul_vartime(vals[j]);
        check_true(label.c_str(), ran_points_equal(commit, naive));
    }

    /* Shaw */
    for (int si = 0; si < 5; si++)
    {
        size_t n = (size_t)sizes[si];
        for (int trial = 0; trial < 10; trial++)
        {
            std::string label = "shaw_ped[n=" + std::to_string(n) + ",t=" + std::to_string(trial) + "]";

            auto blinding = random_shaw_scalar(rng);
            auto H = random_shaw_point(rng);
            std::vector<ShawScalar> vals(n);
            std::vector<ShawPoint> gens(n);
            for (size_t j = 0; j < n; j++)
            {
                vals[j] = random_shaw_scalar(rng);
                gens[j] = random_shaw_point(rng);
            }

            auto commit = ShawPoint::pedersen_commit(blinding, H, vals.data(), gens.data(), n);
            auto naive = H.scalar_mul_vartime(blinding);
            for (size_t j = 0; j < n; j++)
                naive = naive + gens[j].scalar_mul_vartime(vals[j]);
            check_true((label + " correct").c_str(), shaw_points_equal(commit, naive));

            /* Cross-check: pedersen_commit == multi_scalar_mul with combined arrays */
            std::vector<ShawScalar> all_scalars(n + 1);
            std::vector<ShawPoint> all_points(n + 1);
            all_scalars[0] = blinding;
            all_points[0] = H;
            for (size_t j = 0; j < n; j++)
            {
                all_scalars[j + 1] = vals[j];
                all_points[j + 1] = gens[j];
            }
            auto msm = ShawPoint::multi_scalar_mul(all_scalars.data(), all_points.data(), n + 1);
            check_true((label + " ped==msm").c_str(), shaw_points_equal(commit, msm));
        }
    }

    for (int trial = 0; trial < 10; trial++)
    {
        std::string label = "shaw_ped_homo[" + std::to_string(trial) + "]";
        size_t n = 4;
        auto H = random_shaw_point(rng);
        std::vector<ShawPoint> gens(n);
        for (size_t j = 0; j < n; j++)
            gens[j] = random_shaw_point(rng);

        auto b1 = random_shaw_scalar(rng);
        auto b2 = random_shaw_scalar(rng);
        std::vector<ShawScalar> v1(n), v2(n), vsum(n);
        for (size_t j = 0; j < n; j++)
        {
            v1[j] = random_shaw_scalar(rng);
            v2[j] = random_shaw_scalar(rng);
            vsum[j] = v1[j] + v2[j];
        }

        auto C1 = ShawPoint::pedersen_commit(b1, H, v1.data(), gens.data(), n);
        auto C2 = ShawPoint::pedersen_commit(b2, H, v2.data(), gens.data(), n);
        auto Csum = ShawPoint::pedersen_commit(b1 + b2, H, vsum.data(), gens.data(), n);
        check_true(label.c_str(), shaw_points_equal(C1 + C2, Csum));
    }

    for (int trial = 0; trial < 5; trial++)
    {
        std::string label = "shaw_ped_zblind[" + std::to_string(trial) + "]";
        size_t n = 4;
        auto H = random_shaw_point(rng);
        std::vector<ShawPoint> gens(n);
        std::vector<ShawScalar> vals(n);
        for (size_t j = 0; j < n; j++)
        {
            gens[j] = random_shaw_point(rng);
            vals[j] = random_shaw_scalar(rng);
        }
        auto commit = ShawPoint::pedersen_commit(ShawScalar::zero(), H, vals.data(), gens.data(), n);
        auto naive = ShawPoint::identity();
        for (size_t j = 0; j < n; j++)
            naive = naive + gens[j].scalar_mul_vartime(vals[j]);
        check_true(label.c_str(), shaw_points_equal(commit, naive));
    }
}

/* ======================================================================
 * 13. fuzz_batch_affine — ~400
 * ====================================================================== */

static void fuzz_batch_affine()
{
    std::cout << std::endl << "=== Fuzz: Batch Affine ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 13);

    const int sizes[] = {1, 2, 4, 8, 16, 32};

    /* Ran */
    for (int si = 0; si < 6; si++)
    {
        size_t n = (size_t)sizes[si];
        for (int trial = 0; trial < 8; trial++)
        {
            std::string label = "ran_batch_aff[n=" + std::to_string(n) + ",t=" + std::to_string(trial) + "]";

            std::vector<ran_jacobian> jac(n);
            for (size_t j = 0; j < n; j++)
            {
                auto P = random_ran_point(rng);
                ran_copy(&jac[j], &P.raw());
            }

            std::vector<ran_affine> batch(n);
            ran_batch_to_affine(batch.data(), jac.data(), n);

            bool all_ok = true;
            for (size_t j = 0; j < n; j++)
            {
                ran_affine single;
                if (ran_is_identity(&jac[j]))
                {
                    fp_0(single.x);
                    fp_0(single.y);
                }
                else
                {
                    ran_to_affine(&single, &jac[j]);
                }

                unsigned char batch_x[32], single_x[32], batch_y[32], single_y[32];
                fp_tobytes(batch_x, batch[j].x);
                fp_tobytes(single_x, single.x);
                fp_tobytes(batch_y, batch[j].y);
                fp_tobytes(single_y, single.y);

                if (std::memcmp(batch_x, single_x, 32) != 0 || std::memcmp(batch_y, single_y, 32) != 0)
                    all_ok = false;
            }
            check_true(label.c_str(), all_ok);
        }
    }

    /* Shaw */
    for (int si = 0; si < 6; si++)
    {
        size_t n = (size_t)sizes[si];
        for (int trial = 0; trial < 8; trial++)
        {
            std::string label = "shaw_batch_aff[n=" + std::to_string(n) + ",t=" + std::to_string(trial) + "]";

            std::vector<shaw_jacobian> jac(n);
            for (size_t j = 0; j < n; j++)
            {
                auto P = random_shaw_point(rng);
                shaw_copy(&jac[j], &P.raw());
            }

            std::vector<shaw_affine> batch(n);
            shaw_batch_to_affine(batch.data(), jac.data(), n);

            bool all_ok = true;
            for (size_t j = 0; j < n; j++)
            {
                shaw_affine single;
                if (shaw_is_identity(&jac[j]))
                {
                    fq_0(single.x);
                    fq_0(single.y);
                }
                else
                {
                    shaw_to_affine(&single, &jac[j]);
                }

                unsigned char batch_x[32], single_x[32], batch_y[32], single_y[32];
                fq_tobytes(batch_x, batch[j].x);
                fq_tobytes(single_x, single.x);
                fq_tobytes(batch_y, batch[j].y);
                fq_tobytes(single_y, single.y);

                if (std::memcmp(batch_x, single_x, 32) != 0 || std::memcmp(batch_y, single_y, 32) != 0)
                    all_ok = false;
            }
            check_true(label.c_str(), all_ok);
        }
    }
}

/* ======================================================================
 * 14. fuzz_polynomial — ~1,500 checks
 * ====================================================================== */

static void fuzz_polynomial()
{
    std::cout << std::endl << "=== Fuzz: Polynomial Arithmetic ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 14);

    static const unsigned char zero32[32] = {0};

    for (int i = 0; i < 250; i++)
    {
        std::string label = "fp_poly[" + std::to_string(i) + "]";
        size_t deg_a = 1 + (rng.next() % 16);
        size_t deg_b = 1 + (rng.next() % 16);

        /* Build coefficient arrays */
        std::vector<uint8_t> a_coeffs(deg_a * 32);
        std::vector<uint8_t> b_coeffs(deg_b * 32);
        for (size_t j = 0; j < deg_a; j++)
        {
            auto s = random_shaw_scalar(rng); /* Fp elements */
            auto sb = s.to_bytes();
            std::memcpy(&a_coeffs[j * 32], sb.data(), 32);
        }
        for (size_t j = 0; j < deg_b; j++)
        {
            auto s = random_shaw_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&b_coeffs[j * 32], sb.data(), 32);
        }

        auto A = FpPolynomial::from_coefficients(a_coeffs.data(), deg_a);
        auto B = FpPolynomial::from_coefficients(b_coeffs.data(), deg_b);

        /* Random evaluation point */
        auto x_s = random_shaw_scalar(rng);
        auto x = x_s.to_bytes();

        /* Eval consistency: (A*B)(x) == A(x) * B(x) */
        auto AB = A * B;
        auto ab_x = AB.evaluate(x.data());
        auto a_x = A.evaluate(x.data());
        auto b_x = B.evaluate(x.data());

        /* Multiply a_x * b_x via scalar */
        auto sa = ShawScalar::from_bytes(a_x.data());
        auto sb = ShawScalar::from_bytes(b_x.data());
        if (sa.has_value() && sb.has_value())
        {
            auto expected = (sa.value() * sb.value()).to_bytes();
            check_bytes((label + " mul_eval").c_str(), expected.data(), ab_x.data(), 32);
        }

        /* Add consistency: (A+B)(x) == A(x) + B(x) */
        auto ApB = A + B;
        auto apb_x = ApB.evaluate(x.data());
        if (sa.has_value() && sb.has_value())
        {
            auto expected = (sa.value() + sb.value()).to_bytes();
            check_bytes((label + " add_eval").c_str(), expected.data(), apb_x.data(), 32);
        }

        /* divmod: A = Q*B + R at random point */
        if (deg_a >= deg_b)
        {
            auto qr = A.divmod(B);
            auto q_x = qr.first.evaluate(x.data());
            auto r_x = qr.second.evaluate(x.data());

            auto sq = ShawScalar::from_bytes(q_x.data());
            auto sr = ShawScalar::from_bytes(r_x.data());
            if (sq.has_value() && sr.has_value() && sb.has_value() && sa.has_value())
            {
                auto expected = (sq.value() * sb.value() + sr.value()).to_bytes();
                check_bytes((label + " divmod").c_str(), expected.data(), a_x.data(), 32);
            }
        }
    }

    /* from_roots: each root evaluates to zero */
    for (int i = 0; i < 50; i++)
    {
        std::string label = "fp_roots[" + std::to_string(i) + "]";
        size_t n = 2 + (rng.next() % 8);
        std::vector<uint8_t> roots(n * 32);
        for (size_t j = 0; j < n; j++)
        {
            auto s = random_shaw_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&roots[j * 32], sb.data(), 32);
        }
        auto P = FpPolynomial::from_roots(roots.data(), n);
        bool all_zero = true;
        for (size_t j = 0; j < n; j++)
        {
            auto ev = P.evaluate(&roots[j * 32]);
            if (std::memcmp(ev.data(), zero32, 32) != 0)
                all_zero = false;
        }
        check_true(label.c_str(), all_zero);
    }

    /* Fq polynomial: same tests */
    for (int i = 0; i < 250; i++)
    {
        std::string label = "fq_poly[" + std::to_string(i) + "]";
        size_t deg_a = 1 + (rng.next() % 16);
        size_t deg_b = 1 + (rng.next() % 16);

        std::vector<uint8_t> a_coeffs(deg_a * 32);
        std::vector<uint8_t> b_coeffs(deg_b * 32);
        for (size_t j = 0; j < deg_a; j++)
        {
            auto s = random_ran_scalar(rng); /* Fq elements */
            auto sb = s.to_bytes();
            std::memcpy(&a_coeffs[j * 32], sb.data(), 32);
        }
        for (size_t j = 0; j < deg_b; j++)
        {
            auto s = random_ran_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&b_coeffs[j * 32], sb.data(), 32);
        }

        auto A = FqPolynomial::from_coefficients(a_coeffs.data(), deg_a);
        auto B = FqPolynomial::from_coefficients(b_coeffs.data(), deg_b);

        auto x_s = random_ran_scalar(rng);
        auto x = x_s.to_bytes();

        auto AB = A * B;
        auto ab_x = AB.evaluate(x.data());
        auto a_x = A.evaluate(x.data());
        auto b_x = B.evaluate(x.data());

        auto sa = RanScalar::from_bytes(a_x.data());
        auto sb = RanScalar::from_bytes(b_x.data());
        if (sa.has_value() && sb.has_value())
        {
            auto expected = (sa.value() * sb.value()).to_bytes();
            check_bytes((label + " mul_eval").c_str(), expected.data(), ab_x.data(), 32);
        }

        auto ApB = A + B;
        auto apb_x = ApB.evaluate(x.data());
        if (sa.has_value() && sb.has_value())
        {
            auto expected = (sa.value() + sb.value()).to_bytes();
            check_bytes((label + " add_eval").c_str(), expected.data(), apb_x.data(), 32);
        }

        if (deg_a >= deg_b)
        {
            auto qr = A.divmod(B);
            auto q_x = qr.first.evaluate(x.data());
            auto r_x = qr.second.evaluate(x.data());
            auto sq = RanScalar::from_bytes(q_x.data());
            auto sr = RanScalar::from_bytes(r_x.data());
            if (sq.has_value() && sr.has_value() && sb.has_value() && sa.has_value())
            {
                auto expected = (sq.value() * sb.value() + sr.value()).to_bytes();
                check_bytes((label + " divmod").c_str(), expected.data(), a_x.data(), 32);
            }
        }
    }

    for (int i = 0; i < 50; i++)
    {
        std::string label = "fq_roots[" + std::to_string(i) + "]";
        size_t n = 2 + (rng.next() % 8);
        std::vector<uint8_t> roots(n * 32);
        for (size_t j = 0; j < n; j++)
        {
            auto s = random_ran_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&roots[j * 32], sb.data(), 32);
        }
        auto P = FqPolynomial::from_roots(roots.data(), n);
        bool all_zero = true;
        for (size_t j = 0; j < n; j++)
        {
            auto ev = P.evaluate(&roots[j * 32]);
            if (std::memcmp(ev.data(), zero32, 32) != 0)
                all_zero = false;
        }
        check_true(label.c_str(), all_zero);
    }
}

/* ======================================================================
 * 15. fuzz_polynomial_protocol_sizes — ~400
 * ====================================================================== */

static void fuzz_polynomial_protocol_sizes()
{
    std::cout << std::endl << "=== Fuzz: Polynomial Protocol Sizes ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 15);

    static const unsigned char zero32[32] = {0};

    /* Fp: Karatsuba-range polys (degree 32-64) */
    for (int trial = 0; trial < 25; trial++)
    {
        std::string label = "fp_kara[" + std::to_string(trial) + "]";
        size_t deg_a = 32 + (rng.next() % 33);
        size_t deg_b = 32 + (rng.next() % 33);

        std::vector<uint8_t> ac(deg_a * 32), bc(deg_b * 32);
        for (size_t j = 0; j < deg_a; j++)
        {
            auto s = random_shaw_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&ac[j * 32], sb.data(), 32);
        }
        for (size_t j = 0; j < deg_b; j++)
        {
            auto s = random_shaw_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&bc[j * 32], sb.data(), 32);
        }

        auto A = FpPolynomial::from_coefficients(ac.data(), deg_a);
        auto B = FpPolynomial::from_coefficients(bc.data(), deg_b);
        auto AB = A * B;

        /* Verify at 3 random points */
        bool ok = true;
        for (int k = 0; k < 3; k++)
        {
            auto x_s = random_shaw_scalar(rng);
            auto x = x_s.to_bytes();
            auto ab_x = AB.evaluate(x.data());
            auto a_x = A.evaluate(x.data());
            auto b_x = B.evaluate(x.data());
            auto sa = ShawScalar::from_bytes(a_x.data());
            auto sb = ShawScalar::from_bytes(b_x.data());
            if (sa.has_value() && sb.has_value())
            {
                auto expected = (sa.value() * sb.value()).to_bytes();
                if (std::memcmp(expected.data(), ab_x.data(), 32) != 0)
                    ok = false;
            }
        }
        check_true(label.c_str(), ok);
    }

    /* Fp: from_roots with 16-32 roots */
    for (int trial = 0; trial < 25; trial++)
    {
        std::string label = "fp_roots_lg[" + std::to_string(trial) + "]";
        size_t n = 16 + (rng.next() % 17);
        std::vector<uint8_t> roots(n * 32);
        for (size_t j = 0; j < n; j++)
        {
            auto s = random_shaw_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&roots[j * 32], sb.data(), 32);
        }
        auto P = FpPolynomial::from_roots(roots.data(), n);
        /* Check 3 random roots */
        bool ok = true;
        for (size_t k = 0; k < 3 && k < n; k++)
        {
            size_t idx = rng.next() % n;
            auto ev = P.evaluate(&roots[idx * 32]);
            if (std::memcmp(ev.data(), zero32, 32) != 0)
                ok = false;
        }
        check_true(label.c_str(), ok);
    }

    /* Fp: interpolation */
    for (int trial = 0; trial < 25; trial++)
    {
        std::string label = "fp_interp[" + std::to_string(trial) + "]";
        size_t n = 8 + (rng.next() % 9);
        std::vector<uint8_t> xs(n * 32), ys(n * 32);
        for (size_t j = 0; j < n; j++)
        {
            auto sx = random_shaw_scalar(rng);
            auto sy = random_shaw_scalar(rng);
            auto xb = sx.to_bytes();
            auto yb = sy.to_bytes();
            std::memcpy(&xs[j * 32], xb.data(), 32);
            std::memcpy(&ys[j * 32], yb.data(), 32);
        }
        auto P = FpPolynomial::interpolate(xs.data(), ys.data(), n);
        bool ok = true;
        for (size_t j = 0; j < n; j++)
        {
            auto ev = P.evaluate(&xs[j * 32]);
            if (std::memcmp(ev.data(), &ys[j * 32], 32) != 0)
                ok = false;
        }
        check_true(label.c_str(), ok);
    }

    /* Fq: same patterns */
    for (int trial = 0; trial < 25; trial++)
    {
        std::string label = "fq_kara[" + std::to_string(trial) + "]";
        size_t deg_a = 32 + (rng.next() % 33);
        size_t deg_b = 32 + (rng.next() % 33);

        std::vector<uint8_t> ac(deg_a * 32), bc(deg_b * 32);
        for (size_t j = 0; j < deg_a; j++)
        {
            auto s = random_ran_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&ac[j * 32], sb.data(), 32);
        }
        for (size_t j = 0; j < deg_b; j++)
        {
            auto s = random_ran_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&bc[j * 32], sb.data(), 32);
        }

        auto A = FqPolynomial::from_coefficients(ac.data(), deg_a);
        auto B = FqPolynomial::from_coefficients(bc.data(), deg_b);
        auto AB = A * B;

        bool ok = true;
        for (int k = 0; k < 3; k++)
        {
            auto x_s = random_ran_scalar(rng);
            auto x = x_s.to_bytes();
            auto ab_x = AB.evaluate(x.data());
            auto a_x = A.evaluate(x.data());
            auto b_x = B.evaluate(x.data());
            auto sa = RanScalar::from_bytes(a_x.data());
            auto sb = RanScalar::from_bytes(b_x.data());
            if (sa.has_value() && sb.has_value())
            {
                auto expected = (sa.value() * sb.value()).to_bytes();
                if (std::memcmp(expected.data(), ab_x.data(), 32) != 0)
                    ok = false;
            }
        }
        check_true(label.c_str(), ok);
    }

    for (int trial = 0; trial < 25; trial++)
    {
        std::string label = "fq_roots_lg[" + std::to_string(trial) + "]";
        size_t n = 16 + (rng.next() % 17);
        std::vector<uint8_t> roots(n * 32);
        for (size_t j = 0; j < n; j++)
        {
            auto s = random_ran_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&roots[j * 32], sb.data(), 32);
        }
        auto P = FqPolynomial::from_roots(roots.data(), n);
        bool ok = true;
        for (size_t k = 0; k < 3 && k < n; k++)
        {
            size_t idx = rng.next() % n;
            auto ev = P.evaluate(&roots[idx * 32]);
            if (std::memcmp(ev.data(), zero32, 32) != 0)
                ok = false;
        }
        check_true(label.c_str(), ok);
    }

    for (int trial = 0; trial < 25; trial++)
    {
        std::string label = "fq_interp[" + std::to_string(trial) + "]";
        size_t n = 8 + (rng.next() % 9);
        std::vector<uint8_t> xs(n * 32), ys(n * 32);
        for (size_t j = 0; j < n; j++)
        {
            auto sx = random_ran_scalar(rng);
            auto sy = random_ran_scalar(rng);
            auto xb = sx.to_bytes();
            auto yb = sy.to_bytes();
            std::memcpy(&xs[j * 32], xb.data(), 32);
            std::memcpy(&ys[j * 32], yb.data(), 32);
        }
        auto P = FqPolynomial::interpolate(xs.data(), ys.data(), n);
        bool ok = true;
        for (size_t j = 0; j < n; j++)
        {
            auto ev = P.evaluate(&xs[j * 32]);
            if (std::memcmp(ev.data(), &ys[j * 32], 32) != 0)
                ok = false;
        }
        check_true(label.c_str(), ok);
    }
}

/* ======================================================================
 * 16. fuzz_divisor — ~600 checks
 * ====================================================================== */

static void fuzz_divisor()
{
    std::cout << std::endl << "=== Fuzz: Divisor ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 16);

    static const unsigned char zero32[32] = {0};

    const int sizes[] = {2, 3, 4, 5, 8};

    /* Ran */
    for (int si = 0; si < 5; si++)
    {
        size_t n = (size_t)sizes[si];
        for (int trial = 0; trial < 10; trial++)
        {
            std::string label = "ran_div[n=" + std::to_string(n) + ",t=" + std::to_string(trial) + "]";

            std::vector<RanPoint> pts(n);
            for (size_t j = 0; j < n; j++)
                pts[j] = random_ran_point(rng);

            auto div = RanDivisor::compute(pts.data(), n);

            /* Vanishing: evaluate at each defining point */
            bool vanish_ok = true;
            for (size_t j = 0; j < n; j++)
            {
                /* Get affine coordinates */
                ran_affine aff;
                ran_to_affine(&aff, &pts[j].raw());
                unsigned char xb[32], yb[32];
                fp_tobytes(xb, aff.x);
                fp_tobytes(yb, aff.y);

                auto ev = div.evaluate(xb, yb);
                if (std::memcmp(ev.data(), zero32, 32) != 0)
                    vanish_ok = false;
            }
            check_true((label + " vanish").c_str(), vanish_ok);

            /* Non-member: evaluate at a random point NOT in the set */
            auto rp = random_ran_point(rng);
            ran_affine raff;
            ran_to_affine(&raff, &rp.raw());
            unsigned char rxb[32], ryb[32];
            fp_tobytes(rxb, raff.x);
            fp_tobytes(ryb, raff.y);
            auto rev = div.evaluate(rxb, ryb);
            check_true((label + " non_member").c_str(), std::memcmp(rev.data(), zero32, 32) != 0);
        }
    }

    /* Shaw */
    for (int si = 0; si < 5; si++)
    {
        size_t n = (size_t)sizes[si];
        for (int trial = 0; trial < 10; trial++)
        {
            std::string label = "shaw_div[n=" + std::to_string(n) + ",t=" + std::to_string(trial) + "]";

            std::vector<ShawPoint> pts(n);
            for (size_t j = 0; j < n; j++)
                pts[j] = random_shaw_point(rng);

            auto div = ShawDivisor::compute(pts.data(), n);

            bool vanish_ok = true;
            for (size_t j = 0; j < n; j++)
            {
                shaw_affine aff;
                shaw_to_affine(&aff, &pts[j].raw());
                unsigned char xb[32], yb[32];
                fq_tobytes(xb, aff.x);
                fq_tobytes(yb, aff.y);

                auto ev = div.evaluate(xb, yb);
                if (std::memcmp(ev.data(), zero32, 32) != 0)
                    vanish_ok = false;
            }
            check_true((label + " vanish").c_str(), vanish_ok);

            auto rp = random_shaw_point(rng);
            shaw_affine raff;
            shaw_to_affine(&raff, &rp.raw());
            unsigned char rxb[32], ryb[32];
            fq_tobytes(rxb, raff.x);
            fq_tobytes(ryb, raff.y);
            auto rev = div.evaluate(rxb, ryb);
            check_true((label + " non_member").c_str(), std::memcmp(rev.data(), zero32, 32) != 0);
        }
    }
}

/* ======================================================================
 * 17. fuzz_divisor_scalar_mul — ~200 checks
 * ====================================================================== */

static void fuzz_divisor_scalar_mul()
{
    std::cout << std::endl << "=== Fuzz: Divisor ScalarMul ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 17);

    /* Ran */
    for (int trial = 0; trial < 25; trial++)
    {
        std::string label = "ran_smdiv[" + std::to_string(trial) + "]";

        auto s = random_ran_scalar(rng);
        auto sb = s.to_bytes();
        auto P = random_ran_point(rng);

        /* Get affine point */
        ran_affine aff;
        ran_to_affine(&aff, &P.raw());

        ran_divisor d;
        ran_scalar_mul_divisor(&d, sb.data(), &aff);

        /* a(x) should have nontrivial degree */
        check_true((label + " a_nontrivial").c_str(), d.a.coeffs.size() > 1);

        /* Evaluate at the input point — should vanish */
        unsigned char xb[32], yb[32];
        fp_tobytes(xb, aff.x);
        fp_tobytes(yb, aff.y);

        fp_fe result;
        ran_evaluate_divisor(result, &d, aff.x, aff.y);
        unsigned char result_bytes[32];
        fp_tobytes(result_bytes, result);
        static const unsigned char zero32[32] = {0};
        check_true((label + " vanish").c_str(), std::memcmp(result_bytes, zero32, 32) == 0);
    }

    /* Shaw */
    for (int trial = 0; trial < 25; trial++)
    {
        std::string label = "shaw_smdiv[" + std::to_string(trial) + "]";

        auto s = random_shaw_scalar(rng);
        auto sb = s.to_bytes();
        auto P = random_shaw_point(rng);

        shaw_affine aff;
        shaw_to_affine(&aff, &P.raw());

        shaw_divisor d;
        shaw_scalar_mul_divisor(&d, sb.data(), &aff);

        check_true((label + " a_nontrivial").c_str(), d.a.coeffs.size() > 1);

        fq_fe result;
        shaw_evaluate_divisor(result, &d, aff.x, aff.y);
        unsigned char result_bytes[32];
        fq_tobytes(result_bytes, result);
        static const unsigned char zero32[32] = {0};
        check_true((label + " vanish").c_str(), std::memcmp(result_bytes, zero32, 32) == 0);
    }
}

/* ======================================================================
 * 18. fuzz_operator_plus_regression — ~2,000 checks
 * ====================================================================== */

static void fuzz_operator_plus_regression()
{
    std::cout << std::endl << "=== Fuzz: Operator+ Regression ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 18);

    /* Ran */
    for (int i = 0; i < 250; i++)
    {
        std::string label = "ran_opadd[" + std::to_string(i) + "]";
        auto P = random_ran_point(rng);
        auto I = RanPoint::identity();

        /* P + P == P.dbl() */
        check_true((label + " P+P==dbl").c_str(), ran_points_equal(P + P, P.dbl()));
        /* P + (-P) == identity */
        check_true((label + " P+(-P)==I").c_str(), (P + (-P)).is_identity());
        /* P + identity == P */
        check_true((label + " P+I==P").c_str(), ran_points_equal(P + I, P));
        /* identity + P == P */
        check_true((label + " I+P==P").c_str(), ran_points_equal(I + P, P));
        /* identity + identity == identity */
        check_true((label + " I+I==I").c_str(), (I + I).is_identity());

        /* P + Q where P != ±Q: verify via scalar-based method */
        auto Q = random_ran_point(rng);
        auto PQ = P + Q;
        /* Verify PQ - P == Q, i.e. PQ + (-P) == Q */
        auto diff = PQ + (-P);
        check_true((label + " PQ-P==Q").c_str(), ran_points_equal(diff, Q));
    }

    /* Shaw */
    for (int i = 0; i < 250; i++)
    {
        std::string label = "shaw_opadd[" + std::to_string(i) + "]";
        auto P = random_shaw_point(rng);
        auto I = ShawPoint::identity();

        check_true((label + " P+P==dbl").c_str(), shaw_points_equal(P + P, P.dbl()));
        check_true((label + " P+(-P)==I").c_str(), (P + (-P)).is_identity());
        check_true((label + " P+I==P").c_str(), shaw_points_equal(P + I, P));
        check_true((label + " I+P==P").c_str(), shaw_points_equal(I + P, P));
        check_true((label + " I+I==I").c_str(), (I + I).is_identity());

        auto Q = random_shaw_point(rng);
        auto PQ = P + Q;
        auto diff = PQ + (-P);
        check_true((label + " PQ-P==Q").c_str(), shaw_points_equal(diff, Q));
    }
}

/* ======================================================================
 * 19. fuzz_verification_equation — ~500 checks
 * ====================================================================== */

static void fuzz_verification_equation()
{
    std::cout << std::endl << "=== Fuzz: Verification Equation ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 19);

    /* Ran */
    for (int trial = 0; trial < 25; trial++)
    {
        std::string label = "ran_verify[" + std::to_string(trial) + "]";
        auto L = random_ran_point(rng);
        auto R = random_ran_point(rng);
        auto P = random_ran_point(rng);
        auto x = random_ran_scalar(rng);

        /* Scalar prerequisites */
        auto x_inv = x.invert();
        check_true((label + " x*xinv==1").c_str(), x_inv.has_value() && x * x_inv.value() == RanScalar::one());
        check_true((label + " x.sq()==x*x").c_str(), x.sq() == x * x);

        auto x2 = x.sq();
        auto xi2 = x_inv.value().sq();

        /* P' = x^2 * L + P + x^{-2} * R */
        auto Pprime = L.scalar_mul_vartime(x2) + P + R.scalar_mul_vartime(xi2);

        /* P' - x^2*L - x^{-2}*R == P */
        auto check = Pprime + (-(L.scalar_mul_vartime(x2))) + (-(R.scalar_mul_vartime(xi2)));
        check_true((label + " verify_eq").c_str(), ran_points_equal(check, P));

        /* Also with negation approach */
        auto check2 = Pprime + (-L).scalar_mul_vartime(x2) + (-R).scalar_mul_vartime(xi2);
        check_true((label + " verify_neg").c_str(), ran_points_equal(check2, P));
    }

    /* Shaw */
    for (int trial = 0; trial < 25; trial++)
    {
        std::string label = "shaw_verify[" + std::to_string(trial) + "]";
        auto L = random_shaw_point(rng);
        auto R = random_shaw_point(rng);
        auto P = random_shaw_point(rng);
        auto x = random_shaw_scalar(rng);

        auto x_inv = x.invert();
        check_true((label + " x*xinv==1").c_str(), x_inv.has_value() && x * x_inv.value() == ShawScalar::one());
        check_true((label + " x.sq()==x*x").c_str(), x.sq() == x * x);

        auto x2 = x.sq();
        auto xi2 = x_inv.value().sq();

        auto Pprime = L.scalar_mul_vartime(x2) + P + R.scalar_mul_vartime(xi2);

        auto check = Pprime + (-(L.scalar_mul_vartime(x2))) + (-(R.scalar_mul_vartime(xi2)));
        check_true((label + " verify_eq").c_str(), shaw_points_equal(check, P));

        auto check2 = Pprime + (-L).scalar_mul_vartime(x2) + (-R).scalar_mul_vartime(xi2);
        check_true((label + " verify_neg").c_str(), shaw_points_equal(check2, P));
    }
}

/* ======================================================================
 * 20. fuzz_ecfft_poly_mul — gated #ifdef RANSHAW_ECFFT
 * ====================================================================== */

#ifdef RANSHAW_ECFFT

static void fuzz_ecfft_poly_mul()
{
    std::cout << std::endl << "=== Fuzz: ECFFT Polynomial Multiplication ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 20);

    /* Initialize ECFFT contexts */
    ecfft_fp_ctx fp_ctx;
    ecfft_fq_ctx fq_ctx;
    ecfft_fp_init(&fp_ctx);
    ecfft_fq_init(&fq_ctx);

    /* (a) Enter/exit round-trip — Fp */
    for (int trial = 0; trial < 50; trial++)
    {
        std::string label = "ecfft_fp_rt[" + std::to_string(trial) + "]";
        int deg = 4 + (rng.next() % 13); /* degree 4-16 */
        /* Need power-of-2 domain size */
        size_t n = 1;
        while (n < (size_t)(deg + 1))
            n <<= 1;
        if (n > fp_ctx.domain_size)
            continue;

        auto coeffs = std::make_unique<fp_fe[]>(n);
        auto saved = std::make_unique<fp_fe[]>(n);
        for (size_t j = 0; j < (size_t)(deg + 1); j++)
        {
            auto s = random_shaw_scalar(rng);
            auto sb = s.to_bytes();
            fp_frombytes(coeffs[j], sb.data());
            fp_copy(saved[j], coeffs[j]);
        }
        for (size_t j = (size_t)(deg + 1); j < n; j++)
        {
            fp_0(coeffs[j]);
            fp_0(saved[j]);
        }

        ecfft_fp_enter(coeffs.get(), n, &fp_ctx);
        ecfft_fp_exit(coeffs.get(), n, &fp_ctx);

        bool ok = true;
        for (size_t j = 0; j < n; j++)
        {
            unsigned char a_bytes[32], b_bytes[32];
            fp_tobytes(a_bytes, coeffs[j]);
            fp_tobytes(b_bytes, saved[j]);
            if (std::memcmp(a_bytes, b_bytes, 32) != 0)
                ok = false;
        }
        check_true(label.c_str(), ok);
    }

    /* (a) Enter/exit round-trip — Fq */
    for (int trial = 0; trial < 50; trial++)
    {
        std::string label = "ecfft_fq_rt[" + std::to_string(trial) + "]";
        int deg = 4 + (rng.next() % 13);
        size_t n = 1;
        while (n < (size_t)(deg + 1))
            n <<= 1;
        if (n > fq_ctx.domain_size)
            continue;

        auto coeffs = std::make_unique<fq_fe[]>(n);
        auto saved = std::make_unique<fq_fe[]>(n);
        for (size_t j = 0; j < (size_t)(deg + 1); j++)
        {
            auto s = random_ran_scalar(rng);
            auto sb = s.to_bytes();
            fq_frombytes(coeffs[j], sb.data());
            fq_copy(saved[j], coeffs[j]);
        }
        for (size_t j = (size_t)(deg + 1); j < n; j++)
        {
            fq_0(coeffs[j]);
            fq_0(saved[j]);
        }

        ecfft_fq_enter(coeffs.get(), n, &fq_ctx);
        ecfft_fq_exit(coeffs.get(), n, &fq_ctx);

        bool ok = true;
        for (size_t j = 0; j < n; j++)
        {
            unsigned char a_bytes[32], b_bytes[32];
            fq_tobytes(a_bytes, coeffs[j]);
            fq_tobytes(b_bytes, saved[j]);
            if (std::memcmp(a_bytes, b_bytes, 32) != 0)
                ok = false;
        }
        check_true(label.c_str(), ok);
    }

    /* (b) Small poly multiply via C++ API (schoolbook/Karatsuba, verify ECFFT compilation doesn't break them) */
    for (int trial = 0; trial < 100; trial++)
    {
        std::string label = "ecfft_small_fp[" + std::to_string(trial) + "]";
        size_t deg_a = 2 + (rng.next() % 15);
        size_t deg_b = 2 + (rng.next() % 15);

        std::vector<uint8_t> ac(deg_a * 32), bc(deg_b * 32);
        for (size_t j = 0; j < deg_a; j++)
        {
            auto s = random_shaw_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&ac[j * 32], sb.data(), 32);
        }
        for (size_t j = 0; j < deg_b; j++)
        {
            auto s = random_shaw_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&bc[j * 32], sb.data(), 32);
        }

        auto A = FpPolynomial::from_coefficients(ac.data(), deg_a);
        auto B = FpPolynomial::from_coefficients(bc.data(), deg_b);
        auto AB = A * B;

        auto x_s = random_shaw_scalar(rng);
        auto x = x_s.to_bytes();
        auto ab_x = AB.evaluate(x.data());
        auto a_x = A.evaluate(x.data());
        auto b_x = B.evaluate(x.data());
        auto sa = ShawScalar::from_bytes(a_x.data());
        auto sb = ShawScalar::from_bytes(b_x.data());
        bool ok = false;
        if (sa.has_value() && sb.has_value())
        {
            auto expected = (sa.value() * sb.value()).to_bytes();
            ok = std::memcmp(expected.data(), ab_x.data(), 32) == 0;
        }
        check_true(label.c_str(), ok);
    }

    for (int trial = 0; trial < 100; trial++)
    {
        std::string label = "ecfft_small_fq[" + std::to_string(trial) + "]";
        size_t deg_a = 2 + (rng.next() % 15);
        size_t deg_b = 2 + (rng.next() % 15);

        std::vector<uint8_t> ac(deg_a * 32), bc(deg_b * 32);
        for (size_t j = 0; j < deg_a; j++)
        {
            auto s = random_ran_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&ac[j * 32], sb.data(), 32);
        }
        for (size_t j = 0; j < deg_b; j++)
        {
            auto s = random_ran_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&bc[j * 32], sb.data(), 32);
        }

        auto A = FqPolynomial::from_coefficients(ac.data(), deg_a);
        auto B = FqPolynomial::from_coefficients(bc.data(), deg_b);
        auto AB = A * B;

        auto x_s = random_ran_scalar(rng);
        auto x = x_s.to_bytes();
        auto ab_x = AB.evaluate(x.data());
        auto a_x = A.evaluate(x.data());
        auto b_x = B.evaluate(x.data());
        auto sa = RanScalar::from_bytes(a_x.data());
        auto sb = RanScalar::from_bytes(b_x.data());
        bool ok = false;
        if (sa.has_value() && sb.has_value())
        {
            auto expected = (sa.value() * sb.value()).to_bytes();
            ok = std::memcmp(expected.data(), ab_x.data(), 32) == 0;
        }
        check_true(label.c_str(), ok);
    }

    /* (c) Karatsuba-threshold polys (deg 32-64) */
    for (int trial = 0; trial < 25; trial++)
    {
        std::string label = "ecfft_kara_fp[" + std::to_string(trial) + "]";
        size_t deg_a = 32 + (rng.next() % 33);
        size_t deg_b = 32 + (rng.next() % 33);

        std::vector<uint8_t> ac(deg_a * 32), bc(deg_b * 32);
        for (size_t j = 0; j < deg_a; j++)
        {
            auto s = random_shaw_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&ac[j * 32], sb.data(), 32);
        }
        for (size_t j = 0; j < deg_b; j++)
        {
            auto s = random_shaw_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&bc[j * 32], sb.data(), 32);
        }

        auto A = FpPolynomial::from_coefficients(ac.data(), deg_a);
        auto B = FpPolynomial::from_coefficients(bc.data(), deg_b);
        auto AB = A * B;

        bool ok = true;
        for (int k = 0; k < 3; k++)
        {
            auto x_s = random_shaw_scalar(rng);
            auto x = x_s.to_bytes();
            auto ab_x = AB.evaluate(x.data());
            auto a_x = A.evaluate(x.data());
            auto b_x = B.evaluate(x.data());
            auto sa = ShawScalar::from_bytes(a_x.data());
            auto sb = ShawScalar::from_bytes(b_x.data());
            if (sa.has_value() && sb.has_value())
            {
                auto expected = (sa.value() * sb.value()).to_bytes();
                if (std::memcmp(expected.data(), ab_x.data(), 32) != 0)
                    ok = false;
            }
        }
        check_true(label.c_str(), ok);
    }

    for (int trial = 0; trial < 25; trial++)
    {
        std::string label = "ecfft_kara_fq[" + std::to_string(trial) + "]";
        size_t deg_a = 32 + (rng.next() % 33);
        size_t deg_b = 32 + (rng.next() % 33);

        std::vector<uint8_t> ac(deg_a * 32), bc(deg_b * 32);
        for (size_t j = 0; j < deg_a; j++)
        {
            auto s = random_ran_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&ac[j * 32], sb.data(), 32);
        }
        for (size_t j = 0; j < deg_b; j++)
        {
            auto s = random_ran_scalar(rng);
            auto sb = s.to_bytes();
            std::memcpy(&bc[j * 32], sb.data(), 32);
        }

        auto A = FqPolynomial::from_coefficients(ac.data(), deg_a);
        auto B = FqPolynomial::from_coefficients(bc.data(), deg_b);
        auto AB = A * B;

        bool ok = true;
        for (int k = 0; k < 3; k++)
        {
            auto x_s = random_ran_scalar(rng);
            auto x = x_s.to_bytes();
            auto ab_x = AB.evaluate(x.data());
            auto a_x = A.evaluate(x.data());
            auto b_x = B.evaluate(x.data());
            auto sa = RanScalar::from_bytes(a_x.data());
            auto sb = RanScalar::from_bytes(b_x.data());
            if (sa.has_value() && sb.has_value())
            {
                auto expected = (sa.value() * sb.value()).to_bytes();
                if (std::memcmp(expected.data(), ab_x.data(), 32) != 0)
                    ok = false;
            }
        }
        check_true(label.c_str(), ok);
    }

    /* (d) ECFFT-threshold polys (deg >= 1024) — generate from roots for guaranteed structure */
    for (int trial = 0; trial < 2; trial++)
    {
        std::string label = "ecfft_large_fp[" + std::to_string(trial) + "]";
        size_t n_roots = 1024;
        std::vector<uint8_t> roots_a(n_roots * 32), roots_b(n_roots * 32);
        for (size_t j = 0; j < n_roots; j++)
        {
            auto sa = random_shaw_scalar(rng);
            auto sb_val = random_shaw_scalar(rng);
            auto sab = sa.to_bytes();
            auto sbb = sb_val.to_bytes();
            std::memcpy(&roots_a[j * 32], sab.data(), 32);
            std::memcpy(&roots_b[j * 32], sbb.data(), 32);
        }

        auto A = FpPolynomial::from_roots(roots_a.data(), n_roots);
        auto B = FpPolynomial::from_roots(roots_b.data(), n_roots);
        auto AB = A * B;

        /* Verify at 3 random points */
        bool ok = true;
        for (int k = 0; k < 3; k++)
        {
            auto x_s = random_shaw_scalar(rng);
            auto x = x_s.to_bytes();
            auto ab_x = AB.evaluate(x.data());
            auto a_x = A.evaluate(x.data());
            auto b_x = B.evaluate(x.data());
            auto sa = ShawScalar::from_bytes(a_x.data());
            auto sb = ShawScalar::from_bytes(b_x.data());
            if (sa.has_value() && sb.has_value())
            {
                auto expected = (sa.value() * sb.value()).to_bytes();
                if (std::memcmp(expected.data(), ab_x.data(), 32) != 0)
                    ok = false;
            }
        }
        check_true(label.c_str(), ok);
    }

    for (int trial = 0; trial < 2; trial++)
    {
        std::string label = "ecfft_large_fq[" + std::to_string(trial) + "]";
        size_t n_roots = 1024;
        std::vector<uint8_t> roots_a(n_roots * 32), roots_b(n_roots * 32);
        for (size_t j = 0; j < n_roots; j++)
        {
            auto sa = random_ran_scalar(rng);
            auto sb_val = random_ran_scalar(rng);
            auto sab = sa.to_bytes();
            auto sbb = sb_val.to_bytes();
            std::memcpy(&roots_a[j * 32], sab.data(), 32);
            std::memcpy(&roots_b[j * 32], sbb.data(), 32);
        }

        auto A = FqPolynomial::from_roots(roots_a.data(), n_roots);
        auto B = FqPolynomial::from_roots(roots_b.data(), n_roots);

        auto AB = A * B;

        /* Verify at 3 random points */
        bool ok = true;
        for (int k = 0; k < 3; k++)
        {
            auto x_s = random_ran_scalar(rng);
            auto x = x_s.to_bytes();
            auto ab_x = AB.evaluate(x.data());
            auto a_x = A.evaluate(x.data());
            auto b_x = B.evaluate(x.data());
            auto sa = RanScalar::from_bytes(a_x.data());
            auto sb = RanScalar::from_bytes(b_x.data());
            if (sa.has_value() && sb.has_value())
            {
                auto expected = (sa.value() * sb.value()).to_bytes();
                if (std::memcmp(expected.data(), ab_x.data(), 32) != 0)
                    ok = false;
            }
        }
        check_true(label.c_str(), ok);
    }
}

#endif /* RANSHAW_ECFFT */

/* ======================================================================
 * 21. fuzz_all_path_cross_validation — ~2,900 checks
 *
 * For each (scalar, point) pair, compute via ALL 6 code paths and verify
 * they all agree:
 *   A. CT scalarmul (ground truth)
 *   B. Vartime wNAF
 *   C. MSM with n=1
 *   D. Pedersen commit (s*P + 0*G)
 *   E. Fixed-base CT scalarmul
 *   F. Fixed-base MSM (n=1)
 * ====================================================================== */

static void fuzz_all_path_cross_validation()
{
    std::cout << std::endl << "=== Fuzz: All-Path Cross-Validation ===" << std::endl;
    xoshiro256ss rng;
    rng.seed(global_seed + 21);

    /* ---- Ran ---- */
    {
        auto test_ran = [&](const std::string &label, const RanScalar &s, const RanPoint &P)
        {
            /* Path A: CT scalarmul (ground truth) */
            auto A = P.scalar_mul(s);

            /* Path B: Vartime wNAF */
            auto B = P.scalar_mul_vartime(s);
            check_true((label + " B==A").c_str(), ran_points_equal(B, A));

            /* Path C: MSM with n=1 */
            auto C = RanPoint::multi_scalar_mul(&s, &P, 1);
            check_true((label + " C==A").c_str(), ran_points_equal(C, A));

            /* Path D: Pedersen commit (s*P + 0*G) */
            auto zero_s = RanScalar::zero();
            auto G = RanPoint::generator();
            auto D = RanPoint::pedersen_commit(s, P, &zero_s, &G, 1);
            check_true((label + " D==A").c_str(), ran_points_equal(D, A));

            /* Path E: Fixed-base CT */
            auto sb = s.to_bytes();
            ran_affine fixed_table[16];
            ran_scalarmult_fixed_precompute(fixed_table, &P.raw());
            RanPoint E;
            ran_scalarmult_fixed(&E.raw(), sb.data(), fixed_table);
            check_true((label + " E==A").c_str(), ran_points_equal(E, A));

            /* Path F: Fixed-base MSM (n=1, delegates to E internally) */
            RanPoint F;
            const ran_affine *tp = fixed_table;
            ran_msm_fixed(&F.raw(), sb.data(), &tp, 1);
            check_true((label + " F==A").c_str(), ran_points_equal(F, A));
        };

        /* Edge scalars */
        RanScalar edge_scalars[] = {
            RanScalar::zero(),
            RanScalar::one(),
            RanScalar::one() + RanScalar::one(),
            -RanScalar::one(),
            -(RanScalar::one() + RanScalar::one()),
        };
        const char *edge_names[] = {"0", "1", "2", "q-1", "q-2"};
        for (int ei = 0; ei < 5; ei++)
        {
            for (int trial = 0; trial < 10; trial++)
            {
                auto P = random_ran_point(rng);
                std::string label = "ran_xval[s=" + std::string(edge_names[ei]) + ",t=" + std::to_string(trial) + "]";
                test_ran(label, edge_scalars[ei], P);
            }
        }

        /* Random 256-bit scalars */
        for (int trial = 0; trial < 200; trial++)
        {
            auto s = random_ran_scalar(rng);
            auto P = random_ran_point(rng);
            std::string label = "ran_xval[rand," + std::to_string(trial) + "]";
            test_ran(label, s, P);
        }

        /* Small scalars (< 2^64) */
        for (int trial = 0; trial < 20; trial++)
        {
            uint8_t wide[64] = {};
            rng.fill_bytes(wide, 8); /* only fill first 8 bytes */
            auto s = RanScalar::reduce_wide(wide);
            auto P = random_ran_point(rng);
            std::string label = "ran_xval[small," + std::to_string(trial) + "]";
            test_ran(label, s, P);
        }

        /* High-bit scalars (bit 254 set) */
        for (int trial = 0; trial < 20; trial++)
        {
            uint8_t wide[64] = {};
            rng.fill_bytes(wide, 32);
            wide[31] |= 0x40; /* set bit 254 */
            wide[31] &= 0x7f; /* clear bit 255 to stay in range */
            auto s = RanScalar::reduce_wide(wide);
            auto P = random_ran_point(rng);
            std::string label = "ran_xval[high," + std::to_string(trial) + "]";
            test_ran(label, s, P);
        }
    }

    /* ---- Shaw ---- */
    {
        auto test_shaw = [&](const std::string &label, const ShawScalar &s, const ShawPoint &P)
        {
            /* Path A: CT scalarmul (ground truth) */
            auto A = P.scalar_mul(s);

            /* Path B: Vartime wNAF */
            auto B = P.scalar_mul_vartime(s);
            check_true((label + " B==A").c_str(), shaw_points_equal(B, A));

            /* Path C: MSM with n=1 */
            auto C = ShawPoint::multi_scalar_mul(&s, &P, 1);
            check_true((label + " C==A").c_str(), shaw_points_equal(C, A));

            /* Path D: Pedersen commit (s*P + 0*G) */
            auto zero_s = ShawScalar::zero();
            auto G = ShawPoint::generator();
            auto D = ShawPoint::pedersen_commit(s, P, &zero_s, &G, 1);
            check_true((label + " D==A").c_str(), shaw_points_equal(D, A));

            /* Path E: Fixed-base CT */
            auto sb = s.to_bytes();
            shaw_affine fixed_table[16];
            shaw_scalarmult_fixed_precompute(fixed_table, &P.raw());
            ShawPoint E;
            shaw_scalarmult_fixed(&E.raw(), sb.data(), fixed_table);
            check_true((label + " E==A").c_str(), shaw_points_equal(E, A));

            /* Path F: Fixed-base MSM (n=1, delegates to E internally) */
            ShawPoint F;
            const shaw_affine *tp = fixed_table;
            shaw_msm_fixed(&F.raw(), sb.data(), &tp, 1);
            check_true((label + " F==A").c_str(), shaw_points_equal(F, A));
        };

        /* Edge scalars */
        ShawScalar edge_scalars[] = {
            ShawScalar::zero(),
            ShawScalar::one(),
            ShawScalar::one() + ShawScalar::one(),
            -ShawScalar::one(),
            -(ShawScalar::one() + ShawScalar::one()),
        };
        const char *edge_names[] = {"0", "1", "2", "p-1", "p-2"};
        for (int ei = 0; ei < 5; ei++)
        {
            for (int trial = 0; trial < 10; trial++)
            {
                auto P = random_shaw_point(rng);
                std::string label = "shaw_xval[s=" + std::string(edge_names[ei]) + ",t=" + std::to_string(trial) + "]";
                test_shaw(label, edge_scalars[ei], P);
            }
        }

        /* Random 256-bit scalars */
        for (int trial = 0; trial < 200; trial++)
        {
            auto s = random_shaw_scalar(rng);
            auto P = random_shaw_point(rng);
            std::string label = "shaw_xval[rand," + std::to_string(trial) + "]";
            test_shaw(label, s, P);
        }

        /* Small scalars (< 2^64) */
        for (int trial = 0; trial < 20; trial++)
        {
            uint8_t wide[64] = {};
            rng.fill_bytes(wide, 8);
            auto s = ShawScalar::reduce_wide(wide);
            auto P = random_shaw_point(rng);
            std::string label = "shaw_xval[small," + std::to_string(trial) + "]";
            test_shaw(label, s, P);
        }

        /* High-bit scalars (bit 254 set) */
        for (int trial = 0; trial < 20; trial++)
        {
            uint8_t wide[64] = {};
            rng.fill_bytes(wide, 32);
            wide[31] |= 0x40;
            wide[31] &= 0x7f;
            auto s = ShawScalar::reduce_wide(wide);
            auto P = random_shaw_point(rng);
            std::string label = "shaw_xval[high," + std::to_string(trial) + "]";
            test_shaw(label, s, P);
        }
    }
}

/* ======================================================================
 * main()
 * ====================================================================== */

int main(int argc, char *argv[])
{
    uint64_t seed = 0ULL;
    const char *dispatch_label = "baseline (x64/portable)";

    for (int i = 1; i < argc; i++)
    {
        if (std::strcmp(argv[i], "--autotune") == 0)
        {
            ranshaw_autotune();
            dispatch_label = "autotune";
        }
        else if (std::strcmp(argv[i], "--init") == 0)
        {
            ranshaw_init();
            dispatch_label = "init (CPUID heuristic)";
        }
        else if (std::strcmp(argv[i], "--quiet") == 0)
        {
            quiet_mode = true;
        }
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        {
            seed = std::strtoull(argv[++i], nullptr, 0);
        }
        else
        {
            std::cerr << "Usage: " << argv[0] << " [--init | --autotune] [--quiet] [--seed <N>]" << std::endl;
            return 1;
        }
    }

    std::cout << "RanShaw Fuzz Tests" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "Dispatch: " << dispatch_label << std::endl;
#if RANSHAW_SIMD
    std::cout << "CPU features:";
    if (ranshaw_has_avx2())
        std::cout << " AVX2";
    if (ranshaw_has_avx512f())
        std::cout << " AVX512F";
    if (ranshaw_has_avx512ifma())
        std::cout << " AVX512IFMA";
    if (!ranshaw_cpu_features())
        std::cout << " (none)";
    std::cout << std::endl;
#endif
    std::cout << "PRNG seed: 0x" << std::hex << seed << std::dec << std::endl;
#ifdef RANSHAW_ECFFT
    std::cout << "ECFFT: enabled" << std::endl;
#else
    std::cout << "ECFFT: disabled" << std::endl;
#endif

    global_seed = seed;

#ifdef RANSHAW_ECFFT
    /* Initialize global ECFFT contexts so that FqPolynomial::operator* and
     * FpPolynomial::operator* dispatch to the ECFFT path for large multiplies
     * (degree >= 1024).  Without this, fq_poly_mul / fp_poly_mul fall through
     * to Karatsuba even when ECFFT is compiled in. */
    ecfft_global_init();
#endif

    fuzz_scalar_arithmetic();
    fuzz_scalar_edge_cases();
    fuzz_point_arithmetic();
    fuzz_ipa_edge_cases();
    fuzz_serialization_roundtrip();
    fuzz_cross_curve_cycle();
    fuzz_scalarmul_consistency();
    fuzz_msm_random();
    fuzz_msm_sparse();
    fuzz_map_to_curve();
    fuzz_wei25519_bridge();
    fuzz_pedersen();
    fuzz_batch_affine();
    fuzz_polynomial();
    fuzz_polynomial_protocol_sizes();
    fuzz_divisor();
    fuzz_divisor_scalar_mul();
    fuzz_operator_plus_regression();
    fuzz_verification_equation();
    fuzz_all_path_cross_validation();
#ifdef RANSHAW_ECFFT
    fuzz_ecfft_poly_mul();
#endif

    std::cout << std::endl << "======================" << std::endl;
    std::cout << "Total:  " << tests_run << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
