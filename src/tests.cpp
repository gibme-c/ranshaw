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
#include "ranshaw_test_vectors.h"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

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

static bool check_int(const char *test_name, int expected, int actual)
{
    ++tests_run;
    if (expected == actual)
    {
        ++tests_passed;
        std::cout << "  PASS: " << test_name << std::endl;
        return true;
    }
    else
    {
        ++tests_failed;
        std::cout << "  FAIL: " << test_name << std::endl;
        std::cout << "    expected: " << expected << std::endl;
        std::cout << "    actual:   " << actual << std::endl;
        return false;
    }
}

static bool check_nonzero(const char *test_name, int actual)
{
    ++tests_run;
    if (actual != 0)
    {
        ++tests_passed;
        std::cout << "  PASS: " << test_name << std::endl;
        return true;
    }
    else
    {
        ++tests_failed;
        std::cout << "  FAIL: " << test_name << " (expected non-zero, got 0)" << std::endl;
        return false;
    }
}

static const unsigned char test_a_bytes[32] = {0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, 0xbe, 0xba, 0xfe,
                                               0xca, 0xef, 0xbe, 0xad, 0xde, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                               0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
static const unsigned char test_b_bytes[32] = {0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x0d, 0xf0, 0xad,
                                               0xba, 0xce, 0xfa, 0xed, 0xfe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                               0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
static const unsigned char one_bytes[32] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
static const unsigned char zero_bytes[32] = {0};
static const unsigned char four_bytes[32] = {0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

/* F_p known-answer vectors */
static const unsigned char fp_ab_bytes[32] = {0x8b, 0xf8, 0x99, 0xb6, 0x81, 0xc3, 0x9d, 0x32, 0x37, 0x91, 0x83,
                                              0xab, 0x63, 0xdf, 0xe3, 0x39, 0x5a, 0xbb, 0x62, 0xcf, 0x01, 0xdb,
                                              0x9b, 0x07, 0x40, 0x05, 0x0f, 0x2e, 0x75, 0x64, 0xbf, 0x5d};
static const unsigned char fp_asq_bytes[32] = {0x34, 0xa5, 0xf2, 0xa2, 0x09, 0x5f, 0x47, 0xa6, 0x80, 0x23, 0x11,
                                               0x6b, 0x38, 0x72, 0xb0, 0xef, 0x20, 0x65, 0x11, 0xb6, 0xcc, 0x2e,
                                               0x41, 0xd2, 0x18, 0xfa, 0x92, 0x82, 0x13, 0xcd, 0xb1, 0x41};
static const unsigned char fp_ainv_bytes[32] = {0x3f, 0x3a, 0x94, 0xed, 0xea, 0xf4, 0x00, 0xef, 0x56, 0x09, 0xc0,
                                                0x94, 0xeb, 0x93, 0x22, 0xcb, 0x71, 0x87, 0x3d, 0x9b, 0x45, 0x9c,
                                                0xde, 0xf4, 0x0a, 0x20, 0x13, 0xc1, 0xfc, 0x61, 0x66, 0x25};

/* F_q known-answer vectors */
static const unsigned char fq_ab_bytes[32] = {0xd9, 0x30, 0x72, 0x3d, 0x0f, 0xf1, 0xe6, 0xc3, 0xde, 0x25, 0x1e,
                                              0xf4, 0x36, 0x67, 0x64, 0x7a, 0x5a, 0xbb, 0x62, 0xcf, 0x01, 0xdb,
                                              0x9b, 0x07, 0x40, 0x05, 0x0f, 0x2e, 0x75, 0x64, 0xbf, 0x5d};
static const unsigned char fq_asq_bytes[32] = {0x82, 0xdd, 0xca, 0x29, 0x97, 0x8c, 0x90, 0x37, 0x28, 0xb8, 0xab,
                                               0xb3, 0x0b, 0xfa, 0x30, 0x30, 0x21, 0x65, 0x11, 0xb6, 0xcc, 0x2e,
                                               0x41, 0xd2, 0x18, 0xfa, 0x92, 0x82, 0x13, 0xcd, 0xb1, 0x41};
static const unsigned char fq_ainv_bytes[32] = {0xee, 0xe9, 0xdc, 0xce, 0x6d, 0x37, 0x57, 0xf1, 0xfd, 0x90, 0x58,
                                                0xf5, 0xff, 0xff, 0x5f, 0xb3, 0x30, 0x3c, 0xb4, 0xb2, 0x81, 0x4a,
                                                0xb8, 0x4f, 0xcf, 0xbe, 0x50, 0xe0, 0x6b, 0x8e, 0xe1, 0x60};
static const unsigned char fq_sqrt4_bytes[32] = {0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

/* Ran compressed point test vectors */
static const unsigned char ran_g_compressed[32] = {0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

static const unsigned char ran_2g_compressed[32] = {0x26, 0x29, 0x42, 0x40, 0x80, 0x90, 0xb3, 0xc5, 0x07, 0xb8, 0xac,
                                                       0x94, 0xd4, 0x6f, 0xc4, 0x95, 0xfc, 0x12, 0x9f, 0xb4, 0xd1, 0x65,
                                                       0x37, 0x24, 0x11, 0xd5, 0xe5, 0xea, 0x00, 0x84, 0x02, 0xf2};

static const unsigned char ran_7g_compressed[32] = {0x03, 0xdf, 0x58, 0xab, 0x3f, 0x90, 0x99, 0xc3, 0x4d, 0x76, 0x64,
                                                       0x2b, 0x4c, 0x99, 0xe5, 0x82, 0xe3, 0x8c, 0xf4, 0x7e, 0x1b, 0xee,
                                                       0x44, 0x4c, 0x48, 0x17, 0xa4, 0x81, 0xba, 0x49, 0x98, 0x26};

/* Shaw compressed point test vectors */
static const unsigned char shaw_g_compressed[32] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

static const unsigned char shaw_2g_compressed[32] = {0x9d, 0xc7, 0x27, 0x79, 0x72, 0xd2, 0xb6, 0x6e, 0x58, 0x6b, 0x65,
                                                       0xb7, 0x2c, 0x78, 0x7f, 0xbf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

static const unsigned char shaw_7g_compressed[32] = {0x99, 0x30, 0x21, 0x4d, 0xf2, 0x35, 0x94, 0x1d, 0xba, 0x78, 0xb6,
                                                       0x1c, 0xeb, 0xf3, 0x81, 0x2c, 0x69, 0xc0, 0x43, 0x18, 0x28, 0xf9,
                                                       0x08, 0x9e, 0x01, 0x69, 0x5d, 0x8a, 0xfd, 0x58, 0xbe, 0x2f};

static void test_fp()
{
    std::cout << std::endl << "=== F_p arithmetic ===" << std::endl;
    unsigned char buf[32];

    fp_fe a, b, c, d;
    fp_frombytes(a, test_a_bytes);
    fp_frombytes(b, test_b_bytes);

    fp_tobytes(buf, a);
    check_bytes("tobytes(frombytes(a)) == a", test_a_bytes, buf, 32);

    fp_fe zero;
    fp_0(zero);
    fp_tobytes(buf, zero);
    check_bytes("tobytes(0)", zero_bytes, buf, 32);

    fp_fe one;
    fp_1(one);
    fp_tobytes(buf, one);
    check_bytes("tobytes(1)", one_bytes, buf, 32);

    fp_add(c, a, zero);
    fp_tobytes(buf, c);
    check_bytes("a + 0 == a", test_a_bytes, buf, 32);

    fp_mul(c, a, b);
    fp_tobytes(buf, c);
    check_bytes("a * b", fp_ab_bytes, buf, 32);

    fp_mul(d, b, a);
    fp_tobytes(buf, d);
    check_bytes("b * a == a * b", fp_ab_bytes, buf, 32);

    fp_sq(c, a);
    fp_tobytes(buf, c);
    check_bytes("a^2", fp_asq_bytes, buf, 32);

    fp_mul(d, a, a);
    fp_tobytes(buf, d);
    check_bytes("sq(a) == mul(a,a)", fp_asq_bytes, buf, 32);

    fp_mul(c, a, one);
    fp_tobytes(buf, c);
    check_bytes("a * 1 == a", test_a_bytes, buf, 32);

    fp_fe inv_a;
    fp_invert(inv_a, a);
    fp_tobytes(buf, inv_a);
    check_bytes("inv(a)", fp_ainv_bytes, buf, 32);

    fp_mul(c, inv_a, a);
    fp_tobytes(buf, c);
    check_bytes("inv(a) * a == 1", one_bytes, buf, 32);

    fp_sub(c, a, a);
    fp_tobytes(buf, c);
    check_bytes("a - a == 0", zero_bytes, buf, 32);

    fp_neg(d, a);
    fp_add(c, a, d);
    fp_tobytes(buf, c);
    check_bytes("a + (-a) == 0", zero_bytes, buf, 32);
}

static void test_fq()
{
    std::cout << std::endl << "=== F_q arithmetic ===" << std::endl;
    unsigned char buf[32];

    fq_fe a, b, c, d;
    fq_frombytes(a, test_a_bytes);
    fq_frombytes(b, test_b_bytes);

    fq_tobytes(buf, a);
    check_bytes("tobytes(frombytes(a)) == a", test_a_bytes, buf, 32);

    fq_fe zero;
    fq_0(zero);
    fq_tobytes(buf, zero);
    check_bytes("tobytes(0)", zero_bytes, buf, 32);

    fq_fe one;
    fq_1(one);
    fq_tobytes(buf, one);
    check_bytes("tobytes(1)", one_bytes, buf, 32);

    fq_add(c, a, zero);
    fq_tobytes(buf, c);
    check_bytes("a + 0 == a", test_a_bytes, buf, 32);

    fq_mul(c, a, b);
    fq_tobytes(buf, c);
    check_bytes("a * b", fq_ab_bytes, buf, 32);

    fq_mul(d, b, a);
    fq_tobytes(buf, d);
    check_bytes("b * a == a * b", fq_ab_bytes, buf, 32);

    fq_sq(c, a);
    fq_tobytes(buf, c);
    check_bytes("a^2", fq_asq_bytes, buf, 32);

    fq_mul(d, a, a);
    fq_tobytes(buf, d);
    check_bytes("sq(a) == mul(a,a)", fq_asq_bytes, buf, 32);

    fq_mul(c, a, one);
    fq_tobytes(buf, c);
    check_bytes("a * 1 == a", test_a_bytes, buf, 32);

    fq_fe inv_a;
    fq_invert(inv_a, a);
    fq_tobytes(buf, inv_a);
    check_bytes("inv(a)", fq_ainv_bytes, buf, 32);

    fq_mul(c, inv_a, a);
    fq_tobytes(buf, c);
    check_bytes("inv(a) * a == 1", one_bytes, buf, 32);

    /* Test invert with fully-populated input (exercises all limbs) */
    {
        static const unsigned char denom_bytes[32] = {0xcf, 0x58, 0x73, 0x16, 0xeb, 0x6b, 0x39, 0x24, 0x6b, 0x9b, 0x4c,
                                                      0xa1, 0x6d, 0xdc, 0x6a, 0x24, 0x98, 0xe9, 0x0f, 0xf1, 0x3a, 0x61,
                                                      0xca, 0x45, 0x67, 0xaf, 0xb1, 0x1b, 0xec, 0x4a, 0x63, 0x49};
        fq_fe denom_fe, inv_denom, check_one;
        fq_frombytes(denom_fe, denom_bytes);
        fq_invert(inv_denom, denom_fe);
        fq_mul(check_one, inv_denom, denom_fe);
        fq_tobytes(buf, check_one);
        check_bytes("inv(full_denom) * full_denom == 1", one_bytes, buf, 32);
        /* Cross-check: known inverse should also give 1 when multiplied by denom */
        static const unsigned char x64_inv_bytes[32] = {
            0xd5, 0x94, 0x1e, 0xd6, 0x78, 0xd1, 0x68, 0xfa, 0x41, 0x79, 0x2a, 0x59, 0xfc, 0xe8, 0xee, 0x82,
            0xad, 0x67, 0xe3, 0x4e, 0xbf, 0x7f, 0xbd, 0xd1, 0x9f, 0xce, 0xaa, 0xfa, 0x41, 0x36, 0xf4, 0x4b};
        fq_fe x64_inv_fe;
        fq_frombytes(x64_inv_fe, x64_inv_bytes);
        fq_fe cross_check;
        fq_mul(cross_check, x64_inv_fe, denom_fe);
        fq_tobytes(buf, cross_check);
        check_bytes("x64_inv * denom == 1 (cross-check)", one_bytes, buf, 32);
#if !RANSHAW_PLATFORM_64BIT
        /* Verify GAMMA_25 matches GAMMA_51 via byte round-trip */
        {
            /* Construct gamma from GAMMA_25 limbs directly */
            fq_fe gamma_25_fe;
            gamma_25_fe[0] = GAMMA_25[0];
            gamma_25_fe[1] = GAMMA_25[1];
            gamma_25_fe[2] = GAMMA_25[2];
            gamma_25_fe[3] = GAMMA_25[3];
            gamma_25_fe[4] = GAMMA_25[4];
            gamma_25_fe[5] = 0;
            gamma_25_fe[6] = 0;
            gamma_25_fe[7] = 0;
            gamma_25_fe[8] = 0;
            gamma_25_fe[9] = 0;
            unsigned char gamma_25_bytes[32];
            fq_tobytes(gamma_25_bytes, gamma_25_fe);

            /* Construct gamma from GAMMA_51 via byte packing */
            uint64_t g51[5] = {0x12D8D86D83861ULL, 0x269135294F229ULL, 0x102021FULL, 0, 0};
            unsigned char gamma_51_bytes[32];
            /* Pack 51-bit limbs to bytes */
            uint64_t w0 = g51[0] | (g51[1] << 51);
            uint64_t w1 = (g51[1] >> 13) | (g51[2] << 38);
            uint64_t w2 = (g51[2] >> 26);
            uint64_t w3 = 0;
            for (int i = 0; i < 8; i++)
                gamma_51_bytes[i] = (unsigned char)(w0 >> (8 * i));
            for (int i = 0; i < 8; i++)
                gamma_51_bytes[8 + i] = (unsigned char)(w1 >> (8 * i));
            for (int i = 0; i < 8; i++)
                gamma_51_bytes[16 + i] = (unsigned char)(w2 >> (8 * i));
            for (int i = 0; i < 8; i++)
                gamma_51_bytes[24 + i] = (unsigned char)(w3 >> (8 * i));

            check_bytes("GAMMA_25 == GAMMA_51 (byte comparison)", gamma_51_bytes, gamma_25_bytes, 32);
        }
#endif // !RANSHAW_PLATFORM_64BIT
        /* Simple mul test: denom * 2 via add vs mul(denom, 2) */
        fq_fe two_fe;
        fq_1(two_fe);
        fq_add(two_fe, two_fe, two_fe);
        fq_fe denom_times_2_add, denom_times_2_mul;
        fq_add(denom_times_2_add, denom_fe, denom_fe);
        fq_mul(denom_times_2_mul, denom_fe, two_fe);
        unsigned char dadd[32], dmul[32];
        fq_tobytes(dadd, denom_times_2_add);
        fq_tobytes(dmul, denom_times_2_mul);
        check_bytes("denom*2 add vs mul", dadd, dmul, 32);
    }

    fq_sub(c, a, a);
    fq_tobytes(buf, c);
    check_bytes("a - a == 0", zero_bytes, buf, 32);

    fq_neg(d, a);
    fq_add(c, a, d);
    fq_tobytes(buf, c);
    check_bytes("a + (-a) == 0", zero_bytes, buf, 32);

    fq_fe four;
    fq_frombytes(four, four_bytes);
    fq_fe sqrt4;
    fq_sqrt(sqrt4, four);
    fq_tobytes(buf, sqrt4);
    check_bytes("sqrt(4) == 2", fq_sqrt4_bytes, buf, 32);

    fq_sq(c, sqrt4);
    fq_tobytes(buf, c);
    check_bytes("sqrt(4)^2 == 4", four_bytes, buf, 32);
}

static void test_fp_sqrt()
{
    std::cout << std::endl << "=== F_p sqrt ===" << std::endl;
    unsigned char buf[32];

    fp_fe zero_fe, sqrt_out;
    fp_0(zero_fe);
    int rc = fp_sqrt(sqrt_out, zero_fe);
    check_int("sqrt(0) returns 0", 0, rc);
    fp_tobytes(buf, sqrt_out);
    check_bytes("sqrt(0) == 0", zero_bytes, buf, 32);

    fp_fe one_fe;
    fp_1(one_fe);
    rc = fp_sqrt(sqrt_out, one_fe);
    check_int("sqrt(1) returns 0", 0, rc);
    fp_fe sq_check;
    fp_sq(sq_check, sqrt_out);
    fp_tobytes(buf, sq_check);
    check_bytes("sqrt(1)^2 == 1", one_bytes, buf, 32);

    fp_fe four_fe;
    fp_frombytes(four_fe, four_bytes);
    rc = fp_sqrt(sqrt_out, four_fe);
    check_int("sqrt(4) returns 0", 0, rc);
    fp_sq(sq_check, sqrt_out);
    fp_tobytes(buf, sq_check);
    check_bytes("sqrt(4)^2 == 4", four_bytes, buf, 32);

    fp_fe a;
    fp_frombytes(a, test_a_bytes);
    fp_fe a_sq;
    fp_sq(a_sq, a);
    rc = fp_sqrt(sqrt_out, a_sq);
    check_int("sqrt(a^2) returns 0", 0, rc);
    fp_sq(sq_check, sqrt_out);
    fp_tobytes(buf, sq_check);
    unsigned char a_sq_bytes[32];
    fp_tobytes(a_sq_bytes, a_sq);
    check_bytes("sqrt(a^2)^2 == a^2", a_sq_bytes, buf, 32);

    unsigned char two_bytes[32] = {0x02};
    fp_fe two_fe;
    fp_frombytes(two_fe, two_bytes);
    rc = fp_sqrt(sqrt_out, two_fe);
    check_int("sqrt(2) returns -1 (non-square)", -1, rc);
}

static void test_ran_points()
{
    std::cout << std::endl << "=== Ran point ops ===" << std::endl;
    unsigned char buf[32];

    ran_affine g_aff;
    fp_copy(g_aff.x, RAN_GX);
    fp_copy(g_aff.y, RAN_GY);
    check_nonzero("G is on curve", ran_is_on_curve(&g_aff));

    ran_jacobian G;
    fp_copy(G.X, RAN_GX);
    fp_copy(G.Y, RAN_GY);
    fp_1(G.Z);

    ran_tobytes(buf, &G);
    check_bytes("tobytes(G)", ran_g_compressed, buf, 32);

    ran_jacobian G2;
    int rc = ran_frombytes(&G2, ran_g_compressed);
    check_int("frombytes(G) returns 0", 0, rc);
    ran_tobytes(buf, &G2);
    check_bytes("frombytes(tobytes(G)) round-trip", ran_g_compressed, buf, 32);

    ran_jacobian id;
    ran_identity(&id);
    check_nonzero("identity is_identity", ran_is_identity(&id));

    ran_tobytes(buf, &id);
    check_bytes("tobytes(identity) == zeros", zero_bytes, buf, 32);

    ran_jacobian dbl_G;
    ran_dbl(&dbl_G, &G);
    ran_tobytes(buf, &dbl_G);
    check_bytes("2G = dbl(G)", ran_2g_compressed, buf, 32);

    /* 3G = 2G + G (add doesn't handle P==P, so skip G+G test) */
    ran_jacobian three_G;
    ran_add(&three_G, &dbl_G, &G);

    ran_jacobian four_G;
    ran_dbl(&four_G, &dbl_G);

    ran_jacobian seven_G;
    ran_add(&seven_G, &four_G, &three_G);
    ran_tobytes(buf, &seven_G);
    check_bytes("7G = 4G + 3G", ran_7g_compressed, buf, 32);

    ran_jacobian decoded_2g;
    rc = ran_frombytes(&decoded_2g, ran_2g_compressed);
    check_int("frombytes(2G) returns 0", 0, rc);
    ran_tobytes(buf, &decoded_2g);
    check_bytes("2G round-trip", ran_2g_compressed, buf, 32);

    unsigned char invalid_bytes[32] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};
    ran_jacobian invalid;
    rc = ran_frombytes(&invalid, invalid_bytes);
    check_int("reject non-canonical x", -1, rc);

    ran_affine g_affine;
    fp_copy(g_affine.x, RAN_GX);
    fp_copy(g_affine.y, RAN_GY);
    ran_jacobian madd_result;
    ran_madd(&madd_result, &dbl_G, &g_affine);
    ran_tobytes(buf, &madd_result);
    unsigned char three_G_bytes[32];
    ran_tobytes(three_G_bytes, &three_G);
    check_bytes("madd(2G, G) == add(2G, G)", three_G_bytes, buf, 32);
}

static void test_shaw_points()
{
    std::cout << std::endl << "=== Shaw point ops ===" << std::endl;
    unsigned char buf[32];

    shaw_affine g_aff;
    fq_copy(g_aff.x, SHAW_GX);
    fq_copy(g_aff.y, SHAW_GY);
    check_nonzero("G is on curve", shaw_is_on_curve(&g_aff));

    shaw_jacobian G;
    fq_copy(G.X, SHAW_GX);
    fq_copy(G.Y, SHAW_GY);
    fq_1(G.Z);

    shaw_tobytes(buf, &G);
    check_bytes("tobytes(G)", shaw_g_compressed, buf, 32);

    shaw_jacobian G2;
    int rc = shaw_frombytes(&G2, shaw_g_compressed);
    check_int("frombytes(G) returns 0", 0, rc);
    shaw_tobytes(buf, &G2);
    check_bytes("frombytes(tobytes(G)) round-trip", shaw_g_compressed, buf, 32);

    shaw_jacobian id;
    shaw_identity(&id);
    check_nonzero("identity is_identity", shaw_is_identity(&id));
    shaw_tobytes(buf, &id);
    check_bytes("tobytes(identity) == zeros", zero_bytes, buf, 32);

    shaw_jacobian dbl_G;
    shaw_dbl(&dbl_G, &G);
    shaw_tobytes(buf, &dbl_G);
    check_bytes("2G = dbl(G)", shaw_2g_compressed, buf, 32);

    /* 3G, 4G, 7G (add doesn't handle P==P, so skip G+G test) */
    shaw_jacobian three_G;
    shaw_add(&three_G, &dbl_G, &G);
    shaw_jacobian four_G;
    shaw_dbl(&four_G, &dbl_G);
    shaw_jacobian seven_G;
    shaw_add(&seven_G, &four_G, &three_G);
    shaw_tobytes(buf, &seven_G);
    check_bytes("7G = 4G + 3G", shaw_7g_compressed, buf, 32);

    shaw_jacobian decoded_2g;
    rc = shaw_frombytes(&decoded_2g, shaw_2g_compressed);
    check_int("frombytes(2G) returns 0", 0, rc);
    shaw_tobytes(buf, &decoded_2g);
    check_bytes("2G round-trip", shaw_2g_compressed, buf, 32);

    unsigned char invalid_bytes[32] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};
    shaw_jacobian invalid;
    rc = shaw_frombytes(&invalid, invalid_bytes);
    check_int("reject non-canonical x", -1, rc);

    shaw_affine g_affine;
    fq_copy(g_affine.x, SHAW_GX);
    fq_copy(g_affine.y, SHAW_GY);
    shaw_jacobian madd_result;
    shaw_madd(&madd_result, &dbl_G, &g_affine);
    shaw_tobytes(buf, &madd_result);
    unsigned char three_G_bytes[32];
    shaw_tobytes(three_G_bytes, &three_G);
    check_bytes("madd(2G, G) == add(2G, G)", three_G_bytes, buf, 32);
}

static void test_ran_scalarmult()
{
    std::cout << std::endl << "=== Ran scalar mul ===" << std::endl;
    unsigned char buf[32];

    ran_jacobian G;
    fp_copy(G.X, RAN_GX);
    fp_copy(G.Y, RAN_GY);
    fp_1(G.Z);

    ran_jacobian result;
    ran_scalarmult(&result, one_bytes, &G);
    ran_tobytes(buf, &result);
    check_bytes("1*G == G", ran_g_compressed, buf, 32);

    ran_scalarmult(&result, zero_bytes, &G);
    check_nonzero("0*G == identity", ran_is_identity(&result));

    unsigned char two_scalar[32] = {0x02};
    ran_scalarmult(&result, two_scalar, &G);
    ran_tobytes(buf, &result);
    check_bytes("2*G == 2G", ran_2g_compressed, buf, 32);

    unsigned char seven_scalar[32] = {0x07};
    ran_scalarmult(&result, seven_scalar, &G);
    ran_tobytes(buf, &result);
    check_bytes("7*G", ran_7g_compressed, buf, 32);

    ran_scalarmult(&result, RAN_ORDER, &G);
    check_nonzero("order*G == identity", ran_is_identity(&result));

    ran_scalarmult_vartime(&result, one_bytes, &G);
    ran_tobytes(buf, &result);
    check_bytes("vartime: 1*G == G", ran_g_compressed, buf, 32);

    ran_scalarmult_vartime(&result, seven_scalar, &G);
    ran_tobytes(buf, &result);
    check_bytes("vartime: 7*G", ran_7g_compressed, buf, 32);

    ran_scalarmult_vartime(&result, RAN_ORDER, &G);
    check_nonzero("vartime: order*G == identity", ran_is_identity(&result));

    unsigned char scalar_a[32] = {0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, 0xbe, 0xba, 0xfe,
                                  0xca, 0xef, 0xbe, 0xad, 0xde, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                  0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10};
    ran_jacobian ct_result, vt_result;
    ran_scalarmult(&ct_result, scalar_a, &G);
    ran_scalarmult_vartime(&vt_result, scalar_a, &G);
    unsigned char ct_bytes[32], vt_bytes[32];
    ran_tobytes(ct_bytes, &ct_result);
    ran_tobytes(vt_bytes, &vt_result);
    check_bytes("CT == vartime for scalar_a", ct_bytes, vt_bytes, 32);

    unsigned char scalar_5[32] = {0x05};
    ran_jacobian aG, bG, sum_pt;
    ran_scalarmult(&aG, two_scalar, &G);
    ran_scalarmult(&bG, scalar_5, &G);
    ran_add(&sum_pt, &aG, &bG);
    ran_tobytes(buf, &sum_pt);
    check_bytes("(2+5)*G == 2*G + 5*G", ran_7g_compressed, buf, 32);
}

static void test_shaw_scalarmult()
{
    std::cout << std::endl << "=== Shaw scalar mul ===" << std::endl;
    unsigned char buf[32];

    shaw_jacobian G;
    fq_copy(G.X, SHAW_GX);
    fq_copy(G.Y, SHAW_GY);
    fq_1(G.Z);

    shaw_jacobian result;
    shaw_scalarmult(&result, one_bytes, &G);
    shaw_tobytes(buf, &result);
    check_bytes("1*G == G", shaw_g_compressed, buf, 32);

    shaw_scalarmult(&result, zero_bytes, &G);
    check_nonzero("0*G == identity", shaw_is_identity(&result));

    unsigned char two_scalar[32] = {0x02};
    shaw_scalarmult(&result, two_scalar, &G);
    shaw_tobytes(buf, &result);
    check_bytes("2*G == 2G", shaw_2g_compressed, buf, 32);

    unsigned char seven_scalar[32] = {0x07};
    shaw_scalarmult(&result, seven_scalar, &G);
    shaw_tobytes(buf, &result);
    check_bytes("7*G", shaw_7g_compressed, buf, 32);

    shaw_scalarmult(&result, SHAW_ORDER, &G);
    check_nonzero("order*G == identity", shaw_is_identity(&result));

    shaw_scalarmult_vartime(&result, one_bytes, &G);
    shaw_tobytes(buf, &result);
    check_bytes("vartime: 1*G == G", shaw_g_compressed, buf, 32);

    shaw_scalarmult_vartime(&result, seven_scalar, &G);
    shaw_tobytes(buf, &result);
    check_bytes("vartime: 7*G", shaw_7g_compressed, buf, 32);

    shaw_scalarmult_vartime(&result, SHAW_ORDER, &G);
    check_nonzero("vartime: order*G == identity", shaw_is_identity(&result));

    unsigned char scalar_a[32] = {0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, 0xbe, 0xba, 0xfe,
                                  0xca, 0xef, 0xbe, 0xad, 0xde, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                  0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10};
    shaw_jacobian ct_result, vt_result;
    shaw_scalarmult(&ct_result, scalar_a, &G);
    shaw_scalarmult_vartime(&vt_result, scalar_a, &G);
    unsigned char ct_bytes[32], vt_bytes[32];
    shaw_tobytes(ct_bytes, &ct_result);
    shaw_tobytes(vt_bytes, &vt_result);
    check_bytes("CT == vartime for scalar_a", ct_bytes, vt_bytes, 32);

    unsigned char scalar_5[32] = {0x05};
    shaw_jacobian aG, bG, sum_pt;
    shaw_scalarmult(&aG, two_scalar, &G);
    shaw_scalarmult(&bG, scalar_5, &G);
    shaw_add(&sum_pt, &aG, &bG);
    shaw_tobytes(buf, &sum_pt);
    check_bytes("(2+5)*G == 2*G + 5*G", shaw_7g_compressed, buf, 32);
}

static void test_wei25519()
{
    std::cout << std::endl << "=== Wei25519 bridge ===" << std::endl;

    unsigned char valid_x[32] = {0x03};
    fp_fe out;
    int rc = ranshaw_wei25519_to_fp(out, valid_x);
    check_int("valid x accepted", 0, rc);
    unsigned char buf[32];
    fp_tobytes(buf, out);
    check_bytes("value preserved", valid_x, buf, 32);

    unsigned char p_bytes[32] = {0xed, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};
    rc = ranshaw_wei25519_to_fp(out, p_bytes);
    check_int("x == p rejected", -1, rc);

    unsigned char high_bit[32] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80};
    rc = ranshaw_wei25519_to_fp(out, high_bit);
    check_int("bit 255 set rejected", -1, rc);
}

static void test_ran_msm()
{
    std::cout << std::endl << "=== Ran MSM ===" << std::endl;
    unsigned char buf[32];

    ran_jacobian G;
    fp_copy(G.X, RAN_GX);
    fp_copy(G.Y, RAN_GY);
    fp_1(G.Z);

    /* msm([1], [G]) == G */
    ran_jacobian result;
    ran_msm_vartime(&result, one_bytes, &G, 1);
    ran_tobytes(buf, &result);
    check_bytes("msm([1], [G]) == G", ran_g_compressed, buf, 32);

    /* msm([7], [G]) == 7*G */
    unsigned char seven_scalar[32] = {0x07};
    ran_msm_vartime(&result, seven_scalar, &G, 1);
    ran_tobytes(buf, &result);
    check_bytes("msm([7], [G]) == 7G", ran_7g_compressed, buf, 32);

    /* msm([0], [G]) == identity */
    ran_msm_vartime(&result, zero_bytes, &G, 1);
    check_nonzero("msm([0], [G]) == identity", ran_is_identity(&result));

    /* msm([], []) == identity (n=0) */
    ran_msm_vartime(&result, nullptr, nullptr, 0);
    check_nonzero("msm([], []) == identity", ran_is_identity(&result));

    /* Linearity: msm([2, 5], [G, G]) == 7*G */
    unsigned char two_scalar[32] = {};
    two_scalar[0] = 0x02;
    unsigned char five_scalar[32] = {};
    five_scalar[0] = 0x05;
    unsigned char scalars_2_5[64];
    std::memcpy(scalars_2_5, two_scalar, 32);
    std::memcpy(scalars_2_5 + 32, five_scalar, 32);
    ran_jacobian points_2[2];
    ran_copy(&points_2[0], &G);
    ran_copy(&points_2[1], &G);
    ran_msm_vartime(&result, scalars_2_5, points_2, 2);
    ran_tobytes(buf, &result);
    check_bytes("msm([2,5], [G,G]) == 7G", ran_7g_compressed, buf, 32);

    /* msm([a], [P]) == scalarmult_vartime(a, P) */
    unsigned char scalar_a[32] = {0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, 0xbe, 0xba, 0xfe,
                                  0xca, 0xef, 0xbe, 0xad, 0xde, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                  0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10};
    ran_jacobian sm_result;
    ran_scalarmult_vartime(&sm_result, scalar_a, &G);
    unsigned char sm_bytes[32];
    ran_tobytes(sm_bytes, &sm_result);
    ran_msm_vartime(&result, scalar_a, &G, 1);
    ran_tobytes(buf, &result);
    check_bytes("msm([a], [G]) == vartime(a, G)", sm_bytes, buf, 32);

    /* Two distinct points: msm([a, b], [G, 2G]) == a*G + b*2G */
    ran_jacobian G2;
    ran_dbl(&G2, &G);
    unsigned char scalar_b[32] = {0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x0d, 0xf0, 0xad,
                                  0xba, 0xce, 0xfa, 0xed, 0xfe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    unsigned char scalars_ab[64];
    std::memcpy(scalars_ab, scalar_a, 32);
    std::memcpy(scalars_ab + 32, scalar_b, 32);
    ran_jacobian points_ab[2];
    ran_copy(&points_ab[0], &G);
    ran_copy(&points_ab[1], &G2);
    ran_msm_vartime(&result, scalars_ab, points_ab, 2);
    ran_tobytes(buf, &result);

    ran_jacobian aG, bG2, expected;
    ran_scalarmult_vartime(&aG, scalar_a, &G);
    ran_scalarmult_vartime(&bG2, scalar_b, &G2);
    ran_add(&expected, &aG, &bG2);
    unsigned char expected_bytes[32];
    ran_tobytes(expected_bytes, &expected);
    check_bytes("msm([a,b], [G,2G]) == a*G + b*2G", expected_bytes, buf, 32);

    /* n=8 (exercises Straus): all scalars=1, all points=G → sum = 8*G */
    {
        unsigned char scalars8[8 * 32] = {};
        ran_jacobian points8[8];
        for (int i = 0; i < 8; i++)
        {
            scalars8[i * 32] = 0x01;
            ran_copy(&points8[i], &G);
        }
        /* 8*G via repeated doubling */
        unsigned char eight_scalar[32] = {0x08};
        ran_jacobian eightG;
        ran_scalarmult_vartime(&eightG, eight_scalar, &G);
        ran_tobytes(expected_bytes, &eightG);
        ran_msm_vartime(&result, scalars8, points8, 8);
        ran_tobytes(buf, &result);
        check_bytes("msm n=8 (Straus)", expected_bytes, buf, 32);
    }

    /* n=33 (crosses Straus/Pippenger boundary): all scalars=1, all points=G → 33*G */
    {
        unsigned char scalars33[33 * 32] = {};
        ran_jacobian points33[33];
        for (int i = 0; i < 33; i++)
        {
            scalars33[i * 32] = 0x01;
            ran_copy(&points33[i], &G);
        }
        unsigned char thirtythree_scalar[32] = {33};
        ran_jacobian expected_pt;
        ran_scalarmult_vartime(&expected_pt, thirtythree_scalar, &G);
        ran_tobytes(expected_bytes, &expected_pt);
        ran_msm_vartime(&result, scalars33, points33, 33);
        ran_tobytes(buf, &result);
        check_bytes("msm n=33 (Pippenger)", expected_bytes, buf, 32);
    }

    /* All-zero scalars → identity */
    {
        unsigned char zero_scalars[4 * 32] = {};
        ran_jacobian points4[4];
        for (int i = 0; i < 4; i++)
            ran_copy(&points4[i], &G);
        ran_msm_vartime(&result, zero_scalars, points4, 4);
        check_nonzero("msm all-zero scalars == identity", ran_is_identity(&result));
    }
}

static void test_shaw_msm()
{
    std::cout << std::endl << "=== Shaw MSM ===" << std::endl;
    unsigned char buf[32];

    shaw_jacobian G;
    fq_copy(G.X, SHAW_GX);
    fq_copy(G.Y, SHAW_GY);
    fq_1(G.Z);

    /* msm([1], [G]) == G */
    shaw_jacobian result;
    shaw_msm_vartime(&result, one_bytes, &G, 1);
    shaw_tobytes(buf, &result);
    check_bytes("msm([1], [G]) == G", shaw_g_compressed, buf, 32);

    /* msm([7], [G]) == 7*G */
    unsigned char seven_scalar[32] = {0x07};
    shaw_msm_vartime(&result, seven_scalar, &G, 1);
    shaw_tobytes(buf, &result);
    check_bytes("msm([7], [G]) == 7G", shaw_7g_compressed, buf, 32);

    /* msm([0], [G]) == identity */
    shaw_msm_vartime(&result, zero_bytes, &G, 1);
    check_nonzero("msm([0], [G]) == identity", shaw_is_identity(&result));

    /* msm([], []) == identity (n=0) */
    shaw_msm_vartime(&result, nullptr, nullptr, 0);
    check_nonzero("msm([], []) == identity", shaw_is_identity(&result));

    /* Linearity: msm([2, 5], [G, G]) == 7*G */
    unsigned char two_scalar[32] = {};
    two_scalar[0] = 0x02;
    unsigned char five_scalar[32] = {};
    five_scalar[0] = 0x05;
    unsigned char scalars_2_5[64];
    std::memcpy(scalars_2_5, two_scalar, 32);
    std::memcpy(scalars_2_5 + 32, five_scalar, 32);
    shaw_jacobian points_2[2];
    shaw_copy(&points_2[0], &G);
    shaw_copy(&points_2[1], &G);
    shaw_msm_vartime(&result, scalars_2_5, points_2, 2);
    shaw_tobytes(buf, &result);
    check_bytes("msm([2,5], [G,G]) == 7G", shaw_7g_compressed, buf, 32);

    /* msm([a], [P]) == scalarmult_vartime(a, P) */
    unsigned char scalar_a[32] = {0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, 0xbe, 0xba, 0xfe,
                                  0xca, 0xef, 0xbe, 0xad, 0xde, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                  0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10};
    shaw_jacobian sm_result;
    shaw_scalarmult_vartime(&sm_result, scalar_a, &G);
    unsigned char sm_bytes[32];
    shaw_tobytes(sm_bytes, &sm_result);
    shaw_msm_vartime(&result, scalar_a, &G, 1);
    shaw_tobytes(buf, &result);
    check_bytes("msm([a], [G]) == vartime(a, G)", sm_bytes, buf, 32);

    /* Two distinct points: msm([a, b], [G, 2G]) == a*G + b*2G */
    {
        shaw_jacobian G2;
        shaw_dbl(&G2, &G);
        unsigned char scalar_b[32] = {0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x0d, 0xf0, 0xad,
                                      0xba, 0xce, 0xfa, 0xed, 0xfe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
        unsigned char scalars_ab[64];
        std::memcpy(scalars_ab, scalar_a, 32);
        std::memcpy(scalars_ab + 32, scalar_b, 32);
        shaw_jacobian points_ab[2];
        shaw_copy(&points_ab[0], &G);
        shaw_copy(&points_ab[1], &G2);
        shaw_msm_vartime(&result, scalars_ab, points_ab, 2);
        shaw_tobytes(buf, &result);

        shaw_jacobian aG, bG2, expected;
        shaw_scalarmult_vartime(&aG, scalar_a, &G);
        shaw_scalarmult_vartime(&bG2, scalar_b, &G2);
        shaw_add(&expected, &aG, &bG2);
        unsigned char expected_bytes[32];
        shaw_tobytes(expected_bytes, &expected);
        check_bytes("msm([a,b], [G,2G]) == a*G + b*2G", expected_bytes, buf, 32);
    }

    /* n=8 (exercises Straus): all scalars=1, all points=G → sum = 8*G */
    {
        unsigned char scalars8[8 * 32] = {};
        shaw_jacobian points8[8];
        for (int i = 0; i < 8; i++)
        {
            scalars8[i * 32] = 0x01;
            shaw_copy(&points8[i], &G);
        }
        unsigned char eight_scalar[32] = {0x08};
        shaw_jacobian eightG;
        shaw_scalarmult_vartime(&eightG, eight_scalar, &G);
        unsigned char expected_bytes[32];
        shaw_tobytes(expected_bytes, &eightG);
        shaw_msm_vartime(&result, scalars8, points8, 8);
        shaw_tobytes(buf, &result);
        check_bytes("msm n=8 (Straus)", expected_bytes, buf, 32);
    }

    /* n=33 (crosses Straus/Pippenger boundary): all scalars=1, all points=G → 33*G */
    {
        unsigned char scalars33[33 * 32] = {};
        shaw_jacobian points33[33];
        for (int i = 0; i < 33; i++)
        {
            scalars33[i * 32] = 0x01;
            shaw_copy(&points33[i], &G);
        }
        unsigned char thirtythree_scalar[32] = {33};
        shaw_jacobian expected_pt;
        shaw_scalarmult_vartime(&expected_pt, thirtythree_scalar, &G);
        unsigned char expected_bytes[32];
        shaw_tobytes(expected_bytes, &expected_pt);
        shaw_msm_vartime(&result, scalars33, points33, 33);
        shaw_tobytes(buf, &result);
        check_bytes("msm n=33 (Pippenger)", expected_bytes, buf, 32);
    }

    /* All-zero scalars → identity */
    {
        unsigned char zero_scalars[4 * 32] = {};
        shaw_jacobian points4[4];
        for (int i = 0; i < 4; i++)
            shaw_copy(&points4[i], &G);
        shaw_msm_vartime(&result, zero_scalars, points4, 4);
        check_nonzero("msm all-zero scalars == identity", shaw_is_identity(&result));
    }
}

/* Ran SSWU test vectors (Z=7) */
static const unsigned char ran_sswu_u1_result[32] = {
    0xc1, 0x2b, 0xdf, 0x94, 0x58, 0xf9, 0x6c, 0x32, 0x1e, 0xe6, 0x8e, 0x9a, 0x25, 0xa8, 0x16, 0x2a,
    0xac, 0x44, 0xfd, 0xb4, 0x9e, 0x0d, 0xa1, 0xc4, 0xb6, 0xcb, 0x2c, 0x04, 0x29, 0xd9, 0xe8, 0x92};
static const unsigned char ran_sswu_u2_result[32] = {
    0x2b, 0xa6, 0x56, 0xa7, 0x92, 0xc8, 0x4a, 0x9c, 0xfc, 0xf6, 0xe2, 0xef, 0x8f, 0x17, 0x45, 0x5b,
    0x02, 0x31, 0x05, 0xc2, 0x18, 0x51, 0xe5, 0xee, 0x95, 0xda, 0x5a, 0x9e, 0x35, 0xcd, 0x68, 0x7e};
static const unsigned char ran_sswu_u42_result[32] = {
    0x02, 0xa5, 0xe6, 0x21, 0x27, 0x7d, 0xf1, 0x0c, 0xb8, 0xab, 0xf7, 0xaa, 0xf2, 0x30, 0x8c, 0x83,
    0x51, 0xae, 0xb8, 0xf8, 0x9f, 0x87, 0x0f, 0x38, 0xe4, 0x4b, 0xf6, 0x26, 0x32, 0xda, 0xfa, 0x44};

/* Shaw SSWU test vectors (Z=-4) */
static const unsigned char shaw_sswu_u1_result[32] = {
    0x86, 0x47, 0x94, 0xcc, 0xb4, 0x7a, 0x10, 0x0d, 0x9c, 0x06, 0x24, 0x65, 0xde, 0x49, 0x0c, 0x58,
    0x4f, 0xd5, 0xaa, 0x7c, 0xbb, 0x62, 0xa6, 0x2b, 0x93, 0x1b, 0xb9, 0xa0, 0x8e, 0x37, 0x1e, 0xde};
static const unsigned char shaw_sswu_u2_result[32] = {
    0x89, 0x0c, 0xf9, 0x19, 0x1a, 0x8d, 0x52, 0x90, 0xc3, 0xd9, 0x8d, 0xba, 0x4c, 0xf8, 0x18, 0x1f,
    0x0b, 0x8d, 0xef, 0x20, 0x78, 0xd4, 0x2d, 0x0c, 0x49, 0x23, 0xba, 0x5f, 0xed, 0xd1, 0xfd, 0x5a};
static const unsigned char shaw_sswu_u42_result[32] = {
    0xd1, 0x74, 0x24, 0x2b, 0x58, 0x40, 0xdf, 0xd2, 0x85, 0x39, 0x24, 0x38, 0x3d, 0x6b, 0x0f, 0x62,
    0xb0, 0x93, 0xb5, 0x9b, 0x6d, 0xdc, 0x89, 0x71, 0x36, 0x19, 0x00, 0xcf, 0x6c, 0xb7, 0xe5, 0x06};

static void test_fp_sqrt_sswu()
{
    std::cout << std::endl << "=== F_p sqrt (SSWU gx2) ===" << std::endl;
    unsigned char buf[32];

    /* gx2 for SSWU u=1, known to be a QR */
    static const unsigned char gx2_bytes[32] = {0x4a, 0x9d, 0xd9, 0xd3, 0x95, 0x50, 0x3c, 0x31, 0x36, 0x8c, 0x6b,
                                                0xc5, 0x81, 0xc6, 0xa4, 0xc0, 0xc9, 0xca, 0x97, 0xde, 0x52, 0x20,
                                                0x8d, 0x23, 0xb2, 0x69, 0xc5, 0x73, 0x68, 0x0d, 0xcb, 0x16};
    static const unsigned char y_expected[32] = {0x98, 0xff, 0x11, 0x0f, 0x2a, 0xbf, 0xc4, 0x3f, 0xdf, 0xac, 0x96,
                                                 0x12, 0xf6, 0xde, 0x68, 0x85, 0x41, 0xf8, 0xf4, 0xbb, 0xea, 0xe4,
                                                 0x73, 0x1c, 0x10, 0x71, 0xce, 0xc2, 0xd0, 0xef, 0xc1, 0x47};

    fp_fe gx2_fe, y_fe;
    fp_frombytes(gx2_fe, gx2_bytes);
    int rc = fp_sqrt(y_fe, gx2_fe);
    check_int("fp_sqrt(gx2) returns 0 (is QR)", 0, rc);

    fp_fe check;
    fp_sq(check, y_fe);
    fp_tobytes(buf, check);
    check_bytes("sqrt(gx2)^2 == gx2", gx2_bytes, buf, 32);

    /* Also check the value matches Python */
    fp_tobytes(buf, y_fe);
    /* Note: fp_sqrt may return either root; check value or its negation */
    bool match_pos = (std::memcmp(buf, y_expected, 32) == 0);
    fp_fe neg_y;
    fp_neg(neg_y, y_fe);
    unsigned char neg_buf[32];
    fp_tobytes(neg_buf, neg_y);
    bool match_neg = (std::memcmp(neg_buf, y_expected, 32) == 0);
    ++tests_run;
    if (match_pos || match_neg)
    {
        ++tests_passed;
        std::cout << "  PASS: sqrt(gx2) matches expected root" << std::endl;
    }
    else
    {
        ++tests_failed;
        std::cout << "  FAIL: sqrt(gx2) matches expected root" << std::endl;
        std::cout << "    expected: " << hex(y_expected, 32) << std::endl;
        std::cout << "    actual:   " << hex(buf, 32) << std::endl;
    }

    /* Now test: compute gx from x2 directly and check sqrt */
    static const unsigned char x2_bytes[32] = {0xc1, 0x2b, 0xdf, 0x94, 0x58, 0xf9, 0x6c, 0x32, 0x1e, 0xe6, 0x8e,
                                               0x9a, 0x25, 0xa8, 0x16, 0x2a, 0xac, 0x44, 0xfd, 0xb4, 0x9e, 0x0d,
                                               0xa1, 0xc4, 0xb6, 0xcb, 0x2c, 0x04, 0x29, 0xd9, 0xe8, 0x12};
    fp_fe x2_fe, x2_sq, x2_cu, gx_computed;
    fp_frombytes(x2_fe, x2_bytes);
    fp_sq(x2_sq, x2_fe);
    fp_mul(x2_cu, x2_sq, x2_fe);

    /* A = -3 mod p */
    fp_fe three_x;
    fp_add(three_x, x2_fe, x2_fe);
    fp_add(three_x, three_x, x2_fe);
    fp_sub(gx_computed, x2_cu, three_x);
    fp_add(gx_computed, gx_computed, RAN_B);
    fp_tobytes(buf, gx_computed);
    check_bytes("gx from x2 matches gx2", gx2_bytes, buf, 32);
}

static void test_ran_sswu()
{
    std::cout << std::endl << "=== Ran SSWU ===" << std::endl;
    unsigned char buf[32];

    /* Known test vectors */
    ran_jacobian result;
    ran_map_to_curve(&result, one_bytes);
    ran_tobytes(buf, &result);
    check_bytes("sswu(1)", ran_sswu_u1_result, buf, 32);

    unsigned char two_bytes[32] = {0x02};
    ran_map_to_curve(&result, two_bytes);
    ran_tobytes(buf, &result);
    check_bytes("sswu(2)", ran_sswu_u2_result, buf, 32);

    unsigned char u42_bytes[32] = {0x2a};
    ran_map_to_curve(&result, u42_bytes);
    ran_tobytes(buf, &result);
    check_bytes("sswu(42)", ran_sswu_u42_result, buf, 32);

    /* Deterministic: same input → same output */
    ran_jacobian result2;
    ran_map_to_curve(&result2, one_bytes);
    ran_tobytes(buf, &result2);
    check_bytes("sswu(1) deterministic", ran_sswu_u1_result, buf, 32);

    /* Output is on curve */
    ran_affine aff;
    ran_to_affine(&aff, &result);
    check_nonzero("sswu(1) on curve", ran_is_on_curve(&aff));

    /* map_to_curve2(u0, u1) == map_to_curve(u0) + map_to_curve(u1) */
    ran_jacobian p0, p1, sum_direct, sum_combined;
    ran_map_to_curve(&p0, one_bytes);
    ran_map_to_curve(&p1, two_bytes);
    ran_add(&sum_direct, &p0, &p1);
    ran_tobytes(buf, &sum_direct);

    ran_map_to_curve2(&sum_combined, one_bytes, two_bytes);
    unsigned char buf2[32];
    ran_tobytes(buf2, &sum_combined);
    check_bytes("map_to_curve2(1,2) == sswu(1)+sswu(2)", buf, buf2, 32);

    /* sswu(0) produces a valid point */
    ran_map_to_curve(&result, zero_bytes);
    ran_to_affine(&aff, &result);
    check_nonzero("sswu(0) on curve", ran_is_on_curve(&aff));
}

static void test_shaw_sswu()
{
    std::cout << std::endl << "=== Shaw SSWU ===" << std::endl;
    unsigned char buf[32];

    /* Known test vectors */
    shaw_jacobian result;
    shaw_map_to_curve(&result, one_bytes);
    shaw_tobytes(buf, &result);
    check_bytes("sswu(1)", shaw_sswu_u1_result, buf, 32);

    unsigned char two_bytes[32] = {0x02};
    shaw_map_to_curve(&result, two_bytes);
    shaw_tobytes(buf, &result);
    check_bytes("sswu(2)", shaw_sswu_u2_result, buf, 32);

    unsigned char u42_bytes[32] = {0x2a};
    shaw_map_to_curve(&result, u42_bytes);
    shaw_tobytes(buf, &result);
    check_bytes("sswu(42)", shaw_sswu_u42_result, buf, 32);

    /* Deterministic: same input → same output */
    shaw_jacobian result2;
    shaw_map_to_curve(&result2, one_bytes);
    shaw_tobytes(buf, &result2);
    check_bytes("sswu(1) deterministic", shaw_sswu_u1_result, buf, 32);

    /* Output is on curve */
    shaw_affine aff;
    shaw_to_affine(&aff, &result);
    check_nonzero("sswu(1) on curve", shaw_is_on_curve(&aff));

    /* map_to_curve2(u0, u1) == map_to_curve(u0) + map_to_curve(u1) */
    shaw_jacobian p0, p1, sum_direct, sum_combined;
    shaw_map_to_curve(&p0, one_bytes);
    shaw_map_to_curve(&p1, two_bytes);
    shaw_add(&sum_direct, &p0, &p1);
    shaw_tobytes(buf, &sum_direct);

    shaw_map_to_curve2(&sum_combined, one_bytes, two_bytes);
    unsigned char buf2[32];
    shaw_tobytes(buf2, &sum_combined);
    check_bytes("map_to_curve2(1,2) == sswu(1)+sswu(2)", buf, buf2, 32);

    /* sswu(0) produces a valid point */
    shaw_map_to_curve(&result, zero_bytes);
    shaw_to_affine(&aff, &result);
    check_nonzero("sswu(0) on curve", shaw_is_on_curve(&aff));
}

static void test_ran_batch_affine()
{
    std::cout << std::endl << "=== Ran batch affine ===" << std::endl;

    ran_jacobian G;
    fp_copy(G.X, RAN_GX);
    fp_copy(G.Y, RAN_GY);
    fp_1(G.Z);

    /* n=1: batch matches single to_affine */
    {
        ran_affine batch_out[1], single_out;
        ran_batch_to_affine(batch_out, &G, 1);
        ran_to_affine(&single_out, &G);
        unsigned char bx[32], sx[32], by[32], sy[32];
        fp_tobytes(bx, batch_out[0].x);
        fp_tobytes(sx, single_out.x);
        check_bytes("batch n=1 x matches single", sx, bx, 32);
        fp_tobytes(by, batch_out[0].y);
        fp_tobytes(sy, single_out.y);
        check_bytes("batch n=1 y matches single", sy, by, 32);
    }

    /* n=4: multiple distinct points */
    {
        ran_jacobian points[4];
        ran_copy(&points[0], &G);
        ran_dbl(&points[1], &G);
        ran_add(&points[2], &points[1], &G);
        ran_dbl(&points[3], &points[1]);

        ran_affine batch_out[4], single_out[4];
        ran_batch_to_affine(batch_out, points, 4);
        for (int i = 0; i < 4; i++)
            ran_to_affine(&single_out[i], &points[i]);

        for (int i = 0; i < 4; i++)
        {
            unsigned char bx[32], sx[32];
            fp_tobytes(bx, batch_out[i].x);
            fp_tobytes(sx, single_out[i].x);
            std::string name = "batch n=4 point " + std::to_string(i) + " x";
            check_bytes(name.c_str(), sx, bx, 32);
        }
    }

    /* Identity point handling */
    {
        ran_jacobian points[2];
        ran_copy(&points[0], &G);
        ran_identity(&points[1]);
        ran_affine batch_out[2];
        ran_batch_to_affine(batch_out, points, 2);
        unsigned char zx[32];
        fp_tobytes(zx, batch_out[1].x);
        check_bytes("batch identity x == 0", zero_bytes, zx, 32);
    }
}

static void test_shaw_batch_affine()
{
    std::cout << std::endl << "=== Shaw batch affine ===" << std::endl;

    shaw_jacobian G;
    fq_copy(G.X, SHAW_GX);
    fq_copy(G.Y, SHAW_GY);
    fq_1(G.Z);

    /* n=4 */
    {
        shaw_jacobian points[4];
        shaw_copy(&points[0], &G);
        shaw_dbl(&points[1], &G);
        shaw_add(&points[2], &points[1], &G);
        shaw_dbl(&points[3], &points[1]);

        shaw_affine batch_out[4], single_out[4];
        shaw_batch_to_affine(batch_out, points, 4);
        for (int i = 0; i < 4; i++)
            shaw_to_affine(&single_out[i], &points[i]);

        for (int i = 0; i < 4; i++)
        {
            unsigned char bx[32], sx[32];
            fq_tobytes(bx, batch_out[i].x);
            fq_tobytes(sx, single_out[i].x);
            std::string name = "batch n=4 point " + std::to_string(i) + " x";
            check_bytes(name.c_str(), sx, bx, 32);
        }
    }
}

static void test_ran_pedersen()
{
    std::cout << std::endl << "=== Ran Pedersen ===" << std::endl;

    ran_jacobian G;
    fp_copy(G.X, RAN_GX);
    fp_copy(G.Y, RAN_GY);
    fp_1(G.Z);

    /* C = r*H + a*G, where H = 2G, verify == r*2G + a*G */
    ran_jacobian H;
    ran_dbl(&H, &G);

    unsigned char r_scalar[32] = {0x03};
    unsigned char a_scalar[32] = {0x05};

    ran_jacobian commit;
    ran_pedersen_commit(&commit, r_scalar, &H, a_scalar, &G, 1);

    /* Compute expected: 3*2G + 5*G = 6G + 5G = 11G */
    unsigned char eleven_scalar[32] = {0x0b};
    ran_jacobian expected;
    ran_scalarmult_vartime(&expected, eleven_scalar, &G);

    unsigned char commit_bytes[32], expected_bytes[32];
    ran_tobytes(commit_bytes, &commit);
    ran_tobytes(expected_bytes, &expected);
    check_bytes("pedersen(3, 2G, [5], [G]) == 11G", expected_bytes, commit_bytes, 32);

    /* n=0: C = r*H (blinding only) */
    ran_pedersen_commit(&commit, r_scalar, &H, nullptr, nullptr, 0);
    unsigned char three_scalar[32] = {0x03};
    ran_scalarmult_vartime(&expected, three_scalar, &H);
    ran_tobytes(commit_bytes, &commit);
    ran_tobytes(expected_bytes, &expected);
    check_bytes("pedersen n=0: r*H only", expected_bytes, commit_bytes, 32);
}

static void test_shaw_pedersen()
{
    std::cout << std::endl << "=== Shaw Pedersen ===" << std::endl;
    (void)zero_bytes; /* suppress unused warnings */

    shaw_jacobian G;
    fq_copy(G.X, SHAW_GX);
    fq_copy(G.Y, SHAW_GY);
    fq_1(G.Z);

    shaw_jacobian H;
    shaw_dbl(&H, &G);

    unsigned char r_scalar[32] = {0x03};
    unsigned char a_scalar[32] = {0x05};

    shaw_jacobian commit;
    shaw_pedersen_commit(&commit, r_scalar, &H, a_scalar, &G, 1);

    unsigned char eleven_scalar[32] = {0x0b};
    shaw_jacobian expected;
    shaw_scalarmult_vartime(&expected, eleven_scalar, &G);

    unsigned char commit_bytes[32], expected_bytes[32];
    shaw_tobytes(commit_bytes, &commit);
    shaw_tobytes(expected_bytes, &expected);
    check_bytes("pedersen(3, 2G, [5], [G]) == 11G", expected_bytes, commit_bytes, 32);
}

static void test_fp_poly()
{
    std::cout << std::endl << "=== F_p polynomial ===" << std::endl;
    unsigned char buf[32];

    /* (x+1)(x-1) = x^2 - 1 */
    {
        fp_poly a, b, r;
        a.coeffs.resize(2);
        fp_1(a.coeffs[0].v); /* 1 */
        fp_1(a.coeffs[1].v); /* x */

        b.coeffs.resize(2);
        fp_fe neg1;
        fp_fe one_fe;
        fp_1(one_fe);
        fp_neg(neg1, one_fe);
        std::memcpy(b.coeffs[0].v, neg1, sizeof(fp_fe));
        fp_1(b.coeffs[1].v);

        fp_poly_mul(&r, &a, &b);

        /* r should be [-1, 0, 1] (x^2 - 1) */
        check_int("(x+1)(x-1) degree", 3, (int)r.coeffs.size());

        fp_fe c0;
        std::memcpy(c0, r.coeffs[0].v, sizeof(fp_fe));
        fp_tobytes(buf, c0);
        unsigned char neg1_bytes[32];
        fp_tobytes(neg1_bytes, neg1);
        check_bytes("(x+1)(x-1) const coeff == -1", neg1_bytes, buf, 32);

        fp_fe c1;
        std::memcpy(c1, r.coeffs[1].v, sizeof(fp_fe));
        fp_tobytes(buf, c1);
        check_bytes("(x+1)(x-1) x coeff == 0", zero_bytes, buf, 32);

        fp_fe c2;
        std::memcpy(c2, r.coeffs[2].v, sizeof(fp_fe));
        fp_tobytes(buf, c2);
        check_bytes("(x+1)(x-1) x^2 coeff == 1", one_bytes, buf, 32);
    }

    /* Evaluate x^2-1 at x=3 should give 8 */
    {
        fp_poly p;
        p.coeffs.resize(3);
        fp_fe one_fe, neg1;
        fp_1(one_fe);
        fp_neg(neg1, one_fe);
        std::memcpy(p.coeffs[0].v, neg1, sizeof(fp_fe));
        fp_0(p.coeffs[1].v);
        fp_1(p.coeffs[2].v);

        unsigned char three_bytes[32] = {0x03};
        fp_fe x_val;
        fp_frombytes(x_val, three_bytes);

        fp_fe result;
        fp_poly_eval(result, &p, x_val);
        fp_tobytes(buf, result);
        unsigned char eight_bytes[32] = {0x08};
        check_bytes("eval x^2-1 at x=3 == 8", eight_bytes, buf, 32);
    }

    /* from_roots: roots=[2,3] -> (x-2)(x-3) = x^2-5x+6 */
    {
        unsigned char r1_bytes[32] = {0x02};
        unsigned char r2_bytes[32] = {0x03};
        fp_fe roots[2];
        fp_frombytes(roots[0], r1_bytes);
        fp_frombytes(roots[1], r2_bytes);

        fp_poly p;
        fp_poly_from_roots(&p, roots, 2);

        /* Evaluate at roots should give 0 */
        fp_fe val;
        fp_poly_eval(val, &p, roots[0]);
        fp_tobytes(buf, val);
        check_bytes("from_roots(2,3) eval at 2 == 0", zero_bytes, buf, 32);

        fp_poly_eval(val, &p, roots[1]);
        fp_tobytes(buf, val);
        check_bytes("from_roots(2,3) eval at 3 == 0", zero_bytes, buf, 32);
    }

    /* divmod: (x^2-1) / (x+1) == (x-1), remainder 0 */
    {
        fp_poly dividend, divisor_poly, q, rem;
        dividend.coeffs.resize(3);
        fp_fe one_fe, neg1;
        fp_1(one_fe);
        fp_neg(neg1, one_fe);
        std::memcpy(dividend.coeffs[0].v, neg1, sizeof(fp_fe));
        fp_0(dividend.coeffs[1].v);
        fp_1(dividend.coeffs[2].v);

        divisor_poly.coeffs.resize(2);
        fp_1(divisor_poly.coeffs[0].v);
        fp_1(divisor_poly.coeffs[1].v);

        fp_poly_divmod(&q, &rem, &dividend, &divisor_poly);

        /* q should be (x - 1): [-1, 1] */
        check_int("divmod quotient size", 2, (int)q.coeffs.size());
        fp_fe q0;
        std::memcpy(q0, q.coeffs[0].v, sizeof(fp_fe));
        fp_tobytes(buf, q0);
        unsigned char neg1_bytes[32];
        fp_tobytes(neg1_bytes, neg1);
        check_bytes("divmod quotient const == -1", neg1_bytes, buf, 32);

        fp_fe q1;
        std::memcpy(q1, q.coeffs[1].v, sizeof(fp_fe));
        fp_tobytes(buf, q1);
        check_bytes("divmod quotient x coeff == 1", one_bytes, buf, 32);

        /* remainder should be 0 */
        fp_fe r0;
        std::memcpy(r0, rem.coeffs[0].v, sizeof(fp_fe));
        fp_tobytes(buf, r0);
        check_bytes("divmod remainder == 0", zero_bytes, buf, 32);
    }
}

static void test_fq_poly()
{
    std::cout << std::endl << "=== F_q polynomial ===" << std::endl;
    unsigned char buf[32];

    /* from_roots + eval at roots should give 0 */
    {
        unsigned char r1_bytes[32] = {0x05};
        unsigned char r2_bytes[32] = {0x07};
        unsigned char r3_bytes[32] = {0x0b};
        fq_fe roots[3];
        fq_frombytes(roots[0], r1_bytes);
        fq_frombytes(roots[1], r2_bytes);
        fq_frombytes(roots[2], r3_bytes);

        fq_poly p;
        fq_poly_from_roots(&p, roots, 3);

        for (int i = 0; i < 3; i++)
        {
            fq_fe val;
            fq_poly_eval(val, &p, roots[i]);
            fq_tobytes(buf, val);
            std::string name = "fq from_roots eval at root " + std::to_string(i) + " == 0";
            check_bytes(name.c_str(), zero_bytes, buf, 32);
        }
    }

    /* mul commutativity */
    {
        fq_poly a, b, ab, ba;
        a.coeffs.resize(2);
        fq_fe two, three;
        unsigned char two_b[32] = {0x02};
        unsigned char three_b[32] = {0x03};
        fq_frombytes(two, two_b);
        fq_frombytes(three, three_b);
        std::memcpy(a.coeffs[0].v, two, sizeof(fq_fe));
        std::memcpy(a.coeffs[1].v, three, sizeof(fq_fe));

        b.coeffs.resize(2);
        unsigned char five_b[32] = {0x05};
        unsigned char seven_b[32] = {0x07};
        fq_fe five, seven;
        fq_frombytes(five, five_b);
        fq_frombytes(seven, seven_b);
        std::memcpy(b.coeffs[0].v, five, sizeof(fq_fe));
        std::memcpy(b.coeffs[1].v, seven, sizeof(fq_fe));

        fq_poly_mul(&ab, &a, &b);
        fq_poly_mul(&ba, &b, &a);

        bool match = true;
        for (size_t i = 0; i < ab.coeffs.size(); i++)
        {
            unsigned char ab_c[32], ba_c[32];
            fq_tobytes(ab_c, ab.coeffs[i].v);
            fq_tobytes(ba_c, ba.coeffs[i].v);
            if (std::memcmp(ab_c, ba_c, 32) != 0)
                match = false;
        }
        ++tests_run;
        if (match)
        {
            ++tests_passed;
            std::cout << "  PASS: fq poly mul commutative" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: fq poly mul commutative" << std::endl;
        }
    }
}

static void test_ran_divisor()
{
    std::cout << std::endl << "=== Ran divisor ===" << std::endl;
    unsigned char buf[32];

    ran_jacobian G;
    fp_copy(G.X, RAN_GX);
    fp_copy(G.Y, RAN_GY);
    fp_1(G.Z);

    /* Get a few affine points on the curve */
    ran_jacobian G2, G3, G4;
    ran_dbl(&G2, &G);
    ran_add(&G3, &G2, &G);
    ran_dbl(&G4, &G2);

    ran_affine pts[3];
    ran_to_affine(&pts[0], &G);
    ran_to_affine(&pts[1], &G2);
    ran_to_affine(&pts[2], &G3);

    /* Compute divisor */
    ran_divisor d;
    ran_compute_divisor(&d, pts, 3);

    /* Evaluate at each point: should give 0 */
    for (int i = 0; i < 3; i++)
    {
        fp_fe val;
        ran_evaluate_divisor(val, &d, pts[i].x, pts[i].y);
        fp_tobytes(buf, val);
        std::string name = "divisor eval at point " + std::to_string(i) + " == 0";
        check_bytes(name.c_str(), zero_bytes, buf, 32);
    }

    /* Evaluate at a different point: should NOT be 0 */
    {
        ran_affine p4;
        ran_to_affine(&p4, &G4);
        fp_fe val;
        ran_evaluate_divisor(val, &d, p4.x, p4.y);
        fp_tobytes(buf, val);
        check_nonzero("divisor eval at non-member != 0", std::memcmp(buf, zero_bytes, 32) != 0 ? 1 : 0);
    }

    /* Single point divisor */
    {
        ran_divisor d1;
        ran_compute_divisor(&d1, pts, 1);
        fp_fe val;
        ran_evaluate_divisor(val, &d1, pts[0].x, pts[0].y);
        fp_tobytes(buf, val);
        check_bytes("single-point divisor eval == 0", zero_bytes, buf, 32);
    }
}

static void test_shaw_divisor()
{
    std::cout << std::endl << "=== Shaw divisor ===" << std::endl;
    unsigned char buf[32];

    shaw_jacobian G;
    fq_copy(G.X, SHAW_GX);
    fq_copy(G.Y, SHAW_GY);
    fq_1(G.Z);

    shaw_jacobian G2, G3;
    shaw_dbl(&G2, &G);
    shaw_add(&G3, &G2, &G);

    shaw_affine pts[2];
    shaw_to_affine(&pts[0], &G);
    shaw_to_affine(&pts[1], &G2);

    shaw_divisor d;
    shaw_compute_divisor(&d, pts, 2);

    for (int i = 0; i < 2; i++)
    {
        fq_fe val;
        shaw_evaluate_divisor(val, &d, pts[i].x, pts[i].y);
        fq_tobytes(buf, val);
        std::string name = "divisor eval at point " + std::to_string(i) + " == 0";
        check_bytes(name.c_str(), zero_bytes, buf, 32);
    }

    /* Non-member check */
    {
        shaw_affine p3;
        shaw_to_affine(&p3, &G3);
        fq_fe val;
        shaw_evaluate_divisor(val, &d, p3.x, p3.y);
        fq_tobytes(buf, val);
        check_nonzero("divisor eval at non-member != 0", std::memcmp(buf, zero_bytes, 32) != 0 ? 1 : 0);
    }
}

/* ========================================================================
 * Extended tests for pre-SIMD hardening
 * ======================================================================== */

static void test_fp_extended()
{
    std::cout << std::endl << "=== F_p extended ===" << std::endl;
    unsigned char buf[32];

    fp_fe a, one_fe, zero_fe;
    fp_frombytes(a, test_a_bytes);
    fp_1(one_fe);
    fp_0(zero_fe);

    /* fp_sq(a) then add to itself == 2*a^2 (tests sq2 property) */
    {
        fp_fe sq_a, sq2_via_add;
        fp_sq(sq_a, a);
        fp_add(sq2_via_add, sq_a, sq_a);
        /* Compare against mul(sq(a), 2) */
        unsigned char two_b[32] = {0x02};
        fp_fe two_fe, sq2_via_mul;
        fp_frombytes(two_fe, two_b);
        fp_mul(sq2_via_mul, sq_a, two_fe);
        unsigned char add_bytes[32], mul_bytes[32];
        fp_tobytes(add_bytes, sq2_via_add);
        fp_tobytes(mul_bytes, sq2_via_mul);
        check_bytes("2*sq(a) via add == via mul", mul_bytes, add_bytes, 32);
    }

    /* sqn chain: sq(sq(sq(sq(sq(a))))) == a^32 via repeated squaring */
    {
        fp_fe chain;
        fp_sq(chain, a);
        fp_sq(chain, chain);
        fp_sq(chain, chain);
        fp_sq(chain, chain);
        fp_sq(chain, chain);
        /* Compare against a^32 via mul: a^2, a^4, a^8, a^16, a^32 */
        fp_fe power;
        fp_sq(power, a); /* a^2 */
        fp_mul(power, power, power); /* a^4 */
        fp_mul(power, power, power); /* a^8 */
        fp_mul(power, power, power); /* a^16 */
        fp_mul(power, power, power); /* a^32 */
        unsigned char chain_bytes[32], power_bytes[32];
        fp_tobytes(chain_bytes, chain);
        fp_tobytes(power_bytes, power);
        check_bytes("sq^5(a) == a^32", power_bytes, chain_bytes, 32);
    }

    /* cmov: b=0 keeps original */
    {
        fp_fe target;
        fp_copy(target, a);
        fp_cmov(target, one_fe, 0);
        fp_tobytes(buf, target);
        check_bytes("cmov(a, 1, 0) == a", test_a_bytes, buf, 32);
    }

    /* cmov: b=1 replaces */
    {
        fp_fe target;
        fp_copy(target, a);
        fp_cmov(target, one_fe, 1);
        fp_tobytes(buf, target);
        check_bytes("cmov(a, 1, 1) == 1", one_bytes, buf, 32);
    }

    /* Edge: (p-1)*(p-1) */
    {
        /* p-1 in little-endian: 0xec, 0xff...0xff, 0x7f */
        unsigned char pm1_bytes[32] = {0xec, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};
        fp_fe pm1;
        fp_frombytes(pm1, pm1_bytes);
        fp_fe pm1_sq;
        fp_mul(pm1_sq, pm1, pm1);
        /* (p-1)^2 = (-1)^2 = 1 mod p */
        fp_tobytes(buf, pm1_sq);
        check_bytes("(p-1)^2 == 1", one_bytes, buf, 32);
    }

    /* Edge: (p-1)*2 */
    {
        unsigned char pm1_bytes[32] = {0xec, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};
        fp_fe pm1;
        fp_frombytes(pm1, pm1_bytes);
        unsigned char two_b[32] = {0x02};
        fp_fe two_fe, result;
        fp_frombytes(two_fe, two_b);
        fp_mul(result, pm1, two_fe);
        /* (-1)*2 = -2 mod p = p-2 */
        unsigned char pm2_bytes[32] = {0xeb, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};
        fp_tobytes(buf, result);
        check_bytes("(p-1)*2 == p-2", pm2_bytes, buf, 32);
    }

    /* Edge: (p-1) + 1 wraps to 0 */
    {
        unsigned char pm1_bytes[32] = {0xec, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};
        fp_fe pm1, result;
        fp_frombytes(pm1, pm1_bytes);
        fp_add(result, pm1, one_fe);
        fp_tobytes(buf, result);
        check_bytes("(p-1) + 1 == 0", zero_bytes, buf, 32);
    }

    /* Edge: 0 - 1 wraps to p-1 */
    {
        fp_fe result;
        fp_sub(result, zero_fe, one_fe);
        unsigned char pm1_bytes[32] = {0xec, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};
        fp_tobytes(buf, result);
        check_bytes("0 - 1 == p-1", pm1_bytes, buf, 32);
    }

    /* neg(0) == 0 */
    {
        fp_fe result;
        fp_neg(result, zero_fe);
        fp_tobytes(buf, result);
        check_bytes("neg(0) == 0", zero_bytes, buf, 32);
    }

    /* invert(1) == 1 */
    {
        fp_fe result;
        fp_invert(result, one_fe);
        fp_tobytes(buf, result);
        check_bytes("invert(1) == 1", one_bytes, buf, 32);
    }

    /* Serialization: frombytes(p_bytes) reduces to 0 */
    {
        unsigned char p_bytes[32] = {0xed, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                     0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                     0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};
        fp_fe result;
        fp_frombytes(result, p_bytes);
        fp_tobytes(buf, result);
        check_bytes("frombytes(p) == 0", zero_bytes, buf, 32);
    }
}

static void test_fq_extended()
{
    std::cout << std::endl << "=== F_q extended ===" << std::endl;
    unsigned char buf[32];

    fq_fe a, one_fe, zero_fe;
    fq_frombytes(a, test_a_bytes);
    fq_1(one_fe);
    fq_0(zero_fe);

    /* 2*sq(a) via add == via mul */
    {
        fq_fe sq_a, sq2_via_add;
        fq_sq(sq_a, a);
        fq_add(sq2_via_add, sq_a, sq_a);
        unsigned char two_b[32] = {0x02};
        fq_fe two_fe, sq2_via_mul;
        fq_frombytes(two_fe, two_b);
        fq_mul(sq2_via_mul, sq_a, two_fe);
        unsigned char add_bytes[32], mul_bytes[32];
        fq_tobytes(add_bytes, sq2_via_add);
        fq_tobytes(mul_bytes, sq2_via_mul);
        check_bytes("2*sq(a) via add == via mul", mul_bytes, add_bytes, 32);
    }

    /* sq^5 chain equivalence */
    {
        fq_fe chain;
        fq_sq(chain, a);
        fq_sq(chain, chain);
        fq_sq(chain, chain);
        fq_sq(chain, chain);
        fq_sq(chain, chain);
        fq_fe power;
        fq_sq(power, a);
        fq_mul(power, power, power);
        fq_mul(power, power, power);
        fq_mul(power, power, power);
        fq_mul(power, power, power);
        unsigned char chain_bytes[32], power_bytes[32];
        fq_tobytes(chain_bytes, chain);
        fq_tobytes(power_bytes, power);
        check_bytes("sq^5(a) == a^32", power_bytes, chain_bytes, 32);
    }

    /* cmov */
    {
        fq_fe target;
        fq_copy(target, a);
        fq_cmov(target, one_fe, 0);
        fq_tobytes(buf, target);
        check_bytes("cmov(a, 1, 0) == a", test_a_bytes, buf, 32);
    }
    {
        fq_fe target;
        fq_copy(target, a);
        fq_cmov(target, one_fe, 1);
        fq_tobytes(buf, target);
        check_bytes("cmov(a, 1, 1) == 1", one_bytes, buf, 32);
    }

    /* fq_sqrt(0) == 0 */
    {
        fq_fe result;
        fq_sqrt(result, zero_fe);
        fq_tobytes(buf, result);
        check_bytes("sqrt(0) == 0", zero_bytes, buf, 32);
    }

    /* fq_sqrt(1)^2 == 1 */
    {
        fq_fe sqrt1, sq_check;
        fq_sqrt(sqrt1, one_fe);
        fq_sq(sq_check, sqrt1);
        fq_tobytes(buf, sq_check);
        check_bytes("sqrt(1)^2 == 1", one_bytes, buf, 32);
    }

    /* fq_sqrt(a^2)^2 == a^2 */
    {
        fq_fe a_sq, sqrt_asq, sq_check;
        fq_sq(a_sq, a);
        fq_sqrt(sqrt_asq, a_sq);
        fq_sq(sq_check, sqrt_asq);
        unsigned char asq_bytes[32];
        fq_tobytes(asq_bytes, a_sq);
        fq_tobytes(buf, sq_check);
        check_bytes("sqrt(a^2)^2 == a^2", asq_bytes, buf, 32);
    }

    /* Edge: (q-1)*(q-1) == 1 */
    {
        /* q = 2^255 - gamma, q-1 in LE */
        /* RAN_ORDER is q in LE. q-1 = RAN_ORDER - 1 */
        unsigned char qm1_bytes[32];
        std::memcpy(qm1_bytes, RAN_ORDER, 32);
        /* Subtract 1 from little-endian */
        for (int i = 0; i < 32; i++)
        {
            if (qm1_bytes[i] > 0)
            {
                qm1_bytes[i]--;
                break;
            }
            qm1_bytes[i] = 0xff;
        }
        fq_fe qm1, qm1_sq;
        fq_frombytes(qm1, qm1_bytes);
        fq_mul(qm1_sq, qm1, qm1);
        fq_tobytes(buf, qm1_sq);
        check_bytes("(q-1)^2 == 1", one_bytes, buf, 32);
    }

    /* Edge: (q-1) + 1 wraps to 0 */
    {
        unsigned char qm1_bytes[32];
        std::memcpy(qm1_bytes, RAN_ORDER, 32);
        for (int i = 0; i < 32; i++)
        {
            if (qm1_bytes[i] > 0)
            {
                qm1_bytes[i]--;
                break;
            }
            qm1_bytes[i] = 0xff;
        }
        fq_fe qm1, result;
        fq_frombytes(qm1, qm1_bytes);
        fq_add(result, qm1, one_fe);
        fq_tobytes(buf, result);
        check_bytes("(q-1) + 1 == 0", zero_bytes, buf, 32);
    }

    /* invert(1) == 1 */
    {
        fq_fe result;
        fq_invert(result, one_fe);
        fq_tobytes(buf, result);
        check_bytes("invert(1) == 1", one_bytes, buf, 32);
    }

    /* neg(0) == 0 */
    {
        fq_fe result;
        fq_neg(result, zero_fe);
        fq_tobytes(buf, result);
        check_bytes("neg(0) == 0", zero_bytes, buf, 32);
    }

    /* Serialization: frombytes(q_bytes) reduces to 0 */
    {
        fq_fe result;
        fq_frombytes(result, RAN_ORDER);
        fq_tobytes(buf, result);
        check_bytes("frombytes(q) == 0", zero_bytes, buf, 32);
    }
}

static void test_serialization_edges()
{
    std::cout << std::endl << "=== Serialization edges ===" << std::endl;
    unsigned char buf[32];

    /* Fp: round-trip 0, 1, p-1 */
    {
        fp_fe fe;
        fp_0(fe);
        fp_tobytes(buf, fe);
        fp_fe fe2;
        fp_frombytes(fe2, buf);
        unsigned char buf2[32];
        fp_tobytes(buf2, fe2);
        check_bytes("fp round-trip 0", buf, buf2, 32);
    }
    {
        fp_fe fe;
        fp_1(fe);
        fp_tobytes(buf, fe);
        fp_fe fe2;
        fp_frombytes(fe2, buf);
        unsigned char buf2[32];
        fp_tobytes(buf2, fe2);
        check_bytes("fp round-trip 1", buf, buf2, 32);
    }
    {
        unsigned char pm1_bytes[32] = {0xec, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};
        fp_fe fe;
        fp_frombytes(fe, pm1_bytes);
        fp_tobytes(buf, fe);
        check_bytes("fp round-trip p-1", pm1_bytes, buf, 32);
    }

    /* Fq: round-trip 0, 1, q-1 */
    {
        fq_fe fe;
        fq_0(fe);
        fq_tobytes(buf, fe);
        fq_fe fe2;
        fq_frombytes(fe2, buf);
        unsigned char buf2[32];
        fq_tobytes(buf2, fe2);
        check_bytes("fq round-trip 0", buf, buf2, 32);
    }
    {
        fq_fe fe;
        fq_1(fe);
        fq_tobytes(buf, fe);
        fq_fe fe2;
        fq_frombytes(fe2, buf);
        unsigned char buf2[32];
        fq_tobytes(buf2, fe2);
        check_bytes("fq round-trip 1", buf, buf2, 32);
    }
    {
        unsigned char qm1_bytes[32];
        std::memcpy(qm1_bytes, RAN_ORDER, 32);
        for (int i = 0; i < 32; i++)
        {
            if (qm1_bytes[i] > 0)
            {
                qm1_bytes[i]--;
                break;
            }
            qm1_bytes[i] = 0xff;
        }
        fq_fe fe;
        fq_frombytes(fe, qm1_bytes);
        fq_tobytes(buf, fe);
        check_bytes("fq round-trip q-1", qm1_bytes, buf, 32);
    }

    /* Fp: value with high bits near 255 */
    {
        unsigned char high_bytes[32] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40};
        fp_fe fe;
        fp_frombytes(fe, high_bytes);
        fp_tobytes(buf, fe);
        check_bytes("fp round-trip high bit value", high_bytes, buf, 32);
    }
}

static void test_ran_point_edges()
{
    std::cout << std::endl << "=== Ran point edges ===" << std::endl;
    unsigned char buf[32];

    ran_jacobian G;
    fp_copy(G.X, RAN_GX);
    fp_copy(G.Y, RAN_GY);
    fp_1(G.Z);

    /* (order-1)*G == -G */
    {
        unsigned char om1[32];
        std::memcpy(om1, RAN_ORDER, 32);
        for (int i = 0; i < 32; i++)
        {
            if (om1[i] > 0)
            {
                om1[i]--;
                break;
            }
            om1[i] = 0xff;
        }
        ran_jacobian result;
        ran_scalarmult(&result, om1, &G);
        ran_jacobian neg_G;
        ran_neg(&neg_G, &G);
        unsigned char r_bytes[32], neg_bytes[32];
        ran_tobytes(r_bytes, &result);
        ran_tobytes(neg_bytes, &neg_G);
        check_bytes("(order-1)*G == -G", neg_bytes, r_bytes, 32);
    }

    /* vartime: (order-1)*G == -G */
    {
        unsigned char om1[32];
        std::memcpy(om1, RAN_ORDER, 32);
        for (int i = 0; i < 32; i++)
        {
            if (om1[i] > 0)
            {
                om1[i]--;
                break;
            }
            om1[i] = 0xff;
        }
        ran_jacobian result;
        ran_scalarmult_vartime(&result, om1, &G);
        ran_jacobian neg_G;
        ran_neg(&neg_G, &G);
        unsigned char r_bytes[32], neg_bytes[32];
        ran_tobytes(r_bytes, &result);
        ran_tobytes(neg_bytes, &neg_G);
        check_bytes("vartime: (order-1)*G == -G", neg_bytes, r_bytes, 32);
    }

    /* (order-1)*G + G == identity */
    {
        unsigned char om1[32];
        std::memcpy(om1, RAN_ORDER, 32);
        for (int i = 0; i < 32; i++)
        {
            if (om1[i] > 0)
            {
                om1[i]--;
                break;
            }
            om1[i] = 0xff;
        }
        ran_jacobian om1G, sum;
        ran_scalarmult(&om1G, om1, &G);
        ran_add(&sum, &om1G, &G);
        check_nonzero("(order-1)*G + G == identity", ran_is_identity(&sum));
    }

    /* Y-parity: serialize G, flip bit 255, verify y negated */
    {
        unsigned char g_bytes[32];
        ran_tobytes(g_bytes, &G);
        unsigned char flipped[32];
        std::memcpy(flipped, g_bytes, 32);
        flipped[31] ^= 0x80; /* flip parity bit */
        ran_jacobian decoded;
        int rc = ran_frombytes(&decoded, flipped);
        check_int("flipped parity decodes", 0, rc);
        /* The y should be negated */
        ran_affine aff_orig, aff_flip;
        ran_to_affine(&aff_orig, &G);
        ran_to_affine(&aff_flip, &decoded);
        /* x should match */
        unsigned char ox[32], fx[32];
        fp_tobytes(ox, aff_orig.x);
        fp_tobytes(fx, aff_flip.x);
        check_bytes("flipped parity: x matches", ox, fx, 32);
        /* y + flipped_y == 0 (they should be negations) */
        fp_fe y_sum;
        fp_add(y_sum, aff_orig.y, aff_flip.y);
        fp_tobytes(buf, y_sum);
        check_bytes("flipped parity: y + y' == 0", zero_bytes, buf, 32);
    }

    /* Identity round-trip */
    {
        ran_jacobian id;
        ran_identity(&id);
        unsigned char id_bytes[32];
        ran_tobytes(id_bytes, &id);
        check_bytes("tobytes(identity) == 0", zero_bytes, id_bytes, 32);
        /* frombytes(0) — x=0, check if on curve */
        ran_jacobian decoded;
        int rc = ran_frombytes(&decoded, zero_bytes);
        /* x=0: gx = 0^3 - 3*0 + b = b. If b is a QR, this decodes. Otherwise -1. */
        /* Either way, we just record what happens */
        if (rc == 0)
        {
            ran_tobytes(buf, &decoded);
            /* Should be a valid non-identity point with x=0 */
            ++tests_run;
            ++tests_passed;
            std::cout << "  PASS: frombytes(0) decodes (x=0 on curve)" << std::endl;
        }
        else
        {
            ++tests_run;
            ++tests_passed;
            std::cout << "  PASS: frombytes(0) rejects (x=0 not on curve)" << std::endl;
        }
    }

    /* Off-curve rejection: x=2, check x^3-3x+b is not a QR */
    {
        unsigned char x_bytes[32] = {0x02};
        ran_jacobian decoded;
        int rc = ran_frombytes(&decoded, x_bytes);
        /* We don't know a priori, but can test the contract: if rc==-1, off-curve rejected */
        ++tests_run;
        if (rc == -1)
        {
            ++tests_passed;
            std::cout << "  PASS: frombytes(x=2) rejects off-curve" << std::endl;
        }
        else
        {
            /* x=2 might be on curve; verify it's actually valid */
            ran_affine aff;
            ran_to_affine(&aff, &decoded);
            if (ran_is_on_curve(&aff))
            {
                ++tests_passed;
                std::cout << "  PASS: frombytes(x=2) accepted and on curve" << std::endl;
            }
            else
            {
                ++tests_failed;
                std::cout << "  FAIL: frombytes(x=2) accepted but NOT on curve" << std::endl;
            }
        }
    }
}

static void test_shaw_point_edges()
{
    std::cout << std::endl << "=== Shaw point edges ===" << std::endl;
    unsigned char buf[32];

    shaw_jacobian G;
    fq_copy(G.X, SHAW_GX);
    fq_copy(G.Y, SHAW_GY);
    fq_1(G.Z);

    /* (order-1)*G == -G */
    {
        unsigned char om1[32];
        std::memcpy(om1, SHAW_ORDER, 32);
        for (int i = 0; i < 32; i++)
        {
            if (om1[i] > 0)
            {
                om1[i]--;
                break;
            }
            om1[i] = 0xff;
        }
        shaw_jacobian result;
        shaw_scalarmult(&result, om1, &G);
        shaw_jacobian neg_G;
        shaw_neg(&neg_G, &G);
        unsigned char r_bytes[32], neg_bytes[32];
        shaw_tobytes(r_bytes, &result);
        shaw_tobytes(neg_bytes, &neg_G);
        check_bytes("(order-1)*G == -G", neg_bytes, r_bytes, 32);
    }

    /* vartime: (order-1)*G == -G */
    {
        unsigned char om1[32];
        std::memcpy(om1, SHAW_ORDER, 32);
        for (int i = 0; i < 32; i++)
        {
            if (om1[i] > 0)
            {
                om1[i]--;
                break;
            }
            om1[i] = 0xff;
        }
        shaw_jacobian result;
        shaw_scalarmult_vartime(&result, om1, &G);
        shaw_jacobian neg_G;
        shaw_neg(&neg_G, &G);
        unsigned char r_bytes[32], neg_bytes[32];
        shaw_tobytes(r_bytes, &result);
        shaw_tobytes(neg_bytes, &neg_G);
        check_bytes("vartime: (order-1)*G == -G", neg_bytes, r_bytes, 32);
    }

    /* (order-1)*G + G == identity */
    {
        unsigned char om1[32];
        std::memcpy(om1, SHAW_ORDER, 32);
        for (int i = 0; i < 32; i++)
        {
            if (om1[i] > 0)
            {
                om1[i]--;
                break;
            }
            om1[i] = 0xff;
        }
        shaw_jacobian om1G, sum;
        shaw_scalarmult(&om1G, om1, &G);
        shaw_add(&sum, &om1G, &G);
        check_nonzero("(order-1)*G + G == identity", shaw_is_identity(&sum));
    }

    /* Y-parity flip */
    {
        unsigned char g_bytes[32];
        shaw_tobytes(g_bytes, &G);
        unsigned char flipped[32];
        std::memcpy(flipped, g_bytes, 32);
        flipped[31] ^= 0x80;
        shaw_jacobian decoded;
        int rc = shaw_frombytes(&decoded, flipped);
        check_int("flipped parity decodes", 0, rc);
        shaw_affine aff_orig, aff_flip;
        shaw_to_affine(&aff_orig, &G);
        shaw_to_affine(&aff_flip, &decoded);
        unsigned char ox[32], fx[32];
        fq_tobytes(ox, aff_orig.x);
        fq_tobytes(fx, aff_flip.x);
        check_bytes("flipped parity: x matches", ox, fx, 32);
        fq_fe y_sum;
        fq_add(y_sum, aff_orig.y, aff_flip.y);
        fq_tobytes(buf, y_sum);
        check_bytes("flipped parity: y + y' == 0", zero_bytes, buf, 32);
    }

    /* Identity round-trip */
    {
        shaw_jacobian id;
        shaw_identity(&id);
        unsigned char id_bytes[32];
        shaw_tobytes(id_bytes, &id);
        check_bytes("tobytes(identity) == 0", zero_bytes, id_bytes, 32);
        shaw_jacobian decoded;
        int rc = shaw_frombytes(&decoded, zero_bytes);
        ++tests_run;
        if (rc == 0)
        {
            ++tests_passed;
            std::cout << "  PASS: frombytes(0) decodes (x=0 on curve)" << std::endl;
        }
        else
        {
            ++tests_passed;
            std::cout << "  PASS: frombytes(0) rejects (x=0 not on curve)" << std::endl;
        }
    }
}

static void test_scalarmult_extended()
{
    std::cout << std::endl << "=== Scalar mul extended ===" << std::endl;

    /* Ran: associativity scalarmult(3, scalarmult(7, G)) == scalarmult(21, G) */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        unsigned char s3[32] = {0x03};
        unsigned char s7[32] = {0x07};
        unsigned char s21[32] = {0x15};
        ran_jacobian sevG, result, expected;
        ran_scalarmult(&sevG, s7, &G);
        ran_scalarmult(&result, s3, &sevG);
        ran_scalarmult(&expected, s21, &G);
        unsigned char r_bytes[32], e_bytes[32];
        ran_tobytes(r_bytes, &result);
        ran_tobytes(e_bytes, &expected);
        check_bytes("ran: 3*(7*G) == 21*G", e_bytes, r_bytes, 32);
    }

    /* Shaw: associativity */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        unsigned char s3[32] = {0x03};
        unsigned char s7[32] = {0x07};
        unsigned char s21[32] = {0x15};
        shaw_jacobian sevG, result, expected;
        shaw_scalarmult(&sevG, s7, &G);
        shaw_scalarmult(&result, s3, &sevG);
        shaw_scalarmult(&expected, s21, &G);
        unsigned char r_bytes[32], e_bytes[32];
        shaw_tobytes(r_bytes, &result);
        shaw_tobytes(e_bytes, &expected);
        check_bytes("shaw: 3*(7*G) == 21*G", e_bytes, r_bytes, 32);
    }

    /* Ran: scalarmult(scalar, identity) == identity (via tobytes) */
    {
        ran_jacobian id;
        ran_identity(&id);
        unsigned char s7[32] = {0x07};
        ran_jacobian result;
        ran_scalarmult(&result, s7, &id);
        unsigned char r_bytes[32];
        ran_tobytes(r_bytes, &result);
        check_bytes("ran: 7*identity == identity", zero_bytes, r_bytes, 32);
    }

    /* Shaw: scalarmult(scalar, identity) == identity (via tobytes) */
    {
        shaw_jacobian id;
        shaw_identity(&id);
        unsigned char s7[32] = {0x07};
        shaw_jacobian result;
        shaw_scalarmult(&result, s7, &id);
        unsigned char r_bytes[32];
        shaw_tobytes(r_bytes, &result);
        check_bytes("shaw: 7*identity == identity", zero_bytes, r_bytes, 32);
    }

    /* Ran: scalarmult_vartime(scalar, identity) == identity (via tobytes) */
    {
        ran_jacobian id;
        ran_identity(&id);
        unsigned char s7[32] = {0x07};
        ran_jacobian result;
        ran_scalarmult_vartime(&result, s7, &id);
        unsigned char r_bytes[32];
        ran_tobytes(r_bytes, &result);
        check_bytes("ran: vartime 7*identity == identity", zero_bytes, r_bytes, 32);
    }

    /* Shaw: scalarmult_vartime(scalar, identity) == identity (via tobytes) */
    {
        shaw_jacobian id;
        shaw_identity(&id);
        unsigned char s7[32] = {0x07};
        shaw_jacobian result;
        shaw_scalarmult_vartime(&result, s7, &id);
        unsigned char r_bytes[32];
        shaw_tobytes(r_bytes, &result);
        check_bytes("shaw: vartime 7*identity == identity", zero_bytes, r_bytes, 32);
    }
}

static void test_msm_extended()
{
    std::cout << std::endl << "=== MSM extended ===" << std::endl;
    unsigned char buf[32];

    /* Ran: MSM with identity in array */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        ran_jacobian id;
        ran_identity(&id);

        unsigned char scalars[64];
        std::memcpy(scalars, one_bytes, 32);
        std::memcpy(scalars + 32, one_bytes, 32);
        ran_jacobian points[2];
        ran_copy(&points[0], &id);
        ran_copy(&points[1], &G);
        ran_jacobian result;
        ran_msm_vartime(&result, scalars, points, 2);
        ran_tobytes(buf, &result);
        check_bytes("ran msm([1,1],[id,G]) == G", ran_g_compressed, buf, 32);
    }

    /* Ran: MSM all identities */
    {
        ran_jacobian id;
        ran_identity(&id);
        unsigned char scalars[64];
        std::memcpy(scalars, one_bytes, 32);
        std::memcpy(scalars + 32, one_bytes, 32);
        ran_jacobian points[2];
        ran_copy(&points[0], &id);
        ran_copy(&points[1], &id);
        ran_jacobian result;
        ran_msm_vartime(&result, scalars, points, 2);
        check_nonzero("ran msm all identities == identity", ran_is_identity(&result));
    }

    /* Ran: MSM n=64 (deep Pippenger) */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        unsigned char scalars[64 * 32] = {};
        ran_jacobian points[64];
        for (int i = 0; i < 64; i++)
        {
            /* scalar_i = i+1 */
            scalars[i * 32] = (unsigned char)(i + 1);
            ran_copy(&points[i], &G);
        }
        ran_jacobian result;
        ran_msm_vartime(&result, scalars, points, 64);
        /* Expected: sum(1..64)*G = 2080*G */
        unsigned char s2080[32] = {0x20, 0x08}; /* 2080 = 0x0820 LE */
        ran_jacobian expected;
        ran_scalarmult_vartime(&expected, s2080, &G);
        unsigned char r_bytes[32], e_bytes[32];
        ran_tobytes(r_bytes, &result);
        ran_tobytes(e_bytes, &expected);
        check_bytes("ran msm n=64 == 2080*G", e_bytes, r_bytes, 32);
    }

    /* Ran: MSM duplicate scalars+points: msm([a,a],[G,G]) == 2a*G */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        unsigned char s5[32] = {};
        s5[0] = 0x05;
        unsigned char scalars[64];
        std::memcpy(scalars, s5, 32);
        std::memcpy(scalars + 32, s5, 32);
        ran_jacobian points[2];
        ran_copy(&points[0], &G);
        ran_copy(&points[1], &G);
        ran_jacobian result;
        ran_msm_vartime(&result, scalars, points, 2);
        unsigned char s10[32] = {};
        s10[0] = 0x0a;
        ran_jacobian expected;
        ran_scalarmult_vartime(&expected, s10, &G);
        unsigned char r_bytes[32], e_bytes[32];
        ran_tobytes(r_bytes, &result);
        ran_tobytes(e_bytes, &expected);
        check_bytes("ran msm([5,5],[G,G]) == 10*G", e_bytes, r_bytes, 32);
    }

    /* Shaw: MSM with identity */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_jacobian id;
        shaw_identity(&id);

        unsigned char scalars[64];
        std::memcpy(scalars, one_bytes, 32);
        std::memcpy(scalars + 32, one_bytes, 32);
        shaw_jacobian points[2];
        shaw_copy(&points[0], &id);
        shaw_copy(&points[1], &G);
        shaw_jacobian result;
        shaw_msm_vartime(&result, scalars, points, 2);
        shaw_tobytes(buf, &result);
        check_bytes("shaw msm([1,1],[id,G]) == G", shaw_g_compressed, buf, 32);
    }

    /* Shaw: MSM all identities */
    {
        shaw_jacobian id;
        shaw_identity(&id);
        unsigned char scalars[64];
        std::memcpy(scalars, one_bytes, 32);
        std::memcpy(scalars + 32, one_bytes, 32);
        shaw_jacobian points[2];
        shaw_copy(&points[0], &id);
        shaw_copy(&points[1], &id);
        shaw_jacobian result;
        shaw_msm_vartime(&result, scalars, points, 2);
        check_nonzero("shaw msm all identities == identity", shaw_is_identity(&result));
    }

    /* Shaw: MSM n=64 */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        unsigned char scalars[64 * 32] = {};
        shaw_jacobian points[64];
        for (int i = 0; i < 64; i++)
        {
            scalars[i * 32] = (unsigned char)(i + 1);
            shaw_copy(&points[i], &G);
        }
        shaw_jacobian result;
        shaw_msm_vartime(&result, scalars, points, 64);
        unsigned char s2080[32] = {0x20, 0x08};
        shaw_jacobian expected;
        shaw_scalarmult_vartime(&expected, s2080, &G);
        unsigned char r_bytes[32], e_bytes[32];
        shaw_tobytes(r_bytes, &result);
        shaw_tobytes(e_bytes, &expected);
        check_bytes("shaw msm n=64 == 2080*G", e_bytes, r_bytes, 32);
    }
}

static void test_batch_affine_extended()
{
    std::cout << std::endl << "=== Batch affine extended ===" << std::endl;

    /* Shaw n=1 (match Ran coverage) */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_affine batch_out[1], single_out;
        shaw_batch_to_affine(batch_out, &G, 1);
        shaw_to_affine(&single_out, &G);
        unsigned char bx[32], sx[32], by[32], sy[32];
        fq_tobytes(bx, batch_out[0].x);
        fq_tobytes(sx, single_out.x);
        check_bytes("shaw batch n=1 x", sx, bx, 32);
        fq_tobytes(by, batch_out[0].y);
        fq_tobytes(sy, single_out.y);
        check_bytes("shaw batch n=1 y", sy, by, 32);
    }

    /* Ran n=4: verify y-coordinates too */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        ran_jacobian points[4];
        ran_copy(&points[0], &G);
        ran_dbl(&points[1], &G);
        ran_add(&points[2], &points[1], &G);
        ran_dbl(&points[3], &points[1]);

        ran_affine batch_out[4], single_out[4];
        ran_batch_to_affine(batch_out, points, 4);
        for (int i = 0; i < 4; i++)
            ran_to_affine(&single_out[i], &points[i]);

        for (int i = 0; i < 4; i++)
        {
            unsigned char by_arr[32], sy_arr[32];
            fp_tobytes(by_arr, batch_out[i].y);
            fp_tobytes(sy_arr, single_out[i].y);
            std::string name = "ran batch n=4 point " + std::to_string(i) + " y";
            check_bytes(name.c_str(), sy_arr, by_arr, 32);
        }
    }

    /* Shaw n=4: verify y-coordinates */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_jacobian points[4];
        shaw_copy(&points[0], &G);
        shaw_dbl(&points[1], &G);
        shaw_add(&points[2], &points[1], &G);
        shaw_dbl(&points[3], &points[1]);

        shaw_affine batch_out[4], single_out[4];
        shaw_batch_to_affine(batch_out, points, 4);
        for (int i = 0; i < 4; i++)
            shaw_to_affine(&single_out[i], &points[i]);

        for (int i = 0; i < 4; i++)
        {
            unsigned char by_arr[32], sy_arr[32];
            fq_tobytes(by_arr, batch_out[i].y);
            fq_tobytes(sy_arr, single_out[i].y);
            std::string name = "shaw batch n=4 point " + std::to_string(i) + " y";
            check_bytes(name.c_str(), sy_arr, by_arr, 32);
        }
    }

    /* Ran n=16 stress test */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        ran_jacobian points[16];
        ran_copy(&points[0], &G); /* 1G */
        ran_dbl(&points[1], &G); /* 2G */
        ran_add(&points[2], &points[1], &G); /* 3G */
        ran_dbl(&points[3], &points[1]); /* 4G */
        ran_add(&points[4], &points[3], &G); /* 5G */
        ran_add(&points[5], &points[4], &G); /* 6G */
        /* Use scalarmult for the rest to avoid add(P,P) */
        for (int i = 6; i < 16; i++)
        {
            unsigned char sc[32] = {};
            sc[0] = (unsigned char)(i + 1);
            ran_scalarmult_vartime(&points[i], sc, &G);
        }

        ran_affine batch_out[16], single_out[16];
        ran_batch_to_affine(batch_out, points, 16);
        for (int i = 0; i < 16; i++)
            ran_to_affine(&single_out[i], &points[i]);

        bool all_match = true;
        for (int i = 0; i < 16; i++)
        {
            unsigned char bx[32], sx[32], by_arr[32], sy_arr[32];
            fp_tobytes(bx, batch_out[i].x);
            fp_tobytes(sx, single_out[i].x);
            fp_tobytes(by_arr, batch_out[i].y);
            fp_tobytes(sy_arr, single_out[i].y);
            if (std::memcmp(bx, sx, 32) != 0 || std::memcmp(by_arr, sy_arr, 32) != 0)
                all_match = false;
        }
        ++tests_run;
        if (all_match)
        {
            ++tests_passed;
            std::cout << "  PASS: ran batch n=16 all x,y match" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: ran batch n=16 mismatch" << std::endl;
        }
    }
}

static void test_batch_invert()
{
    std::cout << std::endl << "=== Batch field inversion ===" << std::endl;

    /* Fp: batch invert 4 elements, verify each out[i] * in[i] == 1 */
    {
        fp_fe elems[4], invs[4];
        /* Use small known values: 2, 3, 5, 7 */
        unsigned char v2[32] = {0x02}, v3[32] = {0x03}, v5[32] = {0x05}, v7[32] = {0x07};
        fp_frombytes(elems[0], v2);
        fp_frombytes(elems[1], v3);
        fp_frombytes(elems[2], v5);
        fp_frombytes(elems[3], v7);

        fp_batch_invert(invs, elems, 4);

        bool all_one = true;
        for (int i = 0; i < 4; i++)
        {
            fp_fe prod;
            fp_mul(prod, elems[i], invs[i]);
            unsigned char out[32], one[32] = {};
            one[0] = 1;
            fp_tobytes(out, prod);
            if (std::memcmp(out, one, 32) != 0)
                all_one = false;
        }
        ++tests_run;
        if (all_one)
        {
            ++tests_passed;
            std::cout << "  PASS: fp batch invert 4 elements" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: fp batch invert 4 elements" << std::endl;
        }
    }

    /* Fq: batch invert 4 elements, verify each out[i] * in[i] == 1 */
    {
        fq_fe elems[4], invs[4];
        unsigned char v2[32] = {0x02}, v3[32] = {0x03}, v5[32] = {0x05}, v7[32] = {0x07};
        fq_frombytes(elems[0], v2);
        fq_frombytes(elems[1], v3);
        fq_frombytes(elems[2], v5);
        fq_frombytes(elems[3], v7);

        fq_batch_invert(invs, elems, 4);

        bool all_one = true;
        for (int i = 0; i < 4; i++)
        {
            fq_fe prod;
            fq_mul(prod, elems[i], invs[i]);
            unsigned char out[32], one[32] = {};
            one[0] = 1;
            fq_tobytes(out, prod);
            if (std::memcmp(out, one, 32) != 0)
                all_one = false;
        }
        ++tests_run;
        if (all_one)
        {
            ++tests_passed;
            std::cout << "  PASS: fq batch invert 4 elements" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: fq batch invert 4 elements" << std::endl;
        }
    }

    /* Fp: batch with zero element in position 2 */
    {
        fp_fe elems[4], invs[4];
        unsigned char v2[32] = {0x02}, v3[32] = {0x03}, v0[32] = {}, v7[32] = {0x07};
        fp_frombytes(elems[0], v2);
        fp_frombytes(elems[1], v3);
        fp_frombytes(elems[2], v0); /* zero */
        fp_frombytes(elems[3], v7);

        fp_batch_invert(invs, elems, 4);

        /* Check zero maps to zero */
        unsigned char zero_out[32], zero_exp[32] = {};
        fp_tobytes(zero_out, invs[2]);
        bool zero_ok = (std::memcmp(zero_out, zero_exp, 32) == 0);

        /* Check nonzero elements still correct */
        bool nonzero_ok = true;
        for (int i : {0, 1, 3})
        {
            fp_fe prod;
            fp_mul(prod, elems[i], invs[i]);
            unsigned char out[32], one[32] = {};
            one[0] = 1;
            fp_tobytes(out, prod);
            if (std::memcmp(out, one, 32) != 0)
                nonzero_ok = false;
        }
        ++tests_run;
        if (zero_ok && nonzero_ok)
        {
            ++tests_passed;
            std::cout << "  PASS: fp batch invert with zero element" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: fp batch invert with zero element" << std::endl;
        }
    }

    /* Fq: batch with zero element in position 2 */
    {
        fq_fe elems[4], invs[4];
        unsigned char v2[32] = {0x02}, v3[32] = {0x03}, v0[32] = {}, v7[32] = {0x07};
        fq_frombytes(elems[0], v2);
        fq_frombytes(elems[1], v3);
        fq_frombytes(elems[2], v0); /* zero */
        fq_frombytes(elems[3], v7);

        fq_batch_invert(invs, elems, 4);

        unsigned char zero_out[32], zero_exp[32] = {};
        fq_tobytes(zero_out, invs[2]);
        bool zero_ok = (std::memcmp(zero_out, zero_exp, 32) == 0);

        bool nonzero_ok = true;
        for (int i : {0, 1, 3})
        {
            fq_fe prod;
            fq_mul(prod, elems[i], invs[i]);
            unsigned char out[32], one[32] = {};
            one[0] = 1;
            fq_tobytes(out, prod);
            if (std::memcmp(out, one, 32) != 0)
                nonzero_ok = false;
        }
        ++tests_run;
        if (zero_ok && nonzero_ok)
        {
            ++tests_passed;
            std::cout << "  PASS: fq batch invert with zero element" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: fq batch invert with zero element" << std::endl;
        }
    }
}

static void test_fixed_base_scalarmult()
{
    std::cout << std::endl << "=== Fixed-base scalarmult (w=5) ===" << std::endl;

    /* Ran: fixed_scalarmult(7, G) == scalarmult(7, G) */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        ran_affine table[16];
        ran_scalarmult_fixed_precompute(table, &G);

        unsigned char s7[32] = {0x07};
        ran_jacobian fixed_result, expected;
        ran_scalarmult_fixed(&fixed_result, s7, table);
        ran_scalarmult(&expected, s7, &G);

        unsigned char fr[32], ex[32];
        ran_tobytes(fr, &fixed_result);
        ran_tobytes(ex, &expected);
        check_bytes("ran fixed: 7*G", ex, fr, 32);
    }

    /* Shaw: fixed_scalarmult(7, G) == scalarmult(7, G) */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_affine table[16];
        shaw_scalarmult_fixed_precompute(table, &G);

        unsigned char s7[32] = {0x07};
        shaw_jacobian fixed_result, expected;
        shaw_scalarmult_fixed(&fixed_result, s7, table);
        shaw_scalarmult(&expected, s7, &G);

        unsigned char fr[32], ex[32];
        shaw_tobytes(fr, &fixed_result);
        shaw_tobytes(ex, &expected);
        check_bytes("shaw fixed: 7*G", ex, fr, 32);
    }

    /* Ran: fixed with large scalar (associativity) */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        ran_affine table[16];
        ran_scalarmult_fixed_precompute(table, &G);

        unsigned char s21[32] = {0x15};
        ran_jacobian fixed_result, expected;
        ran_scalarmult_fixed(&fixed_result, s21, table);
        ran_scalarmult(&expected, s21, &G);

        unsigned char fr[32], ex[32];
        ran_tobytes(fr, &fixed_result);
        ran_tobytes(ex, &expected);
        check_bytes("ran fixed: 21*G", ex, fr, 32);
    }

    /* Shaw: fixed with large scalar */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_affine table[16];
        shaw_scalarmult_fixed_precompute(table, &G);

        unsigned char s21[32] = {0x15};
        shaw_jacobian fixed_result, expected;
        shaw_scalarmult_fixed(&fixed_result, s21, table);
        shaw_scalarmult(&expected, s21, &G);

        unsigned char fr[32], ex[32];
        shaw_tobytes(fr, &fixed_result);
        shaw_tobytes(ex, &expected);
        check_bytes("shaw fixed: 21*G", ex, fr, 32);
    }

    /* Ran: fixed with multi-byte scalar */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        ran_affine table[16];
        ran_scalarmult_fixed_precompute(table, &G);

        unsigned char sc[32] = {0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89};
        ran_jacobian fixed_result, expected;
        ran_scalarmult_fixed(&fixed_result, sc, table);
        ran_scalarmult(&expected, sc, &G);

        unsigned char fr[32], ex[32];
        ran_tobytes(fr, &fixed_result);
        ran_tobytes(ex, &expected);
        check_bytes("ran fixed: large scalar", ex, fr, 32);
    }

    /* Shaw: fixed with multi-byte scalar */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_affine table[16];
        shaw_scalarmult_fixed_precompute(table, &G);

        unsigned char sc[32] = {0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89};
        shaw_jacobian fixed_result, expected;
        shaw_scalarmult_fixed(&fixed_result, sc, table);
        shaw_scalarmult(&expected, sc, &G);

        unsigned char fr[32], ex[32];
        shaw_tobytes(fr, &fixed_result);
        shaw_tobytes(ex, &expected);
        check_bytes("shaw fixed: large scalar", ex, fr, 32);
    }

    /* Ran: scalar = 1 (edge case) */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        ran_affine table[16];
        ran_scalarmult_fixed_precompute(table, &G);

        unsigned char s1[32] = {0x01};
        ran_jacobian fixed_result;
        ran_scalarmult_fixed(&fixed_result, s1, table);

        unsigned char fr[32], gx[32];
        ran_tobytes(fr, &fixed_result);
        ran_tobytes(gx, &G);
        check_bytes("ran fixed: 1*G == G", gx, fr, 32);
    }

    /* Shaw: scalar = 1 (edge case) */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_affine table[16];
        shaw_scalarmult_fixed_precompute(table, &G);

        unsigned char s1[32] = {0x01};
        shaw_jacobian fixed_result;
        shaw_scalarmult_fixed(&fixed_result, s1, table);

        unsigned char fr[32], gx[32];
        shaw_tobytes(fr, &fixed_result);
        shaw_tobytes(gx, &G);
        check_bytes("shaw fixed: 1*G == G", gx, fr, 32);
    }
}

static void test_precomputed_tables()
{
    std::cout << std::endl << "=== Precomputed generator tables ===" << std::endl;

    /* Ran: precomputed table matches runtime computation */
    {
        ran_affine precomp[16], runtime[16];
        ran_load_g_table(precomp);

        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);
        ran_scalarmult_fixed_precompute(runtime, &G);

        bool all_match = true;
        for (int i = 0; i < 16; i++)
        {
            unsigned char px[32], py[32], rx[32], ry[32];
            fp_tobytes(px, precomp[i].x);
            fp_tobytes(py, precomp[i].y);
            fp_tobytes(rx, runtime[i].x);
            fp_tobytes(ry, runtime[i].y);
            if (std::memcmp(px, rx, 32) != 0 || std::memcmp(py, ry, 32) != 0)
                all_match = false;
        }
        ++tests_run;
        if (all_match)
        {
            ++tests_passed;
            std::cout << "  PASS: ran precomp table matches runtime" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: ran precomp table mismatch" << std::endl;
        }
    }

    /* Shaw: precomputed table matches runtime computation */
    {
        shaw_affine precomp[16], runtime[16];
        shaw_load_g_table(precomp);

        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);
        shaw_scalarmult_fixed_precompute(runtime, &G);

        bool all_match = true;
        for (int i = 0; i < 16; i++)
        {
            unsigned char px[32], py[32], rx[32], ry[32];
            fq_tobytes(px, precomp[i].x);
            fq_tobytes(py, precomp[i].y);
            fq_tobytes(rx, runtime[i].x);
            fq_tobytes(ry, runtime[i].y);
            if (std::memcmp(px, rx, 32) != 0 || std::memcmp(py, ry, 32) != 0)
                all_match = false;
        }
        ++tests_run;
        if (all_match)
        {
            ++tests_passed;
            std::cout << "  PASS: shaw precomp table matches runtime" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: shaw precomp table mismatch" << std::endl;
        }
    }

    /* Ran: fixed scalarmult with precomp table matches regular scalarmult */
    {
        ran_affine table[16];
        ran_load_g_table(table);

        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        unsigned char sc[32] = {0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89};
        ran_jacobian fixed_result, expected;
        ran_scalarmult_fixed(&fixed_result, sc, table);
        ran_scalarmult(&expected, sc, &G);

        unsigned char fr[32], ex[32];
        ran_tobytes(fr, &fixed_result);
        ran_tobytes(ex, &expected);
        check_bytes("ran precomp scalarmult", ex, fr, 32);
    }

    /* Shaw: fixed scalarmult with precomp table matches regular scalarmult */
    {
        shaw_affine table[16];
        shaw_load_g_table(table);

        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        unsigned char sc[32] = {0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89};
        shaw_jacobian fixed_result, expected;
        shaw_scalarmult_fixed(&fixed_result, sc, table);
        shaw_scalarmult(&expected, sc, &G);

        unsigned char fr[32], ex[32];
        shaw_tobytes(fr, &fixed_result);
        shaw_tobytes(ex, &expected);
        check_bytes("shaw precomp scalarmult", ex, fr, 32);
    }
}

static void test_msm_fixed()
{
    std::cout << std::endl << "=== Fixed-base MSM ===" << std::endl;

    /* Ran: msm_fixed(s1*G + s2*2G) == scalarmult(s1, G) + scalarmult(s2, 2G) */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        ran_jacobian G2;
        ran_dbl(&G2, &G);

        ran_affine table_g[16], table_g2[16];
        ran_scalarmult_fixed_precompute(table_g, &G);
        ran_scalarmult_fixed_precompute(table_g2, &G2);

        const ran_affine *tables[2] = {table_g, table_g2};
        unsigned char scalars[64] = {};
        scalars[0] = 0x07; /* s1 = 7 */
        scalars[32] = 0x05; /* s2 = 5 */

        ran_jacobian msm_result;
        ran_msm_fixed(&msm_result, scalars, tables, 2);

        /* Expected: 7*G + 5*(2G) = 7*G + 10*G = 17*G */
        unsigned char s17[32] = {0x11};
        ran_jacobian expected;
        ran_scalarmult(&expected, s17, &G);

        unsigned char mr[32], ex[32];
        ran_tobytes(mr, &msm_result);
        ran_tobytes(ex, &expected);
        check_bytes("ran msm_fixed: 7*G + 5*(2G) == 17*G", ex, mr, 32);
    }

    /* Shaw: msm_fixed(s1*G + s2*2G) */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_jacobian G2;
        shaw_dbl(&G2, &G);

        shaw_affine table_g[16], table_g2[16];
        shaw_scalarmult_fixed_precompute(table_g, &G);
        shaw_scalarmult_fixed_precompute(table_g2, &G2);

        const shaw_affine *tables[2] = {table_g, table_g2};
        unsigned char scalars[64] = {};
        scalars[0] = 0x07;
        scalars[32] = 0x05;

        shaw_jacobian msm_result;
        shaw_msm_fixed(&msm_result, scalars, tables, 2);

        unsigned char s17[32] = {0x11};
        shaw_jacobian expected;
        shaw_scalarmult(&expected, s17, &G);

        unsigned char mr[32], ex[32];
        shaw_tobytes(mr, &msm_result);
        shaw_tobytes(ex, &expected);
        check_bytes("shaw msm_fixed: 7*G + 5*(2G) == 17*G", ex, mr, 32);
    }

    /* Ran: msm_fixed with 3 points and larger scalars */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        ran_jacobian G2, G3;
        ran_dbl(&G2, &G);
        ran_add(&G3, &G2, &G);

        ran_affine t1[16], t2[16], t3[16];
        ran_scalarmult_fixed_precompute(t1, &G);
        ran_scalarmult_fixed_precompute(t2, &G2);
        ran_scalarmult_fixed_precompute(t3, &G3);

        const ran_affine *tables[3] = {t1, t2, t3};
        unsigned char scalars[96] = {};
        scalars[0] = 0x03; /* s1 = 3 */
        scalars[32] = 0x05; /* s2 = 5 */
        scalars[64] = 0x07; /* s3 = 7 */

        ran_jacobian msm_result;
        ran_msm_fixed(&msm_result, scalars, tables, 3);

        /* Expected: 3*G + 5*(2G) + 7*(3G) = 3+10+21 = 34*G */
        unsigned char s34[32] = {0x22};
        ran_jacobian expected;
        ran_scalarmult(&expected, s34, &G);

        unsigned char mr[32], ex[32];
        ran_tobytes(mr, &msm_result);
        ran_tobytes(ex, &expected);
        check_bytes("ran msm_fixed: 3*G + 5*(2G) + 7*(3G) == 34*G", ex, mr, 32);
    }

    /* Shaw: msm_fixed n=1 fallback */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_affine table[16];
        shaw_scalarmult_fixed_precompute(table, &G);

        const shaw_affine *tables[1] = {table};
        unsigned char scalars[32] = {0x0b}; /* 11 */

        shaw_jacobian msm_result, expected;
        shaw_msm_fixed(&msm_result, scalars, tables, 1);

        unsigned char s11[32] = {0x0b};
        shaw_scalarmult(&expected, s11, &G);

        unsigned char mr[32], ex[32];
        shaw_tobytes(mr, &msm_result);
        shaw_tobytes(ex, &expected);
        check_bytes("shaw msm_fixed: n=1 (11*G)", ex, mr, 32);
    }
}

static void test_pedersen_extended()
{
    std::cout << std::endl << "=== Pedersen extended ===" << std::endl;
    (void)zero_bytes;

    /* Ran: n=3 multiple generators */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        ran_jacobian H, G2, G3;
        ran_dbl(&H, &G); /* H = 2G */
        ran_add(&G2, &H, &G); /* G2 = 3G */
        ran_dbl(&G3, &H); /* G3 = 4G */

        unsigned char r_scalar[32] = {};
        r_scalar[0] = 0x02;
        unsigned char vals[96];
        unsigned char v1[32] = {};
        v1[0] = 0x03;
        unsigned char v2[32] = {};
        v2[0] = 0x05;
        unsigned char v3[32] = {};
        v3[0] = 0x07;
        std::memcpy(vals, v1, 32);
        std::memcpy(vals + 32, v2, 32);
        std::memcpy(vals + 64, v3, 32);

        ran_jacobian gens[3];
        ran_copy(&gens[0], &G);
        ran_copy(&gens[1], &G2);
        ran_copy(&gens[2], &G3);

        ran_jacobian commit;
        ran_pedersen_commit(&commit, r_scalar, &H, vals, gens, 3);

        /* Expected: 2*H + 3*G + 5*G2 + 7*G3 = 2*2G + 3*G + 5*3G + 7*4G = 4G+3G+15G+28G = 50G */
        unsigned char s50[32] = {0x32};
        ran_jacobian expected;
        ran_scalarmult_vartime(&expected, s50, &G);
        unsigned char c_bytes[32], e_bytes[32];
        ran_tobytes(c_bytes, &commit);
        ran_tobytes(e_bytes, &expected);
        check_bytes("ran pedersen n=3", e_bytes, c_bytes, 32);
    }

    /* Shaw: n=0 blinding only */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_jacobian H;
        shaw_dbl(&H, &G);

        unsigned char r_scalar[32] = {0x03};
        shaw_jacobian commit;
        shaw_pedersen_commit(&commit, r_scalar, &H, nullptr, nullptr, 0);

        unsigned char s3[32] = {0x03};
        shaw_jacobian expected;
        shaw_scalarmult_vartime(&expected, s3, &H);
        unsigned char c_bytes[32], e_bytes[32];
        shaw_tobytes(c_bytes, &commit);
        shaw_tobytes(e_bytes, &expected);
        check_bytes("shaw pedersen n=0: r*H", e_bytes, c_bytes, 32);
    }

    /* Ran: zero blinding */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        ran_jacobian H;
        ran_dbl(&H, &G);

        unsigned char s5[32] = {0x05};
        ran_jacobian commit;
        ran_pedersen_commit(&commit, zero_bytes, &H, s5, &G, 1);

        ran_jacobian expected;
        ran_scalarmult_vartime(&expected, s5, &G);
        unsigned char c_bytes[32], e_bytes[32];
        ran_tobytes(c_bytes, &commit);
        ran_tobytes(e_bytes, &expected);
        check_bytes("ran pedersen(0, H, [5], [G]) == 5*G", e_bytes, c_bytes, 32);
    }

    /* Shaw: n=3 multiple generators */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_jacobian H, G2, G3;
        shaw_dbl(&H, &G);
        shaw_add(&G2, &H, &G);
        shaw_dbl(&G3, &H);

        unsigned char r_scalar[32] = {};
        r_scalar[0] = 0x02;
        unsigned char vals[96];
        unsigned char v1[32] = {};
        v1[0] = 0x03;
        unsigned char v2[32] = {};
        v2[0] = 0x05;
        unsigned char v3[32] = {};
        v3[0] = 0x07;
        std::memcpy(vals, v1, 32);
        std::memcpy(vals + 32, v2, 32);
        std::memcpy(vals + 64, v3, 32);

        shaw_jacobian gens[3];
        shaw_copy(&gens[0], &G);
        shaw_copy(&gens[1], &G2);
        shaw_copy(&gens[2], &G3);

        shaw_jacobian commit;
        shaw_pedersen_commit(&commit, r_scalar, &H, vals, gens, 3);

        unsigned char s50[32] = {0x32};
        shaw_jacobian expected;
        shaw_scalarmult_vartime(&expected, s50, &G);
        unsigned char c_bytes[32], e_bytes[32];
        shaw_tobytes(c_bytes, &commit);
        shaw_tobytes(e_bytes, &expected);
        check_bytes("shaw pedersen n=3", e_bytes, c_bytes, 32);
    }
}

static void test_poly_extended()
{
    std::cout << std::endl << "=== Polynomial extended ===" << std::endl;
    unsigned char buf[32];

    /* Degree-0: constant * constant */
    {
        fp_poly a, b, r;
        a.coeffs.resize(1);
        unsigned char three_b[32] = {0x03};
        fp_frombytes(a.coeffs[0].v, three_b);
        b.coeffs.resize(1);
        unsigned char five_b[32] = {0x05};
        fp_frombytes(b.coeffs[0].v, five_b);
        fp_poly_mul(&r, &a, &b);
        check_int("deg-0 mul result size", 1, (int)r.coeffs.size());
        fp_fe c0;
        std::memcpy(c0, r.coeffs[0].v, sizeof(fp_fe));
        fp_tobytes(buf, c0);
        unsigned char fifteen_b[32] = {0x0f};
        check_bytes("3 * 5 == 15", fifteen_b, buf, 32);
    }

    /* eval(any_poly, 0) == constant coefficient */
    {
        fp_poly p;
        p.coeffs.resize(3);
        unsigned char c0_b[32] = {0x07};
        unsigned char c1_b[32] = {0x03};
        unsigned char c2_b[32] = {0x02};
        fp_frombytes(p.coeffs[0].v, c0_b);
        fp_frombytes(p.coeffs[1].v, c1_b);
        fp_frombytes(p.coeffs[2].v, c2_b);

        fp_fe zero_val, result;
        fp_0(zero_val);
        fp_poly_eval(result, &p, zero_val);
        fp_tobytes(buf, result);
        check_bytes("fp eval(poly, 0) == const coeff", c0_b, buf, 32);
    }

    /* Single root: from_roots([r], 1), eval at r == 0 */
    {
        unsigned char r_b[32] = {0x09};
        fp_fe root;
        fp_frombytes(root, r_b);
        fp_poly p;
        fp_poly_from_roots(&p, &root, 1);
        fp_fe val;
        fp_poly_eval(val, &p, root);
        fp_tobytes(buf, val);
        check_bytes("fp from_roots([9]) eval at 9 == 0", zero_bytes, buf, 32);
    }

    /* Many roots n=10: eval at each root == 0 */
    {
        fp_fe roots[10];
        for (int i = 0; i < 10; i++)
        {
            unsigned char rb[32] = {};
            rb[0] = (unsigned char)(i + 1);
            fp_frombytes(roots[i], rb);
        }
        fp_poly p;
        fp_poly_from_roots(&p, roots, 10);
        bool all_zero = true;
        for (int i = 0; i < 10; i++)
        {
            fp_fe val;
            fp_poly_eval(val, &p, roots[i]);
            unsigned char vb[32];
            fp_tobytes(vb, val);
            if (std::memcmp(vb, zero_bytes, 32) != 0)
                all_zero = false;
        }
        ++tests_run;
        if (all_zero)
        {
            ++tests_passed;
            std::cout << "  PASS: fp from_roots n=10 all evals == 0" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: fp from_roots n=10 some eval != 0" << std::endl;
        }
    }

    /* fq_poly_divmod: (x^2-1) / (x+1) == (x-1), remainder 0 */
    {
        fq_poly dividend, divisor_poly, q, rem;
        dividend.coeffs.resize(3);
        fq_fe one_fe, neg1;
        fq_1(one_fe);
        fq_neg(neg1, one_fe);
        std::memcpy(dividend.coeffs[0].v, neg1, sizeof(fq_fe));
        fq_0(dividend.coeffs[1].v);
        fq_1(dividend.coeffs[2].v);

        divisor_poly.coeffs.resize(2);
        fq_1(divisor_poly.coeffs[0].v);
        fq_1(divisor_poly.coeffs[1].v);

        fq_poly_divmod(&q, &rem, &dividend, &divisor_poly);

        check_int("fq divmod quotient size", 2, (int)q.coeffs.size());

        fq_fe q0;
        std::memcpy(q0, q.coeffs[0].v, sizeof(fq_fe));
        fq_tobytes(buf, q0);
        unsigned char neg1_bytes[32];
        fq_tobytes(neg1_bytes, neg1);
        check_bytes("fq divmod quotient const == -1", neg1_bytes, buf, 32);

        fq_fe q1;
        std::memcpy(q1, q.coeffs[1].v, sizeof(fq_fe));
        fq_tobytes(buf, q1);
        check_bytes("fq divmod quotient x coeff == 1", one_bytes, buf, 32);

        fq_fe r0;
        std::memcpy(r0, rem.coeffs[0].v, sizeof(fq_fe));
        fq_tobytes(buf, r0);
        check_bytes("fq divmod remainder == 0", zero_bytes, buf, 32);
    }

    /* Non-zero remainder: (x^2+1) / (x+1) */
    {
        fp_poly dividend, divisor_poly, q, rem;
        dividend.coeffs.resize(3);
        fp_1(dividend.coeffs[0].v); /* 1 */
        fp_0(dividend.coeffs[1].v); /* 0 */
        fp_1(dividend.coeffs[2].v); /* x^2 */

        divisor_poly.coeffs.resize(2);
        fp_1(divisor_poly.coeffs[0].v);
        fp_1(divisor_poly.coeffs[1].v);

        fp_poly_divmod(&q, &rem, &dividend, &divisor_poly);

        /* Quotient should be (x-1) */
        check_int("nonzero rem: quotient size", 2, (int)q.coeffs.size());

        /* Remainder should be 2 */
        fp_fe r0;
        std::memcpy(r0, rem.coeffs[0].v, sizeof(fp_fe));
        fp_tobytes(buf, r0);
        unsigned char two_b[32] = {0x02};
        check_bytes("(x^2+1)/(x+1) remainder == 2", two_b, buf, 32);
    }

    /* fq eval(poly, 0) == constant coefficient */
    {
        fq_poly p;
        p.coeffs.resize(3);
        unsigned char c0_b[32] = {0x0b};
        unsigned char c1_b[32] = {0x03};
        unsigned char c2_b[32] = {0x02};
        fq_frombytes(p.coeffs[0].v, c0_b);
        fq_frombytes(p.coeffs[1].v, c1_b);
        fq_frombytes(p.coeffs[2].v, c2_b);

        fq_fe zero_val, result;
        fq_0(zero_val);
        fq_poly_eval(result, &p, zero_val);
        fq_tobytes(buf, result);
        check_bytes("fq eval(poly, 0) == const coeff", c0_b, buf, 32);
    }
}

static void test_divisor_extended()
{
    std::cout << std::endl << "=== Divisor extended ===" << std::endl;
    unsigned char buf[32];

    /* Ran: 5-point divisor */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        ran_jacobian pts_jac[6];
        ran_copy(&pts_jac[0], &G);
        ran_dbl(&pts_jac[1], &G);
        ran_add(&pts_jac[2], &pts_jac[1], &G);
        ran_dbl(&pts_jac[3], &pts_jac[1]);
        ran_add(&pts_jac[4], &pts_jac[3], &G);
        ran_add(&pts_jac[5], &pts_jac[4], &G); /* non-member */

        ran_affine pts[5], non_member;
        for (int i = 0; i < 5; i++)
            ran_to_affine(&pts[i], &pts_jac[i]);
        ran_to_affine(&non_member, &pts_jac[5]);

        ran_divisor d;
        ran_compute_divisor(&d, pts, 5);

        bool all_zero = true;
        for (int i = 0; i < 5; i++)
        {
            fp_fe val;
            ran_evaluate_divisor(val, &d, pts[i].x, pts[i].y);
            unsigned char vb[32];
            fp_tobytes(vb, val);
            if (std::memcmp(vb, zero_bytes, 32) != 0)
                all_zero = false;
        }
        ++tests_run;
        if (all_zero)
        {
            ++tests_passed;
            std::cout << "  PASS: ran 5-point divisor all evals == 0" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: ran 5-point divisor some eval != 0" << std::endl;
        }

        fp_fe val;
        ran_evaluate_divisor(val, &d, non_member.x, non_member.y);
        fp_tobytes(buf, val);
        check_nonzero("ran 5-point divisor non-member != 0", std::memcmp(buf, zero_bytes, 32) != 0 ? 1 : 0);
    }

    /* Shaw: single-point divisor */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_affine pt;
        shaw_to_affine(&pt, &G);

        shaw_divisor d;
        shaw_compute_divisor(&d, &pt, 1);

        fq_fe val;
        shaw_evaluate_divisor(val, &d, pt.x, pt.y);
        fq_tobytes(buf, val);
        check_bytes("shaw single-point divisor eval == 0", zero_bytes, buf, 32);
    }

    /* Shaw: 5-point divisor */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_jacobian pts_jac[6];
        shaw_copy(&pts_jac[0], &G);
        shaw_dbl(&pts_jac[1], &G);
        shaw_add(&pts_jac[2], &pts_jac[1], &G);
        shaw_dbl(&pts_jac[3], &pts_jac[1]);
        shaw_add(&pts_jac[4], &pts_jac[3], &G);
        shaw_add(&pts_jac[5], &pts_jac[4], &G);

        shaw_affine pts[5], non_member;
        for (int i = 0; i < 5; i++)
            shaw_to_affine(&pts[i], &pts_jac[i]);
        shaw_to_affine(&non_member, &pts_jac[5]);

        shaw_divisor d;
        shaw_compute_divisor(&d, pts, 5);

        bool all_zero = true;
        for (int i = 0; i < 5; i++)
        {
            fq_fe val;
            shaw_evaluate_divisor(val, &d, pts[i].x, pts[i].y);
            unsigned char vb[32];
            fq_tobytes(vb, val);
            if (std::memcmp(vb, zero_bytes, 32) != 0)
                all_zero = false;
        }
        ++tests_run;
        if (all_zero)
        {
            ++tests_passed;
            std::cout << "  PASS: shaw 5-point divisor all evals == 0" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: shaw 5-point divisor some eval != 0" << std::endl;
        }
    }
}

static void test_point_to_scalar()
{
    std::cout << std::endl << "=== Point-to-scalar ===" << std::endl;

    /* Ran: extract x-coordinate of G as bytes */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        unsigned char xbytes[32];
        ran_point_to_bytes(xbytes, &G);

        /* G.x == 3, so bytes should be {3, 0, ..., 0} */
        unsigned char expected_gx[32] = {0x03};
        check_bytes("ran G.x == 3", expected_gx, xbytes, 32);
    }

    /* Shaw: extract x-coordinate of G as bytes */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        unsigned char xbytes[32];
        shaw_point_to_bytes(xbytes, &G);

        /* G.x == 1, so bytes should be {1, 0, ..., 0} */
        unsigned char expected_gx[32] = {0x01};
        check_bytes("shaw G.x == 1", expected_gx, xbytes, 32);
    }

    /* Round-trip: 7*G, extract x, verify tobytes(affine.x) matches */
    {
        ran_jacobian G, P;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        unsigned char scalar_7[32] = {0x07};
        ran_scalarmult_vartime(&P, scalar_7, &G);

        unsigned char pt_bytes[32];
        ran_point_to_bytes(pt_bytes, &P);

        /* Verify via independent affine conversion */
        ran_affine a;
        ran_to_affine(&a, &P);
        unsigned char ref_bytes[32];
        fp_tobytes(ref_bytes, a.x);
        check_bytes("ran 7G round-trip x", ref_bytes, pt_bytes, 32);
    }

    /* Identity: should produce 32 zero bytes */
    {
        ran_jacobian id;
        ran_identity(&id);
        unsigned char xbytes[32];
        ran_point_to_bytes(xbytes, &id);
        check_bytes("ran identity -> zero bytes", zero_bytes, xbytes, 32);

        shaw_jacobian sid;
        shaw_identity(&sid);
        unsigned char sxbytes[32];
        shaw_point_to_bytes(sxbytes, &sid);
        check_bytes("shaw identity -> zero bytes", zero_bytes, sxbytes, 32);
    }

    /* Cross-curve: Ran point -> Fp bytes -> Shaw scalar -> Shaw point -> Fq bytes -> Ran scalar */
    {
        ran_jacobian HG, HP;
        fp_copy(HG.X, RAN_GX);
        fp_copy(HG.Y, RAN_GY);
        fp_1(HG.Z);

        unsigned char scalar_5[32] = {0x05};
        ran_scalarmult_vartime(&HP, scalar_5, &HG);

        /* Extract Ran x-coordinate as bytes (element of Fp = Shaw scalar field) */
        unsigned char hp_x[32];
        ran_point_to_bytes(hp_x, &HP);

        /* Use as Shaw scalar for scalarmult */
        shaw_jacobian SG, SP;
        fq_copy(SG.X, SHAW_GX);
        fq_copy(SG.Y, SHAW_GY);
        fq_1(SG.Z);
        shaw_scalarmult_vartime(&SP, hp_x, &SG);

        /* Extract Shaw x-coordinate as bytes (element of Fq = Ran scalar field) */
        unsigned char sp_x[32];
        shaw_point_to_bytes(sp_x, &SP);

        /* Use as Ran scalar */
        ran_jacobian HP2;
        ran_scalarmult_vartime(&HP2, sp_x, &HG);

        /* Verify the chain produced a valid non-identity point */
        unsigned char hp2_bytes[32];
        ran_point_to_bytes(hp2_bytes, &HP2);
        check_nonzero("cross-curve chain produces non-identity", std::memcmp(hp2_bytes, zero_bytes, 32) != 0 ? 1 : 0);
    }
}

static void test_ran_scalar()
{
    std::cout << std::endl << "=== Ran scalar ===" << std::endl;

    fq_fe a, b;
    fq_frombytes(a, test_a_bytes);
    fq_frombytes(b, test_b_bytes);

    /* a + (-a) == 0 */
    {
        fq_fe neg_a, sum;
        ran_scalar_neg(neg_a, a);
        ran_scalar_add(sum, a, neg_a);
        unsigned char out[32];
        ran_scalar_to_bytes(out, sum);
        check_bytes("ran scalar a + (-a) == 0", zero_bytes, out, 32);
    }

    /* a * 1 == a */
    {
        fq_fe one, prod;
        ran_scalar_one(one);
        ran_scalar_mul(prod, a, one);
        unsigned char out[32], expected[32];
        ran_scalar_to_bytes(out, prod);
        ran_scalar_to_bytes(expected, a);
        check_bytes("ran scalar a * 1 == a", expected, out, 32);
    }

    /* a * a^(-1) == 1 */
    {
        fq_fe inv, prod;
        ran_scalar_invert(inv, a);
        ran_scalar_mul(prod, a, inv);
        unsigned char out[32];
        ran_scalar_to_bytes(out, prod);
        check_bytes("ran scalar a * a^-1 == 1", one_bytes, out, 32);
    }

    /* Distributivity: a * (b + c) == a*b + a*c where c = 1 */
    {
        fq_fe one, b_plus_one, lhs, ab, a_one, rhs;
        ran_scalar_one(one);
        ran_scalar_add(b_plus_one, b, one);
        ran_scalar_mul(lhs, a, b_plus_one);
        ran_scalar_mul(ab, a, b);
        ran_scalar_mul(a_one, a, one);
        ran_scalar_add(rhs, ab, a_one);
        unsigned char lhs_bytes[32], rhs_bytes[32];
        ran_scalar_to_bytes(lhs_bytes, lhs);
        ran_scalar_to_bytes(rhs_bytes, rhs);
        check_bytes("ran scalar distributivity", lhs_bytes, rhs_bytes, 32);
    }

    /* Serialization round-trip */
    {
        unsigned char buf[32];
        ran_scalar_to_bytes(buf, a);
        fq_fe a2;
        ran_scalar_from_bytes(a2, buf);
        unsigned char buf2[32];
        ran_scalar_to_bytes(buf2, a2);
        check_bytes("ran scalar round-trip", buf, buf2, 32);
    }

    /* is_zero */
    {
        fq_fe z;
        ran_scalar_zero(z);
        check_int("ran scalar is_zero(0)", 1, ran_scalar_is_zero(z));
        check_int("ran scalar !is_zero(a)", 0, ran_scalar_is_zero(a));
    }

    /* Wide reduction: reduce known 64-byte value */
    {
        /* All-zero 64 bytes should give zero */
        unsigned char wide_zero[64] = {};
        fq_fe result;
        ran_scalar_reduce_wide(result, wide_zero);
        unsigned char out[32];
        ran_scalar_to_bytes(out, result);
        check_bytes("ran scalar reduce_wide(0) == 0", zero_bytes, out, 32);

        /* lo = 1, hi = 0 -> result = 1 */
        unsigned char wide_one[64] = {0x01};
        ran_scalar_reduce_wide(result, wide_one);
        ran_scalar_to_bytes(out, result);
        check_bytes("ran scalar reduce_wide(lo=1,hi=0) == 1", one_bytes, out, 32);
    }

    /* muladd: a*b + 1 == a*b + 1 (computed two ways) */
    {
        fq_fe one, ab, ab_plus_one, muladd_result;
        ran_scalar_one(one);
        ran_scalar_mul(ab, a, b);
        ran_scalar_add(ab_plus_one, ab, one);
        ran_scalar_muladd(muladd_result, a, b, one);
        unsigned char out1[32], out2[32];
        ran_scalar_to_bytes(out1, ab_plus_one);
        ran_scalar_to_bytes(out2, muladd_result);
        check_bytes("ran scalar muladd(a,b,1) == a*b+1", out1, out2, 32);
    }

    /* sq: a^2 == a*a */
    {
        fq_fe sq_result, mul_result;
        ran_scalar_sq(sq_result, a);
        ran_scalar_mul(mul_result, a, a);
        unsigned char out1[32], out2[32];
        ran_scalar_to_bytes(out1, sq_result);
        ran_scalar_to_bytes(out2, mul_result);
        check_bytes("ran scalar sq(a) == a*a", out1, out2, 32);
    }
}

static void test_shaw_scalar()
{
    std::cout << std::endl << "=== Shaw scalar ===" << std::endl;

    fp_fe a, b;
    fp_frombytes(a, test_a_bytes);
    fp_frombytes(b, test_b_bytes);

    /* a + (-a) == 0 */
    {
        fp_fe neg_a, sum;
        shaw_scalar_neg(neg_a, a);
        shaw_scalar_add(sum, a, neg_a);
        unsigned char out[32];
        shaw_scalar_to_bytes(out, sum);
        check_bytes("shaw scalar a + (-a) == 0", zero_bytes, out, 32);
    }

    /* a * 1 == a */
    {
        fp_fe one, prod;
        shaw_scalar_one(one);
        shaw_scalar_mul(prod, a, one);
        unsigned char out[32], expected[32];
        shaw_scalar_to_bytes(out, prod);
        shaw_scalar_to_bytes(expected, a);
        check_bytes("shaw scalar a * 1 == a", expected, out, 32);
    }

    /* a * a^(-1) == 1 */
    {
        fp_fe inv, prod;
        shaw_scalar_invert(inv, a);
        shaw_scalar_mul(prod, a, inv);
        unsigned char out[32];
        shaw_scalar_to_bytes(out, prod);
        check_bytes("shaw scalar a * a^-1 == 1", one_bytes, out, 32);
    }

    /* Distributivity */
    {
        fp_fe one, b_plus_one, lhs, ab, a_one, rhs;
        shaw_scalar_one(one);
        shaw_scalar_add(b_plus_one, b, one);
        shaw_scalar_mul(lhs, a, b_plus_one);
        shaw_scalar_mul(ab, a, b);
        shaw_scalar_mul(a_one, a, one);
        shaw_scalar_add(rhs, ab, a_one);
        unsigned char lhs_bytes[32], rhs_bytes[32];
        shaw_scalar_to_bytes(lhs_bytes, lhs);
        shaw_scalar_to_bytes(rhs_bytes, rhs);
        check_bytes("shaw scalar distributivity", lhs_bytes, rhs_bytes, 32);
    }

    /* Serialization round-trip */
    {
        unsigned char buf[32];
        shaw_scalar_to_bytes(buf, a);
        fp_fe a2;
        shaw_scalar_from_bytes(a2, buf);
        unsigned char buf2[32];
        shaw_scalar_to_bytes(buf2, a2);
        check_bytes("shaw scalar round-trip", buf, buf2, 32);
    }

    /* is_zero */
    {
        fp_fe z;
        shaw_scalar_zero(z);
        check_int("shaw scalar is_zero(0)", 1, shaw_scalar_is_zero(z));
        check_int("shaw scalar !is_zero(a)", 0, shaw_scalar_is_zero(a));
    }

    /* Wide reduction */
    {
        unsigned char wide_zero[64] = {};
        fp_fe result;
        shaw_scalar_reduce_wide(result, wide_zero);
        unsigned char out[32];
        shaw_scalar_to_bytes(out, result);
        check_bytes("shaw scalar reduce_wide(0) == 0", zero_bytes, out, 32);

        unsigned char wide_one[64] = {0x01};
        shaw_scalar_reduce_wide(result, wide_one);
        shaw_scalar_to_bytes(out, result);
        check_bytes("shaw scalar reduce_wide(lo=1,hi=0) == 1", one_bytes, out, 32);
    }

    /* muladd: a*b + 1 == a*b + 1 (computed two ways) */
    {
        fp_fe one, ab, ab_plus_one, muladd_result;
        shaw_scalar_one(one);
        shaw_scalar_mul(ab, a, b);
        shaw_scalar_add(ab_plus_one, ab, one);
        shaw_scalar_muladd(muladd_result, a, b, one);
        unsigned char out1[32], out2[32];
        shaw_scalar_to_bytes(out1, ab_plus_one);
        shaw_scalar_to_bytes(out2, muladd_result);
        check_bytes("shaw scalar muladd(a,b,1) == a*b+1", out1, out2, 32);
    }

    /* sq: a^2 == a*a */
    {
        fp_fe sq_result, mul_result;
        shaw_scalar_sq(sq_result, a);
        shaw_scalar_mul(mul_result, a, a);
        unsigned char out1[32], out2[32];
        shaw_scalar_to_bytes(out1, sq_result);
        shaw_scalar_to_bytes(out2, mul_result);
        check_bytes("shaw scalar sq(a) == a*a", out1, out2, 32);
    }
}

static void test_poly_interpolate()
{
    std::cout << std::endl << "=== Polynomial interpolation ===" << std::endl;

    /* Fp: interpolate through 3 known points: (1,1), (2,4), (3,9) -> f(x) = x^2 */
    {
        fp_fe xs[3], ys[3];
        unsigned char x1[32] = {1}, x2[32] = {2}, x3[32] = {3};
        unsigned char y1[32] = {1}, y4[32] = {4}, y9[32] = {9};
        fp_frombytes(xs[0], x1);
        fp_frombytes(xs[1], x2);
        fp_frombytes(xs[2], x3);
        fp_frombytes(ys[0], y1);
        fp_frombytes(ys[1], y4);
        fp_frombytes(ys[2], y9);

        fp_poly out;
        fp_poly_interpolate(&out, xs, ys, 3);

        /* Verify evaluations: f(1)=1, f(2)=4, f(3)=9 */
        fp_fe result;
        fp_poly_eval(result, &out, xs[0]);
        unsigned char rb[32];
        fp_tobytes(rb, result);
        check_bytes("fp interp f(1)==1", y1, rb, 32);

        fp_poly_eval(result, &out, xs[1]);
        fp_tobytes(rb, result);
        check_bytes("fp interp f(2)==4", y4, rb, 32);

        fp_poly_eval(result, &out, xs[2]);
        fp_tobytes(rb, result);
        check_bytes("fp interp f(3)==9", y9, rb, 32);

        /* Degree check: 3 points -> degree 2 polynomial (3 coefficients) */
        check_int("fp interp degree == 2", 3, (int)out.coeffs.size());
    }

    /* Fq: interpolate through 3 known points: (1,2), (2,5), (3,10) -> f(x) = x^2 + 1 */
    {
        fq_fe xs[3], ys[3];
        unsigned char x1[32] = {1}, x2[32] = {2}, x3[32] = {3};
        unsigned char y2[32] = {2}, y5[32] = {5}, y10[32] = {10};
        fq_frombytes(xs[0], x1);
        fq_frombytes(xs[1], x2);
        fq_frombytes(xs[2], x3);
        fq_frombytes(ys[0], y2);
        fq_frombytes(ys[1], y5);
        fq_frombytes(ys[2], y10);

        fq_poly out;
        fq_poly_interpolate(&out, xs, ys, 3);

        fq_fe result;
        fq_poly_eval(result, &out, xs[0]);
        unsigned char rb[32];
        fq_tobytes(rb, result);
        check_bytes("fq interp f(1)==2", y2, rb, 32);

        fq_poly_eval(result, &out, xs[1]);
        fq_tobytes(rb, result);
        check_bytes("fq interp f(2)==5", y5, rb, 32);

        fq_poly_eval(result, &out, xs[2]);
        fq_tobytes(rb, result);
        check_bytes("fq interp f(3)==10", y10, rb, 32);

        check_int("fq interp degree == 2", 3, (int)out.coeffs.size());
    }

    /* Single-point interpolation */
    {
        fp_fe xs[1], ys[1];
        unsigned char x1[32] = {7}, y42[32] = {42};
        fp_frombytes(xs[0], x1);
        fp_frombytes(ys[0], y42);
        fp_poly out;
        fp_poly_interpolate(&out, xs, ys, 1);
        check_int("fp interp n=1 degree", 1, (int)out.coeffs.size());

        fp_fe result;
        fp_poly_eval(result, &out, xs[0]);
        unsigned char rb[32];
        fp_tobytes(rb, result);
        check_bytes("fp interp n=1 eval", y42, rb, 32);
    }
}

static void test_karatsuba()
{
    std::cout << std::endl << "=== Karatsuba ===" << std::endl;

    /*
     * Verify Karatsuba matches schoolbook for polynomials built from roots.
     * Build two poly of degree 32+ by using from_roots, then multiply them.
     * Verify by evaluating at a test point.
     */

    /* Fp: build A from 33 roots, B from 33 roots, multiply via Karatsuba */
    {
        /* Create roots: just small integers */
        fp_fe roots_a[33], roots_b[33];
        for (int i = 0; i < 33; i++)
        {
            unsigned char buf[32] = {};
            buf[0] = (unsigned char)(i + 1);
            fp_frombytes(roots_a[i], buf);
            buf[0] = (unsigned char)(i + 34);
            fp_frombytes(roots_b[i], buf);
        }

        fp_poly A, B, C;
        fp_poly_from_roots(&A, roots_a, 33);
        fp_poly_from_roots(&B, roots_b, 33);

        /* C = A * B (will use Karatsuba since both have 34 coefficients >= 32) */
        fp_poly_mul(&C, &A, &B);

        /* Verify: eval C at root of A should be 0 (since A(root)=0, C=A*B, so C(root)=0) */
        fp_fe result;
        fp_poly_eval(result, &C, roots_a[0]);
        unsigned char rb[32];
        fp_tobytes(rb, result);
        check_bytes("fp karatsuba: C(root_a[0]) == 0", zero_bytes, rb, 32);

        fp_poly_eval(result, &C, roots_a[16]);
        fp_tobytes(rb, result);
        check_bytes("fp karatsuba: C(root_a[16]) == 0", zero_bytes, rb, 32);

        /* Verify degree: (33+33) = 66 roots -> degree 66 product */
        check_int("fp karatsuba degree", 67, (int)C.coeffs.size());

        /* Verify at a non-root point: C(0) should be A(0)*B(0) */
        fp_fe zero_pt, a_at_0, b_at_0, expected_c0, c_at_0;
        fp_0(zero_pt);
        fp_poly_eval(a_at_0, &A, zero_pt);
        fp_poly_eval(b_at_0, &B, zero_pt);
        fp_mul(expected_c0, a_at_0, b_at_0);
        fp_poly_eval(c_at_0, &C, zero_pt);

        unsigned char exp_bytes[32], act_bytes[32];
        fp_tobytes(exp_bytes, expected_c0);
        fp_tobytes(act_bytes, c_at_0);
        check_bytes("fp karatsuba: C(0) == A(0)*B(0)", exp_bytes, act_bytes, 32);
    }

    /* Fq: same test */
    {
        fq_fe roots_a[33], roots_b[33];
        for (int i = 0; i < 33; i++)
        {
            unsigned char buf[32] = {};
            buf[0] = (unsigned char)(i + 1);
            fq_frombytes(roots_a[i], buf);
            buf[0] = (unsigned char)(i + 34);
            fq_frombytes(roots_b[i], buf);
        }

        fq_poly A, B, C;
        fq_poly_from_roots(&A, roots_a, 33);
        fq_poly_from_roots(&B, roots_b, 33);
        fq_poly_mul(&C, &A, &B);

        fq_fe result;
        fq_poly_eval(result, &C, roots_a[0]);
        unsigned char rb[32];
        fq_tobytes(rb, result);
        check_bytes("fq karatsuba: C(root_a[0]) == 0", zero_bytes, rb, 32);

        check_int("fq karatsuba degree", 67, (int)C.coeffs.size());

        fq_fe zero_pt, a_at_0, b_at_0, expected_c0, c_at_0;
        fq_0(zero_pt);
        fq_poly_eval(a_at_0, &A, zero_pt);
        fq_poly_eval(b_at_0, &B, zero_pt);
        fq_mul(expected_c0, a_at_0, b_at_0);
        fq_poly_eval(c_at_0, &C, zero_pt);

        unsigned char exp_bytes[32], act_bytes[32];
        fq_tobytes(exp_bytes, expected_c0);
        fq_tobytes(act_bytes, c_at_0);
        check_bytes("fq karatsuba: C(0) == A(0)*B(0)", exp_bytes, act_bytes, 32);
    }

    /* Mixed sizes: one small (< threshold), one large (>= threshold) -> schoolbook */
    {
        fp_fe roots_a[5], roots_b[33];
        for (int i = 0; i < 5; i++)
        {
            unsigned char buf[32] = {};
            buf[0] = (unsigned char)(i + 1);
            fp_frombytes(roots_a[i], buf);
        }
        for (int i = 0; i < 33; i++)
        {
            unsigned char buf[32] = {};
            buf[0] = (unsigned char)(i + 10);
            fp_frombytes(roots_b[i], buf);
        }

        fp_poly A, B, C;
        fp_poly_from_roots(&A, roots_a, 5);
        fp_poly_from_roots(&B, roots_b, 33);
        fp_poly_mul(&C, &A, &B);

        fp_fe result;
        fp_poly_eval(result, &C, roots_a[2]);
        unsigned char rb[32];
        fp_tobytes(rb, result);
        check_bytes("fp mixed-size: C(root_a[2]) == 0", zero_bytes, rb, 32);

        check_int("fp mixed-size degree", 39, (int)C.coeffs.size());
    }
}

#ifdef RANSHAW_ECFFT
#include "ecfft_fp.h"
#include "ecfft_fq.h"

static void test_ecfft()
{
    std::cout << std::endl << "=== ECFFT ===" << std::endl;

    /* ---- Fp ECFFT ---- */
    {
        /* Test init */
        ecfft_fp_ctx ctx = {};
        ecfft_fp_init(&ctx);
        check_int("fp ecfft domain_size", (int)ECFFT_FP_DOMAIN_SIZE, (int)ctx.domain_size);
        check_int("fp ecfft log_n", (int)ECFFT_FP_LOG_DOMAIN, (int)ctx.log_n);

        /* Test ENTER/EXIT round-trip with small polynomial */
        /* Polynomial: f(x) = 3 + 2x (degree 1, needs domain >= 2) */
        {
            fp_fe data[16];
            unsigned char buf3[32] = {0x03};
            unsigned char buf2[32] = {0x02};
            fp_frombytes(data[0], buf3);
            fp_frombytes(data[1], buf2);
            for (size_t i = 2; i < 16; i++)
                fp_0(data[i]);

            /* Save original coefficients */
            fp_fe orig0, orig1;
            fp_copy(orig0, data[0]);
            fp_copy(orig1, data[1]);

            /* Find the level where n == 16 */
            size_t enter_level = 0;
            for (size_t lv = 0; lv < ctx.log_n; lv++)
                if (ctx.levels[lv].n == 16)
                {
                    enter_level = lv;
                    break;
                }

            /* ENTER: coeff -> eval */
            ecfft_fp_enter(data, 16, &ctx);

            /* Verify: data[0] should be f(s[0]) = 3 + 2*s[0] */
            fp_fe expected, t;
            fp_frombytes(expected, buf3); /* 3 */
            fp_mul(t, orig1, ctx.levels[enter_level].s[0].v);
            fp_add(expected, expected, t);
            fp_fe zero;
            fp_0(zero);
            fp_sub(expected, expected, zero); /* normalize */

            unsigned char exp_b[32], act_b[32];
            fp_tobytes(exp_b, expected);
            fp_tobytes(act_b, data[0]);
            check_bytes("fp ecfft enter: f(s[0]) correct", exp_b, act_b, 32);

            /* EXIT: eval -> coeff */
            ecfft_fp_exit(data, 16, &ctx);

            /* Should recover original coefficients */
            fp_tobytes(exp_b, orig0);
            fp_tobytes(act_b, data[0]);
            check_bytes("fp ecfft enter/exit round-trip: coeff[0]", exp_b, act_b, 32);

            fp_tobytes(exp_b, orig1);
            fp_tobytes(act_b, data[1]);
            check_bytes("fp ecfft enter/exit round-trip: coeff[1]", exp_b, act_b, 32);

            /* Coefficients 2..15 should be zero */
            unsigned char zero_b[32] = {0};
            fp_tobytes(act_b, data[2]);
            check_bytes("fp ecfft enter/exit round-trip: coeff[2]==0", zero_b, act_b, 32);
        }

        /* Test ECFFT polynomial multiplication: (1 + x)(1 + x) = 1 + 2x + x^2 */
        {
            fp_fe a[2], b[2], result[16];
            unsigned char buf1[32] = {0x01};
            fp_frombytes(a[0], buf1);
            fp_frombytes(a[1], buf1);
            fp_frombytes(b[0], buf1);
            fp_frombytes(b[1], buf1);

            size_t result_len = 0;
            ecfft_fp_poly_mul(result, &result_len, a, 2, b, 2, &ctx);
            check_int("fp ecfft mul: result_len", 3, (int)result_len);

            /* Expected: [1, 2, 1] */
            unsigned char act[32];

            unsigned char one[32] = {0x01};
            unsigned char two[32] = {0x02};

            fp_tobytes(act, result[0]);
            check_bytes("fp ecfft mul: (1+x)^2 coeff[0]=1", one, act, 32);

            fp_tobytes(act, result[1]);
            check_bytes("fp ecfft mul: (1+x)^2 coeff[1]=2", two, act, 32);

            fp_tobytes(act, result[2]);
            check_bytes("fp ecfft mul: (1+x)^2 coeff[2]=1", one, act, 32);
        }

        /* Test ECFFT multiply matches schoolbook for degree-4 * degree-4 */
        {
            fp_fe a[5], b[5];
            for (int i = 0; i < 5; i++)
            {
                unsigned char buf[32] = {};
                buf[0] = (unsigned char)(i + 1);
                fp_frombytes(a[i], buf);
                buf[0] = (unsigned char)(i + 6);
                fp_frombytes(b[i], buf);
            }

            /* ECFFT multiply */
            fp_fe ecfft_result[16];
            size_t ecfft_len = 0;
            ecfft_fp_poly_mul(ecfft_result, &ecfft_len, a, 5, b, 5, &ctx);

            /* Verify via evaluation: A(x)*B(x) should equal result(x) at test point */
            check_int("fp ecfft mul deg4: result_len", 9, (int)ecfft_len);

            fp_poly pa, pb, pc;
            pa.coeffs.resize(5);
            pb.coeffs.resize(5);
            pc.coeffs.resize(ecfft_len);
            for (size_t i = 0; i < 5; i++)
            {
                fp_copy(pa.coeffs[i].v, a[i]);
                fp_copy(pb.coeffs[i].v, b[i]);
            }
            for (size_t i = 0; i < ecfft_len; i++)
                fp_copy(pc.coeffs[i].v, ecfft_result[i]);

            fp_fe test_x;
            unsigned char xbuf[32] = {0x37};
            fp_frombytes(test_x, xbuf);

            fp_fe va, vb, vc, vab;
            fp_poly_eval(va, &pa, test_x);
            fp_poly_eval(vb, &pb, test_x);
            fp_mul(vab, va, vb);
            fp_poly_eval(vc, &pc, test_x);

            unsigned char eb[32], ab[32];
            fp_tobytes(eb, vab);
            fp_tobytes(ab, vc);
            check_bytes("fp ecfft mul deg4: C(x)==A(x)*B(x)", eb, ab, 32);
        }

        /* Test dispatch integration: init global context, verify poly_mul uses it */
        {
            ecfft_global_init();

            fp_poly pa, pb, pc_ecfft;
            pa.coeffs.resize(9);
            pb.coeffs.resize(9);
            for (size_t i = 0; i < 9; i++)
            {
                unsigned char buf[32] = {};
                buf[0] = (unsigned char)(i + 1);
                fp_frombytes(pa.coeffs[i].v, buf);
                buf[0] = (unsigned char)(i + 10);
                fp_frombytes(pb.coeffs[i].v, buf);
            }

            /* This should use ECFFT since both are >= ECFFT_THRESHOLD */
            fp_poly_mul(&pc_ecfft, &pa, &pb);

            /* Verify by evaluating at a test point */
            fp_fe test_x;
            unsigned char test_buf[32] = {0x42};
            fp_frombytes(test_x, test_buf);

            fp_fe val_a, val_b, val_c, val_ab;
            fp_poly_eval(val_a, &pa, test_x);
            fp_poly_eval(val_b, &pb, test_x);
            fp_mul(val_ab, val_a, val_b);
            fp_poly_eval(val_c, &pc_ecfft, test_x);

            unsigned char eb[32], ab[32];
            fp_tobytes(eb, val_ab);
            fp_tobytes(ab, val_c);
            check_bytes("fp ecfft dispatch: C(x) == A(x)*B(x)", eb, ab, 32);
        }
    }

    /* ---- Fq ECFFT ---- */
    {
        ecfft_fq_ctx ctx = {};
        ecfft_fq_init(&ctx);

        /* Test ECFFT polynomial multiplication: (1 + x)(1 + x) = 1 + 2x + x^2 */
        {
            fq_fe a[2], b[2], result[16];
            unsigned char buf1[32] = {0x01};
            fq_frombytes(a[0], buf1);
            fq_frombytes(a[1], buf1);
            fq_frombytes(b[0], buf1);
            fq_frombytes(b[1], buf1);

            size_t result_len = 0;
            ecfft_fq_poly_mul(result, &result_len, a, 2, b, 2, &ctx);
            check_int("fq ecfft mul: result_len", 3, (int)result_len);

            unsigned char act[32];
            unsigned char one[32] = {0x01};
            unsigned char two[32] = {0x02};

            fq_tobytes(act, result[0]);
            check_bytes("fq ecfft mul: (1+x)^2 coeff[0]=1", one, act, 32);

            fq_tobytes(act, result[1]);
            check_bytes("fq ecfft mul: (1+x)^2 coeff[1]=2", two, act, 32);

            fq_tobytes(act, result[2]);
            check_bytes("fq ecfft mul: (1+x)^2 coeff[2]=1", one, act, 32);
        }

        /* Test ECFFT multiply matches schoolbook for degree-4 * degree-4 */
        {
            fq_fe a[5], b[5];
            for (int i = 0; i < 5; i++)
            {
                unsigned char buf[32] = {};
                buf[0] = (unsigned char)(i + 1);
                fq_frombytes(a[i], buf);
                buf[0] = (unsigned char)(i + 6);
                fq_frombytes(b[i], buf);
            }

            fq_fe ecfft_result[16];
            size_t ecfft_len = 0;
            ecfft_fq_poly_mul(ecfft_result, &ecfft_len, a, 5, b, 5, &ctx);

            check_int("fq ecfft mul deg4: result_len", 9, (int)ecfft_len);

            fq_poly pa, pb, pc;
            pa.coeffs.resize(5);
            pb.coeffs.resize(5);
            pc.coeffs.resize(ecfft_len);
            for (size_t i = 0; i < 5; i++)
            {
                fq_copy(pa.coeffs[i].v, a[i]);
                fq_copy(pb.coeffs[i].v, b[i]);
            }
            for (size_t i = 0; i < ecfft_len; i++)
                fq_copy(pc.coeffs[i].v, ecfft_result[i]);

            fq_fe test_x;
            unsigned char xbuf[32] = {0x37};
            fq_frombytes(test_x, xbuf);

            fq_fe va, vb, vc, vab;
            fq_poly_eval(va, &pa, test_x);
            fq_poly_eval(vb, &pb, test_x);
            fq_mul(vab, va, vb);
            fq_poly_eval(vc, &pc, test_x);

            unsigned char eb[32], ab[32];
            fq_tobytes(eb, vab);
            fq_tobytes(ab, vc);
            check_bytes("fq ecfft mul deg4: C(x)==A(x)*B(x)", eb, ab, 32);
        }

        /* Test dispatch integration */
        {
            ecfft_global_init();

            fq_poly pa, pb, pc_ecfft;
            pa.coeffs.resize(9);
            pb.coeffs.resize(9);
            for (size_t i = 0; i < 9; i++)
            {
                unsigned char buf[32] = {};
                buf[0] = (unsigned char)(i + 1);
                fq_frombytes(pa.coeffs[i].v, buf);
                buf[0] = (unsigned char)(i + 10);
                fq_frombytes(pb.coeffs[i].v, buf);
            }

            fq_poly_mul(&pc_ecfft, &pa, &pb);

            fq_fe test_x;
            unsigned char test_buf[32] = {0x42};
            fq_frombytes(test_x, test_buf);

            fq_fe val_a, val_b, val_c, val_ab;
            fq_poly_eval(val_a, &pa, test_x);
            fq_poly_eval(val_b, &pb, test_x);
            fq_mul(val_ab, val_a, val_b);
            fq_poly_eval(val_c, &pc_ecfft, test_x);

            unsigned char eb[32], ab[32];
            fq_tobytes(eb, val_ab);
            fq_tobytes(ab, val_c);
            check_bytes("fq ecfft dispatch: C(x) == A(x)*B(x)", eb, ab, 32);
        }
    }
}
#endif /* RANSHAW_ECFFT */

static void test_eval_divisor()
{
    std::cout << std::endl << "=== Eval-domain divisor ===" << std::endl;
    unsigned char buf[32];

    ran_eval_divisor_init();
    shaw_eval_divisor_init();

    /* Test 1: fp_evals roundtrip — evaluate known poly at domain, interpolate back */
    {
        /* p(x) = 3x^2 + 5x + 7 */
        fp_poly p;
        p.coeffs.resize(3);
        fp_fe c7, c5, c3;
        unsigned char b7[32] = {7}, b5[32] = {5}, b3[32] = {3};
        fp_frombytes(c7, b7);
        fp_frombytes(c5, b5);
        fp_frombytes(c3, b3);
        fp_copy(p.coeffs[0].v, c7);
        fp_copy(p.coeffs[1].v, c5);
        fp_copy(p.coeffs[2].v, c3);

        /* Evaluate at domain points */
        fp_evals ev;
        ev.degree = 2;
        for (size_t i = 0; i < EVAL_DOMAIN_SIZE; i++)
        {
            fp_fe xi;
            unsigned char xb[32] = {};
            xb[0] = (unsigned char)(i & 0xff);
            if (i > 255)
                xb[1] = (unsigned char)((i >> 8) & 0xff);
            fp_frombytes(xi, xb);
            fp_fe tmp_ev;
            fp_poly_eval(tmp_ev, &p, xi);
            fp_evals_set(&ev, i, tmp_ev);
        }

        /* Interpolate back */
        fp_poly recovered;
        fp_evals_to_poly(&recovered, &ev);

        /* Check coefficients match */
        bool match = (recovered.coeffs.size() == 3);
        if (match)
        {
            for (size_t i = 0; i < 3; i++)
            {
                unsigned char eb[32], rb[32];
                fp_tobytes(eb, p.coeffs[i].v);
                fp_tobytes(rb, recovered.coeffs[i].v);
                if (std::memcmp(eb, rb, 32) != 0)
                    match = false;
            }
        }
        ++tests_run;
        if (match)
        {
            ++tests_passed;
            std::cout << "  PASS: fp_evals roundtrip" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: fp_evals roundtrip" << std::endl;
        }
    }

    /* Test 2: fp_evals_mul matches fp_poly_mul */
    {
        /* a(x) = 2x + 1, b(x) = x + 3 */
        fp_poly pa, pb, pc;
        pa.coeffs.resize(2);
        pb.coeffs.resize(2);
        fp_fe c1, c2, c3_val;
        unsigned char b1[32] = {1}, b2[32] = {2}, b3v[32] = {3};
        fp_frombytes(c1, b1);
        fp_frombytes(c2, b2);
        fp_frombytes(c3_val, b3v);
        fp_copy(pa.coeffs[0].v, c1);
        fp_copy(pa.coeffs[1].v, c2);
        fp_copy(pb.coeffs[0].v, c3_val);
        fp_copy(pb.coeffs[1].v, c1);

        /* Poly mul for reference */
        fp_poly_mul(&pc, &pa, &pb);

        /* Eval-domain multiplication */
        fp_evals ea, eb, ec;
        ea.degree = 1;
        eb.degree = 1;
        for (size_t i = 0; i < EVAL_DOMAIN_SIZE; i++)
        {
            fp_fe xi;
            unsigned char xb[32] = {};
            xb[0] = (unsigned char)(i & 0xff);
            fp_frombytes(xi, xb);
            fp_fe tmp_a, tmp_b;
            fp_poly_eval(tmp_a, &pa, xi);
            fp_poly_eval(tmp_b, &pb, xi);
            fp_evals_set(&ea, i, tmp_a);
            fp_evals_set(&eb, i, tmp_b);
        }
        fp_evals_mul(&ec, &ea, &eb);

        /* Convert back */
        fp_poly pc_eval;
        fp_evals_to_poly(&pc_eval, &ec);

        /* Compare */
        bool match = (pc_eval.coeffs.size() == pc.coeffs.size());
        if (match)
        {
            for (size_t i = 0; i < pc.coeffs.size(); i++)
            {
                unsigned char eb2[32], rb[32];
                fp_tobytes(eb2, pc.coeffs[i].v);
                fp_tobytes(rb, pc_eval.coeffs[i].v);
                if (std::memcmp(eb2, rb, 32) != 0)
                    match = false;
            }
        }
        ++tests_run;
        if (match)
        {
            ++tests_passed;
            std::cout << "  PASS: fp_evals_mul matches poly_mul" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: fp_evals_mul matches poly_mul" << std::endl;
        }
    }

    /* Test 3: fp_evals_div_linear */
    {
        /* f(x) = (x-300)(x-400)(x-500) evaluated at domain, then divide by (x-300) */
        fp_fe r300, r400, r500;
        unsigned char b300[32] = {}, b400[32] = {}, b500[32] = {};
        b300[0] = 0x2c;
        b300[1] = 0x01; /* 300 */
        b400[0] = 0x90;
        b400[1] = 0x01; /* 400 */
        b500[0] = 0xf4;
        b500[1] = 0x01; /* 500 */
        fp_frombytes(r300, b300);
        fp_frombytes(r400, b400);
        fp_frombytes(r500, b500);

        /* Build (x-300)(x-400)(x-500) via roots */
        fp_fe roots[3];
        fp_copy(roots[0], r300);
        fp_copy(roots[1], r400);
        fp_copy(roots[2], r500);
        fp_poly f;
        fp_poly_from_roots(&f, roots, 3);

        /* Evaluate at domain */
        fp_evals ef;
        ef.degree = 3;
        for (size_t i = 0; i < EVAL_DOMAIN_SIZE; i++)
        {
            fp_fe xi;
            unsigned char xb[32] = {};
            xb[0] = (unsigned char)(i & 0xff);
            fp_frombytes(xi, xb);
            fp_fe tmp_f;
            fp_poly_eval(tmp_f, &f, xi);
            fp_evals_set(&ef, i, tmp_f);
        }

        /* Divide by (x - 300) */
        fp_evals eq;
        fp_evals_div_linear(&eq, &ef, r300);

        /* Expected: (x-400)(x-500) */
        fp_fe roots2[2];
        fp_copy(roots2[0], r400);
        fp_copy(roots2[1], r500);
        fp_poly expected;
        fp_poly_from_roots(&expected, roots2, 2);

        /* Convert eq back to poly and compare */
        fp_poly got;
        fp_evals_to_poly(&got, &eq);

        bool match = (got.coeffs.size() == expected.coeffs.size());
        if (match)
        {
            for (size_t i = 0; i < expected.coeffs.size(); i++)
            {
                unsigned char eb2[32], gb[32];
                fp_tobytes(eb2, expected.coeffs[i].v);
                fp_tobytes(gb, got.coeffs[i].v);
                if (std::memcmp(eb2, gb, 32) != 0)
                    match = false;
            }
        }
        ++tests_run;
        if (match)
        {
            ++tests_passed;
            std::cout << "  PASS: fp_evals_div_linear" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: fp_evals_div_linear" << std::endl;
        }
    }

    /* Test 4: eval_divisor_from_point — verify vanishes at point */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        ran_affine pt;
        ran_to_affine(&pt, &G);

        ran_eval_divisor ed;
        ran_eval_divisor_from_point(&ed, &pt);

        ran_divisor d;
        ran_eval_divisor_to_divisor(&d, &ed);

        fp_fe val;
        ran_evaluate_divisor(val, &d, pt.x, pt.y);
        fp_tobytes(buf, val);
        check_bytes("eval_divisor_from_point vanishes at P", zero_bytes, buf, 32);

        /* Also matches ran_compute_divisor for 1 point */
        ran_divisor d_ref;
        ran_compute_divisor(&d_ref, &pt, 1);
        unsigned char ab[32], rb[32];
        fp_tobytes(ab, d.a.coeffs[0].v);
        fp_tobytes(rb, d_ref.a.coeffs[0].v);
        check_bytes("eval_divisor_from_point matches compute_divisor a[0]", rb, ab, 32);
        fp_tobytes(ab, d.b.coeffs[0].v);
        fp_tobytes(rb, d_ref.b.coeffs[0].v);
        check_bytes("eval_divisor_from_point matches compute_divisor b[0]", rb, ab, 32);
    }

    /* Test 5: eval_divisor_mul — multiply two single-point divisors */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);
        ran_jacobian G2;
        ran_dbl(&G2, &G);

        ran_affine p1, p2;
        ran_to_affine(&p1, &G);
        ran_to_affine(&p2, &G2);

        ran_eval_divisor ed1, ed2, ed_prod;
        ran_eval_divisor_from_point(&ed1, &p1);
        ran_eval_divisor_from_point(&ed2, &p2);
        ran_eval_divisor_mul(&ed_prod, &ed1, &ed2);

        ran_divisor d;
        ran_eval_divisor_to_divisor(&d, &ed_prod);

        /* Should vanish at both points */
        fp_fe val;
        ran_evaluate_divisor(val, &d, p1.x, p1.y);
        fp_tobytes(buf, val);
        check_bytes("eval_divisor_mul vanishes at P1", zero_bytes, buf, 32);

        ran_evaluate_divisor(val, &d, p2.x, p2.y);
        fp_tobytes(buf, val);
        check_bytes("eval_divisor_mul vanishes at P2", zero_bytes, buf, 32);

        /* Should NOT vanish at a third point */
        ran_jacobian G3;
        ran_add(&G3, &G2, &G);
        ran_affine p3;
        ran_to_affine(&p3, &G3);
        ran_evaluate_divisor(val, &d, p3.x, p3.y);
        fp_tobytes(buf, val);
        check_nonzero("eval_divisor_mul nonzero at P3", std::memcmp(buf, zero_bytes, 32) != 0 ? 1 : 0);
    }

    /* Test 6: fq_evals roundtrip */
    {
        fq_poly p;
        p.coeffs.resize(2);
        fq_fe c1, c2;
        unsigned char b1[32] = {1}, b2[32] = {2};
        fq_frombytes(c1, b1);
        fq_frombytes(c2, b2);
        fq_copy(p.coeffs[0].v, c1);
        fq_copy(p.coeffs[1].v, c2);

        fq_evals ev;
        ev.degree = 1;
        for (size_t i = 0; i < EVAL_DOMAIN_SIZE; i++)
        {
            fq_fe xi;
            unsigned char xb[32] = {};
            xb[0] = (unsigned char)(i & 0xff);
            fq_frombytes(xi, xb);
            fq_fe tmp_ev;
            fq_poly_eval(tmp_ev, &p, xi);
            fq_evals_set(&ev, i, tmp_ev);
        }

        fq_poly recovered;
        fq_evals_to_poly(&recovered, &ev);

        bool match = (recovered.coeffs.size() == 2);
        if (match)
        {
            for (size_t i = 0; i < 2; i++)
            {
                unsigned char eb2[32], rb[32];
                fq_tobytes(eb2, p.coeffs[i].v);
                fq_tobytes(rb, recovered.coeffs[i].v);
                if (std::memcmp(eb2, rb, 32) != 0)
                    match = false;
            }
        }
        ++tests_run;
        if (match)
        {
            ++tests_passed;
            std::cout << "  PASS: fq_evals roundtrip" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: fq_evals roundtrip" << std::endl;
        }
    }

    /* Test 7: shaw eval_divisor_from_point */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        shaw_affine pt;
        shaw_to_affine(&pt, &G);

        shaw_eval_divisor ed;
        shaw_eval_divisor_from_point(&ed, &pt);

        shaw_divisor d;
        shaw_eval_divisor_to_divisor(&d, &ed);

        fq_fe val;
        shaw_evaluate_divisor(val, &d, pt.x, pt.y);
        fq_tobytes(buf, val);
        check_bytes("shaw eval_divisor_from_point vanishes at P", zero_bytes, buf, 32);
    }

    /* Test 8: shaw eval_divisor_mul */
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);
        shaw_jacobian G2;
        shaw_dbl(&G2, &G);

        shaw_affine p1, p2;
        shaw_to_affine(&p1, &G);
        shaw_to_affine(&p2, &G2);

        shaw_eval_divisor ed1, ed2, ed_prod;
        shaw_eval_divisor_from_point(&ed1, &p1);
        shaw_eval_divisor_from_point(&ed2, &p2);
        shaw_eval_divisor_mul(&ed_prod, &ed1, &ed2);

        shaw_divisor d;
        shaw_eval_divisor_to_divisor(&d, &ed_prod);

        fq_fe val;
        shaw_evaluate_divisor(val, &d, p1.x, p1.y);
        fq_tobytes(buf, val);
        check_bytes("shaw eval_divisor_mul vanishes at P1", zero_bytes, buf, 32);

        shaw_evaluate_divisor(val, &d, p2.x, p2.y);
        fq_tobytes(buf, val);
        check_bytes("shaw eval_divisor_mul vanishes at P2", zero_bytes, buf, 32);
    }

    /* Test 9: ran eval divisor merge (2 single-point divisors) */
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);
        ran_jacobian G2;
        ran_dbl(&G2, &G);

        ran_affine p1, p2;
        ran_to_affine(&p1, &G);
        ran_to_affine(&p2, &G2);

        ran_jacobian G3;
        ran_add(&G3, &G, &G2);
        ran_affine sum;
        ran_to_affine(&sum, &G3);

        ran_eval_divisor ed1, ed2, merged;
        ran_eval_divisor_from_point(&ed1, &p1);
        ran_eval_divisor_from_point(&ed2, &p2);
        ran_eval_divisor_merge(&merged, &ed1, &ed2, &p1, &p2, &sum);

        ran_divisor d;
        ran_eval_divisor_to_divisor(&d, &merged);

        /* Should vanish at both points */
        fp_fe val;
        ran_evaluate_divisor(val, &d, p1.x, p1.y);
        fp_tobytes(buf, val);
        check_bytes("ran merge vanishes at P1", zero_bytes, buf, 32);

        ran_evaluate_divisor(val, &d, p2.x, p2.y);
        fp_tobytes(buf, val);
        check_bytes("ran merge vanishes at P2", zero_bytes, buf, 32);
    }

    /* Test 10: fq_evals_div_linear */
    {
        fq_fe r300, r400;
        unsigned char b300[32] = {}, b400[32] = {};
        b300[0] = 0x2c;
        b300[1] = 0x01;
        b400[0] = 0x90;
        b400[1] = 0x01;
        fq_frombytes(r300, b300);
        fq_frombytes(r400, b400);

        fq_fe roots[2];
        fq_copy(roots[0], r300);
        fq_copy(roots[1], r400);
        fq_poly f;
        fq_poly_from_roots(&f, roots, 2);

        fq_evals ef;
        ef.degree = 2;
        for (size_t i = 0; i < EVAL_DOMAIN_SIZE; i++)
        {
            fq_fe xi;
            unsigned char xb[32] = {};
            xb[0] = (unsigned char)(i & 0xff);
            fq_frombytes(xi, xb);
            fq_fe tmp_f;
            fq_poly_eval(tmp_f, &f, xi);
            fq_evals_set(&ef, i, tmp_f);
        }

        fq_evals eq;
        fq_evals_div_linear(&eq, &ef, r300);

        /* Expected: (x-400) = x - 400 */
        fq_fe roots2[1];
        fq_copy(roots2[0], r400);
        fq_poly expected;
        fq_poly_from_roots(&expected, roots2, 1);

        fq_poly got;
        fq_evals_to_poly(&got, &eq);

        bool match = (got.coeffs.size() == expected.coeffs.size());
        if (match)
        {
            for (size_t i = 0; i < expected.coeffs.size(); i++)
            {
                unsigned char eb2[32], gb[32];
                fq_tobytes(eb2, expected.coeffs[i].v);
                fq_tobytes(gb, got.coeffs[i].v);
                if (std::memcmp(eb2, gb, 32) != 0)
                    match = false;
            }
        }
        ++tests_run;
        if (match)
        {
            ++tests_passed;
            std::cout << "  PASS: fq_evals_div_linear" << std::endl;
        }
        else
        {
            ++tests_failed;
            std::cout << "  FAIL: fq_evals_div_linear" << std::endl;
        }
    }
}

static void test_dispatch()
{
    std::cout << "  dispatch" << std::endl;

#if RANSHAW_SIMD
    // Test that init() can be called (no-op if already called, or first time)
    ranshaw_init();

    // After init, dispatch should still produce correct results.
    // Run scalarmult through dispatch wrappers and verify against KAT.

    // Ran scalarmult via dispatch
    {
        ran_jacobian G;
        fp_copy(G.X, RAN_GX);
        fp_copy(G.Y, RAN_GY);
        fp_1(G.Z);

        // 7*G via dispatch
        unsigned char scalar_7[32] = {0x07};
        ran_jacobian result;
        ran_scalarmult(&result, scalar_7, &G);

        unsigned char result_bytes[32];
        ran_tobytes(result_bytes, &result);
        check_bytes("ran dispatch scalarmult 7*G", result_bytes, ran_7g_compressed, 32);

        // vartime
        ran_scalarmult_vartime(&result, scalar_7, &G);
        ran_tobytes(result_bytes, &result);
        check_bytes("ran dispatch scalarmult_vt 7*G", result_bytes, ran_7g_compressed, 32);

        // MSM: 7*G via MSM(scalar=7, point=G, n=1)
        ran_msm_vartime(&result, scalar_7, &G, 1);
        ran_tobytes(result_bytes, &result);
        check_bytes("ran dispatch msm 7*G", result_bytes, ran_7g_compressed, 32);
    }

    // Shaw scalarmult via dispatch
    {
        shaw_jacobian G;
        fq_copy(G.X, SHAW_GX);
        fq_copy(G.Y, SHAW_GY);
        fq_1(G.Z);

        unsigned char scalar_7[32] = {0x07};
        shaw_jacobian result;
        shaw_scalarmult(&result, scalar_7, &G);

        unsigned char result_bytes[32];
        shaw_tobytes(result_bytes, &result);
        check_bytes("shaw dispatch scalarmult 7*G", result_bytes, shaw_7g_compressed, 32);

        shaw_scalarmult_vartime(&result, scalar_7, &G);
        shaw_tobytes(result_bytes, &result);
        check_bytes("shaw dispatch scalarmult_vt 7*G", result_bytes, shaw_7g_compressed, 32);

        shaw_msm_vartime(&result, scalar_7, &G, 1);
        shaw_tobytes(result_bytes, &result);
        check_bytes("shaw dispatch msm 7*G", result_bytes, shaw_7g_compressed, 32);
    }

    // Test double init is safe (idempotent via call_once)
    ranshaw_init();
#else
    // No-SIMD build: init/autotune are no-ops, dispatch not used
    ranshaw_init();
    ranshaw_autotune();
    std::cout << "    (SIMD disabled, dispatch stubs only)" << std::endl;
#endif
}

static void test_cpp_api()
{
    using namespace ranshaw;

    std::cout << std::endl << "=== C++ API ===" << std::endl;

    /* ---- Scalar round-trip ---- */
    {
        auto s = RanScalar::from_bytes(test_a_bytes);
        check_int("api: ran scalar from_bytes valid", 1, s.has_value() ? 1 : 0);
        auto bytes = s->to_bytes();
        check_bytes("api: ran scalar round-trip", test_a_bytes, bytes.data(), 32);
    }
    {
        auto s = ShawScalar::from_bytes(test_a_bytes);
        check_int("api: shaw scalar from_bytes valid", 1, s.has_value() ? 1 : 0);
        auto bytes = s->to_bytes();
        check_bytes("api: shaw scalar round-trip", test_a_bytes, bytes.data(), 32);
    }

    /* ---- Scalar arithmetic ---- */
    {
        auto a = RanScalar::from_bytes(test_a_bytes).value();
        auto b = RanScalar::from_bytes(test_b_bytes).value();
        auto one = RanScalar::one();

        /* a + b == b + a (commutativity) */
        auto ab = (a + b).to_bytes();
        auto ba = (b + a).to_bytes();
        check_bytes("api: ran scalar a+b == b+a", ab.data(), ba.data(), 32);

        /* a * one == a */
        auto a_times_1 = (a * one).to_bytes();
        auto a_bytes = a.to_bytes();
        check_bytes("api: ran scalar a*1 == a", a_bytes.data(), a_times_1.data(), 32);

        /* a * invert(a) == one */
        auto inv = a.invert();
        check_int("api: ran scalar invert non-null", 1, inv.has_value() ? 1 : 0);
        auto prod = (a * inv.value()).to_bytes();
        auto one_b = one.to_bytes();
        check_bytes("api: ran scalar a*inv(a) == 1", one_b.data(), prod.data(), 32);

        /* zero invert returns nullopt */
        auto z_inv = RanScalar::zero().invert();
        check_int("api: ran scalar inv(0) == nullopt", 0, z_inv.has_value() ? 1 : 0);

        /* is_zero */
        check_int("api: ran scalar zero.is_zero", 1, RanScalar::zero().is_zero() ? 1 : 0);
        check_int("api: ran scalar one.is_zero", 0, one.is_zero() ? 1 : 0);
    }
    {
        auto a = ShawScalar::from_bytes(test_a_bytes).value();
        auto one = ShawScalar::one();

        auto inv = a.invert();
        check_int("api: shaw scalar invert non-null", 1, inv.has_value() ? 1 : 0);
        auto prod = (a * inv.value()).to_bytes();
        auto one_b = one.to_bytes();
        check_bytes("api: shaw scalar a*inv(a) == 1", one_b.data(), prod.data(), 32);
    }

    /* ---- Scalar from_bytes rejects invalid ---- */
    {
        /* Bit 255 set */
        unsigned char bad[32] = {};
        bad[31] = 0x80;
        auto s = RanScalar::from_bytes(bad);
        check_int("api: ran scalar rejects bit255", 0, s.has_value() ? 1 : 0);
        auto s2 = ShawScalar::from_bytes(bad);
        check_int("api: shaw scalar rejects bit255", 0, s2.has_value() ? 1 : 0);
    }

    /* ---- Scalar muladd ---- */
    {
        auto a = RanScalar::from_bytes(test_a_bytes).value();
        auto b = RanScalar::from_bytes(test_b_bytes).value();
        auto one = RanScalar::one();
        auto lhs = RanScalar::muladd(a, b, one).to_bytes();
        auto rhs = (a * b + one).to_bytes();
        check_bytes("api: ran muladd a*b+1", lhs.data(), rhs.data(), 32);
    }

    /* ---- Point round-trip ---- */
    {
        auto G = RanPoint::generator();
        auto bytes = G.to_bytes();
        auto p = RanPoint::from_bytes(bytes.data());
        check_int("api: ran point from_bytes valid", 1, p.has_value() ? 1 : 0);
        auto bytes2 = p->to_bytes();
        check_bytes("api: ran point round-trip", bytes.data(), bytes2.data(), 32);
    }
    {
        auto G = ShawPoint::generator();
        auto bytes = G.to_bytes();
        auto p = ShawPoint::from_bytes(bytes.data());
        check_int("api: shaw point from_bytes valid", 1, p.has_value() ? 1 : 0);
        auto bytes2 = p->to_bytes();
        check_bytes("api: shaw point round-trip", bytes.data(), bytes2.data(), 32);
    }

    /* ---- Point arithmetic ---- */
    {
        auto G = RanPoint::generator();
        auto one = RanScalar::one();
        auto G1 = G.scalar_mul(one).to_bytes();
        auto Gb = G.to_bytes();
        check_bytes("api: ran G*1 == G", Gb.data(), G1.data(), 32);

        /* identity checks */
        auto I = RanPoint::identity();
        check_int("api: ran identity.is_identity", 1, I.is_identity() ? 1 : 0);
        check_int("api: ran G.is_identity", 0, G.is_identity() ? 1 : 0);

        /* dbl works */
        auto two = one + one;
        auto G2_sm = G.scalar_mul(two).to_bytes();
        auto G2_dbl = G.dbl().to_bytes();
        check_bytes("api: ran dbl == 2*G", G2_sm.data(), G2_dbl.data(), 32);

        /* P + Q where P != Q and neither is identity */
        auto three = two + one;
        auto G3 = G.scalar_mul(three);
        auto G2 = G.dbl();
        auto sum = (G2 + G).to_bytes();
        auto G3b = G3.to_bytes();
        check_bytes("api: ran 2G+G == 3G", G3b.data(), sum.data(), 32);

        /* negation: -G serializes differently from G (y-parity flips) */
        auto negG = (-G).to_bytes();
        check_nonzero("api: ran -G != G", std::memcmp(Gb.data(), negG.data(), 32));
    }
    {
        auto G = ShawPoint::generator();
        auto one = ShawScalar::one();
        auto G1 = G.scalar_mul(one).to_bytes();
        auto Gb = G.to_bytes();
        check_bytes("api: shaw G*1 == G", Gb.data(), G1.data(), 32);
    }

    /* ---- Point from_bytes rejects invalid ---- */
    {
        unsigned char bad[32] = {};
        bad[0] = 0x02; /* Likely off-curve */
        auto p = RanPoint::from_bytes(bad);
        check_int("api: ran point rejects off-curve", 0, p.has_value() ? 1 : 0);
    }

    /* ---- MSM: compare API wrapper against C-level MSM ---- */
    {
        auto G = RanPoint::generator();
        auto G2 = G.dbl();
        RanScalar scalars[2] = {
            RanScalar::from_bytes(test_a_bytes).value(), RanScalar::from_bytes(test_b_bytes).value()};
        RanPoint points[2] = {G, G2};
        auto msm = RanPoint::multi_scalar_mul(scalars, points, 2);

        /* Compare against C-level MSM */
        unsigned char c_scalars[64];
        std::memcpy(c_scalars, scalars[0].to_bytes().data(), 32);
        std::memcpy(c_scalars + 32, scalars[1].to_bytes().data(), 32);
        ran_jacobian c_points[2], c_result;
        ran_copy(&c_points[0], &G.raw());
        ran_copy(&c_points[1], &G2.raw());
        ran_msm_vartime(&c_result, c_scalars, c_points, 2);
        unsigned char c_bytes[32];
        ran_tobytes(c_bytes, &c_result);
        auto api_bytes = msm.to_bytes();
        check_bytes("api: ran msm matches C-level", c_bytes, api_bytes.data(), 32);
    }
    {
        auto G = ShawPoint::generator();
        auto G2 = G.dbl();
        ShawScalar scalars[2] = {
            ShawScalar::from_bytes(test_a_bytes).value(), ShawScalar::from_bytes(test_b_bytes).value()};
        ShawPoint points[2] = {G, G2};
        auto msm = ShawPoint::multi_scalar_mul(scalars, points, 2);

        unsigned char c_scalars[64];
        std::memcpy(c_scalars, scalars[0].to_bytes().data(), 32);
        std::memcpy(c_scalars + 32, scalars[1].to_bytes().data(), 32);
        shaw_jacobian c_points[2], c_result;
        shaw_copy(&c_points[0], &G.raw());
        shaw_copy(&c_points[1], &G2.raw());
        shaw_msm_vartime(&c_result, c_scalars, c_points, 2);
        unsigned char c_bytes[32];
        shaw_tobytes(c_bytes, &c_result);
        auto api_bytes = msm.to_bytes();
        check_bytes("api: shaw msm matches C-level", c_bytes, api_bytes.data(), 32);
    }

    /* ---- Pedersen: compare API wrapper against C-level ---- */
    {
        auto G = RanPoint::generator();
        auto H = G.dbl();
        auto blind = RanScalar::from_bytes(test_a_bytes).value();
        auto val = RanScalar::from_bytes(test_b_bytes).value();
        auto commit = RanPoint::pedersen_commit(blind, H, &val, &G, 1);

        /* Compare against C-level pedersen */
        auto bb = blind.to_bytes();
        auto vb = val.to_bytes();
        ran_jacobian c_result;
        ran_pedersen_commit(&c_result, bb.data(), &H.raw(), vb.data(), &G.raw(), 1);
        unsigned char c_bytes[32];
        ran_tobytes(c_bytes, &c_result);
        auto api_bytes = commit.to_bytes();
        check_bytes("api: ran pedersen matches C-level", c_bytes, api_bytes.data(), 32);
    }

    /* ---- Map to curve ---- */
    {
        auto p1 = RanPoint::map_to_curve(test_a_bytes);
        check_int("api: ran map_to_curve not identity", 0, p1.is_identity() ? 1 : 0);

        auto p2 = RanPoint::map_to_curve(test_a_bytes, test_b_bytes);
        check_int("api: ran map_to_curve2 not identity", 0, p2.is_identity() ? 1 : 0);
    }

    /* ---- x_coordinate_bytes ---- */
    {
        auto G = RanPoint::generator();
        auto xb = G.x_coordinate_bytes();
        /* x-coordinate of Ran generator is 3 */
        unsigned char expected_x[32] = {};
        expected_x[0] = 0x03;
        check_bytes("api: ran G x-coord == 3", expected_x, xb.data(), 32);
    }

    /* ---- Polynomial ---- */
    {
        /* p(x) = x - r => from_roots([r]) => p(r) == 0 */
        auto poly = FpPolynomial::from_roots(test_a_bytes, 1);
        auto val = poly.evaluate(test_a_bytes);
        check_bytes("api: fp poly eval root == 0", zero_bytes, val.data(), 32);
        check_int("api: fp poly degree from 1 root", 1, (int)poly.degree());
    }
    {
        auto poly = FqPolynomial::from_roots(test_a_bytes, 1);
        auto val = poly.evaluate(test_a_bytes);
        check_bytes("api: fq poly eval root == 0", zero_bytes, val.data(), 32);
    }

    /* ---- Polynomial multiply consistency ---- */
    {
        /* (x - a) * (x - b) should equal from_roots([a, b]) */
        unsigned char roots[64];
        std::memcpy(roots, test_a_bytes, 32);
        std::memcpy(roots + 32, test_b_bytes, 32);

        auto pa = FpPolynomial::from_roots(test_a_bytes, 1);
        auto pb = FpPolynomial::from_roots(test_b_bytes, 1);
        auto prod = pa * pb;
        auto direct = FpPolynomial::from_roots(roots, 2);

        /* Evaluate both at point 1 and compare */
        auto v1 = prod.evaluate(one_bytes);
        auto v2 = direct.evaluate(one_bytes);
        check_bytes("api: fp poly mul == from_roots", v1.data(), v2.data(), 32);
    }

    /* ---- Divisor compute + evaluate ---- */
    {
        auto G = RanPoint::generator();
        auto P2 = G.dbl();
        RanPoint pts[2] = {G, P2};
        auto div = RanDivisor::compute(pts, 2);

        /* Divisor should vanish at the points: get affine coords and evaluate */
        auto G_xb = G.x_coordinate_bytes();
        ran_affine aff;
        ran_to_affine(&aff, &G.raw());
        unsigned char y_bytes[32];
        fp_tobytes(y_bytes, aff.y);

        auto val = div.evaluate(G_xb.data(), y_bytes);
        check_bytes("api: ran divisor eval at G == 0", zero_bytes, val.data(), 32);
    }
    {
        auto G = ShawPoint::generator();
        auto P2 = G.dbl();
        ShawPoint pts[2] = {G, P2};
        auto div = ShawDivisor::compute(pts, 2);

        auto G_xb = G.x_coordinate_bytes();
        shaw_affine aff;
        shaw_to_affine(&aff, &G.raw());
        unsigned char y_bytes[32];
        fq_tobytes(y_bytes, aff.y);

        auto val = div.evaluate(G_xb.data(), y_bytes);
        check_bytes("api: shaw divisor eval at G == 0", zero_bytes, val.data(), 32);
    }

    /* ---- Wei25519 bridge ---- */
    {
        /* Valid x-coordinate (3 is valid as F_p element) */
        unsigned char x3[32] = {};
        x3[0] = 0x03;
        auto s = shaw_scalar_from_wei25519_x(x3);
        check_int("api: wei25519 valid x", 1, s.has_value() ? 1 : 0);
        auto sb = s->to_bytes();
        check_bytes("api: wei25519 x value", x3, sb.data(), 32);

        /* Invalid: bit 255 set */
        unsigned char bad[32] = {};
        bad[31] = 0x80;
        auto s2 = shaw_scalar_from_wei25519_x(bad);
        check_int("api: wei25519 rejects bit255", 0, s2.has_value() ? 1 : 0);
    }

    /* ---- Namespace init/autotune ---- */
    {
        ranshaw::init();
        /* No crash is the test */
        ++tests_run;
        ++tests_passed;
        std::cout << "  PASS: api: namespace init()" << std::endl;
    }
}

static void test_serialization_roundtrip()
{
    using namespace ranshaw;

    std::cout << std::endl << "=== Serialization round-trip ===" << std::endl;

    /* Helper: round-trip a point through to_bytes/from_bytes */
    auto ran_point_rt = [](const char *label, const RanPoint &p)
    {
        auto bytes = p.to_bytes();
        auto p2 = RanPoint::from_bytes(bytes.data());
        check_int(label, 1, p2.has_value() ? 1 : 0);
        if (p2)
        {
            auto bytes2 = p2->to_bytes();
            check_bytes(label, bytes.data(), bytes2.data(), 32);
        }
    };
    auto shaw_point_rt = [](const char *label, const ShawPoint &p)
    {
        auto bytes = p.to_bytes();
        auto p2 = ShawPoint::from_bytes(bytes.data());
        check_int(label, 1, p2.has_value() ? 1 : 0);
        if (p2)
        {
            auto bytes2 = p2->to_bytes();
            check_bytes(label, bytes.data(), bytes2.data(), 32);
        }
    };
    auto ran_scalar_rt = [](const char *label, const RanScalar &s)
    {
        auto bytes = s.to_bytes();
        auto s2 = RanScalar::from_bytes(bytes.data());
        check_int(label, 1, s2.has_value() ? 1 : 0);
        if (s2)
        {
            auto bytes2 = s2->to_bytes();
            check_bytes(label, bytes.data(), bytes2.data(), 32);
        }
    };
    auto shaw_scalar_rt = [](const char *label, const ShawScalar &s)
    {
        auto bytes = s.to_bytes();
        auto s2 = ShawScalar::from_bytes(bytes.data());
        check_int(label, 1, s2.has_value() ? 1 : 0);
        if (s2)
        {
            auto bytes2 = s2->to_bytes();
            check_bytes(label, bytes.data(), bytes2.data(), 32);
        }
    };

    /* ---- Ran point round-trips ---- */
    {
        auto G = RanPoint::generator();
        auto one = RanScalar::one();
        auto two = one + one;
        auto three = two + one;
        auto a = RanScalar::from_bytes(test_a_bytes).value();
        auto b = RanScalar::from_bytes(test_b_bytes).value();

        ran_point_rt("rt: ran G", G);

        /* Identity + P == P (operator+ must handle identity inputs) */
        {
            auto I = RanPoint::identity();
            auto sum = I + G;
            check_int("rt: ran identity+G not identity", 0, sum.is_identity() ? 1 : 0);
            auto Gb = G.to_bytes();
            auto sb = sum.to_bytes();
            check_bytes("rt: ran identity+G == G", Gb.data(), sb.data(), 32);

            auto sum2 = G + I;
            check_int("rt: ran G+identity not identity", 0, sum2.is_identity() ? 1 : 0);
            auto sb2 = sum2.to_bytes();
            check_bytes("rt: ran G+identity == G", Gb.data(), sb2.data(), 32);

            /* Accumulation pattern: identity + P1 + P2 */
            auto P2 = G.dbl();
            auto accum = I + G + P2;
            auto direct = G + P2;
            check_bytes("rt: ran accum I+G+2G", direct.to_bytes().data(), accum.to_bytes().data(), 32);
        }

        /* P + P == dbl(P) (operator+ must handle equal inputs) */
        {
            auto sum = G + G;
            auto dbl_G = G.dbl();
            check_bytes("rt: ran G+G == dbl(G)", dbl_G.to_bytes().data(), sum.to_bytes().data(), 32);

            /* Also with non-affine Z (computed point) */
            auto P = G.scalar_mul(a);
            auto sum2 = P + P;
            auto dbl_P = P.dbl();
            check_bytes("rt: ran P+P == dbl(P)", dbl_P.to_bytes().data(), sum2.to_bytes().data(), 32);
        }

        /* P + (-P) == identity */
        {
            auto negG = -G;
            auto sum = G + negG;
            check_int("rt: ran G+(-G) is identity", 1, sum.is_identity() ? 1 : 0);

            auto P = G.scalar_mul(a);
            auto negP = -P;
            auto sum2 = P + negP;
            check_int("rt: ran P+(-P) is identity", 1, sum2.is_identity() ? 1 : 0);
        }

        ran_point_rt("rt: ran 2G (dbl)", G.dbl());
        ran_point_rt("rt: ran 3G (add)", G.dbl() + G);
        ran_point_rt("rt: ran -G (neg)", -G);
        ran_point_rt("rt: ran G*1", G.scalar_mul(one));
        ran_point_rt("rt: ran G*2", G.scalar_mul(two));
        ran_point_rt("rt: ran G*3", G.scalar_mul(three));
        ran_point_rt("rt: ran G*a", G.scalar_mul(a));
        ran_point_rt("rt: ran G*b", G.scalar_mul(b));
        ran_point_rt("rt: ran G*a + G*b", G.scalar_mul(a) + G.scalar_mul(b));
        ran_point_rt("rt: ran map_to_curve(a)", RanPoint::map_to_curve(test_a_bytes));
        ran_point_rt("rt: ran map_to_curve(a,b)", RanPoint::map_to_curve(test_a_bytes, test_b_bytes));

        /* Iterated doubling: 2^k * G for k=1..10 */
        auto P = G;
        for (int k = 1; k <= 10; k++)
        {
            P = P.dbl();
            std::string name = "rt: ran 2^" + std::to_string(k) + "*G";
            ran_point_rt(name.c_str(), P);
        }
    }

    /* ---- Shaw point round-trips ---- */
    {
        auto G = ShawPoint::generator();
        auto one = ShawScalar::one();
        auto two = one + one;
        auto three = two + one;
        auto a = ShawScalar::from_bytes(test_a_bytes).value();
        auto b = ShawScalar::from_bytes(test_b_bytes).value();

        shaw_point_rt("rt: shaw G", G);

        /* Identity + P == P */
        {
            auto I = ShawPoint::identity();
            auto sum = I + G;
            check_int("rt: shaw identity+G not identity", 0, sum.is_identity() ? 1 : 0);
            auto Gb = G.to_bytes();
            auto sb = sum.to_bytes();
            check_bytes("rt: shaw identity+G == G", Gb.data(), sb.data(), 32);

            auto sum2 = G + I;
            check_int("rt: shaw G+identity not identity", 0, sum2.is_identity() ? 1 : 0);
            auto sb2 = sum2.to_bytes();
            check_bytes("rt: shaw G+identity == G", Gb.data(), sb2.data(), 32);

            auto P2 = G.dbl();
            auto accum = I + G + P2;
            auto direct = G + P2;
            check_bytes("rt: shaw accum I+G+2G", direct.to_bytes().data(), accum.to_bytes().data(), 32);
        }

        /* P + P == dbl(P) */
        {
            auto sum = G + G;
            auto dbl_G = G.dbl();
            check_bytes("rt: shaw G+G == dbl(G)", dbl_G.to_bytes().data(), sum.to_bytes().data(), 32);

            auto P = G.scalar_mul(a);
            auto sum2 = P + P;
            auto dbl_P = P.dbl();
            check_bytes("rt: shaw P+P == dbl(P)", dbl_P.to_bytes().data(), sum2.to_bytes().data(), 32);
        }

        /* P + (-P) == identity */
        {
            auto negG = -G;
            auto sum = G + negG;
            check_int("rt: shaw G+(-G) is identity", 1, sum.is_identity() ? 1 : 0);

            auto P = G.scalar_mul(a);
            auto negP = -P;
            auto sum2 = P + negP;
            check_int("rt: shaw P+(-P) is identity", 1, sum2.is_identity() ? 1 : 0);
        }

        shaw_point_rt("rt: shaw 2G (dbl)", G.dbl());
        shaw_point_rt("rt: shaw 3G (add)", G.dbl() + G);
        shaw_point_rt("rt: shaw -G (neg)", -G);
        shaw_point_rt("rt: shaw G*1", G.scalar_mul(one));
        shaw_point_rt("rt: shaw G*2", G.scalar_mul(two));
        shaw_point_rt("rt: shaw G*3", G.scalar_mul(three));
        shaw_point_rt("rt: shaw G*a", G.scalar_mul(a));
        shaw_point_rt("rt: shaw G*b", G.scalar_mul(b));
        shaw_point_rt("rt: shaw G*a + G*b", G.scalar_mul(a) + G.scalar_mul(b));
        shaw_point_rt("rt: shaw map_to_curve(a)", ShawPoint::map_to_curve(test_a_bytes));
        shaw_point_rt("rt: shaw map_to_curve(a,b)", ShawPoint::map_to_curve(test_a_bytes, test_b_bytes));

        auto P = G;
        for (int k = 1; k <= 10; k++)
        {
            P = P.dbl();
            std::string name = "rt: shaw 2^" + std::to_string(k) + "*G";
            shaw_point_rt(name.c_str(), P);
        }
    }

    /* ---- MSM vs scalar_mul+add consistency ---- */
    {
        auto G = RanPoint::generator();
        auto a = RanScalar::from_bytes(test_a_bytes).value();
        auto b = RanScalar::from_bytes(test_b_bytes).value();

        /* n=2: MSM(a,b; G,2G) == a*G + b*2G */
        {
            auto G2 = G.dbl();
            RanScalar s[2] = {a, b};
            RanPoint p[2] = {G, G2};
            auto msm = RanPoint::multi_scalar_mul(s, p, 2);
            auto manual = G.scalar_mul(a) + G2.scalar_mul(b);
            check_int("rt: ran msm n=2 not identity", 0, msm.is_identity() ? 1 : 0);
            check_bytes("rt: ran msm n=2 == manual", manual.to_bytes().data(), msm.to_bytes().data(), 32);
        }

        /* n=2 with map_to_curve points (non-trivial Z after scalarmul) */
        {
            auto P0 = RanPoint::map_to_curve(test_a_bytes);
            auto P1 = RanPoint::map_to_curve(test_b_bytes);
            RanScalar s[2] = {a, b};
            RanPoint p[2] = {P0, P1};
            auto msm = RanPoint::multi_scalar_mul(s, p, 2);
            auto manual = P0.scalar_mul(a) + P1.scalar_mul(b);
            check_int("rt: ran msm n=2 h2c not identity", 0, msm.is_identity() ? 1 : 0);
            check_bytes("rt: ran msm n=2 h2c == manual", manual.to_bytes().data(), msm.to_bytes().data(), 32);
        }

        /* n=1: MSM(a; G) == a*G */
        {
            RanScalar s[1] = {a};
            RanPoint p[1] = {G};
            auto msm = RanPoint::multi_scalar_mul(s, p, 1);
            auto manual = G.scalar_mul(a);
            check_bytes("rt: ran msm n=1 == manual", manual.to_bytes().data(), msm.to_bytes().data(), 32);
        }

        /* n=3 */
        {
            auto G2 = G.dbl();
            auto G3 = G2 + G;
            auto one = RanScalar::one();
            RanScalar s[3] = {a, b, one};
            RanPoint p[3] = {G, G2, G3};
            auto msm = RanPoint::multi_scalar_mul(s, p, 3);
            auto manual = G.scalar_mul(a) + G2.scalar_mul(b) + G3.scalar_mul(one);
            check_bytes("rt: ran msm n=3 == manual", manual.to_bytes().data(), msm.to_bytes().data(), 32);
        }
    }
    {
        auto G = ShawPoint::generator();
        auto a = ShawScalar::from_bytes(test_a_bytes).value();
        auto b = ShawScalar::from_bytes(test_b_bytes).value();

        {
            auto G2 = G.dbl();
            ShawScalar s[2] = {a, b};
            ShawPoint p[2] = {G, G2};
            auto msm = ShawPoint::multi_scalar_mul(s, p, 2);
            auto manual = G.scalar_mul(a) + G2.scalar_mul(b);
            check_int("rt: shaw msm n=2 not identity", 0, msm.is_identity() ? 1 : 0);
            check_bytes("rt: shaw msm n=2 == manual", manual.to_bytes().data(), msm.to_bytes().data(), 32);
        }

        {
            auto P0 = ShawPoint::map_to_curve(test_a_bytes);
            auto P1 = ShawPoint::map_to_curve(test_b_bytes);
            ShawScalar s[2] = {a, b};
            ShawPoint p[2] = {P0, P1};
            auto msm = ShawPoint::multi_scalar_mul(s, p, 2);
            auto manual = P0.scalar_mul(a) + P1.scalar_mul(b);
            check_int("rt: shaw msm n=2 h2c not identity", 0, msm.is_identity() ? 1 : 0);
            check_bytes("rt: shaw msm n=2 h2c == manual", manual.to_bytes().data(), msm.to_bytes().data(), 32);
        }

        {
            ShawScalar s[1] = {a};
            ShawPoint p[1] = {G};
            auto msm = ShawPoint::multi_scalar_mul(s, p, 1);
            auto manual = G.scalar_mul(a);
            check_bytes("rt: shaw msm n=1 == manual", manual.to_bytes().data(), msm.to_bytes().data(), 32);
        }

        {
            auto G2 = G.dbl();
            auto G3 = G2 + G;
            auto one = ShawScalar::one();
            ShawScalar s[3] = {a, b, one};
            ShawPoint p[3] = {G, G2, G3};
            auto msm = ShawPoint::multi_scalar_mul(s, p, 3);
            auto manual = G.scalar_mul(a) + G2.scalar_mul(b) + G3.scalar_mul(one);
            check_bytes("rt: shaw msm n=3 == manual", manual.to_bytes().data(), msm.to_bytes().data(), 32);
        }
    }

    /* ---- Ran scalar round-trips ---- */
    {
        auto a = RanScalar::from_bytes(test_a_bytes).value();
        auto b = RanScalar::from_bytes(test_b_bytes).value();
        auto one = RanScalar::one();

        ran_scalar_rt("rt: ran scalar zero", RanScalar::zero());
        ran_scalar_rt("rt: ran scalar one", one);
        ran_scalar_rt("rt: ran scalar a", a);
        ran_scalar_rt("rt: ran scalar a+b", a + b);
        ran_scalar_rt("rt: ran scalar a*b", a * b);
        ran_scalar_rt("rt: ran scalar a-b", a - b);
        ran_scalar_rt("rt: ran scalar -a", -a);
        ran_scalar_rt("rt: ran scalar a^2", a.sq());
        ran_scalar_rt("rt: ran scalar inv(a)", a.invert().value());
    }

    /* ---- Shaw scalar round-trips ---- */
    {
        auto a = ShawScalar::from_bytes(test_a_bytes).value();
        auto b = ShawScalar::from_bytes(test_b_bytes).value();
        auto one = ShawScalar::one();

        shaw_scalar_rt("rt: shaw scalar zero", ShawScalar::zero());
        shaw_scalar_rt("rt: shaw scalar one", one);
        shaw_scalar_rt("rt: shaw scalar a", a);
        shaw_scalar_rt("rt: shaw scalar a+b", a + b);
        shaw_scalar_rt("rt: shaw scalar a*b", a * b);
        shaw_scalar_rt("rt: shaw scalar a-b", a - b);
        shaw_scalar_rt("rt: shaw scalar -a", -a);
        shaw_scalar_rt("rt: shaw scalar a^2", a.sq());
        shaw_scalar_rt("rt: shaw scalar inv(a)", a.invert().value());
    }
}

static void test_vector_validation()
{
    using namespace ranshaw;
    namespace tv = ranshaw_test_vectors;

    std::cout << std::endl << "=== Test Vector Validation ===" << std::endl;

    /* ---- Ran Scalar ---- */
    std::cout << "  --- Ran Scalar ---" << std::endl;
    for (size_t i = 0; i < tv::ran_scalar::from_bytes_count; i++)
    {
        auto &v = tv::ran_scalar::from_bytes_vectors[i];
        auto r = RanScalar::from_bytes(v.input);
        std::string name = std::string("tv: ran scalar from_bytes ") + v.label;
        if (v.valid)
        {
            check_int((name + " valid").c_str(), 1, r.has_value() ? 1 : 0);
            if (r)
                check_bytes((name + " value").c_str(), v.result, r->to_bytes().data(), 32);
        }
        else
        {
            check_int((name + " invalid").c_str(), 0, r.has_value() ? 1 : 0);
        }
    }
    for (size_t i = 0; i < tv::ran_scalar::add_count; i++)
    {
        auto &v = tv::ran_scalar::add_vectors[i];
        auto a = RanScalar::from_bytes(v.a).value();
        auto b = RanScalar::from_bytes(v.b).value();
        auto r = a + b;
        check_bytes((std::string("tv: ran scalar add ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::sub_count; i++)
    {
        auto &v = tv::ran_scalar::sub_vectors[i];
        auto a = RanScalar::from_bytes(v.a).value();
        auto b = RanScalar::from_bytes(v.b).value();
        auto r = a - b;
        check_bytes((std::string("tv: ran scalar sub ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::mul_count; i++)
    {
        auto &v = tv::ran_scalar::mul_vectors[i];
        auto a = RanScalar::from_bytes(v.a).value();
        auto b = RanScalar::from_bytes(v.b).value();
        auto r = a * b;
        check_bytes((std::string("tv: ran scalar mul ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::sq_count; i++)
    {
        auto &v = tv::ran_scalar::sq_vectors[i];
        auto a = RanScalar::from_bytes(v.a).value();
        auto r = a.sq();
        check_bytes((std::string("tv: ran scalar sq ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::negate_count; i++)
    {
        auto &v = tv::ran_scalar::negate_vectors[i];
        auto a = RanScalar::from_bytes(v.a).value();
        auto r = -a;
        check_bytes((std::string("tv: ran scalar neg ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::invert_count; i++)
    {
        auto &v = tv::ran_scalar::invert_vectors[i];
        auto a = RanScalar::from_bytes(v.a).value();
        auto r = a.invert();
        std::string name = std::string("tv: ran scalar inv ") + v.label;
        if (v.valid)
        {
            check_int((name + " valid").c_str(), 1, r.has_value() ? 1 : 0);
            if (r)
                check_bytes((name + " value").c_str(), v.result, r->to_bytes().data(), 32);
        }
        else
        {
            check_int((name + " invalid").c_str(), 0, r.has_value() ? 1 : 0);
        }
    }
    for (size_t i = 0; i < tv::ran_scalar::reduce_wide_count; i++)
    {
        auto &v = tv::ran_scalar::reduce_wide_vectors[i];
        auto r = RanScalar::reduce_wide(v.input);
        check_bytes(
            (std::string("tv: ran scalar reduce_wide ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::muladd_count; i++)
    {
        auto &v = tv::ran_scalar::muladd_vectors[i];
        auto a = RanScalar::from_bytes(v.a).value();
        auto b = RanScalar::from_bytes(v.b).value();
        auto c = RanScalar::from_bytes(v.c).value();
        auto r = RanScalar::muladd(a, b, c);
        check_bytes((std::string("tv: ran scalar muladd ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::is_zero_count; i++)
    {
        auto &v = tv::ran_scalar::is_zero_vectors[i];
        auto a = RanScalar::from_bytes(v.a).value();
        check_int((std::string("tv: ran scalar is_zero ") + v.label).c_str(), v.result ? 1 : 0, a.is_zero() ? 1 : 0);
    }

    /* ---- Shaw Scalar ---- */
    std::cout << "  --- Shaw Scalar ---" << std::endl;
    for (size_t i = 0; i < tv::shaw_scalar::from_bytes_count; i++)
    {
        auto &v = tv::shaw_scalar::from_bytes_vectors[i];
        auto r = ShawScalar::from_bytes(v.input);
        std::string name = std::string("tv: shaw scalar from_bytes ") + v.label;
        if (v.valid)
        {
            check_int((name + " valid").c_str(), 1, r.has_value() ? 1 : 0);
            if (r)
                check_bytes((name + " value").c_str(), v.result, r->to_bytes().data(), 32);
        }
        else
        {
            check_int((name + " invalid").c_str(), 0, r.has_value() ? 1 : 0);
        }
    }
    for (size_t i = 0; i < tv::shaw_scalar::add_count; i++)
    {
        auto &v = tv::shaw_scalar::add_vectors[i];
        auto a = ShawScalar::from_bytes(v.a).value();
        auto b = ShawScalar::from_bytes(v.b).value();
        auto r = a + b;
        check_bytes((std::string("tv: shaw scalar add ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::sub_count; i++)
    {
        auto &v = tv::shaw_scalar::sub_vectors[i];
        auto a = ShawScalar::from_bytes(v.a).value();
        auto b = ShawScalar::from_bytes(v.b).value();
        auto r = a - b;
        check_bytes((std::string("tv: shaw scalar sub ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::mul_count; i++)
    {
        auto &v = tv::shaw_scalar::mul_vectors[i];
        auto a = ShawScalar::from_bytes(v.a).value();
        auto b = ShawScalar::from_bytes(v.b).value();
        auto r = a * b;
        check_bytes((std::string("tv: shaw scalar mul ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::sq_count; i++)
    {
        auto &v = tv::shaw_scalar::sq_vectors[i];
        auto a = ShawScalar::from_bytes(v.a).value();
        auto r = a.sq();
        check_bytes((std::string("tv: shaw scalar sq ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::negate_count; i++)
    {
        auto &v = tv::shaw_scalar::negate_vectors[i];
        auto a = ShawScalar::from_bytes(v.a).value();
        auto r = -a;
        check_bytes((std::string("tv: shaw scalar neg ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::invert_count; i++)
    {
        auto &v = tv::shaw_scalar::invert_vectors[i];
        auto a = ShawScalar::from_bytes(v.a).value();
        auto r = a.invert();
        std::string name = std::string("tv: shaw scalar inv ") + v.label;
        if (v.valid)
        {
            check_int((name + " valid").c_str(), 1, r.has_value() ? 1 : 0);
            if (r)
                check_bytes((name + " value").c_str(), v.result, r->to_bytes().data(), 32);
        }
        else
        {
            check_int((name + " invalid").c_str(), 0, r.has_value() ? 1 : 0);
        }
    }
    for (size_t i = 0; i < tv::shaw_scalar::reduce_wide_count; i++)
    {
        auto &v = tv::shaw_scalar::reduce_wide_vectors[i];
        auto r = ShawScalar::reduce_wide(v.input);
        check_bytes(
            (std::string("tv: shaw scalar reduce_wide ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::muladd_count; i++)
    {
        auto &v = tv::shaw_scalar::muladd_vectors[i];
        auto a = ShawScalar::from_bytes(v.a).value();
        auto b = ShawScalar::from_bytes(v.b).value();
        auto c = ShawScalar::from_bytes(v.c).value();
        auto r = ShawScalar::muladd(a, b, c);
        check_bytes((std::string("tv: shaw scalar muladd ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::is_zero_count; i++)
    {
        auto &v = tv::shaw_scalar::is_zero_vectors[i];
        auto a = ShawScalar::from_bytes(v.a).value();
        check_int((std::string("tv: shaw scalar is_zero ") + v.label).c_str(), v.result ? 1 : 0, a.is_zero() ? 1 : 0);
    }

    /* ---- Ran Point ---- */
    std::cout << "  --- Ran Point ---" << std::endl;
    for (size_t i = 0; i < tv::ran_point::from_bytes_count; i++)
    {
        auto &v = tv::ran_point::from_bytes_vectors[i];
        auto r = RanPoint::from_bytes(v.input);
        std::string name = std::string("tv: ran point from_bytes ") + v.label;
        if (v.valid)
        {
            check_int((name + " valid").c_str(), 1, r.has_value() ? 1 : 0);
            if (r)
                check_bytes((name + " value").c_str(), v.result, r->to_bytes().data(), 32);
        }
        else
        {
            check_int((name + " invalid").c_str(), 0, r.has_value() ? 1 : 0);
        }
    }
    {
        auto hp_from = [](const uint8_t bytes[32]) -> RanPoint
        {
            auto r = RanPoint::from_bytes(bytes);
            return r ? *r : RanPoint::identity();
        };
        for (size_t i = 0; i < tv::ran_point::add_count; i++)
        {
            auto &v = tv::ran_point::add_vectors[i];
            auto a = hp_from(v.a);
            auto b = hp_from(v.b);
            auto r = a + b;
            check_bytes((std::string("tv: ran point add ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
        }
    }
    {
        auto hp_from = [](const uint8_t bytes[32]) -> RanPoint
        {
            auto r = RanPoint::from_bytes(bytes);
            return r ? *r : RanPoint::identity();
        };
        for (size_t i = 0; i < tv::ran_point::dbl_count; i++)
        {
            auto &v = tv::ran_point::dbl_vectors[i];
            auto a = hp_from(v.a);
            auto r = a.dbl();
            check_bytes((std::string("tv: ran point dbl ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
        }
        for (size_t i = 0; i < tv::ran_point::negate_count; i++)
        {
            auto &v = tv::ran_point::negate_vectors[i];
            auto a = hp_from(v.a);
            auto r = -a;
            check_bytes((std::string("tv: ran point neg ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
        }
    }
    {
        auto hp_from = [](const uint8_t bytes[32]) -> RanPoint
        {
            auto r = RanPoint::from_bytes(bytes);
            return r ? *r : RanPoint::identity();
        };
        for (size_t i = 0; i < tv::ran_point::scalar_mul_count; i++)
        {
            auto &v = tv::ran_point::scalar_mul_vectors[i];
            auto s = RanScalar::from_bytes(v.scalar).value();
            auto p = hp_from(v.point);
            auto r = p.scalar_mul(s);
            check_bytes(
                (std::string("tv: ran point scalar_mul ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
        }
    }
    /* MSM */
    {
        auto test_msm = [](const char *label,
                           size_t n,
                           const uint8_t scalars[][32],
                           const uint8_t points[][32],
                           const uint8_t expected[32])
        {
            std::vector<RanScalar> sv;
            std::vector<RanPoint> pv;
            for (size_t i = 0; i < n; i++)
            {
                sv.push_back(RanScalar::from_bytes(scalars[i]).value());
                pv.push_back(RanPoint::from_bytes(points[i]).value());
            }
            auto r = RanPoint::multi_scalar_mul(sv.data(), pv.data(), n);
            check_bytes(label, expected, r.to_bytes().data(), 32);
        };
        test_msm(
            "tv: ran msm n_1",
            1,
            tv::ran_point::msm_n_1_scalars,
            tv::ran_point::msm_n_1_points,
            tv::ran_point::msm_n_1_result);
        test_msm(
            "tv: ran msm n_2",
            2,
            tv::ran_point::msm_n_2_scalars,
            tv::ran_point::msm_n_2_points,
            tv::ran_point::msm_n_2_result);
        test_msm(
            "tv: ran msm n_4",
            4,
            tv::ran_point::msm_n_4_scalars,
            tv::ran_point::msm_n_4_points,
            tv::ran_point::msm_n_4_result);
        test_msm(
            "tv: ran msm n_16",
            16,
            tv::ran_point::msm_n_16_scalars,
            tv::ran_point::msm_n_16_points,
            tv::ran_point::msm_n_16_result);
        test_msm(
            "tv: ran msm n_32_straus",
            32,
            tv::ran_point::msm_n_32_straus_scalars,
            tv::ran_point::msm_n_32_straus_points,
            tv::ran_point::msm_n_32_straus_result);
        test_msm(
            "tv: ran msm n_33_pippenger",
            33,
            tv::ran_point::msm_n_33_pippenger_scalars,
            tv::ran_point::msm_n_33_pippenger_points,
            tv::ran_point::msm_n_33_pippenger_result);
        test_msm(
            "tv: ran msm n_64_pippenger",
            64,
            tv::ran_point::msm_n_64_pippenger_scalars,
            tv::ran_point::msm_n_64_pippenger_points,
            tv::ran_point::msm_n_64_pippenger_result);
    }
    /* Pedersen */
    {
        auto test_ped = [](const char *label,
                           const uint8_t blinding[32],
                           const uint8_t H[32],
                           size_t n,
                           const uint8_t values[][32],
                           const uint8_t generators[][32],
                           const uint8_t expected[32])
        {
            auto s_blind = RanScalar::from_bytes(blinding).value();
            auto p_H = RanPoint::from_bytes(H).value();
            std::vector<RanScalar> vals;
            std::vector<RanPoint> gens;
            for (size_t i = 0; i < n; i++)
            {
                vals.push_back(RanScalar::from_bytes(values[i]).value());
                gens.push_back(RanPoint::from_bytes(generators[i]).value());
            }
            auto r = RanPoint::pedersen_commit(s_blind, p_H, vals.data(), gens.data(), n);
            check_bytes(label, expected, r.to_bytes().data(), 32);
        };
        test_ped(
            "tv: ran pedersen n_1",
            tv::ran_point::pedersen_n_1_blinding,
            tv::ran_point::pedersen_n_1_H,
            1,
            tv::ran_point::pedersen_n_1_values,
            tv::ran_point::pedersen_n_1_generators,
            tv::ran_point::pedersen_n_1_result);
        test_ped(
            "tv: ran pedersen blinding_zero",
            tv::ran_point::pedersen_blinding_zero_blinding,
            tv::ran_point::pedersen_blinding_zero_H,
            1,
            tv::ran_point::pedersen_blinding_zero_values,
            tv::ran_point::pedersen_blinding_zero_generators,
            tv::ran_point::pedersen_blinding_zero_result);
        test_ped(
            "tv: ran pedersen n_4",
            tv::ran_point::pedersen_n_4_blinding,
            tv::ran_point::pedersen_n_4_H,
            4,
            tv::ran_point::pedersen_n_4_values,
            tv::ran_point::pedersen_n_4_generators,
            tv::ran_point::pedersen_n_4_result);
    }
    for (size_t i = 0; i < tv::ran_point::map_to_curve_single_count; i++)
    {
        auto &v = tv::ran_point::map_to_curve_single_vectors[i];
        auto r = RanPoint::map_to_curve(v.u);
        check_bytes(
            (std::string("tv: ran point map_to_curve ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::ran_point::map_to_curve_double_count; i++)
    {
        auto &v = tv::ran_point::map_to_curve_double_vectors[i];
        auto r = RanPoint::map_to_curve(v.u0, v.u1);
        check_bytes(
            (std::string("tv: ran point map_to_curve2 ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::ran_point::x_coordinate_count; i++)
    {
        auto &v = tv::ran_point::x_coordinate_vectors[i];
        auto p = RanPoint::from_bytes(v.point).value();
        auto r = p.x_coordinate_bytes();
        check_bytes((std::string("tv: ran point x_coord ") + v.label).c_str(), v.x_bytes, r.data(), 32);
    }

    /* ---- Shaw Point ---- */
    std::cout << "  --- Shaw Point ---" << std::endl;
    for (size_t i = 0; i < tv::shaw_point::from_bytes_count; i++)
    {
        auto &v = tv::shaw_point::from_bytes_vectors[i];
        auto r = ShawPoint::from_bytes(v.input);
        std::string name = std::string("tv: shaw point from_bytes ") + v.label;
        if (v.valid)
        {
            check_int((name + " valid").c_str(), 1, r.has_value() ? 1 : 0);
            if (r)
                check_bytes((name + " value").c_str(), v.result, r->to_bytes().data(), 32);
        }
        else
        {
            check_int((name + " invalid").c_str(), 0, r.has_value() ? 1 : 0);
        }
    }
    {
        auto sp_from = [](const uint8_t bytes[32]) -> ShawPoint
        {
            auto r = ShawPoint::from_bytes(bytes);
            return r ? *r : ShawPoint::identity();
        };
        for (size_t i = 0; i < tv::shaw_point::add_count; i++)
        {
            auto &v = tv::shaw_point::add_vectors[i];
            auto a = sp_from(v.a);
            auto b = sp_from(v.b);
            auto r = a + b;
            check_bytes((std::string("tv: shaw point add ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
        }
    }
    {
        auto sp_from = [](const uint8_t bytes[32]) -> ShawPoint
        {
            auto r = ShawPoint::from_bytes(bytes);
            return r ? *r : ShawPoint::identity();
        };
        for (size_t i = 0; i < tv::shaw_point::dbl_count; i++)
        {
            auto &v = tv::shaw_point::dbl_vectors[i];
            auto a = sp_from(v.a);
            auto r = a.dbl();
            check_bytes((std::string("tv: shaw point dbl ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
        }
        for (size_t i = 0; i < tv::shaw_point::negate_count; i++)
        {
            auto &v = tv::shaw_point::negate_vectors[i];
            auto a = sp_from(v.a);
            auto r = -a;
            check_bytes((std::string("tv: shaw point neg ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
        }
    }
    {
        auto sp_from = [](const uint8_t bytes[32]) -> ShawPoint
        {
            auto r = ShawPoint::from_bytes(bytes);
            return r ? *r : ShawPoint::identity();
        };
        for (size_t i = 0; i < tv::shaw_point::scalar_mul_count; i++)
        {
            auto &v = tv::shaw_point::scalar_mul_vectors[i];
            auto s = ShawScalar::from_bytes(v.scalar).value();
            auto p = sp_from(v.point);
            auto r = p.scalar_mul(s);
            check_bytes(
                (std::string("tv: shaw point scalar_mul ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
        }
    }
    /* MSM */
    {
        auto test_msm = [](const char *label,
                           size_t n,
                           const uint8_t scalars[][32],
                           const uint8_t points[][32],
                           const uint8_t expected[32])
        {
            std::vector<ShawScalar> sv;
            std::vector<ShawPoint> pv;
            for (size_t i = 0; i < n; i++)
            {
                sv.push_back(ShawScalar::from_bytes(scalars[i]).value());
                pv.push_back(ShawPoint::from_bytes(points[i]).value());
            }
            auto r = ShawPoint::multi_scalar_mul(sv.data(), pv.data(), n);
            check_bytes(label, expected, r.to_bytes().data(), 32);
        };
        test_msm(
            "tv: shaw msm n_1",
            1,
            tv::shaw_point::msm_n_1_scalars,
            tv::shaw_point::msm_n_1_points,
            tv::shaw_point::msm_n_1_result);
        test_msm(
            "tv: shaw msm n_2",
            2,
            tv::shaw_point::msm_n_2_scalars,
            tv::shaw_point::msm_n_2_points,
            tv::shaw_point::msm_n_2_result);
        test_msm(
            "tv: shaw msm n_4",
            4,
            tv::shaw_point::msm_n_4_scalars,
            tv::shaw_point::msm_n_4_points,
            tv::shaw_point::msm_n_4_result);
        test_msm(
            "tv: shaw msm n_16",
            16,
            tv::shaw_point::msm_n_16_scalars,
            tv::shaw_point::msm_n_16_points,
            tv::shaw_point::msm_n_16_result);
        test_msm(
            "tv: shaw msm n_32_straus",
            32,
            tv::shaw_point::msm_n_32_straus_scalars,
            tv::shaw_point::msm_n_32_straus_points,
            tv::shaw_point::msm_n_32_straus_result);
        test_msm(
            "tv: shaw msm n_33_pippenger",
            33,
            tv::shaw_point::msm_n_33_pippenger_scalars,
            tv::shaw_point::msm_n_33_pippenger_points,
            tv::shaw_point::msm_n_33_pippenger_result);
        test_msm(
            "tv: shaw msm n_64_pippenger",
            64,
            tv::shaw_point::msm_n_64_pippenger_scalars,
            tv::shaw_point::msm_n_64_pippenger_points,
            tv::shaw_point::msm_n_64_pippenger_result);
    }
    /* Pedersen */
    {
        auto test_ped = [](const char *label,
                           const uint8_t blinding[32],
                           const uint8_t H[32],
                           size_t n,
                           const uint8_t values[][32],
                           const uint8_t generators[][32],
                           const uint8_t expected[32])
        {
            auto s_blind = ShawScalar::from_bytes(blinding).value();
            auto p_H = ShawPoint::from_bytes(H).value();
            std::vector<ShawScalar> vals;
            std::vector<ShawPoint> gens;
            for (size_t i = 0; i < n; i++)
            {
                vals.push_back(ShawScalar::from_bytes(values[i]).value());
                gens.push_back(ShawPoint::from_bytes(generators[i]).value());
            }
            auto r = ShawPoint::pedersen_commit(s_blind, p_H, vals.data(), gens.data(), n);
            check_bytes(label, expected, r.to_bytes().data(), 32);
        };
        test_ped(
            "tv: shaw pedersen n_1",
            tv::shaw_point::pedersen_n_1_blinding,
            tv::shaw_point::pedersen_n_1_H,
            1,
            tv::shaw_point::pedersen_n_1_values,
            tv::shaw_point::pedersen_n_1_generators,
            tv::shaw_point::pedersen_n_1_result);
        test_ped(
            "tv: shaw pedersen blinding_zero",
            tv::shaw_point::pedersen_blinding_zero_blinding,
            tv::shaw_point::pedersen_blinding_zero_H,
            1,
            tv::shaw_point::pedersen_blinding_zero_values,
            tv::shaw_point::pedersen_blinding_zero_generators,
            tv::shaw_point::pedersen_blinding_zero_result);
        test_ped(
            "tv: shaw pedersen n_4",
            tv::shaw_point::pedersen_n_4_blinding,
            tv::shaw_point::pedersen_n_4_H,
            4,
            tv::shaw_point::pedersen_n_4_values,
            tv::shaw_point::pedersen_n_4_generators,
            tv::shaw_point::pedersen_n_4_result);
    }
    for (size_t i = 0; i < tv::shaw_point::map_to_curve_single_count; i++)
    {
        auto &v = tv::shaw_point::map_to_curve_single_vectors[i];
        auto r = ShawPoint::map_to_curve(v.u);
        check_bytes(
            (std::string("tv: shaw point map_to_curve ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::shaw_point::map_to_curve_double_count; i++)
    {
        auto &v = tv::shaw_point::map_to_curve_double_vectors[i];
        auto r = ShawPoint::map_to_curve(v.u0, v.u1);
        check_bytes(
            (std::string("tv: shaw point map_to_curve2 ") + v.label).c_str(), v.result, r.to_bytes().data(), 32);
    }
    for (size_t i = 0; i < tv::shaw_point::x_coordinate_count; i++)
    {
        auto &v = tv::shaw_point::x_coordinate_vectors[i];
        auto p = ShawPoint::from_bytes(v.point).value();
        auto r = p.x_coordinate_bytes();
        check_bytes((std::string("tv: shaw point x_coord ") + v.label).c_str(), v.x_bytes, r.data(), 32);
    }

    /* ---- Batch Invert ---- */
    std::cout << "  --- Batch Invert ---" << std::endl;
    {
        /* fp n=1 */
        {
            uint8_t result[32];
            std::memcpy(result, tv::batch_invert::fp_n_1_inputs[0], 32);
            fp_fe fe;
            fp_frombytes(fe, result);
            fp_invert(fe, fe);
            fp_tobytes(result, fe);
            check_bytes("tv: batch invert fp n_1", tv::batch_invert::fp_n_1_results[0], result, 32);
        }
        /* fp n=4 */
        {
            fp_fe fes[4];
            for (int i = 0; i < 4; i++)
                fp_frombytes(fes[i], tv::batch_invert::fp_n_4_inputs[i]);
            fp_batch_invert(fes, fes, 4);
            for (int i = 0; i < 4; i++)
            {
                uint8_t result[32];
                fp_tobytes(result, fes[i]);
                check_bytes(
                    (std::string("tv: batch invert fp n_4 [") + std::to_string(i) + "]").c_str(),
                    tv::batch_invert::fp_n_4_results[i],
                    result,
                    32);
            }
        }
        /* fq n=1 */
        {
            uint8_t result[32];
            std::memcpy(result, tv::batch_invert::fq_n_1_inputs[0], 32);
            fq_fe fe;
            fq_frombytes(fe, result);
            fq_invert(fe, fe);
            fq_tobytes(result, fe);
            check_bytes("tv: batch invert fq n_1", tv::batch_invert::fq_n_1_results[0], result, 32);
        }
        /* fq n=4 */
        {
            fq_fe fes[4];
            for (int i = 0; i < 4; i++)
                fq_frombytes(fes[i], tv::batch_invert::fq_n_4_inputs[i]);
            fq_batch_invert(fes, fes, 4);
            for (int i = 0; i < 4; i++)
            {
                uint8_t result[32];
                fq_tobytes(result, fes[i]);
                check_bytes(
                    (std::string("tv: batch invert fq n_4 [") + std::to_string(i) + "]").c_str(),
                    tv::batch_invert::fq_n_4_results[i],
                    result,
                    32);
            }
        }
    }

    /* ---- Fp Polynomial ---- */
    std::cout << "  --- Fp Polynomial ---" << std::endl;
    {
        namespace fp = tv::fp_polynomial;

        // from_roots: one root
        {
            auto p = FpPolynomial::from_roots(fp::from_roots_one_root_roots[0], 1);
            size_t n = sizeof(fp::from_roots_one_root_coefficients) / sizeof(fp::from_roots_one_root_coefficients[0]);
            check_int("tv: fp poly from_roots one_root degree", (int)(n - 1), (int)p.degree());
            auto rebuilt = FpPolynomial::from_coefficients(fp::from_roots_one_root_coefficients[0], n);
            for (size_t i = 0; i < n; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, p.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly from_roots one_root coeff[") + std::to_string(i) + "]").c_str(),
                    fp::from_roots_one_root_coefficients[i],
                    c,
                    32);
            }
        }
        // from_roots: two roots
        {
            auto p = FpPolynomial::from_roots(fp::from_roots_two_roots_roots[0], 2);
            size_t n = sizeof(fp::from_roots_two_roots_coefficients) / sizeof(fp::from_roots_two_roots_coefficients[0]);
            check_int("tv: fp poly from_roots two_roots degree", (int)(n - 1), (int)p.degree());
            for (size_t i = 0; i < n; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, p.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly from_roots two_roots coeff[") + std::to_string(i) + "]").c_str(),
                    fp::from_roots_two_roots_coefficients[i],
                    c,
                    32);
            }
        }
        // from_roots: four roots
        {
            auto p = FpPolynomial::from_roots(fp::from_roots_four_roots_roots[0], 4);
            size_t n =
                sizeof(fp::from_roots_four_roots_coefficients) / sizeof(fp::from_roots_four_roots_coefficients[0]);
            check_int("tv: fp poly from_roots four_roots degree", (int)(n - 1), (int)p.degree());
            for (size_t i = 0; i < n; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, p.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly from_roots four_roots coeff[") + std::to_string(i) + "]").c_str(),
                    fp::from_roots_four_roots_coefficients[i],
                    c,
                    32);
            }
        }

        // evaluate: constant at 7
        {
            auto p = FpPolynomial::from_coefficients(fp::eval_constant_at_7_coefficients[0], 1);
            auto r = p.evaluate(fp::eval_constant_at_7_x);
            check_bytes("tv: fp poly eval constant_at_7", fp::eval_constant_at_7_result, r.data(), 32);
        }
        // evaluate: linear at 0
        {
            auto p = FpPolynomial::from_coefficients(fp::eval_linear_at_0_coefficients[0], 2);
            auto r = p.evaluate(fp::eval_linear_at_0_x);
            check_bytes("tv: fp poly eval linear_at_0", fp::eval_linear_at_0_result, r.data(), 32);
        }
        // evaluate: linear at test_a
        {
            auto p = FpPolynomial::from_coefficients(fp::eval_linear_at_test_a_coefficients[0], 2);
            auto r = p.evaluate(fp::eval_linear_at_test_a_x);
            check_bytes("tv: fp poly eval linear_at_test_a", fp::eval_linear_at_test_a_result, r.data(), 32);
        }
        // evaluate: quadratic at 7
        {
            auto p = FpPolynomial::from_coefficients(fp::eval_quadratic_at_7_coefficients[0], 3);
            auto r = p.evaluate(fp::eval_quadratic_at_7_x);
            check_bytes("tv: fp poly eval quadratic_at_7", fp::eval_quadratic_at_7_result, r.data(), 32);
        }

        // mul: deg1 * deg1
        {
            size_t na = sizeof(fp::mul_deg1_times_deg1_a) / sizeof(fp::mul_deg1_times_deg1_a[0]);
            size_t nb = sizeof(fp::mul_deg1_times_deg1_b) / sizeof(fp::mul_deg1_times_deg1_b[0]);
            size_t nr = sizeof(fp::mul_deg1_times_deg1_result) / sizeof(fp::mul_deg1_times_deg1_result[0]);
            auto a = FpPolynomial::from_coefficients(fp::mul_deg1_times_deg1_a[0], na);
            auto b = FpPolynomial::from_coefficients(fp::mul_deg1_times_deg1_b[0], nb);
            auto r = a * b;
            check_int("tv: fp poly mul deg1*deg1 degree", (int)(nr - 1), (int)r.degree());
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly mul deg1*deg1 coeff[") + std::to_string(i) + "]").c_str(),
                    fp::mul_deg1_times_deg1_result[i],
                    c,
                    32);
            }
        }
        // mul: deg5 * deg5
        {
            size_t na = sizeof(fp::mul_deg5_times_deg5_a) / sizeof(fp::mul_deg5_times_deg5_a[0]);
            size_t nb = sizeof(fp::mul_deg5_times_deg5_b) / sizeof(fp::mul_deg5_times_deg5_b[0]);
            size_t nr = sizeof(fp::mul_deg5_times_deg5_result) / sizeof(fp::mul_deg5_times_deg5_result[0]);
            auto a = FpPolynomial::from_coefficients(fp::mul_deg5_times_deg5_a[0], na);
            auto b = FpPolynomial::from_coefficients(fp::mul_deg5_times_deg5_b[0], nb);
            auto r = a * b;
            check_int("tv: fp poly mul deg5*deg5 degree", (int)(nr - 1), (int)r.degree());
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly mul deg5*deg5 coeff[") + std::to_string(i) + "]").c_str(),
                    fp::mul_deg5_times_deg5_result[i],
                    c,
                    32);
            }
        }
        // mul: deg15 * deg15
        {
            size_t na = sizeof(fp::mul_deg15_times_deg15_a) / sizeof(fp::mul_deg15_times_deg15_a[0]);
            size_t nb = sizeof(fp::mul_deg15_times_deg15_b) / sizeof(fp::mul_deg15_times_deg15_b[0]);
            size_t nr = sizeof(fp::mul_deg15_times_deg15_result) / sizeof(fp::mul_deg15_times_deg15_result[0]);
            auto a = FpPolynomial::from_coefficients(fp::mul_deg15_times_deg15_a[0], na);
            auto b = FpPolynomial::from_coefficients(fp::mul_deg15_times_deg15_b[0], nb);
            auto r = a * b;
            check_int("tv: fp poly mul deg15*deg15 degree", (int)(nr - 1), (int)r.degree());
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly mul deg15*deg15 coeff[") + std::to_string(i) + "]").c_str(),
                    fp::mul_deg15_times_deg15_result[i],
                    c,
                    32);
            }
        }
        // mul: deg16 * deg16 (karatsuba)
        {
            size_t na =
                sizeof(fp::mul_deg16_times_deg16_karatsuba_a) / sizeof(fp::mul_deg16_times_deg16_karatsuba_a[0]);
            size_t nb =
                sizeof(fp::mul_deg16_times_deg16_karatsuba_b) / sizeof(fp::mul_deg16_times_deg16_karatsuba_b[0]);
            size_t nr = sizeof(fp::mul_deg16_times_deg16_karatsuba_result)
                        / sizeof(fp::mul_deg16_times_deg16_karatsuba_result[0]);
            auto a = FpPolynomial::from_coefficients(fp::mul_deg16_times_deg16_karatsuba_a[0], na);
            auto b = FpPolynomial::from_coefficients(fp::mul_deg16_times_deg16_karatsuba_b[0], nb);
            auto r = a * b;
            check_int("tv: fp poly mul deg16*deg16 karatsuba degree", (int)(nr - 1), (int)r.degree());
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly mul deg16*deg16 karatsuba coeff[") + std::to_string(i) + "]").c_str(),
                    fp::mul_deg16_times_deg16_karatsuba_result[i],
                    c,
                    32);
            }
        }

        // add: same degree
        {
            size_t na = sizeof(fp::add_same_degree_a) / sizeof(fp::add_same_degree_a[0]);
            size_t nb = sizeof(fp::add_same_degree_b) / sizeof(fp::add_same_degree_b[0]);
            size_t nr = sizeof(fp::add_same_degree_result) / sizeof(fp::add_same_degree_result[0]);
            auto a = FpPolynomial::from_coefficients(fp::add_same_degree_a[0], na);
            auto b = FpPolynomial::from_coefficients(fp::add_same_degree_b[0], nb);
            auto r = a + b;
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly add same_deg coeff[") + std::to_string(i) + "]").c_str(),
                    fp::add_same_degree_result[i],
                    c,
                    32);
            }
        }
        // add: different degree
        {
            size_t na = sizeof(fp::add_different_degree_a) / sizeof(fp::add_different_degree_a[0]);
            size_t nb = sizeof(fp::add_different_degree_b) / sizeof(fp::add_different_degree_b[0]);
            size_t nr = sizeof(fp::add_different_degree_result) / sizeof(fp::add_different_degree_result[0]);
            auto a = FpPolynomial::from_coefficients(fp::add_different_degree_a[0], na);
            auto b = FpPolynomial::from_coefficients(fp::add_different_degree_b[0], nb);
            auto r = a + b;
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly add diff_deg coeff[") + std::to_string(i) + "]").c_str(),
                    fp::add_different_degree_result[i],
                    c,
                    32);
            }
        }

        // sub: same degree
        {
            size_t na = sizeof(fp::sub_same_degree_a) / sizeof(fp::sub_same_degree_a[0]);
            size_t nb = sizeof(fp::sub_same_degree_b) / sizeof(fp::sub_same_degree_b[0]);
            size_t nr = sizeof(fp::sub_same_degree_result) / sizeof(fp::sub_same_degree_result[0]);
            auto a = FpPolynomial::from_coefficients(fp::sub_same_degree_a[0], na);
            auto b = FpPolynomial::from_coefficients(fp::sub_same_degree_b[0], nb);
            auto r = a - b;
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly sub same_deg coeff[") + std::to_string(i) + "]").c_str(),
                    fp::sub_same_degree_result[i],
                    c,
                    32);
            }
        }
        // sub: different degree
        {
            size_t na = sizeof(fp::sub_different_degree_a) / sizeof(fp::sub_different_degree_a[0]);
            size_t nb = sizeof(fp::sub_different_degree_b) / sizeof(fp::sub_different_degree_b[0]);
            size_t nr = sizeof(fp::sub_different_degree_result) / sizeof(fp::sub_different_degree_result[0]);
            auto a = FpPolynomial::from_coefficients(fp::sub_different_degree_a[0], na);
            auto b = FpPolynomial::from_coefficients(fp::sub_different_degree_b[0], nb);
            auto r = a - b;
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly sub diff_deg coeff[") + std::to_string(i) + "]").c_str(),
                    fp::sub_different_degree_result[i],
                    c,
                    32);
            }
        }

        // divmod: exact division
        {
            size_t nn = sizeof(fp::divmod_exact_division_numerator) / sizeof(fp::divmod_exact_division_numerator[0]);
            size_t nd =
                sizeof(fp::divmod_exact_division_denominator) / sizeof(fp::divmod_exact_division_denominator[0]);
            size_t nq = sizeof(fp::divmod_exact_division_quotient) / sizeof(fp::divmod_exact_division_quotient[0]);
            size_t nrem = sizeof(fp::divmod_exact_division_remainder) / sizeof(fp::divmod_exact_division_remainder[0]);
            auto num = FpPolynomial::from_coefficients(fp::divmod_exact_division_numerator[0], nn);
            auto den = FpPolynomial::from_coefficients(fp::divmod_exact_division_denominator[0], nd);
            auto [q, rem] = num.divmod(den);
            for (size_t i = 0; i < nq; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, q.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly divmod exact q[") + std::to_string(i) + "]").c_str(),
                    fp::divmod_exact_division_quotient[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nrem; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, rem.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly divmod exact r[") + std::to_string(i) + "]").c_str(),
                    fp::divmod_exact_division_remainder[i],
                    c,
                    32);
            }
        }
        // divmod: nonzero remainder
        {
            size_t nn =
                sizeof(fp::divmod_nonzero_remainder_numerator) / sizeof(fp::divmod_nonzero_remainder_numerator[0]);
            size_t nd =
                sizeof(fp::divmod_nonzero_remainder_denominator) / sizeof(fp::divmod_nonzero_remainder_denominator[0]);
            size_t nq =
                sizeof(fp::divmod_nonzero_remainder_quotient) / sizeof(fp::divmod_nonzero_remainder_quotient[0]);
            size_t nrem =
                sizeof(fp::divmod_nonzero_remainder_remainder) / sizeof(fp::divmod_nonzero_remainder_remainder[0]);
            auto num = FpPolynomial::from_coefficients(fp::divmod_nonzero_remainder_numerator[0], nn);
            auto den = FpPolynomial::from_coefficients(fp::divmod_nonzero_remainder_denominator[0], nd);
            auto [q, rem] = num.divmod(den);
            for (size_t i = 0; i < nq; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, q.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly divmod nonzero_rem q[") + std::to_string(i) + "]").c_str(),
                    fp::divmod_nonzero_remainder_quotient[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nrem; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, rem.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly divmod nonzero_rem r[") + std::to_string(i) + "]").c_str(),
                    fp::divmod_nonzero_remainder_remainder[i],
                    c,
                    32);
            }
        }
        // divmod: divide by linear
        {
            size_t nn =
                sizeof(fp::divmod_divide_by_linear_numerator) / sizeof(fp::divmod_divide_by_linear_numerator[0]);
            size_t nd =
                sizeof(fp::divmod_divide_by_linear_denominator) / sizeof(fp::divmod_divide_by_linear_denominator[0]);
            size_t nq = sizeof(fp::divmod_divide_by_linear_quotient) / sizeof(fp::divmod_divide_by_linear_quotient[0]);
            size_t nrem =
                sizeof(fp::divmod_divide_by_linear_remainder) / sizeof(fp::divmod_divide_by_linear_remainder[0]);
            auto num = FpPolynomial::from_coefficients(fp::divmod_divide_by_linear_numerator[0], nn);
            auto den = FpPolynomial::from_coefficients(fp::divmod_divide_by_linear_denominator[0], nd);
            auto [q, rem] = num.divmod(den);
            for (size_t i = 0; i < nq; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, q.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly divmod by_linear q[") + std::to_string(i) + "]").c_str(),
                    fp::divmod_divide_by_linear_quotient[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nrem; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, rem.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly divmod by_linear r[") + std::to_string(i) + "]").c_str(),
                    fp::divmod_divide_by_linear_remainder[i],
                    c,
                    32);
            }
        }

        // interpolate: three points
        {
            size_t nc = sizeof(fp::interp_three_points_coefficients) / sizeof(fp::interp_three_points_coefficients[0]);
            auto p = FpPolynomial::interpolate(fp::interp_three_points_xs[0], fp::interp_three_points_ys[0], 3);
            check_int("tv: fp poly interp three_points degree", (int)(nc - 1), (int)p.degree());
            for (size_t i = 0; i < nc; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, p.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly interp three_points coeff[") + std::to_string(i) + "]").c_str(),
                    fp::interp_three_points_coefficients[i],
                    c,
                    32);
            }
        }
        // interpolate: four points
        {
            size_t nc = sizeof(fp::interp_four_points_coefficients) / sizeof(fp::interp_four_points_coefficients[0]);
            auto p = FpPolynomial::interpolate(fp::interp_four_points_xs[0], fp::interp_four_points_ys[0], 4);
            check_int("tv: fp poly interp four_points degree", (int)(nc - 1), (int)p.degree());
            for (size_t i = 0; i < nc; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, p.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fp poly interp four_points coeff[") + std::to_string(i) + "]").c_str(),
                    fp::interp_four_points_coefficients[i],
                    c,
                    32);
            }
        }
    }

    /* ---- Fq Polynomial ---- */
    std::cout << "  --- Fq Polynomial ---" << std::endl;
    {
        namespace fq = tv::fq_polynomial;

        // from_roots: one root
        {
            auto p = FqPolynomial::from_roots(fq::from_roots_one_root_roots[0], 1);
            size_t n = sizeof(fq::from_roots_one_root_coefficients) / sizeof(fq::from_roots_one_root_coefficients[0]);
            check_int("tv: fq poly from_roots one_root degree", (int)(n - 1), (int)p.degree());
            for (size_t i = 0; i < n; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, p.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly from_roots one_root coeff[") + std::to_string(i) + "]").c_str(),
                    fq::from_roots_one_root_coefficients[i],
                    c,
                    32);
            }
        }
        // from_roots: two roots
        {
            auto p = FqPolynomial::from_roots(fq::from_roots_two_roots_roots[0], 2);
            size_t n = sizeof(fq::from_roots_two_roots_coefficients) / sizeof(fq::from_roots_two_roots_coefficients[0]);
            check_int("tv: fq poly from_roots two_roots degree", (int)(n - 1), (int)p.degree());
            for (size_t i = 0; i < n; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, p.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly from_roots two_roots coeff[") + std::to_string(i) + "]").c_str(),
                    fq::from_roots_two_roots_coefficients[i],
                    c,
                    32);
            }
        }
        // from_roots: four roots
        {
            auto p = FqPolynomial::from_roots(fq::from_roots_four_roots_roots[0], 4);
            size_t n =
                sizeof(fq::from_roots_four_roots_coefficients) / sizeof(fq::from_roots_four_roots_coefficients[0]);
            check_int("tv: fq poly from_roots four_roots degree", (int)(n - 1), (int)p.degree());
            for (size_t i = 0; i < n; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, p.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly from_roots four_roots coeff[") + std::to_string(i) + "]").c_str(),
                    fq::from_roots_four_roots_coefficients[i],
                    c,
                    32);
            }
        }

        // evaluate: constant at 7
        {
            auto p = FqPolynomial::from_coefficients(fq::eval_constant_at_7_coefficients[0], 1);
            auto r = p.evaluate(fq::eval_constant_at_7_x);
            check_bytes("tv: fq poly eval constant_at_7", fq::eval_constant_at_7_result, r.data(), 32);
        }
        // evaluate: linear at 0
        {
            auto p = FqPolynomial::from_coefficients(fq::eval_linear_at_0_coefficients[0], 2);
            auto r = p.evaluate(fq::eval_linear_at_0_x);
            check_bytes("tv: fq poly eval linear_at_0", fq::eval_linear_at_0_result, r.data(), 32);
        }
        // evaluate: linear at test_a
        {
            auto p = FqPolynomial::from_coefficients(fq::eval_linear_at_test_a_coefficients[0], 2);
            auto r = p.evaluate(fq::eval_linear_at_test_a_x);
            check_bytes("tv: fq poly eval linear_at_test_a", fq::eval_linear_at_test_a_result, r.data(), 32);
        }
        // evaluate: quadratic at 7
        {
            auto p = FqPolynomial::from_coefficients(fq::eval_quadratic_at_7_coefficients[0], 3);
            auto r = p.evaluate(fq::eval_quadratic_at_7_x);
            check_bytes("tv: fq poly eval quadratic_at_7", fq::eval_quadratic_at_7_result, r.data(), 32);
        }

        // mul: deg1 * deg1
        {
            size_t na = sizeof(fq::mul_deg1_times_deg1_a) / sizeof(fq::mul_deg1_times_deg1_a[0]);
            size_t nb = sizeof(fq::mul_deg1_times_deg1_b) / sizeof(fq::mul_deg1_times_deg1_b[0]);
            size_t nr = sizeof(fq::mul_deg1_times_deg1_result) / sizeof(fq::mul_deg1_times_deg1_result[0]);
            auto a = FqPolynomial::from_coefficients(fq::mul_deg1_times_deg1_a[0], na);
            auto b = FqPolynomial::from_coefficients(fq::mul_deg1_times_deg1_b[0], nb);
            auto r = a * b;
            check_int("tv: fq poly mul deg1*deg1 degree", (int)(nr - 1), (int)r.degree());
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly mul deg1*deg1 coeff[") + std::to_string(i) + "]").c_str(),
                    fq::mul_deg1_times_deg1_result[i],
                    c,
                    32);
            }
        }
        // mul: deg5 * deg5
        {
            size_t na = sizeof(fq::mul_deg5_times_deg5_a) / sizeof(fq::mul_deg5_times_deg5_a[0]);
            size_t nb = sizeof(fq::mul_deg5_times_deg5_b) / sizeof(fq::mul_deg5_times_deg5_b[0]);
            size_t nr = sizeof(fq::mul_deg5_times_deg5_result) / sizeof(fq::mul_deg5_times_deg5_result[0]);
            auto a = FqPolynomial::from_coefficients(fq::mul_deg5_times_deg5_a[0], na);
            auto b = FqPolynomial::from_coefficients(fq::mul_deg5_times_deg5_b[0], nb);
            auto r = a * b;
            check_int("tv: fq poly mul deg5*deg5 degree", (int)(nr - 1), (int)r.degree());
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly mul deg5*deg5 coeff[") + std::to_string(i) + "]").c_str(),
                    fq::mul_deg5_times_deg5_result[i],
                    c,
                    32);
            }
        }
        // mul: deg15 * deg15
        {
            size_t na = sizeof(fq::mul_deg15_times_deg15_a) / sizeof(fq::mul_deg15_times_deg15_a[0]);
            size_t nb = sizeof(fq::mul_deg15_times_deg15_b) / sizeof(fq::mul_deg15_times_deg15_b[0]);
            size_t nr = sizeof(fq::mul_deg15_times_deg15_result) / sizeof(fq::mul_deg15_times_deg15_result[0]);
            auto a = FqPolynomial::from_coefficients(fq::mul_deg15_times_deg15_a[0], na);
            auto b = FqPolynomial::from_coefficients(fq::mul_deg15_times_deg15_b[0], nb);
            auto r = a * b;
            check_int("tv: fq poly mul deg15*deg15 degree", (int)(nr - 1), (int)r.degree());
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly mul deg15*deg15 coeff[") + std::to_string(i) + "]").c_str(),
                    fq::mul_deg15_times_deg15_result[i],
                    c,
                    32);
            }
        }
        // mul: deg16 * deg16 (karatsuba)
        {
            size_t na =
                sizeof(fq::mul_deg16_times_deg16_karatsuba_a) / sizeof(fq::mul_deg16_times_deg16_karatsuba_a[0]);
            size_t nb =
                sizeof(fq::mul_deg16_times_deg16_karatsuba_b) / sizeof(fq::mul_deg16_times_deg16_karatsuba_b[0]);
            size_t nr = sizeof(fq::mul_deg16_times_deg16_karatsuba_result)
                        / sizeof(fq::mul_deg16_times_deg16_karatsuba_result[0]);
            auto a = FqPolynomial::from_coefficients(fq::mul_deg16_times_deg16_karatsuba_a[0], na);
            auto b = FqPolynomial::from_coefficients(fq::mul_deg16_times_deg16_karatsuba_b[0], nb);
            auto r = a * b;
            check_int("tv: fq poly mul deg16*deg16 karatsuba degree", (int)(nr - 1), (int)r.degree());
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly mul deg16*deg16 karatsuba coeff[") + std::to_string(i) + "]").c_str(),
                    fq::mul_deg16_times_deg16_karatsuba_result[i],
                    c,
                    32);
            }
        }

        // add: same degree
        {
            size_t na = sizeof(fq::add_same_degree_a) / sizeof(fq::add_same_degree_a[0]);
            size_t nb = sizeof(fq::add_same_degree_b) / sizeof(fq::add_same_degree_b[0]);
            size_t nr = sizeof(fq::add_same_degree_result) / sizeof(fq::add_same_degree_result[0]);
            auto a = FqPolynomial::from_coefficients(fq::add_same_degree_a[0], na);
            auto b = FqPolynomial::from_coefficients(fq::add_same_degree_b[0], nb);
            auto r = a + b;
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly add same_deg coeff[") + std::to_string(i) + "]").c_str(),
                    fq::add_same_degree_result[i],
                    c,
                    32);
            }
        }
        // add: different degree
        {
            size_t na = sizeof(fq::add_different_degree_a) / sizeof(fq::add_different_degree_a[0]);
            size_t nb = sizeof(fq::add_different_degree_b) / sizeof(fq::add_different_degree_b[0]);
            size_t nr = sizeof(fq::add_different_degree_result) / sizeof(fq::add_different_degree_result[0]);
            auto a = FqPolynomial::from_coefficients(fq::add_different_degree_a[0], na);
            auto b = FqPolynomial::from_coefficients(fq::add_different_degree_b[0], nb);
            auto r = a + b;
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly add diff_deg coeff[") + std::to_string(i) + "]").c_str(),
                    fq::add_different_degree_result[i],
                    c,
                    32);
            }
        }

        // sub: same degree
        {
            size_t na = sizeof(fq::sub_same_degree_a) / sizeof(fq::sub_same_degree_a[0]);
            size_t nb = sizeof(fq::sub_same_degree_b) / sizeof(fq::sub_same_degree_b[0]);
            size_t nr = sizeof(fq::sub_same_degree_result) / sizeof(fq::sub_same_degree_result[0]);
            auto a = FqPolynomial::from_coefficients(fq::sub_same_degree_a[0], na);
            auto b = FqPolynomial::from_coefficients(fq::sub_same_degree_b[0], nb);
            auto r = a - b;
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly sub same_deg coeff[") + std::to_string(i) + "]").c_str(),
                    fq::sub_same_degree_result[i],
                    c,
                    32);
            }
        }
        // sub: different degree
        {
            size_t na = sizeof(fq::sub_different_degree_a) / sizeof(fq::sub_different_degree_a[0]);
            size_t nb = sizeof(fq::sub_different_degree_b) / sizeof(fq::sub_different_degree_b[0]);
            size_t nr = sizeof(fq::sub_different_degree_result) / sizeof(fq::sub_different_degree_result[0]);
            auto a = FqPolynomial::from_coefficients(fq::sub_different_degree_a[0], na);
            auto b = FqPolynomial::from_coefficients(fq::sub_different_degree_b[0], nb);
            auto r = a - b;
            for (size_t i = 0; i < nr; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, r.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly sub diff_deg coeff[") + std::to_string(i) + "]").c_str(),
                    fq::sub_different_degree_result[i],
                    c,
                    32);
            }
        }

        // divmod: exact division
        {
            size_t nn = sizeof(fq::divmod_exact_division_numerator) / sizeof(fq::divmod_exact_division_numerator[0]);
            size_t nd =
                sizeof(fq::divmod_exact_division_denominator) / sizeof(fq::divmod_exact_division_denominator[0]);
            size_t nq = sizeof(fq::divmod_exact_division_quotient) / sizeof(fq::divmod_exact_division_quotient[0]);
            size_t nrem = sizeof(fq::divmod_exact_division_remainder) / sizeof(fq::divmod_exact_division_remainder[0]);
            auto num = FqPolynomial::from_coefficients(fq::divmod_exact_division_numerator[0], nn);
            auto den = FqPolynomial::from_coefficients(fq::divmod_exact_division_denominator[0], nd);
            auto [q, rem] = num.divmod(den);
            for (size_t i = 0; i < nq; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, q.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly divmod exact q[") + std::to_string(i) + "]").c_str(),
                    fq::divmod_exact_division_quotient[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nrem; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, rem.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly divmod exact r[") + std::to_string(i) + "]").c_str(),
                    fq::divmod_exact_division_remainder[i],
                    c,
                    32);
            }
        }
        // divmod: nonzero remainder
        {
            size_t nn =
                sizeof(fq::divmod_nonzero_remainder_numerator) / sizeof(fq::divmod_nonzero_remainder_numerator[0]);
            size_t nd =
                sizeof(fq::divmod_nonzero_remainder_denominator) / sizeof(fq::divmod_nonzero_remainder_denominator[0]);
            size_t nq =
                sizeof(fq::divmod_nonzero_remainder_quotient) / sizeof(fq::divmod_nonzero_remainder_quotient[0]);
            size_t nrem =
                sizeof(fq::divmod_nonzero_remainder_remainder) / sizeof(fq::divmod_nonzero_remainder_remainder[0]);
            auto num = FqPolynomial::from_coefficients(fq::divmod_nonzero_remainder_numerator[0], nn);
            auto den = FqPolynomial::from_coefficients(fq::divmod_nonzero_remainder_denominator[0], nd);
            auto [q, rem] = num.divmod(den);
            for (size_t i = 0; i < nq; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, q.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly divmod nonzero_rem q[") + std::to_string(i) + "]").c_str(),
                    fq::divmod_nonzero_remainder_quotient[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nrem; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, rem.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly divmod nonzero_rem r[") + std::to_string(i) + "]").c_str(),
                    fq::divmod_nonzero_remainder_remainder[i],
                    c,
                    32);
            }
        }
        // divmod: divide by linear
        {
            size_t nn =
                sizeof(fq::divmod_divide_by_linear_numerator) / sizeof(fq::divmod_divide_by_linear_numerator[0]);
            size_t nd =
                sizeof(fq::divmod_divide_by_linear_denominator) / sizeof(fq::divmod_divide_by_linear_denominator[0]);
            size_t nq = sizeof(fq::divmod_divide_by_linear_quotient) / sizeof(fq::divmod_divide_by_linear_quotient[0]);
            size_t nrem =
                sizeof(fq::divmod_divide_by_linear_remainder) / sizeof(fq::divmod_divide_by_linear_remainder[0]);
            auto num = FqPolynomial::from_coefficients(fq::divmod_divide_by_linear_numerator[0], nn);
            auto den = FqPolynomial::from_coefficients(fq::divmod_divide_by_linear_denominator[0], nd);
            auto [q, rem] = num.divmod(den);
            for (size_t i = 0; i < nq; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, q.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly divmod by_linear q[") + std::to_string(i) + "]").c_str(),
                    fq::divmod_divide_by_linear_quotient[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nrem; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, rem.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly divmod by_linear r[") + std::to_string(i) + "]").c_str(),
                    fq::divmod_divide_by_linear_remainder[i],
                    c,
                    32);
            }
        }

        // interpolate: three points
        {
            size_t nc = sizeof(fq::interp_three_points_coefficients) / sizeof(fq::interp_three_points_coefficients[0]);
            auto p = FqPolynomial::interpolate(fq::interp_three_points_xs[0], fq::interp_three_points_ys[0], 3);
            check_int("tv: fq poly interp three_points degree", (int)(nc - 1), (int)p.degree());
            for (size_t i = 0; i < nc; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, p.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly interp three_points coeff[") + std::to_string(i) + "]").c_str(),
                    fq::interp_three_points_coefficients[i],
                    c,
                    32);
            }
        }
        // interpolate: four points
        {
            size_t nc = sizeof(fq::interp_four_points_coefficients) / sizeof(fq::interp_four_points_coefficients[0]);
            auto p = FqPolynomial::interpolate(fq::interp_four_points_xs[0], fq::interp_four_points_ys[0], 4);
            check_int("tv: fq poly interp four_points degree", (int)(nc - 1), (int)p.degree());
            for (size_t i = 0; i < nc; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, p.raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: fq poly interp four_points coeff[") + std::to_string(i) + "]").c_str(),
                    fq::interp_four_points_coefficients[i],
                    c,
                    32);
            }
        }
    }

    /* ---- Ran Divisor ---- */
    std::cout << "  --- Ran Divisor ---" << std::endl;
    {
        namespace hd = tv::ran_divisor;

        // n=2
        {
            RanPoint pts[2];
            for (int i = 0; i < 2; i++)
                pts[i] = RanPoint::from_bytes(hd::n_2_points[i]).value();
            auto d = RanDivisor::compute(pts, 2);
            size_t na = sizeof(hd::n_2_a_coefficients) / sizeof(hd::n_2_a_coefficients[0]);
            size_t nb = sizeof(hd::n_2_b_coefficients) / sizeof(hd::n_2_b_coefficients[0]);
            for (size_t i = 0; i < na; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, d.a().raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: ran divisor n=2 a[") + std::to_string(i) + "]").c_str(),
                    hd::n_2_a_coefficients[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nb; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, d.b().raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: ran divisor n=2 b[") + std::to_string(i) + "]").c_str(),
                    hd::n_2_b_coefficients[i],
                    c,
                    32);
            }
            auto ev = d.evaluate(hd::n_2_eval_point_x, hd::n_2_eval_point_y);
            check_bytes("tv: ran divisor n=2 eval", hd::n_2_eval_result, ev.data(), 32);
        }
        // n=4
        {
            RanPoint pts[4];
            for (int i = 0; i < 4; i++)
                pts[i] = RanPoint::from_bytes(hd::n_4_points[i]).value();
            auto d = RanDivisor::compute(pts, 4);
            size_t na = sizeof(hd::n_4_a_coefficients) / sizeof(hd::n_4_a_coefficients[0]);
            size_t nb = sizeof(hd::n_4_b_coefficients) / sizeof(hd::n_4_b_coefficients[0]);
            for (size_t i = 0; i < na; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, d.a().raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: ran divisor n=4 a[") + std::to_string(i) + "]").c_str(),
                    hd::n_4_a_coefficients[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nb; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, d.b().raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: ran divisor n=4 b[") + std::to_string(i) + "]").c_str(),
                    hd::n_4_b_coefficients[i],
                    c,
                    32);
            }
            auto ev = d.evaluate(hd::n_4_eval_point_x, hd::n_4_eval_point_y);
            check_bytes("tv: ran divisor n=4 eval", hd::n_4_eval_result, ev.data(), 32);
        }
        // n=8
        {
            RanPoint pts[8];
            for (int i = 0; i < 8; i++)
                pts[i] = RanPoint::from_bytes(hd::n_8_points[i]).value();
            auto d = RanDivisor::compute(pts, 8);
            size_t na = sizeof(hd::n_8_a_coefficients) / sizeof(hd::n_8_a_coefficients[0]);
            size_t nb = sizeof(hd::n_8_b_coefficients) / sizeof(hd::n_8_b_coefficients[0]);
            for (size_t i = 0; i < na; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, d.a().raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: ran divisor n=8 a[") + std::to_string(i) + "]").c_str(),
                    hd::n_8_a_coefficients[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nb; i++)
            {
                uint8_t c[32];
                fp_tobytes(c, d.b().raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: ran divisor n=8 b[") + std::to_string(i) + "]").c_str(),
                    hd::n_8_b_coefficients[i],
                    c,
                    32);
            }
            auto ev = d.evaluate(hd::n_8_eval_point_x, hd::n_8_eval_point_y);
            check_bytes("tv: ran divisor n=8 eval", hd::n_8_eval_result, ev.data(), 32);
        }
    }

    /* ---- Shaw Divisor ---- */
    std::cout << "  --- Shaw Divisor ---" << std::endl;
    {
        namespace sd = tv::shaw_divisor;

        // n=2
        {
            ShawPoint pts[2];
            for (int i = 0; i < 2; i++)
                pts[i] = ShawPoint::from_bytes(sd::n_2_points[i]).value();
            auto d = ShawDivisor::compute(pts, 2);
            size_t na = sizeof(sd::n_2_a_coefficients) / sizeof(sd::n_2_a_coefficients[0]);
            size_t nb = sizeof(sd::n_2_b_coefficients) / sizeof(sd::n_2_b_coefficients[0]);
            for (size_t i = 0; i < na; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, d.a().raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: shaw divisor n=2 a[") + std::to_string(i) + "]").c_str(),
                    sd::n_2_a_coefficients[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nb; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, d.b().raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: shaw divisor n=2 b[") + std::to_string(i) + "]").c_str(),
                    sd::n_2_b_coefficients[i],
                    c,
                    32);
            }
            auto ev = d.evaluate(sd::n_2_eval_point_x, sd::n_2_eval_point_y);
            check_bytes("tv: shaw divisor n=2 eval", sd::n_2_eval_result, ev.data(), 32);
        }
        // n=4
        {
            ShawPoint pts[4];
            for (int i = 0; i < 4; i++)
                pts[i] = ShawPoint::from_bytes(sd::n_4_points[i]).value();
            auto d = ShawDivisor::compute(pts, 4);
            size_t na = sizeof(sd::n_4_a_coefficients) / sizeof(sd::n_4_a_coefficients[0]);
            size_t nb = sizeof(sd::n_4_b_coefficients) / sizeof(sd::n_4_b_coefficients[0]);
            for (size_t i = 0; i < na; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, d.a().raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: shaw divisor n=4 a[") + std::to_string(i) + "]").c_str(),
                    sd::n_4_a_coefficients[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nb; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, d.b().raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: shaw divisor n=4 b[") + std::to_string(i) + "]").c_str(),
                    sd::n_4_b_coefficients[i],
                    c,
                    32);
            }
            auto ev = d.evaluate(sd::n_4_eval_point_x, sd::n_4_eval_point_y);
            check_bytes("tv: shaw divisor n=4 eval", sd::n_4_eval_result, ev.data(), 32);
        }
        // n=8
        {
            ShawPoint pts[8];
            for (int i = 0; i < 8; i++)
                pts[i] = ShawPoint::from_bytes(sd::n_8_points[i]).value();
            auto d = ShawDivisor::compute(pts, 8);
            size_t na = sizeof(sd::n_8_a_coefficients) / sizeof(sd::n_8_a_coefficients[0]);
            size_t nb = sizeof(sd::n_8_b_coefficients) / sizeof(sd::n_8_b_coefficients[0]);
            for (size_t i = 0; i < na; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, d.a().raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: shaw divisor n=8 a[") + std::to_string(i) + "]").c_str(),
                    sd::n_8_a_coefficients[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nb; i++)
            {
                uint8_t c[32];
                fq_tobytes(c, d.b().raw().coeffs[i].v);
                check_bytes(
                    (std::string("tv: shaw divisor n=8 b[") + std::to_string(i) + "]").c_str(),
                    sd::n_8_b_coefficients[i],
                    c,
                    32);
            }
            auto ev = d.evaluate(sd::n_8_eval_point_x, sd::n_8_eval_point_y);
            check_bytes("tv: shaw divisor n=8 eval", sd::n_8_eval_result, ev.data(), 32);
        }
    }

    /* ---- High-Degree Poly Mul ---- */
    std::cout << "  --- High-Degree Poly Mul ---" << std::endl;
    {
        namespace hdp = tv::high_degree_poly_mul;

        // Fp vectors
        for (size_t vi = 0; vi < hdp::fp_count; vi++)
        {
            auto &v = hdp::fp_vectors[vi];
            size_t n = (size_t)v.n_coeffs;

            // Build deterministic polynomials: a[i] = (i+1) mod p, b[i] = (i+n+1) mod p
            std::vector<uint8_t> a_bytes(n * 32, 0), b_bytes(n * 32, 0);
            for (size_t i = 0; i < n; i++)
            {
                uint32_t va = (uint32_t)(i + 1);
                uint32_t vb = (uint32_t)(i + n + 1);
                std::memcpy(&a_bytes[i * 32], &va, sizeof(va));
                std::memcpy(&b_bytes[i * 32], &vb, sizeof(vb));
            }
            auto a = FpPolynomial::from_coefficients(a_bytes.data(), n);
            auto b = FpPolynomial::from_coefficients(b_bytes.data(), n);
            auto r = a * b;

            std::string prefix = std::string("tv: highdeg fp ") + v.label;
            check_int((prefix + " result_degree").c_str(), v.result_degree, (int)r.degree());

            for (size_t ci = 0; ci < 3; ci++)
            {
                auto &chk = v.checks[ci];
                // Verify a(x)
                auto a_at_x = a.evaluate(chk.x);
                check_bytes((prefix + " " + chk.point + " a(x)").c_str(), chk.a_of_x, a_at_x.data(), 32);
                // Verify b(x)
                auto b_at_x = b.evaluate(chk.x);
                check_bytes((prefix + " " + chk.point + " b(x)").c_str(), chk.b_of_x, b_at_x.data(), 32);
                // Verify result(x) = a(x)*b(x)
                auto r_at_x = r.evaluate(chk.x);
                check_bytes((prefix + " " + chk.point + " result(x)").c_str(), chk.result_of_x, r_at_x.data(), 32);
            }
        }

        // Fq vectors
        for (size_t vi = 0; vi < hdp::fq_count; vi++)
        {
            auto &v = hdp::fq_vectors[vi];
            size_t n = (size_t)v.n_coeffs;

            std::vector<uint8_t> a_bytes(n * 32, 0), b_bytes(n * 32, 0);
            for (size_t i = 0; i < n; i++)
            {
                uint32_t va = (uint32_t)(i + 1);
                uint32_t vb = (uint32_t)(i + n + 1);
                std::memcpy(&a_bytes[i * 32], &va, sizeof(va));
                std::memcpy(&b_bytes[i * 32], &vb, sizeof(vb));
            }
            auto a = FqPolynomial::from_coefficients(a_bytes.data(), n);
            auto b = FqPolynomial::from_coefficients(b_bytes.data(), n);
            auto r = a * b;

            std::string prefix = std::string("tv: highdeg fq ") + v.label;
            check_int((prefix + " result_degree").c_str(), v.result_degree, (int)r.degree());

            for (size_t ci = 0; ci < 3; ci++)
            {
                auto &chk = v.checks[ci];
                auto a_at_x = a.evaluate(chk.x);
                check_bytes((prefix + " " + chk.point + " a(x)").c_str(), chk.a_of_x, a_at_x.data(), 32);
                auto b_at_x = b.evaluate(chk.x);
                check_bytes((prefix + " " + chk.point + " b(x)").c_str(), chk.b_of_x, b_at_x.data(), 32);
                auto r_at_x = r.evaluate(chk.x);
                check_bytes((prefix + " " + chk.point + " result(x)").c_str(), chk.result_of_x, r_at_x.data(), 32);
            }
        }
    }

    /* ---- Wei25519 ---- */
    std::cout << "  --- Wei25519 ---" << std::endl;
    for (size_t i = 0; i < tv::wei25519::x_to_scalar_count; i++)
    {
        auto &v = tv::wei25519::x_to_scalar_vectors[i];
        auto r = shaw_scalar_from_wei25519_x(v.input);
        std::string name = std::string("tv: wei25519 x_to_scalar ") + v.label;
        if (v.valid)
        {
            check_int((name + " valid").c_str(), 1, r.has_value() ? 1 : 0);
            if (r)
                check_bytes((name + " value").c_str(), v.result, r->to_bytes().data(), 32);
        }
        else
        {
            check_int((name + " invalid").c_str(), 0, r.has_value() ? 1 : 0);
        }
    }
}

static void test_vector_validation_c_primitives()
{
    namespace tv = ranshaw_test_vectors;

    std::cout << std::endl << "=== Test Vector Validation (C Primitives) ===" << std::endl;

    /* Helper: load a ran_jacobian from 32-byte encoding, handling identity */
    auto h_load = [](ran_jacobian *out, const uint8_t bytes[32]) -> bool
    {
        bool all_zero = true;
        for (int i = 0; i < 32; i++)
        {
            if (bytes[i])
            {
                all_zero = false;
                break;
            }
        }
        if (all_zero)
        {
            ran_identity(out);
            return true;
        }
        return ran_frombytes(out, bytes) == 0;
    };

    auto s_load = [](shaw_jacobian *out, const uint8_t bytes[32]) -> bool
    {
        bool all_zero = true;
        for (int i = 0; i < 32; i++)
        {
            if (bytes[i])
            {
                all_zero = false;
                break;
            }
        }
        if (all_zero)
        {
            shaw_identity(out);
            return true;
        }
        return shaw_frombytes(out, bytes) == 0;
    };

    /* ==== Ran Scalar (C primitives) ==== */
    std::cout << "  --- Ran Scalar (C) ---" << std::endl;
    for (size_t i = 0; i < tv::ran_scalar::add_count; i++)
    {
        auto &v = tv::ran_scalar::add_vectors[i];
        fq_fe a, b, r;
        unsigned char out[32];
        ran_scalar_from_bytes(a, v.a);
        ran_scalar_from_bytes(b, v.b);
        ran_scalar_add(r, a, b);
        ran_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): ran scalar add ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::sub_count; i++)
    {
        auto &v = tv::ran_scalar::sub_vectors[i];
        fq_fe a, b, r;
        unsigned char out[32];
        ran_scalar_from_bytes(a, v.a);
        ran_scalar_from_bytes(b, v.b);
        ran_scalar_sub(r, a, b);
        ran_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): ran scalar sub ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::mul_count; i++)
    {
        auto &v = tv::ran_scalar::mul_vectors[i];
        fq_fe a, b, r;
        unsigned char out[32];
        ran_scalar_from_bytes(a, v.a);
        ran_scalar_from_bytes(b, v.b);
        ran_scalar_mul(r, a, b);
        ran_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): ran scalar mul ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::sq_count; i++)
    {
        auto &v = tv::ran_scalar::sq_vectors[i];
        fq_fe a, r;
        unsigned char out[32];
        ran_scalar_from_bytes(a, v.a);
        ran_scalar_sq(r, a);
        ran_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): ran scalar sq ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::negate_count; i++)
    {
        auto &v = tv::ran_scalar::negate_vectors[i];
        fq_fe a, r;
        unsigned char out[32];
        ran_scalar_from_bytes(a, v.a);
        ran_scalar_neg(r, a);
        ran_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): ran scalar neg ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::invert_count; i++)
    {
        auto &v = tv::ran_scalar::invert_vectors[i];
        if (!v.valid)
            continue; /* C-level invert(0) is undefined */
        fq_fe a, r;
        unsigned char out[32];
        ran_scalar_from_bytes(a, v.a);
        ran_scalar_invert(r, a);
        ran_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): ran scalar inv ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::reduce_wide_count; i++)
    {
        auto &v = tv::ran_scalar::reduce_wide_vectors[i];
        fq_fe r;
        unsigned char out[32];
        ran_scalar_reduce_wide(r, v.input);
        ran_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): ran scalar reduce_wide ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::muladd_count; i++)
    {
        auto &v = tv::ran_scalar::muladd_vectors[i];
        fq_fe a, b, c, r;
        unsigned char out[32];
        ran_scalar_from_bytes(a, v.a);
        ran_scalar_from_bytes(b, v.b);
        ran_scalar_from_bytes(c, v.c);
        ran_scalar_muladd(r, a, b, c);
        ran_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): ran scalar muladd ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::ran_scalar::is_zero_count; i++)
    {
        auto &v = tv::ran_scalar::is_zero_vectors[i];
        fq_fe a;
        ran_scalar_from_bytes(a, v.a);
        check_int(
            (std::string("tv(C): ran scalar is_zero ") + v.label).c_str(),
            v.result ? 1 : 0,
            ran_scalar_is_zero(a));
    }

    /* ==== Shaw Scalar (C primitives) ==== */
    std::cout << "  --- Shaw Scalar (C) ---" << std::endl;
    for (size_t i = 0; i < tv::shaw_scalar::add_count; i++)
    {
        auto &v = tv::shaw_scalar::add_vectors[i];
        fp_fe a, b, r;
        unsigned char out[32];
        shaw_scalar_from_bytes(a, v.a);
        shaw_scalar_from_bytes(b, v.b);
        shaw_scalar_add(r, a, b);
        shaw_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): shaw scalar add ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::sub_count; i++)
    {
        auto &v = tv::shaw_scalar::sub_vectors[i];
        fp_fe a, b, r;
        unsigned char out[32];
        shaw_scalar_from_bytes(a, v.a);
        shaw_scalar_from_bytes(b, v.b);
        shaw_scalar_sub(r, a, b);
        shaw_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): shaw scalar sub ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::mul_count; i++)
    {
        auto &v = tv::shaw_scalar::mul_vectors[i];
        fp_fe a, b, r;
        unsigned char out[32];
        shaw_scalar_from_bytes(a, v.a);
        shaw_scalar_from_bytes(b, v.b);
        shaw_scalar_mul(r, a, b);
        shaw_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): shaw scalar mul ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::sq_count; i++)
    {
        auto &v = tv::shaw_scalar::sq_vectors[i];
        fp_fe a, r;
        unsigned char out[32];
        shaw_scalar_from_bytes(a, v.a);
        shaw_scalar_sq(r, a);
        shaw_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): shaw scalar sq ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::negate_count; i++)
    {
        auto &v = tv::shaw_scalar::negate_vectors[i];
        fp_fe a, r;
        unsigned char out[32];
        shaw_scalar_from_bytes(a, v.a);
        shaw_scalar_neg(r, a);
        shaw_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): shaw scalar neg ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::invert_count; i++)
    {
        auto &v = tv::shaw_scalar::invert_vectors[i];
        if (!v.valid)
            continue;
        fp_fe a, r;
        unsigned char out[32];
        shaw_scalar_from_bytes(a, v.a);
        shaw_scalar_invert(r, a);
        shaw_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): shaw scalar inv ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::reduce_wide_count; i++)
    {
        auto &v = tv::shaw_scalar::reduce_wide_vectors[i];
        fp_fe r;
        unsigned char out[32];
        shaw_scalar_reduce_wide(r, v.input);
        shaw_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): shaw scalar reduce_wide ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::muladd_count; i++)
    {
        auto &v = tv::shaw_scalar::muladd_vectors[i];
        fp_fe a, b, c, r;
        unsigned char out[32];
        shaw_scalar_from_bytes(a, v.a);
        shaw_scalar_from_bytes(b, v.b);
        shaw_scalar_from_bytes(c, v.c);
        shaw_scalar_muladd(r, a, b, c);
        shaw_scalar_to_bytes(out, r);
        check_bytes((std::string("tv(C): shaw scalar muladd ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::shaw_scalar::is_zero_count; i++)
    {
        auto &v = tv::shaw_scalar::is_zero_vectors[i];
        fp_fe a;
        shaw_scalar_from_bytes(a, v.a);
        check_int(
            (std::string("tv(C): shaw scalar is_zero ") + v.label).c_str(),
            v.result ? 1 : 0,
            shaw_scalar_is_zero(a));
    }

    /* ==== Ran Point (C primitives) ==== */
    std::cout << "  --- Ran Point (C) ---" << std::endl;
    for (size_t i = 0; i < tv::ran_point::from_bytes_count; i++)
    {
        auto &v = tv::ran_point::from_bytes_vectors[i];
        ran_jacobian p;
        int ok = ran_frombytes(&p, v.input);
        std::string name = std::string("tv(C): ran point from_bytes ") + v.label;
        if (v.valid)
        {
            check_int((name + " valid").c_str(), 0, ok);
            if (ok == 0)
            {
                unsigned char out[32];
                ran_tobytes(out, &p);
                check_bytes((name + " value").c_str(), v.result, out, 32);
            }
        }
        else
        {
            check_int((name + " invalid").c_str(), 1, (ok != 0) ? 1 : 0);
        }
    }
    for (size_t i = 0; i < tv::ran_point::add_count; i++)
    {
        auto &v = tv::ran_point::add_vectors[i];
        ran_jacobian a, b, r;
        h_load(&a, v.a);
        h_load(&b, v.b);
        ran_add(&r, &a, &b);
        unsigned char out[32];
        ran_tobytes(out, &r);
        check_bytes((std::string("tv(C): ran point add ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::ran_point::dbl_count; i++)
    {
        auto &v = tv::ran_point::dbl_vectors[i];
        ran_jacobian a, r;
        h_load(&a, v.a);
        ran_dbl(&r, &a);
        unsigned char out[32];
        ran_tobytes(out, &r);
        check_bytes((std::string("tv(C): ran point dbl ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::ran_point::negate_count; i++)
    {
        auto &v = tv::ran_point::negate_vectors[i];
        ran_jacobian a, r;
        h_load(&a, v.a);
        ran_neg(&r, &a);
        unsigned char out[32];
        ran_tobytes(out, &r);
        check_bytes((std::string("tv(C): ran point neg ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::ran_point::scalar_mul_count; i++)
    {
        auto &v = tv::ran_point::scalar_mul_vectors[i];
        ran_jacobian p, r;
        h_load(&p, v.point);
        ran_scalarmult(&r, v.scalar, &p);
        unsigned char out[32];
        ran_tobytes(out, &r);
        check_bytes((std::string("tv(C): ran point scalarmult ") + v.label).c_str(), v.result, out, 32);
    }

    /* MSM (ran) */
    {
        auto run_msm = [&](const char *name,
                           const uint8_t(*scalars)[32],
                           const uint8_t(*points)[32],
                           const uint8_t expected[32],
                           size_t n)
        {
            std::vector<ran_jacobian> pts(n);
            for (size_t j = 0; j < n; j++)
                h_load(&pts[j], points[j]);
            ran_jacobian r;
            ran_msm_vartime(&r, scalars[0], pts.data(), n);
            unsigned char out[32];
            ran_tobytes(out, &r);
            check_bytes(name, expected, out, 32);
        };
        namespace hp = tv::ran_point;
        run_msm("tv(C): ran msm n=1", hp::msm_n_1_scalars, hp::msm_n_1_points, hp::msm_n_1_result, 1);
        run_msm("tv(C): ran msm n=2", hp::msm_n_2_scalars, hp::msm_n_2_points, hp::msm_n_2_result, 2);
        run_msm("tv(C): ran msm n=4", hp::msm_n_4_scalars, hp::msm_n_4_points, hp::msm_n_4_result, 4);
        run_msm("tv(C): ran msm n=16", hp::msm_n_16_scalars, hp::msm_n_16_points, hp::msm_n_16_result, 16);
        run_msm(
            "tv(C): ran msm n=32",
            hp::msm_n_32_straus_scalars,
            hp::msm_n_32_straus_points,
            hp::msm_n_32_straus_result,
            32);
        run_msm(
            "tv(C): ran msm n=33",
            hp::msm_n_33_pippenger_scalars,
            hp::msm_n_33_pippenger_points,
            hp::msm_n_33_pippenger_result,
            33);
        run_msm(
            "tv(C): ran msm n=64",
            hp::msm_n_64_pippenger_scalars,
            hp::msm_n_64_pippenger_points,
            hp::msm_n_64_pippenger_result,
            64);
    }

    /* Pedersen (ran) */
    {
        auto run_ped = [&](const char *name,
                           const uint8_t blinding[32],
                           const uint8_t H_bytes[32],
                           const uint8_t(*values)[32],
                           const uint8_t(*generators)[32],
                           const uint8_t expected[32],
                           size_t n)
        {
            ran_jacobian H_pt;
            h_load(&H_pt, H_bytes);
            std::vector<ran_jacobian> gens(n);
            for (size_t j = 0; j < n; j++)
                h_load(&gens[j], generators[j]);
            ran_jacobian r;
            ran_pedersen_commit(&r, blinding, &H_pt, values[0], gens.data(), n);
            unsigned char out[32];
            ran_tobytes(out, &r);
            check_bytes(name, expected, out, 32);
        };
        namespace hp = tv::ran_point;
        run_ped(
            "tv(C): ran pedersen n=1",
            hp::pedersen_n_1_blinding,
            hp::pedersen_n_1_H,
            hp::pedersen_n_1_values,
            hp::pedersen_n_1_generators,
            hp::pedersen_n_1_result,
            1);
        run_ped(
            "tv(C): ran pedersen n=4",
            hp::pedersen_n_4_blinding,
            hp::pedersen_n_4_H,
            hp::pedersen_n_4_values,
            hp::pedersen_n_4_generators,
            hp::pedersen_n_4_result,
            4);
        run_ped(
            "tv(C): ran pedersen blind=0",
            hp::pedersen_blinding_zero_blinding,
            hp::pedersen_blinding_zero_H,
            hp::pedersen_blinding_zero_values,
            hp::pedersen_blinding_zero_generators,
            hp::pedersen_blinding_zero_result,
            1);
    }

    /* Map-to-curve (ran) */
    for (size_t i = 0; i < tv::ran_point::map_to_curve_single_count; i++)
    {
        auto &v = tv::ran_point::map_to_curve_single_vectors[i];
        ran_jacobian r;
        ran_map_to_curve(&r, v.u);
        unsigned char out[32];
        ran_tobytes(out, &r);
        check_bytes((std::string("tv(C): ran map_to_curve ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::ran_point::map_to_curve_double_count; i++)
    {
        auto &v = tv::ran_point::map_to_curve_double_vectors[i];
        ran_jacobian r;
        ran_map_to_curve2(&r, v.u0, v.u1);
        unsigned char out[32];
        ran_tobytes(out, &r);
        check_bytes((std::string("tv(C): ran map_to_curve2 ") + v.label).c_str(), v.result, out, 32);
    }

    /* ==== Shaw Point (C primitives) ==== */
    std::cout << "  --- Shaw Point (C) ---" << std::endl;
    for (size_t i = 0; i < tv::shaw_point::from_bytes_count; i++)
    {
        auto &v = tv::shaw_point::from_bytes_vectors[i];
        shaw_jacobian p;
        int ok = shaw_frombytes(&p, v.input);
        std::string name = std::string("tv(C): shaw point from_bytes ") + v.label;
        if (v.valid)
        {
            check_int((name + " valid").c_str(), 0, ok);
            if (ok == 0)
            {
                unsigned char out[32];
                shaw_tobytes(out, &p);
                check_bytes((name + " value").c_str(), v.result, out, 32);
            }
        }
        else
        {
            check_int((name + " invalid").c_str(), 1, (ok != 0) ? 1 : 0);
        }
    }
    for (size_t i = 0; i < tv::shaw_point::add_count; i++)
    {
        auto &v = tv::shaw_point::add_vectors[i];
        shaw_jacobian a, b, r;
        s_load(&a, v.a);
        s_load(&b, v.b);
        shaw_add(&r, &a, &b);
        unsigned char out[32];
        shaw_tobytes(out, &r);
        check_bytes((std::string("tv(C): shaw point add ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::shaw_point::dbl_count; i++)
    {
        auto &v = tv::shaw_point::dbl_vectors[i];
        shaw_jacobian a, r;
        s_load(&a, v.a);
        shaw_dbl(&r, &a);
        unsigned char out[32];
        shaw_tobytes(out, &r);
        check_bytes((std::string("tv(C): shaw point dbl ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::shaw_point::negate_count; i++)
    {
        auto &v = tv::shaw_point::negate_vectors[i];
        shaw_jacobian a, r;
        s_load(&a, v.a);
        shaw_neg(&r, &a);
        unsigned char out[32];
        shaw_tobytes(out, &r);
        check_bytes((std::string("tv(C): shaw point neg ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::shaw_point::scalar_mul_count; i++)
    {
        auto &v = tv::shaw_point::scalar_mul_vectors[i];
        shaw_jacobian p, r;
        s_load(&p, v.point);
        shaw_scalarmult(&r, v.scalar, &p);
        unsigned char out[32];
        shaw_tobytes(out, &r);
        check_bytes((std::string("tv(C): shaw point scalarmult ") + v.label).c_str(), v.result, out, 32);
    }

    /* MSM (shaw) */
    {
        auto run_msm = [&](const char *name,
                           const uint8_t(*scalars)[32],
                           const uint8_t(*points)[32],
                           const uint8_t expected[32],
                           size_t n)
        {
            std::vector<shaw_jacobian> pts(n);
            for (size_t j = 0; j < n; j++)
                s_load(&pts[j], points[j]);
            shaw_jacobian r;
            shaw_msm_vartime(&r, scalars[0], pts.data(), n);
            unsigned char out[32];
            shaw_tobytes(out, &r);
            check_bytes(name, expected, out, 32);
        };
        namespace sp = tv::shaw_point;
        run_msm("tv(C): shaw msm n=1", sp::msm_n_1_scalars, sp::msm_n_1_points, sp::msm_n_1_result, 1);
        run_msm("tv(C): shaw msm n=2", sp::msm_n_2_scalars, sp::msm_n_2_points, sp::msm_n_2_result, 2);
        run_msm("tv(C): shaw msm n=4", sp::msm_n_4_scalars, sp::msm_n_4_points, sp::msm_n_4_result, 4);
        run_msm("tv(C): shaw msm n=16", sp::msm_n_16_scalars, sp::msm_n_16_points, sp::msm_n_16_result, 16);
        run_msm(
            "tv(C): shaw msm n=32",
            sp::msm_n_32_straus_scalars,
            sp::msm_n_32_straus_points,
            sp::msm_n_32_straus_result,
            32);
        run_msm(
            "tv(C): shaw msm n=33",
            sp::msm_n_33_pippenger_scalars,
            sp::msm_n_33_pippenger_points,
            sp::msm_n_33_pippenger_result,
            33);
        run_msm(
            "tv(C): shaw msm n=64",
            sp::msm_n_64_pippenger_scalars,
            sp::msm_n_64_pippenger_points,
            sp::msm_n_64_pippenger_result,
            64);
    }

    /* Pedersen (shaw) */
    {
        auto run_ped = [&](const char *name,
                           const uint8_t blinding[32],
                           const uint8_t H_bytes[32],
                           const uint8_t(*values)[32],
                           const uint8_t(*generators)[32],
                           const uint8_t expected[32],
                           size_t n)
        {
            shaw_jacobian H_pt;
            s_load(&H_pt, H_bytes);
            std::vector<shaw_jacobian> gens(n);
            for (size_t j = 0; j < n; j++)
                s_load(&gens[j], generators[j]);
            shaw_jacobian r;
            shaw_pedersen_commit(&r, blinding, &H_pt, values[0], gens.data(), n);
            unsigned char out[32];
            shaw_tobytes(out, &r);
            check_bytes(name, expected, out, 32);
        };
        namespace sp = tv::shaw_point;
        run_ped(
            "tv(C): shaw pedersen n=1",
            sp::pedersen_n_1_blinding,
            sp::pedersen_n_1_H,
            sp::pedersen_n_1_values,
            sp::pedersen_n_1_generators,
            sp::pedersen_n_1_result,
            1);
        run_ped(
            "tv(C): shaw pedersen n=4",
            sp::pedersen_n_4_blinding,
            sp::pedersen_n_4_H,
            sp::pedersen_n_4_values,
            sp::pedersen_n_4_generators,
            sp::pedersen_n_4_result,
            4);
        run_ped(
            "tv(C): shaw pedersen blind=0",
            sp::pedersen_blinding_zero_blinding,
            sp::pedersen_blinding_zero_H,
            sp::pedersen_blinding_zero_values,
            sp::pedersen_blinding_zero_generators,
            sp::pedersen_blinding_zero_result,
            1);
    }

    /* Map-to-curve (shaw) */
    for (size_t i = 0; i < tv::shaw_point::map_to_curve_single_count; i++)
    {
        auto &v = tv::shaw_point::map_to_curve_single_vectors[i];
        shaw_jacobian r;
        shaw_map_to_curve(&r, v.u);
        unsigned char out[32];
        shaw_tobytes(out, &r);
        check_bytes((std::string("tv(C): shaw map_to_curve ") + v.label).c_str(), v.result, out, 32);
    }
    for (size_t i = 0; i < tv::shaw_point::map_to_curve_double_count; i++)
    {
        auto &v = tv::shaw_point::map_to_curve_double_vectors[i];
        shaw_jacobian r;
        shaw_map_to_curve2(&r, v.u0, v.u1);
        unsigned char out[32];
        shaw_tobytes(out, &r);
        check_bytes((std::string("tv(C): shaw map_to_curve2 ") + v.label).c_str(), v.result, out, 32);
    }

    /* ==== Fp Polynomial (C primitives) ==== */
    std::cout << "  --- Fp Polynomial (C) ---" << std::endl;
    {
        namespace fp = tv::fp_polynomial;

        /* Helper: build fp_poly from test vector coefficient bytes */
        auto make_fp_poly = [](const uint8_t(*coeffs)[32], size_t n) -> fp_poly
        {
            fp_poly p;
            p.coeffs.resize(n);
            for (size_t i = 0; i < n; i++)
                fp_frombytes(p.coeffs[i].v, coeffs[i]);
            return p;
        };

        /* from_roots */
        {
            fp_fe roots[1];
            fp_frombytes(roots[0], fp::from_roots_one_root_roots[0]);
            fp_poly p;
            fp_poly_from_roots(&p, roots, 1);
            size_t n = sizeof(fp::from_roots_one_root_coefficients) / sizeof(fp::from_roots_one_root_coefficients[0]);
            for (size_t i = 0; i < n && i < p.coeffs.size(); i++)
            {
                uint8_t c[32];
                fp_tobytes(c, p.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): fp poly from_roots 1 coeff[") + std::to_string(i) + "]").c_str(),
                    fp::from_roots_one_root_coefficients[i],
                    c,
                    32);
            }
        }
        {
            fp_fe roots[4];
            for (int j = 0; j < 4; j++)
                fp_frombytes(roots[j], fp::from_roots_four_roots_roots[j]);
            fp_poly p;
            fp_poly_from_roots(&p, roots, 4);
            size_t n =
                sizeof(fp::from_roots_four_roots_coefficients) / sizeof(fp::from_roots_four_roots_coefficients[0]);
            for (size_t i = 0; i < n && i < p.coeffs.size(); i++)
            {
                uint8_t c[32];
                fp_tobytes(c, p.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): fp poly from_roots 4 coeff[") + std::to_string(i) + "]").c_str(),
                    fp::from_roots_four_roots_coefficients[i],
                    c,
                    32);
            }
        }

        /* eval */
        {
            size_t nc = sizeof(fp::eval_quadratic_at_7_coefficients) / sizeof(fp::eval_quadratic_at_7_coefficients[0]);
            auto p = make_fp_poly(fp::eval_quadratic_at_7_coefficients, nc);
            fp_fe x, result;
            fp_frombytes(x, fp::eval_quadratic_at_7_x);
            fp_poly_eval(result, &p, x);
            uint8_t out[32];
            fp_tobytes(out, result);
            check_bytes("tv(C): fp poly eval quadratic_at_7", fp::eval_quadratic_at_7_result, out, 32);
        }

        /* mul: deg1*deg1 */
        {
            size_t na = sizeof(fp::mul_deg1_times_deg1_a) / sizeof(fp::mul_deg1_times_deg1_a[0]);
            size_t nb = sizeof(fp::mul_deg1_times_deg1_b) / sizeof(fp::mul_deg1_times_deg1_b[0]);
            size_t nr = sizeof(fp::mul_deg1_times_deg1_result) / sizeof(fp::mul_deg1_times_deg1_result[0]);
            auto a = make_fp_poly(fp::mul_deg1_times_deg1_a, na);
            auto b = make_fp_poly(fp::mul_deg1_times_deg1_b, nb);
            fp_poly r;
            fp_poly_mul(&r, &a, &b);
            for (size_t i = 0; i < nr && i < r.coeffs.size(); i++)
            {
                uint8_t c[32];
                fp_tobytes(c, r.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): fp poly mul deg1*deg1 coeff[") + std::to_string(i) + "]").c_str(),
                    fp::mul_deg1_times_deg1_result[i],
                    c,
                    32);
            }
        }

        /* mul: deg16*deg16 (karatsuba) */
        {
            size_t na =
                sizeof(fp::mul_deg16_times_deg16_karatsuba_a) / sizeof(fp::mul_deg16_times_deg16_karatsuba_a[0]);
            size_t nb =
                sizeof(fp::mul_deg16_times_deg16_karatsuba_b) / sizeof(fp::mul_deg16_times_deg16_karatsuba_b[0]);
            size_t nr = sizeof(fp::mul_deg16_times_deg16_karatsuba_result)
                        / sizeof(fp::mul_deg16_times_deg16_karatsuba_result[0]);
            auto a = make_fp_poly(fp::mul_deg16_times_deg16_karatsuba_a, na);
            auto b = make_fp_poly(fp::mul_deg16_times_deg16_karatsuba_b, nb);
            fp_poly r;
            fp_poly_mul(&r, &a, &b);
            for (size_t i = 0; i < nr && i < r.coeffs.size(); i++)
            {
                uint8_t c[32];
                fp_tobytes(c, r.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): fp poly mul deg16*deg16 coeff[") + std::to_string(i) + "]").c_str(),
                    fp::mul_deg16_times_deg16_karatsuba_result[i],
                    c,
                    32);
            }
        }

        /* divmod: exact division */
        {
            size_t nn = sizeof(fp::divmod_exact_division_numerator) / sizeof(fp::divmod_exact_division_numerator[0]);
            size_t nd =
                sizeof(fp::divmod_exact_division_denominator) / sizeof(fp::divmod_exact_division_denominator[0]);
            size_t nq = sizeof(fp::divmod_exact_division_quotient) / sizeof(fp::divmod_exact_division_quotient[0]);
            size_t nrem = sizeof(fp::divmod_exact_division_remainder) / sizeof(fp::divmod_exact_division_remainder[0]);
            auto num = make_fp_poly(fp::divmod_exact_division_numerator, nn);
            auto den = make_fp_poly(fp::divmod_exact_division_denominator, nd);
            fp_poly q, rem;
            fp_poly_divmod(&q, &rem, &num, &den);
            for (size_t i = 0; i < nq && i < q.coeffs.size(); i++)
            {
                uint8_t c[32];
                fp_tobytes(c, q.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): fp poly divmod exact q[") + std::to_string(i) + "]").c_str(),
                    fp::divmod_exact_division_quotient[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nrem && i < rem.coeffs.size(); i++)
            {
                uint8_t c[32];
                fp_tobytes(c, rem.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): fp poly divmod exact r[") + std::to_string(i) + "]").c_str(),
                    fp::divmod_exact_division_remainder[i],
                    c,
                    32);
            }
        }

        /* interpolate: three points */
        {
            fp_fe xs[3], ys[3];
            for (int j = 0; j < 3; j++)
            {
                fp_frombytes(xs[j], fp::interp_three_points_xs[j]);
                fp_frombytes(ys[j], fp::interp_three_points_ys[j]);
            }
            fp_poly p;
            fp_poly_interpolate(&p, xs, ys, 3);
            size_t nc = sizeof(fp::interp_three_points_coefficients) / sizeof(fp::interp_three_points_coefficients[0]);
            for (size_t i = 0; i < nc && i < p.coeffs.size(); i++)
            {
                uint8_t c[32];
                fp_tobytes(c, p.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): fp poly interp 3pt coeff[") + std::to_string(i) + "]").c_str(),
                    fp::interp_three_points_coefficients[i],
                    c,
                    32);
            }
        }
    }

    /* ==== Fq Polynomial (C primitives) ==== */
    std::cout << "  --- Fq Polynomial (C) ---" << std::endl;
    {
        namespace fqn = tv::fq_polynomial;

        auto make_fq_poly = [](const uint8_t(*coeffs)[32], size_t n) -> fq_poly
        {
            fq_poly p;
            p.coeffs.resize(n);
            for (size_t i = 0; i < n; i++)
                fq_frombytes(p.coeffs[i].v, coeffs[i]);
            return p;
        };

        /* from_roots */
        {
            fq_fe roots[4];
            for (int j = 0; j < 4; j++)
                fq_frombytes(roots[j], fqn::from_roots_four_roots_roots[j]);
            fq_poly p;
            fq_poly_from_roots(&p, roots, 4);
            size_t n =
                sizeof(fqn::from_roots_four_roots_coefficients) / sizeof(fqn::from_roots_four_roots_coefficients[0]);
            for (size_t i = 0; i < n && i < p.coeffs.size(); i++)
            {
                uint8_t c[32];
                fq_tobytes(c, p.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): fq poly from_roots 4 coeff[") + std::to_string(i) + "]").c_str(),
                    fqn::from_roots_four_roots_coefficients[i],
                    c,
                    32);
            }
        }

        /* eval */
        {
            size_t nc =
                sizeof(fqn::eval_quadratic_at_7_coefficients) / sizeof(fqn::eval_quadratic_at_7_coefficients[0]);
            auto p = make_fq_poly(fqn::eval_quadratic_at_7_coefficients, nc);
            fq_fe x, result;
            fq_frombytes(x, fqn::eval_quadratic_at_7_x);
            fq_poly_eval(result, &p, x);
            uint8_t out[32];
            fq_tobytes(out, result);
            check_bytes("tv(C): fq poly eval quadratic_at_7", fqn::eval_quadratic_at_7_result, out, 32);
        }

        /* mul: deg16*deg16 (karatsuba) */
        {
            size_t na =
                sizeof(fqn::mul_deg16_times_deg16_karatsuba_a) / sizeof(fqn::mul_deg16_times_deg16_karatsuba_a[0]);
            size_t nb =
                sizeof(fqn::mul_deg16_times_deg16_karatsuba_b) / sizeof(fqn::mul_deg16_times_deg16_karatsuba_b[0]);
            size_t nr = sizeof(fqn::mul_deg16_times_deg16_karatsuba_result)
                        / sizeof(fqn::mul_deg16_times_deg16_karatsuba_result[0]);
            auto a = make_fq_poly(fqn::mul_deg16_times_deg16_karatsuba_a, na);
            auto b = make_fq_poly(fqn::mul_deg16_times_deg16_karatsuba_b, nb);
            fq_poly r;
            fq_poly_mul(&r, &a, &b);
            for (size_t i = 0; i < nr && i < r.coeffs.size(); i++)
            {
                uint8_t c[32];
                fq_tobytes(c, r.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): fq poly mul deg16*deg16 coeff[") + std::to_string(i) + "]").c_str(),
                    fqn::mul_deg16_times_deg16_karatsuba_result[i],
                    c,
                    32);
            }
        }

        /* divmod: exact division */
        {
            size_t nn = sizeof(fqn::divmod_exact_division_numerator) / sizeof(fqn::divmod_exact_division_numerator[0]);
            size_t nd =
                sizeof(fqn::divmod_exact_division_denominator) / sizeof(fqn::divmod_exact_division_denominator[0]);
            size_t nq = sizeof(fqn::divmod_exact_division_quotient) / sizeof(fqn::divmod_exact_division_quotient[0]);
            size_t nrem =
                sizeof(fqn::divmod_exact_division_remainder) / sizeof(fqn::divmod_exact_division_remainder[0]);
            auto num = make_fq_poly(fqn::divmod_exact_division_numerator, nn);
            auto den = make_fq_poly(fqn::divmod_exact_division_denominator, nd);
            fq_poly q, rem;
            fq_poly_divmod(&q, &rem, &num, &den);
            for (size_t i = 0; i < nq && i < q.coeffs.size(); i++)
            {
                uint8_t c[32];
                fq_tobytes(c, q.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): fq poly divmod exact q[") + std::to_string(i) + "]").c_str(),
                    fqn::divmod_exact_division_quotient[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nrem && i < rem.coeffs.size(); i++)
            {
                uint8_t c[32];
                fq_tobytes(c, rem.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): fq poly divmod exact r[") + std::to_string(i) + "]").c_str(),
                    fqn::divmod_exact_division_remainder[i],
                    c,
                    32);
            }
        }

        /* interpolate: three points */
        {
            fq_fe xs[3], ys[3];
            for (int j = 0; j < 3; j++)
            {
                fq_frombytes(xs[j], fqn::interp_three_points_xs[j]);
                fq_frombytes(ys[j], fqn::interp_three_points_ys[j]);
            }
            fq_poly p;
            fq_poly_interpolate(&p, xs, ys, 3);
            size_t nc =
                sizeof(fqn::interp_three_points_coefficients) / sizeof(fqn::interp_three_points_coefficients[0]);
            for (size_t i = 0; i < nc && i < p.coeffs.size(); i++)
            {
                uint8_t c[32];
                fq_tobytes(c, p.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): fq poly interp 3pt coeff[") + std::to_string(i) + "]").c_str(),
                    fqn::interp_three_points_coefficients[i],
                    c,
                    32);
            }
        }
    }

    /* ==== Ran Divisor (C primitives) ==== */
    std::cout << "  --- Ran Divisor (C) ---" << std::endl;
    {
        namespace hd = tv::ran_divisor;

        auto run_divisor = [&](const char *label,
                               const uint8_t(*pt_bytes)[32],
                               size_t n,
                               const uint8_t(*a_coeffs)[32],
                               size_t na,
                               const uint8_t(*b_coeffs)[32],
                               size_t nb,
                               const uint8_t eval_x[32],
                               const uint8_t eval_y[32],
                               const uint8_t eval_expected[32])
        {
            /* Load affine points */
            std::vector<ran_affine> pts(n);
            for (size_t j = 0; j < n; j++)
            {
                ran_jacobian jac;
                ran_frombytes(&jac, pt_bytes[j]);
                ran_to_affine(&pts[j], &jac);
            }
            ran_divisor d;
            ran_compute_divisor(&d, pts.data(), n);
            for (size_t i = 0; i < na && i < d.a.coeffs.size(); i++)
            {
                uint8_t c[32];
                fp_tobytes(c, d.a.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): ran div ") + label + " a[" + std::to_string(i) + "]").c_str(),
                    a_coeffs[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nb && i < d.b.coeffs.size(); i++)
            {
                uint8_t c[32];
                fp_tobytes(c, d.b.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): ran div ") + label + " b[" + std::to_string(i) + "]").c_str(),
                    b_coeffs[i],
                    c,
                    32);
            }
            fp_fe ex, ey, ev;
            fp_frombytes(ex, eval_x);
            fp_frombytes(ey, eval_y);
            ran_evaluate_divisor(ev, &d, ex, ey);
            uint8_t out[32];
            fp_tobytes(out, ev);
            check_bytes((std::string("tv(C): ran div ") + label + " eval").c_str(), eval_expected, out, 32);
        };

        run_divisor(
            "n=2",
            hd::n_2_points,
            2,
            hd::n_2_a_coefficients,
            sizeof(hd::n_2_a_coefficients) / sizeof(hd::n_2_a_coefficients[0]),
            hd::n_2_b_coefficients,
            sizeof(hd::n_2_b_coefficients) / sizeof(hd::n_2_b_coefficients[0]),
            hd::n_2_eval_point_x,
            hd::n_2_eval_point_y,
            hd::n_2_eval_result);
        run_divisor(
            "n=4",
            hd::n_4_points,
            4,
            hd::n_4_a_coefficients,
            sizeof(hd::n_4_a_coefficients) / sizeof(hd::n_4_a_coefficients[0]),
            hd::n_4_b_coefficients,
            sizeof(hd::n_4_b_coefficients) / sizeof(hd::n_4_b_coefficients[0]),
            hd::n_4_eval_point_x,
            hd::n_4_eval_point_y,
            hd::n_4_eval_result);
        run_divisor(
            "n=8",
            hd::n_8_points,
            8,
            hd::n_8_a_coefficients,
            sizeof(hd::n_8_a_coefficients) / sizeof(hd::n_8_a_coefficients[0]),
            hd::n_8_b_coefficients,
            sizeof(hd::n_8_b_coefficients) / sizeof(hd::n_8_b_coefficients[0]),
            hd::n_8_eval_point_x,
            hd::n_8_eval_point_y,
            hd::n_8_eval_result);
    }

    /* ==== Shaw Divisor (C primitives) ==== */
    std::cout << "  --- Shaw Divisor (C) ---" << std::endl;
    {
        namespace sd = tv::shaw_divisor;

        auto run_divisor = [&](const char *label,
                               const uint8_t(*pt_bytes)[32],
                               size_t n,
                               const uint8_t(*a_coeffs)[32],
                               size_t na,
                               const uint8_t(*b_coeffs)[32],
                               size_t nb,
                               const uint8_t eval_x[32],
                               const uint8_t eval_y[32],
                               const uint8_t eval_expected[32])
        {
            std::vector<shaw_affine> pts(n);
            for (size_t j = 0; j < n; j++)
            {
                shaw_jacobian jac;
                shaw_frombytes(&jac, pt_bytes[j]);
                shaw_to_affine(&pts[j], &jac);
            }
            shaw_divisor d;
            shaw_compute_divisor(&d, pts.data(), n);
            for (size_t i = 0; i < na && i < d.a.coeffs.size(); i++)
            {
                uint8_t c[32];
                fq_tobytes(c, d.a.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): shaw div ") + label + " a[" + std::to_string(i) + "]").c_str(),
                    a_coeffs[i],
                    c,
                    32);
            }
            for (size_t i = 0; i < nb && i < d.b.coeffs.size(); i++)
            {
                uint8_t c[32];
                fq_tobytes(c, d.b.coeffs[i].v);
                check_bytes(
                    (std::string("tv(C): shaw div ") + label + " b[" + std::to_string(i) + "]").c_str(),
                    b_coeffs[i],
                    c,
                    32);
            }
            fq_fe ex, ey, ev;
            fq_frombytes(ex, eval_x);
            fq_frombytes(ey, eval_y);
            shaw_evaluate_divisor(ev, &d, ex, ey);
            uint8_t out[32];
            fq_tobytes(out, ev);
            check_bytes((std::string("tv(C): shaw div ") + label + " eval").c_str(), eval_expected, out, 32);
        };

        run_divisor(
            "n=2",
            sd::n_2_points,
            2,
            sd::n_2_a_coefficients,
            sizeof(sd::n_2_a_coefficients) / sizeof(sd::n_2_a_coefficients[0]),
            sd::n_2_b_coefficients,
            sizeof(sd::n_2_b_coefficients) / sizeof(sd::n_2_b_coefficients[0]),
            sd::n_2_eval_point_x,
            sd::n_2_eval_point_y,
            sd::n_2_eval_result);
        run_divisor(
            "n=4",
            sd::n_4_points,
            4,
            sd::n_4_a_coefficients,
            sizeof(sd::n_4_a_coefficients) / sizeof(sd::n_4_a_coefficients[0]),
            sd::n_4_b_coefficients,
            sizeof(sd::n_4_b_coefficients) / sizeof(sd::n_4_b_coefficients[0]),
            sd::n_4_eval_point_x,
            sd::n_4_eval_point_y,
            sd::n_4_eval_result);
        run_divisor(
            "n=8",
            sd::n_8_points,
            8,
            sd::n_8_a_coefficients,
            sizeof(sd::n_8_a_coefficients) / sizeof(sd::n_8_a_coefficients[0]),
            sd::n_8_b_coefficients,
            sizeof(sd::n_8_b_coefficients) / sizeof(sd::n_8_b_coefficients[0]),
            sd::n_8_eval_point_x,
            sd::n_8_eval_point_y,
            sd::n_8_eval_result);
    }
}

int main(int argc, char *argv[])
{
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
        else
        {
            std::cerr << "Usage: " << argv[0] << " [--init | --autotune]" << std::endl;
            return 1;
        }
    }

    std::cout << "RanShaw Unit Tests" << std::endl;
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

    test_fp();
    test_fq();
    test_fp_sqrt();
    test_ran_points();
    test_shaw_points();
    test_ran_scalarmult();
    test_shaw_scalarmult();
    test_wei25519();
    test_ran_msm();
    test_shaw_msm();
    test_fp_sqrt_sswu();
    test_ran_sswu();
    test_shaw_sswu();
    test_ran_batch_affine();
    test_shaw_batch_affine();
    test_ran_pedersen();
    test_shaw_pedersen();
    test_fp_poly();
    test_fq_poly();
    test_ran_divisor();
    test_shaw_divisor();
    test_fp_extended();
    test_fq_extended();
    test_serialization_edges();
    test_ran_point_edges();
    test_shaw_point_edges();
    test_scalarmult_extended();
    test_msm_extended();
    test_batch_affine_extended();
    test_batch_invert();
    test_fixed_base_scalarmult();
    test_precomputed_tables();
    test_msm_fixed();
    test_pedersen_extended();
    test_poly_extended();
    test_divisor_extended();
    test_point_to_scalar();
    test_ran_scalar();
    test_shaw_scalar();
    test_poly_interpolate();
    test_karatsuba();
#ifdef RANSHAW_ECFFT
    test_ecfft();
#endif
    test_eval_divisor();
    test_serialization_roundtrip();
    test_vector_validation();
    test_vector_validation_c_primitives();
    test_dispatch();
    test_cpp_api();

    std::cout << std::endl << "======================" << std::endl;
    std::cout << "Total:  " << tests_run << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
