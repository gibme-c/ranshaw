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
 * @file gen_test_vectors.cpp
 * @brief Deterministic test vector generator for downstream consumers.
 *
 * Outputs canonical JSON to stdout covering all public API operations
 * for both Ran and Shaw curves. All inputs are hardcoded for
 * reproducibility — no randomness.
 *
 * Usage:
 *   ranshaw-gen-testvectors > test_vectors/ranshaw_test_vectors.json
 *
 * MUST be built with -DFORCE_PORTABLE=1 to ensure vectors come from
 * the reference (portable) backend, not an optimized SIMD path.
 */

#include "ranshaw.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ranshaw;

/* ── JSON helpers ── */

static int indent_level = 0;

static void emit_indent()
{
    for (int i = 0; i < indent_level; i++)
        printf("  ");
}

static std::string hex_str(const uint8_t *data, size_t len)
{
    std::string s;
    s.reserve(len * 2);
    for (size_t i = 0; i < len; i++)
    {
        char buf[3];
        snprintf(buf, sizeof(buf), "%02x", data[i]);
        s += buf;
    }
    return s;
}

static void emit_hex(const char *key, const uint8_t *data, size_t len, bool last = false)
{
    emit_indent();
    printf("\"%s\": \"%s\"%s\n", key, hex_str(data, len).c_str(), last ? "" : ",");
}

static void emit_hex_arr32(const char *key, const std::array<uint8_t, 32> &arr, bool last = false)
{
    emit_hex(key, arr.data(), 32, last);
}

static void emit_bool(const char *key, bool val, bool last = false)
{
    emit_indent();
    printf("\"%s\": %s%s\n", key, val ? "true" : "false", last ? "" : ",");
}

static void emit_int(const char *key, int val, bool last = false)
{
    emit_indent();
    printf("\"%s\": %d%s\n", key, val, last ? "" : ",");
}

static void emit_null(const char *key, bool last = false)
{
    emit_indent();
    printf("\"%s\": null%s\n", key, last ? "" : ",");
}

static void emit_string(const char *key, const char *val, bool last = false)
{
    emit_indent();
    printf("\"%s\": \"%s\"%s\n", key, val, last ? "" : ",");
}

static void open_obj(const char *key = nullptr)
{
    emit_indent();
    if (key)
        printf("\"%s\": {\n", key);
    else
        printf("{\n");
    indent_level++;
}

static void close_obj(bool last = false)
{
    indent_level--;
    emit_indent();
    printf("}%s\n", last ? "" : ",");
}

static void open_arr(const char *key)
{
    emit_indent();
    printf("\"%s\": [\n", key);
    indent_level++;
}

static void close_arr(bool last = false)
{
    indent_level--;
    emit_indent();
    printf("]%s\n", last ? "" : ",");
}

/* Helper: emit a JSON array of hex-encoded 32-byte elements from a flat buffer */
static void emit_hex_array(const char *key, const uint8_t *data, int count, bool last = false)
{
    open_arr(key);
    for (int j = 0; j < count; j++)
    {
        emit_indent();
        printf("\"%s\"%s\n", hex_str(data + j * 32, 32).c_str(), j == count - 1 ? "" : ",");
    }
    close_arr(last);
}

/* Helper: emit a JSON array of hex-encoded coefficients from an fp_poly */
static void emit_fp_poly_coeffs(const char *key, const fp_poly &poly, bool last = false)
{
    open_arr(key);
    for (size_t j = 0; j < poly.coeffs.size(); j++)
    {
        uint8_t buf[32];
        fp_tobytes(buf, poly.coeffs[j].v);
        emit_indent();
        printf("\"%s\"%s\n", hex_str(buf, 32).c_str(), j == poly.coeffs.size() - 1 ? "" : ",");
    }
    close_arr(last);
}

/* Helper: emit a JSON array of hex-encoded coefficients from an fq_poly */
static void emit_fq_poly_coeffs(const char *key, const fq_poly &poly, bool last = false)
{
    open_arr(key);
    for (size_t j = 0; j < poly.coeffs.size(); j++)
    {
        uint8_t buf[32];
        fq_tobytes(buf, poly.coeffs[j].v);
        emit_indent();
        printf("\"%s\"%s\n", hex_str(buf, 32).c_str(), j == poly.coeffs.size() - 1 ? "" : ",");
    }
    close_arr(last);
}

/* ── Deterministic test inputs ── */

static const uint8_t test_a_bytes[32] = {0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, 0xbe, 0xba, 0xfe,
                                         0xca, 0xef, 0xbe, 0xad, 0xde, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

static const uint8_t test_b_bytes[32] = {0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x0d, 0xf0, 0xad,
                                         0xba, 0xce, 0xfa, 0xed, 0xfe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

static const uint8_t zero_bytes[32] = {0};
static const uint8_t one_bytes[32] = {1};
static const uint8_t two_bytes[32] = {2};
static const uint8_t seven_bytes[32] = {7};
static const uint8_t fortytwo_bytes[32] = {42};

/* order - 1 for each curve (computed at init time) */
static uint8_t ran_order_m1[32];
static uint8_t shaw_order_m1[32];

/* all-0xFF (invalid scalar for both curves) */
static const uint8_t all_ff_bytes[32] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

/* 64-byte wide reduction inputs */
static const uint8_t wide_zero[64] = {0};
static const uint8_t wide_small[64] = {42};
static const uint8_t wide_large[64] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                       0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
static const uint8_t wide_hash[64] = {0x48, 0x65, 0x6c, 0x69, 0x6f, 0x73, 0x65, 0x6c, 0x65, 0x6e, 0x65, 0x5f, 0x74,
                                      0x65, 0x73, 0x74, 0x5f, 0x76, 0x65, 0x63, 0x74, 0x6f, 0x72, 0x5f, 0x30, 0x30,
                                      0x30, 0x31, 0x00, 0x00, 0x00, 0x00, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67,
                                      0x89, 0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe, 0x01, 0x02, 0x03, 0x04,
                                      0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10};

/* off-curve point (valid x but wrong parity to produce invalid decompression) */
static const uint8_t off_curve_bytes[32] = {0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

/* x >= p (non-canonical field element) */
static const uint8_t x_ge_p_bytes[32] = {0xee, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f};

static void compute_order_minus_one()
{
    /* RAN_ORDER - 1 */
    memcpy(ran_order_m1, RAN_ORDER, 32);
    /* subtract 1 with borrow */
    int borrow = 1;
    for (int i = 0; i < 32 && borrow; i++)
    {
        int v = (int)ran_order_m1[i] - borrow;
        ran_order_m1[i] = (uint8_t)(v & 0xff);
        borrow = (v < 0) ? 1 : 0;
    }

    memcpy(shaw_order_m1, SHAW_ORDER, 32);
    borrow = 1;
    for (int i = 0; i < 32 && borrow; i++)
    {
        int v = (int)shaw_order_m1[i] - borrow;
        shaw_order_m1[i] = (uint8_t)(v & 0xff);
        borrow = (v < 0) ? 1 : 0;
    }
}

/* Small scalar from integer */
static void small_scalar_bytes(uint8_t out[32], uint64_t val)
{
    memset(out, 0, 32);
    for (int i = 0; i < 8; i++)
        out[i] = (uint8_t)(val >> (8 * i));
}

/* ── Ran scalar vectors ── */

static void emit_ran_scalar()
{
    open_obj("ran_scalar");

    /* from_bytes */
    open_arr("from_bytes");
    {
        auto cases = std::vector<std::pair<const char *, const uint8_t *>> {
            {"zero", zero_bytes},
            {"one", one_bytes},
            {"fortytwo", fortytwo_bytes},
            {"test_a", test_a_bytes},
            {"order_minus_1", ran_order_m1},
            {"order", RAN_ORDER},
            {"order_plus_1", nullptr}, /* special */
            {"all_ff", all_ff_bytes}};

        uint8_t order_plus_1[32];
        memcpy(order_plus_1, RAN_ORDER, 32);
        int carry = 1;
        for (int i = 0; i < 32 && carry; i++)
        {
            int v = (int)order_plus_1[i] + carry;
            order_plus_1[i] = (uint8_t)(v & 0xff);
            carry = (v >> 8) & 1;
        }
        cases[6].second = order_plus_1;

        for (size_t i = 0; i < cases.size(); i++)
        {
            open_obj();
            emit_string("label", cases[i].first);
            emit_hex("input", cases[i].second, 32);
            auto s = RanScalar::from_bytes(cases[i].second);
            if (s)
                emit_hex_arr32("result", s->to_bytes(), true);
            else
                emit_null("result", true);
            close_obj(i == cases.size() - 1);
        }
    }
    close_arr();

    /* add */
    open_arr("add");
    {
        auto a = *RanScalar::from_bytes(test_a_bytes);
        auto b = *RanScalar::from_bytes(test_b_bytes);
        auto z = RanScalar::zero();
        auto om1 = *RanScalar::from_bytes(ran_order_m1);
        auto one = RanScalar::one();

        struct
        {
            const char *label;
            RanScalar x, y;
        } cases[] = {
            {"a_plus_b", a, b},
            {"a_plus_zero", a, z},
            {"a_plus_neg_a_eq_zero", a, -a},
            {"order_m1_plus_1_eq_zero", om1, one},
            {"one_plus_one", one, one},
        };
        for (size_t i = 0; i < 5; i++)
        {
            auto r = cases[i].x + cases[i].y;
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_hex_arr32("b", cases[i].y.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 4);
        }
    }
    close_arr();

    /* sub */
    open_arr("sub");
    {
        auto a = *RanScalar::from_bytes(test_a_bytes);
        auto b = *RanScalar::from_bytes(test_b_bytes);
        auto z = RanScalar::zero();

        struct
        {
            const char *label;
            RanScalar x, y;
        } cases[] = {
            {"a_minus_b", a, b},
            {"a_minus_zero", a, z},
            {"zero_minus_a", z, a},
            {"a_minus_a", a, a},
        };
        for (size_t i = 0; i < 4; i++)
        {
            auto r = cases[i].x - cases[i].y;
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_hex_arr32("b", cases[i].y.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 3);
        }
    }
    close_arr();

    /* mul */
    open_arr("mul");
    {
        auto a = *RanScalar::from_bytes(test_a_bytes);
        auto b = *RanScalar::from_bytes(test_b_bytes);
        auto z = RanScalar::zero();
        auto one = RanScalar::one();
        auto om1 = *RanScalar::from_bytes(ran_order_m1);
        auto seven = *RanScalar::from_bytes(seven_bytes);
        auto ft = *RanScalar::from_bytes(fortytwo_bytes);

        struct
        {
            const char *label;
            RanScalar x, y;
        } cases[] = {
            {"a_times_b", a, b},
            {"a_times_one", a, one},
            {"a_times_zero", a, z},
            {"order_m1_times_order_m1", om1, om1},
            {"seven_times_fortytwo", seven, ft},
        };
        for (size_t i = 0; i < 5; i++)
        {
            auto r = cases[i].x * cases[i].y;
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_hex_arr32("b", cases[i].y.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 4);
        }
    }
    close_arr();

    /* sq */
    open_arr("sq");
    {
        auto a = *RanScalar::from_bytes(test_a_bytes);
        auto one = RanScalar::one();
        auto om1 = *RanScalar::from_bytes(ran_order_m1);

        struct
        {
            const char *label;
            RanScalar x;
        } cases[] = {
            {"a_squared", a},
            {"one_squared", one},
            {"order_m1_squared", om1},
        };
        for (size_t i = 0; i < 3; i++)
        {
            auto r = cases[i].x.sq();
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 2);
        }
    }
    close_arr();

    /* negate */
    open_arr("negate");
    {
        auto z = RanScalar::zero();
        auto one = RanScalar::one();
        auto a = *RanScalar::from_bytes(test_a_bytes);
        auto om1 = *RanScalar::from_bytes(ran_order_m1);

        struct
        {
            const char *label;
            RanScalar x;
        } cases[] = {
            {"neg_zero", z},
            {"neg_one", one},
            {"neg_a", a},
            {"neg_order_m1", om1},
        };
        for (size_t i = 0; i < 4; i++)
        {
            auto r = -cases[i].x;
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 3);
        }
    }
    close_arr();

    /* invert */
    open_arr("invert");
    {
        auto a = *RanScalar::from_bytes(test_a_bytes);
        auto one = RanScalar::one();
        auto om1 = *RanScalar::from_bytes(ran_order_m1);
        auto z = RanScalar::zero();

        struct
        {
            const char *label;
            RanScalar x;
        } cases[] = {
            {"inv_a", a},
            {"inv_one", one},
            {"inv_order_m1", om1},
            {"inv_zero", z},
        };
        for (size_t i = 0; i < 4; i++)
        {
            auto r = cases[i].x.invert();
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            if (r)
                emit_hex_arr32("result", r->to_bytes(), true);
            else
                emit_null("result", true);
            close_obj(i == 3);
        }
    }
    close_arr();

    /* reduce_wide */
    open_arr("reduce_wide");
    {
        struct
        {
            const char *label;
            const uint8_t *input;
        } cases[] = {
            {"all_zero", wide_zero},
            {"small", wide_small},
            {"large", wide_large},
            {"hash_derived", wide_hash},
        };
        for (size_t i = 0; i < 4; i++)
        {
            auto r = RanScalar::reduce_wide(cases[i].input);
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex("input", cases[i].input, 64);
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 3);
        }
    }
    close_arr();

    /* muladd */
    open_arr("muladd");
    {
        auto a = *RanScalar::from_bytes(test_a_bytes);
        auto b = *RanScalar::from_bytes(test_b_bytes);
        auto one = RanScalar::one();
        auto seven = *RanScalar::from_bytes(seven_bytes);
        auto ft = *RanScalar::from_bytes(fortytwo_bytes);

        struct
        {
            const char *label;
            RanScalar x, y, z;
        } cases[] = {
            {"a_times_b_plus_one", a, b, one},
            {"seven_times_ft_plus_a", seven, ft, a},
            {"one_times_one_plus_one", one, one, one},
        };
        for (size_t i = 0; i < 3; i++)
        {
            auto r = RanScalar::muladd(cases[i].x, cases[i].y, cases[i].z);
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_hex_arr32("b", cases[i].y.to_bytes());
            emit_hex_arr32("c", cases[i].z.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 2);
        }
    }
    close_arr();

    /* is_zero */
    open_arr("is_zero");
    {
        auto z = RanScalar::zero();
        auto one = RanScalar::one();
        auto a = *RanScalar::from_bytes(test_a_bytes);

        struct
        {
            const char *label;
            RanScalar x;
            bool expected;
        } cases[] = {
            {"zero_is_zero", z, true},
            {"one_is_not_zero", one, false},
            {"a_is_not_zero", a, false},
        };
        for (size_t i = 0; i < 3; i++)
        {
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_bool("result", cases[i].expected, true);
            close_obj(i == 2);
        }
    }
    close_arr(true);

    close_obj(); /* ran_scalar */
}

/* ── Shaw scalar vectors (symmetric to Ran) ── */

static void emit_shaw_scalar()
{
    open_obj("shaw_scalar");

    /* from_bytes */
    open_arr("from_bytes");
    {
        uint8_t order_plus_1[32];
        memcpy(order_plus_1, SHAW_ORDER, 32);
        int carry = 1;
        for (int i = 0; i < 32 && carry; i++)
        {
            int v = (int)order_plus_1[i] + carry;
            order_plus_1[i] = (uint8_t)(v & 0xff);
            carry = (v >> 8) & 1;
        }

        struct
        {
            const char *label;
            const uint8_t *input;
        } cases[] = {
            {"zero", zero_bytes},
            {"one", one_bytes},
            {"fortytwo", fortytwo_bytes},
            {"test_a", test_a_bytes},
            {"order_minus_1", shaw_order_m1},
            {"order", SHAW_ORDER},
            {"order_plus_1", order_plus_1},
            {"all_ff", all_ff_bytes},
        };
        for (size_t i = 0; i < 8; i++)
        {
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex("input", cases[i].input, 32);
            auto s = ShawScalar::from_bytes(cases[i].input);
            if (s)
                emit_hex_arr32("result", s->to_bytes(), true);
            else
                emit_null("result", true);
            close_obj(i == 7);
        }
    }
    close_arr();

    /* add */
    open_arr("add");
    {
        auto a = *ShawScalar::from_bytes(test_a_bytes);
        auto b = *ShawScalar::from_bytes(test_b_bytes);
        auto z = ShawScalar::zero();
        auto om1 = *ShawScalar::from_bytes(shaw_order_m1);
        auto one = ShawScalar::one();

        struct
        {
            const char *label;
            ShawScalar x, y;
        } cases[] = {
            {"a_plus_b", a, b},
            {"a_plus_zero", a, z},
            {"a_plus_neg_a_eq_zero", a, -a},
            {"order_m1_plus_1_eq_zero", om1, one},
            {"one_plus_one", one, one},
        };
        for (size_t i = 0; i < 5; i++)
        {
            auto r = cases[i].x + cases[i].y;
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_hex_arr32("b", cases[i].y.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 4);
        }
    }
    close_arr();

    /* sub */
    open_arr("sub");
    {
        auto a = *ShawScalar::from_bytes(test_a_bytes);
        auto b = *ShawScalar::from_bytes(test_b_bytes);
        auto z = ShawScalar::zero();

        struct
        {
            const char *label;
            ShawScalar x, y;
        } cases[] = {
            {"a_minus_b", a, b},
            {"a_minus_zero", a, z},
            {"zero_minus_a", z, a},
            {"a_minus_a", a, a},
        };
        for (size_t i = 0; i < 4; i++)
        {
            auto r = cases[i].x - cases[i].y;
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_hex_arr32("b", cases[i].y.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 3);
        }
    }
    close_arr();

    /* mul */
    open_arr("mul");
    {
        auto a = *ShawScalar::from_bytes(test_a_bytes);
        auto b = *ShawScalar::from_bytes(test_b_bytes);
        auto z = ShawScalar::zero();
        auto one = ShawScalar::one();
        auto om1 = *ShawScalar::from_bytes(shaw_order_m1);
        auto seven = *ShawScalar::from_bytes(seven_bytes);
        auto ft = *ShawScalar::from_bytes(fortytwo_bytes);

        struct
        {
            const char *label;
            ShawScalar x, y;
        } cases[] = {
            {"a_times_b", a, b},
            {"a_times_one", a, one},
            {"a_times_zero", a, z},
            {"order_m1_times_order_m1", om1, om1},
            {"seven_times_fortytwo", seven, ft},
        };
        for (size_t i = 0; i < 5; i++)
        {
            auto r = cases[i].x * cases[i].y;
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_hex_arr32("b", cases[i].y.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 4);
        }
    }
    close_arr();

    /* sq */
    open_arr("sq");
    {
        auto a = *ShawScalar::from_bytes(test_a_bytes);
        auto one = ShawScalar::one();
        auto om1 = *ShawScalar::from_bytes(shaw_order_m1);

        struct
        {
            const char *label;
            ShawScalar x;
        } cases[] = {
            {"a_squared", a},
            {"one_squared", one},
            {"order_m1_squared", om1},
        };
        for (size_t i = 0; i < 3; i++)
        {
            auto r = cases[i].x.sq();
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 2);
        }
    }
    close_arr();

    /* negate */
    open_arr("negate");
    {
        auto z = ShawScalar::zero();
        auto one = ShawScalar::one();
        auto a = *ShawScalar::from_bytes(test_a_bytes);
        auto om1 = *ShawScalar::from_bytes(shaw_order_m1);

        struct
        {
            const char *label;
            ShawScalar x;
        } cases[] = {
            {"neg_zero", z},
            {"neg_one", one},
            {"neg_a", a},
            {"neg_order_m1", om1},
        };
        for (size_t i = 0; i < 4; i++)
        {
            auto r = -cases[i].x;
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 3);
        }
    }
    close_arr();

    /* invert */
    open_arr("invert");
    {
        auto a = *ShawScalar::from_bytes(test_a_bytes);
        auto one = ShawScalar::one();
        auto om1 = *ShawScalar::from_bytes(shaw_order_m1);
        auto z = ShawScalar::zero();

        struct
        {
            const char *label;
            ShawScalar x;
        } cases[] = {
            {"inv_a", a},
            {"inv_one", one},
            {"inv_order_m1", om1},
            {"inv_zero", z},
        };
        for (size_t i = 0; i < 4; i++)
        {
            auto r = cases[i].x.invert();
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            if (r)
                emit_hex_arr32("result", r->to_bytes(), true);
            else
                emit_null("result", true);
            close_obj(i == 3);
        }
    }
    close_arr();

    /* reduce_wide */
    open_arr("reduce_wide");
    {
        struct
        {
            const char *label;
            const uint8_t *input;
        } cases[] = {
            {"all_zero", wide_zero},
            {"small", wide_small},
            {"large", wide_large},
            {"hash_derived", wide_hash},
        };
        for (size_t i = 0; i < 4; i++)
        {
            auto r = ShawScalar::reduce_wide(cases[i].input);
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex("input", cases[i].input, 64);
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 3);
        }
    }
    close_arr();

    /* muladd */
    open_arr("muladd");
    {
        auto a = *ShawScalar::from_bytes(test_a_bytes);
        auto b = *ShawScalar::from_bytes(test_b_bytes);
        auto one = ShawScalar::one();
        auto seven = *ShawScalar::from_bytes(seven_bytes);
        auto ft = *ShawScalar::from_bytes(fortytwo_bytes);

        struct
        {
            const char *label;
            ShawScalar x, y, z;
        } cases[] = {
            {"a_times_b_plus_one", a, b, one},
            {"seven_times_ft_plus_a", seven, ft, a},
            {"one_times_one_plus_one", one, one, one},
        };
        for (size_t i = 0; i < 3; i++)
        {
            auto r = ShawScalar::muladd(cases[i].x, cases[i].y, cases[i].z);
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_hex_arr32("b", cases[i].y.to_bytes());
            emit_hex_arr32("c", cases[i].z.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 2);
        }
    }
    close_arr();

    /* is_zero */
    open_arr("is_zero");
    {
        auto z = ShawScalar::zero();
        auto one = ShawScalar::one();
        auto a = *ShawScalar::from_bytes(test_a_bytes);

        struct
        {
            const char *label;
            ShawScalar x;
            bool expected;
        } cases[] = {
            {"zero_is_zero", z, true},
            {"one_is_not_zero", one, false},
            {"a_is_not_zero", a, false},
        };
        for (size_t i = 0; i < 3; i++)
        {
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].x.to_bytes());
            emit_bool("result", cases[i].expected, true);
            close_obj(i == 2);
        }
    }
    close_arr(true);

    close_obj(); /* shaw_scalar */
}

/* ── Ran point vectors ── */

static void emit_ran_point()
{
    open_obj("ran_point");

    auto G = RanPoint::generator();
    auto O = RanPoint::identity();

    /* generator & identity */
    emit_hex_arr32("generator", G.to_bytes());
    emit_hex_arr32("identity", O.to_bytes());

    /* from_bytes */
    open_arr("from_bytes");
    {
        auto g_bytes = G.to_bytes();
        auto g2 = G.dbl();
        auto g2_bytes = g2.to_bytes();
        /* flip y-parity of G for a valid alternate decompression */
        uint8_t g_flip[32];
        memcpy(g_flip, g_bytes.data(), 32);
        g_flip[31] ^= 0x80;

        struct
        {
            const char *label;
            const uint8_t *input;
        } cases[] = {
            {"generator", g_bytes.data()},
            {"double_generator", g2_bytes.data()},
            {"off_curve", off_curve_bytes},
            {"x_ge_p", x_ge_p_bytes},
            {"flipped_parity", g_flip},
            {"identity_encoding", zero_bytes},
        };
        for (size_t i = 0; i < 6; i++)
        {
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex("input", cases[i].input, 32);
            auto p = RanPoint::from_bytes(cases[i].input);
            if (p)
                emit_hex_arr32("result", p->to_bytes(), true);
            else
                emit_null("result", true);
            close_obj(i == 5);
        }
    }
    close_arr();

    /* add — including identity edge cases */
    open_arr("add");
    {
        auto s2 = *RanScalar::from_bytes(two_bytes);
        auto s3 = *RanScalar::from_bytes(seven_bytes); /* 7, not 3, for variety */
        auto s4 = *RanScalar::from_bytes(fortytwo_bytes);
        auto g2 = G.scalar_mul(s2);
        auto g7 = G.scalar_mul(s3);
        auto g42 = G.scalar_mul(s4);

        /* Use scalar_mul to compute expected results independently */
        auto s_two = *RanScalar::from_bytes(two_bytes);
        auto s_seven = *RanScalar::from_bytes(seven_bytes);
        auto s_nine = s_two + s_seven; /* 2 + 7 = 9 */

        struct
        {
            const char *label;
            RanPoint a, b;
            RanPoint expected;
        } cases[] = {
            {"2G_plus_7G", g2, g7, G.scalar_mul(s_nine)},
            {"G_plus_42G", G, g42, G.scalar_mul(RanScalar::one() + s4)},
            {"7G_plus_42G", g7, g42, G.scalar_mul(s3 + s4)},
            {"P_plus_neg_P", G, -G, O},
            /* Identity shenanigans */
            {"O_plus_G", O, G, G},
            {"G_plus_O", G, O, G},
            {"O_plus_O", O, O, O},
            {"O_plus_7G", O, g7, g7},
            {"42G_plus_O", g42, O, g42},
            /* P + P (doubling via add) */
            {"G_plus_G", G, G, g2},
        };
        const size_t n_cases = sizeof(cases) / sizeof(cases[0]);
        for (size_t i = 0; i < n_cases; i++)
        {
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].a.to_bytes());
            emit_hex_arr32("b", cases[i].b.to_bytes());
            emit_hex_arr32("result", cases[i].expected.to_bytes(), true);
            close_obj(i == n_cases - 1);
        }
    }
    close_arr();

    /* dbl */
    open_arr("dbl");
    {
        auto s2 = *RanScalar::from_bytes(two_bytes);
        auto g2 = G.scalar_mul(s2);

        struct
        {
            const char *label;
            RanPoint a;
        } cases[] = {
            {"dbl_G", G},
            {"dbl_2G", g2},
            {"dbl_O", O},
        };
        for (size_t i = 0; i < 3; i++)
        {
            auto r = cases[i].a.dbl();
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].a.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 2);
        }
    }
    close_arr();

    /* negate */
    open_arr("negate");
    {
        auto s2 = *RanScalar::from_bytes(two_bytes);
        auto g2 = G.scalar_mul(s2);

        struct
        {
            const char *label;
            RanPoint a;
        } cases[] = {
            {"neg_G", G},
            {"neg_2G", g2},
            {"neg_O", O},
        };
        for (size_t i = 0; i < 3; i++)
        {
            auto r = -cases[i].a;
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].a.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 2);
        }
    }
    close_arr();

    /* scalar_mul */
    open_arr("scalar_mul");
    {
        auto s0 = RanScalar::zero();
        auto s1 = RanScalar::one();
        auto s2 = *RanScalar::from_bytes(two_bytes);
        auto s7 = *RanScalar::from_bytes(seven_bytes);
        auto s42 = *RanScalar::from_bytes(fortytwo_bytes);
        auto som1 = *RanScalar::from_bytes(ran_order_m1);
        auto sa = *RanScalar::from_bytes(test_a_bytes);

        /* arbitrary point = 7*G */
        auto p7 = G.scalar_mul(s7);

        struct
        {
            const char *label;
            RanScalar s;
            RanPoint p;
        } cases[] = {
            {"zero_times_G", s0, G},
            {"one_times_G", s1, G},
            {"two_times_G", s2, G},
            {"seven_times_G", s7, G},
            {"fortytwo_times_G", s42, G},
            {"order_m1_times_G", som1, G},
            {"a_times_G", sa, G},
            {"a_times_7G", sa, p7},
            /* Identity shenanigans */
            {"zero_times_O", s0, O},
            {"one_times_O", s1, O},
            {"a_times_O", sa, O},
            {"order_m1_times_O", som1, O},
        };
        const size_t n_cases = sizeof(cases) / sizeof(cases[0]);
        for (size_t i = 0; i < n_cases; i++)
        {
            auto r = cases[i].p.scalar_mul(cases[i].s);
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("scalar", cases[i].s.to_bytes());
            emit_hex_arr32("point", cases[i].p.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == n_cases - 1);
        }
    }
    close_arr();

    /* msm — expected results computed independently via scalar_mul */
    open_arr("msm");
    {
        /* Points: i*G for i=1..64. Scalars: i for i=1..64.
         * MSM(n) = sum(scs[j]*pts[j], j=0..n-1) = sum((j+1)*(j+1)*G) = sum((j+1)^2)*G
         * Compute expected result via scalar_mul of the equivalent scalar. */
        std::vector<RanPoint> pts;
        std::vector<RanScalar> scs;
        for (int i = 1; i <= 64; i++)
        {
            uint8_t sb[32];
            small_scalar_bytes(sb, (uint64_t)i);
            auto s = *RanScalar::from_bytes(sb);
            pts.push_back(G.scalar_mul(s));
            scs.push_back(s);
        }

        int sizes[] = {1, 2, 4, 16, 32, 33, 64};
        const char *labels[] = {"n_1", "n_2", "n_4", "n_16", "n_32_straus", "n_33_pippenger", "n_64_pippenger"};
        for (int ci = 0; ci < 7; ci++)
        {
            int n = sizes[ci];
            /* Compute equivalent scalar: sum((j+1)^2, j=0..n-1) */
            auto eq_scalar = RanScalar::zero();
            for (int j = 0; j < n; j++)
                eq_scalar = eq_scalar + scs[j] * scs[j];
            auto expected = G.scalar_mul(eq_scalar);

            open_obj();
            emit_string("label", labels[ci]);
            emit_int("n", n);

            open_arr("scalars");
            for (int j = 0; j < n; j++)
            {
                emit_indent();
                printf("\"%s\"%s\n", hex_str(scs[j].to_bytes().data(), 32).c_str(), j == n - 1 ? "" : ",");
            }
            close_arr();

            open_arr("points");
            for (int j = 0; j < n; j++)
            {
                emit_indent();
                printf("\"%s\"%s\n", hex_str(pts[j].to_bytes().data(), 32).c_str(), j == n - 1 ? "" : ",");
            }
            close_arr();

            emit_hex_arr32("result", expected.to_bytes(), true);
            close_obj(ci == 6);
        }
    }
    close_arr();

    /* pedersen_commit — expected results computed independently via scalar_mul */
    open_arr("pedersen_commit");
    {
        /* H = 2*G. pedersen = blind*H + sum(val[i]*gen[i])
         * Since H=2G and gen[i]=k_i*G, equivalent scalar = 2*blind + sum(val[i]*k_i) */
        auto s2 = *RanScalar::from_bytes(two_bytes);
        auto H = G.scalar_mul(s2);

        /* n=1: blind=test_a, val=test_b, gen=G(=1*G). eq = 2*test_a + test_b */
        {
            auto blind = *RanScalar::from_bytes(test_a_bytes);
            auto val = *RanScalar::from_bytes(test_b_bytes);
            auto gen = G;
            auto eq_scalar = s2 * blind + val;
            auto expected = G.scalar_mul(eq_scalar);

            open_obj();
            emit_string("label", "n_1");
            emit_hex_arr32("blinding", blind.to_bytes());
            emit_hex_arr32("H", H.to_bytes());
            emit_int("n", 1);
            open_arr("values");
            emit_indent();
            printf("\"%s\"\n", hex_str(val.to_bytes().data(), 32).c_str());
            close_arr();
            open_arr("generators");
            emit_indent();
            printf("\"%s\"\n", hex_str(gen.to_bytes().data(), 32).c_str());
            close_arr();
            emit_hex_arr32("result", expected.to_bytes(), true);
            close_obj();
        }

        /* n=4: blind=test_a, vals=[1,2,3,4], gens=[3G,4G,5G,6G]
         * eq = 2*test_a + 1*3 + 2*4 + 3*5 + 4*6 = 2*test_a + 50 */
        {
            auto blind = *RanScalar::from_bytes(test_a_bytes);
            RanScalar vals[4];
            RanPoint gens[4];
            RanScalar gen_scalars[4];
            auto eq_scalar = s2 * blind;
            for (int i = 0; i < 4; i++)
            {
                uint8_t sb[32];
                small_scalar_bytes(sb, (uint64_t)(i + 1));
                vals[i] = *RanScalar::from_bytes(sb);
                small_scalar_bytes(sb, (uint64_t)(i + 3));
                gen_scalars[i] = *RanScalar::from_bytes(sb);
                gens[i] = G.scalar_mul(gen_scalars[i]);
                eq_scalar = eq_scalar + vals[i] * gen_scalars[i];
            }
            auto expected = G.scalar_mul(eq_scalar);

            open_obj();
            emit_string("label", "n_4");
            emit_hex_arr32("blinding", blind.to_bytes());
            emit_hex_arr32("H", H.to_bytes());
            emit_int("n", 4);
            open_arr("values");
            for (int j = 0; j < 4; j++)
            {
                emit_indent();
                printf("\"%s\"%s\n", hex_str(vals[j].to_bytes().data(), 32).c_str(), j == 3 ? "" : ",");
            }
            close_arr();
            open_arr("generators");
            for (int j = 0; j < 4; j++)
            {
                emit_indent();
                printf("\"%s\"%s\n", hex_str(gens[j].to_bytes().data(), 32).c_str(), j == 3 ? "" : ",");
            }
            close_arr();
            emit_hex_arr32("result", expected.to_bytes(), true);
            close_obj();
        }

        /* blinding=0: eq = 0 + 1*1 = 1, result = G */
        {
            auto blind = RanScalar::zero();
            auto val = RanScalar::one();
            auto gen = G;
            auto expected = G; /* 0*2G + 1*G = G */

            open_obj();
            emit_string("label", "blinding_zero");
            emit_hex_arr32("blinding", blind.to_bytes());
            emit_hex_arr32("H", H.to_bytes());
            emit_int("n", 1);
            open_arr("values");
            emit_indent();
            printf("\"%s\"\n", hex_str(val.to_bytes().data(), 32).c_str());
            close_arr();
            open_arr("generators");
            emit_indent();
            printf("\"%s\"\n", hex_str(gen.to_bytes().data(), 32).c_str());
            close_arr();
            emit_hex_arr32("result", expected.to_bytes(), true);
            close_obj(true);
        }
    }
    close_arr();

    /* map_to_curve (single) */
    open_arr("map_to_curve_single");
    {
        const uint8_t *inputs[] = {zero_bytes, one_bytes, test_a_bytes, test_b_bytes};
        const char *labels[] = {"u_zero", "u_one", "u_test_a", "u_test_b"};
        for (int i = 0; i < 4; i++)
        {
            auto r = RanPoint::map_to_curve(inputs[i]);
            open_obj();
            emit_string("label", labels[i]);
            emit_hex("u", inputs[i], 32);
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 3);
        }
    }
    close_arr();

    /* map_to_curve (double) */
    open_arr("map_to_curve_double");
    {
        struct
        {
            const char *label;
            const uint8_t *u0;
            const uint8_t *u1;
        } cases[] = {
            {"zero_one", zero_bytes, one_bytes},
            {"a_b", test_a_bytes, test_b_bytes},
            {"one_a", one_bytes, test_a_bytes},
        };
        for (int i = 0; i < 3; i++)
        {
            auto r = RanPoint::map_to_curve(cases[i].u0, cases[i].u1);
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex("u0", cases[i].u0, 32);
            emit_hex("u1", cases[i].u1, 32);
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 2);
        }
    }
    close_arr();

    /* x_coordinate */
    open_arr("x_coordinate");
    {
        auto s7 = *RanScalar::from_bytes(seven_bytes);
        RanPoint pts[] = {G, G.dbl(), G.scalar_mul(s7)};
        const char *labels[] = {"G", "2G", "7G"};
        for (int i = 0; i < 3; i++)
        {
            open_obj();
            emit_string("label", labels[i]);
            emit_hex_arr32("point", pts[i].to_bytes());
            emit_hex_arr32("x_bytes", pts[i].x_coordinate_bytes(), true);
            close_obj(i == 2);
        }
    }
    close_arr(true);

    close_obj(); /* ran_point */
}

/* ── Shaw point vectors ── */

static void emit_shaw_point()
{
    open_obj("shaw_point");

    auto G = ShawPoint::generator();
    auto O = ShawPoint::identity();

    emit_hex_arr32("generator", G.to_bytes());
    emit_hex_arr32("identity", O.to_bytes());

    /* from_bytes */
    open_arr("from_bytes");
    {
        auto g_bytes = G.to_bytes();
        auto g2 = G.dbl();
        auto g2_bytes = g2.to_bytes();
        uint8_t g_flip[32];
        memcpy(g_flip, g_bytes.data(), 32);
        g_flip[31] ^= 0x80;

        struct
        {
            const char *label;
            const uint8_t *input;
        } cases[] = {
            {"generator", g_bytes.data()},
            {"double_generator", g2_bytes.data()},
            {"off_curve", off_curve_bytes},
            {"x_ge_p", x_ge_p_bytes},
            {"flipped_parity", g_flip},
            {"identity_encoding", zero_bytes},
        };
        for (size_t i = 0; i < 6; i++)
        {
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex("input", cases[i].input, 32);
            auto p = ShawPoint::from_bytes(cases[i].input);
            if (p)
                emit_hex_arr32("result", p->to_bytes(), true);
            else
                emit_null("result", true);
            close_obj(i == 5);
        }
    }
    close_arr();

    /* add — including identity edge cases */
    open_arr("add");
    {
        auto s2 = *ShawScalar::from_bytes(two_bytes);
        auto s7 = *ShawScalar::from_bytes(seven_bytes);
        auto s42 = *ShawScalar::from_bytes(fortytwo_bytes);
        auto g2 = G.scalar_mul(s2);
        auto g7 = G.scalar_mul(s7);
        auto g42 = G.scalar_mul(s42);

        auto s_nine = s2 + s7;

        struct
        {
            const char *label;
            ShawPoint a, b;
            ShawPoint expected;
        } cases[] = {
            {"2G_plus_7G", g2, g7, G.scalar_mul(s_nine)},
            {"G_plus_42G", G, g42, G.scalar_mul(ShawScalar::one() + s42)},
            {"7G_plus_42G", g7, g42, G.scalar_mul(s7 + s42)},
            {"P_plus_neg_P", G, -G, O},
            /* Identity shenanigans */
            {"O_plus_G", O, G, G},
            {"G_plus_O", G, O, G},
            {"O_plus_O", O, O, O},
            {"O_plus_7G", O, g7, g7},
            {"42G_plus_O", g42, O, g42},
            /* P + P (doubling via add) */
            {"G_plus_G", G, G, g2},
        };
        const size_t n_cases = sizeof(cases) / sizeof(cases[0]);
        for (size_t i = 0; i < n_cases; i++)
        {
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].a.to_bytes());
            emit_hex_arr32("b", cases[i].b.to_bytes());
            emit_hex_arr32("result", cases[i].expected.to_bytes(), true);
            close_obj(i == n_cases - 1);
        }
    }
    close_arr();

    /* dbl */
    open_arr("dbl");
    {
        auto s2 = *ShawScalar::from_bytes(two_bytes);
        auto g2 = G.scalar_mul(s2);

        struct
        {
            const char *label;
            ShawPoint a;
        } cases[] = {
            {"dbl_G", G},
            {"dbl_2G", g2},
            {"dbl_O", O},
        };
        for (size_t i = 0; i < 3; i++)
        {
            auto r = cases[i].a.dbl();
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].a.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 2);
        }
    }
    close_arr();

    /* negate */
    open_arr("negate");
    {
        auto s2 = *ShawScalar::from_bytes(two_bytes);
        auto g2 = G.scalar_mul(s2);

        struct
        {
            const char *label;
            ShawPoint a;
        } cases[] = {
            {"neg_G", G},
            {"neg_2G", g2},
            {"neg_O", O},
        };
        for (size_t i = 0; i < 3; i++)
        {
            auto r = -cases[i].a;
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("a", cases[i].a.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 2);
        }
    }
    close_arr();

    /* scalar_mul */
    open_arr("scalar_mul");
    {
        auto s0 = ShawScalar::zero();
        auto s1 = ShawScalar::one();
        auto s2 = *ShawScalar::from_bytes(two_bytes);
        auto s7 = *ShawScalar::from_bytes(seven_bytes);
        auto s42 = *ShawScalar::from_bytes(fortytwo_bytes);
        auto som1 = *ShawScalar::from_bytes(shaw_order_m1);
        auto sa = *ShawScalar::from_bytes(test_a_bytes);
        auto p7 = G.scalar_mul(s7);

        struct
        {
            const char *label;
            ShawScalar s;
            ShawPoint p;
        } cases[] = {
            {"zero_times_G", s0, G},
            {"one_times_G", s1, G},
            {"two_times_G", s2, G},
            {"seven_times_G", s7, G},
            {"fortytwo_times_G", s42, G},
            {"order_m1_times_G", som1, G},
            {"a_times_G", sa, G},
            {"a_times_7G", sa, p7},
            /* Identity shenanigans */
            {"zero_times_O", s0, O},
            {"one_times_O", s1, O},
            {"a_times_O", sa, O},
            {"order_m1_times_O", som1, O},
        };
        const size_t n_cases = sizeof(cases) / sizeof(cases[0]);
        for (size_t i = 0; i < n_cases; i++)
        {
            auto r = cases[i].p.scalar_mul(cases[i].s);
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex_arr32("scalar", cases[i].s.to_bytes());
            emit_hex_arr32("point", cases[i].p.to_bytes());
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == n_cases - 1);
        }
    }
    close_arr();

    /* msm — expected results computed independently via scalar_mul */
    open_arr("msm");
    {
        /* Points: i*G for i=1..64. Scalars: i for i=1..64.
         * MSM(n) = sum(scs[j]*pts[j], j=0..n-1) = sum((j+1)*(j+1)*G) = sum((j+1)^2)*G
         * Compute expected result via scalar_mul of the equivalent scalar. */
        std::vector<ShawPoint> pts;
        std::vector<ShawScalar> scs;
        for (int i = 1; i <= 64; i++)
        {
            uint8_t sb[32];
            small_scalar_bytes(sb, (uint64_t)i);
            auto s = *ShawScalar::from_bytes(sb);
            pts.push_back(G.scalar_mul(s));
            scs.push_back(s);
        }

        int sizes[] = {1, 2, 4, 16, 32, 33, 64};
        const char *labels[] = {"n_1", "n_2", "n_4", "n_16", "n_32_straus", "n_33_pippenger", "n_64_pippenger"};
        for (int ci = 0; ci < 7; ci++)
        {
            int n = sizes[ci];
            /* Compute equivalent scalar: sum((j+1)^2, j=0..n-1) */
            auto eq_scalar = ShawScalar::zero();
            for (int j = 0; j < n; j++)
                eq_scalar = eq_scalar + scs[j] * scs[j];
            auto expected = G.scalar_mul(eq_scalar);

            open_obj();
            emit_string("label", labels[ci]);
            emit_int("n", n);

            open_arr("scalars");
            for (int j = 0; j < n; j++)
            {
                emit_indent();
                printf("\"%s\"%s\n", hex_str(scs[j].to_bytes().data(), 32).c_str(), j == n - 1 ? "" : ",");
            }
            close_arr();

            open_arr("points");
            for (int j = 0; j < n; j++)
            {
                emit_indent();
                printf("\"%s\"%s\n", hex_str(pts[j].to_bytes().data(), 32).c_str(), j == n - 1 ? "" : ",");
            }
            close_arr();

            emit_hex_arr32("result", expected.to_bytes(), true);
            close_obj(ci == 6);
        }
    }
    close_arr();

    /* pedersen_commit — expected results computed independently via scalar_mul */
    open_arr("pedersen_commit");
    {
        /* H = 2*G. pedersen = blind*H + sum(val[i]*gen[i])
         * Since H=2G and gen[i]=k_i*G, equivalent scalar = 2*blind + sum(val[i]*k_i) */
        auto s2 = *ShawScalar::from_bytes(two_bytes);
        auto H = G.scalar_mul(s2);

        /* n=1: blind=test_a, val=test_b, gen=G(=1*G). eq = 2*test_a + test_b */
        {
            auto blind = *ShawScalar::from_bytes(test_a_bytes);
            auto val = *ShawScalar::from_bytes(test_b_bytes);
            auto gen = G;
            auto eq_scalar = s2 * blind + val;
            auto expected = G.scalar_mul(eq_scalar);

            open_obj();
            emit_string("label", "n_1");
            emit_hex_arr32("blinding", blind.to_bytes());
            emit_hex_arr32("H", H.to_bytes());
            emit_int("n", 1);
            open_arr("values");
            emit_indent();
            printf("\"%s\"\n", hex_str(val.to_bytes().data(), 32).c_str());
            close_arr();
            open_arr("generators");
            emit_indent();
            printf("\"%s\"\n", hex_str(gen.to_bytes().data(), 32).c_str());
            close_arr();
            emit_hex_arr32("result", expected.to_bytes(), true);
            close_obj();
        }

        /* n=4: blind=test_a, vals=[1,2,3,4], gens=[3G,4G,5G,6G]
         * eq = 2*test_a + 1*3 + 2*4 + 3*5 + 4*6 = 2*test_a + 50 */
        {
            auto blind = *ShawScalar::from_bytes(test_a_bytes);
            ShawScalar vals[4];
            ShawPoint gens[4];
            ShawScalar gen_scalars[4];
            auto eq_scalar = s2 * blind;
            for (int i = 0; i < 4; i++)
            {
                uint8_t sb[32];
                small_scalar_bytes(sb, (uint64_t)(i + 1));
                vals[i] = *ShawScalar::from_bytes(sb);
                small_scalar_bytes(sb, (uint64_t)(i + 3));
                gen_scalars[i] = *ShawScalar::from_bytes(sb);
                gens[i] = G.scalar_mul(gen_scalars[i]);
                eq_scalar = eq_scalar + vals[i] * gen_scalars[i];
            }
            auto expected = G.scalar_mul(eq_scalar);

            open_obj();
            emit_string("label", "n_4");
            emit_hex_arr32("blinding", blind.to_bytes());
            emit_hex_arr32("H", H.to_bytes());
            emit_int("n", 4);
            open_arr("values");
            for (int j = 0; j < 4; j++)
            {
                emit_indent();
                printf("\"%s\"%s\n", hex_str(vals[j].to_bytes().data(), 32).c_str(), j == 3 ? "" : ",");
            }
            close_arr();
            open_arr("generators");
            for (int j = 0; j < 4; j++)
            {
                emit_indent();
                printf("\"%s\"%s\n", hex_str(gens[j].to_bytes().data(), 32).c_str(), j == 3 ? "" : ",");
            }
            close_arr();
            emit_hex_arr32("result", expected.to_bytes(), true);
            close_obj();
        }

        /* blinding=0: eq = 0 + 1*1 = 1, result = G */
        {
            auto blind = ShawScalar::zero();
            auto val = ShawScalar::one();
            auto gen = G;
            auto expected = G; /* 0*2G + 1*G = G */

            open_obj();
            emit_string("label", "blinding_zero");
            emit_hex_arr32("blinding", blind.to_bytes());
            emit_hex_arr32("H", H.to_bytes());
            emit_int("n", 1);
            open_arr("values");
            emit_indent();
            printf("\"%s\"\n", hex_str(val.to_bytes().data(), 32).c_str());
            close_arr();
            open_arr("generators");
            emit_indent();
            printf("\"%s\"\n", hex_str(gen.to_bytes().data(), 32).c_str());
            close_arr();
            emit_hex_arr32("result", expected.to_bytes(), true);
            close_obj(true);
        }
    }
    close_arr();

    /* map_to_curve (single) */
    open_arr("map_to_curve_single");
    {
        const uint8_t *inputs[] = {zero_bytes, one_bytes, test_a_bytes, test_b_bytes};
        const char *labels[] = {"u_zero", "u_one", "u_test_a", "u_test_b"};
        for (int i = 0; i < 4; i++)
        {
            auto r = ShawPoint::map_to_curve(inputs[i]);
            open_obj();
            emit_string("label", labels[i]);
            emit_hex("u", inputs[i], 32);
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 3);
        }
    }
    close_arr();

    /* map_to_curve (double) */
    open_arr("map_to_curve_double");
    {
        struct
        {
            const char *label;
            const uint8_t *u0;
            const uint8_t *u1;
        } cases[] = {
            {"zero_one", zero_bytes, one_bytes},
            {"a_b", test_a_bytes, test_b_bytes},
            {"one_a", one_bytes, test_a_bytes},
        };
        for (int i = 0; i < 3; i++)
        {
            auto r = ShawPoint::map_to_curve(cases[i].u0, cases[i].u1);
            open_obj();
            emit_string("label", cases[i].label);
            emit_hex("u0", cases[i].u0, 32);
            emit_hex("u1", cases[i].u1, 32);
            emit_hex_arr32("result", r.to_bytes(), true);
            close_obj(i == 2);
        }
    }
    close_arr();

    /* x_coordinate */
    open_arr("x_coordinate");
    {
        auto s7 = *ShawScalar::from_bytes(seven_bytes);
        ShawPoint pts[] = {G, G.dbl(), G.scalar_mul(s7)};
        const char *labels[] = {"G", "2G", "7G"};
        for (int i = 0; i < 3; i++)
        {
            open_obj();
            emit_string("label", labels[i]);
            emit_hex_arr32("point", pts[i].to_bytes());
            emit_hex_arr32("x_bytes", pts[i].x_coordinate_bytes(), true);
            close_obj(i == 2);
        }
    }
    close_arr(true);

    close_obj(); /* shaw_point */
}

/* ── Polynomial vectors (Fp) ── */

static void emit_fp_polynomial()
{
    open_obj("fp_polynomial");

    /* from_roots */
    open_arr("from_roots");
    {
        /* 1 root */
        {
            auto p = FpPolynomial::from_roots(one_bytes, 1);
            auto &raw = p.raw();
            open_obj();
            emit_string("label", "one_root");
            emit_int("n", 1);
            open_arr("roots");
            emit_indent();
            printf("\"%s\"\n", hex_str(one_bytes, 32).c_str());
            close_arr();
            emit_int("degree", (int)p.degree());
            open_arr("coefficients");
            for (size_t j = 0; j < raw.coeffs.size(); j++)
            {
                uint8_t buf[32];
                fp_tobytes(buf, raw.coeffs[j].v);
                emit_indent();
                printf("\"%s\"%s\n", hex_str(buf, 32).c_str(), j == raw.coeffs.size() - 1 ? "" : ",");
            }
            close_arr(true);
            close_obj();
        }
        /* 2 roots */
        {
            uint8_t roots[64];
            memcpy(roots, one_bytes, 32);
            memcpy(roots + 32, two_bytes, 32);
            auto p = FpPolynomial::from_roots(roots, 2);
            auto &raw = p.raw();
            open_obj();
            emit_string("label", "two_roots");
            emit_int("n", 2);
            open_arr("roots");
            emit_indent();
            printf("\"%s\",\n", hex_str(one_bytes, 32).c_str());
            emit_indent();
            printf("\"%s\"\n", hex_str(two_bytes, 32).c_str());
            close_arr();
            emit_int("degree", (int)p.degree());
            open_arr("coefficients");
            for (size_t j = 0; j < raw.coeffs.size(); j++)
            {
                uint8_t buf[32];
                fp_tobytes(buf, raw.coeffs[j].v);
                emit_indent();
                printf("\"%s\"%s\n", hex_str(buf, 32).c_str(), j == raw.coeffs.size() - 1 ? "" : ",");
            }
            close_arr(true);
            close_obj();
        }
        /* 4 roots */
        {
            uint8_t roots[128];
            for (int i = 0; i < 4; i++)
            {
                uint8_t sb[32];
                small_scalar_bytes(sb, (uint64_t)(i + 1));
                memcpy(roots + i * 32, sb, 32);
            }
            auto p = FpPolynomial::from_roots(roots, 4);
            auto &raw = p.raw();
            open_obj();
            emit_string("label", "four_roots");
            emit_int("n", 4);
            open_arr("roots");
            for (int i = 0; i < 4; i++)
            {
                emit_indent();
                printf("\"%s\"%s\n", hex_str(roots + i * 32, 32).c_str(), i == 3 ? "" : ",");
            }
            close_arr();
            emit_int("degree", (int)p.degree());
            open_arr("coefficients");
            for (size_t j = 0; j < raw.coeffs.size(); j++)
            {
                uint8_t buf[32];
                fp_tobytes(buf, raw.coeffs[j].v);
                emit_indent();
                printf("\"%s\"%s\n", hex_str(buf, 32).c_str(), j == raw.coeffs.size() - 1 ? "" : ",");
            }
            close_arr(true);
            close_obj(true);
        }
    }
    close_arr();

    /* evaluate */
    open_arr("evaluate");
    {
        /* constant poly: [42] at x=7 -> 42 */
        {
            auto p = FpPolynomial::from_coefficients(fortytwo_bytes, 1);
            auto r = p.evaluate(seven_bytes);
            open_obj();
            emit_string("label", "constant_at_7");
            open_arr("coefficients");
            emit_indent();
            printf("\"%s\"\n", hex_str(fortytwo_bytes, 32).c_str());
            close_arr();
            emit_hex("x", seven_bytes, 32);
            emit_hex_arr32("result", r, true);
            close_obj();
        }
        /* linear: [1, 2] (= 1 + 2x) at x=0 -> 1 */
        {
            uint8_t coeffs[64];
            memcpy(coeffs, one_bytes, 32);
            memcpy(coeffs + 32, two_bytes, 32);
            auto p = FpPolynomial::from_coefficients(coeffs, 2);
            auto r = p.evaluate(zero_bytes);
            open_obj();
            emit_string("label", "linear_at_0");
            open_arr("coefficients");
            emit_indent();
            printf("\"%s\",\n", hex_str(one_bytes, 32).c_str());
            emit_indent();
            printf("\"%s\"\n", hex_str(two_bytes, 32).c_str());
            close_arr();
            emit_hex("x", zero_bytes, 32);
            emit_hex_arr32("result", r, true);
            close_obj();
        }
        /* linear: [1, 2] at x=test_a */
        {
            uint8_t coeffs[64];
            memcpy(coeffs, one_bytes, 32);
            memcpy(coeffs + 32, two_bytes, 32);
            auto p = FpPolynomial::from_coefficients(coeffs, 2);
            auto r = p.evaluate(test_a_bytes);
            open_obj();
            emit_string("label", "linear_at_test_a");
            open_arr("coefficients");
            emit_indent();
            printf("\"%s\",\n", hex_str(one_bytes, 32).c_str());
            emit_indent();
            printf("\"%s\"\n", hex_str(two_bytes, 32).c_str());
            close_arr();
            emit_hex("x", test_a_bytes, 32);
            emit_hex_arr32("result", r, true);
            close_obj();
        }
        /* quadratic: [1, 0, 1] (= 1 + x^2) at x=7 -> 50 */
        {
            uint8_t coeffs[96];
            memcpy(coeffs, one_bytes, 32);
            memset(coeffs + 32, 0, 32);
            memcpy(coeffs + 64, one_bytes, 32);
            auto p = FpPolynomial::from_coefficients(coeffs, 3);
            auto r = p.evaluate(seven_bytes);
            open_obj();
            emit_string("label", "quadratic_at_7");
            open_arr("coefficients");
            emit_indent();
            printf("\"%s\",\n", hex_str(coeffs, 32).c_str());
            emit_indent();
            printf("\"%s\",\n", hex_str(coeffs + 32, 32).c_str());
            emit_indent();
            printf("\"%s\"\n", hex_str(coeffs + 64, 32).c_str());
            close_arr();
            emit_hex("x", seven_bytes, 32);
            emit_hex_arr32("result", r, true);
            close_obj(true);
        }
    }
    close_arr();

    /* mul */
    open_arr("mul");
    {
        /* deg 1 x 1 */
        {
            uint8_t c1[64], c2[64];
            memcpy(c1, one_bytes, 32);
            memcpy(c1 + 32, two_bytes, 32);
            memcpy(c2, seven_bytes, 32);
            memcpy(c2 + 32, one_bytes, 32);
            auto p1 = FpPolynomial::from_coefficients(c1, 2);
            auto p2 = FpPolynomial::from_coefficients(c2, 2);
            auto r = p1 * p2;
            open_obj();
            emit_string("label", "deg1_times_deg1");
            emit_hex_array("a_coefficients", c1, 2);
            emit_hex_array("b_coefficients", c2, 2);
            emit_int("degree", (int)r.degree());
            emit_fp_poly_coeffs("coefficients", r.raw(), true);
            close_obj();
        }
        /* deg 5 x 5 */
        {
            uint8_t c1[192], c2[192];
            for (int i = 0; i < 6; i++)
            {
                small_scalar_bytes(c1 + i * 32, (uint64_t)(i + 1));
                small_scalar_bytes(c2 + i * 32, (uint64_t)(i + 7));
            }
            auto p1 = FpPolynomial::from_coefficients(c1, 6);
            auto p2 = FpPolynomial::from_coefficients(c2, 6);
            auto r = p1 * p2;
            open_obj();
            emit_string("label", "deg5_times_deg5");
            emit_hex_array("a_coefficients", c1, 6);
            emit_hex_array("b_coefficients", c2, 6);
            emit_int("degree", (int)r.degree());
            emit_fp_poly_coeffs("coefficients", r.raw(), true);
            close_obj();
        }
        /* deg 15 x 15 (schoolbook) */
        {
            uint8_t c1[512], c2[512];
            for (int i = 0; i < 16; i++)
            {
                small_scalar_bytes(c1 + i * 32, (uint64_t)(i + 1));
                small_scalar_bytes(c2 + i * 32, (uint64_t)(i + 17));
            }
            auto p1 = FpPolynomial::from_coefficients(c1, 16);
            auto p2 = FpPolynomial::from_coefficients(c2, 16);
            auto r = p1 * p2;
            open_obj();
            emit_string("label", "deg15_times_deg15");
            emit_hex_array("a_coefficients", c1, 16);
            emit_hex_array("b_coefficients", c2, 16);
            emit_int("degree", (int)r.degree());
            emit_fp_poly_coeffs("coefficients", r.raw(), true);
            close_obj();
        }
        /* deg 16 x 16 (Karatsuba) */
        {
            uint8_t c1[544], c2[544];
            for (int i = 0; i < 17; i++)
            {
                small_scalar_bytes(c1 + i * 32, (uint64_t)(i + 1));
                small_scalar_bytes(c2 + i * 32, (uint64_t)(i + 18));
            }
            auto p1 = FpPolynomial::from_coefficients(c1, 17);
            auto p2 = FpPolynomial::from_coefficients(c2, 17);
            auto r = p1 * p2;
            open_obj();
            emit_string("label", "deg16_times_deg16_karatsuba");
            emit_hex_array("a_coefficients", c1, 17);
            emit_hex_array("b_coefficients", c2, 17);
            emit_int("degree", (int)r.degree());
            emit_fp_poly_coeffs("coefficients", r.raw(), true);
            close_obj(true);
        }
    }
    close_arr();

    /* add */
    open_arr("add");
    {
        uint8_t c1[96], c2[96];
        for (int i = 0; i < 3; i++)
            small_scalar_bytes(c1 + i * 32, (uint64_t)(i + 1));
        for (int i = 0; i < 3; i++)
            small_scalar_bytes(c2 + i * 32, (uint64_t)(i + 10));
        auto p1 = FpPolynomial::from_coefficients(c1, 3);
        auto p2 = FpPolynomial::from_coefficients(c2, 3);
        auto r = p1 + p2;

        open_obj();
        emit_string("label", "same_degree");
        emit_hex_array("a_coefficients", c1, 3);
        emit_hex_array("b_coefficients", c2, 3);
        emit_fp_poly_coeffs("coefficients", r.raw(), true);
        close_obj();

        /* different degree */
        uint8_t c3[64];
        small_scalar_bytes(c3, 5);
        small_scalar_bytes(c3 + 32, 3);
        auto p3 = FpPolynomial::from_coefficients(c3, 2);
        auto r2 = p1 + p3;

        open_obj();
        emit_string("label", "different_degree");
        emit_hex_array("a_coefficients", c1, 3);
        emit_hex_array("b_coefficients", c3, 2);
        emit_fp_poly_coeffs("coefficients", r2.raw(), true);
        close_obj(true);
    }
    close_arr();

    /* sub */
    open_arr("sub");
    {
        uint8_t c1[96], c2[96];
        for (int i = 0; i < 3; i++)
            small_scalar_bytes(c1 + i * 32, (uint64_t)(i + 10));
        for (int i = 0; i < 3; i++)
            small_scalar_bytes(c2 + i * 32, (uint64_t)(i + 1));
        auto p1 = FpPolynomial::from_coefficients(c1, 3);
        auto p2 = FpPolynomial::from_coefficients(c2, 3);
        auto r = p1 - p2;

        open_obj();
        emit_string("label", "same_degree");
        emit_hex_array("a_coefficients", c1, 3);
        emit_hex_array("b_coefficients", c2, 3);
        emit_fp_poly_coeffs("coefficients", r.raw(), true);
        close_obj();

        uint8_t c3[64];
        small_scalar_bytes(c3, 5);
        small_scalar_bytes(c3 + 32, 3);
        auto p3 = FpPolynomial::from_coefficients(c3, 2);
        auto r2 = p1 - p3;

        open_obj();
        emit_string("label", "different_degree");
        emit_hex_array("a_coefficients", c1, 3);
        emit_hex_array("b_coefficients", c3, 2);
        emit_fp_poly_coeffs("coefficients", r2.raw(), true);
        close_obj(true);
    }
    close_arr();

    /* divmod */
    open_arr("divmod");
    {
        /* exact division: (x-1)(x-2) / (x-1) = (x-2) */
        {
            uint8_t roots2[64];
            memcpy(roots2, one_bytes, 32);
            memcpy(roots2 + 32, two_bytes, 32);
            auto num = FpPolynomial::from_roots(roots2, 2);
            auto den = FpPolynomial::from_roots(one_bytes, 1);
            auto [q, r] = num.divmod(den);

            open_obj();
            emit_string("label", "exact_division");
            emit_fp_poly_coeffs("numerator", num.raw());
            emit_fp_poly_coeffs("denominator", den.raw());
            emit_fp_poly_coeffs("quotient", q.raw());
            emit_fp_poly_coeffs("remainder", r.raw(), true);
            close_obj();
        }
        /* non-zero remainder */
        {
            uint8_t c1[96];
            small_scalar_bytes(c1, 7);
            small_scalar_bytes(c1 + 32, 3);
            small_scalar_bytes(c1 + 64, 1);
            auto num = FpPolynomial::from_coefficients(c1, 3);
            auto den = FpPolynomial::from_roots(two_bytes, 1);
            auto [q, r] = num.divmod(den);

            open_obj();
            emit_string("label", "nonzero_remainder");
            emit_fp_poly_coeffs("numerator", num.raw());
            emit_fp_poly_coeffs("denominator", den.raw());
            emit_fp_poly_coeffs("quotient", q.raw());
            emit_fp_poly_coeffs("remainder", r.raw(), true);
            close_obj();
        }
        /* divide by linear */
        {
            uint8_t roots3[96];
            for (int i = 0; i < 3; i++)
                small_scalar_bytes(roots3 + i * 32, (uint64_t)(i + 1));
            auto num = FpPolynomial::from_roots(roots3, 3);
            auto den = FpPolynomial::from_roots(roots3, 1);
            auto [q, r] = num.divmod(den);

            open_obj();
            emit_string("label", "divide_by_linear");
            emit_fp_poly_coeffs("numerator", num.raw());
            emit_fp_poly_coeffs("denominator", den.raw());
            emit_fp_poly_coeffs("quotient", q.raw());
            emit_fp_poly_coeffs("remainder", r.raw(), true);
            close_obj(true);
        }
    }
    close_arr();

    /* interpolate */
    open_arr("interpolate");
    {
        /* 3 points */
        {
            uint8_t xs[96], ys[96];
            for (int i = 0; i < 3; i++)
            {
                small_scalar_bytes(xs + i * 32, (uint64_t)(i + 1));
                small_scalar_bytes(ys + i * 32, (uint64_t)((i + 1) * (i + 1)));
            }
            auto p = FpPolynomial::interpolate(xs, ys, 3);
            open_obj();
            emit_string("label", "three_points");
            emit_int("n", 3);
            emit_hex_array("xs", xs, 3);
            emit_hex_array("ys", ys, 3);
            emit_int("degree", (int)p.degree());
            emit_fp_poly_coeffs("coefficients", p.raw(), true);
            close_obj();
        }
        /* 4 points */
        {
            uint8_t xs[128], ys[128];
            for (int i = 0; i < 4; i++)
            {
                small_scalar_bytes(xs + i * 32, (uint64_t)(i + 1));
                small_scalar_bytes(ys + i * 32, (uint64_t)((i + 1) * 3 + 2));
            }
            auto p = FpPolynomial::interpolate(xs, ys, 4);
            open_obj();
            emit_string("label", "four_points");
            emit_int("n", 4);
            emit_hex_array("xs", xs, 4);
            emit_hex_array("ys", ys, 4);
            emit_int("degree", (int)p.degree());
            emit_fp_poly_coeffs("coefficients", p.raw(), true);
            close_obj(true);
        }
    }
    close_arr(true);

    close_obj(); /* fp_polynomial */
}

/* ── Polynomial vectors (Fq) — same structure, different field ── */

static void emit_fq_polynomial()
{
    open_obj("fq_polynomial");

    /* from_roots */
    open_arr("from_roots");
    {
        {
            auto p = FqPolynomial::from_roots(one_bytes, 1);
            open_obj();
            emit_string("label", "one_root");
            emit_int("n", 1);
            emit_hex_array("roots", one_bytes, 1);
            emit_int("degree", (int)p.degree());
            emit_fq_poly_coeffs("coefficients", p.raw(), true);
            close_obj();
        }
        {
            uint8_t roots[64];
            memcpy(roots, one_bytes, 32);
            memcpy(roots + 32, two_bytes, 32);
            auto p = FqPolynomial::from_roots(roots, 2);
            open_obj();
            emit_string("label", "two_roots");
            emit_int("n", 2);
            emit_hex_array("roots", roots, 2);
            emit_int("degree", (int)p.degree());
            emit_fq_poly_coeffs("coefficients", p.raw(), true);
            close_obj();
        }
        {
            uint8_t roots[128];
            for (int i = 0; i < 4; i++)
                small_scalar_bytes(roots + i * 32, (uint64_t)(i + 1));
            auto p = FqPolynomial::from_roots(roots, 4);
            open_obj();
            emit_string("label", "four_roots");
            emit_int("n", 4);
            emit_hex_array("roots", roots, 4);
            emit_int("degree", (int)p.degree());
            emit_fq_poly_coeffs("coefficients", p.raw(), true);
            close_obj(true);
        }
    }
    close_arr();

    /* evaluate */
    open_arr("evaluate");
    {
        {
            auto p = FqPolynomial::from_coefficients(fortytwo_bytes, 1);
            auto r = p.evaluate(seven_bytes);
            open_obj();
            emit_string("label", "constant_at_7");
            emit_hex_array("coefficients", fortytwo_bytes, 1);
            emit_hex("x", seven_bytes, 32);
            emit_hex_arr32("result", r, true);
            close_obj();
        }
        {
            uint8_t coeffs[64];
            memcpy(coeffs, one_bytes, 32);
            memcpy(coeffs + 32, two_bytes, 32);
            auto p = FqPolynomial::from_coefficients(coeffs, 2);
            auto r = p.evaluate(zero_bytes);
            open_obj();
            emit_string("label", "linear_at_0");
            emit_hex_array("coefficients", coeffs, 2);
            emit_hex("x", zero_bytes, 32);
            emit_hex_arr32("result", r, true);
            close_obj();
        }
        {
            uint8_t coeffs[64];
            memcpy(coeffs, one_bytes, 32);
            memcpy(coeffs + 32, two_bytes, 32);
            auto p = FqPolynomial::from_coefficients(coeffs, 2);
            auto r = p.evaluate(test_a_bytes);
            open_obj();
            emit_string("label", "linear_at_test_a");
            emit_hex_array("coefficients", coeffs, 2);
            emit_hex("x", test_a_bytes, 32);
            emit_hex_arr32("result", r, true);
            close_obj();
        }
        {
            uint8_t coeffs[96];
            memcpy(coeffs, one_bytes, 32);
            memset(coeffs + 32, 0, 32);
            memcpy(coeffs + 64, one_bytes, 32);
            auto p = FqPolynomial::from_coefficients(coeffs, 3);
            auto r = p.evaluate(seven_bytes);
            open_obj();
            emit_string("label", "quadratic_at_7");
            emit_hex_array("coefficients", coeffs, 3);
            emit_hex("x", seven_bytes, 32);
            emit_hex_arr32("result", r, true);
            close_obj(true);
        }
    }
    close_arr();

    /* mul */
    open_arr("mul");
    {
        /* deg 1 x 1 */
        {
            uint8_t c1[64], c2[64];
            memcpy(c1, one_bytes, 32);
            memcpy(c1 + 32, two_bytes, 32);
            memcpy(c2, seven_bytes, 32);
            memcpy(c2 + 32, one_bytes, 32);
            auto p1 = FqPolynomial::from_coefficients(c1, 2);
            auto p2 = FqPolynomial::from_coefficients(c2, 2);
            auto r = p1 * p2;
            open_obj();
            emit_string("label", "deg1_times_deg1");
            emit_hex_array("a_coefficients", c1, 2);
            emit_hex_array("b_coefficients", c2, 2);
            emit_int("degree", (int)r.degree());
            emit_fq_poly_coeffs("coefficients", r.raw(), true);
            close_obj();
        }
        /* deg 5 x 5 */
        {
            uint8_t c1[192], c2[192];
            for (int i = 0; i < 6; i++)
            {
                small_scalar_bytes(c1 + i * 32, (uint64_t)(i + 1));
                small_scalar_bytes(c2 + i * 32, (uint64_t)(i + 7));
            }
            auto p1 = FqPolynomial::from_coefficients(c1, 6);
            auto p2 = FqPolynomial::from_coefficients(c2, 6);
            auto r = p1 * p2;
            open_obj();
            emit_string("label", "deg5_times_deg5");
            emit_hex_array("a_coefficients", c1, 6);
            emit_hex_array("b_coefficients", c2, 6);
            emit_int("degree", (int)r.degree());
            emit_fq_poly_coeffs("coefficients", r.raw(), true);
            close_obj();
        }
        /* deg 15 x 15 (schoolbook) */
        {
            uint8_t c1[512], c2[512];
            for (int i = 0; i < 16; i++)
            {
                small_scalar_bytes(c1 + i * 32, (uint64_t)(i + 1));
                small_scalar_bytes(c2 + i * 32, (uint64_t)(i + 17));
            }
            auto p1 = FqPolynomial::from_coefficients(c1, 16);
            auto p2 = FqPolynomial::from_coefficients(c2, 16);
            auto r = p1 * p2;
            open_obj();
            emit_string("label", "deg15_times_deg15");
            emit_hex_array("a_coefficients", c1, 16);
            emit_hex_array("b_coefficients", c2, 16);
            emit_int("degree", (int)r.degree());
            emit_fq_poly_coeffs("coefficients", r.raw(), true);
            close_obj();
        }
        /* deg 16 x 16 (Karatsuba) */
        {
            uint8_t c1[544], c2[544];
            for (int i = 0; i < 17; i++)
            {
                small_scalar_bytes(c1 + i * 32, (uint64_t)(i + 1));
                small_scalar_bytes(c2 + i * 32, (uint64_t)(i + 18));
            }
            auto p1 = FqPolynomial::from_coefficients(c1, 17);
            auto p2 = FqPolynomial::from_coefficients(c2, 17);
            auto r = p1 * p2;
            open_obj();
            emit_string("label", "deg16_times_deg16_karatsuba");
            emit_hex_array("a_coefficients", c1, 17);
            emit_hex_array("b_coefficients", c2, 17);
            emit_int("degree", (int)r.degree());
            emit_fq_poly_coeffs("coefficients", r.raw(), true);
            close_obj(true);
        }
    }
    close_arr();

    /* add */
    open_arr("add");
    {
        uint8_t c1[96], c2[96];
        for (int i = 0; i < 3; i++)
            small_scalar_bytes(c1 + i * 32, (uint64_t)(i + 1));
        for (int i = 0; i < 3; i++)
            small_scalar_bytes(c2 + i * 32, (uint64_t)(i + 10));
        auto p1 = FqPolynomial::from_coefficients(c1, 3);
        auto p2 = FqPolynomial::from_coefficients(c2, 3);
        auto r = p1 + p2;

        open_obj();
        emit_string("label", "same_degree");
        emit_hex_array("a_coefficients", c1, 3);
        emit_hex_array("b_coefficients", c2, 3);
        emit_fq_poly_coeffs("coefficients", r.raw(), true);
        close_obj();

        /* different degree */
        uint8_t c3[64];
        small_scalar_bytes(c3, 5);
        small_scalar_bytes(c3 + 32, 3);
        auto p3 = FqPolynomial::from_coefficients(c3, 2);
        auto r2 = p1 + p3;

        open_obj();
        emit_string("label", "different_degree");
        emit_hex_array("a_coefficients", c1, 3);
        emit_hex_array("b_coefficients", c3, 2);
        emit_fq_poly_coeffs("coefficients", r2.raw(), true);
        close_obj(true);
    }
    close_arr();

    /* sub */
    open_arr("sub");
    {
        uint8_t c1[96], c2[96];
        for (int i = 0; i < 3; i++)
            small_scalar_bytes(c1 + i * 32, (uint64_t)(i + 10));
        for (int i = 0; i < 3; i++)
            small_scalar_bytes(c2 + i * 32, (uint64_t)(i + 1));
        auto p1 = FqPolynomial::from_coefficients(c1, 3);
        auto p2 = FqPolynomial::from_coefficients(c2, 3);
        auto r = p1 - p2;

        open_obj();
        emit_string("label", "same_degree");
        emit_hex_array("a_coefficients", c1, 3);
        emit_hex_array("b_coefficients", c2, 3);
        emit_fq_poly_coeffs("coefficients", r.raw(), true);
        close_obj();

        uint8_t c3[64];
        small_scalar_bytes(c3, 5);
        small_scalar_bytes(c3 + 32, 3);
        auto p3 = FqPolynomial::from_coefficients(c3, 2);
        auto r2 = p1 - p3;

        open_obj();
        emit_string("label", "different_degree");
        emit_hex_array("a_coefficients", c1, 3);
        emit_hex_array("b_coefficients", c3, 2);
        emit_fq_poly_coeffs("coefficients", r2.raw(), true);
        close_obj(true);
    }
    close_arr();

    /* divmod */
    open_arr("divmod");
    {
        /* exact division: (x-1)(x-2) / (x-1) = (x-2) */
        {
            uint8_t roots2[64];
            memcpy(roots2, one_bytes, 32);
            memcpy(roots2 + 32, two_bytes, 32);
            auto num = FqPolynomial::from_roots(roots2, 2);
            auto den = FqPolynomial::from_roots(one_bytes, 1);
            auto [q, r] = num.divmod(den);

            open_obj();
            emit_string("label", "exact_division");
            emit_fq_poly_coeffs("numerator", num.raw());
            emit_fq_poly_coeffs("denominator", den.raw());
            emit_fq_poly_coeffs("quotient", q.raw());
            emit_fq_poly_coeffs("remainder", r.raw(), true);
            close_obj();
        }
        /* non-zero remainder */
        {
            uint8_t c1[96];
            small_scalar_bytes(c1, 7);
            small_scalar_bytes(c1 + 32, 3);
            small_scalar_bytes(c1 + 64, 1);
            auto num = FqPolynomial::from_coefficients(c1, 3);
            auto den = FqPolynomial::from_roots(two_bytes, 1);
            auto [q, r] = num.divmod(den);

            open_obj();
            emit_string("label", "nonzero_remainder");
            emit_fq_poly_coeffs("numerator", num.raw());
            emit_fq_poly_coeffs("denominator", den.raw());
            emit_fq_poly_coeffs("quotient", q.raw());
            emit_fq_poly_coeffs("remainder", r.raw(), true);
            close_obj();
        }
        /* divide by linear */
        {
            uint8_t roots3[96];
            for (int i = 0; i < 3; i++)
                small_scalar_bytes(roots3 + i * 32, (uint64_t)(i + 1));
            auto num = FqPolynomial::from_roots(roots3, 3);
            auto den = FqPolynomial::from_roots(roots3, 1);
            auto [q, r] = num.divmod(den);

            open_obj();
            emit_string("label", "divide_by_linear");
            emit_fq_poly_coeffs("numerator", num.raw());
            emit_fq_poly_coeffs("denominator", den.raw());
            emit_fq_poly_coeffs("quotient", q.raw());
            emit_fq_poly_coeffs("remainder", r.raw(), true);
            close_obj(true);
        }
    }
    close_arr();

    /* interpolate */
    open_arr("interpolate");
    {
        /* 3 points */
        {
            uint8_t xs[96], ys[96];
            for (int i = 0; i < 3; i++)
            {
                small_scalar_bytes(xs + i * 32, (uint64_t)(i + 1));
                small_scalar_bytes(ys + i * 32, (uint64_t)((i + 1) * (i + 1)));
            }
            auto p = FqPolynomial::interpolate(xs, ys, 3);
            open_obj();
            emit_string("label", "three_points");
            emit_int("n", 3);
            emit_hex_array("xs", xs, 3);
            emit_hex_array("ys", ys, 3);
            emit_int("degree", (int)p.degree());
            emit_fq_poly_coeffs("coefficients", p.raw(), true);
            close_obj();
        }
        /* 4 points */
        {
            uint8_t xs[128], ys[128];
            for (int i = 0; i < 4; i++)
            {
                small_scalar_bytes(xs + i * 32, (uint64_t)(i + 1));
                small_scalar_bytes(ys + i * 32, (uint64_t)((i + 1) * 3 + 2));
            }
            auto p = FqPolynomial::interpolate(xs, ys, 4);
            open_obj();
            emit_string("label", "four_points");
            emit_int("n", 4);
            emit_hex_array("xs", xs, 4);
            emit_hex_array("ys", ys, 4);
            emit_int("degree", (int)p.degree());
            emit_fq_poly_coeffs("coefficients", p.raw(), true);
            close_obj(true);
        }
    }
    close_arr(true);

    close_obj(); /* fq_polynomial */
}

/* ── Divisor vectors ── */

static void emit_ran_divisor()
{
    open_obj("ran_divisor");

    auto G = RanPoint::generator();

    open_arr("compute");
    {
        int sizes[] = {2, 4, 8};
        const char *labels[] = {"n_2", "n_4", "n_8"};
        for (int ci = 0; ci < 3; ci++)
        {
            int n = sizes[ci];
            std::vector<RanPoint> pts;
            for (int i = 1; i <= n; i++)
            {
                uint8_t sb[32];
                small_scalar_bytes(sb, (uint64_t)i);
                pts.push_back(G.scalar_mul(*RanScalar::from_bytes(sb)));
            }
            auto div = RanDivisor::compute(pts.data(), (size_t)n);

            open_obj();
            emit_string("label", labels[ci]);
            emit_int("n", n);

            open_arr("points");
            for (int j = 0; j < n; j++)
            {
                emit_indent();
                printf("\"%s\"%s\n", hex_str(pts[j].to_bytes().data(), 32).c_str(), j == n - 1 ? "" : ",");
            }
            close_arr();

            /* a polynomial coefficients */
            auto &araw = div.a().raw();
            open_arr("a_coefficients");
            for (size_t j = 0; j < araw.coeffs.size(); j++)
            {
                uint8_t buf[32];
                fp_tobytes(buf, araw.coeffs[j].v);
                emit_indent();
                printf("\"%s\"%s\n", hex_str(buf, 32).c_str(), j == araw.coeffs.size() - 1 ? "" : ",");
            }
            close_arr();

            /* b polynomial coefficients */
            auto &braw = div.b().raw();
            open_arr("b_coefficients");
            for (size_t j = 0; j < braw.coeffs.size(); j++)
            {
                uint8_t buf[32];
                fp_tobytes(buf, braw.coeffs[j].v);
                emit_indent();
                printf("\"%s\"%s\n", hex_str(buf, 32).c_str(), j == braw.coeffs.size() - 1 ? "" : ",");
            }
            close_arr();

            /* evaluate at a non-member point */
            uint8_t sb_test[32];
            small_scalar_bytes(sb_test, (uint64_t)(n + 1));
            auto test_pt = G.scalar_mul(*RanScalar::from_bytes(sb_test));
            auto x_bytes = test_pt.x_coordinate_bytes();
            /* need y-coordinate too — extract from compressed form */
            auto pt_bytes = test_pt.to_bytes();
            /* decompress to get full (x,y) */
            auto decompressed = RanPoint::from_bytes(pt_bytes.data());
            /* we need raw y bytes — get from the affine form */
            ran_affine aff;
            ran_to_affine(&aff, &decompressed->raw());
            uint8_t y_bytes_arr[32];
            fp_tobytes(y_bytes_arr, aff.y);

            auto eval_result = div.evaluate(x_bytes.data(), y_bytes_arr);

            emit_hex("eval_point_x", x_bytes.data(), 32);
            emit_hex("eval_point_y", y_bytes_arr, 32);
            emit_hex_arr32("eval_result", eval_result, true);
            close_obj(ci == 2);
        }
    }
    close_arr(true);

    close_obj(); /* ran_divisor */
}

static void emit_shaw_divisor()
{
    open_obj("shaw_divisor");

    auto G = ShawPoint::generator();

    open_arr("compute");
    {
        int sizes[] = {2, 4, 8};
        const char *labels[] = {"n_2", "n_4", "n_8"};
        for (int ci = 0; ci < 3; ci++)
        {
            int n = sizes[ci];
            std::vector<ShawPoint> pts;
            for (int i = 1; i <= n; i++)
            {
                uint8_t sb[32];
                small_scalar_bytes(sb, (uint64_t)i);
                pts.push_back(G.scalar_mul(*ShawScalar::from_bytes(sb)));
            }
            auto div = ShawDivisor::compute(pts.data(), (size_t)n);

            open_obj();
            emit_string("label", labels[ci]);
            emit_int("n", n);

            open_arr("points");
            for (int j = 0; j < n; j++)
            {
                emit_indent();
                printf("\"%s\"%s\n", hex_str(pts[j].to_bytes().data(), 32).c_str(), j == n - 1 ? "" : ",");
            }
            close_arr();

            auto &araw = div.a().raw();
            open_arr("a_coefficients");
            for (size_t j = 0; j < araw.coeffs.size(); j++)
            {
                uint8_t buf[32];
                fq_tobytes(buf, araw.coeffs[j].v);
                emit_indent();
                printf("\"%s\"%s\n", hex_str(buf, 32).c_str(), j == araw.coeffs.size() - 1 ? "" : ",");
            }
            close_arr();

            auto &braw = div.b().raw();
            open_arr("b_coefficients");
            for (size_t j = 0; j < braw.coeffs.size(); j++)
            {
                uint8_t buf[32];
                fq_tobytes(buf, braw.coeffs[j].v);
                emit_indent();
                printf("\"%s\"%s\n", hex_str(buf, 32).c_str(), j == braw.coeffs.size() - 1 ? "" : ",");
            }
            close_arr();

            uint8_t sb_test[32];
            small_scalar_bytes(sb_test, (uint64_t)(n + 1));
            auto test_pt = G.scalar_mul(*ShawScalar::from_bytes(sb_test));
            auto x_bytes = test_pt.x_coordinate_bytes();
            auto pt_bytes = test_pt.to_bytes();
            auto decompressed = ShawPoint::from_bytes(pt_bytes.data());
            shaw_affine aff;
            shaw_to_affine(&aff, &decompressed->raw());
            uint8_t y_bytes_arr[32];
            fq_tobytes(y_bytes_arr, aff.y);

            auto eval_result = div.evaluate(x_bytes.data(), y_bytes_arr);

            emit_hex("eval_point_x", x_bytes.data(), 32);
            emit_hex("eval_point_y", y_bytes_arr, 32);
            emit_hex_arr32("eval_result", eval_result, true);
            close_obj(ci == 2);
        }
    }
    close_arr(true);

    close_obj(); /* shaw_divisor */
}

/* ── Wei25519 bridge vectors ── */

static void emit_wei25519()
{
    open_obj("wei25519");

    open_arr("x_to_shaw_scalar");
    {
        /* valid small x */
        {
            auto r = shaw_scalar_from_wei25519_x(seven_bytes);
            open_obj();
            emit_string("label", "small_x");
            emit_hex("input", seven_bytes, 32);
            if (r)
                emit_hex_arr32("result", r->to_bytes(), true);
            else
                emit_null("result", true);
            close_obj();
        }
        /* valid test_a x */
        {
            auto r = shaw_scalar_from_wei25519_x(test_a_bytes);
            open_obj();
            emit_string("label", "test_a_x");
            emit_hex("input", test_a_bytes, 32);
            if (r)
                emit_hex_arr32("result", r->to_bytes(), true);
            else
                emit_null("result", true);
            close_obj();
        }
        /* invalid: x >= p */
        {
            auto r = shaw_scalar_from_wei25519_x(x_ge_p_bytes);
            open_obj();
            emit_string("label", "x_ge_p");
            emit_hex("input", x_ge_p_bytes, 32);
            if (r)
                emit_hex_arr32("result", r->to_bytes(), true);
            else
                emit_null("result", true);
            close_obj(true);
        }
    }
    close_arr(true);

    close_obj(); /* wei25519 */
}

/* ── Batch inversion vectors ── */

static void emit_batch_invert()
{
    open_obj("batch_invert");

    /* fp */
    open_arr("fp");
    {
        /* n=1 */
        {
            fp_fe in, out;
            fp_frombytes(in, test_a_bytes);
            fp_batch_invert(&out, &in, 1);
            uint8_t result[32];
            fp_tobytes(result, out);

            open_obj();
            emit_string("label", "n_1");
            emit_int("n", 1);
            open_arr("inputs");
            emit_indent();
            printf("\"%s\"\n", hex_str(test_a_bytes, 32).c_str());
            close_arr();
            open_arr("results");
            emit_indent();
            printf("\"%s\"\n", hex_str(result, 32).c_str());
            close_arr(true);
            close_obj();
        }
        /* n=4 */
        {
            fp_fe in[4], out[4];
            const uint8_t *inputs[] = {one_bytes, two_bytes, seven_bytes, test_a_bytes};
            for (int i = 0; i < 4; i++)
                fp_frombytes(in[i], inputs[i]);
            fp_batch_invert(out, in, 4);

            open_obj();
            emit_string("label", "n_4");
            emit_int("n", 4);
            open_arr("inputs");
            for (int i = 0; i < 4; i++)
            {
                emit_indent();
                printf("\"%s\"%s\n", hex_str(inputs[i], 32).c_str(), i == 3 ? "" : ",");
            }
            close_arr();
            open_arr("results");
            for (int i = 0; i < 4; i++)
            {
                uint8_t result[32];
                fp_tobytes(result, out[i]);
                emit_indent();
                printf("\"%s\"%s\n", hex_str(result, 32).c_str(), i == 3 ? "" : ",");
            }
            close_arr(true);
            close_obj(true);
        }
    }
    close_arr();

    /* fq */
    open_arr("fq");
    {
        /* n=1 */
        {
            fq_fe in, out;
            fq_frombytes(in, test_a_bytes);
            fq_batch_invert(&out, &in, 1);
            uint8_t result[32];
            fq_tobytes(result, out);

            open_obj();
            emit_string("label", "n_1");
            emit_int("n", 1);
            open_arr("inputs");
            emit_indent();
            printf("\"%s\"\n", hex_str(test_a_bytes, 32).c_str());
            close_arr();
            open_arr("results");
            emit_indent();
            printf("\"%s\"\n", hex_str(result, 32).c_str());
            close_arr(true);
            close_obj();
        }
        /* n=4 */
        {
            fq_fe in[4], out[4];
            const uint8_t *inputs[] = {one_bytes, two_bytes, seven_bytes, test_a_bytes};
            for (int i = 0; i < 4; i++)
                fq_frombytes(in[i], inputs[i]);
            fq_batch_invert(out, in, 4);

            open_obj();
            emit_string("label", "n_4");
            emit_int("n", 4);
            open_arr("inputs");
            for (int i = 0; i < 4; i++)
            {
                emit_indent();
                printf("\"%s\"%s\n", hex_str(inputs[i], 32).c_str(), i == 3 ? "" : ",");
            }
            close_arr();
            open_arr("results");
            for (int i = 0; i < 4; i++)
            {
                uint8_t result[32];
                fq_tobytes(result, out[i]);
                emit_indent();
                printf("\"%s\"%s\n", hex_str(result, 32).c_str(), i == 3 ? "" : ",");
            }
            close_arr(true);
            close_obj(true);
        }
    }
    close_arr(true);

    close_obj(); /* batch_invert */
}

/* ── Main ── */

int main()
{
#if !defined(FORCE_PORTABLE) || !FORCE_PORTABLE
    fprintf(stderr, "ERROR: gen_test_vectors MUST be built with -DFORCE_PORTABLE=1\n");
    fprintf(stderr, "       to ensure vectors come from the reference portable backend.\n");
    return 1;
#endif

    ranshaw::init();
    compute_order_minus_one();

    fprintf(stderr, "Generating ranshaw test vectors...\n");

    open_obj();
    emit_string("generator", "ranshaw-gen-testvectors");
    emit_string("version", "1.0.0");

    /* Curve parameters for reference */
    open_obj("parameters");
    emit_hex("ran_order", RAN_ORDER, 32);
    emit_hex("shaw_order", SHAW_ORDER, 32);
    printf("    \"curve_a\": -3,\n");
    {
        uint8_t buf[32];
        fp_tobytes(buf, RAN_B);
        emit_hex("ran_b", buf, 32);
        fq_tobytes(buf, SHAW_B);
        emit_hex("shaw_b", buf, 32);
        fp_tobytes(buf, RAN_GX);
        emit_hex("ran_gx", buf, 32);
        fp_tobytes(buf, RAN_GY);
        emit_hex("ran_gy", buf, 32);
        fq_tobytes(buf, SHAW_GX);
        emit_hex("shaw_gx", buf, 32);
        fq_tobytes(buf, SHAW_GY);
        emit_hex("shaw_gy", buf, 32, true);
    }
    close_obj();

    fprintf(stderr, "  Ran scalar...\n");
    emit_ran_scalar();

    fprintf(stderr, "  Shaw scalar...\n");
    emit_shaw_scalar();

    fprintf(stderr, "  Ran point...\n");
    emit_ran_point();

    fprintf(stderr, "  Shaw point...\n");
    emit_shaw_point();

    fprintf(stderr, "  Fp polynomial...\n");
    emit_fp_polynomial();

    fprintf(stderr, "  Fq polynomial...\n");
    emit_fq_polynomial();

    fprintf(stderr, "  Ran divisor...\n");
    emit_ran_divisor();

    fprintf(stderr, "  Shaw divisor...\n");
    emit_shaw_divisor();

    fprintf(stderr, "  Wei25519 bridge...\n");
    emit_wei25519();

    fprintf(stderr, "  Batch invert...\n");
    emit_batch_invert();

    /* High-degree polynomial multiplication vectors.
     * Compact format: inputs are deterministic (coeff[i] = i+1 for a, i+n+1 for b),
     * so we only store multi-point eval checks: a(x_j)*b(x_j) == result(x_j)
     * at 3 independent points. This validates correctness without storing
     * millions of coefficients.
     *
     * Sizes walk up in base-2 increments: 1, 2, 4, ..., 8192
     * ECFFT threshold is 1024 coefficients per operand.
     */
    fprintf(stderr, "  High-degree polynomial mul...\n");

    /* 3 eval points for multi-point verification */
    const uint8_t *eval_points[] = {test_a_bytes, test_b_bytes, seven_bytes};
    const char *eval_point_names[] = {"test_a", "test_b", "seven"};

    open_obj("high_degree_poly_mul");
    {
        auto emit_highdeg_mul_fp = [&](const char *label, int n_coeffs, bool last)
        {
            fprintf(stderr, "    fp %s (%d coeffs)...\n", label, n_coeffs);
            std::vector<uint8_t> c1(n_coeffs * 32), c2(n_coeffs * 32);
            for (int i = 0; i < n_coeffs; i++)
            {
                small_scalar_bytes(c1.data() + i * 32, (uint64_t)(i + 1));
                small_scalar_bytes(c2.data() + i * 32, (uint64_t)(i + n_coeffs + 1));
            }
            auto p1 = FpPolynomial::from_coefficients(c1.data(), n_coeffs);
            auto p2 = FpPolynomial::from_coefficients(c2.data(), n_coeffs);
            auto result = p1 * p2;

            open_obj();
            emit_string("label", label);
            emit_int("n_coeffs", n_coeffs);
            emit_int("result_degree", (int)result.degree());

            /* Multi-point eval checks */
            open_arr("eval_checks");
            for (int j = 0; j < 3; j++)
            {
                auto ev_a = p1.evaluate(eval_points[j]);
                auto ev_b = p2.evaluate(eval_points[j]);
                auto ev_r = result.evaluate(eval_points[j]);
                open_obj();
                emit_string("point", eval_point_names[j]);
                emit_hex("x", eval_points[j], 32);
                emit_hex_arr32("a_of_x", ev_a);
                emit_hex_arr32("b_of_x", ev_b);
                emit_hex_arr32("result_of_x", ev_r, true);
                close_obj(j == 2);
            }
            close_arr(true);

            close_obj(last);
        };

        auto emit_highdeg_mul_fq = [&](const char *label, int n_coeffs, bool last)
        {
            fprintf(stderr, "    fq %s (%d coeffs)...\n", label, n_coeffs);
            std::vector<uint8_t> c1(n_coeffs * 32), c2(n_coeffs * 32);
            for (int i = 0; i < n_coeffs; i++)
            {
                small_scalar_bytes(c1.data() + i * 32, (uint64_t)(i + 1));
                small_scalar_bytes(c2.data() + i * 32, (uint64_t)(i + n_coeffs + 1));
            }
            auto p1 = FqPolynomial::from_coefficients(c1.data(), n_coeffs);
            auto p2 = FqPolynomial::from_coefficients(c2.data(), n_coeffs);
            auto result = p1 * p2;

            open_obj();
            emit_string("label", label);
            emit_int("n_coeffs", n_coeffs);
            emit_int("result_degree", (int)result.degree());

            open_arr("eval_checks");
            for (int j = 0; j < 3; j++)
            {
                auto ev_a = p1.evaluate(eval_points[j]);
                auto ev_b = p2.evaluate(eval_points[j]);
                auto ev_r = result.evaluate(eval_points[j]);
                open_obj();
                emit_string("point", eval_point_names[j]);
                emit_hex("x", eval_points[j], 32);
                emit_hex_arr32("a_of_x", ev_a);
                emit_hex_arr32("b_of_x", ev_b);
                emit_hex_arr32("result_of_x", ev_r, true);
                close_obj(j == 2);
            }
            close_arr(true);

            close_obj(last);
        };

        /* Fp high-degree: base-2 walk from 1 to 8192 */
        open_arr("fp");
        {
            const int sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
            const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
            for (int si = 0; si < n_sizes; si++)
            {
                char label[64];
                snprintf(label, sizeof(label), "n%d", sizes[si]);
                emit_highdeg_mul_fp(label, sizes[si], si == n_sizes - 1);
            }
        }
        close_arr();

        /* Fq high-degree: same base-2 walk */
        open_arr("fq");
        {
            const int sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
            const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
            for (int si = 0; si < n_sizes; si++)
            {
                char label[64];
                snprintf(label, sizeof(label), "n%d", sizes[si]);
                emit_highdeg_mul_fq(label, sizes[si], si == n_sizes - 1);
            }
        }
        close_arr(true);
    }
    close_obj(true); /* high_degree_poly_mul */

    close_obj(true);

    fprintf(stderr, "Done.\n");
    return 0;
}
