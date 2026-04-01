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
 * FCMP++ workload benchmark — measures the exact operations Monero's FCMP++
 * protocol calls from the ranshaw library.
 *
 * Groups:
 *   1. Node benchmarks (tree construction): Pedersen hash via MSM + to_affine
 *   2. Wallet benchmarks (proof construction): scalar_mul_divisor pipeline + multiexp
 *   3. Verification benchmarks (batch multiexp)
 *   4. Composite scores: weighted real-world timing estimates
 */

#include "ranshaw.h"
#include "ranshaw_benchmark.h"

#include <cstring>
#include <iostream>
#include <vector>

static const unsigned char test_scalar[32] = {0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, 0xbe, 0xba, 0xfe,
                                              0xca, 0xef, 0xbe, 0xad, 0xde, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                              0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10};

/* ── Helper: generate n Jacobian points via successive doubling from generator ── */

static void generate_ran_points(std::vector<ran_jacobian> &pts, size_t n)
{
    pts.resize(n);
    fp_copy(pts[0].X, RAN_GX);
    fp_copy(pts[0].Y, RAN_GY);
    fp_1(pts[0].Z);
    for (size_t i = 1; i < n; i++)
        ran_dbl(&pts[i], &pts[i - 1]);
}

static void generate_shaw_points(std::vector<shaw_jacobian> &pts, size_t n)
{
    pts.resize(n);
    fq_copy(pts[0].X, SHAW_GX);
    fq_copy(pts[0].Y, SHAW_GY);
    fq_1(pts[0].Z);
    for (size_t i = 1; i < n; i++)
        shaw_dbl(&pts[i], &pts[i - 1]);
}

/* ── Helper: generate n small test scalars (32 bytes each) ── */

static void generate_scalars(std::vector<unsigned char> &scalars, size_t n)
{
    scalars.resize(n * 32, 0);
    for (size_t i = 0; i < n; i++)
    {
        scalars[i * 32 + 0] = static_cast<unsigned char>((i + 1) & 0xff);
        scalars[i * 32 + 1] = static_cast<unsigned char>(((i + 1) >> 8) & 0xff);
        scalars[i * 32 + 2] = static_cast<unsigned char>(((i + 1) >> 16) & 0xff);
        scalars[i * 32 + 3] = static_cast<unsigned char>(0x01);
    }
}

/* ── Helper: time a simple loop, return per-call average in microseconds ── */

template<typename T> static double time_average_us(T &&function, size_t iters)
{
    const auto start = NOW();
    for (size_t i = 0; i < iters; i++)
        function();
    return NOW_DIFF(start) / static_cast<double>(iters);
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
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            std::cerr << "Usage: ranshaw-benchmark-fcmpp [--init|--autotune]" << std::endl;
            return 1;
        }
    }

    auto state = benchmark_setup();

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

    /* ================================================================
     * Group 1: Node Benchmarks (Tree Construction)
     *
     * Each "tree hash" = MSM vartime (Pedersen hash) + to_affine (extract x-coord).
     * ================================================================ */

    std::cout << std::endl;
    std::cout << "=== FCMP++ Node Benchmarks (Tree Construction) ===" << std::endl;
    std::cout << std::endl;
    benchmark_header();

    /* shaw_tree_hash(228) — leaf layer */
    constexpr size_t SHAW_LEAF_N = 228;
    std::vector<shaw_jacobian> s_leaf_gens;
    generate_shaw_points(s_leaf_gens, SHAW_LEAF_N);
    std::vector<unsigned char> s_leaf_scalars;
    generate_scalars(s_leaf_scalars, SHAW_LEAF_N);

    shaw_jacobian s_msm_result;
    shaw_affine s_aff_result;

    benchmark(
        [&]()
        {
            shaw_msm_vartime(&s_msm_result, s_leaf_scalars.data(), s_leaf_gens.data(), SHAW_LEAF_N);
            shaw_to_affine(&s_aff_result, &s_msm_result);
            benchmark_do_not_optimize(s_aff_result);
        },
        "shaw_tree_hash(228)",
        1000,
        100);

    /* shaw_tree_hash(38) — Shaw branch */
    constexpr size_t SHAW_BRANCH_N = 38;
    std::vector<shaw_jacobian> s_branch_gens;
    generate_shaw_points(s_branch_gens, SHAW_BRANCH_N);
    std::vector<unsigned char> s_branch_scalars;
    generate_scalars(s_branch_scalars, SHAW_BRANCH_N);

    benchmark(
        [&]()
        {
            shaw_msm_vartime(&s_msm_result, s_branch_scalars.data(), s_branch_gens.data(), SHAW_BRANCH_N);
            shaw_to_affine(&s_aff_result, &s_msm_result);
            benchmark_do_not_optimize(s_aff_result);
        },
        "shaw_tree_hash(38)",
        5000,
        500);

    /* ran_tree_hash(18) — Ran branch */
    constexpr size_t RAN_BRANCH_N = 18;
    std::vector<ran_jacobian> h_branch_gens;
    generate_ran_points(h_branch_gens, RAN_BRANCH_N);
    std::vector<unsigned char> h_branch_scalars;
    generate_scalars(h_branch_scalars, RAN_BRANCH_N);

    ran_jacobian h_msm_result;
    ran_affine h_aff_result;

    benchmark(
        [&]()
        {
            ran_msm_vartime(&h_msm_result, h_branch_scalars.data(), h_branch_gens.data(), RAN_BRANCH_N);
            ran_to_affine(&h_aff_result, &h_msm_result);
            benchmark_do_not_optimize(h_aff_result);
        },
        "ran_tree_hash(18)",
        5000,
        500);

    /* ================================================================
     * Group 2: Wallet Benchmarks — Divisors
     *
     * scalar_mul_divisor = scalarmult_vartime + 253 doublings + batch_to_affine(254) + compute_divisor(254)
     * ================================================================ */

    std::cout << std::endl;
    std::cout << "=== FCMP++ Wallet Benchmarks (Proof Construction) ===" << std::endl;
    std::cout << std::endl;
    benchmark_header();

    /* Set up affine generator points for scalar_mul_divisor */
    shaw_affine s_gen_aff;
    {
        shaw_jacobian s_gen;
        fq_copy(s_gen.X, SHAW_GX);
        fq_copy(s_gen.Y, SHAW_GY);
        fq_1(s_gen.Z);
        shaw_to_affine(&s_gen_aff, &s_gen);
    }

    ran_affine h_gen_aff;
    {
        ran_jacobian h_gen;
        fp_copy(h_gen.X, RAN_GX);
        fp_copy(h_gen.Y, RAN_GY);
        fp_1(h_gen.Z);
        ran_to_affine(&h_gen_aff, &h_gen);
    }

    shaw_divisor s_div;
    ran_divisor h_div;

    /* shaw_scalar_mul_divisor(253) */
    benchmark(
        [&]()
        {
            shaw_scalar_mul_divisor(&s_div, test_scalar, &s_gen_aff);
            benchmark_do_not_optimize(s_div.a.coeffs[0]);
        },
        "shaw_scalar_mul_divisor(253)",
        10,
        1);

    /* ran_scalar_mul_divisor(253) */
    benchmark(
        [&]()
        {
            ran_scalar_mul_divisor(&h_div, test_scalar, &h_gen_aff);
            benchmark_do_not_optimize(h_div.a.coeffs[0]);
        },
        "ran_scalar_mul_divisor(253)",
        10,
        1);

    /* Standalone polynomial multiplication at degree 253 over Fp */
    {
        fp_poly fp_a, fp_b, fp_r;
        fp_a.coeffs.resize(254);
        fp_b.coeffs.resize(254);
        for (size_t i = 0; i < 254; i++)
        {
            fp_1(fp_a.coeffs[i].v);
            fp_a.coeffs[i].v[0] = static_cast<uint64_t>(i + 1);
            fp_1(fp_b.coeffs[i].v);
            fp_b.coeffs[i].v[0] = static_cast<uint64_t>(i + 100);
        }

        benchmark(
            [&]()
            {
                fp_poly_mul(&fp_r, &fp_a, &fp_b);
                benchmark_do_not_optimize(fp_r.coeffs[0]);
            },
            "fp_poly_mul(253)",
            50,
            5);
    }

    /* Standalone polynomial multiplication at degree 253 over Fq */
    {
        fq_poly fq_a, fq_b, fq_r;
        fq_a.coeffs.resize(254);
        fq_b.coeffs.resize(254);
        for (size_t i = 0; i < 254; i++)
        {
            fq_1(fq_a.coeffs[i].v);
            fq_a.coeffs[i].v[0] = static_cast<uint64_t>(i + 1);
            fq_1(fq_b.coeffs[i].v);
            fq_b.coeffs[i].v[0] = static_cast<uint64_t>(i + 100);
        }

        benchmark(
            [&]()
            {
                fq_poly_mul(&fq_r, &fq_a, &fq_b);
                benchmark_do_not_optimize(fq_r.coeffs[0]);
            },
            "fq_poly_mul(253)",
            50,
            5);
    }

    /* ── Group 3: Wallet Benchmarks — GBP Prover (Multiexp) ── */

    std::cout << std::endl;
    benchmark_header();

    /* shaw_multiexp(256) */
    constexpr size_t SHAW_MULTIEXP_N = 256;
    std::vector<shaw_jacobian> s_multiexp_gens;
    generate_shaw_points(s_multiexp_gens, SHAW_MULTIEXP_N);
    std::vector<unsigned char> s_multiexp_scalars;
    generate_scalars(s_multiexp_scalars, SHAW_MULTIEXP_N);

    benchmark(
        [&]()
        {
            shaw_msm_vartime(&s_msm_result, s_multiexp_scalars.data(), s_multiexp_gens.data(), SHAW_MULTIEXP_N);
            benchmark_do_not_optimize(s_msm_result);
        },
        "shaw_multiexp(256)",
        500,
        50);

    /* ran_multiexp(128) */
    constexpr size_t RAN_MULTIEXP_N = 128;
    std::vector<ran_jacobian> h_multiexp_gens;
    generate_ran_points(h_multiexp_gens, RAN_MULTIEXP_N);
    std::vector<unsigned char> h_multiexp_scalars;
    generate_scalars(h_multiexp_scalars, RAN_MULTIEXP_N);

    benchmark(
        [&]()
        {
            ran_msm_vartime(&h_msm_result, h_multiexp_scalars.data(), h_multiexp_gens.data(), RAN_MULTIEXP_N);
            benchmark_do_not_optimize(h_msm_result);
        },
        "ran_multiexp(128)",
        500,
        50);

    /* ================================================================
     * Group 3: Verification Benchmarks (Batch Multiexp)
     *
     * FCMP++ verification = shaw_msm(522 + 80*batch) + ran_msm(265 + 80*batch)
     * Fixed generators: Shaw 522 (g,h,g_bold[256],h_bold[256],h_sum[8])
     *                   Ran 265 (g,h,g_bold[128],h_bold[128],h_sum[7])
     * Per-proof additional: ~80 points per curve
     * ================================================================ */

    std::cout << std::endl;
    std::cout << "=== FCMP++ Verification (Batch Multiexp) ===" << std::endl;
    std::cout << std::endl;
    benchmark_header();

    constexpr size_t FCMPP_SHAW_FIXED = 522;
    constexpr size_t FCMPP_RAN_FIXED = 265;
    constexpr size_t FCMPP_PER_PROOF = 80;

    auto verify_iters = [](size_t n) -> size_t
    {
        if (n >= 1000)
            return 1000;
        if (n >= 600)
            return 2000;
        return 5000;
    };

    auto verify_warmup = [](size_t n) -> size_t
    {
        if (n >= 1000)
            return 50;
        if (n >= 600)
            return 100;
        return 200;
    };

    constexpr size_t verify_batches[] = {1, 2, 4, 10};

    for (const auto batch : verify_batches)
    {
        const size_t s_n = FCMPP_SHAW_FIXED + FCMPP_PER_PROOF * batch;
        const size_t h_n = FCMPP_RAN_FIXED + FCMPP_PER_PROOF * batch;

        std::vector<shaw_jacobian> s_verify_pts;
        generate_shaw_points(s_verify_pts, s_n);
        std::vector<unsigned char> s_verify_sc;
        generate_scalars(s_verify_sc, s_n);

        std::vector<ran_jacobian> h_verify_pts;
        generate_ran_points(h_verify_pts, h_n);
        std::vector<unsigned char> h_verify_sc;
        generate_scalars(h_verify_sc, h_n);

        char label[64];

        /* Shaw MSM */
        std::snprintf(label, sizeof(label), "shaw_verify batch=%zu n=%zu", batch, s_n);
        benchmark(
            [&]()
            {
                shaw_msm_vartime(&s_msm_result, s_verify_sc.data(), s_verify_pts.data(), s_n);
                benchmark_do_not_optimize(s_msm_result);
            },
            label,
            verify_iters(s_n),
            verify_warmup(s_n));

        /* Ran MSM */
        std::snprintf(label, sizeof(label), "ran_verify batch=%zu n=%zu", batch, h_n);
        benchmark(
            [&]()
            {
                ran_msm_vartime(&h_msm_result, h_verify_sc.data(), h_verify_pts.data(), h_n);
                benchmark_do_not_optimize(h_msm_result);
            },
            label,
            verify_iters(h_n),
            verify_warmup(h_n));

        /* Combined verify */
        std::snprintf(label, sizeof(label), "fcmpp_verify batch=%zu (%zu+%zu)", batch, s_n, h_n);
        benchmark(
            [&]()
            {
                shaw_msm_vartime(&s_msm_result, s_verify_sc.data(), s_verify_pts.data(), s_n);
                ran_msm_vartime(&h_msm_result, h_verify_sc.data(), h_verify_pts.data(), h_n);
                benchmark_do_not_optimize(s_msm_result);
                benchmark_do_not_optimize(h_msm_result);
            },
            label,
            verify_iters(s_n + h_n),
            verify_warmup(s_n + h_n));
    }

    /* ================================================================
     * Group 4: Composite Scores
     *
     * Capture per-call averages and compute weighted real-world scores.
     * ================================================================ */

    std::cout << std::endl;
    std::cout << "=== FCMP++ Composite Scores ===" << std::endl;
    std::cout << std::endl;
    std::cout << "  Measuring per-call averages for composite scoring..." << std::endl;

    /* Tree hash timings (us per call) */
    constexpr size_t COMPOSITE_TREE_ITERS = 100;
    constexpr size_t COMPOSITE_DIV_ITERS = 3;
    constexpr size_t COMPOSITE_MSM_ITERS = 50;

    const double shaw_tree_228_us = time_average_us(
        [&]()
        {
            shaw_msm_vartime(&s_msm_result, s_leaf_scalars.data(), s_leaf_gens.data(), SHAW_LEAF_N);
            shaw_to_affine(&s_aff_result, &s_msm_result);
            benchmark_do_not_optimize(s_aff_result);
        },
        COMPOSITE_TREE_ITERS);

    const double ran_tree_18_us = time_average_us(
        [&]()
        {
            ran_msm_vartime(&h_msm_result, h_branch_scalars.data(), h_branch_gens.data(), RAN_BRANCH_N);
            ran_to_affine(&h_aff_result, &h_msm_result);
            benchmark_do_not_optimize(h_aff_result);
        },
        COMPOSITE_TREE_ITERS);

    /* Divisor timings */
    const double shaw_div_us = time_average_us(
        [&]()
        {
            shaw_scalar_mul_divisor(&s_div, test_scalar, &s_gen_aff);
            benchmark_do_not_optimize(s_div.a.coeffs[0]);
        },
        COMPOSITE_DIV_ITERS);

    const double ran_div_us = time_average_us(
        [&]()
        {
            ran_scalar_mul_divisor(&h_div, test_scalar, &h_gen_aff);
            benchmark_do_not_optimize(h_div.a.coeffs[0]);
        },
        COMPOSITE_DIV_ITERS);

    /* Multiexp timings */
    const double shaw_multiexp_256_us = time_average_us(
        [&]()
        {
            shaw_msm_vartime(&s_msm_result, s_multiexp_scalars.data(), s_multiexp_gens.data(), SHAW_MULTIEXP_N);
            benchmark_do_not_optimize(s_msm_result);
        },
        COMPOSITE_MSM_ITERS);

    const double ran_multiexp_128_us = time_average_us(
        [&]()
        {
            ran_msm_vartime(&h_msm_result, h_multiexp_scalars.data(), h_multiexp_gens.data(), RAN_MULTIEXP_N);
            benchmark_do_not_optimize(h_msm_result);
        },
        COMPOSITE_MSM_ITERS);

    /* Verification timings */
    constexpr size_t COMPOSITE_VERIFY_ITERS = 50;

    const size_t verify1_s_n = FCMPP_SHAW_FIXED + FCMPP_PER_PROOF * 1;
    const size_t verify1_h_n = FCMPP_RAN_FIXED + FCMPP_PER_PROOF * 1;
    std::vector<shaw_jacobian> s_v1_pts;
    generate_shaw_points(s_v1_pts, verify1_s_n);
    std::vector<unsigned char> s_v1_sc;
    generate_scalars(s_v1_sc, verify1_s_n);
    std::vector<ran_jacobian> h_v1_pts;
    generate_ran_points(h_v1_pts, verify1_h_n);
    std::vector<unsigned char> h_v1_sc;
    generate_scalars(h_v1_sc, verify1_h_n);

    const double verify1_us = time_average_us(
        [&]()
        {
            shaw_msm_vartime(&s_msm_result, s_v1_sc.data(), s_v1_pts.data(), verify1_s_n);
            ran_msm_vartime(&h_msm_result, h_v1_sc.data(), h_v1_pts.data(), verify1_h_n);
            benchmark_do_not_optimize(s_msm_result);
            benchmark_do_not_optimize(h_msm_result);
        },
        COMPOSITE_VERIFY_ITERS);

    const size_t verify10_s_n = FCMPP_SHAW_FIXED + FCMPP_PER_PROOF * 10;
    const size_t verify10_h_n = FCMPP_RAN_FIXED + FCMPP_PER_PROOF * 10;
    std::vector<shaw_jacobian> s_v10_pts;
    generate_shaw_points(s_v10_pts, verify10_s_n);
    std::vector<unsigned char> s_v10_sc;
    generate_scalars(s_v10_sc, verify10_s_n);
    std::vector<ran_jacobian> h_v10_pts;
    generate_ran_points(h_v10_pts, verify10_h_n);
    std::vector<unsigned char> h_v10_sc;
    generate_scalars(h_v10_sc, verify10_h_n);

    const double verify10_us = time_average_us(
        [&]()
        {
            shaw_msm_vartime(&s_msm_result, s_v10_sc.data(), s_v10_pts.data(), verify10_s_n);
            ran_msm_vartime(&h_msm_result, h_v10_sc.data(), h_v10_pts.data(), verify10_h_n);
            benchmark_do_not_optimize(s_msm_result);
            benchmark_do_not_optimize(h_msm_result);
        },
        COMPOSITE_VERIFY_ITERS);

    /* Node (100M outputs):
     *   2,631,579 × shaw_tree_hash(228) + 146,199 × ran_tree_hash(18) */
    const double node_us = 2631579.0 * shaw_tree_228_us + 146199.0 * ran_tree_18_us;
    const double node_s = node_us / 1e6;

    /* Wallet (1-input tx):
     *   4 × shaw_scalar_mul_divisor(253) + 8 × ran_scalar_mul_divisor(253)
     *   + 1 × shaw_multiexp(256) + 1 × ran_multiexp(128) */
    const double wallet_us = 4.0 * shaw_div_us + 8.0 * ran_div_us + shaw_multiexp_256_us + ran_multiexp_128_us;
    const double wallet_s = wallet_us / 1e6;

    /* Wallet + Privacy = wallet + node */
    const double wallet_privacy_s = wallet_s + node_s;

    constexpr int LABEL_W = 38;
    constexpr int VALUE_W = 12;

    std::cout << std::endl;
    std::cout << "  Per-call averages:" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << std::setw(LABEL_W) << "shaw_tree_hash(228):" << std::setw(VALUE_W) << shaw_tree_228_us << " us"
              << std::endl;
    std::cout << std::setw(LABEL_W) << "ran_tree_hash(18):" << std::setw(VALUE_W) << ran_tree_18_us << " us"
              << std::endl;
    std::cout << std::setw(LABEL_W) << "shaw_scalar_mul_divisor(253):" << std::setw(VALUE_W) << shaw_div_us << " us"
              << std::endl;
    std::cout << std::setw(LABEL_W) << "ran_scalar_mul_divisor(253):" << std::setw(VALUE_W) << ran_div_us << " us"
              << std::endl;
    std::cout << std::setw(LABEL_W) << "shaw_multiexp(256):" << std::setw(VALUE_W) << shaw_multiexp_256_us << " us"
              << std::endl;
    std::cout << std::setw(LABEL_W) << "ran_multiexp(128):" << std::setw(VALUE_W) << ran_multiexp_128_us << " us"
              << std::endl;

    std::cout << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << std::setw(LABEL_W) << "Verify (batch=1):" << std::setw(VALUE_W) << (verify1_us / 1000.0) << " ms"
              << std::endl;
    std::cout << std::setw(LABEL_W) << "Verify (batch=10):" << std::setw(VALUE_W) << (verify10_us / 1000.0) << " ms"
              << std::endl;

    std::cout << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(LABEL_W) << "Node (100M outputs):" << std::setw(VALUE_W) << node_s << " seconds"
              << std::endl;
    std::cout << std::setw(LABEL_W) << "Wallet (1-input tx):" << std::setw(VALUE_W) << (wallet_us / 1000.0) << " ms"
              << std::endl;
    std::cout << std::setw(LABEL_W) << "Wallet + Privacy:" << std::setw(VALUE_W) << wallet_privacy_s << " seconds"
              << std::endl;

    benchmark_teardown(state);

    return 0;
}
