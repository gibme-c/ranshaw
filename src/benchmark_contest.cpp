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
 * FCMP++ Optimization Competition benchmark — matches the Rust competition's methodology.
 *
 * Methodology (matching ranshaw-contest/benches/ranshaw.rs):
 *   - Simple for loop with N iterations (no adaptive batching, no warmup)
 *   - black_box / benchmark_do_not_optimize to prevent dead code elimination
 *   - Reports total time in microseconds for all N iterations
 *   - Computes improvement vs Rust reference and winner
 *
 * Rust competition source:
 *   https://github.com/j-berman/fcmp-plus-plus-optimization-competition
 */

#include "ranshaw.h"
#include "ranshaw_primitives.h"
#include "ranshaw_benchmark.h"

#include <cstring>
#include <iostream>
#include <vector>

static const unsigned char test_scalar[32] = {0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, 0xbe, 0xba, 0xfe,
                                              0xca, 0xef, 0xbe, 0xad, 0xde, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                              0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10};

/* ── Rust competition reference numbers (us, from cargo bench on same machine) ──
 * These are the total times for all iterations of each operation. */
struct rust_times
{
    double reference;
    double winner;
};

static const rust_times RUST_SHAW_POINT_ADD = {801440, 525255};
static const rust_times RUST_RAN_POINT_ADD = {811546, 685835};
static const rust_times RUST_FIELD_MUL = {876441, 711976};
static const rust_times RUST_FIELD_INVERT = {818190, 677614};
static const rust_times RUST_SHAW_DECOMPRESS = {1144528, 414891};
static const rust_times RUST_RAN_DECOMPRESS = {1177135, 1055547};
static const rust_times RUST_FIELD_ADD = {1019172, 559506};
static const rust_times RUST_FIELD_SUB = {862869, 547949};
static const rust_times RUST_SHAW_SCALAR_MUL = {910035, 728880};
static const rust_times RUST_RAN_SCALAR_MUL = {945016, 893773};

/* EC-Divisors contest (single invocation time, us) */
static const rust_times RUST_EC_DIVISORS = {466230, 10784};

/**
 * Run a benchmark matching the Rust competition's run_bench! macro:
 *   start = Instant::now()
 *   for _ in 0..N { black_box(op); }
 *   elapsed = (now - start).as_micros()
 *
 * Returns total elapsed microseconds.
 */
template<typename T> static double run_bench(T &&function, const char *name, size_t n_iters, const rust_times &rust)
{
    const auto start = NOW();
    for (size_t i = 0; i < n_iters; ++i)
        function();
    const double elapsed_us = NOW_DIFF(start);

    const double vs_ref = (rust.reference - elapsed_us) / rust.reference * 100.0;
    const double vs_winner = (rust.winner - elapsed_us) / rust.winner * 100.0;

    std::cout << name << "..." << std::endl;
    std::cout << "  C++ took " << std::fixed << std::setprecision(0) << elapsed_us << "us" << std::endl;
    std::cout << "  Rust reference took " << std::setprecision(0) << rust.reference << "us"
              << "  (C++ is " << std::setprecision(2) << std::abs(vs_ref) << "% " << (vs_ref >= 0 ? "faster" : "slower")
              << ")" << std::endl;
    std::cout << "  Rust winner took " << std::setprecision(0) << rust.winner << "us"
              << "  (C++ is " << std::setprecision(2) << std::abs(vs_winner) << "% "
              << (vs_winner >= 0 ? "faster" : "slower") << ")" << std::endl;
    std::cout << std::endl;

    return elapsed_us;
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
            std::cerr << "Usage: ranshaw-benchmark-contest [--init|--autotune]" << std::endl;
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
     * Section 1: RanShaw Contest — 10 weighted ops
     * Matches ranshaw-contest/benches/ranshaw.rs exactly:
     *   same ops, same iteration counts, same timing methodology
     * ================================================================ */

    std::cout << std::endl;
    std::cout << "=== FCMP++ RanShaw Contest Benchmark ===" << std::endl;
    std::cout << std::endl;

    /* Set up field elements */
    static const unsigned char test_a_bytes[32] = {0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, 0xbe, 0xba, 0xfe,
                                                   0xca, 0xef, 0xbe, 0xad, 0xde, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    static const unsigned char test_b_bytes[32] = {0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x0d, 0xf0, 0xad,
                                                   0xba, 0xce, 0xfa, 0xed, 0xfe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

    fq_fe fq_a, fq_b, fq_c;
    fq_frombytes(fq_a, test_a_bytes);
    fq_frombytes(fq_b, test_b_bytes);

    /* Set up curve points */
    ran_jacobian h_G, h_2G, h_result;
    fp_copy(h_G.X, RAN_GX);
    fp_copy(h_G.Y, RAN_GY);
    fp_1(h_G.Z);
    ran_dbl(&h_2G, &h_G);

    shaw_jacobian s_G, s_2G, s_result;
    fq_copy(s_G.X, SHAW_GX);
    fq_copy(s_G.Y, SHAW_GY);
    fq_1(s_G.Z);
    shaw_dbl(&s_2G, &s_G);

    /* Pre-compute valid compressed bytes for decompression benchmarks */
    unsigned char ran_compressed[32];
    unsigned char shaw_compressed[32];
    ran_tobytes(ran_compressed, &h_G);
    shaw_tobytes(shaw_compressed, &s_G);

    /* Weights for scoring */
    constexpr double weights[10] = {0.30, 0.15, 0.15, 0.10, 0.075, 0.075, 0.05, 0.05, 0.025, 0.025};
    double times[10];

    /* 1. Shaw Point Add — 2,000,000 iters (weight 0.30) */
    times[0] = run_bench(
        [&]()
        {
            shaw_add(&s_result, &s_2G, &s_G);
            benchmark_do_not_optimize(s_result);
        },
        "Shaw Point Add",
        2000000,
        RUST_SHAW_POINT_ADD);

    /* 2. Ran Point Add — 2,000,000 iters (weight 0.15) */
    times[1] = run_bench(
        [&]()
        {
            ran_add(&h_result, &h_2G, &h_G);
            benchmark_do_not_optimize(h_result);
        },
        "Ran Point Add",
        2000000,
        RUST_RAN_POINT_ADD);

    /* 3. Field Mul [Fq] — 50,000,000 iters (weight 0.15) */
    fq_copy(fq_c, fq_a);
    times[2] = run_bench(
        [&]()
        {
            fq_mul(fq_c, fq_c, fq_b);
            benchmark_do_not_optimize(fq_c);
        },
        "ranshaw Mul",
        50000000,
        RUST_FIELD_MUL);

    /* 4. Field Invert [Fq] — 200,000 iters (weight 0.10) */
    times[3] = run_bench(
        [&]()
        {
            fq_invert(fq_c, fq_a);
            benchmark_do_not_optimize(fq_c);
        },
        "ranshaw invert",
        200000,
        RUST_FIELD_INVERT);

    /* 5. Shaw Decompress — 100,000 iters (weight 0.075) */
    times[4] = run_bench(
        [&]()
        {
            shaw_frombytes(&s_result, shaw_compressed);
            benchmark_do_not_optimize(s_result);
        },
        "Shaw Point Decompression",
        100000,
        RUST_SHAW_DECOMPRESS);

    /* 6. Ran Decompress — 100,000 iters (weight 0.075) */
    times[5] = run_bench(
        [&]()
        {
            ran_frombytes(&h_result, ran_compressed);
            benchmark_do_not_optimize(h_result);
        },
        "Ran Point Decompression",
        100000,
        RUST_RAN_DECOMPRESS);

    /* 7. Field Add [Fq] — 200,000,000 iters (weight 0.05) */
    fq_copy(fq_c, fq_a);
    times[6] = run_bench(
        [&]()
        {
            fq_add(fq_c, fq_c, fq_b);
            benchmark_do_not_optimize(fq_c);
        },
        "ranshaw Add",
        200000000,
        RUST_FIELD_ADD);

    /* 8. Field Sub [Fq] — 200,000,000 iters (weight 0.05) */
    fq_copy(fq_c, fq_a);
    times[7] = run_bench(
        [&]()
        {
            fq_sub(fq_c, fq_c, fq_b);
            benchmark_do_not_optimize(fq_c);
        },
        "ranshaw Sub",
        200000000,
        RUST_FIELD_SUB);

    /* 9. Shaw Scalar Mul — 10,000 iters (weight 0.025) */
    times[8] = run_bench(
        [&]()
        {
            shaw_scalarmult(&s_result, test_scalar, &s_G);
            benchmark_do_not_optimize(s_result);
        },
        "Shaw Point Mul",
        10000,
        RUST_SHAW_SCALAR_MUL);

    /* 10. Ran Scalar Mul — 10,000 iters (weight 0.025) */
    times[9] = run_bench(
        [&]()
        {
            ran_scalarmult(&h_result, test_scalar, &h_G);
            benchmark_do_not_optimize(h_result);
        },
        "Ran Point Mul",
        10000,
        RUST_RAN_SCALAR_MUL);

    /* Compute weighted improvement vs Rust reference (same formula as competition) */
    static const char *op_names[10] = {
        "Shaw Point Add",
        "Ran Point Add",
        "ranshaw Mul",
        "ranshaw invert",
        "Shaw Decompress",
        "Ran Decompress",
        "ranshaw Add",
        "ranshaw Sub",
        "Shaw Point Mul",
        "Ran Point Mul"};

    static const rust_times *rust_all[10] = {
        &RUST_SHAW_POINT_ADD,
        &RUST_RAN_POINT_ADD,
        &RUST_FIELD_MUL,
        &RUST_FIELD_INVERT,
        &RUST_SHAW_DECOMPRESS,
        &RUST_RAN_DECOMPRESS,
        &RUST_FIELD_ADD,
        &RUST_FIELD_SUB,
        &RUST_SHAW_SCALAR_MUL,
        &RUST_RAN_SCALAR_MUL};

    std::cout << "--- Summary (vs Rust reference, competition scoring) ---" << std::endl;
    std::cout << std::endl;

    double weighted_improvement_vs_ref = 0.0;
    double weighted_improvement_vs_winner = 0.0;

    for (int i = 0; i < 10; i++)
    {
        const double improvement_vs_ref = (rust_all[i]->reference - times[i]) / rust_all[i]->reference * 100.0;
        const double improvement_vs_winner = (rust_all[i]->winner - times[i]) / rust_all[i]->winner * 100.0;
        const double weighted_ref = weights[i] * improvement_vs_ref;
        const double weighted_win = weights[i] * improvement_vs_winner;
        weighted_improvement_vs_ref += weighted_ref;
        weighted_improvement_vs_winner += weighted_win;

        std::cout << std::setw(28) << op_names[i] << ":  C++ " << std::fixed << std::setprecision(0) << std::setw(10)
                  << times[i] << "us  vs ref " << std::setprecision(2) << std::setw(7) << improvement_vs_ref
                  << "%  vs winner " << std::setw(7) << improvement_vs_winner << "%" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "  Overall improvement vs Rust reference: " << std::fixed << std::setprecision(2)
              << weighted_improvement_vs_ref << "%" << std::endl;
    std::cout << "  Overall improvement vs Rust winner:    " << std::fixed << std::setprecision(2)
              << weighted_improvement_vs_winner << "%" << std::endl;

    /* ================================================================
     * Section 2: EC-Divisors Contest
     * Matches ec-divisors-contest/benches/divisors.rs:
     *   ScalarDecomposition::new(scalar) + scalar.scalar_mul_divisor(point)
     *   = 254 points (NUM_BITS+1 for 253-bit scalar) -> divisor construction
     * ================================================================ */

    std::cout << std::endl;
    std::cout << "=== FCMP++ EC-Divisors Contest Benchmark ===" << std::endl;
    std::cout << std::endl;

    constexpr size_t DIVISOR_N = 254;

    /* Pre-compute 254 affine points for "divisor only" benchmarks */
    std::vector<ran_jacobian> h_jac_pts(DIVISOR_N);
    ran_copy(&h_jac_pts[0], &h_G);
    for (size_t i = 1; i < DIVISOR_N; i++)
        ran_dbl(&h_jac_pts[i], &h_jac_pts[i - 1]);

    std::vector<ran_affine> h_aff_pts(DIVISOR_N);
    ran_batch_to_affine(h_aff_pts.data(), h_jac_pts.data(), DIVISOR_N);

    std::vector<shaw_jacobian> s_jac_pts(DIVISOR_N);
    shaw_copy(&s_jac_pts[0], &s_G);
    for (size_t i = 1; i < DIVISOR_N; i++)
        shaw_dbl(&s_jac_pts[i], &s_jac_pts[i - 1]);

    std::vector<shaw_affine> s_aff_pts(DIVISOR_N);
    shaw_batch_to_affine(s_aff_pts.data(), s_jac_pts.data(), DIVISOR_N);

    ran_divisor h_div;
    shaw_divisor s_div;

    /* Pre-allocate buffers for full pipeline benchmarks */
    std::vector<ran_jacobian> h_pipeline_jac(DIVISOR_N);
    std::vector<ran_affine> h_pipeline_aff(DIVISOR_N);
    std::vector<shaw_jacobian> s_pipeline_jac(DIVISOR_N);
    std::vector<shaw_affine> s_pipeline_aff(DIVISOR_N);

    /* Full pipeline (Ran, n=254) — direct comparison to Rust ec-divisors contest */
    run_bench(
        [&]()
        {
            ran_scalarmult_vartime(&h_pipeline_jac[0], test_scalar, &h_G);
            for (size_t i = 1; i < DIVISOR_N; i++)
                ran_dbl(&h_pipeline_jac[i], &h_pipeline_jac[i - 1]);

            ran_batch_to_affine(h_pipeline_aff.data(), h_pipeline_jac.data(), DIVISOR_N);

            ran_compute_divisor(&h_div, h_pipeline_aff.data(), DIVISOR_N);
            benchmark_do_not_optimize(h_div.a.coeffs[0]);
        },
        "Ran full pipeline n=254 (x1)",
        1,
        RUST_EC_DIVISORS);

    /* Full pipeline (Shaw, n=254) — not in the Rust competition */
    {
        const auto start = NOW();
        shaw_scalarmult_vartime(&s_pipeline_jac[0], test_scalar, &s_G);
        for (size_t i = 1; i < DIVISOR_N; i++)
            shaw_dbl(&s_pipeline_jac[i], &s_pipeline_jac[i - 1]);

        shaw_batch_to_affine(s_pipeline_aff.data(), s_pipeline_jac.data(), DIVISOR_N);

        shaw_compute_divisor(&s_div, s_pipeline_aff.data(), DIVISOR_N);
        benchmark_do_not_optimize(s_div.a.coeffs[0]);
        const double elapsed_us = NOW_DIFF(start);
        std::cout << "Shaw full pipeline n=254 (x1)..." << std::endl;
        std::cout << "  C++ took " << std::fixed << std::setprecision(0) << elapsed_us << "us" << std::endl;
        std::cout << std::endl;
    }

    /* Divisor only (Ran, n=254) — isolates the Lagrange interpolation cost */
    {
        const auto start = NOW();
        ran_compute_divisor(&h_div, h_aff_pts.data(), DIVISOR_N);
        benchmark_do_not_optimize(h_div.a.coeffs[0]);
        const double elapsed_us = NOW_DIFF(start);
        std::cout << "Ran divisor only n=254 (x1)..." << std::endl;
        std::cout << "  C++ took " << std::fixed << std::setprecision(0) << elapsed_us << "us" << std::endl;
        std::cout << std::endl;
    }

    /* Divisor only (Shaw, n=254) */
    {
        const auto start = NOW();
        shaw_compute_divisor(&s_div, s_aff_pts.data(), DIVISOR_N);
        benchmark_do_not_optimize(s_div.a.coeffs[0]);
        const double elapsed_us = NOW_DIFF(start);
        std::cout << "Shaw divisor only n=254 (x1)..." << std::endl;
        std::cout << "  C++ took " << std::fixed << std::setprecision(0) << elapsed_us << "us" << std::endl;
        std::cout << std::endl;
    }

    benchmark_teardown(state);

    return 0;
}
