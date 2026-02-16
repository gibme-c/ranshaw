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
 * @file ranshaw_benchmark.h
 * @brief Micro-benchmarking framework with adaptive batching and progress display.
 *
 * Provides benchmark() / benchmark_long() templates that measure median, min, max, and total
 * time per operation. Uses priority elevation and CPU pinning for stable results on Windows/Linux.
 * Batch sizes are auto-tuned from warmup timing to target ~10ms per batch.
 */

#ifndef RANSHAW_BENCHMARK_H
#define RANSHAW_BENCHMARK_H

#ifndef BENCHMARK_PERFORMANCE_ITERATIONS
#define BENCHMARK_PERFORMANCE_ITERATIONS 50000
#endif

#ifndef BENCHMARK_PERFORMANCE_ITERATIONS_LONG_MULTIPLIER
#define BENCHMARK_PERFORMANCE_ITERATIONS_LONG_MULTIPLIER 10
#endif

#ifndef BENCHMARK_PREFIX_WIDTH
#define BENCHMARK_PREFIX_WIDTH 40
#endif

#ifndef BENCHMARK_COLUMN_WIDTH
#define BENCHMARK_COLUMN_WIDTH 14
#endif

#ifndef BENCHMARK_PRECISION
#define BENCHMARK_PRECISION 3
#endif

#ifndef BENCHMARK_WARMUP_ITERATIONS
#define BENCHMARK_WARMUP_ITERATIONS 10000
#endif

#ifndef BENCHMARK_BATCH_SIZE
#define BENCHMARK_BATCH_SIZE 1000
#endif

#ifndef BENCHMARK_TARGET_BATCH_US
#define BENCHMARK_TARGET_BATCH_US 10000.0
#endif

#define BENCHMARK_PERFORMANCE_ITERATIONS_LONG \
    BENCHMARK_PERFORMANCE_ITERATIONS *BENCHMARK_PERFORMANCE_ITERATIONS_LONG_MULTIPLIER

#define NOW() std::chrono::high_resolution_clock::now()
#define NOW_DIFF(b) \
    static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(NOW() - b).count()) / 1'000.0

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

struct benchmark_state
{
#ifdef _WIN32
    DWORD original_priority_class;
    int original_thread_priority;
    DWORD_PTR original_affinity_mask;
#endif
};

static inline benchmark_state benchmark_setup()
{
    benchmark_state state = {};

#ifdef _WIN32
    const auto process = GetCurrentProcess();
    const auto thread = GetCurrentThread();

    state.original_priority_class = GetPriorityClass(process);
    state.original_thread_priority = GetThreadPriority(thread);
    state.original_affinity_mask = SetThreadAffinityMask(thread, 1);

    SetPriorityClass(process, HIGH_PRIORITY_CLASS);
    SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST);
#elif defined(__linux__)
    if (nice(-20) == -1)
    {
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif

    return state;
}

static inline void benchmark_teardown(const benchmark_state &state)
{
#ifdef _WIN32
    const auto process = GetCurrentProcess();
    const auto thread = GetCurrentThread();

    SetPriorityClass(process, state.original_priority_class);
    SetThreadPriority(thread, state.original_thread_priority);

    if (state.original_affinity_mask != 0)
    {
        SetThreadAffinityMask(thread, state.original_affinity_mask);
    }
#else
    (void)state;
#endif
}

template<typename T> static inline void benchmark_do_not_optimize(const T &value)
{
#if defined(_MSC_VER)
    const volatile auto *sink = &value;
    (void)sink;
#else
    asm volatile("" : : "g"(value) : "memory");
#endif
}

static inline void
    benchmark_header(int8_t prefix_width = BENCHMARK_PREFIX_WIDTH, int8_t column_width = BENCHMARK_COLUMN_WIDTH)
{
    std::cout << std::setw(prefix_width) << "BENCHMARK TESTS"
              << ": " << std::setw(10) << " " << std::setw(column_width) << "Median" << std::setw(column_width)
              << "Minimum" << std::setw(column_width) << "Maximum" << std::setw(column_width + 8) << "Total"
              << std::endl;
}

template<typename T>
void benchmark(
    T &&function,
    const std::string &functionName = "",
    const size_t iterations = BENCHMARK_PERFORMANCE_ITERATIONS,
    const size_t warmup = BENCHMARK_WARMUP_ITERATIONS,
    int8_t prefix_width = BENCHMARK_PREFIX_WIDTH,
    int8_t column_width = BENCHMARK_COLUMN_WIDTH,
    int8_t precision = BENCHMARK_PRECISION)
{
    if (static_cast<double>(iterations) > DBL_MAX)
    {
        throw std::invalid_argument("iterations exceeds bounds of double");
    }

    if (!functionName.empty())
    {
        std::cout << std::setw(prefix_width) << functionName.substr(0, prefix_width) << ": " << std::flush;
    }

    const auto warmup_timer = NOW();

    for (size_t i = 0; i < warmup; ++i)
    {
        function();
    }

    const double warmup_us = NOW_DIFF(warmup_timer);

    const size_t max_batch = (iterations >= 10) ? (iterations / 10) : 1;
    size_t batch_size;

    if (warmup_us > 0.0)
    {
        const double est_per_op_us = warmup_us / static_cast<double>(warmup);
        const double ideal = BENCHMARK_TARGET_BATCH_US / est_per_op_us;

        if (ideal < 1.0)
        {
            batch_size = 1;
        }
        else if (ideal > static_cast<double>(max_batch))
        {
            batch_size = max_batch;
        }
        else
        {
            batch_size = static_cast<size_t>(ideal);
        }
    }
    else
    {
        batch_size = (BENCHMARK_BATCH_SIZE < max_batch) ? BENCHMARK_BATCH_SIZE : max_batch;
    }

    const size_t num_batches = (iterations + batch_size - 1) / batch_size;

    constexpr size_t progress_width = 10;
    size_t dots_printed = 0;

    std::vector<double> batch_times(num_batches);

    for (size_t b = 0; b < num_batches; ++b)
    {
        const auto batch_timer = NOW();

        for (size_t i = 0; i < batch_size; ++i)
        {
            function();
        }

        batch_times[b] = NOW_DIFF(batch_timer) / static_cast<double>(batch_size);

        if (num_batches < progress_width)
        {
            std::cout << "." << std::flush;
            ++dots_printed;
        }
        else
        {
            const size_t target_raw = ((b + 1) * progress_width) / num_batches;
            const size_t target_dots = (target_raw < progress_width) ? target_raw : progress_width;
            while (dots_printed < target_dots)
            {
                std::cout << "." << std::flush;
                ++dots_printed;
            }
        }
    }

    while (dots_printed < progress_width)
    {
        std::cout << " " << std::flush;
        ++dots_printed;
    }

    std::sort(batch_times.begin(), batch_times.end());

    const auto median_time = batch_times[num_batches / 2];
    const auto minimum_time = batch_times.front();
    const auto maximum_time = batch_times.back();

    double total_time = 0;
    for (const auto &t : batch_times)
    {
        total_time += t;
    }
    total_time *= static_cast<double>(batch_size);

    std::cout << std::fixed << std::setprecision(precision) << std::setw(column_width) << median_time << std::fixed
              << std::setprecision(precision) << std::setw(column_width) << minimum_time << std::fixed
              << std::setprecision(precision) << std::setw(column_width) << maximum_time << std::fixed
              << std::setprecision(precision) << std::setw(column_width + 8) << total_time << " us" << std::endl;
}

template<typename T>
void benchmark_long(
    T &&function,
    const std::string &functionName = "",
    const size_t iterations = BENCHMARK_PERFORMANCE_ITERATIONS_LONG,
    const size_t warmup = BENCHMARK_WARMUP_ITERATIONS,
    int8_t prefix_width = BENCHMARK_PREFIX_WIDTH,
    int8_t column_width = BENCHMARK_COLUMN_WIDTH,
    int8_t precision = BENCHMARK_PRECISION)
{
    benchmark(function, functionName, iterations, warmup, prefix_width, column_width, precision);
}

#endif // RANSHAW_BENCHMARK_H
