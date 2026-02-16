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

// api_divisor.cpp — Implementation of RanDivisor/ShawDivisor C++ API methods.
// Batch-converts Jacobian points to affine before computing the divisor witness.

#include "fp_frombytes.h"
#include "fp_tobytes.h"
#include "fq_frombytes.h"
#include "fq_tobytes.h"
#include "ran_batch_affine.h"
#include "ranshaw_divisor.h"
#include "shaw_batch_affine.h"

#include <vector>

namespace ranshaw
{

    /* Upper bound on divisor size: 1M points. Prevents unbounded allocations. */
    static constexpr size_t MAX_DIVISOR_SIZE = 1u << 20;

    /* ---- RanDivisor ---- */

    void RanDivisor::sync_wrappers()
    {
        a_.raw() = div_.a;
        b_.raw() = div_.b;
    }

    RanDivisor RanDivisor::compute(const RanPoint *points, size_t n)
    {
        if (n == 0 || !points || n > MAX_DIVISOR_SIZE)
            return RanDivisor();

        std::vector<ran_jacobian> jac(n);
        for (size_t i = 0; i < n; i++)
            ran_copy(&jac[i], &points[i].raw());

        std::vector<ran_affine> aff(n);
        ran_batch_to_affine(aff.data(), jac.data(), n);

        RanDivisor d;
        ran_compute_divisor(&d.div_, aff.data(), n);
        d.sync_wrappers();
        return d;
    }

    std::array<uint8_t, 32> RanDivisor::evaluate(const uint8_t x_bytes[32], const uint8_t y_bytes[32]) const
    {
        fp_fe x, y, result;
        fp_frombytes(x, x_bytes);
        fp_frombytes(y, y_bytes);
        ran_evaluate_divisor(result, &div_, x, y);

        std::array<uint8_t, 32> out;
        fp_tobytes(out.data(), result);
        return out;
    }

    /* ---- ShawDivisor ---- */

    void ShawDivisor::sync_wrappers()
    {
        a_.raw() = div_.a;
        b_.raw() = div_.b;
    }

    ShawDivisor ShawDivisor::compute(const ShawPoint *points, size_t n)
    {
        if (n == 0 || !points || n > MAX_DIVISOR_SIZE)
            return ShawDivisor();

        std::vector<shaw_jacobian> jac(n);
        for (size_t i = 0; i < n; i++)
            shaw_copy(&jac[i], &points[i].raw());

        std::vector<shaw_affine> aff(n);
        shaw_batch_to_affine(aff.data(), jac.data(), n);

        ShawDivisor d;
        shaw_compute_divisor(&d.div_, aff.data(), n);
        d.sync_wrappers();
        return d;
    }

    std::array<uint8_t, 32> ShawDivisor::evaluate(const uint8_t x_bytes[32], const uint8_t y_bytes[32]) const
    {
        fq_fe x, y, result;
        fq_frombytes(x, x_bytes);
        fq_frombytes(y, y_bytes);
        shaw_evaluate_divisor(result, &div_, x, y);

        std::array<uint8_t, 32> out;
        fq_tobytes(out.data(), result);
        return out;
    }

} // namespace ranshaw
