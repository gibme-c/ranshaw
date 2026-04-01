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
 * @file ranshaw.h
 * @brief Public C++ API for the RanShaw library.
 *
 * RanShaw is an elliptic curve library implementing the Ran/Shaw
 * curve cycle for FCMP++ integration. The two curves form a cycle:
 *
 * - **Ran**: y^2 = x^3 - 3x + b over F_p (p = 2^255 - 19), group order q
 * - **Shaw**: y^2 = x^3 - 3x + b over F_q (q = 2^255 - gamma), group order p
 *
 * This header provides the idiomatic C++ API with type-safe classes,
 * std::optional validation, and RAII:
 *
 * - **RanScalar / ShawScalar**: Scalar field elements with arithmetic operators.
 * - **RanPoint / ShawPoint**: Curve points with scalar multiplication and MSM.
 * - **FpPolynomial / FqPolynomial**: Polynomial arithmetic over the base fields.
 * - **RanDivisor / ShawDivisor**: EC-divisor witness computation and evaluation.
 *
 * All classes live in the `ranshaw` namespace.
 */

#ifndef RANSHAW_H
#define RANSHAW_H

/* Runtime dispatch (ranshaw_init / ranshaw_autotune) */
#include "ranshaw_dispatch.h"

/* Public C++ API classes */
#include "ranshaw_divisor.h"
#include "ranshaw_point.h"
#include "ranshaw_polynomial.h"
#include "ranshaw_scalar.h"

namespace ranshaw
{

    /// Initialize the library: detect CPU features and select optimal backends. Thread-safe (std::call_once).
    inline void init()
    {
        ranshaw_init();
    }

    /// Benchmark all available backends and select the fastest for each dispatch slot.
    inline void autotune()
    {
        ranshaw_autotune();
    }

} // namespace ranshaw

#endif /* RANSHAW_H */
