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
 * @file ranshaw_divisor.h
 * @brief Type-safe C++ wrappers for elliptic curve divisors on Ran and Shaw.
 *
 * A divisor D on an elliptic curve E is represented as a rational function f(x,y) = a(x) + y*b(x),
 * where a and b are univariate polynomials. Divisors are the core primitive in FCMP++ membership
 * proofs: given a set of curve points, compute() builds the unique divisor whose zeros are those
 * points, and evaluate() evaluates that function at an arbitrary (x, y).
 */

#ifndef RANSHAW_API_DIVISOR_H
#define RANSHAW_API_DIVISOR_H

#include "divisor.h"
#include "ranshaw_point.h"
#include "ranshaw_polynomial.h"

#include <array>
#include <cstddef>
#include <ostream>

namespace ranshaw
{

    /**
     * @brief Divisor on the Ran curve, represented as f(x,y) = a(x) + y*b(x).
     *
     * The polynomials a(x) and b(x) are over F_p. Use compute() to build from a set of points,
     * then evaluate() to probe the divisor at any field point.
     */
    class RanDivisor
    {
      public:
        RanDivisor() = default;

        /// Build the divisor whose zeros are the given n points.
        static RanDivisor compute(const RanPoint *points, size_t n);

        /// Evaluate f(x,y) = a(x) + y*b(x) at the given serialized coordinates.
        std::array<uint8_t, 32> evaluate(const uint8_t x_bytes[32], const uint8_t y_bytes[32]) const;

        /// The a(x) polynomial component.
        const FpPolynomial &a() const
        {
            return a_;
        }

        const FpPolynomial &b() const
        {
            return b_;
        }

        const ran_divisor &raw() const
        {
            return div_;
        }

        ran_divisor &raw()
        {
            return div_;
        }

      private:
        ran_divisor div_;
        FpPolynomial a_;
        FpPolynomial b_;

        void sync_wrappers();
    };

    /**
     * @brief Divisor on the Shaw curve, represented as f(x,y) = a(x) + y*b(x).
     *
     * The polynomials a(x) and b(x) are over F_q. Use compute() to build from a set of points,
     * then evaluate() to probe the divisor at any field point.
     */
    class ShawDivisor
    {
      public:
        ShawDivisor() = default;

        /// Build the divisor whose zeros are the given n points.
        static ShawDivisor compute(const ShawPoint *points, size_t n);

        /// Evaluate f(x,y) = a(x) + y*b(x) at the given serialized coordinates.
        std::array<uint8_t, 32> evaluate(const uint8_t x_bytes[32], const uint8_t y_bytes[32]) const;

        /// The a(x) polynomial component.
        const FqPolynomial &a() const
        {
            return a_;
        }

        const FqPolynomial &b() const
        {
            return b_;
        }

        const shaw_divisor &raw() const
        {
            return div_;
        }

        shaw_divisor &raw()
        {
            return div_;
        }

      private:
        shaw_divisor div_;
        FqPolynomial a_;
        FqPolynomial b_;

        void sync_wrappers();
    };

    inline std::ostream &operator<<(std::ostream &os, const RanDivisor &d)
    {
        os << "RanDivisor {a: " << d.a() << ", b: " << d.b() << "}";
        return os;
    }

    inline std::ostream &operator<<(std::ostream &os, const ShawDivisor &d)
    {
        os << "ShawDivisor {a: " << d.a() << ", b: " << d.b() << "}";
        return os;
    }

} // namespace ranshaw

#endif // RANSHAW_API_DIVISOR_H
