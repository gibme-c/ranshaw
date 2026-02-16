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
 * @file ranshaw_polynomial.h
 * @brief Type-safe C++ wrappers for univariate polynomials over F_p and F_q.
 *
 * FpPolynomial and FqPolynomial represent dense univariate polynomials with coefficients
 * in the respective prime fields. Used for divisor computation in FCMP++ proofs.
 * Multiplication uses schoolbook (deg < 32), Karatsuba (32 <= deg < 1024), or ECFFT (deg >= 1024).
 */

#ifndef RANSHAW_API_POLYNOMIAL_H
#define RANSHAW_API_POLYNOMIAL_H

#include "fp_tobytes.h"
#include "fq_tobytes.h"
#include "poly.h"

#include <array>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <utility>

namespace ranshaw
{

    /**
     * @brief Univariate polynomial over F_p (the Ran base field).
     *
     * Coefficients stored in ascending degree order: coeffs[i] is the coefficient of x^i.
     */
    class FpPolynomial
    {
      public:
        FpPolynomial() = default;

        size_t degree() const;

        /// Build from n serialized 32-byte LE coefficients (ascending degree order).
        static FpPolynomial from_coefficients(const uint8_t *coeff_bytes, size_t n);

        /// Build the monic polynomial (x - r0)(x - r1)...(x - r_{n-1}) from n serialized roots.
        static FpPolynomial from_roots(const uint8_t *root_bytes, size_t n);

        /// Evaluate the polynomial at x using Horner's method. Returns 32-byte LE result.
        std::array<uint8_t, 32> evaluate(const uint8_t x[32]) const;

        FpPolynomial operator*(const FpPolynomial &other) const;
        FpPolynomial operator+(const FpPolynomial &other) const;
        FpPolynomial operator-(const FpPolynomial &other) const;

        /// Polynomial division with remainder. Returns (quotient, remainder).
        std::pair<FpPolynomial, FpPolynomial> divmod(const FpPolynomial &divisor) const;

        /// Lagrange interpolation through n points given as serialized 32-byte (x, y) pairs.
        static FpPolynomial interpolate(const uint8_t *x_bytes, const uint8_t *y_bytes, size_t n);

        const fp_poly &raw() const
        {
            return poly_;
        }

        fp_poly &raw()
        {
            return poly_;
        }

      private:
        fp_poly poly_;
    };

    /**
     * @brief Univariate polynomial over F_q (the Shaw base field).
     *
     * Coefficients stored in ascending degree order: coeffs[i] is the coefficient of x^i.
     */
    class FqPolynomial
    {
      public:
        FqPolynomial() = default;

        size_t degree() const;

        /// Build from n serialized 32-byte LE coefficients (ascending degree order).
        static FqPolynomial from_coefficients(const uint8_t *coeff_bytes, size_t n);

        /// Build the monic polynomial (x - r0)(x - r1)...(x - r_{n-1}) from n serialized roots.
        static FqPolynomial from_roots(const uint8_t *root_bytes, size_t n);

        /// Evaluate the polynomial at x using Horner's method. Returns 32-byte LE result.
        std::array<uint8_t, 32> evaluate(const uint8_t x[32]) const;

        FqPolynomial operator*(const FqPolynomial &other) const;
        FqPolynomial operator+(const FqPolynomial &other) const;
        FqPolynomial operator-(const FqPolynomial &other) const;

        /// Polynomial division with remainder. Returns (quotient, remainder).
        std::pair<FqPolynomial, FqPolynomial> divmod(const FqPolynomial &divisor) const;

        /// Lagrange interpolation through n points given as serialized 32-byte (x, y) pairs.
        static FqPolynomial interpolate(const uint8_t *x_bytes, const uint8_t *y_bytes, size_t n);

        const fq_poly &raw() const
        {
            return poly_;
        }

        fq_poly &raw()
        {
            return poly_;
        }

      private:
        fq_poly poly_;
    };

    inline std::ostream &operator<<(std::ostream &os, const FpPolynomial &poly)
    {
        const auto &raw = poly.raw();
        const auto flags = os.flags();
        size_t fp_deg = raw.coeffs.empty() ? 0 : (raw.coeffs.size() - 1);
        os << "FpPolynomial(deg=" << fp_deg << ") [";
        for (size_t c = 0; c < raw.coeffs.size(); ++c)
        {
            if (c > 0)
                os << ", ";
            unsigned char bytes[32];
            fp_tobytes(bytes, raw.coeffs[c].v);
            for (int i = 31; i >= 0; --i)
                os << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(bytes[i]);
        }
        os << "]";
        os.flags(flags);
        return os;
    }

    inline std::ostream &operator<<(std::ostream &os, const FqPolynomial &poly)
    {
        const auto &raw = poly.raw();
        const auto flags = os.flags();
        size_t fq_deg = raw.coeffs.empty() ? 0 : (raw.coeffs.size() - 1);
        os << "FqPolynomial(deg=" << fq_deg << ") [";
        for (size_t c = 0; c < raw.coeffs.size(); ++c)
        {
            if (c > 0)
                os << ", ";
            unsigned char bytes[32];
            fq_tobytes(bytes, raw.coeffs[c].v);
            for (int i = 31; i >= 0; --i)
                os << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(bytes[i]);
        }
        os << "]";
        os.flags(flags);
        return os;
    }

} // namespace ranshaw

#endif // RANSHAW_API_POLYNOMIAL_H
