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
 * @file ranshaw_scalar.h
 * @brief Type-safe C++ wrappers for Ran and Shaw scalar field elements.
 *
 * RanScalar wraps fq_fe (elements of F_q, the Ran scalar field / Shaw base field).
 * ShawScalar wraps fp_fe (elements of F_p, the Shaw scalar field / Ran base field).
 * This duality is the cycle property: each curve's scalar field is the other's base field.
 *
 * All arithmetic is modular and constant-time. Equality comparison uses CT XOR-accumulate.
 */

#ifndef RANSHAW_API_SCALAR_H
#define RANSHAW_API_SCALAR_H

#include "ran_scalar.h"
#include "ranshaw_wei25519.h"
#include "shaw_scalar.h"

#include <array>
#include <cstring>
#include <iomanip>
#include <optional>
#include <ostream>

namespace ranshaw
{

    /**
     * @brief Scalar field element for the Ran curve (element of F_q).
     *
     * Represents an integer mod q where q = 2^255 - gamma (a Crandall prime, gamma ~ 2^126).
     * Internally stored as fq_fe in the active backend's representation.
     */
    class RanScalar
    {
      public:
        RanScalar()
        {
            ran_scalar_zero(fe_);
        }

        RanScalar(const RanScalar &other)
        {
            std::memcpy(fe_, other.fe_, sizeof(fq_fe));
        }

        RanScalar &operator=(const RanScalar &other)
        {
            std::memcpy(fe_, other.fe_, sizeof(fq_fe));
            return *this;
        }

        static RanScalar zero()
        {
            return RanScalar();
        }

        static RanScalar one()
        {
            RanScalar s;
            ran_scalar_one(s.fe_);
            return s;
        }

        bool is_zero() const
        {
            return ran_scalar_is_zero(fe_) != 0;
        }

        bool operator==(const RanScalar &other) const
        {
            auto a = to_bytes();
            auto b = other.to_bytes();
            unsigned diff = 0;
            for (size_t i = 0; i < 32; i++)
                diff |= static_cast<unsigned>(a[i] ^ b[i]);
            return diff == 0;
        }

        bool operator!=(const RanScalar &other) const
        {
            return !(*this == other);
        }

        RanScalar operator+(const RanScalar &other) const
        {
            RanScalar r;
            ran_scalar_add(r.fe_, fe_, other.fe_);
            return r;
        }

        RanScalar operator-(const RanScalar &other) const
        {
            RanScalar r;
            ran_scalar_sub(r.fe_, fe_, other.fe_);
            return r;
        }

        RanScalar operator*(const RanScalar &other) const
        {
            RanScalar r;
            ran_scalar_mul(r.fe_, fe_, other.fe_);
            return r;
        }

        RanScalar operator-() const
        {
            RanScalar r;
            ran_scalar_neg(r.fe_, fe_);
            return r;
        }

        RanScalar sq() const
        {
            RanScalar r;
            ran_scalar_sq(r.fe_, fe_);
            return r;
        }

        /// Serialize to 32-byte little-endian canonical form.
        std::array<uint8_t, 32> to_bytes() const;

        /// Deserialize from 32-byte LE. Returns nullopt if value >= q.
        static std::optional<RanScalar> from_bytes(const uint8_t bytes[32]);

        /// Modular inverse via Fermat's little theorem (a^{q-2} mod q). Returns nullopt for zero.
        std::optional<RanScalar> invert() const;

        /// Reduce a 64-byte (512-bit) wide integer mod q. Used for hash-to-scalar.
        static RanScalar reduce_wide(const uint8_t bytes[64]);

        /// Fused multiply-add: returns a*b + c (mod q).
        static RanScalar muladd(const RanScalar &a, const RanScalar &b, const RanScalar &c);

        /// Direct access to the underlying field element.
        const fq_fe &raw() const
        {
            return fe_;
        }

        fq_fe &raw()
        {
            return fe_;
        }

      private:
        fq_fe fe_;
    };

    /**
     * @brief Scalar field element for the Shaw curve (element of F_p).
     *
     * Represents an integer mod p where p = 2^255 - 19.
     * Internally stored as fp_fe in the active backend's representation.
     */
    class ShawScalar
    {
      public:
        ShawScalar()
        {
            shaw_scalar_zero(fe_);
        }

        ShawScalar(const ShawScalar &other)
        {
            std::memcpy(fe_, other.fe_, sizeof(fp_fe));
        }

        ShawScalar &operator=(const ShawScalar &other)
        {
            std::memcpy(fe_, other.fe_, sizeof(fp_fe));
            return *this;
        }

        static ShawScalar zero()
        {
            return ShawScalar();
        }

        static ShawScalar one()
        {
            ShawScalar s;
            shaw_scalar_one(s.fe_);
            return s;
        }

        bool is_zero() const
        {
            return shaw_scalar_is_zero(fe_) != 0;
        }

        bool operator==(const ShawScalar &other) const
        {
            auto a = to_bytes();
            auto b = other.to_bytes();
            unsigned diff = 0;
            for (size_t i = 0; i < 32; i++)
                diff |= static_cast<unsigned>(a[i] ^ b[i]);
            return diff == 0;
        }

        bool operator!=(const ShawScalar &other) const
        {
            return !(*this == other);
        }

        ShawScalar operator+(const ShawScalar &other) const
        {
            ShawScalar r;
            shaw_scalar_add(r.fe_, fe_, other.fe_);
            return r;
        }

        ShawScalar operator-(const ShawScalar &other) const
        {
            ShawScalar r;
            shaw_scalar_sub(r.fe_, fe_, other.fe_);
            return r;
        }

        ShawScalar operator*(const ShawScalar &other) const
        {
            ShawScalar r;
            shaw_scalar_mul(r.fe_, fe_, other.fe_);
            return r;
        }

        ShawScalar operator-() const
        {
            ShawScalar r;
            shaw_scalar_neg(r.fe_, fe_);
            return r;
        }

        ShawScalar sq() const
        {
            ShawScalar r;
            shaw_scalar_sq(r.fe_, fe_);
            return r;
        }

        /// Serialize to 32-byte little-endian canonical form.
        std::array<uint8_t, 32> to_bytes() const;

        /// Deserialize from 32-byte LE. Returns nullopt if value >= p.
        static std::optional<ShawScalar> from_bytes(const uint8_t bytes[32]);

        /// Modular inverse via Fermat's little theorem (a^{p-2} mod p). Returns nullopt for zero.
        std::optional<ShawScalar> invert() const;

        /// Reduce a 64-byte (512-bit) wide integer mod p. Used for hash-to-scalar.
        static ShawScalar reduce_wide(const uint8_t bytes[64]);

        /// Fused multiply-add: returns a*b + c (mod p).
        static ShawScalar muladd(const ShawScalar &a, const ShawScalar &b, const ShawScalar &c);

        /// Direct access to the underlying field element.
        const fp_fe &raw() const
        {
            return fe_;
        }

        fp_fe &raw()
        {
            return fe_;
        }

      private:
        fp_fe fe_;
    };

    /// Convert a Wei25519 x-coordinate to a Shaw scalar. Returns nullopt if not a valid Shaw field element.
    std::optional<ShawScalar> shaw_scalar_from_wei25519_x(const uint8_t x_bytes[32]);

    inline std::ostream &operator<<(std::ostream &os, const RanScalar &s)
    {
        const auto bytes = s.to_bytes();
        const auto flags = os.flags();
        for (size_t i = 32; i-- > 0;)
            os << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(bytes[i]);
        os.flags(flags);
        return os;
    }

    inline std::ostream &operator<<(std::ostream &os, const ShawScalar &s)
    {
        const auto bytes = s.to_bytes();
        const auto flags = os.flags();
        for (size_t i = 32; i-- > 0;)
            os << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(bytes[i]);
        os.flags(flags);
        return os;
    }

} // namespace ranshaw

#endif // RANSHAW_API_SCALAR_H
