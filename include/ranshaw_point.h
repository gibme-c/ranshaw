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
 * @file ranshaw_point.h
 * @brief Type-safe C++ wrappers for Ran and Shaw elliptic curve points.
 *
 * RanPoint and ShawPoint represent points on the Ran/Shaw curve cycle
 * (y^2 = x^3 - 3x + b). Internally stored in Jacobian projective coordinates (X:Y:Z)
 * for efficient group operations. Serialization uses compressed form (32 bytes, bit 255
 * encodes y-parity).
 *
 * Constant-time scalar multiplication is available via scalar_mul(); use the _vartime
 * variants only when the scalar is public knowledge.
 */

#ifndef RANSHAW_API_POINT_H
#define RANSHAW_API_POINT_H

#include "ran_add.h"
#include "ran_constants.h"
#include "ran_dbl.h"
#include "ran_ops.h"
#include "ranshaw_scalar.h"
#include "shaw_add.h"
#include "shaw_constants.h"
#include "shaw_dbl.h"
#include "shaw_ops.h"

#include <array>
#include <cstddef>
#include <iomanip>
#include <optional>
#include <ostream>

namespace ranshaw
{

    /**
     * @brief Point on the Ran curve: y^2 = x^3 - 3x + b over F_p (p = 2^255 - 19).
     *
     * Group order is q (the Shaw base field prime). Cofactor 1.
     * Internally stored as Jacobian coordinates (X:Y:Z) where affine x = X/Z^2, y = Y/Z^3.
     */
    class RanPoint
    {
      public:
        RanPoint()
        {
            ran_identity(&jac_);
        }

        RanPoint(const RanPoint &other)
        {
            ran_copy(&jac_, &other.jac_);
        }

        RanPoint &operator=(const RanPoint &other)
        {
            ran_copy(&jac_, &other.jac_);
            return *this;
        }

        static RanPoint identity()
        {
            return RanPoint();
        }

        static RanPoint generator()
        {
            RanPoint p;
            fp_copy(p.jac_.X, RAN_GX);
            fp_copy(p.jac_.Y, RAN_GY);
            fp_1(p.jac_.Z);
            return p;
        }

        bool is_identity() const
        {
            return ran_is_identity(&jac_) != 0;
        }

        RanPoint operator-() const
        {
            RanPoint r;
            ran_neg(&r.jac_, &jac_);
            return r;
        }

        RanPoint operator+(const RanPoint &other) const
        {
            if (is_identity())
                return other;
            if (other.is_identity())
                return *this;
            /* Check if x-coordinates match (projective: X1*Z2^2 == X2*Z1^2) */
            fp_fe z1z1, z2z2, u1, u2, diff;
            fp_sq(z1z1, jac_.Z);
            fp_sq(z2z2, other.jac_.Z);
            fp_mul(u1, jac_.X, z2z2);
            fp_mul(u2, other.jac_.X, z1z1);
            fp_sub(diff, u1, u2);
            if (!fp_isnonzero(diff))
            {
                fp_fe s1, s2, t;
                fp_mul(t, other.jac_.Z, z2z2);
                fp_mul(s1, jac_.Y, t);
                fp_mul(t, jac_.Z, z1z1);
                fp_mul(s2, other.jac_.Y, t);
                fp_sub(diff, s1, s2);
                if (!fp_isnonzero(diff))
                    return dbl(); /* P == P */
                return identity(); /* P == -P */
            }
            RanPoint r;
            ran_add(&r.jac_, &jac_, &other.jac_);
            return r;
        }

        RanPoint dbl() const
        {
            RanPoint r;
            ran_dbl(&r.jac_, &jac_);
            return r;
        }

        /// Decompress from 32-byte encoding. Returns nullopt if not a valid on-curve point.
        static std::optional<RanPoint> from_bytes(const uint8_t bytes[32]);

        /// Compress to 32 bytes (x-coordinate LE, bit 255 = y parity).
        std::array<uint8_t, 32> to_bytes() const;

        /// Return just the x-coordinate as 32-byte LE (no y-parity bit).
        std::array<uint8_t, 32> x_coordinate_bytes() const;

        /// Constant-time scalar multiplication using signed 4-bit windowed method.
        RanPoint scalar_mul(const RanScalar &s) const;

        /// Variable-time scalar multiplication using wNAF w=5. Only use with public scalars.
        RanPoint scalar_mul_vartime(const RanScalar &s) const;

        /// Multi-scalar multiplication: sum(scalars[i] * points[i]). Uses Straus (n<=32) or Pippenger.
        static RanPoint multi_scalar_mul(const RanScalar *scalars, const RanPoint *points, size_t n);

        /// Pedersen commitment: blinding*H + sum(values[i]*generators[i]).
        static RanPoint pedersen_commit(
            const RanScalar &blinding,
            const RanPoint &H,
            const RanScalar *values,
            const RanPoint *generators,
            size_t n);

        /// Hash-to-curve (single field element) via RFC 9380 Simplified SWU.
        static RanPoint map_to_curve(const uint8_t u[32]);

        /// Hash-to-curve (two field elements, full RFC 9380 encode-to-curve).
        static RanPoint map_to_curve(const uint8_t u0[32], const uint8_t u1[32]);

        const ran_jacobian &raw() const
        {
            return jac_;
        }

#ifdef RANSHAW_INTERNAL_ACCESS
        ran_jacobian &raw()
        {
            return jac_;
        }
#endif

      private:
        ran_jacobian jac_;
    };

    /**
     * @brief Point on the Shaw curve: y^2 = x^3 - 3x + b over F_q (q = 2^255 - gamma).
     *
     * Group order is p (the Ran base field prime, 2^255 - 19). Cofactor 1.
     * Internally stored as Jacobian coordinates (X:Y:Z) where affine x = X/Z^2, y = Y/Z^3.
     */
    class ShawPoint
    {
      public:
        ShawPoint()
        {
            shaw_identity(&jac_);
        }

        ShawPoint(const ShawPoint &other)
        {
            shaw_copy(&jac_, &other.jac_);
        }

        ShawPoint &operator=(const ShawPoint &other)
        {
            shaw_copy(&jac_, &other.jac_);
            return *this;
        }

        static ShawPoint identity()
        {
            return ShawPoint();
        }

        static ShawPoint generator()
        {
            ShawPoint p;
            fq_copy(p.jac_.X, SHAW_GX);
            fq_copy(p.jac_.Y, SHAW_GY);
            fq_1(p.jac_.Z);
            return p;
        }

        bool is_identity() const
        {
            return shaw_is_identity(&jac_) != 0;
        }

        ShawPoint operator-() const
        {
            ShawPoint r;
            shaw_neg(&r.jac_, &jac_);
            return r;
        }

        ShawPoint operator+(const ShawPoint &other) const
        {
            if (is_identity())
                return other;
            if (other.is_identity())
                return *this;
            fq_fe z1z1, z2z2, u1, u2, diff;
            fq_sq(z1z1, jac_.Z);
            fq_sq(z2z2, other.jac_.Z);
            fq_mul(u1, jac_.X, z2z2);
            fq_mul(u2, other.jac_.X, z1z1);
            fq_sub(diff, u1, u2);
            if (!fq_isnonzero(diff))
            {
                fq_fe s1, s2, t;
                fq_mul(t, other.jac_.Z, z2z2);
                fq_mul(s1, jac_.Y, t);
                fq_mul(t, jac_.Z, z1z1);
                fq_mul(s2, other.jac_.Y, t);
                fq_sub(diff, s1, s2);
                if (!fq_isnonzero(diff))
                    return dbl();
                return identity();
            }
            ShawPoint r;
            shaw_add(&r.jac_, &jac_, &other.jac_);
            return r;
        }

        ShawPoint dbl() const
        {
            ShawPoint r;
            shaw_dbl(&r.jac_, &jac_);
            return r;
        }

        /// Decompress from 32-byte encoding. Returns nullopt if not a valid on-curve point.
        static std::optional<ShawPoint> from_bytes(const uint8_t bytes[32]);

        /// Compress to 32 bytes (x-coordinate LE, bit 255 = y parity).
        std::array<uint8_t, 32> to_bytes() const;

        /// Return just the x-coordinate as 32-byte LE (no y-parity bit).
        std::array<uint8_t, 32> x_coordinate_bytes() const;

        /// Constant-time scalar multiplication using signed 4-bit windowed method.
        ShawPoint scalar_mul(const ShawScalar &s) const;

        /// Variable-time scalar multiplication using wNAF w=5. Only use with public scalars.
        ShawPoint scalar_mul_vartime(const ShawScalar &s) const;

        /// Multi-scalar multiplication: sum(scalars[i] * points[i]). Uses Straus (n<=32) or Pippenger.
        static ShawPoint multi_scalar_mul(const ShawScalar *scalars, const ShawPoint *points, size_t n);

        /// Pedersen commitment: blinding*H + sum(values[i]*generators[i]).
        static ShawPoint pedersen_commit(
            const ShawScalar &blinding,
            const ShawPoint &H,
            const ShawScalar *values,
            const ShawPoint *generators,
            size_t n);

        /// Hash-to-curve (single field element) via RFC 9380 Simplified SWU.
        static ShawPoint map_to_curve(const uint8_t u[32]);

        /// Hash-to-curve (two field elements, full RFC 9380 encode-to-curve).
        static ShawPoint map_to_curve(const uint8_t u0[32], const uint8_t u1[32]);

        const shaw_jacobian &raw() const
        {
            return jac_;
        }

#ifdef RANSHAW_INTERNAL_ACCESS
        shaw_jacobian &raw()
        {
            return jac_;
        }
#endif

      private:
        shaw_jacobian jac_;
    };

    inline std::ostream &operator<<(std::ostream &os, const RanPoint &p)
    {
        const auto bytes = p.to_bytes();
        const auto flags = os.flags();
        for (size_t i = 32; i-- > 0;)
            os << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(bytes[i]);
        os.flags(flags);
        return os;
    }

    inline std::ostream &operator<<(std::ostream &os, const ShawPoint &p)
    {
        const auto bytes = p.to_bytes();
        const auto flags = os.flags();
        for (size_t i = 32; i-- > 0;)
            os << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(bytes[i]);
        os.flags(flags);
        return os;
    }

} // namespace ranshaw

#endif // RANSHAW_API_POINT_H
