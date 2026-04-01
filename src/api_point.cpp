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

// api_point.cpp — Implementation of RanPoint/ShawPoint C++ API methods
// (serialization, scalar multiplication, MSM, Pedersen commit, hash-to-curve).

#include "ran_frombytes.h"
#include "ran_map_to_curve.h"
#include "ran_msm_vartime.h"
#include "ran_pedersen.h"
#include "ran_scalarmult.h"
#include "ran_scalarmult_vartime.h"
#include "ran_to_scalar.h"
#include "ran_tobytes.h"
#include "ranshaw_point.h"
#include "ranshaw_secure_erase.h"
#include "shaw_frombytes.h"
#include "shaw_map_to_curve.h"
#include "shaw_msm_vartime.h"
#include "shaw_pedersen.h"
#include "shaw_scalarmult.h"
#include "shaw_scalarmult_vartime.h"
#include "shaw_to_scalar.h"
#include "shaw_tobytes.h"

#include <climits>
#include <cstdint>
#include <vector>

namespace ranshaw
{

    /* ---- RanPoint ---- */

    std::optional<RanPoint> RanPoint::from_bytes(const uint8_t bytes[32])
    {
        RanPoint p;
        if (ran_frombytes(&p.jac_, bytes) != 0)
            return std::nullopt;
        return p;
    }

    std::array<uint8_t, 32> RanPoint::to_bytes() const
    {
        std::array<uint8_t, 32> out;
        ran_tobytes(out.data(), &jac_);
        return out;
    }

    std::array<uint8_t, 32> RanPoint::x_coordinate_bytes() const
    {
        std::array<uint8_t, 32> out;
        ran_point_to_bytes(out.data(), &jac_);
        return out;
    }

    RanPoint RanPoint::scalar_mul(const RanScalar &s) const
    {
        auto sb = s.to_bytes();
        RanPoint r;
        ran_scalarmult(&r.jac_, sb.data(), &jac_);
        ranshaw_secure_erase(sb.data(), 32);
        return r;
    }

    RanPoint RanPoint::scalar_mul_vartime(const RanScalar &s) const
    {
        auto sb = s.to_bytes();
        RanPoint r;
        ran_scalarmult_vartime(&r.jac_, sb.data(), &jac_);
        ranshaw_secure_erase(sb.data(), 32);
        return r;
    }

    RanPoint RanPoint::multi_scalar_mul(const RanScalar *scalars, const RanPoint *points, size_t n)
    {
        if (n == 0 || !scalars || !points || n > SIZE_MAX / 32)
            return RanPoint();

        std::vector<unsigned char> scalar_bytes(32 * n);
        std::vector<ran_jacobian> jac_points(n);

        for (size_t i = 0; i < n; i++)
        {
            auto sb = scalars[i].to_bytes();
            std::memcpy(scalar_bytes.data() + 32 * i, sb.data(), 32);
            ran_copy(&jac_points[i], &points[i].raw());
        }

        RanPoint r;
        ran_msm_vartime(&r.jac_, scalar_bytes.data(), jac_points.data(), n);
        ranshaw_secure_erase(scalar_bytes.data(), scalar_bytes.size());
        return r;
    }

    RanPoint RanPoint::pedersen_commit(
        const RanScalar &blinding,
        const RanPoint &H,
        const RanScalar *values,
        const RanPoint *generators,
        size_t n)
    {
        if (n == 0 || !values || !generators || n > SIZE_MAX / 32)
            return RanPoint();

        auto blind_bytes = blinding.to_bytes();
        std::vector<unsigned char> val_bytes(32 * n);
        std::vector<ran_jacobian> gen_points(n);

        for (size_t i = 0; i < n; i++)
        {
            auto vb = values[i].to_bytes();
            std::memcpy(val_bytes.data() + 32 * i, vb.data(), 32);
            ran_copy(&gen_points[i], &generators[i].raw());
        }

        RanPoint r;
        ran_pedersen_commit(&r.jac_, blind_bytes.data(), &H.raw(), val_bytes.data(), gen_points.data(), n);
        ranshaw_secure_erase(blind_bytes.data(), 32);
        ranshaw_secure_erase(val_bytes.data(), val_bytes.size());
        return r;
    }

    RanPoint RanPoint::map_to_curve(const uint8_t u[32])
    {
        RanPoint r;
        ran_map_to_curve(&r.jac_, u);
        return r;
    }

    RanPoint RanPoint::map_to_curve(const uint8_t u0[32], const uint8_t u1[32])
    {
        RanPoint r;
        ran_map_to_curve2(&r.jac_, u0, u1);
        return r;
    }

    /* ---- ShawPoint ---- */

    std::optional<ShawPoint> ShawPoint::from_bytes(const uint8_t bytes[32])
    {
        ShawPoint p;
        if (shaw_frombytes(&p.jac_, bytes) != 0)
            return std::nullopt;
        return p;
    }

    std::array<uint8_t, 32> ShawPoint::to_bytes() const
    {
        std::array<uint8_t, 32> out;
        shaw_tobytes(out.data(), &jac_);
        return out;
    }

    std::array<uint8_t, 32> ShawPoint::x_coordinate_bytes() const
    {
        std::array<uint8_t, 32> out;
        shaw_point_to_bytes(out.data(), &jac_);
        return out;
    }

    ShawPoint ShawPoint::scalar_mul(const ShawScalar &s) const
    {
        auto sb = s.to_bytes();
        ShawPoint r;
        shaw_scalarmult(&r.jac_, sb.data(), &jac_);
        ranshaw_secure_erase(sb.data(), 32);
        return r;
    }

    ShawPoint ShawPoint::scalar_mul_vartime(const ShawScalar &s) const
    {
        auto sb = s.to_bytes();
        ShawPoint r;
        shaw_scalarmult_vartime(&r.jac_, sb.data(), &jac_);
        ranshaw_secure_erase(sb.data(), 32);
        return r;
    }

    ShawPoint ShawPoint::multi_scalar_mul(const ShawScalar *scalars, const ShawPoint *points, size_t n)
    {
        if (n == 0 || !scalars || !points || n > SIZE_MAX / 32)
            return ShawPoint();

        std::vector<unsigned char> scalar_bytes(32 * n);
        std::vector<shaw_jacobian> jac_points(n);

        for (size_t i = 0; i < n; i++)
        {
            auto sb = scalars[i].to_bytes();
            std::memcpy(scalar_bytes.data() + 32 * i, sb.data(), 32);
            shaw_copy(&jac_points[i], &points[i].raw());
        }

        ShawPoint r;
        shaw_msm_vartime(&r.jac_, scalar_bytes.data(), jac_points.data(), n);
        ranshaw_secure_erase(scalar_bytes.data(), scalar_bytes.size());
        return r;
    }

    ShawPoint ShawPoint::pedersen_commit(
        const ShawScalar &blinding,
        const ShawPoint &H,
        const ShawScalar *values,
        const ShawPoint *generators,
        size_t n)
    {
        if (n == 0 || !values || !generators || n > SIZE_MAX / 32)
            return ShawPoint();

        auto blind_bytes = blinding.to_bytes();
        std::vector<unsigned char> val_bytes(32 * n);
        std::vector<shaw_jacobian> gen_points(n);

        for (size_t i = 0; i < n; i++)
        {
            auto vb = values[i].to_bytes();
            std::memcpy(val_bytes.data() + 32 * i, vb.data(), 32);
            shaw_copy(&gen_points[i], &generators[i].raw());
        }

        ShawPoint r;
        shaw_pedersen_commit(&r.jac_, blind_bytes.data(), &H.raw(), val_bytes.data(), gen_points.data(), n);
        ranshaw_secure_erase(blind_bytes.data(), 32);
        ranshaw_secure_erase(val_bytes.data(), val_bytes.size());
        return r;
    }

    ShawPoint ShawPoint::map_to_curve(const uint8_t u[32])
    {
        ShawPoint r;
        shaw_map_to_curve(&r.jac_, u);
        return r;
    }

    ShawPoint ShawPoint::map_to_curve(const uint8_t u0[32], const uint8_t u1[32])
    {
        ShawPoint r;
        shaw_map_to_curve2(&r.jac_, u0, u1);
        return r;
    }

} // namespace ranshaw
