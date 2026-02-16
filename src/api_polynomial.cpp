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

// api_polynomial.cpp — Implementation of FpPolynomial/FqPolynomial C++ API methods.
// Handles byte serialization/deserialization and delegates to the C-style poly routines.

#include "fp_frombytes.h"
#include "fp_ops.h"
#include "fp_tobytes.h"
#include "fp_utils.h"
#include "fq_frombytes.h"
#include "fq_ops.h"
#include "fq_tobytes.h"
#include "fq_utils.h"
#include "ranshaw_polynomial.h"

#include <climits>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ranshaw
{

    /* Upper bound on polynomial size: 1M coefficients (~40MB). Prevents
     * unbounded allocations from causing memory exhaustion. */
    static constexpr size_t MAX_POLY_SIZE = 1u << 20;

    /* ---- helpers ---- */

    static inline void fp_fe_store(fp_fe_storage *dst, const fp_fe src)
    {
        std::memcpy(dst->v, src, sizeof(fp_fe));
    }

    static inline void fp_fe_load(fp_fe dst, const fp_fe_storage *src)
    {
        std::memcpy(dst, src->v, sizeof(fp_fe));
    }

    static inline void fq_fe_store(fq_fe_storage *dst, const fq_fe src)
    {
        std::memcpy(dst->v, src, sizeof(fq_fe));
    }

    static inline void fq_fe_load(fq_fe dst, const fq_fe_storage *src)
    {
        std::memcpy(dst, src->v, sizeof(fq_fe));
    }

    static void fp_poly_add_impl(fp_poly *r, const fp_poly *a, const fp_poly *b)
    {
        size_t na = a->coeffs.size();
        size_t nb = b->coeffs.size();
        size_t nr = (na > nb) ? na : nb;
        r->coeffs.resize(nr);
        for (size_t i = 0; i < nr; i++)
        {
            fp_fe ai_val, bi_val;
            if (i < na)
                fp_fe_load(ai_val, &a->coeffs[i]);
            else
                fp_0(ai_val);
            if (i < nb)
                fp_fe_load(bi_val, &b->coeffs[i]);
            else
                fp_0(bi_val);
            fp_fe sum;
            fp_add(sum, ai_val, bi_val);
            fp_fe_store(&r->coeffs[i], sum);
        }
    }

    static void fp_poly_sub_impl(fp_poly *r, const fp_poly *a, const fp_poly *b)
    {
        size_t na = a->coeffs.size();
        size_t nb = b->coeffs.size();
        size_t nr = (na > nb) ? na : nb;
        r->coeffs.resize(nr);
        for (size_t i = 0; i < nr; i++)
        {
            fp_fe ai_val, bi_val;
            if (i < na)
                fp_fe_load(ai_val, &a->coeffs[i]);
            else
                fp_0(ai_val);
            if (i < nb)
                fp_fe_load(bi_val, &b->coeffs[i]);
            else
                fp_0(bi_val);
            fp_fe diff;
            fp_sub(diff, ai_val, bi_val);
            fp_fe_store(&r->coeffs[i], diff);
        }
    }

    static void fq_poly_add_impl(fq_poly *r, const fq_poly *a, const fq_poly *b)
    {
        size_t na = a->coeffs.size();
        size_t nb = b->coeffs.size();
        size_t nr = (na > nb) ? na : nb;
        r->coeffs.resize(nr);
        for (size_t i = 0; i < nr; i++)
        {
            fq_fe ai_val, bi_val;
            if (i < na)
                fq_fe_load(ai_val, &a->coeffs[i]);
            else
                fq_0(ai_val);
            if (i < nb)
                fq_fe_load(bi_val, &b->coeffs[i]);
            else
                fq_0(bi_val);
            fq_fe sum;
            fq_add(sum, ai_val, bi_val);
            fq_fe_store(&r->coeffs[i], sum);
        }
    }

    static void fq_poly_sub_impl(fq_poly *r, const fq_poly *a, const fq_poly *b)
    {
        size_t na = a->coeffs.size();
        size_t nb = b->coeffs.size();
        size_t nr = (na > nb) ? na : nb;
        r->coeffs.resize(nr);
        for (size_t i = 0; i < nr; i++)
        {
            fq_fe ai_val, bi_val;
            if (i < na)
                fq_fe_load(ai_val, &a->coeffs[i]);
            else
                fq_0(ai_val);
            if (i < nb)
                fq_fe_load(bi_val, &b->coeffs[i]);
            else
                fq_0(bi_val);
            fq_fe diff;
            fq_sub(diff, ai_val, bi_val);
            fq_fe_store(&r->coeffs[i], diff);
        }
    }

    /* Strip trailing zero coefficients */
    static void fp_poly_strip(fp_poly *p)
    {
        while (p->coeffs.size() > 1)
        {
            fp_fe tmp;
            fp_fe_load(tmp, &p->coeffs.back());
            if (fp_isnonzero(tmp))
                break;
            p->coeffs.pop_back();
        }
    }

    static void fq_poly_strip(fq_poly *p)
    {
        while (p->coeffs.size() > 1)
        {
            fq_fe tmp;
            fq_fe_load(tmp, &p->coeffs.back());
            if (fq_isnonzero(tmp))
                break;
            p->coeffs.pop_back();
        }
    }

    /* ---- FpPolynomial ---- */

    size_t FpPolynomial::degree() const
    {
        if (poly_.coeffs.empty())
            return 0;
        return poly_.coeffs.size() - 1;
    }

    FpPolynomial FpPolynomial::from_coefficients(const uint8_t *coeff_bytes, size_t n)
    {
        if (n == 0 || !coeff_bytes || n > MAX_POLY_SIZE || n > SIZE_MAX / 32)
            return FpPolynomial();

        FpPolynomial p;
        p.poly_.coeffs.resize(n);
        for (size_t i = 0; i < n; i++)
            fp_frombytes(p.poly_.coeffs[i].v, coeff_bytes + 32 * i);
        return p;
    }

    FpPolynomial FpPolynomial::from_roots(const uint8_t *root_bytes, size_t n)
    {
        if (n == 0 || !root_bytes || n > MAX_POLY_SIZE || n > SIZE_MAX / 32)
            return FpPolynomial();

        struct fp_fe_s
        {
            fp_fe v;
        };
        std::vector<fp_fe_s> roots(n);
        for (size_t i = 0; i < n; i++)
            fp_frombytes(roots[i].v, root_bytes + 32 * i);

        FpPolynomial p;
        fp_poly_from_roots(&p.poly_, &roots[0].v, n);
        return p;
    }

    std::array<uint8_t, 32> FpPolynomial::evaluate(const uint8_t x[32]) const
    {
        fp_fe xval, result;
        fp_frombytes(xval, x);
        fp_poly_eval(result, &poly_, xval);

        std::array<uint8_t, 32> out;
        fp_tobytes(out.data(), result);
        return out;
    }

    FpPolynomial FpPolynomial::operator*(const FpPolynomial &other) const
    {
        FpPolynomial r;
        fp_poly_mul(&r.poly_, &poly_, &other.poly_);
        return r;
    }

    FpPolynomial FpPolynomial::operator+(const FpPolynomial &other) const
    {
        FpPolynomial r;
        fp_poly_add_impl(&r.poly_, &poly_, &other.poly_);
        fp_poly_strip(&r.poly_);
        return r;
    }

    FpPolynomial FpPolynomial::operator-(const FpPolynomial &other) const
    {
        FpPolynomial r;
        fp_poly_sub_impl(&r.poly_, &poly_, &other.poly_);
        fp_poly_strip(&r.poly_);
        return r;
    }

    std::pair<FpPolynomial, FpPolynomial> FpPolynomial::divmod(const FpPolynomial &divisor) const
    {
        FpPolynomial q, rem;
        fp_poly_divmod(&q.poly_, &rem.poly_, &poly_, &divisor.poly_);
        return {q, rem};
    }

    FpPolynomial FpPolynomial::interpolate(const uint8_t *x_bytes, const uint8_t *y_bytes, size_t n)
    {
        if (n == 0 || !x_bytes || !y_bytes || n > MAX_POLY_SIZE || n > SIZE_MAX / 32)
            return FpPolynomial();

        struct fp_fe_s
        {
            fp_fe v;
        };
        std::vector<fp_fe_s> xs(n);
        std::vector<fp_fe_s> ys(n);
        for (size_t i = 0; i < n; i++)
        {
            fp_frombytes(xs[i].v, x_bytes + 32 * i);
            fp_frombytes(ys[i].v, y_bytes + 32 * i);
        }

        FpPolynomial p;
        fp_poly_interpolate(&p.poly_, &xs[0].v, &ys[0].v, n);
        return p;
    }

    /* ---- FqPolynomial ---- */

    size_t FqPolynomial::degree() const
    {
        if (poly_.coeffs.empty())
            return 0;
        return poly_.coeffs.size() - 1;
    }

    FqPolynomial FqPolynomial::from_coefficients(const uint8_t *coeff_bytes, size_t n)
    {
        if (n == 0 || !coeff_bytes || n > MAX_POLY_SIZE || n > SIZE_MAX / 32)
            return FqPolynomial();

        FqPolynomial p;
        p.poly_.coeffs.resize(n);
        for (size_t i = 0; i < n; i++)
            fq_frombytes(p.poly_.coeffs[i].v, coeff_bytes + 32 * i);
        return p;
    }

    FqPolynomial FqPolynomial::from_roots(const uint8_t *root_bytes, size_t n)
    {
        if (n == 0 || !root_bytes || n > MAX_POLY_SIZE || n > SIZE_MAX / 32)
            return FqPolynomial();

        struct fq_fe_s
        {
            fq_fe v;
        };
        std::vector<fq_fe_s> roots(n);
        for (size_t i = 0; i < n; i++)
            fq_frombytes(roots[i].v, root_bytes + 32 * i);

        FqPolynomial p;
        fq_poly_from_roots(&p.poly_, &roots[0].v, n);
        return p;
    }

    std::array<uint8_t, 32> FqPolynomial::evaluate(const uint8_t x[32]) const
    {
        fq_fe xval, result;
        fq_frombytes(xval, x);
        fq_poly_eval(result, &poly_, xval);

        std::array<uint8_t, 32> out;
        fq_tobytes(out.data(), result);
        return out;
    }

    FqPolynomial FqPolynomial::operator*(const FqPolynomial &other) const
    {
        FqPolynomial r;
        fq_poly_mul(&r.poly_, &poly_, &other.poly_);
        return r;
    }

    FqPolynomial FqPolynomial::operator+(const FqPolynomial &other) const
    {
        FqPolynomial r;
        fq_poly_add_impl(&r.poly_, &poly_, &other.poly_);
        fq_poly_strip(&r.poly_);
        return r;
    }

    FqPolynomial FqPolynomial::operator-(const FqPolynomial &other) const
    {
        FqPolynomial r;
        fq_poly_sub_impl(&r.poly_, &poly_, &other.poly_);
        fq_poly_strip(&r.poly_);
        return r;
    }

    std::pair<FqPolynomial, FqPolynomial> FqPolynomial::divmod(const FqPolynomial &divisor) const
    {
        FqPolynomial q, rem;
        fq_poly_divmod(&q.poly_, &rem.poly_, &poly_, &divisor.poly_);
        return {q, rem};
    }

    FqPolynomial FqPolynomial::interpolate(const uint8_t *x_bytes, const uint8_t *y_bytes, size_t n)
    {
        if (n == 0 || !x_bytes || !y_bytes || n > MAX_POLY_SIZE || n > SIZE_MAX / 32)
            return FqPolynomial();

        struct fq_fe_s
        {
            fq_fe v;
        };
        std::vector<fq_fe_s> xs(n);
        std::vector<fq_fe_s> ys(n);
        for (size_t i = 0; i < n; i++)
        {
            fq_frombytes(xs[i].v, x_bytes + 32 * i);
            fq_frombytes(ys[i].v, y_bytes + 32 * i);
        }

        FqPolynomial p;
        fq_poly_interpolate(&p.poly_, &xs[0].v, &ys[0].v, n);
        return p;
    }

} // namespace ranshaw
