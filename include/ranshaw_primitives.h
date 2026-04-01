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
 * @file ranshaw_primitives.h
 * @brief Low-level C-style primitives for the RanShaw library.
 *
 * RanShaw is an elliptic curve library implementing the Ran/Shaw
 * curve cycle for FCMP++ integration. The two curves form a cycle:
 *
 * - **Ran**: y^2 = x^3 - 3x + b over F_p (p = 2^255 - 19), group order q
 * - **Shaw**: y^2 = x^3 - 3x + b over F_q (q = 2^255 - gamma), group order p
 *
 * The library is organized in layers:
 *
 * - **Field elements (fp_*, fq_*)**: Arithmetic modulo p and q respectively.
 * - **Curve points (ran_*, shaw_*)**: Jacobian coordinate point operations
 *   including addition, doubling, scalar multiplication, and MSM.
 * - **Polynomials (fp_poly_*, fq_poly_*)**: EC-divisor polynomial arithmetic.
 * - **Divisors (ran_divisor_*, shaw_divisor_*)**: EC-divisor witness
 *   computation and evaluation.
 *
 * This header pulls in all low-level C-style primitives. For the idiomatic
 * C++ API with type safety and std::optional validation, use ranshaw.h.
 *
 * @note **This is a low-level cryptographic primitive library.** Callers must:
 *
 * 1. Validate all externally-received points via frombytes (returns error for
 *    off-curve points).
 * 2. Use constant-time scalar multiplication for secret scalars, and _vartime
 *    functions only for public data.
 * 3. Zero sensitive data after use via ranshaw_secure_erase().
 */

#ifndef RANSHAW_PRIMITIVES_H
#define RANSHAW_PRIMITIVES_H

/* Platform detection, CPUID, dispatch, and secure erase */
#include "ranshaw_cpuid.h"
#include "ranshaw_ct_barrier.h"
#include "ranshaw_dispatch.h"
#include "ranshaw_platform.h"
#include "ranshaw_secure_erase.h"

/* F_p field arithmetic (p = 2^255 - 19) */
#include "fp.h"
#include "fp_batch_invert.h"
#include "fp_cmov.h"
#include "fp_frombytes.h"
#include "fp_invert.h"
#include "fp_mul.h"
#include "fp_ops.h"
#include "fp_pow22523.h"
#include "fp_sq.h"
#include "fp_sqrt.h"
#include "fp_tobytes.h"
#include "fp_utils.h"

/* F_q field arithmetic (q = 2^255 - gamma) */
#include "fq.h"
#include "fq_batch_invert.h"
#include "fq_cmov.h"
#include "fq_frombytes.h"
#include "fq_invert.h"
#include "fq_mul.h"
#include "fq_ops.h"
#include "fq_sq.h"
#include "fq_sqrt.h"
#include "fq_tobytes.h"
#include "fq_utils.h"

/* Ran curve operations (over F_p) */
#include "ran.h"
#include "ran_add.h"
#include "ran_batch_affine.h"
#include "ran_constants.h"
#include "ran_dbl.h"
#include "ran_frombytes.h"
#include "ran_madd.h"
#include "ran_map_to_curve.h"
#include "ran_msm_fixed.h"
#include "ran_msm_vartime.h"
#include "ran_ops.h"
#include "ran_pedersen.h"
#include "ran_precomp.h"
#include "ran_scalar.h"
#include "ran_scalarmult.h"
#include "ran_scalarmult_fixed.h"
#include "ran_scalarmult_vartime.h"
#include "ran_to_scalar.h"
#include "ran_tobytes.h"
#include "ran_validate.h"

/* Shaw curve operations (over F_q) */
#include "shaw.h"
#include "shaw_add.h"
#include "shaw_batch_affine.h"
#include "shaw_constants.h"
#include "shaw_dbl.h"
#include "shaw_frombytes.h"
#include "shaw_madd.h"
#include "shaw_map_to_curve.h"
#include "shaw_msm_fixed.h"
#include "shaw_msm_vartime.h"
#include "shaw_ops.h"
#include "shaw_pedersen.h"
#include "shaw_precomp.h"
#include "shaw_scalar.h"
#include "shaw_scalarmult.h"
#include "shaw_scalarmult_fixed.h"
#include "shaw_scalarmult_vartime.h"
#include "shaw_to_scalar.h"
#include "shaw_tobytes.h"
#include "shaw_validate.h"

/* Wei25519 bridge */
#include "ranshaw_wei25519.h"

/* EC-divisor polynomials and divisors */
#include "divisor.h"
#include "divisor_eval.h"
#include "poly.h"

#endif /* RANSHAW_PRIMITIVES_H */
