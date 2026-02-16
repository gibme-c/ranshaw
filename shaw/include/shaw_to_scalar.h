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
 * @file shaw_to_scalar.h
 * @brief Convert a Shaw point's x-coordinate to a Ran scalar (via the cycle property).
 */

#ifndef RANSHAW_SHAW_TO_SCALAR_H
#define RANSHAW_SHAW_TO_SCALAR_H

#include "fq_tobytes.h"
#include "shaw_ops.h"

/*
 * Extract the affine x-coordinate of a Shaw point as 32 canonical LE bytes.
 *
 * This is the core primitive for curve-cycle layer alternation: the output
 * bytes can be used as a Ran scalar (since Shaw base field = Ran
 * scalar field).
 *
 * Identity (Z=0) maps to 32 zero bytes (scalar 0).
 */
static inline void shaw_point_to_bytes(unsigned char out[32], const shaw_jacobian *P)
{
    shaw_affine a;
    shaw_to_affine(&a, P);
    fq_tobytes(out, a.x);
}

#endif // RANSHAW_SHAW_TO_SCALAR_H
