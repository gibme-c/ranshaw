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
 * @file fq_mul.h
 * @brief F_q multiplication dispatching to the active backend.
 *
 * Uses Crandall reduction (not the 2^255-19 shortcut).
 */

#ifndef RANSHAW_FQ_MUL_H
#define RANSHAW_FQ_MUL_H

#include "fq.h"

#if RANSHAW_PLATFORM_64BIT
void fq_mul_x64(fq_fe h, const fq_fe f, const fq_fe g);
static inline void fq_mul(fq_fe h, const fq_fe f, const fq_fe g)
{
    fq_mul_x64(h, f, g);
}
#else
void fq_mul_portable(fq_fe h, const fq_fe f, const fq_fe g);
static inline void fq_mul(fq_fe h, const fq_fe f, const fq_fe g)
{
    fq_mul_portable(h, f, g);
}
#endif

#endif // RANSHAW_FQ_MUL_H
