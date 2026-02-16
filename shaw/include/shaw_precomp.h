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

#ifndef RANSHAW_SHAW_PRECOMP_H
#define RANSHAW_SHAW_PRECOMP_H

/**
 * @file shaw_precomp.h
 * @brief Precomputed fixed-base table for the Shaw base generator.
 *
 * Provides a statically-embedded w=5 affine table for the Shaw base generator G,
 * avoiding runtime precomputation for the most commonly used base point.
 */

#include "fq_frombytes.h"
#include "shaw.h"
#include "shaw_g_table.inl"

/**
 * Load the precomputed Shaw base generator table from static byte data.
 *
 * @param table Output array of 16 affine points [1G, 2G, ..., 16G]
 */
static inline void shaw_load_g_table(shaw_affine table[16])
{
    for (int i = 0; i < 16; i++)
    {
        fq_frombytes(table[i].x, SHAW_G_TABLE_BYTES + i * 64);
        fq_frombytes(table[i].y, SHAW_G_TABLE_BYTES + i * 64 + 32);
    }
}

#endif // RANSHAW_SHAW_PRECOMP_H
