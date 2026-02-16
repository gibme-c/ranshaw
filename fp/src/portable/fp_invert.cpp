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

#include "portable/fp_invert.h"

#include "fp_mul.h"
#include "fp_sq.h"
#include "ranshaw_secure_erase.h"

void fp_invert_portable(fp_fe out, const fp_fe z)
{
    fp_fe t0, t1, t2, t3;
    int i;

    fp_sq(t0, z);
    fp_sq(t1, t0);
    fp_sq(t1, t1);
    fp_mul(t1, z, t1);
    fp_mul(t0, t0, t1);
    fp_sq(t2, t0);
    fp_mul(t1, t1, t2);
    fp_sq(t2, t1);
    for (i = 0; i < 4; ++i)
        fp_sq(t2, t2);
    fp_mul(t1, t2, t1);
    fp_sq(t2, t1);
    for (i = 0; i < 9; ++i)
        fp_sq(t2, t2);
    fp_mul(t2, t2, t1);
    fp_sq(t3, t2);
    for (i = 0; i < 19; ++i)
        fp_sq(t3, t3);
    fp_mul(t2, t3, t2);
    fp_sq(t2, t2);
    for (i = 0; i < 9; ++i)
        fp_sq(t2, t2);
    fp_mul(t1, t2, t1);
    fp_sq(t2, t1);
    for (i = 0; i < 49; ++i)
        fp_sq(t2, t2);
    fp_mul(t2, t2, t1);
    fp_sq(t3, t2);
    for (i = 0; i < 99; ++i)
        fp_sq(t3, t3);
    fp_mul(t2, t3, t2);
    fp_sq(t2, t2);
    for (i = 0; i < 49; ++i)
        fp_sq(t2, t2);
    fp_mul(t1, t2, t1);
    fp_sq(t1, t1);
    for (i = 0; i < 4; ++i)
        fp_sq(t1, t1);
    fp_mul(out, t1, t0);

    ranshaw_secure_erase(t0, sizeof(fp_fe));
    ranshaw_secure_erase(t1, sizeof(fp_fe));
    ranshaw_secure_erase(t2, sizeof(fp_fe));
    ranshaw_secure_erase(t3, sizeof(fp_fe));
}
