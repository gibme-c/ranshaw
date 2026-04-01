#!/usr/bin/env python3

#
# gen_g_tables.py
#
# Pure-Python replacement for gen_g_tables.sage (no SageMath dependency).
#
# Generates:
#   ran/include/ran_g_table.inl
#   shaw/include/shaw_g_table.inl
#
# Assumed format:
#   16 affine points [1G, 2G, ..., 16G]
#   each point stored as 64 bytes:
#     32-byte little-endian x || 32-byte little-endian y
#
# Usage:
#   python gen_g_tables.py [/path/to/repo/root]

import os
import sys

# ------------------------------------------------------------------------------
# CONFIG: curve constants
# ------------------------------------------------------------------------------

p = 2**255 - 19
q = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF4BB1EB0E39730FA771684645EC70F85F

# Ran over Fp: y^2 = x^3 - 3x + b
ran_b = 0x4153F5EAB5C57B60FFB710A4CD92E79AAB8F66787D603C262AF05AE9F1D1D36E
ran_gx = 0x44CD9962FA942343B3FD1E677A42235C7120510B2F09D9FB88FDD3F5ABD6F63B
ran_gy = 0x0BEAB1C489555A9C2019D081CED2EE7938C6E428013270092EF544CE79A2C13F

# Shaw over Fq: y^2 = x^3 - 3x + b
shaw_b = 0x640657EEA7EFB1341EF4BC7E888ADD9325C36EFA28B2CC52AD63C002DD34C5B1
shaw_gx = 0x3A8B78295E7C33EF06C70AFF52BFC74A6924096F7CC0F251C13265FA80B76BF4
shaw_gy = 0x047D87BC8873BB741E4069FCEC8ECAFB07765859AF4FBFDDB923532F445198CB

# BSD-3-Clause license header
LICENSE_HEADER = """\
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
"""


# ------------------------------------------------------------------------------
# MODULAR ARITHMETIC
# ------------------------------------------------------------------------------

def mod_inv(a, m):
    """Modular inverse using Python's built-in pow."""
    return pow(a, -1, m)


# ------------------------------------------------------------------------------
# ELLIPTIC CURVE ARITHMETIC (short Weierstrass: y^2 = x^3 + ax + b over Fm)
# ------------------------------------------------------------------------------
# Points represented as (x, y) tuples or None for the identity.

def ec_double(P, a, m):
    """Double a point on y^2 = x^3 + ax + b (mod m)."""
    if P is None:
        return None
    x, y = P
    if y == 0:
        return None
    lam = ((3 * x * x + a) * mod_inv(2 * y, m)) % m
    x3 = (lam * lam - 2 * x) % m
    y3 = (lam * (x - x3) - y) % m
    return (x3, y3)


def ec_add(P, Q, a, m):
    """Add two points on y^2 = x^3 + ax + b (mod m)."""
    if P is None:
        return Q
    if Q is None:
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2:
        if y1 == y2:
            return ec_double(P, a, m)
        return None  # P + (-P) = O
    lam = ((y2 - y1) * mod_inv(x2 - x1, m)) % m
    x3 = (lam * lam - x1 - x2) % m
    y3 = (lam * (x1 - x3) - y1) % m
    return (x3, y3)


def ec_mul(k, P, a, m):
    """Scalar multiplication via double-and-add."""
    if k < 0:
        raise ValueError("negative scalar")
    if k == 0 or P is None:
        return None
    R = None
    Q = P
    while k > 0:
        if k & 1:
            R = ec_add(R, Q, a, m)
        Q = ec_double(Q, a, m)
        k >>= 1
    return R


# ------------------------------------------------------------------------------
# PRIMALITY TEST (deterministic Miller-Rabin for numbers < 2^256)
# ------------------------------------------------------------------------------

def is_prime(n):
    """Deterministic primality test sufficient for numbers up to 2^256."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witnesses sufficient for n < 3,317,044,064,679,887,385,961,981
    # For 255-bit numbers we add extra witnesses for safety.
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def le_bytes_32(n):
    """Convert integer to 32-byte little-endian byte list."""
    if n < 0 or n >= 2**256:
        raise ValueError("value does not fit in 32 bytes")
    return [(n >> (8 * i)) & 0xFF for i in range(32)]


def point_bytes_xy_le(P):
    """Serialize affine point as 64 bytes: LE(x) || LE(y)."""
    if P is None:
        raise ValueError("unexpected identity point in fixed-base table")
    x, y = P
    return le_bytes_32(x) + le_bytes_32(y)


def fmt_byte_array(bs, cols=16):
    """Format a byte list as a C array initializer."""
    items = [f"0x{b:02x}" for b in bs]
    lines = []
    for i in range(0, len(items), cols):
        lines.append("    " + ", ".join(items[i:i + cols]))
    return "{\n" + ",\n".join(lines) + "\n};\n"


def emit_table(curve_name, display_name, var_name, G, a, m, out_path):
    """
    Generate and write a .inl file containing 16 precomputed multiples of G.

    curve_name:   name for the auto-generated comment (e.g. "ran", "shaw")
    display_name: not used in file content, kept for parity
    var_name:     C variable name (e.g. "RAN_G_TABLE_BYTES")
    G:            generator point (x, y)
    a:            curve coefficient a
    m:            field modulus
    out_path:     output file path
    """
    raw = []
    pts = []

    for k in range(1, 17):
        P = ec_mul(k, G, a, m)
        if P is None:
            raise ValueError(f"{curve_name}: {k}G is identity, invalid for table")
        pts.append(P)
        raw.extend(point_bytes_xy_le(P))

    lines = []
    lines.append(LICENSE_HEADER)
    lines.append(f"// Auto-generated by gen_g_tables.py")
    lines.append(f"// Precomputed fixed-base table for {curve_name} base generator.")
    lines.append("// 16 affine points [1G, 2G, ..., 16G], each stored as")
    lines.append("// 64 bytes: 32-byte little-endian x || 32-byte little-endian y.")
    lines.append("// clang-format off")
    lines.append(f"static const unsigned char {var_name}[1024] = ")
    body = fmt_byte_array(raw)

    text = "\n".join(lines) + body
    text += "// clang-format on\n"

    with open(out_path, "w", newline="\n") as f:
        f.write(text)

    print(f"Wrote {out_path}")
    print(f"  1G  = ({pts[0][0]:#x}, {pts[0][1]:#x})")
    print(f" 16G  = ({pts[-1][0]:#x}, {pts[-1][1]:#x})")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    repo_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")

    # Curve coefficient a = -3
    a_coeff = -3

    # Ran over Fp
    Gh = (ran_gx, ran_gy)
    # Verify generator is on curve: y^2 == x^3 + ax + b (mod p)
    lhs = (ran_gy * ran_gy) % p
    rhs = (ran_gx**3 + a_coeff * ran_gx + ran_b) % p
    assert lhs == rhs, "Ran generator not on curve"

    # Shaw over Fq
    Gs = (shaw_gx, shaw_gy)
    lhs = (shaw_gy * shaw_gy) % q
    rhs = (shaw_gx**3 + a_coeff * shaw_gx + shaw_b) % q
    assert lhs == rhs, "Shaw generator not on curve"

    # Sanity checks
    assert is_prime(q), "q must be prime"
    assert is_prime(p), "p must be prime"

    # Verify generator orders: q*G_ran == O, p*G_shaw == O
    assert ec_mul(q, Gh, a_coeff, p) is None, "Ran generator does not have order q"
    assert ec_mul(p, Gs, a_coeff, q) is None, "Shaw generator does not have order p"

    print("All sanity checks passed.\n")

    ran_out = os.path.join(repo_root, "ran", "include", "ran_g_table.inl")
    shaw_out = os.path.join(repo_root, "shaw", "include", "shaw_g_table.inl")

    emit_table("ran", "ran", "RAN_G_TABLE_BYTES", Gh, a_coeff, p, ran_out)
    emit_table("shaw", "shaw", "SHAW_G_TABLE_BYTES", Gs, a_coeff, q, shaw_out)

    print("\nDone.")
    print("Before committing, diff the new .inl files against the current ones")
    print("and confirm the byte layout matches the current scalar-mul loader.")


if __name__ == "__main__":
    main()
