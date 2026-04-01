#!/usr/bin/env python3
"""
compute_curve_constants.py

Compute all derived constants needed by the ranshaw library from raw curve parameters.
Paste candidate values into the CONFIG section below, then run:

    python tools/compute_curve_constants.py

Outputs all limb decompositions, bias values, exponent nibble sequences, SSWU constants,
FQ_MODULUS_S62, and verification checks needed to patch the repo.
"""

# ==============================================================================
# CONFIG — paste candidate constants here
# ==============================================================================

p = 2**255 - 19

q = 57896044618658097711785492504343953926395325869620403790519050891226336262239

# Ran (curve_a) over Fp: y^2 = x^3 - 3x + b
ran_b = 29548680719914169098707364166668229174524292605831040732610367861908860687214
ran_gx = 31120535987596528787543654099550989366369335255403568395040253315362187572795
ran_gy = 5390110455289897431419207572205632273070988968129752366550933652483657089343

# Shaw (curve_b) over Fq: y^2 = x^3 - 3x + b
shaw_b = 45242492826800543065362849020183379778794895326127509704283577990932728169905
shaw_gx = 26480566284703523680650692398329233407086306779424130090964995851557635845108
shaw_gy = 2031044096027234894369401215051179692991555584293980444498707708508980222155

# ==============================================================================
# HELPERS
# ==============================================================================

def limbs_51(n, count=5):
    mask = (1 << 51) - 1
    return [(n >> (51 * i)) & mask for i in range(count)]

def limbs_25_5(n, count=10):
    widths = [26, 25, 26, 25, 26, 25, 26, 25, 26, 25]
    shift = 0
    out = []
    for i in range(count):
        w = widths[i]
        out.append((n >> shift) & ((1 << w) - 1))
        shift += w
    return out

def limbs_64(n, count=4):
    mask = (1 << 64) - 1
    return [(n >> (64 * i)) & mask for i in range(count)]

def limbs_62_signed(n):
    """Decompose n into 5 signed-62 limbs (as used by divsteps)."""
    w = limbs_64(n, 4)
    M62 = (1 << 62) - 1
    s = []
    s.append(w[0] & M62)
    s.append(((w[0] >> 62) | (w[1] << 2)) & M62)
    s.append(((w[1] >> 60) | (w[2] << 4)) & M62)
    s.append(((w[2] >> 58) | (w[3] << 6)) & M62)
    s.append(w[3] >> 56)
    return s

def le_bytes_32(n):
    return [(n >> (8 * i)) & 0xFF for i in range(32)]

def count_nonzero_limbs(limbs):
    return max((i + 1 for i in range(len(limbs)) if limbs[i] != 0), default=0)

def modinv(a, m):
    return pow(a, m - 2, m)

def nibbles_msb_first(n, count=32):
    return [(n >> (4 * i)) & 0xF for i in range(count - 1, -1, -1)]

def fmt_hex_ull(vals):
    return "{ " + ", ".join(f"0x{v:X}ULL" for v in vals) + " }"

def fmt_hex_ull_padded(vals):
    return "{ " + ", ".join(f"0x{v:016X}ULL" for v in vals) + " }"

def fmt_dec(vals):
    return "{ " + ", ".join(str(v) for v in vals) + " }"

def fmt_bytes(vals):
    return "{ " + ", ".join(f"0x{v:02x}" for v in vals) + " }"

def fmt_s62(vals):
    return "{{ " + ", ".join(f"(int64_t)0x{v:016X}LL" for v in vals) + " }}"

def section(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

# ==============================================================================
# DERIVED VALUES
# ==============================================================================

gamma = 2**255 - q

section("Basic parameters")
print(f"p         = {p}")
print(f"q         = {q}")
print(f"gamma     = {gamma}")
print(f"gamma hex = 0x{gamma:x}")
print(f"gamma bits= {gamma.bit_length()}")
print(f"q mod 8   = {q % 8}")
print(f"q mod 4   = {q % 4}")
print(f"q hex     = 0x{q:064x}")

# ==============================================================================
# LIMBS DEFINES
# ==============================================================================

section("LIMBS defines")

g25 = limbs_25_5(gamma, 10)
g51 = limbs_51(gamma, 5)
tg51 = limbs_51(2 * gamma, 5)
tg64 = limbs_64(2 * gamma, 4)

GAMMA_25_LIMBS = count_nonzero_limbs(g25)
GAMMA_51_LIMBS = count_nonzero_limbs(g51)
TWO_GAMMA_51_LIMBS = count_nonzero_limbs(tg51)
TWO_GAMMA_64_LIMBS = count_nonzero_limbs(tg64)

print(f"#define GAMMA_25_LIMBS {GAMMA_25_LIMBS}")
print(f"#define GAMMA_51_LIMBS {GAMMA_51_LIMBS}")
print(f"#define TWO_GAMMA_51_LIMBS {TWO_GAMMA_51_LIMBS}")
print(f"#define TWO_GAMMA_64_LIMBS {TWO_GAMMA_64_LIMBS}")

# ==============================================================================
# Fq FIELD CONSTANTS
# ==============================================================================

section("fq/include/portable/fq25.h")

print(f"static const int32_t GAMMA_25[10] = {fmt_dec(g25)};")
q25 = limbs_25_5(q, 10)
print(f"static const int32_t Q_25[10] = {fmt_dec(q25)};")

section("fq/include/x64/fq51.h")

q51 = limbs_51(q, 5)
print(f"static const uint64_t GAMMA_51[5] = {fmt_hex_ull(g51)};")
print(f"static const uint64_t TWO_GAMMA_51[5] = {fmt_hex_ull(tg51)};")
print(f"static const uint64_t TWO_GAMMA_64[4] = {fmt_hex_ull_padded(tg64)};")
print(f"static const uint64_t Q_51[5] = {fmt_hex_ull(q51)};")

# Bias: 128 * Q_51[i] per-limb
bias_q51 = [128 * x for x in q51]
print()
print(f"/* 128q bias for subtraction (all limbs >= 2^53, max {max(v.bit_length() for v in bias_q51)} bits) */")
print(f"static const uint64_t BIAS_Q_51[5] = {fmt_hex_ull(bias_q51)};")
print()
for i, v in enumerate(bias_q51):
    status = "OK" if v >= (1 << 53) else "FAIL"
    print(f"  BIAS_Q_51[{i}] = {v.bit_length()} bits [{status}]")

# ==============================================================================
# RAN CONSTANTS
# ==============================================================================

section("ran/include/ran_constants.h")

print("/* 64-bit (radix-2^51) */")
print(f"static const fp_fe RAN_B = {fmt_hex_ull(limbs_51(ran_b))};")
print(f"static const fp_fe RAN_GX = {fmt_hex_ull(limbs_51(ran_gx))};")
print(f"static const fp_fe RAN_GY = {fmt_hex_ull(limbs_51(ran_gy))};")
print()
print("/* 32-bit (radix-2^25.5) */")
print(f"static const fp_fe RAN_B = {fmt_dec(limbs_25_5(ran_b))};")
print(f"static const fp_fe RAN_GX = {fmt_dec(limbs_25_5(ran_gx))};")
print(f"static const fp_fe RAN_GY = {fmt_dec(limbs_25_5(ran_gy))};")
print()
print("/* RAN_ORDER (q as 32 bytes LE) */")
print(f"static const unsigned char RAN_ORDER[32] = {fmt_bytes(le_bytes_32(q))};")

# ==============================================================================
# SHAW CONSTANTS
# ==============================================================================

section("shaw/include/shaw_constants.h")

print("/* 64-bit (radix-2^51) */")
print(f"static const fq_fe SHAW_B = {fmt_hex_ull(limbs_51(shaw_b))};")
print(f"static const fq_fe SHAW_GX = {fmt_hex_ull(limbs_51(shaw_gx))};")
print(f"static const fq_fe SHAW_GY = {fmt_hex_ull(limbs_51(shaw_gy))};")
print()
print("/* 32-bit (radix-2^25.5) */")
print(f"static const fq_fe SHAW_B = {fmt_dec(limbs_25_5(shaw_b))};")
print(f"static const fq_fe SHAW_GX = {fmt_dec(limbs_25_5(shaw_gx))};")
print(f"static const fq_fe SHAW_GY = {fmt_dec(limbs_25_5(shaw_gy))};")
print()
print("/* SHAW_ORDER = p = 2^255 - 19 (unchanged) */")
print(f"static const unsigned char SHAW_ORDER[32] = {fmt_bytes(le_bytes_32(p))};")

# ==============================================================================
# FQ_DIVSTEPS
# ==============================================================================

section("fq/include/x64/fq_divsteps.h — FQ_MODULUS_S62")

w64 = limbs_64(q, 4)
print("q as 4x uint64_t (LE):")
for i, v in enumerate(w64):
    print(f"  w[{i}] = 0x{v:016X}")
print()
s62 = limbs_62_signed(q)
print(f"static const fq_signed62 FQ_MODULUS_S62 = {fmt_s62(s62)};")

# ==============================================================================
# EXPONENT CHAINS
# ==============================================================================

section("Exponent chain: (q+1)/4 for fq_sqrt")

qp1_4 = (q + 1) // 4
print(f"(q+1)/4 = 0x{qp1_4:064x}")
lower128_sqrt = qp1_4 & ((1 << 128) - 1)
print(f"lower 128 = 0x{lower128_sqrt:032x}")
nibs_sqrt = nibbles_msb_first(lower128_sqrt, 32)
print(f"nibbles (MSB first): {','.join(f'{n:x}' for n in nibs_sqrt)}")
zero_count_sqrt = sum(1 for n in nibs_sqrt if n == 0)
nonzero_sqrt = 32 - zero_count_sqrt
print(f"zero nibbles: {zero_count_sqrt}, nonzero: {nonzero_sqrt}")
print(f"total cost: 252 sq + {nonzero_sqrt + 18} mul")
print()
print("/* Addition chain nibble scan (paste into fq_sqrt): */")
for idx, n in enumerate(nibs_sqrt):
    if n == 0:
        print(f"    fq_chain_sqn(acc, acc, 4);")
        print(f"    /* nibble 0: shift only, no multiply */")
    elif n == 1:
        print(f"    fq_chain_sqn(acc, acc, 4);")
        print(f"    fq_chain_mul(acc, acc, z); /* 1 */")
    else:
        print(f"    fq_chain_sqn(acc, acc, 4);")
        print(f"    fq_chain_mul(acc, acc, zt{n}); /* {n:x} = {n} */")

section("Exponent chain: (q-2) for fq_invert (portable Fermat)")

qm2 = q - 2
print(f"q-2 = 0x{qm2:064x}")
lower128_inv = qm2 & ((1 << 128) - 1)
print(f"lower 128 = 0x{lower128_inv:032x}")
nibs_inv = nibbles_msb_first(lower128_inv, 32)
print(f"nibbles (MSB first): {','.join(f'{n:x}' for n in nibs_inv)}")
zero_count_inv = sum(1 for n in nibs_inv if n == 0)
nonzero_inv = 32 - zero_count_inv
print(f"zero nibbles: {zero_count_inv}, nonzero: {nonzero_inv}")
used_nibs = sorted(set(nibs_inv) - {0})
print(f"table entries needed (nonzero): zt{', zt'.join(str(n) for n in used_nibs)}")
print(f"total cost: 254 sq + {nonzero_inv + 18} mul")
print()
print("/* Addition chain nibble scan (paste into fq_invert): */")
for idx, n in enumerate(nibs_inv):
    if n == 0:
        print(f"    fq_chain_sqn(acc, acc, 4);")
        print(f"    /* nibble 0: shift only, no multiply */")
    elif n == 1:
        print(f"    fq_chain_sqn(acc, acc, 4);")
        print(f"    fq_chain_mul(acc, acc, z); /* 1 */")
    else:
        print(f"    fq_chain_sqn(acc, acc, 4);")
        print(f"    fq_chain_mul(acc, acc, zt{n}); /* {n:x} = {n} */")

# ==============================================================================
# SSWU CONSTANTS
# ==============================================================================

def find_sswu_z(field_p, a_coeff, b_coeff):
    """Find SSWU Z: first z in -1, -2, -3, ... that is non-square and g(b/(z*a)) is square."""
    for z_cand in range(1, 1000):
        z_int = -z_cand
        z = z_int % field_p
        if pow(z, (field_p - 1) // 2, field_p) == 1:
            continue  # must be non-square
        x_test = (b_coeff * modinv((z * a_coeff) % field_p, field_p)) % field_p
        gx_test = (pow(x_test, 3, field_p) + a_coeff * x_test + b_coeff) % field_p
        if gx_test == 0 or pow(gx_test, (field_p - 1) // 2, field_p) == 1:
            return z_int
    raise RuntimeError("Could not find SSWU Z")


def emit_sswu_constants(name, field_p, a_coeff, b_coeff):
    z_int = find_sswu_z(field_p, a_coeff, b_coeff)
    z_mod = z_int % field_p
    za = (z_int * a_coeff) % field_p
    neg_b_over_a = (b_coeff * modinv((-a_coeff) % field_p, field_p)) % field_p  # -B/A = B/(-A) = b/3
    b_over_za = (b_coeff * modinv(za, field_p)) % field_p
    a_mod = a_coeff % field_p

    print(f"SSWU_Z = {z_int} mod {'q' if name == 'Shaw' else 'p'}:")
    print(f"  {fmt_hex_ull(limbs_51(z_mod))}")
    print(f"SSWU_NEG_B_OVER_A = -B/A = b/{-a_coeff} mod {'q' if name == 'Shaw' else 'p'}:")
    print(f"  {fmt_hex_ull(limbs_51(neg_b_over_a))}")
    print(f"SSWU_B_OVER_ZA = B/(Z*A) = b/{z_int * a_coeff} mod {'q' if name == 'Shaw' else 'p'} (Z={z_int}, A={a_coeff}):")
    print(f"  {fmt_hex_ull(limbs_51(b_over_za))}")
    print(f"SSWU_A = {a_coeff} mod {'q' if name == 'Shaw' else 'p'}:")
    print(f"  {fmt_hex_ull(limbs_51(a_mod))}")
    print(f"B_LIMBS:")
    print(f"  {fmt_hex_ull(limbs_51(b_coeff))}")

    # Verify
    euler = pow(z_mod, (field_p - 1) // 2, field_p)
    assert euler == field_p - 1, f"Z={z_int} is NOT non-square!"
    x_test = b_over_za
    gx_test = (pow(x_test, 3, field_p) + a_coeff * x_test + b_coeff) % field_p
    assert gx_test == 0 or pow(gx_test, (field_p - 1) // 2, field_p) == 1, f"g(B/(Z*A)) is NOT square!"
    print()
    print(f"Z={z_int} is non-square: True [OK]")
    print(f"g(B/(Z*A)) is square: True [OK]")


section("SSWU hash-to-curve constants (Shaw over Fq)")
emit_sswu_constants("Shaw", q, -3, shaw_b)

# ==============================================================================
# RAN SSWU CONSTANTS (over Fp)
# ==============================================================================

section("Ran SSWU hash-to-curve constants (Ran over Fp)")
emit_sswu_constants("Ran", p, -3, ran_b)

# ==============================================================================
# AVX2 BIAS VALUES
# ==============================================================================

section("AVX2 bias values")

print("/* fq10_avx2.h: 2*Q_25 for scalar sub (signed path, values only need updating) */")
two_q25 = [2 * x for x in q25]
print(f"2*Q_25 = {fmt_dec(two_q25)}")
print()
print("/* fq10x4_avx2.h: 128*Q_25 for 4-way sub (unsigned path, must exceed max subtrahend) */")
bias_q25 = [128 * x for x in q25]
print(f"128*Q_25 = {fmt_dec(bias_q25)}")
# Check 26-bit even limbs: bias must exceed max 27-bit value (134217727)
# Check 25-bit odd limbs: bias must exceed max 26-bit value (67108863)
print()
for i, v in enumerate(bias_q25):
    threshold = (1 << 27) if (i % 2 == 0) else (1 << 26)
    status = "OK" if v >= threshold else "FAIL"
    print(f"  128*Q_25[{i}] = {v} ({v.bit_length()} bits, >= {threshold}: {v >= threshold}) [{status}]")

# ==============================================================================
# SUMMARY
# ==============================================================================

section("Summary")
print(f"gamma bits: {gamma.bit_length()}")
print(f"GAMMA_25_LIMBS: {GAMMA_25_LIMBS}")
print(f"GAMMA_51_LIMBS: {GAMMA_51_LIMBS}")
print(f"TWO_GAMMA_51_LIMBS: {TWO_GAMMA_51_LIMBS}")
print(f"TWO_GAMMA_64_LIMBS: {TWO_GAMMA_64_LIMBS}")
print(f"Bias multiplier: 128q")
print(f"BIAS_Q_51 min bits: {min(v.bit_length() for v in bias_q51)}")
print(f"BIAS_Q_51 max bits: {max(v.bit_length() for v in bias_q51)}")
print(f"All BIAS_Q_51 >= 2^53: {all(v >= (1 << 53) for v in bias_q51)}")
print(f"SSWU Z values auto-detected (see above)")
print(f"sqrt chain zero nibbles: {zero_count_sqrt}")
print(f"invert chain zero nibbles: {zero_count_inv}")
