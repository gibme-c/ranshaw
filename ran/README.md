# ran — Ran Curve

Ran is one half of the Ran/Shaw curve cycle — a short Weierstrass curve y² = x³ − 3x + b over the field Fp (p = 2^255 − 19). This module provides everything you need to work with Ran points: creating them, adding them, multiplying by scalars, and encoding them as bytes.

The curve was constructed by tevador for Monero's FCMP++ integration. Its group order is q — the same prime used as Shaw's base field. This is the cycle property: a scalar on Ran is a field element on Shaw, and vice versa. That duality is what enables recursive proof composition.


## Point Representation

Points are stored in Jacobian coordinates (X : Y : Z), where the affine point (x, y) is recovered as x = X/Z², y = Y/Z³. This uses three field elements instead of two, but it means addition and doubling never need a field inversion (which costs ~255 multiplications). You only pay for inversion once, when you convert back to affine at the end.

The curve equation has a = −3, which saves one multiplication per point doubling (3M + 5S instead of 4M + 4S using the [dbl-2001-b](https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html) formula). Since doubling is the most frequent operation in scalar multiplication, this adds up.

```cpp
// Core types
typedef struct { fp_fe X, Y, Z; } ran_jacobian;  // Z=0 means identity
typedef struct { fp_fe x, y; }    ran_affine;
```


## Scalar Multiplication

The core operation — computing s · P for a scalar s and point P.

**Constant-time** (for secret scalars): Uses a signed 4-bit fixed-window method. The scalar is recoded into signed digits, a table of 8 multiples is precomputed, and each digit selects from the table using constant-time conditional moves. No branches or memory access patterns depend on the scalar value.

**Variable-time** (for public data): Uses width-5 wNAF (windowed Non-Adjacent Form), which is faster but leaks timing information about the scalar. Appropriate for signature verification and other public operations.

**Fixed-base**: When you'll multiply the same base point repeatedly (like the generator G), a precomputed table eliminates the per-call table setup. The generator table is stored in `ran_g_table.inl`.


## Multi-Scalar Multiplication (MSM)

Computing s₁·P₁ + s₂·P₂ + ... + sₙ·Pₙ is the performance-critical operation in FCMP++. Two algorithms, selected by n:

| n | Algorithm | Strategy |
|---|-----------|----------|
| ≤ 32 | Straus | Interleaved wNAF — process all scalars bit-by-bit in lockstep |
| > 32 | Pippenger | Bucket method — sort partial sums by scalar digit, accumulate per-bucket |

Both are variable-time (MSM inputs are public in the FCMP++ context). The AVX2 and IFMA backends accelerate MSM with 4-way and 8-way parallel point arithmetic respectively.


## Serialization and Validation

Points are compressed to 32 bytes: the x-coordinate in little-endian, with the y-parity bit stored in bit 255. Deserialization (`ran_frombytes`) recovers y via square root and validates that the point is on the curve. It returns 0 on failure — if the x-coordinate doesn't correspond to a curve point, you get nothing.

This validation matters for twist security. The Ran twist has ~107-bit security, which is strong but not infinite. Rejecting off-curve points at deserialization prevents twist attacks entirely.


## Hash-to-Curve

`ran_map_to_curve` implements the Simplified SWU (Shallue-van de Woestijne-Ulas) map from [RFC 9380](https://www.rfc-editor.org/rfc/rfc9380) §6.6.2. It's a deterministic mapping from a field element to a curve point — not dispatched, always runs the compiled platform path.

The two-input variant `ran_map_to_curve2` maps two field elements independently and adds the results, which is the standard construction for a full hash-to-curve with uniform distribution.


## Pedersen Commitments

`ran_pedersen_commit` computes C = blind · H + Σ valᵢ · genᵢ using MSM internally. This is the commitment scheme used in FCMP++ proof generation.


## Cycle Bridge

`ran_point_to_bytes` extracts the affine x-coordinate of a Ran point as 32 raw bytes. Since Ran coordinates live in Fp and Shaw scalars also live in Fp, these bytes can be directly interpreted as a Shaw scalar. This is the bridge that connects the two curves in the cycle.


## Backends

| Backend | Point arithmetic | MSM acceleration |
|---------|-----------------|------------------|
| Portable | Scalar Fp, `int32_t[10]` | Sequential |
| x64 | Scalar Fp, `uint64_t[5]` with BMI2 asm | Sequential |
| AVX2 | 4-way parallel (`ran_jacobian_4x`, `fp10x4`) | 4 points in SIMD |
| IFMA | 8-way parallel (`fp51x8`, `__m512i[5]`) | 8 points in SIMD |

The dispatch layer selects among backends at runtime for the three hottest operations: `ran_scalarmult`, `ran_scalarmult_vartime`, and `ran_msm_vartime`. All other operations (hash-to-curve, serialization, validation) run the compile-time-selected path.
