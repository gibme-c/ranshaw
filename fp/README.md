# fp — Field Arithmetic (F_p, p = 2^255 − 19)

All elliptic curve math ultimately reduces to arithmetic on big numbers modulo a prime. This module implements that arithmetic for the prime p = 2^255 − 19 — the same field used by Ed25519 and Curve25519, and the base field of the Ran curve.


## What's a Field Element

A field element is a number between 0 and p − 1. You can add, subtract, multiply, and divide them (where "division" means multiplying by the modular inverse). These are the numbers that curve point coordinates are made of — every Ran point is a pair of field elements satisfying the curve equation.

The type is `fp_fe`, and it looks different depending on your platform (more on that below), but the API is the same everywhere.


## Why 2^255 − 19 Is a Good Prime

When you multiply two 255-bit numbers, the result can be up to 510 bits. You need to reduce it back to 255 bits, which normally means an expensive division. But 2^255 − 19 has a trick: since 2^255 ≡ 19 (mod p), any overflow past 255 bits folds back in by multiplying by 19. That's a single small multiply instead of a full modular reduction.

This is why Ed25519 and Curve25519 chose this prime, and it's why Fp arithmetic is fast. See Bernstein's [Curve25519 paper](https://cr.yp.to/ecdh/curve25519-20060209.pdf) for the original construction.


## Representations

The same field element gets stored differently depending on what your CPU can do efficiently:

| Backend | Layout | Limb type | Used when |
|---------|--------|-----------|-----------|
| Portable | 10 limbs, radix-2^25.5 (alternating 26/25 bits) | `int32_t[10]` | 32-bit platforms, or `FORCE_PORTABLE` |
| x64 | 5 limbs, radix-2^51 | `uint64_t[5]` | Any 64-bit x86 platform |

The 5×51 layout is faster because each limb-multiply fits in a 64×64→128-bit hardware multiply, and you only have 5 of them instead of 10. The portable layout splits things into smaller pieces that fit in 32-bit arithmetic.

Both layouts use *lazy* addition — `fp_add` just adds corresponding limbs without carrying, letting them temporarily exceed their nominal width. Carries propagate during multiplication, squaring, or explicit normalization. This saves work when additions chain together (which happens constantly in point arithmetic).


## Operations

**Basic arithmetic**: `fp_add`, `fp_sub`, `fp_neg`, `fp_mul`, `fp_sq` — the building blocks. Subtraction uses a bias trick (adding 4p before subtracting) to keep limbs positive without branching.

**Serialization**: `fp_tobytes` and `fp_frombytes` convert between the internal representation and a canonical 32-byte little-endian encoding. Bit 255 is stripped on input (it's used for y-parity in point serialization).

**Inversion**: `fp_invert` computes a^(p−2) using an optimized addition chain specific to this prime. One inversion costs roughly the same as ~255 multiplications.

**Batch inversion**: `fp_batch_invert` inverts n elements using only one actual inversion plus 3(n−1) multiplications, via Montgomery's trick. This is critical for batch Jacobian-to-affine conversion.

**Square root**: `fp_sqrt` exploits the fact that p ≡ 5 (mod 8) for an efficient extraction. Returns success/failure since not every field element has a square root.

**Constant-time utilities**: `fp_cmov` (conditional move), `fp_cneg` (conditional negate), `fp_isnonzero`, `fp_isnegative` — all branchless, using the barriers from [common](../common/README.md).


## Backends

| Backend | Mul strategy | Notes |
|---------|-------------|-------|
| Portable | Schoolbook 10×10 in `int64_t` | Works everywhere |
| x64 scalar | Schoolbook 5×5 in `__int128` or BMI2 `mulx`/`adcx`/`adox` asm | GCC uses dedicated inline assembly for the hot kernel |
| AVX2 | 4-way parallel (`fp10x4`, `__m256i[10]`) | Used by Ran MSM and divisor evaluation |

The AVX2 backend operates on a 4-wide SIMD type (`fp10x4`) in the 10-limb radix-2^25.5 layout, with pack/unpack conversions at the boundary. It doesn't replace the scalar backend — it accelerates specific multi-point operations where you can keep 4 independent computations in flight.
