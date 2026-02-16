# fq — Field Arithmetic (F_q, q = 2^255 − γ)

The companion field to Fp. Where Fp uses the well-known Ed25519 prime 2^255 − 19, Fq uses a different prime — a Crandall prime where q = 2^255 − γ and γ is approximately 2^127. This is the base field of the Shaw curve and the scalar field of the Ran curve. It exists because the Ran/Shaw curve cycle requires two distinct primes, and this one was chosen to make the cycle work.


## Why a Different Prime

The cycle property requires that each curve's group order equals the other curve's field prime. Ran already uses p = 2^255 − 19 as its base field, so Shaw's base field must be something else — specifically, it must be the group order of Ran. That group order turns out to be a Crandall prime (2^255 minus a "small-ish" constant), which is good news for performance but introduces real complexity compared to Fp.


## What Makes It Harder

With Fp, when a product overflows 2^255, you fold the overflow back by multiplying by 19 — a single small constant. Here, γ is about 127 bits wide, so the fold-back is a full multi-limb multiply. Same idea, significantly more work.

This is the core complexity difference between the two fields. The API is identical — `fq_add`, `fq_sub`, `fq_mul`, `fq_invert`, and so on — but every multiplication and squaring does substantially more reduction work under the hood. See [fp](../fp/README.md) for the shared algorithmic patterns.


## Crandall Reduction

Multiplication uses a 3-stage fold based on the identity 2^255 ≡ γ (mod q):

1. **Schoolbook accumulation**: Standard 5×5 limb multiplication produces 9 result limbs (positions 0–8)
2. **First fold**: Limbs 5–8 (the overflow) are convolved with γ's 3 limbs and folded back into positions 0–4. This is a 4×3 product — not free, but much cheaper than a full modular division
3. **Second fold**: The first fold can itself overflow, so a second 3×3 convolution handles the residual
4. **Final carry chain**: Propagate carries through all 5 limbs and fold any last-limb overflow one more time

The name comes from Crandall & Pomerance's *Prime Numbers* (Springer, 2005), which analyzes arithmetic for primes of the form 2^n − c.


## Subtraction

Subtraction is where the Crandall structure bites hardest. Fp uses a 4p bias (add 4p before subtracting to keep limbs positive). Fq needs an **8q bias** because γ ≈ 2^127 makes the lower limbs of q much smaller than the limb capacity. With a 4q bias, the bias limbs can be smaller than the subtrahend, causing unsigned underflow. The 8q bias ensures every bias limb exceeds even the largest inputs from two chained lazy additions.

This same issue shows up in every representation — the AVX2 radix-2^25.5 backend also uses 8q bias for the same reason.


## Inversion

Fp inversion uses a Fermat addition chain (a^(p−2)) that's been hand-optimized for that specific prime. Fq can't reuse the same chain — different prime, different bit pattern.

On x64, Fq uses **Bernstein-Yang divsteps** instead: a general-purpose constant-time algorithm that runs in exactly 744 iterations regardless of input. It works in a signed 62-bit representation (`fq_signed62`) and uses 2×2 transition matrices to update the GCD state. The fixed iteration count is the constant-time guarantee — there's no early exit that could leak information about the input.

On portable (32-bit), a Fermat chain is used since divsteps requires efficient 64-bit arithmetic.

Reference: Bernstein & Yang, ["Fast constant-time gcd computation and modular inversion"](https://gcd.cr.yp.to/safegcd-20190413.pdf), 2019.


## Representations and Backends

Same structure as Fp — portable 10×25.5-bit, x64 5×51-bit, AVX2 4-way (`fq10x4`), IFMA 8-way (`fq51x8`). The IFMA backend expresses the Crandall fold using `_mm512_madd52lo_epu64` / `_mm512_madd52hi_epu64` intrinsics, which map the 3-limb γ convolution directly onto hardware 52-bit multiply-accumulate.

| Backend | Mul strategy | Notes |
|---------|-------------|-------|
| Portable | Schoolbook 10×10 + Crandall fold | Both gamma folds use radix-2^25.5 offset correction |
| x64 scalar | Schoolbook 5×5 + 3-stage Crandall fold | BMI2 `mulx` assembly for the inner kernel on GCC |
| AVX2 | 4-way parallel (`fq10x4`) | 8q bias subtraction, same fold pattern |
| IFMA | 8-way parallel (`fq51x8`, `__m512i[5]`) | Crandall fold via IFMA multiply-accumulate |
