# ecfft — Elliptic Curve FFT

Classical FFT needs roots of unity — special numbers where ω^n = 1 in the field. The fields used by Ran and Shaw don't have large power-of-2 roots of unity, so classical FFT (and NTT) can't be used for fast polynomial multiplication. The ECFFT works around this by replacing roots of unity with structured point sets derived from isogeny chains on auxiliary elliptic curves.

This module is optional — it requires building with `-DENABLE_ECFFT=ON` and is used by [ec-divisors](../ec-divisors/README.md) for polynomial multiplication at degree ≥ 1024.


## The Problem

Polynomial multiplication via FFT requires an evaluation domain with specific algebraic structure: you need to be able to repeatedly split the domain in half, with each half related to the other by a simple map. In fields with 2^k-th roots of unity, squaring provides that map. Our fields don't have large enough roots of unity.


## The Solution

Instead of roots of unity, use x-coordinates from a carefully chosen auxiliary elliptic curve. A 2-to-1 isogeny (a special map between curves) sends pairs of x-coordinates to a single x-coordinate on the next curve — this gives the "split in half" step. A chain of 16 such isogenies takes a domain of 65,536 points down to a single point, one level at a time, mirroring the recursive halving in classical FFT.

At each level, a 2×2 "butterfly" matrix transforms pairs of values, analogous to the twiddle factors in classical FFT. These matrices are precomputed from the isogeny fiber structure and stored as part of the context.

Reference: Ben-Sasson, Carmon, Kopparty, Levit, ["Elliptic Curve Fast Fourier Transform (ECFFT) Part I"](https://arxiv.org/abs/2107.08473), 2021.


## Operations

| Operation | Description |
|-----------|-------------|
| ENTER | Coefficients → evaluations (evaluate polynomial at all domain points) |
| EXIT | Evaluations → coefficients (interpolate from evaluations) |
| EXTEND | Grow evaluation domain (exit to coefficients, zero-pad, re-enter) |
| REDUCE | Shrink evaluation domain (exit to coefficients, re-enter at smaller size) |

`ecfft_fp_poly_mul` / `ecfft_fq_poly_mul` multiplies two polynomials by entering both into the evaluation domain, multiplying pointwise (256 independent field multiplies), and exiting the result back to coefficients.

ENTER and EXIT are currently O(n²) — they evaluate/interpolate via Horner's method and Newton divided differences. The linear-time butterfly recursion that would make ECFFT truly O(n log n) is the target for future optimization. Even at O(n²), the ECFFT path is competitive for large degrees because the evaluation-domain multiply itself is O(n).


## Precomputed Data

The auxiliary curves and isogeny chains are precomputed and stored as `.inl` files included at compile time:

| Field | Auxiliary curve | Domain size | Levels | Data file |
|-------|----------------|-------------|--------|-----------|
| Fp | y² = x³ + x + 3427 (a = 1) | 65,536 | 16 | `ecfft_fp_data.inl` |
| Fq | y² = x³ − 3x + b (a = −3) | 65,536 | 16 | `ecfft_fq_data.inl` |

Each `.inl` file contains the 65,536 coset x-coordinates (as 32-byte field elements) and the rational map coefficients (numerator/denominator polynomials) for each of the 16 isogeny levels.

Both global contexts are initialized together by a single call to `ecfft_global_init()`, which loads the coset data, applies a bit-reversal permutation, and builds the per-level butterfly matrices using batch inversion for both Fp and Fq. Initialization is init-once and thread-safe — an atomic 3-state gate (uninitialized → initializing → ready) ensures exactly one thread does the work while concurrent callers spin until it completes. The ready state is terminal; there is no free or reset. If initialization fails (e.g. allocation failure), the gate resets so the next caller can retry.


## Regeneration

The precomputed data only needs to be regenerated if the auxiliary curve changes (which would break backwards compatibility with all downstream proofs). The process uses the `ranshaw-gen-ecfft` tool — see the project README for the exact commands and auxiliary curve parameters.

The auxiliary curves were found by the `ranshaw-find-ecfft` tool using native 2-descent to identify curves with smooth group orders (power-of-2 domain sizes). A SageMath cross-check is available in `tools/ecfft_params.sage`.
