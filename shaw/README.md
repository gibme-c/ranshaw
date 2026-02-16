# shaw — Shaw Curve

Shaw is the other half of the curve cycle — same curve shape (y² = x³ − 3x + b), same algorithms, same API, but operating over Fq instead of Fp. If you understand [Ran](../ran/README.md), you understand Shaw.

```cpp
typedef struct { fq_fe X, Y, Z; } shaw_jacobian;
typedef struct { fq_fe x, y; }    shaw_affine;
```


## The Key Difference

Shaw's base field is Fq (the Crandall prime q = 2^255 − γ), and its scalar field is Fp (p = 2^255 − 19). This is the mirror of Ran, and it's what closes the cycle: Ran group order = q = Shaw base field, and Shaw group order = p = Ran base field.

The practical consequence is that Shaw point arithmetic is more expensive than Ran. Every field multiply involves the 3-stage Crandall reduction (see [fq](../fq/README.md)) instead of the simple ×19 fold that Fp enjoys. The algorithmic structure is identical — same Jacobian formulas, same window sizes, same MSM strategies — but each underlying field operation does more work.

The generator point has affine x = 1, and the curve constant b is a different value than Ran's b (necessarily, since it's over a different field).


## Scalar Operations

Shaw scalars live in Fp (not Fq). This means `shaw_scalar_add`, `shaw_scalar_mul`, and friends are thin wrappers around `fp_*` operations. The reversal is the cycle property in action.

`shaw_point_to_bytes` extracts the affine x-coordinate — which is an Fq element — as 32 bytes. Since Fq is the Ran scalar field, these bytes can be interpreted as a Ran scalar. This is Shaw's half of the cycle bridge.


## Everything Else

Serialization, validation, hash-to-curve, Pedersen commitments, batch affine conversion, MSM (Straus/Pippenger) — all structurally identical to Ran with `shaw_` prefixes and `fq_fe` coordinates. The same backend tiers exist (portable, x64, AVX2, IFMA), and the same 3 operations are runtime-dispatched (`shaw_scalarmult`, `shaw_scalarmult_vartime`, `shaw_msm_vartime`).

The twist security is ~99 bits (vs ~107 for Ran), so on-curve validation at deserialization is equally important here.
