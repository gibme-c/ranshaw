# ec-divisors — Polynomials & EC-Divisor Witnesses

This module implements polynomial arithmetic over both fields and the EC-divisor witness system used by FCMP++. If scalar multiplication is the engine of the curve library, divisors are the engine of the proof system — they encode sets of curve points as polynomial pairs that can be efficiently verified.


## Polynomials

FCMP++ represents anonymity sets as polynomials. The degree of the polynomial corresponds to the size of the set. Larger anonymity sets mean higher-degree polynomials, which means more arithmetic. So polynomial multiplication speed directly affects proof generation time.

Polynomials are stored as coefficient vectors in ascending degree order:

```cpp
struct fp_poly { std::vector<fp_fe_storage> coeffs; };  // coeffs[i] = coefficient of x^i
struct fq_poly { std::vector<fq_fe_storage> coeffs; };
```

### Multiplication Strategies

Three algorithms, selected by degree:

| Degree range | Algorithm | Complexity | Notes |
|-------------|-----------|------------|-------|
| < 32 | Schoolbook | O(n²) | Simple, low overhead |
| 32 – 1023 | Karatsuba | O(n^1.585) | Recursive divide-and-conquer |
| ≥ 1024 | ECFFT | ~O(n²) pointwise, O(n) multiply | Requires `ENABLE_ECFFT`, see [ecfft](../ecfft/README.md) |

The ECFFT path transforms polynomials into an evaluation domain where multiplication is pointwise, then transforms back. The transforms are currently O(n²), but the infrastructure enables future evaluation-domain workflows.

Other polynomial operations — `eval` (Horner), `from_roots`, `divmod`, `interpolate` (Lagrange) — use standard algorithms with careful normalization to handle the lazy arithmetic from [fp](../fp/README.md) and [fq](../fq/README.md).


## Divisors

A divisor is a pair of polynomials D(x, y) = a(x) − y · b(x) that "vanishes" at exactly a specified set of curve points. If P = (x₀, y₀) is in the set, then a(x₀) − y₀ · b(x₀) = 0.

```cpp
struct ran_divisor { fp_poly a; fp_poly b; };
struct shaw_divisor { fq_poly a; fq_poly b; };
```

This is how FCMP++ proves "I know a point in this set" without revealing which one. The divisor witness encodes the relationship between the committed point and the anonymity set in a form that the verifier can check algebraically.

`ran_compute_divisor` / `shaw_compute_divisor` build the divisor for a given set of affine points. `ran_evaluate_divisor` / `shaw_evaluate_divisor` evaluate D(x, y) at a specific point.


## Evaluation Domain

For FCMP++ verification, divisors need to be combined (multiplied, merged) many times. Doing this in coefficient form would require repeated polynomial multiplications. Instead, the eval-domain types represent divisors as their values at 256 fixed points:

```cpp
struct fp_evals {
    alignas(64) fp_evals_limb_t limbs[FP_EVALS_NLIMBS][EVAL_DOMAIN_SIZE];
    size_t degree;
};

struct ran_eval_divisor { fp_evals a; fp_evals b; };
```

In this representation, multiplication is just 256 independent field multiplies — perfect for SIMD. The AVX2 and IFMA backends provide accelerated versions of `fp_evals_mul` and `fq_evals_mul`, selected once at init time via CPUID (best available: IFMA > AVX2 > scalar). This is separate from the main 6-slot runtime dispatch and isn't affected by `--init` or `--autotune` — it's purely a compile-time/CPUID decision.

Key eval-domain operations:

- **Tree reduction**: `ran_eval_divisor_tree_reduce` combines n single-point divisors into one using divide-and-conquer — O(n log n) divisor multiplications instead of O(n²)
- **Scalar-mul divisor**: `ran_scalar_mul_divisor` constructs the divisor witness for a scalar multiplication, used for Pedersen commitment opening proofs
- **Finalization**: `ran_eval_divisor_to_divisor` converts from evaluation domain back to coefficient polynomials for the final proof output
