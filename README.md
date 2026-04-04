# ranshaw

[![CI](https://github.com/gibme-c/ranshaw/actions/workflows/ci.yml/badge.svg)](https://github.com/gibme-c/ranshaw/actions/workflows/ci.yml)

A standalone, zero-dependency C++17 elliptic curve library implementing the Ran/Shaw curve cycle for [FCMP++](https://github.com/kayabaNerve/fcmp-plus-plus) integration. It replaces the Rust FFI approach with a native C++ implementation.

On 64-bit platforms (x86_64, ARM64), the library automatically uses an optimized radix-2^51 backend with 128-bit multiplication. On x86_64, **SIMD-accelerated backends** -- AVX2 and AVX-512 IFMA -- are selected at runtime via CPUID detection.

### The Curve Cycle

Ran and Shaw are a pair of prime-order short Weierstrass curves that form a **cycle** ([tevador's construction](https://gist.github.com/tevador/4524c2092178df08996487d4e272b096)): Ran operates over the [Ed25519](https://cr.yp.to/ecdh/curve25519-20060209.pdf) base field **F_p** (p = 2^255 - 19), and its group order is the base field of Shaw, and vice versa. This cycle property makes them ideal for recursive proof composition -- a proof verified on one curve produces elements that live natively in the other curve's scalar field, eliminating expensive non-native field arithmetic.

Both curves have the form **y² = x³ - 3x + b** (the a = -3 optimization enables faster point doubling). Both are cofactor 1, so every point on the curve is in the prime-order group.

## Features

### Core Curve Operations

- **Two complete curve implementations** -- Ran (over F_p) and Shaw (over F_q), with independent field arithmetic, point operations, and scalar math for each
- **Field arithmetic** over F_p (2^255 - 19) and F_q (2^255 - γ, a Crandall prime) -- add, subtract, multiply, square, invert, square root, batch invert, and more
- **Jacobian coordinate curve operations** -- point addition, mixed addition, doubling with a = -3 optimization (3M+5S)
- **Constant-time scalar multiplication** -- variable-base signed 4-bit fixed-window, plus fixed-base w=5 with precomputed 16-entry affine tables for known/cached base points
- **Variable-time scalar multiplication** -- wNAF w=5 for verification and public-data operations
- **Multi-scalar multiplication** -- Straus (n ≤ 32) and [Pippenger](https://cr.yp.to/papers/pippenger.pdf) (n > 32) with signed-digit encoding and bucket accumulation, plus a fixed-base MSM specialization that shares 255 doublings across all points
- **Precomputed generator tables** -- static w=5 affine tables for the Ran and Shaw base generators, deserialized from `.inl` data with zero runtime precomputation
- **Scalar arithmetic** -- scalar muladd (`a*x + b mod order`) and scalar squaring for both Ran and Shaw, built on the curve cycle's field operations

### Higher-Level Primitives

- **Hash-to-curve** -- [RFC 9380](https://www.rfc-editor.org/rfc/rfc9380.html) Simplified SWU (SSWU) mapping from field elements to curve points
- **Pedersen commitments** -- `r*H + Σ(s_i * P_i)` computed via MSM
- **Batch affine conversion** -- [Montgomery's trick](https://cr.yp.to/bib/1987/montgomery.pdf) for converting multiple Jacobian points to affine coordinates with a single inversion
- **Batch field inversion** -- standalone [Montgomery's trick](https://cr.yp.to/bib/1987/montgomery.pdf) utilities (`fp_batch_invert`, `fq_batch_invert`) for amortizing inversions across multiple elements
- **Wei25519 bridge** -- converts Ed25519 x-coordinates (as raw 32-byte values) to Shaw scalars

### Polynomial & Divisor System

- **Polynomial arithmetic** -- multiplication (schoolbook, Karatsuba, ECFFT), evaluation, interpolation, division, and construction from roots (see [Polynomials](#polynomials) below)
- **EC-divisor witnesses** -- compute and evaluate divisor polynomials a(x) - y·b(x) for sets of curve points, with SIMD-accelerated evaluation (AVX2 4-way, IFMA 8-way)

### Security & Platform

- **Constant-time discipline** -- no secret-dependent branches or memory access, branchless scalar recode, `ct_barrier` + XOR-blend throughout (see [Constant-Time Discipline](#constant-time-discipline) below)
- **Secure memory erasure** -- `ranshaw_secure_erase` zeros secret data using platform-specific methods the compiler can't optimize away
- **SIMD acceleration** (x86_64) -- runtime-dispatched AVX2 and AVX-512 IFMA backends for scalar multiplication and MSM, with automatic CPU feature detection via CPUID
- **Public C++ API** -- type-safe `RanScalar`, `ShawScalar`, `RanPoint`, `ShawPoint`, `FpPolynomial`, `FqPolynomial`, `RanDivisor`, `ShawDivisor` classes with `std::optional` validation, operator overloads, and RAII
- **Cross-platform** -- MSVC, GCC, Clang, MinGW

## Building

Requires CMake 3.10+ and a C++17 compiler. No external dependencies.

```bash
# Configure (Linux/macOS)
cmake -S . -B build -DBUILD_TESTS=ON

# Configure (Windows -- use Ninja for single-config output)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

# Build
cmake --build build --config Release -j

# Run tests
./build/ranshaw-tests         # Linux/macOS
.\build\ranshaw-tests.exe     # Windows
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | `OFF` | Build unit tests (`ranshaw-tests`, `ranshaw-fuzz-tests`) |
| `BUILD_BENCHMARKS` | `OFF` | Build benchmark executables (`ranshaw-benchmark`, etc.) |
| `BUILD_TOOLS` | `OFF` | Build auxiliary tools (ECFFT precomputation, test vector generator) |
| `FORCE_PORTABLE` | `OFF` | Force the 32-bit portable implementation on 64-bit platforms (for testing/comparison) |
| `ENABLE_AVX2` | `ON`* | Enable AVX2 SIMD backend with runtime dispatch |
| `ENABLE_AVX512` | `ON`* | Enable AVX-512 IFMA SIMD backend with runtime dispatch |
| `ENABLE_ECFFT` | `OFF` | Enable ECFFT polynomial multiplication for large degrees |
| `ENABLE_SPECTRE_MITIGATIONS` | `OFF` | Enable MSVC Spectre mitigations (`/Qspectre`) |
| `ENABLE_LTO` | `OFF` | Enable link-time optimization |
| `ENABLE_ASAN` | `OFF` | Enable AddressSanitizer (GCC/Clang only) |
| `ENABLE_UBSAN` | `OFF` | Enable UndefinedBehaviorSanitizer (GCC/Clang only) |
| `ARCH` | `native` | Target CPU architecture for `-march` (`native`, `default`, or a specific arch) |

\* On x86_64 only. Both default to `OFF` on other architectures or when `FORCE_PORTABLE` is set.

## Usage

Include the public header to access the C++ API:

```cpp
#include "ranshaw.h"

// Initialize runtime dispatch (CPUID heuristic)
ranshaw::init();

// Scalar arithmetic
auto s = ranshaw::RanScalar::one();
auto t = s + s;
auto bytes = t.to_bytes();

// Point operations
auto G = ranshaw::RanPoint::generator();
auto P = G.scalar_mul(s);
auto compressed = P.to_bytes();

// Deserialization with validation
auto pt = ranshaw::RanPoint::from_bytes(compressed.data()); // std::optional
if (pt) { /* valid on-curve point */ }

// Multi-scalar multiplication
std::vector<ranshaw::RanScalar> scalars = {s, t};
std::vector<ranshaw::RanPoint> points = {G, P};
auto msm = ranshaw::RanPoint::multi_scalar_mul(scalars.data(), points.data(), 2);

// Polynomials and divisors
auto div = ranshaw::RanDivisor::compute(points.data(), points.size());
```

The C++ API is the only public interface. All point deserialization goes through `from_bytes()`, which returns `std::optional` and rejects off-curve points. Low-level C-style primitives (`ranshaw_primitives.h`) are internal to the library and not available to downstream consumers.

Link against the `ranshaw` static library target in your CMake project:

```cmake
add_subdirectory(ranshaw)
target_link_libraries(your_target ranshaw)
```

## Architecture

The library is organized as a set of independent modules, each with its own `include/` and `src/` directories:

| Module | Directory | Description |
|--------|-----------|-------------|
| **common** | `common/` | Platform abstraction, CT barriers, secure erase, 128-bit multiply |
| **fp** | `fp/` | F_p (2^255 - 19) field arithmetic -- all backends |
| **fq** | `fq/` | F_q (Crandall prime) field arithmetic -- all backends |
| **ran** | `ran/` | Ran curve operations -- point ops, scalarmult, MSM, batch affine, map-to-curve |
| **shaw** | `shaw/` | Shaw curve operations -- point ops, scalarmult, MSM, batch affine, map-to-curve |
| **ec-divisors** | `ec-divisors/` | Polynomial arithmetic, EC-divisor witnesses, divisor evaluation (with SIMD) |
| **ecfft** | `ecfft/` | ECFFT infrastructure -- precomputed domains, enter/exit/extend/reduce |
| **api** | `src/api_*.cpp` | Public C++ API classes (`ranshaw.h`) -- the only supported interface for downstream consumers |

These modules are layered, matching the math of elliptic curve cryptography:

1. **Field elements (`fp_*`, `fq_*`)** -- Integers modulo p or q. These are the coordinates of curve points. Two independent fields because the two curves operate over different primes.
2. **Curve points (`ran_*`, `shaw_*`)** -- Points on each curve in Jacobian coordinates. Addition, doubling, scalar multiplication, encoding/decoding.
3. **Polynomials (`fp_poly_*`, `fq_poly_*`)** -- Polynomial arithmetic over each field, used by the divisor module and the FCMP++ proof system.
4. **Divisors (`ran_divisor_*`, `shaw_divisor_*`)** -- EC-divisor witness computation and evaluation for the FCMP++ membership proof protocol.

Each module is a separate CMake target. The top-level `ranshaw` target links all modules together.

### Platform Dispatch

The library uses two levels of dispatch:

**Compile-time dispatch** via `ranshaw_platform.h` selects the field element representation:

- **64-bit** (x86_64, ARM64): Field elements are `uint64_t[5]` in radix-2^51. Multiplication uses 128-bit products (`__int128` on GCC/Clang, `_umul128` on MSVC).
- **Everything else**: Falls back to the portable implementation with `int32_t[10]` in alternating 26/25-bit limbs (radix-2^25.5).

The `FORCE_PORTABLE` CMake option forces the 32-bit path on 64-bit platforms for testing.

**Runtime dispatch** (x86_64 only) selects SIMD-accelerated implementations for six hot-path operations:

| Slot | Function |
|------|----------|
| `ran_scalarmult` | Constant-time scalar multiplication (Ran) |
| `ran_scalarmult_vartime` | Variable-time scalar multiplication (Ran) |
| `ran_msm_vartime` | Multi-scalar multiplication (Ran) |
| `shaw_scalarmult` | Constant-time scalar multiplication (Shaw) |
| `shaw_scalarmult_vartime` | Variable-time scalar multiplication (Shaw) |
| `shaw_msm_vartime` | Multi-scalar multiplication (Shaw) |

CPU features (AVX2, AVX-512F, AVX-512 IFMA) are detected via CPUID+XGETBV at startup. Call `ranshaw_init()` for fast heuristic selection, or `ranshaw_autotune()` to benchmark all available implementations and pick the fastest per-function. Both are thread-safe (only the first call executes; subsequent calls are no-ops).

### SIMD Backends (x86_64)

Two SIMD backends accelerate scalar multiplication and MSM:

**AVX2** -- Scalar multiplication uses `int64_t[10]` radix-2^25.5 representation (`fp10`/`fq10`) to avoid 128-bit multiply overhead while keeping values in-register. MSM Straus uses 4-way horizontal parallelism via `fp10x4`/`fq10x4` (10 × `__m256i`), processing 4 independent bucket accumulations per iteration. MSM Pippenger falls back to x64 scalar (irregular access patterns don't benefit from SIMD).

**AVX-512 IFMA** -- Uses `vpmadd52lo`/`vpmadd52hi` for field element multiplication, replacing the multi-instruction schoolbook multiply with hardware 52-bit fused multiply-accumulate. MSM Straus uses 8-way horizontal parallelism via `fp51x8`/`fq51x8` (5 × `__m512i`), processing 8 independent bucket accumulations per iteration. Scalar multiplication uses the same `fp10`/`fq10` path as AVX2 (IFMA's 8-way parallelism doesn't help single-scalar operations).

### Crandall Reduction

The F_q field uses the [Crandall prime](https://link.springer.com/book/10.1007/0-387-28979-8) 2^255 - γ, where γ is 128 bits. Unlike F_p (where reduction by 2^255 - 19 folds back with a single-digit multiply by 19), Crandall reduction requires a wide multiply by the 128-bit value 2γ. This is the fundamental difference from Ed25519-style field arithmetic and the source of most implementation complexity in the Fq backends.

## Polynomials

### What is polynomial degree?

A polynomial is an expression like **3x² + 5x + 1**. The **degree** is the highest power of x that appears -- this example has degree 2. In the context of this library, polynomials are used to represent relationships between curve points. The coefficients are not ordinary numbers but field elements (integers modulo a large prime), and the degree corresponds to the number of points being described.

When FCMP++ constructs a membership proof for a set of transaction outputs, the size of that set determines the degree of the polynomials involved. A proof covering 256 outputs produces polynomials of roughly degree 256. Larger anonymity sets mean higher-degree polynomials, which means more arithmetic -- so polynomial multiplication speed directly affects proof generation time.

### Multiplication strategies

The library automatically selects the best multiplication algorithm based on degree:

| Method | Degree range | Complexity | Description |
|--------|-------------|------------|-------------|
| Schoolbook | < 32 | O(n²) | Direct term-by-term multiplication. Simple and fast at small sizes. |
| Karatsuba | 32 – 1023 | O(n^1.585) | Divide-and-conquer algorithm that trades additions for multiplications. Handles all practical FCMP++ polynomial degrees. |
| ECFFT | ≥ 1024 | O(n²)* | Elliptic Curve Fast Fourier Transform. Uses structured evaluation domains derived from isogeny chains on auxiliary elliptic curves. |

\* The ECFFT's ENTER (evaluation) and EXIT (interpolation) operations are currently O(n²), which means coefficient-space polynomial multiplication via ECFFT cannot beat Karatsuba at any practical size. The ECFFT infrastructure is in place for future protocol-level optimizations that work in the evaluation domain (see below).

### Polynomial API

```cpp
using namespace ranshaw;

auto p = FpPolynomial::from_roots(root_bytes, n);   // (x - r0)(x - r1)...
auto q = FpPolynomial::from_coefficients(coeff_bytes, n);
auto product = p * q;                                // auto-selects schoolbook/Karatsuba/ECFFT
auto sum = p + q;
auto [quot, rem] = p.divmod(q);                      // polynomial division
auto val = p.evaluate(x_bytes);                      // Horner's method
auto interp = FpPolynomial::interpolate(x_bytes, y_bytes, n);  // Lagrange
```

All operations are mirrored for F_q (`FqPolynomial`).

### EC-Divisor Witnesses

EC-divisors represent sets of curve points as polynomial pairs a(x) - y·b(x). The divisor "vanishes" (evaluates to zero) at exactly the points it encodes. This is the core primitive used by FCMP++ to prove set membership without revealing which element is being proven.

```cpp
using namespace ranshaw;

auto div = RanDivisor::compute(points, n);           // compute divisor witness
auto result = div.evaluate(x_bytes, y_bytes);           // evaluate D(x, y)
const FpPolynomial& a = div.a();                        // access a(x) polynomial
const FpPolynomial& b = div.b();                        // access b(x) polynomial
```

Divisor evaluation is SIMD-accelerated: AVX2 uses 4-way parallelism and IFMA uses 8-way parallelism for batch point evaluation across the evaluation domain.

### ECFFT and Evaluation-Domain Operations

The [ECFFT](https://arxiv.org/abs/2107.08473) (Elliptic Curve Fast Fourier Transform) replaces the multiplicative subgroups used in classical NTT/FFT with structured point sets derived from isogeny chains on auxiliary elliptic curves. This matters because the prime fields used by Ran and Shaw lack the large power-of-2 roots of unity that classical FFT requires.

The ECFFT infrastructure provides ENTER (coefficient → evaluation), EXIT (evaluation → coefficient), EXTEND, and REDUCE operations. While ENTER and EXIT are currently O(n²), the real value of the ECFFT lies in evaluation-domain workflows: once polynomials are represented as evaluation vectors, multiplication becomes O(n) pointwise products, and domain size management uses O(n log n) butterfly operations. Unlocking this performance requires protocol-level changes in how FCMP++ constructs and manipulates divisor polynomials -- a library-level optimization alone is not sufficient.

The precomputed ECFFT data (isogeny chain coefficients and domain cosets) is generated by the `ranshaw-gen-ecfft` tool and checked into the repository as `.inl` files. F_p uses a 16-level domain (65,536 evaluation points) and F_q uses a 15-level domain (32,768 evaluation points).

#### ECFFT Auxiliary Curves

| Field | Auxiliary curve | b (hex) | Domain | Levels | Seed |
|-------|----------------|---------|--------|--------|------|
| F_p (2^255 - 19) | y² = x³ + x + 3427 | `0x0d63` | 65,536 | 16 | 1771386560 |
| F_q (2^255 - γ) | y² = x³ - 3x + b | `0x551062156348f4c77cae38d089493516cd556000d142d22eceb09d208fa12c1c` | 32,768 | 15 | 1775007806 |

These auxiliary curves were selected for having group orders with high 2-adic valuation (large power-of-2 factor), which determines the maximum domain size. **Changing the auxiliary curve breaks backwards compatibility** -- the precomputed `.inl` data encodes the isogeny chain for a specific curve, and all downstream protocol proofs depend on it.

## Constant-Time Discipline

All operations on secret data (private scalars, signing keys) are constant-time:

- No branches on secret-dependent values -- branchless conditional select via `ct_barrier` + XOR-blend, branchless scalar recode in all CT scalarmult paths (w=4 and w=5)
- No secret-dependent memory access -- full-table scan with masked selection for lookups
- No variable-time instructions in hot paths
- `ct_barrier` applied in all conditional-move functions, including AVX2 `fp10_cmov`/`fq10_cmov`
- `ranshaw_secure_erase()` on stack locals in constant-time scalar multiplication paths, and defense-in-depth in vartime paths (`wnaf_encode` scalar copies, `cneg` temporaries)
- Verification-only paths may use variable-time operations (explicitly tagged `_vartime` in the API)

The public API also includes defensive input validation (null pointers and n=0 return identity/empty rather than crashing; polynomial `divmod` guards against zero divisor). These are robustness measures, not side-channel mitigations.

### Twist Security

Both Ran and Shaw have ~254 bits of twist security (twist orders factor as 3 × large-prime). **Every externally-received point must pass on-curve validation** via the `frombytes` functions, which return an error for off-curve points. The embedding degrees are ~253 bits (Ran) and ~255 bits (Shaw, = p − 1), making the MOV/Frey-Rück attack completely infeasible.

## Benchmarking

The benchmark tool (`ranshaw-benchmark`) measures all library operations:

- `--init` -- Use CPUID heuristic dispatch (default: x64 baseline)
- `--autotune` -- Use benchmarked best-per-function dispatch

Operations benchmarked include field arithmetic (add, sub, mul, sq, invert, sqrt), point operations (dbl, madd, add), serialization, scalar multiplication (CT and vartime, variable-base and fixed-base), MSM at multiple sizes (n = 1, 8, 32, 64, 256), fixed-base MSM, SSWU hash-to-curve, batch affine conversion, batch field inversion, Pedersen commitments, polynomial multiplication (Karatsuba and ECFFT), divisor computation, and divisor evaluation.

Additional benchmark executables:

- `ranshaw-benchmark-contest` -- comparative benchmarks across backend tiers (portable, x64, AVX2, IFMA)
- `ranshaw-benchmark-fcmpp` -- FCMP++-specific benchmarks (Pedersen, divisor evaluation, polynomial operations)

## Testing

Two test suites verify correctness, both using CTest with no external test framework (no Google Test, no Catch2 -- zero dependencies).

```bash
cmake -S . -B build -DBUILD_TESTS=ON -DENABLE_ECFFT=ON
cmake --build build --config Release -j

# Unit tests
./build/ranshaw-tests

# Fuzz tests (deterministic property-based)
./build/ranshaw-fuzz-tests --quiet
```

### Unit Tests (`ranshaw-tests`)

1,681–1,707 tests (depending on config) across 50+ test groups covering: F_p/F_q arithmetic, square roots, point operations, scalar multiplication (variable-base and fixed-base), MSM (variable-base and fixed-base), precomputed generator tables, SSWU hash-to-curve, batch affine, batch field inversion, Pedersen commitments, scalar muladd/sq, polynomials (schoolbook, Karatsuba, interpolation, ECFFT), divisors, divisor evaluation, serialization, edge cases, Wei25519 bridge, point-to-scalar conversion, dispatch verification, the public C++ API (scalar/point/polynomial/divisor classes), and 1,146 cross-validated test vector checks (C++ API + C primitives).

### Fuzz Tests (`ranshaw-fuzz-tests`)

38,128 tests (no ECFFT) or 38,482 tests (with ECFFT) -- deterministic property-based tests using a seeded xoshiro256** PRNG. Covers 21 test categories:

- **Scalar arithmetic** -- commutativity, associativity, distributivity, identity, inverse, muladd, inversion (~10,000 checks)
- **Point arithmetic** -- commutativity, double consistency, associativity, identity (~2,000 checks)
- **IPA/Bulletproof edge cases** -- zero-scalar mul, identity handling, negation, n=1 MSM (~120 checks)
- **Serialization round-trip** -- points and scalars through to_bytes/from_bytes (~2,000 checks)
- **Cross-curve cycle** -- Ran x-coord → Shaw scalar → Shaw point → Ran scalar chain (~1,000 checks)
- **Scalar mul consistency** -- CT vs vartime agreement, linearity, composition (~1,500 checks)
- **MSM** -- random inputs at sizes 2–64 covering Straus and Pippenger, sparse/zero/degenerate cases (~800 checks)
- **Map-to-curve** -- single and two-element SSWU, determinism (~1,000 checks)
- **Wei25519 bridge** -- valid conversions, rejection of invalid inputs (~500 checks)
- **Pedersen commitments** -- correctness, homomorphism, zero-value/zero-blinding (~800 checks)
- **Batch affine** -- individual vs batch conversion agreement (~400 checks)
- **Polynomial arithmetic** -- eval consistency for mul/add/divmod/from_roots (~1,500 checks)
- **Polynomial protocol sizes** -- Karatsuba-threshold and FCMP++-realistic degrees (~400 checks)
- **Divisors** -- vanishing property, non-member rejection, degree checks (~600 checks)
- **Divisor scalar mul** -- FCMP++ critical path, degree 254 verification (~200 checks)
- **Operator+ regression** -- projective equality edge cases from a past bug (~2,000 checks)
- **Verification equation** -- simplified IPA fold simulation (~500 checks)
- **All-path cross-validation** -- compares all 6 scalar multiplication code paths (CT, vartime wNAF, MSM n=1, Pedersen, fixed-base CT, fixed-base MSM) against each other for both curves with edge, random, small, and high-bit scalars (~2,900 checks)
- **ECFFT polynomial multiplication** (when ECFFT enabled) -- enter/exit round-trip, small/Karatsuba/large poly mul verification (~354 checks)

Options: `--seed N` (override PRNG seed, default `0xDEADBEEFCAFE1234`), `--quiet` (suppress PASS lines).

### Test Vectors

Portable test vectors are provided for downstream consumers implementing the Ran/Shaw curve cycle in other languages. The test vectors cover every public API operation (scalar arithmetic, point operations, MSM, Pedersen commitments, hash-to-curve, polynomials, divisors, Wei25519 bridge, and batch inversion) and are independently cross-validated by a Python script using the ecpy library.

- **JSON**: `test_vectors/ranshaw_test_vectors.json` (~320 vectors, canonical JSON)
- **C++ header**: `include/ranshaw_test_vectors.h` (generated, for `#include` in C++ projects)
- **Python validator**: `tools/validate_test_vectors.py` (independent cross-validation via ecpy)

To regenerate (requires `BUILD_TOOLS=ON`):
```bash
./ranshaw-gen-testvectors > test_vectors/ranshaw_test_vectors.json
python tools/json_to_header.py test_vectors/ranshaw_test_vectors.json include/ranshaw_test_vectors.h
python tools/validate_test_vectors.py test_vectors/ranshaw_test_vectors.json
```

### Full Test Matrix

CI runs 30 jobs across Linux (gcc-11, gcc-12, clang-14, clang-15), macOS (Homebrew clang, AppleClang), and Windows (MSVC, MinGW GCC). Each x86_64 compiler tests all four backend configurations with ECFFT enabled; ARM64 compilers test portable and native. Both `ranshaw-tests` and `ranshaw-fuzz-tests` are run for every configuration under three dispatch modes: default (baseline), `--init` (CPUID heuristic), and `--autotune` (per-slot benchmarking).

| Config | CMake flags | Unit Tests | Fuzz Tests |
|--------|-------------|------------|------------|
| FORCE_PORTABLE | `-DFORCE_PORTABLE=1` | 1,682–1,702 | 38,128–38,482 |
| x64 no SIMD | `-DENABLE_AVX2=OFF -DENABLE_AVX512=OFF` | 1,681–1,701 | 38,128–38,482 |
| x64 + AVX2 | `-DENABLE_AVX512=OFF` | 1,687–1,707 | 38,128–38,482 |
| x64 + AVX2 + IFMA | (default) | 1,687–1,707 | 38,128–38,482 |

Ranges reflect ECFFT off (lower) vs ECFFT on (higher). SIMD configurations include additional dispatch verification tests. ECFFT configurations include additional ECFFT-specific polynomial multiplication tests.

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
