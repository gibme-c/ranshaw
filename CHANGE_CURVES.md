# Changing the Ran/Shaw Curve Pair

Complete procedure for replacing the Ran/Shaw curve pair or the ECFFT auxiliary curve.

## What You Need Before Starting

A full curve swap requires these raw inputs:
- **New q** (the Fq prime, group order of Ran / base field of Shaw)
- **gamma** = `2^255 - q` (the Crandall reduction constant)
- **ran_b** (the b coefficient for y^2 = x^3 - 3x + b over Fp)
- **ran_gx, ran_gy** (generator point for Ran, must be on the curve with cofactor 1)
- **shaw_b** (the b coefficient for y^2 = x^3 - 3x + b over Fq)
- **shaw_gx, shaw_gy** (generator point for Shaw)

p = 2^255 - 19 is fixed (ed25519 base field). Both curves use a = -3.

## Tools

| Tool | What it does |
|------|-------------|
| `tools/compute_curve_constants.py` | Computes ALL derived constants from the raw inputs above |
| `tools/gen_g_tables.py` | Generates 16-point precomputed generator tables |
| `tools/find_ecfft_curve.cpp` | Searches for ECFFT-friendly auxiliary curves (hours) |
| `tools/gen_ecfft_data.cpp` | Generates ECFFT precomputed `.inl` data from a found curve |
| `tools/gen_test_vectors.cpp` | Generates canonical test vectors (must build with FORCE_PORTABLE) |
| `tools/json_to_header.py` | Converts JSON test vectors to C++ header |
| `tools/validate_test_vectors.py` | Cross-validates test vectors independently (Python + ecpy) |

## Phase 1: Compute All Derived Constants

Edit the CONFIG section at the top of `tools/compute_curve_constants.py` with the new raw values (q, ran_b, ran_gx, ran_gy, shaw_b, shaw_gx, shaw_gy). Then run:

```bash
python tools/compute_curve_constants.py > curve_constants_output.txt
```

Save the output — you will copy-paste from it in every subsequent phase. The script:
- Auto-detects the SSWU Z parameter for both curves (never hardcode Z)
- Computes the bias multiplier and verifies all bias limbs >= 2^53
- Generates paste-ready addition chain code for sqrt and invert
- Tells you which `zt` precompute variables are needed for the invert chain
- Runs verification checks (prints `[OK]` or `[FAIL]` for each)

**If any check prints `[FAIL]`, stop.** The curve parameters are invalid.

The output is organized into labeled sections. Each section header tells you which file to patch.

---

## Phase 2: Patch Field Constant Headers

Copy limb arrays from the Phase 1 output into these files. The output labels match the constant names in the source.

### `fq/include/x64/fq51.h`

| Constant | What to look for in output |
|----------|---------------------------|
| `GAMMA_51[5]` | Section "Fq gamma/q constants (radix-2^51)" |
| `TWO_GAMMA_51[5]` | Same section |
| `TWO_GAMMA_64[4]` | Same section |
| `Q_51[5]` | Same section |
| `BIAS_Q_51[5]` | Same section — labeled "128q bias for subtraction" |
| `GAMMA_51_LIMBS` | Section "LIMBS defines" |
| `TWO_GAMMA_51_LIMBS` | Same |
| `TWO_GAMMA_64_LIMBS` | Same |

If `TWO_GAMMA_64_LIMBS` changes, verify `fq/include/x64/fq51_inline.h` has a `#if` branch for the new count. Missing branches produce a `#error` at compile time.

### `fq/include/portable/fq25.h`

| Constant | What to look for |
|----------|-----------------|
| `GAMMA_25[10]` | Section "Fq gamma/q constants (radix-2^25.5)" |
| `Q_25[10]` | Same section |
| `GAMMA_25_LIMBS` | Section "LIMBS defines" |

### `ran/include/ran_constants.h`

| Constant | Representation |
|----------|---------------|
| `RAN_B` | Both 64-bit (`uint64_t[5]`) and 32-bit (`int32_t[10]`) — two separate arrays |
| `RAN_GX` | Both representations |
| `RAN_GY` | Both representations |
| `RAN_ORDER[32]` | Little-endian byte array |

### `shaw/include/shaw_constants.h`

Same as ran_constants.h but for Shaw. **SHAW_ORDER is p = 2^255-19 and does NOT change.**

### `fq/include/x64/fq_divsteps.h`

| Constant | What to look for |
|----------|-----------------|
| `FQ_MODULUS_S62` | Section "FQ_MODULUS_S62 (signed-62 decomposition)" |

The `FQ_NEG_QINV62` constant is derived via constexpr from `FQ_MODULUS_S62` — no manual change needed.

---

## Phase 3: Bias Multiplier

The script output includes a section "128q bias for subtraction" with verification:
```
BIAS_Q_51[0] = 54 bits [OK]
...
All BIAS_Q_51 >= 2^53: True
```

If all checks pass and the bias multiplier is already 128q (current default), **no code changes needed** — just update the constant values (already done in Phase 2).

**If the bias is insufficient** (any limb < 2^53), you must increase the multiplier. This requires changing:

| File | What to change |
|------|---------------|
| `fq/include/fq_ops.h` | `BIAS_Q_51[i]` references in `fq_sub` — update comment |
| `fq/include/x64/avx2/fq10x4_avx2.h` | Change `128LL * Q_25[i]` to `NNN * Q_25[i]` in all 7 bias functions |
| `fq/include/x64/avx2/fq10_avx2.h` | Update 10 hardcoded `2*Q_25[i]` literal values (the script outputs these) |
| `shaw/include/x64/ifma/fq51x8_ifma.h` | `BIAS_Q_51[i]` references in IFMA `fq_sub` — update comment |

Then re-run `compute_curve_constants.py` with the new multiplier (edit the `bias_mult = 128` line near the bias section).

---

## Phase 4: Exponent Addition Chains

The script outputs paste-ready code for both chains.

### `(q+1)/4` for sqrt — files: `fq/src/x64/fq_sqrt.cpp` and `fq/src/portable/fq_sqrt.cpp`

1. Update the comment with the new `(q+1)/4` hex value
2. Replace the nibble scan section (the `fq_chain_sqn` / `fq_chain_mul` block) with the code from the script output section "Addition chain nibble scan (paste into fq_sqrt)"
3. Update the total operation count comment (the script prints "Total: N sq + M mul")

The upper 125 bits of `(q+1)/4` are all 1s for any q close to 2^255 — the `z^(2^125-1)` chain at the top of the function stays identical. Only the lower 128-bit nibble scan changes.

### `(q-2)` for Fermat invert — file: `fq/src/portable/fq_invert.cpp`

1. Update the comment with the new `(q-2)` hex value
2. Replace the nibble scan section with the code from "Addition chain nibble scan (paste into fq_invert)"
3. **Check the precompute table.** The script output lists which nibble values appear. Each nibble `n` requires a precomputed `zt{2n}` variable. If a new nibble appears that wasn't used before (e.g., nibble `a`=10 requires `zt10`), you must:
   - Declare `fq_fe zt{2n};` alongside the other precomputes
   - Add `fq25_chain_mul(zt{2n}, zt{2n-2}, zt2);` to the precompute block
   - Add `ranshaw_secure_erase(zt{2n}, sizeof(zt{2n}));` at the cleanup

The script output explicitly lists "Required precompute values: zt2, zt3, ..., ztN" — check this against the existing declarations.

**Note**: `fq/src/x64/fq_invert.cpp` uses divsteps (via `FQ_MODULUS_S62`), NOT Fermat inversion. It has no chain to update — the modulus was already handled in Phase 2.

---

## Phase 5: SSWU Hash-to-Curve Constants

**The SSWU Z value depends on both the prime AND the curve coefficient b.** When b changes, Z must be recomputed. The `compute_curve_constants.py` script auto-detects Z by searching {-1, -2, -3, ...} for the first non-square where `g(B/(Z*A))` is square.

The script output sections "SSWU hash-to-curve constants (Shaw over Fq)" and "Ran SSWU hash-to-curve constants (Ran over Fp)" contain all values.

Update **4 files** (same constants appear in x64 and portable paths):

| File | Constants to update |
|------|-------------------|
| `ran/src/x64/ran_map_to_curve.cpp` | `SSWU_Z`, `SSWU_NEG_B_OVER_A`, `SSWU_B_OVER_ZA`, `SSWU_A` |
| `ran/src/portable/ran_map_to_curve.cpp` | `SSWU_Z_LIMBS`, `SSWU_NEG_B_OVER_A_LIMBS`, `SSWU_B_OVER_ZA_LIMBS`, `SSWU_A_LIMBS`, `RAN_B_LIMBS` |
| `shaw/src/x64/shaw_map_to_curve.cpp` | `SSWU_Z`, `SSWU_NEG_B_OVER_A`, `SSWU_B_OVER_ZA`, `SSWU_A` |
| `shaw/src/portable/shaw_map_to_curve.cpp` | `SSWU_Z_LIMBS`, `SSWU_NEG_B_OVER_A_LIMBS`, `SSWU_B_OVER_ZA_LIMBS`, `SSWU_A_LIMBS`, `SHAW_B_LIMBS` |

Update the comments too (e.g., `/* Z = -2 mod p */` not the old `/* Z = 7 */`).

The `compute_curve_constants.py` script handles the `-B/A` formula correctly (`b/3`, since A=-3). Do not manually compute these — use the script output.

---

## Phase 6: Generator Precompute Tables

```bash
python tools/gen_g_tables.py
```

Writes directly to `ran/include/ran_g_table.inl` and `shaw/include/shaw_g_table.inl`. No manual patching — the script reads generators from the constant headers you already updated in Phase 2.

**Run this AFTER Phase 2** — it reads `RAN_GX`/`RAN_GY`/`SHAW_GX`/`SHAW_GY` from the source.

---

## Phase 7: ECFFT Auxiliary Curve (Fq)

The Fq ECFFT auxiliary curve depends on q. If q changed, you need a new one. If q is unchanged (only b/generators changed), skip Phase 7 and Phase 8 entirely — the existing ECFFT data is still valid.

```bash
# Build the search tool (non-portable, native x64, needs BUILD_TOOLS)
# Use cmake-build skill: gcc Release -DBUILD_TOOLS=ON

./build/gcc-release-tools/ranshaw-find-ecfft.exe \
  --field fq --trials 500000 --min-levels 12 --cpus auto
```

This can run for **minutes to hours**. The output looks like:
```
*** HIT: field=fq a=-3 b=0x<hex> levels=15 (v2=16, domain=32768) ***
```

Save the `b` value.

The Fp ECFFT data (`ecfft/include/ecfft_fp_data.inl`) only changes if p changes, which it doesn't (p = 2^255 - 19 is fixed).

---

## Phase 8: Generate ECFFT Data

The gen tool writes data to **stdout** and diagnostics to **stderr**. You must redirect correctly or the diagnostics will pollute the `.inl` file.

```bash
./build/gcc-release-tools/ranshaw-gen-ecfft.exe fq \
  --known-b 0x<b_from_phase_7> [--seed <seed_from_phase_7>] \
  2>/dev/null > ecfft/include/ecfft_fq_data.inl
```

**Verify the file starts with `//`**, not diagnostic output:
```bash
head -1 ecfft/include/ecfft_fq_data.inl
# Should print: // Auto-generated by ranshaw-gen-ecfft — DO NOT EDIT
```

The `ecfft_fq.h` and `ecfft_fp.h` init functions use generated pointer arrays (`ECFFT_FQ_ISO_NUM_PTRS`, etc.) from the `.inl` data. They adapt automatically to any level count — no manual edits needed.

---

## Phase 9: Regenerate Test Vectors

The test vector generator MUST be built with `FORCE_PORTABLE=1` to ensure vectors come from the reference backend.

```bash
# Build (use cmake-build skill: gcc Release -DFORCE_PORTABLE=1 -DBUILD_TOOLS=ON)

# Generate JSON — stderr to /dev/null, stdout is the JSON
./build/gcc-release-portable-tools/ranshaw-gen-testvectors.exe \
  2>/dev/null > test_vectors/ranshaw_test_vectors.json

# Verify it's valid JSON (not polluted by stderr)
python -c "import json; json.load(open('test_vectors/ranshaw_test_vectors.json')); print('OK')"

# Convert to C++ header
python tools/json_to_header.py \
  test_vectors/ranshaw_test_vectors.json \
  include/ranshaw_test_vectors.h
```


---

## Phase 10: Cross-Validate (MUST PASS)

```bash
pip install ecpy  # if not already installed
python tools/validate_test_vectors.py test_vectors/ranshaw_test_vectors.json
```

This independently recomputes every test vector using Python + ecpy and compares against the JSON. It covers:
- Scalar arithmetic (Ran + Shaw): add, sub, mul, sq, negate, invert, reduce_wide, muladd
- Point operations: add, dbl, negate, scalar_mul, MSM, Pedersen commit
- Raw field arithmetic: Fp/Fq mul, sq, invert, sqrt
- Compressed points: G, 2G, 7G for both curves
- SSWU map-to-curve: single and double, all test inputs
- Polynomial arithmetic, divisors, batch invert, high-degree poly mul

**Do not proceed if any test fails.**

---

## Phase 11: Full Test Matrix

**Clean build every config.**

4 configs x 3 compilers = 12 builds minimum:

| Config | Extra CMake flags |
|--------|-------------------|
| Default (AVX2+IFMA) | `-DBUILD_TESTS=ON` |
| FORCE_PORTABLE | `-DBUILD_TESTS=ON -DFORCE_PORTABLE=1` |
| No SIMD | `-DBUILD_TESTS=ON -DENABLE_AVX2=OFF -DENABLE_AVX512=OFF` |
| AVX2 only | `-DBUILD_TESTS=ON -DENABLE_AVX512=OFF` |

For each build, run both:
```bash
./ranshaw-tests.exe          # unit tests — must be 0 failures
./ranshaw-fuzz-tests.exe     # algebraic identity tests — must be 0 failures
```

Add `--init` and `--autotune` dispatch modes for full coverage:
```bash
./ranshaw-tests.exe --init
./ranshaw-tests.exe --autotune
./ranshaw-fuzz-tests.exe --init
./ranshaw-fuzz-tests.exe --autotune
```

---

## ECFFT Auxiliary Curve Replacement Only

If only replacing the Fq ECFFT auxiliary curve (same q, same b, just a better auxiliary curve):

1. Run `ranshaw-find-ecfft.exe --field fq ...` (Phase 7 above)
2. Run `ranshaw-gen-ecfft.exe fq --known-b <b> 2>/dev/null > ecfft/include/ecfft_fq_data.inl`
3. Verify output: `head -1 ecfft/include/ecfft_fq_data.inl` should start with `//`
4. Rebuild ECFFT-enabled configs and test

No constant changes, no test vector regeneration, no cross-validation needed.

---

## File Manifest

### All files that change during a full curve swap

| Phase | File | What changes |
|-------|------|-------------|
| 2 | `fq/include/x64/fq51.h` | GAMMA, Q, BIAS, TWO_GAMMA arrays + LIMBS defines |
| 2 | `fq/include/portable/fq25.h` | GAMMA_25, Q_25 arrays + LIMBS defines |
| 2 | `ran/include/ran_constants.h` | RAN_B, RAN_GX, RAN_GY (both 32/64-bit), RAN_ORDER |
| 2 | `shaw/include/shaw_constants.h` | SHAW_B, SHAW_GX, SHAW_GY (both 32/64-bit) |
| 2 | `fq/include/x64/fq_divsteps.h` | FQ_MODULUS_S62 |
| 3 | `fq/include/fq_ops.h` | BIAS_Q_51 values (only if multiplier changes) |
| 3 | `fq/include/x64/avx2/fq10x4_avx2.h` | Bias multiplier in bias functions |
| 3 | `fq/include/x64/avx2/fq10_avx2.h` | Hardcoded 2*Q_25 values |
| 3 | `shaw/include/x64/ifma/fq51x8_ifma.h` | BIAS_Q_51 references |
| 4 | `fq/src/x64/fq_sqrt.cpp` | (q+1)/4 nibble scan section |
| 4 | `fq/src/portable/fq_sqrt.cpp` | (q+1)/4 nibble scan section |
| 4 | `fq/src/portable/fq_invert.cpp` | (q-2) nibble scan + precompute table |
| 5 | `ran/src/x64/ran_map_to_curve.cpp` | SSWU_Z, SSWU_NEG_B_OVER_A, SSWU_B_OVER_ZA, SSWU_A |
| 5 | `ran/src/portable/ran_map_to_curve.cpp` | Same (as _LIMBS) + RAN_B_LIMBS |
| 5 | `shaw/src/x64/shaw_map_to_curve.cpp` | SSWU_Z, SSWU_NEG_B_OVER_A, SSWU_B_OVER_ZA, SSWU_A |
| 5 | `shaw/src/portable/shaw_map_to_curve.cpp` | Same (as _LIMBS) + SHAW_B_LIMBS |
| 6 | `ran/include/ran_g_table.inl` | Generator precompute table (auto-generated) |
| 6 | `shaw/include/shaw_g_table.inl` | Generator precompute table (auto-generated) |
| 8 | `ecfft/include/ecfft_fq_data.inl` | ECFFT precomputed data (auto-generated) |
| 9 | `test_vectors/ranshaw_test_vectors.json` | All test vectors (auto-generated) |
| 9 | `include/ranshaw_test_vectors.h` | C++ header (auto-generated from JSON) |

### Files that do NOT change

- `fq/src/x64/fq_invert.cpp` — uses divsteps via FQ_MODULUS_S62, no Fermat chain
- `ecfft/include/ecfft_fp_data.inl` — Fp ECFFT, depends on p which is fixed
- `ecfft/include/ecfft_fp.h` — same
- All Fp field arithmetic — p = 2^255-19 is fixed, Fp code never changes
- `SHAW_ORDER` in `shaw/include/shaw_constants.h` — it's p, not q

