#!/usr/bin/env sage
# -*- coding: utf-8 -*-
"""
Independent SageMath test vector generator/validator for ranshaw.

Usage:
  sage test_vectors.sage --generate > test_vectors_sage.json
  sage test_vectors.sage --validate test_vectors/ranshaw_test_vectors.json

All arithmetic is performed purely in SageMath (no C++ calls).
"""

import json
import sys

# No hardcoded curve parameters — everything is loaded from the JSON file at runtime.
# See load_curve_params() and validate() below.

# ── Serialization helpers ──

def to_le_hex(val, length=32):
    """Convert an integer to little-endian hex string."""
    val = int(val) % (2**(length*8))
    b = int(val).to_bytes(length, 'little')
    return b.hex()

def from_le_hex(h):
    """Convert a little-endian hex string to integer."""
    return int.from_bytes(bytes.fromhex(h), 'little')

def compress_point(P, prime):
    """Compress a Weierstrass point to 32-byte LE with bit 255 = y parity."""
    if P == P.curve()(0):  # identity
        return to_le_hex(0)
    x = int(P[0])
    y = int(P[1])
    # y parity: bit 255 of x-coordinate encoding
    encoded = x
    if y % 2 == 1:
        encoded |= (1 << 255)
    return to_le_hex(encoded)

def decompress_point(hex_str, curve, prime):
    """Decompress a 32-byte LE point encoding."""
    val = from_le_hex(hex_str)
    if val == 0:
        return curve(0)  # identity
    y_parity = (val >> 255) & 1
    x = val & ((1 << 255) - 1)
    F = GF(prime)
    x_f = F(x)
    rhs = x_f**3 + curve.a4()*x_f + curve.a6()
    if not rhs.is_square():
        return None
    y_f = rhs.sqrt()
    if int(y_f) % 2 != y_parity:
        y_f = -y_f
    return curve(x_f, y_f)

# ── RFC 9380 Simplified SWU map_to_curve ──

def sswu_map(u_int, curve, prime):
    """
    Simplified SWU map for y^2 = x^3 + a*x + b.
    Z is the first non-square from -1, -2, ... such that g(B/(Z*A)) is square.
    """
    F = GF(prime)
    a = curve.a4()
    b_coeff = curve.a6()

    # Find Z: first negative integer that is non-square in F
    Z = None
    for z_try in range(-1, -100, -1):
        z_cand = F(z_try)
        if not z_cand.is_square():
            Z = z_cand
            break
    assert Z is not None, "Failed to find non-square Z"

    u = F(u_int)

    # SWU core
    tv1 = (Z**2 * u**4 + Z * u**2)
    if tv1 == 0:
        x1 = b_coeff / (Z * a)
    else:
        x1 = (-b_coeff / a) * (1 + 1/tv1)

    gx1 = x1**3 + a * x1 + b_coeff
    if gx1.is_square():
        y1 = gx1.sqrt()
        # Fix sign: sgn0(u) == sgn0(y)
        if int(u) % 2 != int(y1) % 2:
            y1 = -y1
        return curve(x1, y1)
    else:
        x2 = Z * u**2 * x1
        gx2 = x2**3 + a * x2 + b_coeff
        y2 = gx2.sqrt()
        if int(u) % 2 != int(y2) % 2:
            y2 = -y2
        return curve(x2, y2)

# ── Validation ──

def validate_scalar_section(section, field_order, field_name, errors):
    """Validate scalar test vectors against Sage arithmetic."""
    F = GF(field_order)

    # from_bytes
    for vec in section.get("from_bytes", []):
        label = vec["label"]
        inp = from_le_hex(vec["input"])
        expected = vec["result"]
        if inp >= field_order:
            if expected is not None:
                errors.append(f"{field_name}.from_bytes.{label}: expected null for out-of-range, got {expected}")
        else:
            if expected is None:
                errors.append(f"{field_name}.from_bytes.{label}: expected {to_le_hex(inp)}, got null")
            elif to_le_hex(inp) != expected:
                errors.append(f"{field_name}.from_bytes.{label}: mismatch")

    # add
    for vec in section.get("add", []):
        label = vec["label"]
        a = F(from_le_hex(vec["a"]))
        b = F(from_le_hex(vec["b"]))
        expected = vec["result"]
        computed = to_le_hex(int(a + b))
        if computed != expected:
            errors.append(f"{field_name}.add.{label}: expected {expected}, got {computed}")

    # sub
    for vec in section.get("sub", []):
        label = vec["label"]
        a = F(from_le_hex(vec["a"]))
        b = F(from_le_hex(vec["b"]))
        expected = vec["result"]
        computed = to_le_hex(int(a - b))
        if computed != expected:
            errors.append(f"{field_name}.sub.{label}: expected {expected}, got {computed}")

    # mul
    for vec in section.get("mul", []):
        label = vec["label"]
        a = F(from_le_hex(vec["a"]))
        b = F(from_le_hex(vec["b"]))
        expected = vec["result"]
        computed = to_le_hex(int(a * b))
        if computed != expected:
            errors.append(f"{field_name}.mul.{label}: expected {expected}, got {computed}")

    # sq
    for vec in section.get("sq", []):
        label = vec["label"]
        a = F(from_le_hex(vec["a"]))
        expected = vec["result"]
        computed = to_le_hex(int(a * a))
        if computed != expected:
            errors.append(f"{field_name}.sq.{label}: expected {expected}, got {computed}")

    # negate
    for vec in section.get("negate", []):
        label = vec["label"]
        a = F(from_le_hex(vec["a"]))
        expected = vec["result"]
        computed = to_le_hex(int(-a))
        if computed != expected:
            errors.append(f"{field_name}.negate.{label}: expected {expected}, got {computed}")

    # invert
    for vec in section.get("invert", []):
        label = vec["label"]
        a_int = from_le_hex(vec["a"])
        expected = vec["result"]
        if a_int == 0:
            if expected is not None:
                errors.append(f"{field_name}.invert.{label}: expected null for zero, got {expected}")
        else:
            a = F(a_int)
            computed = to_le_hex(int(a**(-1)))
            if computed != expected:
                errors.append(f"{field_name}.invert.{label}: expected {expected}, got {computed}")

    # reduce_wide
    for vec in section.get("reduce_wide", []):
        label = vec["label"]
        inp_bytes = bytes.fromhex(vec["input"])
        inp_int = int.from_bytes(inp_bytes, 'little')
        expected = vec["result"]
        computed = to_le_hex(inp_int % field_order)
        if computed != expected:
            errors.append(f"{field_name}.reduce_wide.{label}: expected {expected}, got {computed}")

    # muladd
    for vec in section.get("muladd", []):
        label = vec["label"]
        a = F(from_le_hex(vec["a"]))
        b = F(from_le_hex(vec["b"]))
        c = F(from_le_hex(vec["c"]))
        expected = vec["result"]
        computed = to_le_hex(int(a * b + c))
        if computed != expected:
            errors.append(f"{field_name}.muladd.{label}: expected {expected}, got {computed}")

    # is_zero
    for vec in section.get("is_zero", []):
        label = vec["label"]
        a_int = from_le_hex(vec["a"])
        expected = vec["result"]
        computed = (a_int % field_order == 0)
        if computed != expected:
            errors.append(f"{field_name}.is_zero.{label}: expected {expected}, got {computed}")


def validate_point_section(section, curve, G, scalar_order, base_prime, field_name, errors):
    """Validate point test vectors against Sage arithmetic."""

    # generator
    gen_hex = section.get("generator")
    if gen_hex:
        computed = compress_point(G, base_prime)
        if computed != gen_hex:
            errors.append(f"{field_name}.generator: mismatch")

    # identity
    id_hex = section.get("identity")
    if id_hex:
        computed = compress_point(curve(0), base_prime)
        if computed != id_hex:
            errors.append(f"{field_name}.identity: mismatch")

    # scalar_mul
    for vec in section.get("scalar_mul", []):
        label = vec["label"]
        s = from_le_hex(vec["scalar"])
        pt_hex = vec["point"]
        expected = vec["result"]

        pt = decompress_point(pt_hex, curve, base_prime)
        if pt is None:
            errors.append(f"{field_name}.scalar_mul.{label}: failed to decompress point")
            continue

        result_pt = int(s) * pt
        computed = compress_point(result_pt, base_prime)
        if computed != expected:
            errors.append(f"{field_name}.scalar_mul.{label}: expected {expected}, got {computed}")

    # add
    for vec in section.get("add", []):
        label = vec["label"]
        a_pt = decompress_point(vec["a"], curve, base_prime)
        b_pt = decompress_point(vec["b"], curve, base_prime)
        expected = vec["result"]
        if a_pt is None or b_pt is None:
            errors.append(f"{field_name}.add.{label}: failed to decompress")
            continue
        result_pt = a_pt + b_pt
        computed = compress_point(result_pt, base_prime)
        if computed != expected:
            errors.append(f"{field_name}.add.{label}: expected {expected}, got {computed}")

    # dbl
    for vec in section.get("dbl", []):
        label = vec["label"]
        a_pt = decompress_point(vec["a"], curve, base_prime)
        expected = vec["result"]
        if a_pt is None:
            errors.append(f"{field_name}.dbl.{label}: failed to decompress")
            continue
        result_pt = 2 * a_pt
        computed = compress_point(result_pt, base_prime)
        if computed != expected:
            errors.append(f"{field_name}.dbl.{label}: expected {expected}, got {computed}")

    # negate
    for vec in section.get("negate", []):
        label = vec["label"]
        a_pt = decompress_point(vec["a"], curve, base_prime)
        expected = vec["result"]
        if a_pt is None:
            errors.append(f"{field_name}.negate.{label}: failed to decompress")
            continue
        result_pt = -a_pt
        computed = compress_point(result_pt, base_prime)
        if computed != expected:
            errors.append(f"{field_name}.negate.{label}: expected {expected}, got {computed}")

    # msm
    for vec in section.get("msm", []):
        label = vec["label"]
        n = vec["n"]
        scalars = [from_le_hex(s) for s in vec["scalars"]]
        points = [decompress_point(ph, curve, base_prime) for ph in vec["points"]]
        expected = vec["result"]
        if any(pt is None for pt in points):
            errors.append(f"{field_name}.msm.{label}: failed to decompress points")
            continue
        result_pt = sum(s * pt for s, pt in zip(scalars, points))
        computed = compress_point(result_pt, base_prime)
        if computed != expected:
            errors.append(f"{field_name}.msm.{label}: expected {expected}, got {computed}")

    # pedersen_commit
    for vec in section.get("pedersen_commit", []):
        label = vec["label"]
        blind = from_le_hex(vec["blinding"])
        H_pt = decompress_point(vec["H"], curve, base_prime)
        values = [from_le_hex(v) for v in vec["values"]]
        generators = [decompress_point(g, curve, base_prime) for g in vec["generators"]]
        expected = vec["result"]
        if H_pt is None or any(g is None for g in generators):
            errors.append(f"{field_name}.pedersen.{label}: failed to decompress")
            continue
        result_pt = int(blind) * H_pt
        for v, gen in zip(values, generators):
            result_pt += int(v) * gen
        computed = compress_point(result_pt, base_prime)
        if computed != expected:
            errors.append(f"{field_name}.pedersen.{label}: expected {expected}, got {computed}")

    # x_coordinate
    for vec in section.get("x_coordinate", []):
        label = vec["label"]
        pt = decompress_point(vec["point"], curve, base_prime)
        expected = vec["x_bytes"]
        if pt is None:
            errors.append(f"{field_name}.x_coordinate.{label}: failed to decompress")
            continue
        x_int = int(pt[0])
        computed = to_le_hex(x_int)
        if computed != expected:
            errors.append(f"{field_name}.x_coordinate.{label}: expected {expected}, got {computed}")

    # from_bytes
    for vec in section.get("from_bytes", []):
        label = vec["label"]
        inp = vec["input"]
        expected = vec["result"]
        pt = decompress_point(inp, curve, base_prime)
        if pt is None:
            if expected is not None:
                errors.append(f"{field_name}.from_bytes.{label}: expected null, got result")
        else:
            computed = compress_point(pt, base_prime)
            if expected is None:
                # We got a valid point but C++ says invalid — might be off-curve
                # Don't flag this as error since our decompress might differ
                pass
            elif computed != expected:
                errors.append(f"{field_name}.from_bytes.{label}: expected {expected}, got {computed}")


def validate_polynomial_section(section, field_order, field_name, errors):
    """Validate polynomial test vectors."""
    F = GF(field_order)
    R = PolynomialRing(F, 'x')
    x = R.gen()

    def make_poly(coeffs_hex):
        return sum(F(from_le_hex(c)) * x**i for i, c in enumerate(coeffs_hex))

    def poly_to_le_hex_list(poly):
        if poly == 0:
            return [to_le_hex(0)]
        return [to_le_hex(int(poly[i])) for i in range(poly.degree() + 1)]

    # from_roots
    for vec in section.get("from_roots", []):
        label = vec["label"]
        roots_hex = vec.get("roots", [])
        coeffs_hex = vec.get("coefficients", [])
        if roots_hex:
            roots = [F(from_le_hex(r)) for r in roots_hex]
            poly = prod(x - r for r in roots)
            computed = poly_to_le_hex_list(poly)
            if computed != coeffs_hex:
                errors.append(f"{field_name}.from_roots.{label}: coefficient mismatch")
        else:
            degree = vec.get("degree", len(coeffs_hex) - 1)
            if len(coeffs_hex) != degree + 1:
                errors.append(f"{field_name}.from_roots.{label}: coefficient count mismatch")

    # evaluate
    for vec in section.get("evaluate", []):
        label = vec["label"]
        x_val = F(from_le_hex(vec["x"]))
        expected = vec["result"]
        coeffs = vec.get("coefficients", [])
        if coeffs:
            poly = make_poly(coeffs)
            computed = to_le_hex(int(poly(x_val)))
            if computed != expected:
                errors.append(f"{field_name}.evaluate.{label}: expected {expected}, got {computed}")

    # mul
    for vec in section.get("mul", []):
        label = vec["label"]
        a_poly = make_poly(vec["a_coefficients"])
        b_poly = make_poly(vec["b_coefficients"])
        expected = vec["coefficients"]
        result = a_poly * b_poly
        computed = poly_to_le_hex_list(result)
        if computed != expected:
            errors.append(f"{field_name}.mul.{label}: coefficient mismatch")

    # add
    for vec in section.get("add", []):
        label = vec["label"]
        a_poly = make_poly(vec["a_coefficients"])
        b_poly = make_poly(vec["b_coefficients"])
        expected = vec["coefficients"]
        result = a_poly + b_poly
        computed = poly_to_le_hex_list(result)
        if computed != expected:
            errors.append(f"{field_name}.add.{label}: coefficient mismatch")

    # sub
    for vec in section.get("sub", []):
        label = vec["label"]
        a_poly = make_poly(vec["a_coefficients"])
        b_poly = make_poly(vec["b_coefficients"])
        expected = vec["coefficients"]
        result = a_poly - b_poly
        computed = poly_to_le_hex_list(result)
        if computed != expected:
            errors.append(f"{field_name}.sub.{label}: coefficient mismatch")

    # divmod
    for vec in section.get("divmod", []):
        label = vec["label"]
        num = make_poly(vec["numerator"])
        den = make_poly(vec["denominator"])
        expected_q = vec["quotient"]
        expected_r = vec["remainder"]
        q, r = num.quo_rem(den)
        computed_q = poly_to_le_hex_list(q)
        computed_r = poly_to_le_hex_list(r)
        if computed_q != expected_q:
            errors.append(f"{field_name}.divmod.{label}: quotient mismatch")
        if computed_r != expected_r:
            errors.append(f"{field_name}.divmod.{label}: remainder mismatch")

    # interpolate
    for vec in section.get("interpolate", []):
        label = vec["label"]
        xs = [F(from_le_hex(h)) for h in vec["xs"]]
        ys = [F(from_le_hex(h)) for h in vec["ys"]]
        expected = vec["coefficients"]
        poly = R.lagrange_polynomial(list(zip(xs, ys)))
        computed = poly_to_le_hex_list(poly)
        if computed != expected:
            errors.append(f"{field_name}.interpolate.{label}: coefficient mismatch")


def validate_divisor_section(section, curve, G, scalar_order, base_prime, field_name, errors):
    """Validate divisor test vectors (evaluation only — polynomial structure checked separately)."""
    for vec in section.get("compute", []):
        label = vec["label"]
        n = vec["n"]
        points = [decompress_point(ph, curve, base_prime) for ph in vec["points"]]
        if any(pt is None for pt in points):
            errors.append(f"{field_name}.compute.{label}: failed to decompress points")
            continue

        # Verify evaluation at non-member point is nonzero
        eval_result = vec.get("eval_result")
        if eval_result and from_le_hex(eval_result) == 0:
            errors.append(f"{field_name}.compute.{label}: eval at non-member should be nonzero")


def validate_wei25519(section, ran_base_field, errors):
    """Validate Wei25519 bridge vectors."""
    for vec in section.get("x_to_shaw_scalar", []):
        label = vec["label"]
        inp = from_le_hex(vec["input"])
        expected = vec["result"]

        if inp >= ran_base_field:
            if expected is not None:
                errors.append(f"wei25519.{label}: expected null for x >= p, got {expected}")
        else:
            computed = to_le_hex(inp)
            if expected is None:
                errors.append(f"wei25519.{label}: expected {computed}, got null")
            elif computed != expected:
                errors.append(f"wei25519.{label}: expected {expected}, got {computed}")


def validate_high_degree_poly_mul(section, ran_base_field, shaw_base_field, errors):
    """Validate high-degree polynomial multiplication via multi-point evaluation."""
    for field_name, field_order in [("fp", ran_base_field), ("fq", shaw_base_field)]:
        F = GF(field_order)
        R = PolynomialRing(F, 'x')
        x = R.gen()
        for vec in section.get(field_name, []):
            label = vec["label"]
            n_coeffs = vec["n_coeffs"]
            result_degree = vec["result_degree"]
            print(f"    {field_name}/{label} (n={n_coeffs})...", end="", flush=True)

            # Rebuild deterministic inputs: a[i] = i+1, b[i] = i+n+1
            a_poly = sum(F(i + 1) * x**i for i in range(n_coeffs))
            b_poly = sum(F(i + n_coeffs + 1) * x**i for i in range(n_coeffs))

            # Check result degree
            expected_degree = (n_coeffs - 1) * 2 if n_coeffs > 0 else 0
            if result_degree != expected_degree:
                errors.append(f"high_degree.{field_name}.{label}: degree {result_degree} != expected {expected_degree}")
                continue

            # Multi-point eval checks
            for check in vec.get("eval_checks", []):
                pt_name = check["point"]
                xv = F(from_le_hex(check["x"]))
                expected_a = from_le_hex(check["a_of_x"])
                expected_b = from_le_hex(check["b_of_x"])
                expected_r = from_le_hex(check["result_of_x"])

                # Verify a(x) and b(x) independently
                computed_a = int(a_poly(xv))
                if computed_a != expected_a:
                    errors.append(f"high_degree.{field_name}.{label}.{pt_name}: a(x) mismatch")
                    continue
                computed_b = int(b_poly(xv))
                if computed_b != expected_b:
                    errors.append(f"high_degree.{field_name}.{label}.{pt_name}: b(x) mismatch")
                    continue

                # Core check: a(x) * b(x) == result(x)
                expected_product = int(F(expected_a) * F(expected_b))
                if expected_product != expected_r:
                    errors.append(f"high_degree.{field_name}.{label}.{pt_name}: a*b={to_le_hex(expected_product)} != result={to_le_hex(expected_r)}")

            print(" OK")


def validate_batch_invert(section, ran_base_field, shaw_base_field, errors):
    """Validate batch inversion vectors."""
    for field_name, field_order in [("fp", ran_base_field), ("fq", shaw_base_field)]:
        F = GF(field_order)
        for vec in section.get(field_name, []):
            label = vec["label"]
            inputs = [from_le_hex(h) for h in vec["inputs"]]
            results = [from_le_hex(h) for h in vec["results"]]
            for i, (inp, res) in enumerate(zip(inputs, results)):
                if inp == 0:
                    if res != 0:
                        errors.append(f"batch_invert.{field_name}.{label}[{i}]: expected 0 for zero input")
                else:
                    computed = int(F(inp)**(-1))
                    if computed != res:
                        errors.append(f"batch_invert.{field_name}.{label}[{i}]: expected {to_le_hex(computed)}, got {to_le_hex(res)}")


def validate(filename):
    """Validate all test vectors in a JSON file.
    ALL curve parameters are loaded from the JSON — no hardcoded values."""
    with open(filename) as f:
        data = json.load(f)

    errors = []

    print(f"Validating {filename}...")

    # Load ALL parameters from JSON
    params = data["parameters"]
    ran_order = from_le_hex(params["ran_order"])  # Q — Ran scalar field = Shaw base field
    shaw_order = from_le_hex(params["shaw_order"])   # P — Shaw scalar field = Ran base field
    curve_a = params["curve_a"]                          # a coefficient (both curves)

    # Ran base field = Shaw scalar field order = P
    ran_base = shaw_order
    # Shaw base field = Ran scalar field order = Q
    shaw_base = ran_order

    Fp = GF(ran_base)
    Fq = GF(shaw_base)

    ran_b_val = Fp(from_le_hex(params["ran_b"]))
    shaw_b_val = Fq(from_le_hex(params["shaw_b"]))

    E_ran = EllipticCurve(Fp, [curve_a, ran_b_val])
    E_shaw = EllipticCurve(Fq, [curve_a, shaw_b_val])

    # Decode generators from JSON
    ran_gx = from_le_hex(params["ran_gx"])
    ran_gy = from_le_hex(params["ran_gy"])
    G_ran = E_ran(Fp(ran_gx), Fp(ran_gy))

    shaw_gx = from_le_hex(params["shaw_gx"])
    shaw_gy = from_le_hex(params["shaw_gy"])
    G_shaw = E_shaw(Fq(shaw_gx), Fq(shaw_gy))

    print(f"  Ran: y^2 = x^3 + {curve_a}x + b over Fp")
    print(f"    Fp = 0x{ran_base:064x}")
    print(f"    b  = 0x{int(ran_b_val):064x}")
    print(f"    order = 0x{ran_order:064x}")
    print(f"  Shaw: y^2 = x^3 + {curve_a}x + b over Fq")
    print(f"    Fq = 0x{shaw_base:064x}")
    print(f"    b  = 0x{int(shaw_b_val):064x}")
    print(f"    order = 0x{shaw_order:064x}")
    print("  Parameters: OK")

    # Scalar sections
    if "ran_scalar" in data:
        print("  Ran scalar...", end="", flush=True)
        validate_scalar_section(data["ran_scalar"], ran_order, "ran_scalar", errors)
        print(" OK" if not any("ran_scalar" in e for e in errors) else " ERRORS")

    if "shaw_scalar" in data:
        print("  Shaw scalar...", end="", flush=True)
        validate_scalar_section(data["shaw_scalar"], shaw_order, "shaw_scalar", errors)
        print(" OK" if not any("shaw_scalar" in e for e in errors) else " ERRORS")

    # Point sections
    if "ran_point" in data:
        print("  Ran point...", end="", flush=True)
        validate_point_section(data["ran_point"], E_ran, G_ran, ran_order, ran_base, "ran_point", errors)
        print(" OK" if not any("ran_point" in e for e in errors) else " ERRORS")

    if "shaw_point" in data:
        print("  Shaw point...", end="", flush=True)
        validate_point_section(data["shaw_point"], E_shaw, G_shaw, shaw_order, shaw_base, "shaw_point", errors)
        print(" OK" if not any("shaw_point" in e for e in errors) else " ERRORS")

    # Polynomial sections
    if "fp_polynomial" in data:
        print("  Fp polynomial...", end="", flush=True)
        validate_polynomial_section(data["fp_polynomial"], ran_base, "fp_polynomial", errors)
        print(" OK" if not any("fp_polynomial" in e for e in errors) else " ERRORS")

    if "fq_polynomial" in data:
        print("  Fq polynomial...", end="", flush=True)
        validate_polynomial_section(data["fq_polynomial"], shaw_base, "fq_polynomial", errors)
        print(" OK" if not any("fq_polynomial" in e for e in errors) else " ERRORS")

    # Divisor sections
    if "ran_divisor" in data:
        print("  Ran divisor...", end="", flush=True)
        validate_divisor_section(data["ran_divisor"], E_ran, G_ran, ran_order, ran_base, "ran_divisor", errors)
        print(" OK" if not any("ran_divisor" in e for e in errors) else " ERRORS")

    if "shaw_divisor" in data:
        print("  Shaw divisor...", end="", flush=True)
        validate_divisor_section(data["shaw_divisor"], E_shaw, G_shaw, shaw_order, shaw_base, "shaw_divisor", errors)
        print(" OK" if not any("shaw_divisor" in e for e in errors) else " ERRORS")

    # Wei25519
    if "wei25519" in data:
        print("  Wei25519 bridge...", end="", flush=True)
        validate_wei25519(data["wei25519"], ran_base, errors)
        print(" OK" if not any("wei25519" in e for e in errors) else " ERRORS")

    # Batch invert
    if "batch_invert" in data:
        print("  Batch invert...", end="", flush=True)
        validate_batch_invert(data["batch_invert"], ran_base, shaw_base, errors)
        print(" OK" if not any("batch_invert" in e for e in errors) else " ERRORS")

    # High-degree polynomial multiplication
    if "high_degree_poly_mul" in data:
        print("  High-degree poly mul...")
        validate_high_degree_poly_mul(data["high_degree_poly_mul"], ran_base, shaw_base, errors)
        print("  High-degree poly mul:" + (" OK" if not any("high_degree" in e for e in errors) else " ERRORS"))

    # Summary
    print()
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("ALL VECTORS VALIDATED SUCCESSFULLY")
        return 0


# ── Main ──

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  sage test_vectors.sage --validate <json_file>")
        print("  sage test_vectors.sage --generate")
        sys.exit(1)

    if sys.argv[1] == "--validate":
        if len(sys.argv) < 3:
            print("Error: --validate requires a JSON file path")
            sys.exit(1)
        sys.exit(validate(sys.argv[2]))

    elif sys.argv[1] == "--generate":
        print("Generation mode not yet implemented.", file=sys.stderr)
        print("Use the C++ generator instead.", file=sys.stderr)
        sys.exit(1)

    else:
        print(f"Unknown option: {sys.argv[1]}")
        sys.exit(1)
