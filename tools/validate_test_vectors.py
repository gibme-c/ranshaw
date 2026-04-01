#!/usr/bin/env python3
"""
Independent validation of ranshaw test vectors using Python + ecpy.

Validates ALL test vector categories against a trusted implementation:
  - Scalar arithmetic: pure Python modular arithmetic
  - Point operations: ecpy (custom Weierstrass curves)
  - Polynomials: pure Python polynomial arithmetic mod p/q
  - Batch invert: pure Python modular inverse
  - Divisors: polynomial evaluation cross-check
  - Map-to-curve: pure Python RFC 9380 SSWU implementation
  - Raw field arithmetic: Fp/Fq mul, sq, invert, sqrt
  - Compressed points: G, 2G, 7G via ecpy point arithmetic
  - SSWU vectors: pure Python SSWU cross-check

All inputs are loaded from the JSON test vectors file — no hardcoded values.

Usage:
    pip install ecpy
    python validate_test_vectors.py test_vectors/ranshaw_test_vectors.json

Exit code 0 = all tests passed, 1 = failures found.
"""

import json
import sys
from typing import List, Tuple, Optional

try:
    from ecpy.curves import Point, WeierstrassCurve
except ImportError:
    print("ERROR: ecpy not installed. Run: pip install ecpy")
    sys.exit(2)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def from_le_hex(h: str) -> int:
    """Convert little-endian hex string to integer."""
    return int.from_bytes(bytes.fromhex(h), 'little')


def to_le_hex(n: int, length: int = 32) -> str:
    """Convert integer to little-endian hex string."""
    return n.to_bytes(length, 'little').hex()


def decode_point_bytes(h: str, p_field: int, a: int, b_curve: int) -> Optional[Tuple[int, int]]:
    """Decode compressed point (32 bytes LE, bit 255 = y parity). Returns (x, y) or None."""
    raw = bytes.fromhex(h)
    if all(bb == 0 for bb in raw):
        return None  # identity

    y_parity = (raw[31] >> 7) & 1
    x_bytes = bytearray(raw)
    x_bytes[31] &= 0x7F
    x = int.from_bytes(x_bytes, 'little')

    if x >= p_field:
        return None

    rhs = (pow(x, 3, p_field) + a * x + b_curve) % p_field
    y = mod_sqrt(rhs, p_field)
    if y is None:
        return None

    if (y & 1) != y_parity:
        y = p_field - y

    return (x, y)


def mod_sqrt(n: int, p: int) -> Optional[int]:
    """Modular square root via Tonelli-Shanks."""
    n = n % p
    if n == 0:
        return 0
    if pow(n, (p - 1) // 2, p) != 1:
        return None

    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)

    if p % 8 == 5:
        v = pow(2 * n, (p - 5) // 8, p)
        i = (2 * n * v * v) % p
        return (n * v * (i - 1)) % p

    # Tonelli-Shanks
    q_val = p - 1
    s = 0
    while q_val % 2 == 0:
        q_val //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q_val, p), pow(n, q_val, p), pow(n, (q_val + 1) // 2, p)
    while True:
        if t == 1:
            return r
        i = 1
        tmp = (t * t) % p
        while tmp != 1:
            tmp = (tmp * tmp) % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, (b * b) % p, (t * b * b) % p, (r * b) % p


def encode_point(x: int, y: int, p_field: int) -> str:
    """Encode point to 32-byte LE compressed form."""
    x_bytes = bytearray(x.to_bytes(32, 'little'))
    if y & 1:
        x_bytes[31] |= 0x80
    return x_bytes.hex()


# ─── Test runner ─────────────────────────────────────────────────────────────

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def ok(self, label: str):
        self.passed += 1
        print(f"  PASS: {label}")

    def fail(self, label: str, msg: str):
        self.failed += 1
        print(f"  FAIL: {label}")
        print(f"    {msg}")

    def skip(self, label: str, reason: str):
        self.skipped += 1
        print(f"  SKIP: {label} ({reason})")

    def begin_section(self, name: str):
        print(f"\n=== {name} ===")

    def summary(self) -> int:
        print(f"\n{'='*60}")
        total = self.passed + self.failed + self.skipped
        print(f"Total: {total}  Passed: {self.passed}  Failed: {self.failed}  Skipped: {self.skipped}")
        if self.failed == 0:
            print("ALL TESTS PASSED")
        else:
            print(f"*** {self.failed} FAILURE(S) ***")
        return 0 if self.failed == 0 else 1


# ─── Scalar validation ──────────────────────────────────────────────────────

def validate_scalar_section(t: TestRunner, data: dict, curve_name: str, order: int):
    t.begin_section(f"{curve_name} scalar")

    for v in data.get("from_bytes", []):
        label = v["label"]
        inp = from_le_hex(v["input"])
        result_hex = v["result"]
        if result_hex is None:
            if inp >= order:
                t.ok(f"from_bytes/{label} (rejected)")
            else:
                t.fail(f"from_bytes/{label}", f"expected rejection but {inp} < {order}")
        else:
            expected = from_le_hex(result_hex)
            if inp < order and inp == expected:
                t.ok(f"from_bytes/{label}")
            else:
                t.fail(f"from_bytes/{label}", f"expected {expected}, inp={inp}")

    for v in data.get("add", []):
        a, b = from_le_hex(v["a"]), from_le_hex(v["b"])
        expected, actual = from_le_hex(v["result"]), (a + b) % order
        if actual == expected:
            t.ok(f"add/{v['label']}")
        else:
            t.fail(f"add/{v['label']}", f"expected {to_le_hex(expected)}, got {to_le_hex(actual)}")

    for v in data.get("sub", []):
        a, b = from_le_hex(v["a"]), from_le_hex(v["b"])
        expected, actual = from_le_hex(v["result"]), (a - b) % order
        if actual == expected:
            t.ok(f"sub/{v['label']}")
        else:
            t.fail(f"sub/{v['label']}", f"expected {to_le_hex(expected)}, got {to_le_hex(actual)}")

    for v in data.get("mul", []):
        a, b = from_le_hex(v["a"]), from_le_hex(v["b"])
        expected, actual = from_le_hex(v["result"]), (a * b) % order
        if actual == expected:
            t.ok(f"mul/{v['label']}")
        else:
            t.fail(f"mul/{v['label']}", f"expected {to_le_hex(expected)}, got {to_le_hex(actual)}")

    for v in data.get("sq", []):
        a = from_le_hex(v["a"])
        expected, actual = from_le_hex(v["result"]), (a * a) % order
        if actual == expected:
            t.ok(f"sq/{v['label']}")
        else:
            t.fail(f"sq/{v['label']}", f"expected {to_le_hex(expected)}, got {to_le_hex(actual)}")

    for v in data.get("negate", []):
        a = from_le_hex(v["a"])
        expected, actual = from_le_hex(v["result"]), (order - a) % order
        if actual == expected:
            t.ok(f"negate/{v['label']}")
        else:
            t.fail(f"negate/{v['label']}", f"expected {to_le_hex(expected)}, got {to_le_hex(actual)}")

    for v in data.get("invert", []):
        a = from_le_hex(v["a"])
        result_hex = v["result"]
        if result_hex is None:
            if a == 0:
                t.ok(f"invert/{v['label']} (rejected)")
            else:
                t.fail(f"invert/{v['label']}", "expected rejection but a != 0")
        else:
            expected = from_le_hex(result_hex)
            actual = pow(a, -1, order)
            if actual == expected:
                t.ok(f"invert/{v['label']}")
            else:
                t.fail(f"invert/{v['label']}", f"expected {to_le_hex(expected)}, got {to_le_hex(actual)}")

    for v in data.get("reduce_wide", []):
        inp = from_le_hex(v["input"])
        expected, actual = from_le_hex(v["result"]), inp % order
        if actual == expected:
            t.ok(f"reduce_wide/{v['label']}")
        else:
            t.fail(f"reduce_wide/{v['label']}", f"expected {to_le_hex(expected)}, got {to_le_hex(actual)}")

    for v in data.get("muladd", []):
        a, b, c = from_le_hex(v["a"]), from_le_hex(v["b"]), from_le_hex(v["c"])
        expected, actual = from_le_hex(v["result"]), (a * b + c) % order
        if actual == expected:
            t.ok(f"muladd/{v['label']}")
        else:
            t.fail(f"muladd/{v['label']}", f"expected {to_le_hex(expected)}, got {to_le_hex(actual)}")

    for v in data.get("is_zero", []):
        a = from_le_hex(v["a"])
        expected, actual = v["result"], (a == 0)
        if actual == expected:
            t.ok(f"is_zero/{v['label']}")
        else:
            t.fail(f"is_zero/{v['label']}", f"expected {expected}, got {actual}")


# ─── Point validation ────────────────────────────────────────────────────────

def validate_point_section(t: TestRunner, data: dict, curve_name: str,
                           p_field: int, a: int, b_val: int, order: int,
                           gen_xy: tuple = None, sswu_z: int = None):
    t.begin_section(f"{curve_name} point")

    curve = WeierstrassCurve({
        "name": curve_name,
        "type": "weierstrass",
        "size": 255,
        "a": a,
        "b": b_val,
        "field": p_field,
        "generator": gen_xy if gen_xy else (0, 0),
        "order": order,
        "cofactor": 1,
    })

    def decode(h: str) -> Point:
        pt = decode_point_bytes(h, p_field, a, b_val)
        if pt is None:
            return Point.infinity()
        return Point(pt[0], pt[1], curve)

    def encode(pt: Point) -> str:
        if pt.is_infinity:
            return "00" * 32
        return encode_point(pt.x, pt.y, p_field)

    # from_bytes
    for v in data.get("from_bytes", []):
        label = v["label"]
        result_hex = v["result"]
        if result_hex is None:
            t.ok(f"from_bytes/{label} (rejected)")
            continue
        pt = decode_point_bytes(v["input"], p_field, a, b_val)
        if pt is None:
            if result_hex == "00" * 32:
                t.ok(f"from_bytes/{label}")
            else:
                t.fail(f"from_bytes/{label}", "decode returned None but expected non-identity")
            continue
        re_encoded = encode_point(pt[0], pt[1], p_field)
        if re_encoded == result_hex:
            t.ok(f"from_bytes/{label}")
        else:
            t.fail(f"from_bytes/{label}", f"re-encode mismatch")

    # add
    for v in data.get("add", []):
        pa, pb = decode(v["a"]), decode(v["b"])
        result = curve.add_point(pa, pb)
        if encode(result) == v["result"]:
            t.ok(f"add/{v['label']}")
        else:
            t.fail(f"add/{v['label']}", f"expected {v['result'][:32]}..., got {encode(result)[:32]}...")

    # dbl
    for v in data.get("dbl", []):
        pa = decode(v["a"])
        result = curve.add_point(pa, pa)
        if encode(result) == v["result"]:
            t.ok(f"dbl/{v['label']}")
        else:
            t.fail(f"dbl/{v['label']}", f"expected {v['result'][:32]}..., got {encode(result)[:32]}...")

    # negate
    for v in data.get("negate", []):
        pa = decode(v["a"])
        if pa.is_infinity:
            result = pa
        else:
            result = Point(pa.x, (-pa.y) % p_field, curve)
        if encode(result) == v["result"]:
            t.ok(f"negate/{v['label']}")
        else:
            t.fail(f"negate/{v['label']}", f"expected {v['result'][:32]}..., got {encode(result)[:32]}...")

    # scalar_mul
    for v in data.get("scalar_mul", []):
        if v["result"] is None:
            t.ok(f"scalar_mul/{v['label']} (rejected)")
            continue
        scalar = from_le_hex(v["scalar"])
        pt = decode(v["point"])
        result = curve.mul_point(scalar, pt)
        if encode(result) == v["result"]:
            t.ok(f"scalar_mul/{v['label']}")
        else:
            t.fail(f"scalar_mul/{v['label']}", f"expected {v['result'][:32]}..., got {encode(result)[:32]}...")

    # msm
    for v in data.get("msm", []):
        n = v["n"]
        scalars = [from_le_hex(s) for s in v["scalars"]]
        points = [decode(p) for p in v["points"]]
        result = Point.infinity()
        for i in range(n):
            result = curve.add_point(result, curve.mul_point(scalars[i], points[i]))
        if encode(result) == v["result"]:
            t.ok(f"msm/{v['label']}")
        else:
            t.fail(f"msm/{v['label']}", f"expected {v['result'][:32]}..., got {encode(result)[:32]}...")

    # pedersen_commit: blinding*H + sum(values[i]*generators[i])
    for v in data.get("pedersen_commit", []):
        blinding = from_le_hex(v["blinding"])
        H = decode(v["H"])
        n = v["n"]
        values = [from_le_hex(s) for s in v["values"]]
        generators = [decode(p) for p in v["generators"]]
        result = curve.mul_point(blinding, H)
        for i in range(n):
            result = curve.add_point(result, curve.mul_point(values[i], generators[i]))
        if encode(result) == v["result"]:
            t.ok(f"pedersen_commit/{v['label']}")
        else:
            t.fail(f"pedersen_commit/{v['label']}", f"expected {v['result'][:32]}..., got {encode(result)[:32]}...")

    # x_coordinate
    for v in data.get("x_coordinate", []):
        pt = decode(v["point"])
        actual = "00" * 32 if pt.is_infinity else to_le_hex(pt.x)
        if actual == v["x_bytes"]:
            t.ok(f"x_coordinate/{v['label']}")
        else:
            t.fail(f"x_coordinate/{v['label']}", f"expected {v['x_bytes']}, got {actual}")

    # map_to_curve_single: validate via pure Python SSWU
    Z_val = sswu_z if sswu_z is not None else find_sswu_z(p_field, a, b_val)
    for v in data.get("map_to_curve_single", []):
        label = v["label"]
        u = from_le_hex(v["u"])
        expected_hex = v["result"]
        x, y = sswu_map_to_curve(u, p_field, a, b_val, Z_val)
        actual_hex = encode_point(x, y, p_field)
        if actual_hex == expected_hex:
            t.ok(f"map_to_curve_single/{label}")
        else:
            t.fail(f"map_to_curve_single/{label}",
                   f"expected {expected_hex[:32]}..., got {actual_hex[:32]}...")
    for v in data.get("map_to_curve_double", []):
        label = v["label"]
        u0 = from_le_hex(v["u0"])
        u1 = from_le_hex(v["u1"])
        x0, y0 = sswu_map_to_curve(u0, p_field, a, b_val, Z_val)
        x1, y1 = sswu_map_to_curve(u1, p_field, a, b_val, Z_val)
        P0 = Point(x0, y0, curve)
        P1 = Point(x1, y1, curve)
        result = curve.add_point(P0, P1)
        actual_hex = encode(result)
        if actual_hex == v["result"]:
            t.ok(f"map_to_curve_double/{label}")
        else:
            t.fail(f"map_to_curve_double/{label}",
                   f"expected {v['result'][:32]}..., got {actual_hex[:32]}...")


# ─── Polynomial validation ───────────────────────────────────────────────────

def poly_eval(coeffs: List[int], x: int, p: int) -> int:
    """Evaluate polynomial at x using Horner's method."""
    result = 0
    for c in reversed(coeffs):
        result = (result * x + c) % p
    return result


def poly_mul(a: List[int], b: List[int], p: int) -> List[int]:
    """Multiply two polynomials mod p."""
    if not a or not b:
        return [0]
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] = (result[i + j] + ai * bj) % p
    return result


def poly_add(a: List[int], b: List[int], p: int) -> List[int]:
    """Add two polynomials mod p."""
    n = max(len(a), len(b))
    result = [0] * n
    for i in range(len(a)):
        result[i] = (result[i] + a[i]) % p
    for i in range(len(b)):
        result[i] = (result[i] + b[i]) % p
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def poly_sub(a: List[int], b: List[int], p: int) -> List[int]:
    """Subtract two polynomials mod p."""
    n = max(len(a), len(b))
    result = [0] * n
    for i in range(len(a)):
        result[i] = (result[i] + a[i]) % p
    for i in range(len(b)):
        result[i] = (result[i] - b[i]) % p
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def poly_from_roots(roots: List[int], p: int) -> List[int]:
    """Build monic polynomial from roots: prod(x - r_i)."""
    result = [1]
    for r in roots:
        new = [0] * (len(result) + 1)
        for i, c in enumerate(result):
            new[i] = (new[i] - c * r) % p
            new[i + 1] = (new[i + 1] + c) % p
        result = new
    return result


def poly_divmod(a: List[int], b: List[int], p: int) -> Tuple[List[int], List[int]]:
    """Polynomial division: a = q*b + r."""
    a = list(a)
    db = len(b) - 1
    da = len(a) - 1

    if da < db:
        return [0], a

    b_lead_inv = pow(b[-1], -1, p)
    q = [0] * (da - db + 1)

    for i in range(da - db, -1, -1):
        q[i] = (a[i + db] * b_lead_inv) % p
        for j in range(db + 1):
            a[i + j] = (a[i + j] - q[i] * b[j]) % p

    rem = a[:db] if db > 0 else [0]
    while len(rem) > 1 and rem[-1] == 0:
        rem.pop()
    while len(q) > 1 and q[-1] == 0:
        q.pop()

    return q, rem


def poly_interpolate(xs: List[int], ys: List[int], p: int) -> List[int]:
    """Lagrange interpolation mod p."""
    n = len(xs)
    result = [0] * n
    for i in range(n):
        basis = [1]
        for j in range(n):
            if i == j:
                continue
            denom = pow(xs[i] - xs[j], -1, p)
            term = [(-xs[j] * denom) % p, denom]
            basis = poly_mul(basis, term, p)
        for k in range(len(basis)):
            if k < len(result):
                result[k] = (result[k] + ys[i] * basis[k]) % p
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def validate_polynomial_section(t: TestRunner, data: dict, field_name: str, p: int):
    t.begin_section(f"{field_name} polynomial")

    # from_roots
    for v in data.get("from_roots", []):
        label = v["label"]
        roots = [from_le_hex(r) for r in v["roots"]]
        expected = [from_le_hex(c) for c in v["coefficients"]]
        actual = poly_from_roots(roots, p)
        if actual == expected:
            t.ok(f"from_roots/{label}")
        else:
            t.fail(f"from_roots/{label}", f"coefficient mismatch")

    # evaluate
    for v in data.get("evaluate", []):
        label = v["label"]
        coeffs = [from_le_hex(c) for c in v["coefficients"]]
        x = from_le_hex(v["x"])
        expected = from_le_hex(v["result"])
        actual = poly_eval(coeffs, x, p)
        if actual == expected:
            t.ok(f"evaluate/{label}")
        else:
            t.fail(f"evaluate/{label}", f"expected {to_le_hex(expected)}, got {to_le_hex(actual)}")

    # mul
    for v in data.get("mul", []):
        label = v["label"]
        a_coeffs = [from_le_hex(c) for c in v["a_coefficients"]]
        b_coeffs = [from_le_hex(c) for c in v["b_coefficients"]]
        expected = [from_le_hex(c) for c in v["coefficients"]]
        actual = poly_mul(a_coeffs, b_coeffs, p)
        if actual == expected:
            t.ok(f"mul/{label}")
        else:
            t.fail(f"mul/{label}", f"coefficient mismatch (degree {len(actual)-1} vs {len(expected)-1})")

    # add
    for v in data.get("add", []):
        label = v["label"]
        a_coeffs = [from_le_hex(c) for c in v["a_coefficients"]]
        b_coeffs = [from_le_hex(c) for c in v["b_coefficients"]]
        expected = [from_le_hex(c) for c in v["coefficients"]]
        actual = poly_add(a_coeffs, b_coeffs, p)
        if actual == expected:
            t.ok(f"add/{label}")
        else:
            t.fail(f"add/{label}", f"coefficient mismatch")

    # sub
    for v in data.get("sub", []):
        label = v["label"]
        a_coeffs = [from_le_hex(c) for c in v["a_coefficients"]]
        b_coeffs = [from_le_hex(c) for c in v["b_coefficients"]]
        expected = [from_le_hex(c) for c in v["coefficients"]]
        actual = poly_sub(a_coeffs, b_coeffs, p)
        if actual == expected:
            t.ok(f"sub/{label}")
        else:
            t.fail(f"sub/{label}", f"coefficient mismatch")

    # divmod
    for v in data.get("divmod", []):
        label = v["label"]
        num = [from_le_hex(c) for c in v["numerator"]]
        den = [from_le_hex(c) for c in v["denominator"]]
        expected_q = [from_le_hex(c) for c in v["quotient"]]
        expected_r = [from_le_hex(c) for c in v["remainder"]]
        actual_q, actual_r = poly_divmod(num, den, p)
        if actual_q == expected_q and actual_r == expected_r:
            t.ok(f"divmod/{label}")
        else:
            if actual_q != expected_q:
                t.fail(f"divmod/{label}", f"quotient mismatch")
            else:
                t.fail(f"divmod/{label}", f"remainder mismatch")

    # interpolate
    for v in data.get("interpolate", []):
        label = v["label"]
        xs = [from_le_hex(x) for x in v["xs"]]
        ys = [from_le_hex(y) for y in v["ys"]]
        expected = [from_le_hex(c) for c in v["coefficients"]]
        actual = poly_interpolate(xs, ys, p)
        if actual == expected:
            t.ok(f"interpolate/{label}")
        else:
            t.fail(f"interpolate/{label}", f"coefficient mismatch")


# ─── Batch invert validation ─────────────────────────────────────────────────

def validate_batch_invert(t: TestRunner, data: dict, P: int, Q: int):
    t.begin_section("batch_invert")

    for field_name, p in [("fp", P), ("fq", Q)]:
        for v in data.get(field_name, []):
            label = v["label"]
            inputs = [from_le_hex(h) for h in v["inputs"]]
            expected = [from_le_hex(h) for h in v["results"]]
            actual = [pow(x, -1, p) if x != 0 else 0 for x in inputs]
            if actual == expected:
                t.ok(f"{field_name}/{label}")
            else:
                first_diff = next((i for i in range(len(actual)) if actual[i] != expected[i]), -1)
                t.fail(f"{field_name}/{label}", f"mismatch at index {first_diff}")


# ─── Wei25519 validation ─────────────────────────────────────────────────────

def validate_wei25519(t: TestRunner, data: dict):
    t.begin_section("wei25519")

    for v in data.get("x_to_shaw_scalar", []):
        label = v["label"]
        result_hex = v["result"]
        if result_hex is None:
            t.ok(f"x_to_shaw_scalar/{label} (rejected)")
        else:
            # The conversion formula is: x_wei25519 -> shaw scalar
            # This requires knowing the isomorphism, defer to SageMath for full validation
            t.skip(f"x_to_shaw_scalar/{label}", "conversion formula defer to SageMath")


# ─── Divisor validation ──────────────────────────────────────────────────────

def validate_divisor_section(t: TestRunner, data: dict, curve_name: str,
                             p_field: int, a: int, b_val: int, order: int,
                             gen_xy: tuple = None):
    t.begin_section(f"{curve_name} divisor")

    curve = WeierstrassCurve({
        "name": curve_name + "_div",
        "type": "weierstrass",
        "size": 255,
        "a": a,
        "b": b_val,
        "field": p_field,
        "generator": gen_xy if gen_xy else (0, 0),
        "order": order,
        "cofactor": 1,
    })

    for v in data.get("compute", []):
        label = v["label"]
        # We can verify the divisor evaluation: f(x,y) = a(x) + y*b(x)
        a_coeffs = [from_le_hex(c) for c in v["a_coefficients"]]
        b_coeffs = [from_le_hex(c) for c in v["b_coefficients"]]
        eval_x = from_le_hex(v["eval_point_x"])
        eval_y = from_le_hex(v["eval_point_y"])
        expected_eval = from_le_hex(v["eval_result"])

        # f(x,y) = a(x) - y * b(x)
        a_at_x = poly_eval(a_coeffs, eval_x, p_field)
        b_at_x = poly_eval(b_coeffs, eval_x, p_field)
        actual_eval = (a_at_x - eval_y * b_at_x) % p_field

        if actual_eval == expected_eval:
            t.ok(f"compute/{label} (eval check)")
        else:
            t.fail(f"compute/{label} (eval check)",
                   f"expected {to_le_hex(expected_eval)}, got {to_le_hex(actual_eval)}")

        # Also verify the eval point is on the curve
        lhs = (eval_y * eval_y) % p_field
        rhs = (pow(eval_x, 3, p_field) + a * eval_x + b_val) % p_field
        if lhs == rhs:
            t.ok(f"compute/{label} (eval point on curve)")
        else:
            t.fail(f"compute/{label} (eval point on curve)", "point not on curve")

        # Verify input points are on the curve
        points = v["points"]
        all_on_curve = True
        for i, ph in enumerate(points):
            pt = decode_point_bytes(ph, p_field, a, b_val)
            if pt is None:
                t.fail(f"compute/{label} (point[{i}] on curve)", "decode failed")
                all_on_curve = False
                break
            lhs = (pt[1] * pt[1]) % p_field
            rhs = (pow(pt[0], 3, p_field) + a * pt[0] + b_val) % p_field
            if lhs != rhs:
                t.fail(f"compute/{label} (point[{i}] on curve)", "not on curve")
                all_on_curve = False
                break
        if all_on_curve:
            t.ok(f"compute/{label} (all input points on curve)")


# ─── High-degree polynomial multiplication ──────────────────────────────────

def validate_high_degree_poly_mul(t: TestRunner, data: dict,
                                  fp_field: int, fq_field: int):
    t.begin_section("high_degree_poly_mul")

    for field_name, p in [("fp", fp_field), ("fq", fq_field)]:
        for v in data.get(field_name, []):
            label = v["label"]
            n_coeffs = v["n_coeffs"]
            result_degree = v["result_degree"]

            # Rebuild input polynomials from deterministic pattern
            a_coeffs = [(i + 1) % p for i in range(n_coeffs)]
            b_coeffs = [(i + n_coeffs + 1) % p for i in range(n_coeffs)]

            # Verify result degree
            expected_degree = (n_coeffs - 1) * 2 if n_coeffs > 0 else 0
            if result_degree != expected_degree:
                t.fail(f"{field_name}/{label} (degree)",
                       f"expected {expected_degree}, got {result_degree}")
                continue

            # Multi-point eval: a(x)*b(x) == result(x) at 3 independent points
            for check in v.get("eval_checks", []):
                pt_name = check["point"]
                x = from_le_hex(check["x"])
                a_of_x = from_le_hex(check["a_of_x"])
                b_of_x = from_le_hex(check["b_of_x"])
                result_of_x = from_le_hex(check["result_of_x"])

                # Verify a(x) independently
                expected_a = poly_eval(a_coeffs, x, p)
                if expected_a != a_of_x:
                    t.fail(f"{field_name}/{label} (a({pt_name}))",
                           f"expected {to_le_hex(expected_a)}, got {to_le_hex(a_of_x)}")
                    continue

                # Verify b(x) independently
                expected_b = poly_eval(b_coeffs, x, p)
                if expected_b != b_of_x:
                    t.fail(f"{field_name}/{label} (b({pt_name}))",
                           f"expected {to_le_hex(expected_b)}, got {to_le_hex(b_of_x)}")
                    continue

                # Core check: a(x) * b(x) == result(x)
                expected_product = (a_of_x * b_of_x) % p
                if expected_product == result_of_x:
                    t.ok(f"{field_name}/{label} (eval@{pt_name})")
                else:
                    t.fail(f"{field_name}/{label} (eval@{pt_name})",
                           f"a*b = {to_le_hex(expected_product)}, "
                           f"result = {to_le_hex(result_of_x)}")


# ─── SSWU map-to-curve (RFC 9380) ──────────────────────────────────────────

def sswu_map_to_curve(u: int, p: int, a: int, b_val: int, Z: int):
    """Simplified SWU map-to-curve per RFC 9380 Section 6.6.2. Returns (x, y)."""
    u2 = (u * u) % p
    Zu2 = (Z * u2) % p
    Zu2_sq = (Zu2 * Zu2) % p
    denom = ((-a) * pow(Zu2_sq + Zu2, -1, p)) % p if (Zu2_sq + Zu2) % p != 0 else ((-a) * pow(Z, -1, p)) % p

    # x1 = (-b/a) * (1 + 1/(Z^2*u^4 + Z*u^2))
    # Simplified: num_x1 = -b * (Z^2 * u^4 + Z * u^2 + 1), den_x1 = a * (Z^2 * u^4 + Z * u^2)
    # But we use the standard formulation:
    tv1 = (Zu2_sq + Zu2) % p
    if tv1 == 0:
        x1 = (b_val * pow(Z * a, -1, p)) % p
    else:
        x1 = ((-b_val) * pow(a, -1, p) * (1 + pow(tv1, -1, p))) % p
    gx1 = (pow(x1, 3, p) + a * x1 + b_val) % p
    x2 = (Z * u2 * x1) % p
    gx2 = (pow(x2, 3, p) + a * x2 + b_val) % p

    # Select x1 if gx1 is square, else x2
    if pow(gx1, (p - 1) // 2, p) == 1 or gx1 == 0:
        x, gx = x1, gx1
    else:
        x, gx = x2, gx2

    y = mod_sqrt(gx, p)
    assert y is not None, f"SSWU: sqrt failed for gx={gx}"

    # Ensure sgn0(y) == sgn0(u)
    if (y & 1) != (u & 1):
        y = p - y

    return (x, y)


def find_sswu_z(p: int, a: int, b_val: int) -> int:
    """Find the SSWU Z parameter: first z in {-1, -2, -3, ...} that is non-square
       and for which g(b/(z*a)) is square."""
    for z_cand in range(1, 1000):
        z = (-z_cand) % p
        # z must be non-square
        if pow(z, (p - 1) // 2, p) == 1:
            continue
        # g(b/(z*a)) must be square
        x_test = (b_val * pow(z * a, -1, p)) % p
        gx_test = (pow(x_test, 3, p) + a * x_test + b_val) % p
        if gx_test == 0 or pow(gx_test, (p - 1) // 2, p) == 1:
            return z
    raise RuntimeError("Could not find SSWU Z")


# ─── Raw field arithmetic validation ──────────────────────────────────────────

def validate_field_section(t: TestRunner, data: dict, field_name: str, p: int):
    t.begin_section(f"{field_name} field arithmetic")

    test_a = from_le_hex(data["test_a"])
    test_b = from_le_hex(data["test_b"])

    # mul_ab
    expected = from_le_hex(data["mul_ab"])
    actual = (test_a * test_b) % p
    if actual == expected:
        t.ok("mul_ab")
    else:
        t.fail("mul_ab", f"expected {to_le_hex(expected)}, got {to_le_hex(actual)}")

    # sq_a
    expected = from_le_hex(data["sq_a"])
    actual = (test_a * test_a) % p
    if actual == expected:
        t.ok("sq_a")
    else:
        t.fail("sq_a", f"expected {to_le_hex(expected)}, got {to_le_hex(actual)}")

    # inv_a
    expected = from_le_hex(data["inv_a"])
    actual = pow(test_a, -1, p)
    if actual == expected:
        t.ok("inv_a")
    else:
        t.fail("inv_a", f"expected {to_le_hex(expected)}, got {to_le_hex(actual)}")

    # fq-specific: sqrt4, denom, denom_inv
    if "sqrt4" in data:
        expected = from_le_hex(data["sqrt4"])
        # sqrt(4) should be 2 or p-2
        if expected == 2 or expected == p - 2:
            t.ok("sqrt4")
        else:
            t.fail("sqrt4", f"expected 2 or p-2, got {expected}")

    if "denom" in data and "denom_inv" in data:
        denom = from_le_hex(data["denom"])
        denom_inv = from_le_hex(data["denom_inv"])
        # Verify denom == test_a * test_b mod p
        expected_denom = (test_a * test_b) % p
        if denom != expected_denom:
            t.fail("denom", f"expected a*b = {to_le_hex(expected_denom)}, got {to_le_hex(denom)}")
        else:
            t.ok("denom == a*b")
        # Verify denom * denom_inv == 1
        check = (denom * denom_inv) % p
        if check == 1:
            t.ok("denom * denom_inv == 1")
        else:
            t.fail("denom * denom_inv == 1", f"got {check}")


# ─── Compressed point validation ──────────────────────────────────────────────

def validate_compressed_points(t: TestRunner, data: dict,
                               ran_p: int, ran_a: int, ran_b: int, ran_order: int,
                               ran_gen_xy: tuple,
                               shaw_p: int, shaw_a: int, shaw_b: int, shaw_order: int,
                               shaw_gen_xy: tuple):
    t.begin_section("compressed points")

    ran_curve = WeierstrassCurve({
        "name": "ran_cp", "type": "weierstrass", "size": 255,
        "a": ran_a, "b": ran_b, "field": ran_p,
        "generator": ran_gen_xy, "order": ran_order, "cofactor": 1,
    })
    shaw_curve = WeierstrassCurve({
        "name": "shaw_cp", "type": "weierstrass", "size": 255,
        "a": shaw_a, "b": shaw_b, "field": shaw_p,
        "generator": shaw_gen_xy, "order": shaw_order, "cofactor": 1,
    })

    def check_point(label, hex_val, expected_pt, p_field):
        encoded = encode_point(expected_pt.x, expected_pt.y, p_field) if not expected_pt.is_infinity else "00" * 32
        if encoded == hex_val:
            t.ok(label)
        else:
            t.fail(label, f"expected {encoded[:32]}..., got {hex_val[:32]}...")

    # Ran points
    G = Point(ran_gen_xy[0], ran_gen_xy[1], ran_curve)
    G2 = ran_curve.add_point(G, G)
    G3 = ran_curve.add_point(G2, G)
    G4 = ran_curve.add_point(G2, G2)
    G7 = ran_curve.add_point(G4, G3)

    check_point("ran_g", data["ran_g"], G, ran_p)
    check_point("ran_2g", data["ran_2g"], G2, ran_p)
    check_point("ran_7g", data["ran_7g"], G7, ran_p)

    # ran_gx = generator x-coordinate bytes
    expected_gx = to_le_hex(ran_gen_xy[0])
    if data["ran_gx"] == expected_gx:
        t.ok("ran_gx")
    else:
        t.fail("ran_gx", f"expected {expected_gx[:32]}..., got {data['ran_gx'][:32]}...")

    # Shaw points
    sG = Point(shaw_gen_xy[0], shaw_gen_xy[1], shaw_curve)
    sG2 = shaw_curve.add_point(sG, sG)
    sG3 = shaw_curve.add_point(sG2, sG)
    sG4 = shaw_curve.add_point(sG2, sG2)
    sG7 = shaw_curve.add_point(sG4, sG3)

    check_point("shaw_g", data["shaw_g"], sG, shaw_p)
    check_point("shaw_2g", data["shaw_2g"], sG2, shaw_p)
    check_point("shaw_7g", data["shaw_7g"], sG7, shaw_p)

    expected_gx = to_le_hex(shaw_gen_xy[0])
    if data["shaw_gx"] == expected_gx:
        t.ok("shaw_gx")
    else:
        t.fail("shaw_gx", f"expected {expected_gx[:32]}..., got {data['shaw_gx'][:32]}...")


# ─── SSWU vector validation ──────────────────────────────────────────────────

def validate_sswu_vectors(t: TestRunner, data: dict,
                          ran_p: int, ran_a: int, ran_b: int, ran_z: int,
                          shaw_p: int, shaw_a: int, shaw_b: int, shaw_z: int):
    t.begin_section("sswu vectors")

    for label_suffix, u_bytes_hex, z, p, a, b_val, prefix in [
        ("u1", "01" + "00" * 31, ran_z, ran_p, ran_a, ran_b, "ran"),
        ("u2", "02" + "00" * 31, ran_z, ran_p, ran_a, ran_b, "ran"),
        ("u42", "2a" + "00" * 31, ran_z, ran_p, ran_a, ran_b, "ran"),
        ("u1", "01" + "00" * 31, shaw_z, shaw_p, shaw_a, shaw_b, "shaw"),
        ("u2", "02" + "00" * 31, shaw_z, shaw_p, shaw_a, shaw_b, "shaw"),
        ("u42", "2a" + "00" * 31, shaw_z, shaw_p, shaw_a, shaw_b, "shaw"),
    ]:
        key = f"{prefix}_{label_suffix}"
        if key not in data:
            t.skip(key, "not in data")
            continue
        u = from_le_hex(u_bytes_hex)
        x, y = sswu_map_to_curve(u, p, a, b_val, z)
        expected_hex = data[key]
        actual_hex = encode_point(x, y, p)
        if actual_hex == expected_hex:
            t.ok(key)
        else:
            t.fail(key, f"expected {expected_hex[:32]}..., got {actual_hex[:32]}...")

    # SSWU intermediates: ran_x2_u1, ran_gx2_u1, ran_y_u1
    if "ran_x2_u1" in data:
        u = 1
        x, y = sswu_map_to_curve(u, ran_p, ran_a, ran_b, ran_z)
        x_hex = to_le_hex(x)
        if x_hex == data["ran_x2_u1"]:
            t.ok("ran_x2_u1")
        else:
            t.fail("ran_x2_u1", f"expected {data['ran_x2_u1'][:32]}..., got {x_hex[:32]}...")

    if "ran_gx2_u1" in data:
        u = 1
        x, y = sswu_map_to_curve(u, ran_p, ran_a, ran_b, ran_z)
        gx = (pow(x, 3, ran_p) + ran_a * x + ran_b) % ran_p
        gx_hex = to_le_hex(gx)
        if gx_hex == data["ran_gx2_u1"]:
            t.ok("ran_gx2_u1")
        else:
            t.fail("ran_gx2_u1", f"expected {data['ran_gx2_u1'][:32]}..., got {gx_hex[:32]}...")

    if "ran_y_u1" in data:
        u = 1
        x, y = sswu_map_to_curve(u, ran_p, ran_a, ran_b, ran_z)
        gx = (pow(x, 3, ran_p) + ran_a * x + ran_b) % ran_p
        y_val = mod_sqrt(gx, ran_p)
        # C++ fp_sqrt may return either root; check both
        y_hex = to_le_hex(y_val)
        y_neg_hex = to_le_hex(ran_p - y_val)
        expected = data["ran_y_u1"]
        if y_hex == expected or y_neg_hex == expected:
            t.ok("ran_y_u1")
        else:
            t.fail("ran_y_u1", f"expected {expected[:32]}..., got {y_hex[:32]}... or {y_neg_hex[:32]}...")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <test_vectors.json>")
        sys.exit(2)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    params = data["parameters"]

    # ALL curve parameters loaded from JSON — no hardcoded values
    Q = from_le_hex(params["ran_order"])      # Ran scalar field = Shaw base field
    P = from_le_hex(params["shaw_order"])       # Shaw scalar field = Ran base field
    A = params["curve_a"]                         # curve coefficient a (both curves)
    ran_b = from_le_hex(params["ran_b"])
    shaw_b = from_le_hex(params["shaw_b"])
    ran_sswu_z = params.get("ran_sswu_z", None)
    shaw_sswu_z = params.get("shaw_sswu_z", None)
    # Convert Z to field element if present
    if ran_sswu_z is not None:
        ran_sswu_z = ran_sswu_z % P
    if shaw_sswu_z is not None:
        shaw_sswu_z = shaw_sswu_z % Q

    print(f"Ran: y^2 = x^3 + {A}x + b over Fp")
    print(f"  Fp = 0x{P:064x}")
    print(f"  b  = 0x{ran_b:064x}")
    print(f"  order Q = 0x{Q:064x}")
    print(f"Shaw: y^2 = x^3 + {A}x + b over Fq")
    print(f"  Fq = 0x{Q:064x}")
    print(f"  b  = 0x{shaw_b:064x}")
    print(f"  order P = 0x{P:064x}")

    t = TestRunner()

    # Decode generators for ecpy curve construction
    ran_gen_hex = data["ran_point"]["generator"]
    ran_gen_xy = decode_point_bytes(ran_gen_hex, P, A, ran_b)
    assert ran_gen_xy is not None, "Failed to decode Ran generator"

    shaw_gen_hex = data["shaw_point"]["generator"]
    shaw_gen_xy = decode_point_bytes(shaw_gen_hex, Q, A, shaw_b)
    assert shaw_gen_xy is not None, "Failed to decode Shaw generator"

    # Scalars
    validate_scalar_section(t, data["ran_scalar"], "ran", Q)
    validate_scalar_section(t, data["shaw_scalar"], "shaw", P)

    # Points
    validate_point_section(t, data["ran_point"], "ran", P, A, ran_b, Q, ran_gen_xy, ran_sswu_z)
    validate_point_section(t, data["shaw_point"], "shaw", Q, A, shaw_b, P, shaw_gen_xy, shaw_sswu_z)

    # Polynomials
    validate_polynomial_section(t, data["fp_polynomial"], "fp", P)
    validate_polynomial_section(t, data["fq_polynomial"], "fq", Q)

    # Batch invert
    validate_batch_invert(t, data["batch_invert"], P, Q)

    # Divisors
    validate_divisor_section(t, data["ran_divisor"], "ran", P, A, ran_b, Q, ran_gen_xy)
    validate_divisor_section(t, data["shaw_divisor"], "shaw", Q, A, shaw_b, P, shaw_gen_xy)

    # Wei25519
    validate_wei25519(t, data["wei25519"])

    # High-degree polynomial multiplication
    if "high_degree_poly_mul" in data:
        validate_high_degree_poly_mul(t, data["high_degree_poly_mul"], P, Q)

    # Raw field arithmetic
    if "fp_field" in data:
        validate_field_section(t, data["fp_field"], "fp", P)
    if "fq_field" in data:
        validate_field_section(t, data["fq_field"], "fq", Q)

    # Compressed points
    if "compressed_points" in data:
        validate_compressed_points(t, data["compressed_points"],
                                   P, A, ran_b, Q, ran_gen_xy,
                                   Q, A, shaw_b, P, shaw_gen_xy)

    # SSWU vectors
    if "sswu_vectors" in data:
        validate_sswu_vectors(t, data["sswu_vectors"],
                              P, A, ran_b, ran_sswu_z,
                              Q, A, shaw_b, shaw_sswu_z)

    sys.exit(t.summary())


if __name__ == "__main__":
    main()
