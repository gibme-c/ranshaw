#!/usr/bin/env python3
"""
Convert ranshaw test vectors JSON to a C++ header file.

Usage:
  python json_to_header.py test_vectors/ranshaw_test_vectors.json include/ranshaw_test_vectors.h
"""

import json
import sys
import textwrap


def hex_to_c_array(hex_str):
    """Convert a hex string to a C byte array initializer."""
    bs = bytes.fromhex(hex_str)
    parts = [f"0x{b:02x}" for b in bs]
    # Wrap at 12 bytes per line
    lines = []
    for i in range(0, len(parts), 12):
        lines.append(", ".join(parts[i:i+12]))
    return "{" + ", ".join(lines) + "}"


def emit_scalar_section(f, section, ns_name):
    """Emit test vector structs for a scalar section."""
    f.write(f"namespace {ns_name} {{\n\n")

    for op_name in ["from_bytes", "add", "sub", "mul", "sq", "negate", "invert",
                     "reduce_wide", "muladd", "is_zero"]:
        vectors = section.get(op_name, [])
        if not vectors:
            continue

        if op_name == "from_bytes":
            f.write(f"struct from_bytes_vector {{ const char *label; uint8_t input[32]; bool valid; uint8_t result[32]; }};\n")
            f.write(f"static const from_bytes_vector from_bytes_vectors[] = {{\n")
            for v in vectors:
                valid = v["result"] is not None
                result = hex_to_c_array(v["result"]) if valid else hex_to_c_array("00" * 32)
                f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['input'])}, {'true' if valid else 'false'}, {result}}},\n")
            f.write(f"}};\n")
            f.write(f"static const size_t from_bytes_count = sizeof(from_bytes_vectors) / sizeof(from_bytes_vectors[0]);\n\n")

        elif op_name in ["add", "sub", "mul"]:
            f.write(f"struct {op_name}_vector {{ const char *label; uint8_t a[32]; uint8_t b[32]; uint8_t result[32]; }};\n")
            f.write(f"static const {op_name}_vector {op_name}_vectors[] = {{\n")
            for v in vectors:
                f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['a'])}, {hex_to_c_array(v['b'])}, {hex_to_c_array(v['result'])}}},\n")
            f.write(f"}};\n")
            f.write(f"static const size_t {op_name}_count = sizeof({op_name}_vectors) / sizeof({op_name}_vectors[0]);\n\n")

        elif op_name in ["sq", "negate"]:
            f.write(f"struct {op_name}_vector {{ const char *label; uint8_t a[32]; uint8_t result[32]; }};\n")
            f.write(f"static const {op_name}_vector {op_name}_vectors[] = {{\n")
            for v in vectors:
                f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['a'])}, {hex_to_c_array(v['result'])}}},\n")
            f.write(f"}};\n")
            f.write(f"static const size_t {op_name}_count = sizeof({op_name}_vectors) / sizeof({op_name}_vectors[0]);\n\n")

        elif op_name == "invert":
            f.write(f"struct invert_vector {{ const char *label; uint8_t a[32]; bool valid; uint8_t result[32]; }};\n")
            f.write(f"static const invert_vector invert_vectors[] = {{\n")
            for v in vectors:
                valid = v["result"] is not None
                result = hex_to_c_array(v["result"]) if valid else hex_to_c_array("00" * 32)
                f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['a'])}, {'true' if valid else 'false'}, {result}}},\n")
            f.write(f"}};\n")
            f.write(f"static const size_t invert_count = sizeof(invert_vectors) / sizeof(invert_vectors[0]);\n\n")

        elif op_name == "reduce_wide":
            f.write(f"struct reduce_wide_vector {{ const char *label; uint8_t input[64]; uint8_t result[32]; }};\n")
            f.write(f"static const reduce_wide_vector reduce_wide_vectors[] = {{\n")
            for v in vectors:
                f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['input'])}, {hex_to_c_array(v['result'])}}},\n")
            f.write(f"}};\n")
            f.write(f"static const size_t reduce_wide_count = sizeof(reduce_wide_vectors) / sizeof(reduce_wide_vectors[0]);\n\n")

        elif op_name == "muladd":
            f.write(f"struct muladd_vector {{ const char *label; uint8_t a[32]; uint8_t b[32]; uint8_t c[32]; uint8_t result[32]; }};\n")
            f.write(f"static const muladd_vector muladd_vectors[] = {{\n")
            for v in vectors:
                f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['a'])}, {hex_to_c_array(v['b'])}, {hex_to_c_array(v['c'])}, {hex_to_c_array(v['result'])}}},\n")
            f.write(f"}};\n")
            f.write(f"static const size_t muladd_count = sizeof(muladd_vectors) / sizeof(muladd_vectors[0]);\n\n")

        elif op_name == "is_zero":
            f.write(f"struct is_zero_vector {{ const char *label; uint8_t a[32]; bool result; }};\n")
            f.write(f"static const is_zero_vector is_zero_vectors[] = {{\n")
            for v in vectors:
                f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['a'])}, {'true' if v['result'] else 'false'}}},\n")
            f.write(f"}};\n")
            f.write(f"static const size_t is_zero_count = sizeof(is_zero_vectors) / sizeof(is_zero_vectors[0]);\n\n")

    f.write(f"}} // namespace {ns_name}\n\n")


def emit_point_section(f, section, ns_name):
    """Emit test vector structs for a point section."""
    f.write(f"namespace {ns_name} {{\n\n")

    # generator & identity
    if "generator" in section:
        f.write(f"static const uint8_t generator[32] = {hex_to_c_array(section['generator'])};\n")
    if "identity" in section:
        f.write(f"static const uint8_t identity[32] = {hex_to_c_array(section['identity'])};\n\n")

    # from_bytes
    vectors = section.get("from_bytes", [])
    if vectors:
        f.write(f"struct from_bytes_vector {{ const char *label; uint8_t input[32]; bool valid; uint8_t result[32]; }};\n")
        f.write(f"static const from_bytes_vector from_bytes_vectors[] = {{\n")
        for v in vectors:
            valid = v["result"] is not None
            result = hex_to_c_array(v["result"]) if valid else hex_to_c_array("00" * 32)
            f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['input'])}, {'true' if valid else 'false'}, {result}}},\n")
        f.write(f"}};\n")
        f.write(f"static const size_t from_bytes_count = sizeof(from_bytes_vectors) / sizeof(from_bytes_vectors[0]);\n\n")

    # add, dbl, negate
    for op_name in ["add", "dbl", "negate"]:
        vectors = section.get(op_name, [])
        if not vectors:
            continue
        if op_name == "add":
            f.write(f"struct add_vector {{ const char *label; uint8_t a[32]; uint8_t b[32]; uint8_t result[32]; }};\n")
            f.write(f"static const add_vector add_vectors[] = {{\n")
            for v in vectors:
                f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['a'])}, {hex_to_c_array(v['b'])}, {hex_to_c_array(v['result'])}}},\n")
        else:
            f.write(f"struct {op_name}_vector {{ const char *label; uint8_t a[32]; uint8_t result[32]; }};\n")
            f.write(f"static const {op_name}_vector {op_name}_vectors[] = {{\n")
            for v in vectors:
                f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['a'])}, {hex_to_c_array(v['result'])}}},\n")
        f.write(f"}};\n")
        f.write(f"static const size_t {op_name}_count = sizeof({op_name}_vectors) / sizeof({op_name}_vectors[0]);\n\n")

    # scalar_mul
    vectors = section.get("scalar_mul", [])
    if vectors:
        f.write(f"struct scalar_mul_vector {{ const char *label; uint8_t scalar[32]; uint8_t point[32]; uint8_t result[32]; }};\n")
        f.write(f"static const scalar_mul_vector scalar_mul_vectors[] = {{\n")
        for v in vectors:
            f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['scalar'])}, {hex_to_c_array(v['point'])}, {hex_to_c_array(v['result'])}}},\n")
        f.write(f"}};\n")
        f.write(f"static const size_t scalar_mul_count = sizeof(scalar_mul_vectors) / sizeof(scalar_mul_vectors[0]);\n\n")

    # x_coordinate
    vectors = section.get("x_coordinate", [])
    if vectors:
        f.write(f"struct x_coordinate_vector {{ const char *label; uint8_t point[32]; uint8_t x_bytes[32]; }};\n")
        f.write(f"static const x_coordinate_vector x_coordinate_vectors[] = {{\n")
        for v in vectors:
            f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['point'])}, {hex_to_c_array(v['x_bytes'])}}},\n")
        f.write(f"}};\n")
        f.write(f"static const size_t x_coordinate_count = sizeof(x_coordinate_vectors) / sizeof(x_coordinate_vectors[0]);\n\n")

    # MSM vectors stored separately (variable-length arrays)
    vectors = section.get("msm", [])
    if vectors:
        for v in vectors:
            label = v["label"]
            n = v["n"]
            f.write(f"static const uint8_t msm_{label}_scalars[{n}][32] = {{\n")
            for s in v["scalars"]:
                f.write(f"  {hex_to_c_array(s)},\n")
            f.write(f"}};\n")
            f.write(f"static const uint8_t msm_{label}_points[{n}][32] = {{\n")
            for pt in v["points"]:
                f.write(f"  {hex_to_c_array(pt)},\n")
            f.write(f"}};\n")
            f.write(f"static const uint8_t msm_{label}_result[32] = {hex_to_c_array(v['result'])};\n\n")

    # Pedersen vectors
    vectors = section.get("pedersen_commit", [])
    if vectors:
        for v in vectors:
            label = v["label"]
            n = v["n"]
            f.write(f"static const uint8_t pedersen_{label}_blinding[32] = {hex_to_c_array(v['blinding'])};\n")
            f.write(f"static const uint8_t pedersen_{label}_H[32] = {hex_to_c_array(v['H'])};\n")
            f.write(f"static const uint8_t pedersen_{label}_values[{n}][32] = {{\n")
            for val in v["values"]:
                f.write(f"  {hex_to_c_array(val)},\n")
            f.write(f"}};\n")
            f.write(f"static const uint8_t pedersen_{label}_generators[{n}][32] = {{\n")
            for gen in v["generators"]:
                f.write(f"  {hex_to_c_array(gen)},\n")
            f.write(f"}};\n")
            f.write(f"static const uint8_t pedersen_{label}_result[32] = {hex_to_c_array(v['result'])};\n\n")

    # map_to_curve_single
    vectors = section.get("map_to_curve_single", [])
    if vectors:
        f.write(f"struct map_to_curve_single_vector {{ const char *label; uint8_t u[32]; uint8_t result[32]; }};\n")
        f.write(f"static const map_to_curve_single_vector map_to_curve_single_vectors[] = {{\n")
        for v in vectors:
            f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['u'])}, {hex_to_c_array(v['result'])}}},\n")
        f.write(f"}};\n")
        f.write(f"static const size_t map_to_curve_single_count = sizeof(map_to_curve_single_vectors) / sizeof(map_to_curve_single_vectors[0]);\n\n")

    # map_to_curve_double
    vectors = section.get("map_to_curve_double", [])
    if vectors:
        f.write(f"struct map_to_curve_double_vector {{ const char *label; uint8_t u0[32]; uint8_t u1[32]; uint8_t result[32]; }};\n")
        f.write(f"static const map_to_curve_double_vector map_to_curve_double_vectors[] = {{\n")
        for v in vectors:
            f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['u0'])}, {hex_to_c_array(v['u1'])}, {hex_to_c_array(v['result'])}}},\n")
        f.write(f"}};\n")
        f.write(f"static const size_t map_to_curve_double_count = sizeof(map_to_curve_double_vectors) / sizeof(map_to_curve_double_vectors[0]);\n\n")

    f.write(f"}} // namespace {ns_name}\n\n")


def emit_polynomial_section(f, section, ns_name):
    """Emit test vector structs for a polynomial section."""
    f.write(f"namespace {ns_name} {{\n\n")

    # from_roots: variable-length arrays
    vectors = section.get("from_roots", [])
    for v in vectors:
        label = v["label"]
        n_roots = v["n"]
        roots = v["roots"]
        coeffs = v["coefficients"]
        n_coeffs = len(coeffs)
        f.write(f"static const uint8_t from_roots_{label}_roots[{n_roots}][32] = {{\n")
        for r in roots:
            f.write(f"  {hex_to_c_array(r)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t from_roots_{label}_coefficients[{n_coeffs}][32] = {{\n")
        for c in coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n\n")

    # evaluate
    vectors = section.get("evaluate", [])
    for v in vectors:
        label = v["label"]
        coeffs = v["coefficients"]
        n_coeffs = len(coeffs)
        f.write(f"static const uint8_t eval_{label}_coefficients[{n_coeffs}][32] = {{\n")
        for c in coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t eval_{label}_x[32] = {hex_to_c_array(v['x'])};\n")
        f.write(f"static const uint8_t eval_{label}_result[32] = {hex_to_c_array(v['result'])};\n\n")

    # mul
    vectors = section.get("mul", [])
    for v in vectors:
        label = v["label"]
        a_coeffs = v["a_coefficients"]
        b_coeffs = v["b_coefficients"]
        r_coeffs = v["coefficients"]
        f.write(f"static const uint8_t mul_{label}_a[{len(a_coeffs)}][32] = {{\n")
        for c in a_coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t mul_{label}_b[{len(b_coeffs)}][32] = {{\n")
        for c in b_coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t mul_{label}_result[{len(r_coeffs)}][32] = {{\n")
        for c in r_coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n\n")

    # add
    vectors = section.get("add", [])
    for v in vectors:
        label = v["label"]
        a_coeffs = v["a_coefficients"]
        b_coeffs = v["b_coefficients"]
        r_coeffs = v["coefficients"]
        f.write(f"static const uint8_t add_{label}_a[{len(a_coeffs)}][32] = {{\n")
        for c in a_coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t add_{label}_b[{len(b_coeffs)}][32] = {{\n")
        for c in b_coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t add_{label}_result[{len(r_coeffs)}][32] = {{\n")
        for c in r_coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n\n")

    # sub
    vectors = section.get("sub", [])
    for v in vectors:
        label = v["label"]
        a_coeffs = v["a_coefficients"]
        b_coeffs = v["b_coefficients"]
        r_coeffs = v["coefficients"]
        f.write(f"static const uint8_t sub_{label}_a[{len(a_coeffs)}][32] = {{\n")
        for c in a_coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t sub_{label}_b[{len(b_coeffs)}][32] = {{\n")
        for c in b_coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t sub_{label}_result[{len(r_coeffs)}][32] = {{\n")
        for c in r_coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n\n")

    # divmod
    vectors = section.get("divmod", [])
    for v in vectors:
        label = v["label"]
        num = v["numerator"]
        den = v["denominator"]
        quot = v["quotient"]
        rem = v["remainder"]
        f.write(f"static const uint8_t divmod_{label}_numerator[{len(num)}][32] = {{\n")
        for c in num:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t divmod_{label}_denominator[{len(den)}][32] = {{\n")
        for c in den:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t divmod_{label}_quotient[{len(quot)}][32] = {{\n")
        for c in quot:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t divmod_{label}_remainder[{len(rem)}][32] = {{\n")
        for c in rem:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n\n")

    # interpolate
    vectors = section.get("interpolate", [])
    for v in vectors:
        label = v["label"]
        n = v["n"]
        xs = v["xs"]
        ys = v["ys"]
        coeffs = v["coefficients"]
        f.write(f"static const uint8_t interp_{label}_xs[{n}][32] = {{\n")
        for x in xs:
            f.write(f"  {hex_to_c_array(x)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t interp_{label}_ys[{n}][32] = {{\n")
        for y in ys:
            f.write(f"  {hex_to_c_array(y)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t interp_{label}_coefficients[{len(coeffs)}][32] = {{\n")
        for c in coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n\n")

    f.write(f"}} // namespace {ns_name}\n\n")


def emit_divisor_section(f, section, ns_name):
    """Emit test vector structs for a divisor section."""
    f.write(f"namespace {ns_name} {{\n\n")

    vectors = section.get("compute", [])
    for v in vectors:
        label = v["label"]
        n = v["n"]
        points = v["points"]
        a_coeffs = v["a_coefficients"]
        b_coeffs = v["b_coefficients"]
        f.write(f"static const uint8_t {label}_points[{n}][32] = {{\n")
        for pt in points:
            f.write(f"  {hex_to_c_array(pt)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t {label}_a_coefficients[{len(a_coeffs)}][32] = {{\n")
        for c in a_coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t {label}_b_coefficients[{len(b_coeffs)}][32] = {{\n")
        for c in b_coeffs:
            f.write(f"  {hex_to_c_array(c)},\n")
        f.write(f"}};\n")
        f.write(f"static const uint8_t {label}_eval_point_x[32] = {hex_to_c_array(v['eval_point_x'])};\n")
        f.write(f"static const uint8_t {label}_eval_point_y[32] = {hex_to_c_array(v['eval_point_y'])};\n")
        f.write(f"static const uint8_t {label}_eval_result[32] = {hex_to_c_array(v['eval_result'])};\n\n")

    f.write(f"}} // namespace {ns_name}\n\n")


def emit_batch_invert_section(f, section):
    """Emit batch inversion vectors."""
    f.write(f"namespace batch_invert {{\n\n")

    for field_name in ["fp", "fq"]:
        vectors = section.get(field_name, [])
        for v in vectors:
            label = v["label"]
            n = v["n"]
            f.write(f"static const uint8_t {field_name}_{label}_inputs[{n}][32] = {{\n")
            for inp in v["inputs"]:
                f.write(f"  {hex_to_c_array(inp)},\n")
            f.write(f"}};\n")
            f.write(f"static const uint8_t {field_name}_{label}_results[{n}][32] = {{\n")
            for res in v["results"]:
                f.write(f"  {hex_to_c_array(res)},\n")
            f.write(f"}};\n\n")

    f.write(f"}} // namespace batch_invert\n\n")


def emit_wei25519_section(f, section):
    """Emit Wei25519 bridge vectors."""
    f.write(f"namespace wei25519 {{\n\n")

    vectors = section.get("x_to_shaw_scalar", [])
    if vectors:
        f.write(f"struct x_to_scalar_vector {{ const char *label; uint8_t input[32]; bool valid; uint8_t result[32]; }};\n")
        f.write(f"static const x_to_scalar_vector x_to_scalar_vectors[] = {{\n")
        for v in vectors:
            valid = v["result"] is not None
            result = hex_to_c_array(v["result"]) if valid else hex_to_c_array("00" * 32)
            f.write(f"  {{\"{v['label']}\", {hex_to_c_array(v['input'])}, {'true' if valid else 'false'}, {result}}},\n")
        f.write(f"}};\n")
        f.write(f"static const size_t x_to_scalar_count = sizeof(x_to_scalar_vectors) / sizeof(x_to_scalar_vectors[0]);\n\n")

    f.write(f"}} // namespace wei25519\n\n")


def emit_high_degree_poly_mul_section(f, data):
    """Emit compact multi-point eval check data for high-degree polynomial multiplication.
    Only stores eval points and expected values — C++ tests rebuild inputs from
    the deterministic coefficient pattern (1, 2, 3, ...)."""
    f.write("namespace high_degree_poly_mul {\n\n")

    f.write("struct eval_check { const char *point; uint8_t x[32]; uint8_t a_of_x[32]; uint8_t b_of_x[32]; uint8_t result_of_x[32]; };\n")
    f.write("struct highdeg_vector { const char *label; int n_coeffs; int result_degree; eval_check checks[3]; };\n\n")

    for field_name in ["fp", "fq"]:
        if field_name not in data:
            continue

        vectors = data[field_name]
        f.write(f"static const highdeg_vector {field_name}_vectors[] = {{\n")
        for vec in vectors:
            label = vec["label"].replace("-", "_")
            n_coeffs = vec["n_coeffs"]
            result_degree = vec["result_degree"]
            checks = vec["eval_checks"]
            f.write(f"  {{\"{label}\", {n_coeffs}, {result_degree}, {{\n")
            for ci, check in enumerate(checks):
                comma = "," if ci < len(checks) - 1 else ""
                f.write(f"    {{\"{check['point']}\", {hex_to_c_array(check['x'])}, "
                        f"{hex_to_c_array(check['a_of_x'])}, "
                        f"{hex_to_c_array(check['b_of_x'])}, "
                        f"{hex_to_c_array(check['result_of_x'])}}}{comma}\n")
            f.write(f"  }}}},\n")
        f.write(f"}};\n")
        f.write(f"static const size_t {field_name}_count = sizeof({field_name}_vectors) / sizeof({field_name}_vectors[0]);\n\n")

    f.write("} // namespace high_degree_poly_mul\n\n")


def emit_flat_field_section(f, section, ns_name):
    """Emit a namespace with flat static const uint8_t[32] arrays from a key-value JSON section."""
    f.write(f"namespace {ns_name} {{\n")
    keys = list(section.keys())
    for key in keys:
        f.write(f"static const uint8_t {key}[32] = {hex_to_c_array(section[key])};\n")
    f.write(f"}} // namespace {ns_name}\n\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: python json_to_header.py <input.json> <output.h>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path) as f:
        data = json.load(f)

    with open(output_path, "w") as f:
        f.write("// Auto-generated by json_to_header.py — DO NOT EDIT\n")
        f.write("// Source: ranshaw_test_vectors.json\n\n")
        f.write("#ifndef RANSHAW_TEST_VECTORS_H\n")
        f.write("#define RANSHAW_TEST_VECTORS_H\n\n")
        f.write("#include <cstddef>\n")
        f.write("#include <cstdint>\n\n")
        f.write("namespace ranshaw_test_vectors {\n\n")

        # Parameters
        params = data.get("parameters", {})
        if params:
            f.write("namespace parameters {\n")
            for key in ["ran_order", "shaw_order", "ran_b", "shaw_b",
                        "ran_gx", "ran_gy", "shaw_gx", "shaw_gy"]:
                if key in params:
                    f.write(f"static const uint8_t {key}[32] = {hex_to_c_array(params[key])};\n")
            f.write("} // namespace parameters\n\n")

        # Scalar sections
        if "ran_scalar" in data:
            emit_scalar_section(f, data["ran_scalar"], "ran_scalar")
        if "shaw_scalar" in data:
            emit_scalar_section(f, data["shaw_scalar"], "shaw_scalar")

        # Point sections
        if "ran_point" in data:
            emit_point_section(f, data["ran_point"], "ran_point")
        if "shaw_point" in data:
            emit_point_section(f, data["shaw_point"], "shaw_point")

        # Polynomial sections
        if "fp_polynomial" in data:
            emit_polynomial_section(f, data["fp_polynomial"], "fp_polynomial")
        if "fq_polynomial" in data:
            emit_polynomial_section(f, data["fq_polynomial"], "fq_polynomial")

        # Divisor sections
        if "ran_divisor" in data:
            emit_divisor_section(f, data["ran_divisor"], "ran_divisor")
        if "shaw_divisor" in data:
            emit_divisor_section(f, data["shaw_divisor"], "shaw_divisor")

        # Wei25519
        if "wei25519" in data:
            emit_wei25519_section(f, data["wei25519"])

        # Batch invert
        if "batch_invert" in data:
            emit_batch_invert_section(f, data["batch_invert"])

        # High-degree polynomial multiplication
        if "high_degree_poly_mul" in data:
            emit_high_degree_poly_mul_section(f, data["high_degree_poly_mul"])

        # Raw field arithmetic
        if "fp_field" in data:
            emit_flat_field_section(f, data["fp_field"], "fp_field")
        if "fq_field" in data:
            emit_flat_field_section(f, data["fq_field"], "fq_field")

        # Compressed point vectors
        if "compressed_points" in data:
            emit_flat_field_section(f, data["compressed_points"], "compressed_points")

        # SSWU map-to-curve vectors
        if "sswu_vectors" in data:
            emit_flat_field_section(f, data["sswu_vectors"], "sswu_vectors")

        f.write("} // namespace ranshaw_test_vectors\n\n")
        f.write("#endif // RANSHAW_TEST_VECTORS_H\n")

    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
