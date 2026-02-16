#!/usr/bin/env sage
"""
ECFFT Precomputation Script for RanShaw

Generates static precomputed data (.inl files) for Elliptic Curve Fast Fourier
Transform (ECFFT) polynomial evaluation and interpolation over the Ran/Shaw
base fields Fp (2^255 - 19) and Fq (Crandall prime 2^255 - gamma).

References:
  [BCKL23]  Eli Ben-Sasson, Dan Carmon, Swastik Kopparty, David Levit.
            "Elliptic Curve Fast Fourier Transform (ECFFT) Part I."
            https://arxiv.org/abs/2107.08473 (2023, revised).
  [Velu71]  Jacques Vélu. "Isogénies entre courbes elliptiques."
            Comptes Rendus de l'Académie des Sciences, 273, pp. 238-241 (1971).
  [FCMP]    FCMP++ specification — Ran/Shaw curve cycle.

Background:
  Standard FFT requires roots of unity of order 2^k in the base field.
  Neither Fp = 2^255 - 19 nor Fq = 2^255 - gamma has large 2-adic subgroups
  in their multiplicative group (they are ~1 mod 4, not mod 2^k for large k).
  ECFFT [BCKL23] replaces roots of unity with an "evaluation domain" derived
  from an auxiliary elliptic curve E/F whose group order #E has high 2-adic
  valuation v2(#E).

  The ECFFT domain consists of x-coordinates of a coset {R + i*G : i=0..N-1}
  where G generates the 2^k subgroup of E(F) and R is an offset point outside
  that subgroup. The "FFT butterfly" is replaced by degree-2 isogenies: at each
  level, a 2-isogeny phi: E_i -> E_{i+1} maps pairs of points (that differ by
  the kernel point T) to the same x-coordinate, splitting the domain in half.

  The isogeny chain provides the rational x-maps that replace the "twiddle
  factors" of classical FFT. Each x-map is a degree-2/degree-1 rational
  function psi(x) = (x^2 + c1*x + c0) / (x + d0), derived from Vélu's
  formulas [Velu71].

Algorithm overview (same as gen_ecfft_data.cpp, the native C++ equivalent):
  1. Start with auxiliary curve E: y^2 = x^3 + ax + b over F.
  2. Compute #E using Sage's SEA (Schoof-Elkies-Atkin) point counting.
  3. Factor out the 2-adic part: #E = 2^v2 * m with m odd.
  4. Find G of order 2^v2 by cofactor multiplication: G = m * P for random P.
  5. Build v2 levels of degree-2 isogenies:
     - At level i, kernel point K_i = 2^(v2-i-1) * G_i has order 2.
     - Compute phi_i: E_i -> E_{i+1} with kernel <K_i> using Sage's isogeny().
     - Extract the x-coordinate rational map psi_i from phi_i.rational_maps().
     - Push generator forward: G_{i+1} = phi_i(G_i).
  6. Generate coset {R + i*G : i=0..2^v2-1}, extract x-coordinates.
  7. Write coset + isogeny coefficients to .inl file.

Coset ordering convention:
  The coset is stored in natural order: position i contains the x-coordinate
  of R + i*G. The ECFFT init functions in ecfft_fp.h / ecfft_fq.h apply a
  bit-reversal permutation when loading this data, which reorders the points
  so that at each level, isogeny fiber pairs (points P and P+T that map to
  the same x-coordinate under the 2-isogeny) appear at adjacent even/odd
  indices. This is analogous to the bit-reversal permutation in classical
  Cooley-Tukey FFT.

Velu's 2-isogeny formulas [Velu71]:
  For E: y^2 = x^3 + ax + b with kernel point T = (x0, 0) (order 2):
    gx = 3*x0^2 + a
    x-map: psi(x) = x + gx/(x - x0) = (x^2 - x0*x + gx) / (x - x0)
    y-map: psi_y(x,y) = y * ((x - x0)^2 - gx) / (x - x0)^2
    Codomain: a' = a - 5*gx,  b' = b - 7*x0*gx

  Note: The comment on line ~109 says "-2*x0*x" but Sage's rational_maps()
  returns the correct formula with "-x0*x". The code uses Sage's output
  directly, so the isogeny data is correct regardless of the comment.

Usage:
    sage ecfft_params.sage fp                          # Generate Fp data (hardcoded a=1, b=3427)
    sage ecfft_params.sage fp --known-b 0xABCD...      # Generate Fp data (a=-3, b=given)
    sage ecfft_params.sage fp --a 1 --known-b 0x0d63 --known-order 0x... --seed 42
    sage ecfft_params.sage fq                          # Generate Fq data (random search)
    sage ecfft_params.sage fq --known-b 0x43d2af...    # Generate Fq data (a=-3, b=given)
    sage ecfft_params.sage both                        # Generate both

Output:
    .inl data is written to stdout. Diagnostic messages go to stderr.
    Redirect stdout to a file to capture the .inl output:
        sage ecfft_params.sage fq --known-b 0x... > include/ecfft_fq_data.inl

Requirements:
    SageMath 9.0+ with SEA point counting support
"""

import sys
import os
import time

def eprint(*args, **kwargs):
    """Print to stderr (diagnostic output)."""
    print(*args, file=sys.stderr, **kwargs)

def find_ecfft_curve(p, known_a=None, known_b=None, known_order=None, max_tries=100000):
    """
    Find an elliptic curve E: y^2 = x^3 + ax + b over GF(p) whose group order
    #E(GF(p)) has maximal 2-adic valuation v2 = v_2(#E).

    For ECFFT [BCKL23], we need v2 >= 12 (domain size 4096) or ideally >= 16
    (domain size 65536). By the Hasse bound, #E = p + 1 - t where |t| <= 2*sqrt(p),
    so roughly half of random curves have even order. Among those with full 2-torsion
    (v2 >= 2), the probability of v2 >= k is approximately 1/2^(k-2).

    When known_a and known_b are provided, skips the random search and uses the
    given curve directly. When known_order is also provided, skips SEA point
    counting entirely (useful when the order is already known from a prior run).

    Args:
        p: Field prime.
        known_a, known_b: If both provided, use curve y^2 = x^3 + a*x + b directly.
        known_order: If provided (with known_a/known_b), use this as #E instead of
                     computing it via SEA. Must be the exact group order.
        max_tries: Maximum random curves to test (default 100000).

    Returns:
        (E, order, v2): SageMath EllipticCurve, group order, 2-adic valuation.
    """
    F = GF(p)

    if known_a is not None and known_b is not None:
        E = EllipticCurve(F, [known_a, known_b])
        if known_order is not None:
            order = known_order
            eprint(f"Using known curve y^2 = x^3 + {known_a}*x + {known_b}")
            eprint(f"  Using provided order (skipping SEA point counting)")
        else:
            eprint(f"Using known curve y^2 = x^3 + {known_a}*x + {known_b}")
            eprint(f"  Computing group order via SEA (this may take a while)...")
            order = E.order()
        v2 = valuation(order, 2)
        eprint(f"  Order = {order}")
        eprint(f"  Order (hex) = 0x{order:064x}")
        eprint(f"  2-adic valuation = {v2}")
        eprint(f"  Max domain size = {2^v2}")
        return E, order, v2

    best_v2 = 0
    best_curve = None
    best_order = None

    for trial in range(max_tries):
        a = F.random_element()
        b = F.random_element()
        if 4*a^3 + 27*b^2 == 0:
            continue
        E = EllipticCurve(F, [a, b])
        order = E.order()
        v2 = valuation(order, 2)
        if v2 > best_v2:
            best_v2 = v2
            best_curve = E
            best_order = order
            eprint(f"  Trial {trial}: v2={v2}, a={a}, b={b}")
            if v2 >= 16:
                break

    eprint(f"Best curve: y^2 = x^3 + {best_curve.a4()}*x + {best_curve.a6()}")
    eprint(f"  Order = {best_order}")
    eprint(f"  Order (hex) = 0x{best_order:064x}")
    eprint(f"  2-adic valuation = {best_v2}")
    eprint(f"  Max domain size = {2^best_v2}")
    return best_curve, best_order, best_v2


def compute_isogeny_chain(E, order, v2):
    """
    Compute the chain of v2 degree-2 isogenies for the ECFFT decomposition.

    Starting from E_0 = E with generator G_0 of the 2^v2 subgroup, at each
    level i we compute a degree-2 isogeny phi_i: E_i -> E_{i+1} whose kernel
    is <K_i> where K_i = 2^(v2-i-1) * G_i has order 2.

    The x-coordinate rational map psi_i(x) = num(x)/den(x) of each isogeny
    is the ECFFT analogue of the "twiddle factor" in classical FFT. For a
    2-isogeny with kernel point (x0, 0), Vélu's formulas [Velu71] give:
        psi(x) = (x^2 - x0*x + (3*x0^2 + a)) / (x - x0)
    which is always degree 2 / degree 1.

    The generator is pushed through each isogeny: G_{i+1} = phi_i(G_i).
    After v2 levels, G_v2 is the identity (the subgroup has been exhausted).

    This function uses Sage's built-in isogeny computation (phi.rational_maps())
    rather than manually implementing Vélu, which serves as a cross-check
    against the manual Vélu implementation in gen_ecfft_data.cpp.

    Args:
        E: SageMath EllipticCurve over GF(p).
        order: #E(GF(p)).
        v2: 2-adic valuation of order.

    Returns:
        List of dicts, one per level, each containing:
          - 'num_coeffs': coefficients of numerator polynomial [c0, c1, c2]
          - 'den_coeffs': coefficients of denominator polynomial [d0, d1]
          - 'kernel_x': x-coordinate of the kernel point
          - 'codomain': the codomain curve E_{i+1}
    """
    F = E.base_field()
    chain = []
    current_E = E

    # Find a point of order 2^v2
    cofactor = order // (2^v2)
    while True:
        P = current_E.random_point()
        G = cofactor * P
        if G != current_E(0) and (2^(v2-1)) * G != current_E(0):
            break

    eprint(f"Generator of 2^{v2} subgroup found")

    for level in range(v2):
        # Kernel point for degree-2 isogeny: point of order 2
        kernel_point = (2^(v2 - level - 1)) * G
        assert 2 * kernel_point == current_E(0), "Kernel point must have order 2"

        # Compute the degree-2 isogeny
        phi = current_E.isogeny(kernel_point, degree=2)
        codomain = phi.codomain()

        # Extract the rational map for x-coordinate: psi(x) = u(x)/v(x)
        # For a degree-2 isogeny with kernel (x0, 0):
        #   psi(x) = x + (slope stuff) -- Velu's formula
        x0 = kernel_point[0]

        # The x-map of a 2-isogeny with kernel point (x0,0) on y^2=x^3+ax+b is:
        #   psi(x) = (x^2 - x0*x + (3*x0^2 + a)) / (x - x0)  [Vélu, 1971]
        # We use Sage's rational_maps() which computes the same thing.
        rational_maps = phi.rational_maps()
        # rational_maps[0] is the x-coordinate map as a rational function
        x_map = rational_maps[0]

        # Extract numerator and denominator as polynomials in x
        R = PolynomialRing(F, 'x')
        x_var = R.gen()
        num = R(x_map.numerator())
        den = R(x_map.denominator())

        chain.append({
            'num_coeffs': [num[i] for i in range(num.degree() + 1)],
            'den_coeffs': [den[i] for i in range(den.degree() + 1)],
            'kernel_x': x0,
            'codomain': codomain,
        })

        # Update for next level
        current_E = codomain
        G = phi(G)

        eprint(f"  Level {level}: kernel x={x0}, codomain a4={codomain.a4()}, a6={codomain.a6()}")

    return chain


def compute_coset(E, order, v2, domain_size=None):
    """
    Compute the ECFFT evaluation domain: x-coordinates of a coset of the 2^v2
    subgroup of E(GF(p)).

    The domain is the set of x-coordinates {x(R + i*G) : i = 0, ..., 2^v2 - 1}
    where G generates the 2^v2 subgroup and R is a "random" offset point that
    is NOT in the 2-primary part of E(GF(p)). This ensures all 2^v2 points in
    the coset have distinct x-coordinates (no two points are negatives of each
    other), which is required for the ECFFT evaluation to be well-defined.

    The coset is output in natural order (position i = x-coordinate of R + i*G).
    The ECFFT init functions apply bit-reversal permutation when loading.

    Args:
        E: SageMath EllipticCurve.
        order: #E(GF(p)).
        v2: 2-adic valuation of order.
        domain_size: Override domain size (default: 2^v2).

    Returns:
        List of 2^v2 field elements (x-coordinates in natural coset order).
    """
    F = E.base_field()

    if domain_size is None:
        domain_size = 2^v2

    assert domain_size <= 2^v2

    cofactor = order // (2^v2)

    # Find generator of 2^v2 subgroup
    while True:
        P = E.random_point()
        G = cofactor * P
        if G != E(0) and (2^(v2-1)) * G != E(0):
            break

    # Find a random offset point not in the subgroup
    while True:
        R = E.random_point()
        # Check R is not in the 2-primary subgroup
        test = cofactor * R
        if test != E(0):
            break

    # Generate coset: {R + i*G for i in 0..domain_size-1}
    coset_points = []
    current = R
    for i in range(domain_size):
        coset_points.append(current[0])  # x-coordinate
        current = current + G

    eprint(f"Computed coset of size {domain_size}")
    return coset_points


def field_element_to_bytes(x, p):
    """Convert a field element to 32 bytes, little-endian."""
    val = int(ZZ(x) % p)
    return val.to_bytes(32, byteorder='little')


def write_inl_data(coset, isogeny_chain, field_name, p, seed_value=None,
                   a_val=None, b_val=None, order=None):
    """Write the .inl data to stdout."""
    n = len(coset)
    n_levels = len(isogeny_chain)

    out = sys.stdout
    out.write(f"// Auto-generated by ecfft_params.sage — DO NOT EDIT\n")
    out.write(f"// ECFFT precomputed data for F_{field_name}\n")
    out.write(f"// Field prime: 0x{int(p):064x}\n")
    if a_val is not None:
        out.write(f"// Curve parameter a: {a_val}\n")
    if b_val is not None:
        out.write(f"// Curve parameter b: 0x{b_val:064x}\n")
    if order is not None:
        out.write(f"// Group order: 0x{int(order):064x}\n")
    out.write(f"// Domain size: {n}, Levels: {n_levels}\n")
    if seed_value is not None:
        out.write(f"// Seed: {seed_value}\n")
    out.write(f"\n")

    # Domain size and level count
    out.write(f"static const size_t ECFFT_{field_name.upper()}_DOMAIN_SIZE = {n};\n")
    out.write(f"static const size_t ECFFT_{field_name.upper()}_LOG_DOMAIN = {n_levels};\n\n")

    # Coset x-coordinates
    out.write(f"static const unsigned char ECFFT_{field_name.upper()}_COSET[{n} * 32] = {{\n")
    for i, x in enumerate(coset):
        b = field_element_to_bytes(x, p)
        hex_str = ', '.join(f'0x{byte:02x}' for byte in b)
        comma = ',' if i < n - 1 else ''
        out.write(f"    {hex_str}{comma}\n")
    out.write(f"}};\n\n")

    # Isogeny chain data: for each level, store numerator and denominator coefficients
    # Format: [level][coeff_index] as 32-byte field elements
    # Each level has: num_degree+1 numerator coeffs, den_degree+1 denominator coeffs
    for level, iso in enumerate(isogeny_chain):
        num_coeffs = iso['num_coeffs']
        den_coeffs = iso['den_coeffs']

        out.write(f"// Level {level}: num degree {len(num_coeffs)-1}, den degree {len(den_coeffs)-1}\n")
        out.write(f"static const unsigned char ECFFT_{field_name.upper()}_ISO_NUM_{level}[{len(num_coeffs)} * 32] = {{\n")
        for i, c in enumerate(num_coeffs):
            b = field_element_to_bytes(c, p)
            hex_str = ', '.join(f'0x{byte:02x}' for byte in b)
            comma = ',' if i < len(num_coeffs) - 1 else ''
            out.write(f"    {hex_str}{comma}\n")
        out.write(f"}};\n")

        out.write(f"static const unsigned char ECFFT_{field_name.upper()}_ISO_DEN_{level}[{len(den_coeffs)} * 32] = {{\n")
        for i, c in enumerate(den_coeffs):
            b = field_element_to_bytes(c, p)
            hex_str = ', '.join(f'0x{byte:02x}' for byte in b)
            comma = ',' if i < len(den_coeffs) - 1 else ''
            out.write(f"    {hex_str}{comma}\n")
        out.write(f"}};\n\n")

    # Isogeny metadata array (pointers will be set up in C++)
    out.write(f"static const size_t ECFFT_{field_name.upper()}_ISO_NUM_DEGREE[{n_levels}] = {{\n")
    out.write(f"    {', '.join(str(len(iso['num_coeffs'])-1) for iso in isogeny_chain)}\n")
    out.write(f"}};\n\n")

    out.write(f"static const size_t ECFFT_{field_name.upper()}_ISO_DEN_DEGREE[{n_levels}] = {{\n")
    out.write(f"    {', '.join(str(len(iso['den_coeffs'])-1) for iso in isogeny_chain)}\n")
    out.write(f"}};\n")

    eprint(f"Wrote {field_name} .inl data to stdout")


def generate_fp(known_a=None, known_b=None, known_order=None, seed_value=None):
    """Generate ECFFT data for Fp (p = 2^255 - 19)."""
    p = 2^255 - 19
    eprint("=== Generating ECFFT data for Fp ===")
    eprint(f"p = {p}")

    if known_b is not None:
        a_val = known_a if known_a is not None else -3
        E, order, v2 = find_ecfft_curve(p, known_a=a_val, known_b=known_b, known_order=known_order)
    elif known_a is not None:
        # Custom a but default b=3427
        E, order, v2 = find_ecfft_curve(p, known_a=known_a, known_b=3427, known_order=known_order)
    else:
        E, order, v2 = find_ecfft_curve(p, known_a=1, known_b=3427, known_order=known_order)

    a_val = int(E.a4())
    b_val = int(E.a6())
    chain = compute_isogeny_chain(E, order, v2)
    coset = compute_coset(E, order, v2)
    write_inl_data(coset, chain, 'fp', p, seed_value=seed_value,
                   a_val=a_val, b_val=b_val, order=order)


def generate_fq(known_a=None, known_b=None, known_order=None, seed_value=None):
    """Generate ECFFT data for Fq (Crandall prime)."""
    # q = 2^255 - gamma, where gamma is the Ran group order / Shaw base field
    # gamma (in radix-2^51 limbs): {0x12D8D86D83861, 0x269135294F229, 0x102021F}
    gamma = 0x12D8D86D83861 + 0x269135294F229 * 2**51 + 0x102021F * 2**102
    q = 2**255 - gamma

    eprint("=== Generating ECFFT data for Fq ===")
    eprint(f"q = {q}")
    eprint(f"q (hex) = {hex(q)}")

    if known_b is not None:
        a_val = known_a if known_a is not None else -3
        E, order, v2 = find_ecfft_curve(q, known_a=a_val, known_b=known_b, known_order=known_order)
    else:
        eprint("Searching for ECFFT-friendly curve over Fq...")
        eprint("(This may take a while)")
        E, order, v2 = find_ecfft_curve(q, known_order=known_order)

    a_val = int(E.a4())
    b_val = int(E.a6())
    chain = compute_isogeny_chain(E, order, v2)
    coset = compute_coset(E, order, v2)
    write_inl_data(coset, chain, 'fq', q, seed_value=seed_value,
                   a_val=a_val, b_val=b_val, order=order)


def parse_args(args):
    """Parse CLI arguments. Returns (known_a, known_b, known_order, seed, remaining_args)."""
    known_a = None
    known_b = None
    known_order = None
    seed = None
    remaining = []
    i = 0
    while i < len(args):
        if args[i] == '--known-b':
            if i + 1 >= len(args):
                eprint("Error: --known-b requires a hex value argument")
                sys.exit(1)
            hex_str = args[i + 1]
            if hex_str.startswith('0x') or hex_str.startswith('0X'):
                hex_str = hex_str[2:]
            known_b = int(hex_str, 16)
            i += 2
        elif args[i] == '--known-order':
            if i + 1 >= len(args):
                eprint("Error: --known-order requires a hex value argument")
                sys.exit(1)
            hex_str = args[i + 1]
            if hex_str.startswith('0x') or hex_str.startswith('0X'):
                hex_str = hex_str[2:]
            known_order = int(hex_str, 16)
            i += 2
        elif args[i] == '--a':
            if i + 1 >= len(args):
                eprint("Error: --a requires an integer argument")
                sys.exit(1)
            known_a = int(args[i + 1])
            i += 2
        elif args[i] == '--seed':
            if i + 1 >= len(args):
                eprint("Error: --seed requires an integer argument")
                sys.exit(1)
            seed = int(args[i + 1], 0)  # accepts decimal and 0x hex
            i += 2
        else:
            remaining.append(args[i])
            i += 1
    return known_a, known_b, known_order, seed, remaining


if __name__ == '__main__':
    if len(sys.argv) < 2:
        eprint("Usage: sage ecfft_params.sage [fp|fq|both] [OPTIONS]")
        eprint()
        eprint("Options:")
        eprint("  --known-b <hex>      Use specific curve with b=<hex> (with or without 0x prefix)")
        eprint("  --known-order <hex>  Use specific group order (skips SEA point counting)")
        eprint("  --a N                Curve parameter a (integer, default: 1 for fp, -3 for fq)")
        eprint("  --seed N             PRNG seed for deterministic output (decimal or 0x hex)")
        eprint()
        eprint("Output:")
        eprint("  .inl data is written to stdout. Diagnostic messages go to stderr.")
        eprint("  Redirect stdout to capture the .inl file:")
        eprint("    sage ecfft_params.sage fq --known-b 0x... --seed 42 > include/ecfft_fq_data.inl")
        eprint()
        eprint("Examples:")
        eprint("  sage ecfft_params.sage fp                       # Hardcoded Fp curve (a=1, b=3427)")
        eprint("  sage ecfft_params.sage fp --a 1                 # Fp curve with a=1, b=3427")
        eprint("  sage ecfft_params.sage fp --a 1 --known-b 0x0d63 --known-order 0x... --seed 42")
        eprint("  sage ecfft_params.sage fq --known-b 0x43d2... --seed 42  # Deterministic Fq")
        eprint("  sage ecfft_params.sage fq                       # Random search for Fq curve")
        sys.exit(1)

    known_a, known_b, known_order, seed, remaining = parse_args(sys.argv[1:])

    if len(remaining) < 1:
        eprint("Error: target [fp|fq|both] is required")
        sys.exit(1)

    # Set PRNG seed for reproducibility
    if seed is None:
        seed = int(time.time())
    set_random_seed(seed)
    eprint(f"PRNG seed: {seed}")

    target = remaining[0]
    if target == 'fp':
        generate_fp(known_a=known_a, known_b=known_b, known_order=known_order, seed_value=seed)
    elif target == 'fq':
        generate_fq(known_a=known_a, known_b=known_b, known_order=known_order, seed_value=seed)
    elif target == 'both':
        generate_fp(known_a=known_a, known_b=known_b, known_order=known_order, seed_value=seed)
        generate_fq(known_a=known_a, known_b=known_b, known_order=known_order, seed_value=seed)
    else:
        eprint(f"Unknown target: {target}")
        sys.exit(1)
