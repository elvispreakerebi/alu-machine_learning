#!/usr/bin/env python3
"""Polynomial integral utilities.

This module exposes `poly_integral(poly, C=0)` to compute the indefinite
integral of a polynomial represented by a coefficient list where index i
corresponds to the coefficient of x**i. The returned coefficient list is as
small as possible and uses integers when coefficients are whole numbers.
"""


def _is_number(value):
    """Return True if value is an int or float (but not bool)."""
    return (isinstance(value, int) and not isinstance(value, bool)) or \
        isinstance(value, float)


def _is_int(value):
    """Return True if value is a whole number (int-like)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    if isinstance(value, float) and value.is_integer():
        return True
    return False


def poly_integral(poly, C=0):
    """Compute the integral coefficients of a polynomial.

    Args:
        poly (list): Coefficient list where poly[i] is coeff of x**i.
        C (int): Integration constant to be placed as the constant term.

    Returns:
        list: Coefficients of the integral (minimal size), or None if invalid.
    """
    # Validate inputs
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(_is_number(c) for c in poly):
        return None
    if not (isinstance(C, int) and not isinstance(C, bool)):
        return None

    # Build integral coefficients: first item is the constant C
    result = [C]
    for i, a_i in enumerate(poly):
        denom = i + 1
        term = a_i / float(denom)
        # Convert whole-number coefficients to int
        if _is_int(term):
            term = int(term)
        result.append(term)

    # Minimize list size: remove trailing zeros
    while len(result) > 1 and (result[-1] == 0 or result[-1] == 0.0):
        result.pop()

    # If all terms removed and C == 0, keep [0]
    if len(result) == 0:
        result = [0]

    return result
