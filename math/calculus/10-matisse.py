#!/usr/bin/env python3
"""Polynomial utilities.

This module provides `poly_derivative(poly)` which computes the derivative
of a polynomial represented by a list of coefficients, where index i is the
coefficient of x**i.

Example: f(x) = x^3 + 3x + 5  ->  poly = [5, 3, 0, 1]
Derivative: f'(x) = 3x^2 + 3  ->  [3, 0, 3]
"""


def _is_number(value):
    """Return True if value is an int or float (but not bool)."""
    return (isinstance(value, int) and not isinstance(value, bool)) or \
        isinstance(value, float)


def poly_derivative(poly):
    """Compute the derivative coefficients of a polynomial.

    Args:
        poly (list): Coefficient list where poly[i] is coeff of x**i.

    Returns:
        list: Coefficients of the derivative, or [0] if derivative is 0.
        None: If `poly` is not a valid polynomial representation.
    """
    # Validate input structure
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(_is_number(c) for c in poly):
        return None

    # If polynomial is constant, derivative is zero
    if len(poly) == 1:
        return [0]

    # Compute derivative: for i >= 1, new_coeff[i-1] = i * poly[i]
    deriv = []
    for i in range(1, len(poly)):
        c = i * poly[i]
        # Normalize floats that are mathematically integers
        if isinstance(c, float) and c.is_integer():
            c = int(c)
        deriv.append(c)

    # If all derivative coefficients are 0, return [0]
    if all((x == 0 or x == 0.0) for x in deriv):
        return [0]

    return deriv
