#!/usr/bin/env python3
"""Summation utilities for squared integers.

This module exposes a single function `summation_i_squared(n)` that computes
the series ∑_{i=1}^{n} i^2 without using loops.

Constraints:
- Python 3.5 compatible
- No imports, no loops
- Return None for invalid inputs
"""


def summation_i_squared(n):
    """Compute the sum of squares from 1 to n (inclusive).

    Uses the closed form: n(n + 1)(2n + 1) / 6.

    Args:
        n (int): Upper bound (stopping condition) for the summation.

    Returns:
        int: Value of ∑_{i=1}^{n} i^2 when n is a valid positive integer.
        None: If n is not a valid number (non-int or n < 1).
    """
    if not isinstance(n, int) or n < 1:
        return None
    # Closed-form expression with integer arithmetic
    return n * (n + 1) * (2 * n + 1) // 6
