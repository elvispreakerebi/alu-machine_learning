#!/usr/bin/env python3
"""Matrix definiteness using NumPy."""

import numpy as np


def definiteness(matrix):
    """Return the definiteness category of a matrix.

    Args:
        matrix (np.ndarray): square symmetric matrix of shape (n, n).

    Returns:
        str | None: One of
            - "Positive definite"
            - "Positive semi-definite"
            - "Negative definite"
            - "Negative semi-definite"
            - "Indefinite"
        or None if the input is invalid.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Require symmetry for definiteness classification
    if not np.allclose(matrix, matrix.T):
        return None

    # For symmetric matrices use eigvalsh (real eigenvalues)
    vals = np.linalg.eigvalsh(matrix)

    eps = 1e-8
    all_pos = np.all(vals > eps)
    any_pos = np.any(vals > eps)
    all_neg = np.all(vals < -eps)
    any_neg = np.any(vals < -eps)
    any_zero = np.any(np.abs(vals) <= eps)

    if all_pos:
        return "Positive definite"
    if any_pos and not any_neg and any_zero:
        return "Positive semi-definite"
    if all_neg:
        return "Negative definite"
    if any_neg and not any_pos and any_zero:
        return "Negative semi-definite"
    if any_pos and any_neg:
        return "Indefinite"
    return None


