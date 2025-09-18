#!/usr/bin/env python3
"""Determinant of a matrix.

Provides a function `determinant(matrix)` that computes the determinant of a
square matrix represented as a list of lists.
"""

from typing import List


def determinant(matrix: List[List[float]]) -> float:
    """Calculate the determinant of a matrix.

    Args:
        matrix: list of lists of numbers. `[[ ]]` represents a 0x0 matrix.

    Returns:
        The determinant as a number (int/float).

    Raises:
        TypeError: if `matrix` is not a list of lists.
        ValueError: if `matrix` is not square.
    """
    # Validate top-level container
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Special-case 0x0 matrix
    if matrix == [[]]:
        return 1

    # Validate rows and squareness
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    row_lengths = []
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        row_lengths.append(len(row))

    n = len(matrix)
    if any(rl != n for rl in row_lengths):
        raise ValueError("matrix must be a square matrix")

    # Base cases
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive Laplace expansion along the first row
    det = 0
    for j, elem in enumerate(matrix[0]):
        # Build minor by excluding row 0 and column j
        minor = [
            [matrix[i][k] for k in range(n) if k != j]
            for i in range(1, n)
        ]
        cofactor = (-1) ** j * elem * determinant(minor)
        det += cofactor
    return det
