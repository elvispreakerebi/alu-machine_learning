#!/usr/bin/env python3
"""Matrix inverse without imports."""


def _determinant(matrix):
    """Compute determinant of a square matrix (list of lists)."""
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for j, elem in enumerate(matrix[0]):
        minor = [
            [matrix[i][k] for k in range(n) if k != j]
            for i in range(1, n)
        ]
        det += ((-1) ** j) * elem * _determinant(minor)
    return det


def _adjugate(matrix):
    """Return adjugate (transpose of cofactor) of a square matrix."""
    n = len(matrix)
    if n == 1:
        return [[1]]
    cof = []
    for i in range(n):
        co_row = []
        for j in range(n):
            sub = [[matrix[r][c] for c in range(n) if c != j]
                   for r in range(n) if r != i]
            sign = -1 if ((i + j) % 2) else 1
            co_row.append(sign * _determinant(sub if sub else [[]]))
        cof.append(co_row)
    # transpose
    return [[cof[r][c] for r in range(n)] for c in range(n)]


def inverse(matrix):
    """Return the inverse of a matrix or None if singular.

    Args:
        matrix: list of lists representing a matrix.

    Returns:
        The inverse matrix (list of lists) or None if matrix is singular.

    Raises:
        TypeError: if matrix is not a list of lists.
        ValueError: if matrix is not a non-empty square matrix.
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    row_lengths = []
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        row_lengths.append(len(row))
    if any(rl != n for rl in row_lengths) or n == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    det = _determinant(matrix)
    if det == 0:
        return None

    adj = _adjugate(matrix)
    inv = []
    for i in range(n):
        inv_row = []
        for j in range(n):
            inv_row.append(adj[i][j] / float(det))
        inv.append(inv_row)
    return inv
