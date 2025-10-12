#!/usr/bin/env python3
"""Binomial distribution class (no external modules)."""


class Binomial:
    """Represents a Binomial(n, p) distribution.

    Attributes:
        n (int): number of trials (> 0 when provided directly).
        p (float): success probability (0 < p < 1 when provided directly).
    """

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize a binomial distribution.

        Args:
            data (list or None): Sample used to estimate n and p when provided.
            n (int/float): Number of trials (used when data is None).
            p (float): Success probability (used when data is None).

        Raises:
            TypeError: If data is provided and is not a list.
            ValueError: If data has fewer than two points.
            ValueError: If provided n <= 0 or p not in (0, 1).
        """
        if data is None:
            # Use provided parameters, validate
            if n is None or n <= 0:
                raise ValueError("n must be a positive value")
            if p is None or not (0 < p < 1):
                raise ValueError(
                    "p must be greater than 0 and less than 1"
                )
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Compute mean and variance from data
            m = 0.0
            for x in data:
                m += x
            m /= float(len(data))

            var = 0.0
            for x in data:
                d = x - m
                var += d * d
            var /= float(len(data))

            # Estimate parameters: p first, then n, then refine p
            # From Binomial: mean = n p, variance = n p (1-p)
            # => p = 1 - variance/mean (assuming mean > 0)
            if m == 0.0:
                est_p = 0.0
            else:
                est_p = 1.0 - (var / m)
            if est_p <= 0.0:
                # Fallback to minimal positive to avoid division by zero
                est_p = 1e-9

            est_n = round(m / est_p) if est_p != 0.0 else 1.0
            if est_n <= 0:
                est_n = 1.0

            self.n = int(est_n)
            # Recalculate p using mean and integer n
            self.p = float(m / float(self.n))
