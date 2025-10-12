#!/usr/bin/env python3
"""Poisson distribution class implementation.

This module provides a `Poisson` class that estimates the rate parameter
`lambtha` from data or uses a provided positive value.
"""


class Poisson:
    """Represents a Poisson distribution.

    Attributes:
        lambtha (float): Expected number of occurrences per time frame.
    """

    def __init__(self, data=None, lambtha=1.):
        """Initialize a Poisson distribution.

        Args:
            data (list or None): Observed counts to estimate `lambtha` from.
            lambtha (float): Rate parameter if data is not provided.

        Raises:
            TypeError: If `data` is provided and is not a list.
            ValueError: If `data` has fewer than two points.
            ValueError: If `lambtha` is not a positive value (> 0) when used.
        """
        if data is None:
            # Use provided lambtha; must be positive
            if lambtha is None or lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Validate data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Estimate lambtha from data (sample mean)
            total = 0.0
            for x in data:
                total += x
            self.lambtha = float(total / float(len(data)))

    def pmf(self, k):
        """Return the PMF value for k successes.

        If k is not an int, it is cast to int. Returns 0 for out-of-range k.
        """
        try:
            if not isinstance(k, int):
                k = int(k)
        except Exception:
            return 0

        if k < 0:
            return 0

        # Use a constant for Euler's number (no imports allowed)
        e_const = 2.7182818285

        # Compute k! iteratively
        fact = 1
        i = 1
        while i <= k:
            fact *= i
            i += 1

        # Compute lambda^k
        lam_pow_k = 1.0
        i = 0
        while i < k:
            lam_pow_k *= self.lambtha
            i += 1

        # Compute e^{-lambda} as 1/(e^{lambda})
        exp_neg_lam = 1.0 / (e_const ** self.lambtha)

        return exp_neg_lam * lam_pow_k / float(fact)
