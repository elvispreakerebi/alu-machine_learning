#!/usr/bin/env python3
"""Exponential distribution class (no external modules)."""


class Exponential:
    """Represents an exponential distribution with rate `lambtha`.

    Attributes:
        lambtha (float): Rate parameter (> 0), expected events per unit time.
    """

    def __init__(self, data=None, lambtha=1.):
        """Initialize an exponential distribution.

        Args:
            data (list or None): Sample of inter-arrival times to estimate
                `lambtha` from (uses 1/mean of data) when provided.
            lambtha (float): Rate parameter when `data is None`.

        Raises:
            TypeError: If `data` is provided and is not a list.
            ValueError: If `data` has fewer than two points.
            ValueError: If provided `lambtha` is not a positive value.
        """
        if data is None:
            if lambtha is None or lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            total = 0.0
            for x in data:
                total += x
            mean = total / float(len(data))
            # Rate is inverse of mean for exponential distribution
            if mean <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(1.0 / mean)

    def pdf(self, x):
        """Return the PDF value at time x.

        For x < 0, returns 0.
        """
        try:
            x = float(x)
        except Exception:
            return 0

        if x < 0.0:
            return 0

        # Use numeric constant for Euler's number
        e_const = 2.7182818285
        # e^{-lambtha * x} = 1 / (e^{lambtha * x})
        exp_term = 1.0 / (e_const ** (self.lambtha * x))
        return self.lambtha * exp_term
