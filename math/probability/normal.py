#!/usr/bin/env python3
"""Normal distribution class (no external modules)."""


class Normal:
    """Represents a normal (Gaussian) distribution.

    Attributes:
        mean (float): Distribution mean.
        stddev (float): Distribution standard deviation (> 0).
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize a normal distribution.

        Args:
            data (list or None): Sample used to estimate mean and stddev when
                provided.
            mean (float): Mean to use when `data is None`.
            stddev (float): Std dev to use when `data is None`.

        Raises:
            TypeError: If `data` is provided and is not a list.
            ValueError: If `data` has fewer than two points.
            ValueError: If provided `stddev` is not a positive value.
        """
        if data is None:
            if stddev is None or stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            n = float(len(data))
            # Compute mean
            total = 0.0
            for x in data:
                total += x
            mu = total / n

            # Compute (population) standard deviation
            sq_sum = 0.0
            for x in data:
                diff = x - mu
                sq_sum += diff * diff
            sigma = (sq_sum / n) ** 0.5

            if sigma <= 0:
                # Degenerate case; still enforce positivity per spec
                raise ValueError("stddev must be a positive value")

            self.mean = float(mu)
            self.stddev = float(sigma)