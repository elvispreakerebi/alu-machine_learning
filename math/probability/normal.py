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

    def z_score(self, x):
        """Return the z-score of value x.

        z = (x - mean) / stddev
        """
        x = float(x)
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Return the x-value for a given z-score.

        x = z * stddev + mean
        """
        z = float(z)
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Return the PDF value for x.

        Uses 1/(σ√(2π)) · e^{-(1/2)((x−μ)/σ)^2} with numeric constants.
        """
        x = float(x)
        z = (x - self.mean) / self.stddev
        pi_const = 3.141592653589793
        e_const = 2.718281828459045
        denom = self.stddev * (2.0 * pi_const) ** 0.5
        exp_term = e_const ** (-0.5 * z * z)
        return exp_term / denom

    @staticmethod
    def _erf(x):
        """Approximate the error function erf(x) with A&S formula.

        Uses Abramowitz and Stegun formula 7.1.26.
        """
        # Save the sign of x
        sign = 1.0
        if x < 0:
            sign = -1.0
            x = -x

        # Constants
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        t = 1.0 / (1.0 + p * x)
        poly = (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t
        e_const = 2.718281828459045
        exp_term = e_const ** (-(x * x))
        y = 1.0 - poly * exp_term
        return sign * y

    def cdf(self, x):
        """Return the CDF value for x.

        Uses Φ(x) = 0.5 * [1 + erf((x−μ)/(σ√2))].
        """
        x = float(x)
        z = (x - self.mean) / self.stddev
        sqrt2 = 2.0 ** 0.5
        return 0.5 * (1.0 + self._erf(z / sqrt2))
