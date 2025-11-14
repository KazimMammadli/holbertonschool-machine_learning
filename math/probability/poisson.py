#!/usr/bin/env python3
"""Initializing Poisson"""
e = 2.7182818285


class Poisson:
    """Class."""

    def __init__(self, data=None, lambtha=1):
        """Constructor."""
        if data is None:

            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")

            self.lambtha = float(lambtha)

        else:

            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = sum(data) / len(data)

    @staticmethod
    def factorial(n):
        """Calculate factorial of given number n"""
        if not isinstance(n, int):
            raise ValueError("The number must be integer")

        f = 1
        for i in range(1, n + 1):
            f *= i
        return f

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        if k < 0:
            return 0
        k = int(k)
        return (e ** (-self.lambtha) * self.lambtha ** k) / self.factorial(k)

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes"""
        if k < 0:
            return 0

        k = int(k)

        return sum((e ** -self.lambtha * self.lambtha ** i) / self.factorial(i)
                   for i in range(k + 1))
