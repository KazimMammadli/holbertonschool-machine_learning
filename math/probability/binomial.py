#!/usr/bin/env python3
"""Initializing Binomial"""
e = 2.7182818285
pi = 3.1415926536


class Binomial:
    """Class."""

    def __init__(self, data=None, n=1, p=0.5):
        """Constructor."""
        if data is None:

            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)

        else:

            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            N = len(data)
            mean = sum(data) / N
            var = 0
            for i in data:
                var += (i - mean) ** 2
            p = 1 - var / (mean * N)
            n = round(mean / p)
            p = mean / n
            self.n = n
            self.p = p

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
        """Calculates PMF for given number of successes."""
        n = self.n
        p = self.p
        k = int(k)

        if k < 0 or k > self.n:
            return 0
        comb = self.factorial(n) / (self.factorial(k) * self.factorial(n - k))
        pmf = comb * (p ** k) * ((1 - p) ** (n - k))

        return pmf

    def cdf(self, k):
        """Calculates the value of CDF for a given number of successes."""
        if k < 0 or k > self.n:
            return 0
        k = int(k)
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
