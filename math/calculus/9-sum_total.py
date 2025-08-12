#!/usr/bin/env python3
"""This module defines a function for sum of squares."""


def summation_i_squared(n):
    """Return the sum of square values."""
    if not isinstance(n, int) or n < 1:
        return None
    return int(n * (n + 1) * (2 * n + 1)/6)
