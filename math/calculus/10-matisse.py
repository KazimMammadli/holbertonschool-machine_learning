#!/usr/env/bin python3
"""
This module defines a function to find
coefficients of derivatives.
"""


def poly_derivative(poly):
    """
    Return list of coefficients representing the
    derivative of the polynomial.
    """
    if not isinstance(poly, list):
        return None
    derivative = [poly[i] * i for i in range(1, len(poly))]
    if all(coef == 0 for coef in derivative):
        return [0]
    return derivative
