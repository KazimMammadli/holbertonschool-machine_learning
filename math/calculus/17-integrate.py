#!/usr/bin/env python3
"""This module is used to define integral function."""


def poly_integral(poly, C=0):
    """
    Return the coefficients representing the
    integral of the polynomial.
    """
    if (not isinstance(poly, list) or
       not all(isinstance(coef, (int, float)) for coef in poly) or
       len(poly) == 0 or not isinstance(C, (int, float))):
        return None

    if all(coef == 0 for coef in poly):
        poly = []

    new_list = [poly[i] / (i + 1) for i in range(len(poly))]

    return [C] + [int(x) if x.is_integer() else x for x in new_list]
