#!/usr/bin/env python3
"""
This module defines a function for
elementwise operations.
"""


def np_elementwise(mat1, mat2):
    """
    Return element-wise addition, subtraction,
    multiplication, and division.
    """
    return (mat1 + mat2, mat1 - mat2,
            mat1 * mat2, mat1 / mat2)
