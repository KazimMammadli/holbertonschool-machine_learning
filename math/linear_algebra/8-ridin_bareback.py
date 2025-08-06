#!/usr/bin/env python3
"""
This module defines a function for
matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """Return the resultant matrix of multiplication."""
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for row in mat1:
        new_row = []
        for k in range(len(mat2[0])):
            element = 0
            for j in range(len(mat2)):
                element += mat2[j][k] * row[j]
            new_row.append(element)
        result.append(new_row)
    return result
