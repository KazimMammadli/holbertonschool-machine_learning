#!/usr/bin/env python3
"""This module defines a function
to add two 2D matrices of ints/floats."""


def add_matrices2D(mat1, mat2):
    """Return the sum of two matrices."""
    if (len(mat1) != len(mat2) or
        len(mat1[0]) != len(mat2[0])):
        return None
    return [[mat1[i][j] + mat2[i][j]
            for j in range(len(mat1[0]))]
            for i in range(len(mat1))]
