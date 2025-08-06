#!/usr/bin/env python3
"""This module is used to add two matrices."""


def add_matrices(mat1, mat2):
    """Return the sum of matrices."""
    if matrix_shape(mat1) == matrix_shape(mat2):
        return _add_matrices_recursive(mat1, mat2)
    return None


def matrix_shape(mat):
    """Return the shape of matrix."""
    shape = []
    while isinstance(mat, list):
        shape.append(len(mat))
        mat = mat[0]
    return shape


def _add_matrices_recursive(m1, m2):
    """Add matrix elements recursively."""
    if not isinstance(m1, list):
        return m1 + m2
    return [_add_matrices_recursive(a, b) for a, b in zip(m1, m2)]
