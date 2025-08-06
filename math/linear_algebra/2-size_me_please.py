#!/usr/bin/env python3
"""This module defines a function to determine shape of the matrix."""


def matrix_shape(matrix):
    """Returns the shape of the matrix as a list."""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
