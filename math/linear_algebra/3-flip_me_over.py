#!/usr/bin/env python3
"""This module defines a function to transpose a matrix."""


def matrix_transpose(matrix):
    """Return transpose of a matrix
    second method
    [[matrix[i][j] for i in range(len(matrix))]
    for j in range(len(matrix[0]))]"""
    new_matrix = []
    for j in range(len(matrix[0])):
        new_row = []
        for i in range(len(matrix)):
            new_row.append(matrix[i][j])
        new_matrix.append(new_row)
    return new_matrix
