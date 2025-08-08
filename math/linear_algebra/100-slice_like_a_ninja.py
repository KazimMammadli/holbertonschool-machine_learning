#!/usr/bin/env python3
"""This module is about slicing of matrices"""


def np_slice(matrix, axes={}):
    """Return the sliced matrix"""
    slicers = [slice(None)] * matrix.ndim 
    for axis, val in axes.items():
        slicers[axis] = slice(*val) 
    return matrix[tuple(slicers)]
