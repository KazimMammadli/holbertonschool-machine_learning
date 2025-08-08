# !/usr/bin/env python3
"""This module is about slicing of matrices"""
import numpy as np


def np_slice(matrix, axes={}):
    """Return the sliced matrix"""
    for key, val in axes.items():
        if key == 0:
            matrix = matrix[slice(*val)]
        if key == 1:
            matrix = matrix[:, slice(*val)]
        if key == 2:
            matrix = matrix[:, :, slice(*val)]
    return matrix
