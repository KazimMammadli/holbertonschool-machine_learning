# !/usr/bin/env python3
"""
This module defines a function to concatenate two
matrices along a specific axis.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Return concatenated matrix"""
    if axis in (0, 1):
        return np.concatenate((mat1, mat2), axis)
    else:
        return None
