#!/usr/bin/env python3
"""One hot coding"""
import numpy as np


def one_hot_encode(Y, classes):
    """Convert a numeric label vector into a one-hot matrix."""
    if not isinstance(Y, np.ndarray):
        return None
    try:
        Y_One_Hot = np.eye(classes)[Y]
        return Y_One_Hot.T
    except Exception:
        return None
