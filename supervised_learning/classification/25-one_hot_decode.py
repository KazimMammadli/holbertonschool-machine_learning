#!/usr/bin/env python3
"""One hot decoding"""
import numpy as np


def one_hot_decode(one_hot):
    """Convert a one-hot matrix into a vector of labels."""
    try:
        
        return np.argmax(one_hot, axis = 0)
    except Exception:
        return None
