#!/usr/bin/env python3
"""Cost"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation
    """
    return (np.sum(np.maximum(P, 1e-12) * np.log(np.maximum(P, 1e-12)
                                                 / (np.maximum(Q, 1e-12)))))
