#!/usr/bin/env python3
"""Precision."""
import numpy as np


def precision(confusion):
    """Return a numpy.ndarray of shape (classes,) containing
    the precision of each class."""
    n = confusion.shape[0]
    sum_rows = np.sum(confusion, axis=0)
    return np.array([confusion[i][i] / sum_rows[i]
                     if sum_rows[i] != 0 else 0
                     for i in range(n)])
