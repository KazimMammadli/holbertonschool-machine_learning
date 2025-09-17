#!/usr/bin/env python3
"""Sensitivity."""
import numpy as np


def sensitivity(confusion):
    """Return a numpy.darray of shape (classes,) containing
    the sensitivity of each class."""
    n = confusion.shape[0]
    sum_cols = np.sum(confusion, axis=0)
    return np.array([confusion[i][i] / sum_cols[i] for i in range(n)])
