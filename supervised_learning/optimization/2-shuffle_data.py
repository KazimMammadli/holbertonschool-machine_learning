#!/usr/bin/env python3
"""Shuffling data."""
import numpy as np


def shuffle_data(X, Y):
    m = len(X)
    perm = np.random.permutation(m)
    return X[perm], Y[perm]
