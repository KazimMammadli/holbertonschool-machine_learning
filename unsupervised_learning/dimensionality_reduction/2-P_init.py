#!/usr/bin/env python3
"""Initialize t-SNE"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to
    calculate the P affinities in t-SNE
    """
    n, _ = X.shape

    D = (np.sum(X**2, axis=1) + np.sum(X**2, axis=1)[..., np.newaxis]
         - 2 * np.dot(X, X.T))
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))

    b = np.ones((n, 1))

    H = np.log2(perplexity)

    return D, P, b, H
