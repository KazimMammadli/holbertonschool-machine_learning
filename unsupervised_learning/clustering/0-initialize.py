#!/usr/bin/env python3
"""Initialize K-means"""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    cluster_center_coords = np.random.uniform(X_min, X_max, (k, X.shape[1]))

    return cluster_center_coords
