#!/usr/bin/env python3
"""K-means"""
import numpy as np


def initialize(X, k):
    """Performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    cluster_center_coords = np.random.uniform(X_min, X_max, (k, X.shape[1]))

    return cluster_center_coords


def compute_centroid_distance(X, centroids_coords):
    """Compute the distance between a point and the K centroids"""
    return np.sqrt(np.sum((X - centroids_coords[:, np.newaxis]) ** 2, axis=2))


def kmeans(X, k, iterations=1000):
    """Run the full Kmean algorithms"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, np.int) or iterations <= 0:
        return None, None

    centroids = initialize(X, k)

    for i in range(iterations):
        centroids_copy = centroids.copy()
        points_centroids_distance = compute_centroid_distance(X, centroids)
        clss = np.argmin(points_centroids_distance, axis=0)

        for j in range(k):
            if len(X[clss == j]) == 0:
                centroids[j] = initialize(X, 1)
            else:
                centroids[j] = np.mean(X[clss == j], axis=0)

        points_centroids_distance = compute_centroid_distance(X, centroids)
        clss = np.argmin(points_centroids_distance, axis=0)
        if np.all(centroids_copy == centroids):
            break

    return centroids, clss
