#!/usr/bin/env python3
"""some comments"""
import sklearn.cluster


def kmeans(X, k):
    """
    Use skleanr Kmean
    """
    kmeans = sklearn.cluster.KMeans(k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
