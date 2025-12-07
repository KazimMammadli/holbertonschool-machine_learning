#!/usr/bin/env python3
"""some comments"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Compute a clustering using the agglomerative algos
    """

    hierarchy = scipy.cluster.hierarchy
    links = hierarchy.linkage(X, method='ward')
    clss = hierarchy.fcluster(links, t=dist, criterion='distance')

    plt.figure()
    hierarchy.dendrogram(links, color_threshold=dist)
    plt.show()

    return clss
