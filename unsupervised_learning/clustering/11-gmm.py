#!/usr/bin/env python3
"""some comments"""
import sklearn.mixture


def gmm(X, k):
    """
    Use the GMN of scikit learn
    """
    gmm = sklearn.mixture.GaussianMixture(k).fit(X)

    return (gmm.weights_, gmm.means_, gmm.covariances_,
            gmm.predict(X), gmm.bic(X))
