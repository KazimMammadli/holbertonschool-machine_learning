#!/usr/bin/env python3
"""Entropy"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P
    affinities relative to a data point
    """
    Di_tmp = np.exp(-Di * beta)
    Di_sum = np.sum(Di_tmp)

    Pi = Di_tmp / Di_sum

    H_pi = - np.sum(Pi * np.log2(Pi))

    return H_pi, Pi
