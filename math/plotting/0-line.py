#!/usr/bin/env python3
"""This module defines a function for plotting"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """Plot graph"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.xlim(0, 10)
    plt.plot(y, "r")
    plt.show()
