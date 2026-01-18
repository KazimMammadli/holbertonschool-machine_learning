#!/usr/bin/env python3
"""
Forward propagation for a simple RNN.
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for an RNN over t time steps.

    rnn_cell: instance of RNNCell
    X: input data (t, m, i)
    h_0: initial hidden state (m, h)

    Returns: H, Y
    H: hidden states (t + 1, m, h)
    Y: outputs (t, m, o)
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0
    h_prev = h_0

    for step in range(t):
        h_prev, y = rnn_cell.forward(h_prev, X[step])
        H[step + 1] = h_prev
        Y[step] = y

    return H, Y
