#!/usr/bin/env python3
"""
Deep RNN forward propagation.
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Forward propagation for a deep RNN.

    rnn_cells: list of RNNCell (length l)
    X: input data (t, m, i)
    h_0: initial hidden states (l, m, h)

    Returns: H, Y
    H: (t + 1, l, m, h)
    Y: (t, m, o)
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        x = X[step]  # input to layer 0 at this time step
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x)
            H[step + 1, layer] = h_next
            x = h_next  # feed to next layer

            if layer == l - 1:
                Y[step] = y

    return H, Y
