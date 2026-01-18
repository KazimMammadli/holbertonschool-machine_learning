#!/usr/bin/env python3
"""
Forward propagation for a deep (stacked) RNN.
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Perform forward propagation for a deep RNN.

    rnn_cells: list of RNNCell instances (length l)
    X: input data (t, m, i)
    h_0: initial hidden states (l, m, h)

    Returns: H, Y
    H: hidden states (t + 1, l, m, h)
    Y: outputs (t, m, o)
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        x = X[step]
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, _ = rnn_cells[layer].forward(h_prev, x)
            H[step + 1, layer] = h_next
            x = h_next  # output of this layer becomes input to next layer

        # Final layer output (recompute with last forward to get y)
        # More efficient: store y from last layer call inside the loop
        _, y = rnn_cells[-1].forward(H[step, l - 1], X[step] if l == 1 else H[step + 1, l - 2])
        # But the above is messy; instead, store y during the last layer pass.
