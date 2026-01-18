#!/usr/bin/env python3
"""
Forward propagation for a bidirectional RNN.
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Perform forward propagation for a bidirectional RNN.

    bi_cell: BidirectionalCell instance
    X: input data (t, m, i)
    h_0: initial forward hidden state (m, h)
    h_t: initial backward hidden state (m, h)

    Returns: H, Y
    H: concatenated hidden states (t, m, 2*h)
    Y: outputs (t, m, o)
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))

    Hf[0] = h_0
    Hb[t] = h_t

    # Forward pass
    for step in range(t):
        Hf[step + 1] = bi_cell.forward(Hf[step], X[step])

    # Backward pass
    for step in range(t - 1, -1, -1):
        Hb[step] = bi_cell.backward(Hb[step + 1], X[step])

    # Concatenate (exclude initialized states)
    H = np.concatenate((Hf[1:], Hb[:-1]), axis=2)  # (t, m, 2*h)

    # Outputs
    Y = bi_cell.output(H)

    return H, Y
