#!/usr/bin/env python3
"""
Bidirectional RNN cell implementation.
"""

import numpy as np


class BidirectionalCell:
    """Represents a bidirectional RNN cell."""

    def __init__(self, i, h, o):
        """
        Initialize the bidirectional cell.

        i: input dimension
        h: hidden state dimension
        o: output dimension
        """
        # Forward direction
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        # Backward direction
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        # Output layer
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Forward direction hidden state for one time step.

        h_prev: previous hidden state (m, h)
        x_t: input data (m, i)
        """
        hx = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(hx @ self.Whf + self.bhf)
        return h_next
