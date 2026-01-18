#!/usr/bin/env python3
"""
Simple RNN cell implementation.
"""

import numpy as np


class RNNCell:
    """Represents a simple RNN cell."""

    def __init__(self, i, h, o):
        """
        Initialize the RNN cell.

        i: input dimension
        h: hidden state dimension
        o: output dimension
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def _softmax(z):
        """Compute softmax."""
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step.

        h_prev: previous hidden state (m, h)
        x_t: input data (m, i)
        """
        hx = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(hx @ self.Wh + self.bh)
        y = self._softmax(h_next @ self.Wy + self.by)
        return h_next, y
