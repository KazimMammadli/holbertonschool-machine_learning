#!/usr/bin/env python3
"""
GRU cell implementation.
"""

import numpy as np


class GRUCell:
    """Represents a GRU cell."""

    def __init__(self, i, h, o):
        """
        Initialize the GRU cell.

        i: input dimension
        h: hidden state dimension
        o: output dimension
        """
        # Update gate
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))

        # Reset gate
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

        # Candidate hidden state
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))

        # Output layer
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    @staticmethod
    def _sigmoid(x):
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _softmax(x):
        """Softmax activation."""
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step.

        h_prev: previous hidden state (m, h)
        x_t: input data (m, i)
        """
        hx = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = self._sigmoid(hx @ self.Wz + self.bz)

        # Reset gate
        r = self._sigmoid(hx @ self.Wr + self.br)

        # Candidate hidden state
        rhx = np.concatenate((r * h_prev, x_t), axis=1)
        h_hat = np.tanh(rhx @ self.Wh + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_hat

        # Output
        y = self._softmax(h_next @ self.Wy + self.by)

        return h_next, y
