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
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def _softmax(x):
        """Compute softmax."""
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Forward hidden state for one time step."""
        hx = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh(hx @ self.Whf + self.bhf)

    def backward(self, h_next, x_t):
        """Backward hidden state for one time step."""
        hx = np.concatenate((h_next, x_t), axis=1)
        return np.tanh(hx @ self.Whb + self.bhb)

    def output(self, H):
        """
        Compute all outputs.

        H: concatenated hidden states (t, m, 2*h)
        Returns: Y (t, m, o)
        """
        t, m, _ = H.shape
        logits = H.reshape(t * m, -1) @ self.Wy + self.by
        Y = self._softmax(logits).reshape(t, m, -1)
        return Y
