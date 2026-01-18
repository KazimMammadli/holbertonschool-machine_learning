#!/usr/bin/env python3
"""
LSTM cell implementation.
"""

import numpy as np


class LSTMCell:
    """Represents an LSTM cell."""

    def __init__(self, i, h, o):
        """
        Initialize the LSTM cell.

        i: input dimension
        h: hidden state dimension
        o: output dimension
        """
        # Forget gate
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))

        # Update (input) gate
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))

        # Candidate cell state
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))

        # Output gate
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))

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

    def forward(self, h_prev, c_prev, x_t):
        """
        Forward propagation for one time step.

        h_prev: previous hidden state (m, h)
        c_prev: previous cell state (m, h)
        x_t: input data (m, i)
        """
        hx = np.concatenate((h_prev, x_t), axis=1)

        # Gates
        f = self._sigmoid(hx @ self.Wf + self.bf)
        u = self._sigmoid(hx @ self.Wu + self.bu)
        o = self._sigmoid(hx @ self.Wo + self.bo)

        # Candidate cell state
        c_hat = np.tanh(hx @ self.Wc + self.bc)

        # Next cell state
        c_next = f * c_prev + u * c_hat

        # Next hidden state
        h_next = o * np.tanh(c_next)

        # Output
        y = self._softmax(h_next @ self.Wy + self.by)

        return h_next, c_next, y
