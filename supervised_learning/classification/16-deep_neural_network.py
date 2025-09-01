#!/usr/bin/env python3
"""This module is about Deep Neural Network."""
import numpy as np


class DeepNeuralNetwork:
    """DNN class."""

    def __init__(self, nx, layers):
        """Constructor of the class."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if layers[i] <= 0 or not isinstance(layers[i], int):
                raise TypeError("layers must be a list of positive integers")

            nodes = layers[i]
            prev_node = nx if i == 0 else layers[i - 1]
            self.weights["W" + str(i + 1)] = (np.random.randn(nodes,
                                                              prev_node) *
                                              np.sqrt(2 / prev_node))
            self.weights["b{}".format(i + 1)] = np.zeros((nodes, 1))
