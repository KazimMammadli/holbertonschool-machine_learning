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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if layers[i] <= 0 or not isinstance(layers[i], int):
                raise TypeError("layers must be a list of positive integers")

            nodes = layers[i]
            prev_node = nx if i == 0 else layers[i - 1]
            self.__weights["W" + str(i + 1)] = (np.random.randn(nodes,
                                                                prev_node) *
                                                np.sqrt(2 / prev_node))
            self.__weights["b{}".format(i + 1)] = np.zeros((nodes, 1))

    @property
    def L(self):
        """L getter."""
        return self.__L

    @property
    def cache(self):
        """cache getter."""
        return self.__cache

    @property
    def weights(self):
        """weights getter."""
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation."""
        self.__cache["A0"] = X
        for i in range(1, self.L + 1):
            W = self.__weights["W{}".format(i)]
            A = self.__cache["A{}".format(i-1)]
            b = self.__weights["b{}".format(i)]
            z = np.dot(W, A) + b
            self.__cache["A{}".format(i)] = 1 / (1 + np.exp(-z))

        return self.__cache["A{}".format(self.L)], self.__cache
