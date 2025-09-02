#!/usr/bin/env python3
"""This module is about Deep Neural Network."""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


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

    def cost(self, Y, A):
        """Cost function"""
        c = -1 / len(Y[0]) * np.sum((Y * np.log(A)) + (1 - Y)
                                    * np.log(1.0000001 - A))
        return c

    def evaluate(self, X, Y):
        """Evaluate the model"""
        self.forward_prop(X)
        pred = self.__cache['A{}'.format(self.__L)]
        return (np.where(pred >= 0.5, 1, 0),
                self.cost(Y, pred))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient descent"""
        m = len(Y[0])
        AL = cache['A{}'.format(self.__L)]
        dZl = AL - Y
        for i in range(self.__L, 0, -1):
            Al = cache['A{}'.format(i-1)]
            dwl = (dZl @ Al.T) / m
            dbl = (np.sum(dZl, axis=1, keepdims=True)) / m

            Al_prev = cache['A{}'.format(i-1)]
            Wl = self.__weights['W{}'.format(i)]
            if i > 1:
                dZl = (Wl.T @ dZl) * (Al_prev * (1-Al_prev))
            self.__weights['W{}'.format(i)] -= alpha * dwl
            self.__weights['b{}'.format(i)] -= alpha * dbl

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Training."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")

            if step <= 0 and step <= iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iteration_list = []

        for i in range(iterations + 1):
            cache_l, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            if i % step == 0 or i == iterations:
                c = self.cost(Y, cache_l)
                costs.append(c)
                iteration_list.append(i)

                if verbose:
                    print(f"Cost after {i} iterations: {c}")

        if graph:
            plt.plot(iteration_list, costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Training Cost")
            plt.show()
        self.forward_prop(X)
        return self.evaluate(X, Y)

    def save(self, filename):
        """Save the instance object to a file in pickle format."""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object."""
        with open(filename, "rb") as f:
            if os.path.exists(filename):
                return pickle.load(f)
            else:
                return None
