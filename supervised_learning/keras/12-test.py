#!/usr/bin/env python3
"""Testing model."""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Return loss and accuracy of the model."""
    verbose = 1 if verbose else 0
    result = network.evaluate(x_test=data, y_test=labels, verbose=verbose)
    return result
