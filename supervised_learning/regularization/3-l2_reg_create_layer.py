#!/usr/bin/env python3
"""Creating a Layer with L2 Regularization."""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Create a layer with l2 regularizer."""
    regularizer = tf.keras.regularizers.l2(lambtha)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer
    )(prev)
    return layer
