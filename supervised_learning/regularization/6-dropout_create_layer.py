#!/usr/bin/env python3
"""Create a Layer with Dropout."""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Create a layer of a neural network using dropout:"""
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation
    )(prev)
    layer = tf.keras.layers.Dropout(rate=1-keep_prob)(layer, training=training)
    return layer
