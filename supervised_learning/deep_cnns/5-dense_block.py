#!/usr/bin/env python3
"""Dense Block."""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Returns: The concatenated output of each layer within the
    Dense Block and the number of filters within the
    concatenated outputs, respectively.
    """
    he_normal = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        output = K.layers.BatchNormalization()(X)
        output = K.layers.ReLU()(output)
        output = K.layers.Conv2D(4 * growth_rate, (1, 1), padding='same',
                                 kernel_initializer=he_normal)(output)

        output = K.layers.BatchNormalization()(output)
        output = K.layers.ReLU()(output)
        output = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                                 kernel_initializer=he_normal)(output)

        X = K.layers.Concatenate()([X, output])
        nb_filters += growth_rate

    return X, nb_filters
