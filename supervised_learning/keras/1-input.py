#!/usr/bin/env python3
import tensorflow.keras as K
"""Input class."""


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Building model with Input class."""
    regularizer = K.regularizers.L2(lambtha)
    dropout = 1 - keep_prob
    inputs = K.layers.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(units=layers[i],
                           activation=activations[i],
                           kernel_regularizer=regularizer)(x)

        if i < len(layers) - 1:
            x = K.layers.Dropout(rate=dropout)(x)

    model = K.models.Model(inputs=inputs, outputs=x)
    return model
