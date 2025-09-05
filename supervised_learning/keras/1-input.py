#!/usr/bin/env python3
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
"""Input class."""


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Building model with Input class."""
    regularizer = regularizers.L2(lambtha)
    dropout = 1 - keep_prob
    iinputs = Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = Dense(units=layers[i],
                  activation=activations[i],
                  kernel_regularizer=regularizer)(x)

        if i < len(layers) - 1:
            x = Dropout(rate=dropout)(x)

    model = Model(inputs=inputs, outputs=x)
    return model
