#!/usr/bin/env python3
"""Adam optimizer."""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var,
                          grad, v, s, t):
    """Return the updated variable, the new first moment,
    and the new second moment, respectively."""
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - (alpha * v / (s + epsilon) ** 0.5)
    return var, v, s
