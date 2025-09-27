#!/usr/bin/env python3
"""Strided Convolution."""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Return a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2

    if padding == 'valid':
        ph = 0
        pw = 0

    if isinstance(padding, tuple):
        ph, pw = padding

    sh, sw = stride
    new_w = (w + 2 * pw - kw) // sw + 1
    new_h = (h + 2 * ph - kh) // sh + 1
    convolved = np.zeros((m, new_h, new_w))
    padded_images = np.pad(images,
                           pad_width=((0, 0), (ph, ph),
                                      (pw, pw)),
                           mode='constant', constant_values=0)

    for i in range(new_h):
        for j in range(new_w):
            convolved[:, i, j] = np.sum(padded_images[:, i * sh:i * sh + kh,
                                        j * sw:j * sw + kw] *
                                        kernel, axis=(1, 2))
    return convolved
