#!/usr/bin/env python3
"""Same Convolution."""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Return a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    padding_h = (kh - 1) // 2
    padding_w = (kw - 1) // 2
    new_w = w   # remains the same
    new_h = h   # remains the same
    convolved = np.zeros((m, new_h, new_w))
    padded_images = np.pad(images,
                           pad_width=((0, 0), (padding_h, padding_h),
                                      (padding_w, padding_w)),
                           mode='constant', constant_values=0)
    for i in range(new_h):
        for j in range(new_w):
            convolved[:, i, j] = np.sum(padded_images[:, i:i + kh, j:j + kw] *
                                        kernel, axis=(1, 2))
    return convolved
