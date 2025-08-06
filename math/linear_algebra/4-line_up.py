#!/usr/bin/env python3
"""This module defines 
a function to add two arrays of int/float."""


def add_arrays(arr1, arr2):
    """Return the sum of two arrays."""
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i]
            for i in range(len(arr1))]
