#!/usr/bin/env python3
"""This module is about slicing of matrices."""


def cat_matrices(mat1, mat2, axis=0):
    """Return the concatenated matrix."""
    if (not matrix_compatibility(mat1, mat2, axis)):
        return None
    if (axis == 0):
        return mat1 + mat2
    result = deepcopy_matrix(mat1)
    if (axis == 1):
        [result[i].append(mat2[i][j])
         for j in range(len(mat2[0]))
         for i in range(len(mat2))]
    if (axis == 2):
        [result[i][j].append(mat2[i][j][k])
         for k in range(len(mat2[0][0]))
         for j in range(len(mat2[0]))
         for i in range(len(mat2))]
    if (axis == 3):
        [result[i][j][k].append(mat2[i][j][k][m])
         for m in range(len(mat2[0][0][0][0]))
         for k in range(len(mat2[0][0]))
         for j in range(len(mat2[0]))
         for i in range(len(mat2))]
    return result


def matrix_shape(mat):
    """Return the shape of matrix."""
    shape = []
    while isinstance(mat, list):
        shape.append(len(mat))
        mat = mat[0]
    return shape


def matrix_compatibility(mat1, mat2, axis):
    """Check the compatibility of two matrices
    for concatenation."""
    for i in range(len(matrix_shape(mat1))):
        if i == axis:
            continue
        if matrix_shape(mat1)[i] != matrix_shape(mat2)[i]:
            return False
    return True


def deepcopy_matrix(mat):
    """Return the deepcopy of the matrix."""
    if isinstance(mat, list):
        return [deepcopy_matrix(element) for element in mat]
    else:
        return mat
