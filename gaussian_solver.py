"""Gaussian elimination with partial pivoting and row-operation snapshots."""

import numpy as np


def gaussian_elimination(A, b):
    """Solve ``A x = b`` and return ``(x, snapshots)``.

    A copy of the augmented matrix is saved after each pivot column has been
    eliminated. Singular or malformed systems raise ``ValueError``.
    """
    coefficients = np.asarray(A, dtype=float)
    constants = np.asarray(b, dtype=float)

    if coefficients.ndim != 2 or coefficients.shape[0] != coefficients.shape[1]:
        raise ValueError("A must be a square 2D array")
    if constants.ndim != 1 or constants.size != coefficients.shape[0]:
        raise ValueError("b must be a 1D array matching A")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(constants)):
        raise ValueError("A and b must contain only finite values")

    size = constants.size
    augmented = np.column_stack((coefficients.copy(), constants.copy()))
    snapshots = []
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    tolerance = np.finfo(float).eps * max(1, size) * scale

    for column in range(size):
        pivot_row = column + int(np.argmax(np.abs(augmented[column:, column])))
        if abs(augmented[pivot_row, column]) <= tolerance:
            raise ValueError("A is singular or numerically singular")

        if pivot_row != column:
            augmented[[column, pivot_row]] = augmented[[pivot_row, column]]

        for row in range(column + 1, size):
            factor = augmented[row, column] / augmented[column, column]
            augmented[row, column:] -= factor * augmented[column, column:]
            augmented[row, column] = 0.0

        snapshots.append(augmented.copy())

    solution = np.zeros(size, dtype=float)
    for row in range(size - 1, -1, -1):
        remainder = augmented[row, -1] - np.dot(
            augmented[row, row + 1 : size], solution[row + 1 :]
        )
        solution[row] = remainder / augmented[row, row]

    return solution, snapshots
