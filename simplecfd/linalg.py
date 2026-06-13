from __future__ import annotations

import numpy as np

from simplecfd.coefficients import LinearSystem


def tdma(system: LinearSystem) -> np.ndarray:
    """Solve a tridiagonal linear system with the Thomas algorithm.

    The system is represented as:

        lower[i] * x[i-1] + diagonal[i] * x[i] + upper[i] * x[i+1] = rhs[i]

    `lower[0]` and `upper[-1]` are ignored. Inputs are copied so the original
    `LinearSystem` remains unchanged.
    """

    lower = np.asarray(system.lower, dtype=float).copy()
    diagonal = np.asarray(system.diagonal, dtype=float).copy()
    upper = np.asarray(system.upper, dtype=float).copy()
    rhs = np.asarray(system.rhs, dtype=float).copy()

    _validate_tridiagonal_arrays(lower, diagonal, upper, rhs)

    n = diagonal.size
    if n == 0:
        return np.array([], dtype=float)

    for i in range(1, n):
        if diagonal[i - 1] == 0.0:
            raise ZeroDivisionError(f"zero pivot encountered at row {i - 1}")
        factor = lower[i] / diagonal[i - 1]
        diagonal[i] -= factor * upper[i - 1]
        rhs[i] -= factor * rhs[i - 1]

    if diagonal[-1] == 0.0:
        raise ZeroDivisionError(f"zero pivot encountered at row {n - 1}")

    solution = np.zeros(n, dtype=float)
    solution[-1] = rhs[-1] / diagonal[-1]

    for i in range(n - 2, -1, -1):
        if diagonal[i] == 0.0:
            raise ZeroDivisionError(f"zero pivot encountered at row {i}")
        solution[i] = (rhs[i] - upper[i] * solution[i + 1]) / diagonal[i]

    return solution


def _validate_tridiagonal_arrays(*arrays: np.ndarray) -> None:
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("all tridiagonal arrays must be 1D")

    sizes = {array.size for array in arrays}
    if len(sizes) != 1:
        raise ValueError("lower, diagonal, upper, and rhs must have the same size")
