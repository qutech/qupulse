import numpy as np

try:
    import numba
    njit = numba.njit
except ImportError:
    numba = None
    njit = lambda x: x


@njit
def _is_monotonic_numba(x: np.ndarray) -> bool:
    # No early return because we optimize for the monotonic case and are branch-free.
    monotonic = True
    for i in range(1, len(x)):
        monotonic &= x[i - 1] <= x[i]
    return monotonic


def _is_monotonic_numpy(arr: np.ndarray) -> bool:
    # A bit faster than np.all(np.diff(arr) > 0) for small arrays
    # No difference for big arrays
    return np.all(arr[1:] >= arr[:-1])


if numba is None:
    is_monotonic = _is_monotonic_numpy
else:
    is_monotonic = _is_monotonic_numba



