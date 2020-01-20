import numpy as np

try:
    import numba
    njit = numba.njit
except ImportError:
    numba = None
    njit = lambda x: x


@njit
def _is_monotonic_numba(x: np.ndarray):
    for i in range(1, len(x)):
        if x[i - 1] >= x[i]:
            return False
    return True


def _is_monotonic_numpy(arr: np.ndarray):
    return np.all(arr[1:] > arr[:-1])


if numba is None:
    is_monotonic = _is_monotonic_numpy
else:
    is_monotonic = _is_monotonic_numba



