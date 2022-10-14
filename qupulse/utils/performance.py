from typing import Tuple, Union, BinaryIO
import numpy as np

try:
    import numba
    njit = numba.njit(cache=True)
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


@njit
def _time_windows_to_samples_sorted_numba(begins, lengths,
                                          sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    begins_as_sample = np.zeros(len(begins), dtype=np.uint64)
    lengths_as_sample = np.zeros(len(lengths), dtype=np.uint64)
    for idx in range(len(begins)):
        begins_as_sample[idx] = round(begins[idx] * sample_rate)
        lengths_as_sample[idx] = np.uint64(lengths[idx] * sample_rate)
    return begins_as_sample, lengths_as_sample


@njit
def _time_windows_to_samples_numba(begins, lengths,
                                   sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    if _is_monotonic_numba(begins):
        # factor 10 faster
        begins_as_sample, lengths_as_sample = _time_windows_to_samples_sorted_numba(begins, lengths, sample_rate)
    else:
        sorting_indices = np.argsort(begins)

        begins_as_sample = np.zeros(len(begins), dtype=np.uint64)
        lengths_as_sample = np.zeros(len(lengths), dtype=np.uint64)
        for new_pos, old_pos in enumerate(sorting_indices):
            begins_as_sample[new_pos] = round(begins[old_pos] * sample_rate)
            lengths_as_sample[new_pos] = np.uint64(lengths[old_pos] * sample_rate)
    return begins_as_sample, lengths_as_sample


def _time_windows_to_samples_numpy(begins: np.ndarray, lengths: np.ndarray,
                                   sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    sorting_indices = np.argsort(begins)
    begins = np.rint(begins * sample_rate).astype(dtype=np.uint64)
    lengths = np.floor(lengths * sample_rate).astype(dtype=np.uint64)

    begins = begins[sorting_indices]
    lengths = lengths[sorting_indices]
    return begins, lengths


def time_windows_to_samples(begins: np.ndarray, lengths: np.ndarray,
                            sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """"""
    if numba is None:
        begins, lengths = _time_windows_to_samples_numpy(begins, lengths, sample_rate)
    else:
        begins, lengths = _time_windows_to_samples_numba(begins, lengths, sample_rate)
    begins.flags.writeable = False
    lengths.flags.writeable = False
    return begins, lengths


@njit
def _fmt_uint_into(i, arr):
    for j in range(len(arr)):
        i, d = divmod(i, 10)
        arr[j] = np.uint8(d) + 48  # 0
        if i == 0:
            break
    else:
        ValueError()

    n = j + 1
    arr[:n] = arr[:n][::-1]
    return n


@njit
def _fmt_int_into(i, arr):
    assert len(arr) > 0
    if i < 0:
        arr[0] = 45  # -
        i = -i
        num_start = 1
    else:
        num_start = 0
    return _fmt_uint_into(i, arr[num_start:]) + num_start


@njit
def _fmt_int_table(arr, sep: np.uint8) -> np.ndarray:
    assert arr.shape[1] == 2
    out_arr = np.empty(16 * len(arr), np.uint8)
    idx = 0
    for a, b in arr:
        idx += _fmt_int_into(a, out_arr[idx:])

        out_arr[idx] = sep
        idx += 1

        idx += _fmt_int_into(b, out_arr[idx:])

        out_arr[idx] = 10  # EOL
        idx += 1
    return out_arr[:idx]


def write_int_table(target: Union[str, BinaryIO], array: np.ndarray, sep: str):
    """Write a table of integers. Optimized for two columns"""
    if isinstance(target, str):
        target = open(target, 'wb')
        close_target = True
    else:
        close_target = False

    try:
        if numba is not None and array.shape[1] == 2 and array.dtype == np.dtype('uint32'):
            sep, = sep.encode('ascii')
            binary = _fmt_int_table(array, np.uint8(sep))
            target.write(binary.tobytes())
        else:
            np.savetxt(target, array, '%u', delimiter=sep)  # pragma: no cover
    finally:
        if close_target:
            target.close()
