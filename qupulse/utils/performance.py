import warnings
from typing import Tuple
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


def _shrink_overlapping_windows_numpy(begins, lengths) -> bool:
    ends = begins + lengths

    overlaps = np.zeros_like(ends)
    np.maximum(ends[:-1] - begins[1:], 0, out=overlaps[1:])

    if np.any(overlaps >= lengths):
        raise ValueError("Overlap is bigger than measurement window")
    if np.any(overlaps > 0):
        begins += overlaps
        lengths -= overlaps
        return True
    return False


@njit
def _shrink_overlapping_windows_numba(begins, lengths) -> bool:
    shrank = False
    for idx in range(len(begins) - 1):
        end = begins[idx] + lengths[idx]
        next_begin = begins[idx + 1]

        if end > next_begin:
            overlap = end - next_begin
            shrank = True
            if lengths[idx + 1] > overlap:
                begins[idx + 1] += overlap
                lengths[idx + 1] -= overlap
            else:
                raise ValueError("Overlap is bigger than measurement window")
    return shrank


class WindowOverlapWarning(RuntimeWarning):
    pass


def shrink_overlapping_windows(begins, lengths, use_numba: bool = numba is not None) -> Tuple[np.array, np.array]:
    """Shrink windows in place if they overlap. Emits WindowOverlapWarning if a window was shrunk.

    Raises:
        ValueError: if the overlap is bigger than a window.

    Warnings:
        WindowOverlapWarning
    """
    if use_numba:
        backend = _shrink_overlapping_windows_numba
    else:
        backend = _shrink_overlapping_windows_numpy
    begins = begins.copy()
    lengths = lengths.copy()
    if backend(begins, lengths):
        warnings.warn("Found overlapping measurement windows which are automatically shrunken if possible.",
                      category=WindowOverlapWarning)
    return begins, lengths


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


if numba is None:
    is_monotonic = _is_monotonic_numpy
else:
    is_monotonic = _is_monotonic_numba
