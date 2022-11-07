from typing import Tuple, Union, BinaryIO
from pathlib import Path
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None

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


def _write_csv_with_pandas(destination, table):
    # roughly a factor of 2 faster in Nov 2022
    df = pd.DataFrame(table)
    df.to_csv(destination, header=False, compression=None, index=False,
              sep=' ', line_terminator='\n')


def _write_csv_with_polars(destination, table):
    # polars only supports file names or binary targets as of version 0.14.24
    # otherwise factor of 1000 faster for large tables
    df = pl.DataFrame(table)
    df.write_csv(file=destination, has_header=False, sep=' ')


def _write_csv_with_numpy(destination, table):
    np.savetxt(destination, table, '%u')


def save_integer_csv(destination: Union[Path, str, BinaryIO], table: np.ndarray):
    if pl is not None and not hasattr(destination, 'encoding'):  # pragma: no cover
        _write_csv_with_polars(destination, table)

    elif pd is not None:  # pragma: no cover
        _write_csv_with_pandas(destination, table)

    else:
        _write_csv_with_numpy(destination, table)
