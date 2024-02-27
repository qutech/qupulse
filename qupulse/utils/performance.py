from typing import *
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

def compress_array_LZ77(array:np.ndarray, allow_intermediates:bool=True, using_diffs:bool=True) -> List[Tuple[int, int, np.ndarray]]:
    """ This function applies LZ77 to compress a array.
    """

    assert len(array.shape) == 2
    assert array.shape[0] > 0

    array_to_compress = array.copy() if not using_diffs else np.concatenate([array[None, 0, :], np.diff(array, axis=0)], axis=0)
    atc = array_to_compress
    # print(f"{atc=}")

    compressed_stack = [(0, 0, array_to_compress[0, :])]
    i = 1
    while i < atc.shape[0]:
        os = [0, ]
        ds = [0, ]
        
        o = 1
        d = 0
        while o <= i:
            d = 0
            while i+d < atc.shape[0] and np.array_equal(atc[i-o+(d%o)], atc[i+d]):
                d += 1
            if d > 0:
                os.append(o)
                ds.append(d)
            o += 1

        os = np.array(os)
        ds = np.array(ds)

        if os[-1] > 0:
            if os[0] == 0:
                os = os[1:]
                ds = ds[1:]
            if not allow_intermediates:
                ds = (ds//os)*os
                mask = (os<=ds) & (ds > 0)
                os = os[mask]
                ds = ds[mask]
            if len(os) == 0:
                os, ds = [0], [0]
            j = len(ds)-np.argmax(ds[::-1])-1
            sos, sds = os[j], ds[j]
            i += ds[j]+1
        else:
            sos, sds = 0, 0
            i += 0+1
        if i-1 < atc.shape[0]:
            sa = atc[i-1]
        else:
            sa = None
        compressed_stack.append((sos, sds, sa))

    return compressed_stack

def uncompress_array_LZ77(compressed_array) -> np.ndarray:
    output = []

    for o, l, c in compressed_array:
        initial_length = len(output)
        for i in range(l):
            output.append(output[initial_length-o+i])
        if c is not None:
            output.append(c)

    return np.array(output)



