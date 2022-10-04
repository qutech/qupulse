from typing import Collection, Sequence, Tuple, Union, Optional
import itertools

import numpy as np

try:
    from autologging import traced
except ImportError:
    def traced(obj):
        """Noop traced that is used if autologging package is not available"""
        return obj

from qupulse._program.waveforms import Waveform
from qupulse.utils.types import TimeType
from qupulse.utils import pairwise

try:
    import numba
    njit = numba.njit
except ImportError:
    numba = None
    njit = lambda x: x

try:
    import zhinst
except ImportError:  # pragma: no cover
    zhinst = None

__all__ = ['voltage_to_uint16', 'get_sample_times', 'traced', 'zhinst_voltage_to_uint16']


@njit
def _voltage_to_uint16_numba(voltage: np.ndarray, output_amplitude: float, output_offset: float, resolution: int) -> np.ndarray:
    """Implementation detail that can be compiled with numba. This code is very slow without numba."""
    out_of_range = False
    scale = (2 ** resolution - 1) / (2 * output_amplitude)
    result = np.empty_like(voltage, dtype=np.uint16)
    for i in range(voltage.size):
        x = voltage[i] - output_offset
        if np.abs(x) > output_amplitude:
            out_of_range = True
        result[i] = np.uint16(np.rint((x + output_amplitude) * scale))

    if out_of_range:
        raise ValueError('Voltage out of range')

    return result


def _voltage_to_uint16_numpy(voltage: np.ndarray, output_amplitude: float, output_offset: float, resolution: int) -> np.ndarray:
    """Implementation detail to be used if numba is not available."""
    non_dc_voltage = voltage - output_offset
    if np.any(np.abs(non_dc_voltage) > output_amplitude):
        # should get more context in wrapper function
        raise ValueError('Voltage out of range')

    non_dc_voltage += output_amplitude
    non_dc_voltage *= (2**resolution - 1) / (2*output_amplitude)
    return np.rint(non_dc_voltage).astype(np.uint16)


def voltage_to_uint16(voltage: np.ndarray, output_amplitude: float, output_offset: float, resolution: int) -> np.ndarray:
    """Convert values of the range
       [output_offset - output_amplitude, output_offset + output_amplitude)
    to uint16 in the range
       [0, 2**resolution)

    output_offset - output_amplitude -> 0
    output_offset                    -> 2**(resolution - 1)
    output_offset + output_amplitude -> 2**resolution - 1

    Args:
        voltage: input voltage. read-only
        output_amplitude: input divided by this
        output_offset: is subtracted from input
        resolution: Target resolution in bits (determines the output range)

    Raises:
        ValueError if the voltage is out of range or the resolution is not an integer

    Returns:
        (voltage - output_offset + output_amplitude) * (2**resolution - 1) / (2*output_amplitude) as uint16
    """
    if resolution < 1 or not isinstance(resolution, int):
        raise ValueError('The resolution must be an integer > 0')

    try:
        if numba:
            impl = _voltage_to_uint16_numba
        else:
            impl = _voltage_to_uint16_numpy
        return impl(voltage, output_amplitude, output_offset, resolution)
    except ValueError as err:
        raise ValueError('Voltage out of range', dict(voltage=voltage,
                                                      output_offset=output_offset,
                                                      output_amplitude=output_amplitude)) from err


def find_positions(data: Sequence, to_find: Sequence) -> np.ndarray:
    """Find indices of the first occurrence of the elements of to_find in data. Elements that are not in data result in
    -1"""
    data_sorter = np.argsort(data)

    pos_left = np.searchsorted(data, to_find, side='left', sorter=data_sorter)
    pos_right = np.searchsorted(data, to_find, side='right', sorter=data_sorter)

    found = pos_left < pos_right

    positions = np.full_like(to_find, fill_value=-1, dtype=np.int64)
    positions[found] = data_sorter[pos_left[found]]

    return positions


def get_waveform_length(waveform: Waveform,
                        sample_rate_in_GHz: TimeType, tolerance: float = 1e-10) -> int:
    """Calculates the number of samples in a waveform

    If only one waveform is given, the number of samples has shape ()

    Raises a ValueError if the waveform has a length that is zero or not a multiple of the inverse sample rate.

    Args:
        waveform: A waveform
        sample_rate_in_GHz: The sample rate in GHz
        tolerance: Allowed deviation from an integer sample count

    Returns:
        Number of samples for the waveform
    """
    segment_length = waveform.duration * sample_rate_in_GHz

    # __round__ is implemented for Fraction and gmpy2.mpq
    rounded_segment_length = round(segment_length)

    if abs(segment_length - rounded_segment_length) > tolerance:
        deviation = abs(segment_length - rounded_segment_length)
        raise ValueError("Error while sampling waveforms. One waveform has a non integer length in samples of "
                         "{segment_length} at the given sample rate of {sample_rate}GHz. This is a deviation of "
                         "{deviation} from the nearest integer {rounded_segment_length}."
                         "".format(segment_length=segment_length,
                                   sample_rate=sample_rate_in_GHz,
                                   deviation=deviation,
                                   rounded_segment_length=rounded_segment_length))
    if rounded_segment_length <= 0:
        raise ValueError("Error while sampling waveform. Waveform has a length <= zero at the given sample "
                         "rate of %rGHz" % sample_rate_in_GHz)
    segment_length = np.uint64(rounded_segment_length)

    return segment_length


def get_sample_times(waveforms: Union[Collection[Waveform], Waveform],
                     sample_rate_in_GHz: TimeType, tolerance: float = 1e-10) -> Tuple[np.array, np.array]:
    """Calculates the sample times required for the longest waveform in waveforms and returns it together with an array
    of the lengths.

    If only one waveform is given, the number of samples has shape ()

    Raises a ValueError if any waveform has a length that is zero or not a multiple of the inverse sample rate.

    Args:
        waveforms: A waveform or a sequence of waveforms
        sample_rate_in_GHz: The sample rate in GHz
        tolerance: Allowed deviation from an integer sample count

    Returns:
        Array of sample times sufficient for the longest waveform
        Number of samples of each waveform
    """
    if not isinstance(waveforms, Collection):
        sample_times, n_samples = get_sample_times([waveforms], sample_rate_in_GHz)
        return sample_times, n_samples.squeeze()

    assert len(waveforms) > 0, "An empty waveform list is not allowed"

    segment_lengths = []
    for waveform in waveforms:
        rounded_segment_length = get_waveform_length(waveform, sample_rate_in_GHz=sample_rate_in_GHz, tolerance=tolerance)
        segment_lengths.append(rounded_segment_length)

    segment_lengths = np.asarray(segment_lengths, dtype=np.uint64)
    time_array = np.arange(np.max(segment_lengths), dtype=float) / float(sample_rate_in_GHz)

    return time_array, segment_lengths


@njit
def _zhinst_voltage_to_uint16_numba(size: int, ch1: Optional[np.ndarray], ch2: Optional[np.ndarray],
                                    m1_front: Optional[np.ndarray], m1_back: Optional[np.ndarray],
                                    m2_front: Optional[np.ndarray], m2_back: Optional[np.ndarray]) -> np.ndarray:
    """Numba targeted implementation"""
    data = np.zeros((size, 3), dtype=np.uint16)

    scale = float(2**15 - 1)

    invalid_value = None

    def has_invalid_size(arr):
        return arr is not None and len(arr) != size

    if has_invalid_size(ch1) or has_invalid_size(ch2) or has_invalid_size(m1_front) or has_invalid_size(m1_back) or has_invalid_size(m2_front) or has_invalid_size(m2_back):
        raise ValueError("One of the inputs does not have the given size.")

    for i in range(size):
        if ch1 is not None:
            if not abs(ch1[i]) <= 1:
                invalid_value = ch1[i]
            data[i, 0] = ch1[i] * scale
        if ch2 is not None:
            if not abs(ch2[i]) <= 1:
                invalid_value = ch2[i]
            data[i, 1] = ch2[i] * scale
        if m1_front is not None:
            data[i, 2] |= (m1_front[i] != 0)
        if m1_back is not None:
            data[i, 2] |= (m1_back[i] != 0) << 1
        if m2_front is not None:
            data[i, 2] |= (m2_front[i] != 0) << 2
        if m2_back is not None:
            data[i, 2] |= (m2_back[i] != 0) << 3

    if invalid_value is not None:
        # we can only use compile time constants here
        raise ValueError('Encountered an invalid value in channel data (not in [-1, 1])')

    return data.ravel()


def _zhinst_voltage_to_uint16_numpy(size: int, ch1: Optional[np.ndarray], ch2: Optional[np.ndarray],
                                    m1_front: Optional[np.ndarray], m1_back: Optional[np.ndarray],
                                    m2_front: Optional[np.ndarray], m2_back: Optional[np.ndarray]) -> np.ndarray:
    """Fallback implementation if numba is not available"""
    markers = (m1_front, m1_back, m2_front, m2_back)

    def check_invalid_values(ch_data):
        # like this to catch NaN
        invalid = ~(np.abs(ch_data) <= 1)
        if np.any(invalid):
            raise ValueError('Encountered an invalid value in channel data (not in [-1, 1])', ch_data[invalid][-1])

    if ch1 is None:
        ch1 = np.zeros(size)
    else:
        check_invalid_values(ch1)
    if ch2 is None:
        ch2 = np.zeros(size)
    else:
        check_invalid_values(ch1)
    marker_data = np.zeros(size, dtype=np.uint16)
    for idx, marker in enumerate(markers):
        if marker is not None:
            marker_data += np.uint16((marker > 0) * 2 ** idx)
    return zhinst.utils.convert_awg_waveform(ch1, ch2, marker_data)


def zhinst_voltage_to_uint16(ch1: Optional[np.ndarray], ch2: Optional[np.ndarray],
                             markers: Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                            Optional[np.ndarray], Optional[np.ndarray]]) -> np.ndarray:
    """Potentially (if numba is installed) faster version of zhinst.utils.convert_awg_waveform

    Args:
        ch1: Sampled data of channel 1 [-1, 1]
        ch2: Sampled data of channel 1 [-1, 1]
        markers: Marker data of (ch1_front, ch1_back, ch2_front, ch2_back)

    Returns:
        Interleaved data in the correct format (u16). The first bit is the sign bit so the data needs to be interpreted
        as i16.
    """
    all_input = (ch1, ch2, *markers)
    size = {x.size for x in all_input if x is not None}
    if not size:
        raise ValueError("No input arrays")
    elif len(size) != 1:
        raise ValueError("Inputs have incompatible dimension")
    size, = size
    size = int(size)

    if numba is not None:
        try:
            return _zhinst_voltage_to_uint16_numba(size, *all_input)
        except ValueError:
            # use the exception from numpy version
            pass
    return _zhinst_voltage_to_uint16_numpy(size, *all_input)


def not_none_indices(seq: Sequence) -> Tuple[Sequence[Optional[int]], int]:
    """Calculate lookup table from sparse to non sparse indices and the total number of not None elements

    assert ([None, 0, 1, None, None, 2], 3) == not_none_indices([None, 'a', 'b', None, None, 'c'])
    """
    indices = []
    idx = 0
    for elem in seq:
        if elem is None:
            indices.append(elem)
        else:
            indices.append(idx)
            idx += 1
    return indices, idx
