from typing import List, Sequence, Tuple, Union, Optional
import itertools

import numpy as np

from qupulse._program.waveforms import Waveform
from qupulse.utils.types import TimeType, Collection

try:
    import numba
    njit = numba.njit
except ImportError:
    numba = None
    njit = lambda x: x

try:
    import zhinst
except ImportError:
    zhinst = None

__all__ = ['voltage_to_uint16', 'get_sample_times', 'zhinst_voltage_to_uint16']


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


def make_combined_wave(segments: List['TaborSegment'], destination_array=None, fill_value=None) -> np.ndarray:
    quantum = 16
    if len(segments) == 0:
        return np.zeros(0, dtype=np.uint16)
    segment_lengths = np.fromiter((segment.num_points for segment in segments), count=len(segments), dtype=int)
    if np.any(segment_lengths % quantum != 0):
        raise ValueError('Segment is not a multiple of 16')
    n_quanta = np.sum(segment_lengths) // quantum + len(segments) - 1

    if destination_array is not None:
        if len(destination_array) != 2*n_quanta*quantum:
            raise ValueError('Destination array has an invalid length')
        destination_array = destination_array.reshape((2*n_quanta, quantum))
    else:
        destination_array = np.empty((2*n_quanta, quantum), dtype=np.uint16)
    if fill_value:
        destination_array[:] = fill_value

    # extract data that already includes the markers
    data, next_data = itertools.tee(((segment.data_a, segment.data_b) for segment in segments), 2)
    next(next_data, None)

    current_quantum = 0
    for (data_a, data_b), next_segment, segment_length in itertools.zip_longest(data, next_data, segment_lengths):
        segment_quanta = 2 * (segment_length // quantum)
        segment_destination = destination_array[current_quantum:current_quantum+segment_quanta, :]

        if data_b is not None:
            segment_destination[::2, :].flat = data_b
        if data_a is not None:
            segment_destination[1::2, :].flat = data_a
        current_quantum += segment_quanta

        if next_segment:
            # fill one quantum with first data point from next segment
            next_data_a, next_data_b = next_segment
            if next_data_b is not None:
                destination_array[current_quantum, :] = next_data_b[0]
            if next_data_a is not None:
                destination_array[current_quantum+1, :] = next_data_a[0]
            current_quantum += 2
    return destination_array.ravel()


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
            raise ValueError("Error while sampling waveforms. One waveform has a length <= zero at the given sample "
                             "rate of %rGHz" % sample_rate_in_GHz)
        segment_lengths.append(rounded_segment_length)

    segment_lengths = np.asarray(segment_lengths, dtype=np.uint64)
    time_array = np.arange(np.max(segment_lengths)) / float(sample_rate_in_GHz)

    return time_array, segment_lengths


@njit
def _zhinst_voltage_to_uint16_numba(size: int, ch1: Optional[np.ndarray], ch2: Optional[np.ndarray],
                                    m1_front: Optional[np.ndarray], m1_back: Optional[np.ndarray],
                                    m2_front: Optional[np.ndarray], m2_back: Optional[np.ndarray]) -> np.ndarray:
    """Numba targeted implementation"""
    data = np.zeros((size, 3), dtype=np.uint16)

    scale = float(2**15 - 1)

    for i in range(size):
        if ch1 is not None:
            data[i, 0] = ch1[i] * scale
        if ch2 is not None:
            data[i, 1] = ch2[i] * scale
        if m1_front is not None:
            data[i, 2] |= (m1_front[i] != 0)
        if m1_back is not None:
            data[i, 2] |= (m1_back[i] != 0) << 1
        if m2_front is not None:
            data[i, 2] |= (m2_front[i] != 0) << 2
        if m2_back is not None:
            data[i, 2] |= (m2_back[i] != 0) << 3
    return data.ravel()


def _zhinst_voltage_to_uint16_numpy(size: int, ch1: Optional[np.ndarray], ch2: Optional[np.ndarray],
                                    m1_front: Optional[np.ndarray], m1_back: Optional[np.ndarray],
                                    m2_front: Optional[np.ndarray], m2_back: Optional[np.ndarray]) -> np.ndarray:
    """Fallback implementation if numba is not available"""
    markers = (m1_front, m1_back, m2_front, m2_back)

    if ch1 is None:
        ch1 = np.zeros(size)
    if ch2 is None:
        ch2 = np.zeros(size)
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
    assert any(x is not None for x in all_input)
    size = {x.size for x in all_input if x is not None}
    assert len(size) == 1, "Inputs have incompatible dimension"
    size, = size
    size = int(size)

    raise NotImplementedError('Check voltage range')

    if numba:
        return _zhinst_voltage_to_uint16_numba(size, *all_input)
    else:
        return _zhinst_voltage_to_uint16_numpy(size, *all_input)
