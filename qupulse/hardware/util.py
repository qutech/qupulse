from typing import List, Sequence, Tuple, Union, Collection
import itertools

import numpy as np

from qupulse._program.waveforms import Waveform
from qupulse.utils.types import TimeType
from qupulse.utils import pairwise

__all__ = ['voltage_to_uint16', 'get_sample_times']


def voltage_to_uint16(voltage: np.ndarray, output_amplitude: float, output_offset: float, resolution: int) -> np.ndarray:
    """

    :param voltage:
    :param output_amplitude:
    :param output_offset:
    :param resolution:
    :return:
    """
    if resolution < 1 or not isinstance(resolution, int):
        raise ValueError('The resolution must be an integer > 0')
    non_dc_voltage = voltage - output_offset

    if np.any(np.abs(non_dc_voltage) > output_amplitude):
        raise ValueError('Voltage of range', dict(voltage=voltage,
                                                  output_offset=output_offset,
                                                  output_amplitude=output_amplitude))
    non_dc_voltage += output_amplitude
    non_dc_voltage *= (2**resolution - 1) / (2*output_amplitude)
    np.rint(non_dc_voltage, out=non_dc_voltage)
    return non_dc_voltage.astype(np.uint16)


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
