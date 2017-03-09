from typing import List, Tuple

import numpy as np

__all__ = ['voltage_to_uint16']


def voltage_to_uint16(voltage: np.ndarray, output_amplitude: float, output_offset: float, resolution: int):
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

    segment_quanta = 2 * segment_lengths[0] // quantum
    if segments[0][1] is not None:
        destination_array[0:segment_quanta:2].flat = segments[0][1]
    if segments[0][0] is not None:
        destination_array[1:segment_quanta:2].flat = segments[0][0]

    current_quantum = segment_quanta
    for (chan_a, chan_b), segment_length in zip(segments[1:], segment_lengths[1:]):
        segment_quanta = 2 * (segment_length // quantum + 1)
        segment_destination = destination_array[current_quantum:current_quantum+segment_quanta, :]

        if chan_b is not None:
            segment_destination[0, :] = chan_b[0]
            segment_destination[2::2, :].flat = chan_b
        if chan_a is not None:
            segment_destination[1, :] = chan_a[0]
            segment_destination[3::2, :].flat = chan_a
        current_quantum += segment_quanta
    return destination_array.ravel()

