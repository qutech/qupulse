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
    non_dc_voltage = voltage - output_offset

    if np.any(np.abs(non_dc_voltage) > output_amplitude):
        raise ValueError('Voltage of range', dict(voltage=voltage,
                                                  output_offset=output_offset,
                                                  output_amplitude=output_amplitude))
    non_dc_voltage += output_amplitude
    non_dc_voltage *= (2**resolution - 1) / (2*output_amplitude)
    np.rint(non_dc_voltage, out=non_dc_voltage)
    return non_dc_voltage.astype(np.uint16)
