import numpy as np


def validate_programs(program_AB, program_CD, loaded_data: dict, parameters):
    for_A = loaded_data['for_A']
    for_B = loaded_data['for_B']
    for_C = loaded_data['for_C']
    for_D = loaded_data['for_D']

    meas_time_multiplier = parameters["charge_scan___meas_time_multiplier"].get_value()
    rep_count = parameters['charge_scan___rep_count'].get_value()

    expected_samples_A = np.tile(for_A, (meas_time_multiplier * 192, 1, rep_count)).T.ravel()
    samples_A = program_AB.get_as_single_waveform(0, expected_samples_A.size)
    np.testing.assert_equal(samples_A, expected_samples_A)

    del samples_A
    del expected_samples_A

    expected_samples_B = np.tile(for_B, (meas_time_multiplier * 192, 1, rep_count)).T.ravel()
    samples_B = program_AB.get_as_single_waveform(1, expected_samples_B.size)
    np.testing.assert_equal(samples_B, expected_samples_B)

    del samples_B
    del expected_samples_B

    samples_C = program_CD.get_as_single_waveform(0)
    np.testing.assert_equal(samples_C, for_C)

    del samples_C

    samples_D = program_CD.get_as_single_waveform(1)
    np.testing.assert_equal(samples_D, for_D)
