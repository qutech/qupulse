import numpy as np


def set_ignored_marker_data_to_zero(wf):
    # np, ch, byte, data
    wf = wf.reshape(-1, 2, 8)

    channel_mask = np.uint16(2**14 - 1)

    wf[:, 0, :] = np.bitwise_and(wf[:, 0, :], channel_mask)


def validate_programs(program_AB, program_CD, loaded_data: dict, parameters):
    for_A = loaded_data['for_A']
    for_B = loaded_data['for_B']
    for_C = loaded_data['for_C']
    for_D = loaded_data['for_D']

    meas_time_multiplier = parameters["charge_scan___meas_time_multiplier"]
    rep_count = parameters['charge_scan___rep_count']

    expected_samples_A = np.tile(for_A, (meas_time_multiplier * 192, 1, rep_count)).T.ravel()
    set_ignored_marker_data_to_zero(expected_samples_A)
    samples_A = program_AB.get_as_single_waveform(0, expected_samples_A.size, with_marker=True)
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
