import unittest

import numpy as np

from qupulse.utils.performance import (_time_windows_to_samples_numba, _time_windows_to_samples_numpy,
                                       shrink_overlapping_windows)


class TimeWindowsToSamplesTest(unittest.TestCase):
    @staticmethod
    def assert_implementations_equal(begins, lengths, sample_rate):
        np.testing.assert_equal(
            _time_windows_to_samples_numba(begins, lengths, sample_rate),
            _time_windows_to_samples_numpy(begins, lengths, sample_rate)
        )

    def test_monotonic(self):
        begins = np.array([101.3, 123.6218764354, 176.31, 763454.776])
        lengths = np.array([6.4234, 24.8654413, 8765.45, 12543.])

        for sr in (0.1, 1/9, 1., 2.764423123563463412342, 100.322):
            self.assert_implementations_equal(begins, lengths, sr)

    def test_unsorted(self):
        begins = np.array([101.3, 176.31, 763454.776, 123.6218764354])
        lengths = np.array([6.4234, 8765.45, 12543., 24.8654413])

        for sr in (0.1, 1/9, 1., 2.764423123563463412342, 100.322):
            self.assert_implementations_equal(begins, lengths, sr)


class TestOverlappingWindowReduction(unittest.TestCase):
    def test_shrink_overlapping_windows_numba(self):
            np.testing.assert_equal(
                (np.array([1, 4, 8]), np.array([3, 4, 4])),
                shrink_overlapping_windows(np.array([1, 4, 7]),
                                           np.array([3, 4, 5]), use_numba=True)
            )

    def test_shrink_overlapping_windows_numpy(self):
            np.testing.assert_equal(
                (np.array([1, 4, 8]), np.array([3, 4, 4])),
                shrink_overlapping_windows(np.array([1, 4, 7]),
                                            np.array([3, 4, 5]), use_numba=False)
            )