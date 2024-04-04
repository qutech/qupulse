import unittest

import numpy as np

from qupulse.utils.performance import (_time_windows_to_samples_numba, _time_windows_to_samples_numpy,
                                       shrink_overlapping_windows)
from qupulse.utils.performance import (_time_windows_to_samples_numba, _time_windows_to_samples_numpy,
                                       _average_windows_numba, _average_windows_numpy, average_windows)


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


class WindowAverageTest(unittest.TestCase):
    @staticmethod
    def assert_implementations_equal(time, values, begins, ends):
        numpy_result = _average_windows_numpy(time, values, begins, ends)
        numba_result = _average_windows_numba(time, values, begins, ends)
        np.testing.assert_allclose(numpy_result, numba_result)

    def setUp(self):
        self.begins = np.array([1., 2., 3.] + [4.] + [6., 7., 8., 9., 10.])
        self.ends = self.begins + np.array([1., 1., 1.] + [3.] + [2., 2., 2., 2., 2.])
        self.time = np.arange(10).astype(float)
        self.values = np.asarray([
            np.sin(self.time),
            np.cos(self.time),
        ]).T

    def test_dispatch(self):
        _ = average_windows(self.time, self.values, self.begins, self.ends)
        _ = average_windows(self.time, self.values[..., 0], self.begins, self.ends)

    def test_single_channel(self):
        self.assert_implementations_equal(self.time, self.values[..., 0], self.begins, self.ends)
        self.assert_implementations_equal(self.time, self.values[..., :1], self.begins, self.ends)

    def test_dual_channel(self):
        self.assert_implementations_equal(self.time, self.values, self.begins, self.ends)
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