import unittest

import numpy as np

from qupulse.utils.performance import _time_windows_to_samples_numba, _time_windows_to_samples_numpy, compress_array_LZ77, uncompress_array_LZ77


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

class TestLZ77(unittest.TestCase):

    def test_compression_and_reconstruction(self, array, intermed, diffs):
        compressed = compress_array_LZ77(array=array, allow_intermediates=intermed, using_diffs=diffs)
        uncompressed = uncompress_array_LZ77(compressed)
        if diffs:
            self.assertTrue(len(array) == len(uncompressed))
            self.assert_implementations_equal(np.allclose(array, np.cumsum(uncompressed, axis=0)))
        else:
            self.assertTrue(len(array) == len(uncompressed))
            self.assert_implementations_equal(np.allclose(array, uncompressed))
        return compressed, uncompressed

    def test_compression_and_reconstruction_helper(self, array):
        _ = self.test_compression_and_reconstruction(array, intermed=True, diffs=True)
        _ = self.test_compression_and_reconstruction(array, intermed=True, diffs=False)
        _ = self.test_compression_and_reconstruction(array, intermed=False, diffs=True)
        _ = self.test_compression_and_reconstruction(array, intermed=False, diffs=False)

    def test_random_arrays(self):
        for _ in range(100):
            self.test_compression_and_reconstruction_helper(np.random.uniform(0, 1, (np.random.randint(1, 100), np.random.randint(1, 100))))

    def test_various(self):
        self.test_compression_and_reconstruction_helper(np.arange(100).reshape((50, 2)))
        self.test_compression_and_reconstruction_helper(np.array([[1, 2, 3, 4, 5]]).T)
        self.test_compression_and_reconstruction_helper(np.array([[1]]))
        self.test_compression_and_reconstruction_helper(np.ones((100, 2)))
        self.test_compression_and_reconstruction_helper(np.ones((1, 2)))
        self.test_compression_and_reconstruction_helper(np.ones((1, 10)))
        self.test_compression_and_reconstruction_helper(np.zeros((1, 10)))
        special_array = np.repeat(np.linspace(0, 100, 100)[:, None], 3, axis=1)
        special_array[:, 0] = special_array[:, 0]%5
        special_array[:, 1] = special_array[:, 1]%10
        special_array = special_array.astype(int)
        self.test_compression_and_reconstruction_helper(special_array.astype(int))


