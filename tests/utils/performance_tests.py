import io
import unittest

import numpy as np

from qupulse.utils.performance import _time_windows_to_samples_numba, _time_windows_to_samples_numpy,\
    _write_csv_with_numpy, _write_csv_with_pandas, _write_csv_with_polars, save_integer_csv, pl, pd


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


class CsvFormattingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.table = (np.arange(192, dtype=np.uint32) + 2**17).reshape(-1, 2)
        b = io.BytesIO()
        _write_csv_with_numpy(b, self.table)
        self.numpy_format = b.getvalue()

    @unittest.skipIf(pd is None, "pandas not installed")
    def test_pandas_format(self):
        b = io.BytesIO()
        _write_csv_with_pandas(b, self.table)
        self.assertEqual(self.numpy_format, b.getvalue())

    @unittest.skipIf(pl is None, "polars not installed")
    def test_polars_format(self):
        b = io.BytesIO()
        _write_csv_with_polars(b, self.table)
        self.assertEqual(self.numpy_format, b.getvalue())
