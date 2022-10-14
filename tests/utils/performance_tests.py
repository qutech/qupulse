import io
import tempfile
import unittest

import numpy as np

from qupulse.utils.performance import _time_windows_to_samples_numba, _time_windows_to_samples_numpy, _fmt_int_table, write_int_table


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


class CsvFormatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.array = np.array([
            [1, 2],
            [2**32 - 1, 2**20],
            [523412, 76422]
        ])
        b = io.BytesIO()
        np.savetxt(b, self.array, '%u')
        self.binary_result = b.getvalue()

        s = io.StringIO()
        np.savetxt(s, self.array, '%u')
        self.unicode_result = s.getvalue()

    def test_raw_write_numba(self):
        formatted = _fmt_int_table(self.array, np.uint8(' '.encode('ascii')[0]))
        self.assertEqual(self.binary_result, formatted.tobytes())

    def test_write_to_buffer(self):
        target = io.BytesIO()
        write_int_table(target, self.array, ' ')
        self.assertEqual(self.binary_result, target.getvalue())

    def test_write_to_file(self):
        with tempfile.TemporaryDirectory() as d:
            target_name = d + '/target.txt'
            write_int_table(target_name, self.array, ' ')
            with open(target_name, 'r') as f:
                written_txt = f.read()
            self.assertEqual(self.unicode_result, written_txt)
