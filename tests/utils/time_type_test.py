import sys
import unittest
import os
import contextlib
import importlib
import fractions
from unittest import mock

try:
    import gmpy2
except ImportError:
    gmpy2 = None

import qupulse.utils.types as qutypes


def mock_missing_gmpy2(exit_stack: contextlib.ExitStack):
    if os.path.exists('gmpy2.py'):
        raise RuntimeError('Cannot mock missing gmpy2 due to existing file')

    with open('gmpy2.py', 'w') as gmpy_file:
        exit_stack.callback(os.remove, 'gmpy2.py')

        gmpy_file.write("raise ImportError()")

    if 'gmpy2' in sys.modules:
        modules_patcher = mock.patch.dict(sys.modules,
                                          values=((name, module)
                                                  for name, module in sys.modules.items()
                                                  if name != 'gmpy2'),
                                          clear=True)
        modules_patcher.__enter__()
        exit_stack.push(modules_patcher)


class TestTimeType(unittest.TestCase):
    _fallback_qutypes = None

    @property
    def fallback_qutypes(self):
        if not self._fallback_qutypes:
            exit_stack = contextlib.ExitStack()

            with exit_stack:

                if gmpy2:
                    # create a local file that raises ImportError on import
                    mock_missing_gmpy2(exit_stack)

                    self._fallback_qutypes = importlib.reload(qutypes)

                else:
                    self._fallback_qutypes = qutypes
        return self._fallback_qutypes

    def test_fraction_fallback(self):
        self.assertIs(fractions.Fraction, self.fallback_qutypes.TimeType)

    @unittest.skipIf(gmpy2 is None, "gmpy2 not available.")
    def test_default_time_from_float(self):
        self.assertEqual(qutypes.time_from_float(123/931), gmpy2.mpq(123, 931))

        self.assertEqual(qutypes.time_from_float(1000000/1000001, 1e-5), gmpy2.mpq(1))

    def test_fraction_time_from_float(self):
        self.assertEqual(self.fallback_qutypes.time_from_float(123 / 931),
                         fractions.Fraction(123, 931))

        self.assertEqual(self.fallback_qutypes.time_from_float(1000000 / 1000001, 1e-5),
                         fractions.Fraction(1))

