import unittest
from qctoolkit.utils import checked_int_cast


class CheckedIntCastTest(unittest.TestCase):
    def test_int_forwarding(self):
        my_int = 6
        self.assertIs(my_int, checked_int_cast(my_int))

    def test_no_int_detection(self):
        with self.assertRaises(ValueError):
            checked_int_cast(0.5)

        with self.assertRaises(ValueError):
            checked_int_cast(-0.5)

        with self.assertRaises(ValueError):
            checked_int_cast(123124.2)

        with self.assertRaises(ValueError):
            checked_int_cast(123124 + 1e-6)

    def test_float_cast(self):
        self.assertEqual(6, checked_int_cast(6+1e-11))

        self.assertEqual(-6, checked_int_cast(-6 + 1e-11))

    def test_variable_epsilon(self):
        self.assertEqual(6, checked_int_cast(6 + 1e-11))

        with self.assertRaises(ValueError):
            checked_int_cast(6 + 1e-11, epsilon=1e-15)


