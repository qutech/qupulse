import unittest

from qupulse.expressions import Expression
from qupulse.pulses.parameters import ParameterNotProvidedException,\
    ParameterConstraint, InvalidParameterNameException


class ParameterConstraintTest(unittest.TestCase):
    def test_ordering(self):
        constraint = ParameterConstraint('a <= b')
        self.assertEqual(constraint.affected_parameters, {'a', 'b'})

        self.assertTrue(constraint.is_fulfilled(dict(a=1, b=2)))
        self.assertTrue(constraint.is_fulfilled(dict(a=2, b=2)))
        self.assertFalse(constraint.is_fulfilled(dict(a=2, b=1)))

    def test_equal(self):
        constraint = ParameterConstraint('a==b')
        self.assertEqual(constraint.affected_parameters, {'a', 'b'})

        self.assertTrue(constraint.is_fulfilled(dict(a=2, b=2)))
        self.assertFalse(constraint.is_fulfilled(dict(a=3, b=2)))

    def test_expressions(self):
        constraint = ParameterConstraint('Max(a, b) < a*c')
        self.assertEqual(constraint.affected_parameters, {'a', 'b', 'c'})

        self.assertTrue(constraint.is_fulfilled(dict(a=2, b=2, c=3)))
        self.assertFalse(constraint.is_fulfilled(dict(a=3, b=5, c=1)))

    def test_no_relation(self):
        with self.assertRaises(ValueError):
            ParameterConstraint('a*b')
        ParameterConstraint('1 < 2')

    def test_str_and_serialization(self):
        self.assertEqual(str(ParameterConstraint('a < b')), 'a < b')
        self.assertEqual(ParameterConstraint('a < b').get_serialization_data(), 'a < b')

        self.assertEqual(str(ParameterConstraint('a==b')), 'a==b')
        self.assertEqual(ParameterConstraint('a==b').get_serialization_data(), 'a==b')

    def test_repr(self):
        pc = ParameterConstraint('a < b')
        self.assertEqual("ParameterConstraint('a < b')", repr(pc))


class ParameterNotProvidedExceptionTests(unittest.TestCase):

    def test(self) -> None:
        exc = ParameterNotProvidedException('foo')
        self.assertEqual("No value was provided for parameter 'foo'.", str(exc))


class InvalidParameterNameExceptionTests(unittest.TestCase):
    def test(self):
        exception = InvalidParameterNameException('asd')

        self.assertEqual(exception.parameter_name, 'asd')
        self.assertEqual(str(exception), 'asd is an invalid parameter name')

        
if __name__ == "__main__":
    unittest.main(verbosity=2)

