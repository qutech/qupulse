import unittest

from qupulse.expressions import Expression
from qupulse.pulses.parameters import ConstantParameter, MappedParameter, ParameterNotProvidedException,\
    ParameterConstraint, InvalidParameterNameException

from tests.pulses.sequencing_dummies import DummyParameter


class ConstantParameterTest(unittest.TestCase):
    def __test_valid_params(self, value: float) -> None:
        constant_parameter = ConstantParameter(value)
        self.assertEqual(value, constant_parameter.get_value())
        
    def test_float_values(self) -> None:
        self.__test_valid_params(-0.3)
        self.__test_valid_params(0)
        self.__test_valid_params(0.3)

    def test_requires_stop(self) -> None:
        constant_parameter = ConstantParameter(0.3)
        self.assertFalse(constant_parameter.requires_stop)

    def test_repr(self) -> None:
        constant_parameter = ConstantParameter(0.2)
        self.assertEqual("<ConstantParameter 0.2>", repr(constant_parameter))


class MappedParameterTest(unittest.TestCase):

    def test_requires_stop_and_get_value(self) -> None:
        p = MappedParameter(Expression("foo + bar * hugo"))
        with self.assertRaises(ParameterNotProvidedException):
            p.requires_stop
        with self.assertRaises(ParameterNotProvidedException):
            p.get_value()

        foo = DummyParameter(-1.1)
        bar = DummyParameter(0.5)
        hugo = DummyParameter(5.2, requires_stop=True)
        ilse = DummyParameter(2356.4, requires_stop=True)

        p.dependencies = {'foo': foo, 'bar': bar, 'ilse': ilse}
        with self.assertRaises(ParameterNotProvidedException):
            p.requires_stop
        with self.assertRaises(ParameterNotProvidedException):
            p.get_value()

        p.dependencies = {'foo': foo, 'bar': bar, 'hugo': hugo}
        self.assertTrue(p.requires_stop)

        hugo = DummyParameter(5.2, requires_stop=False)
        p.dependencies = {'foo': foo, 'bar': bar, 'hugo': hugo, 'ilse': ilse}
        self.assertFalse(p.requires_stop)
        self.assertEqual(1.5, p.get_value())

    def test_repr(self) -> None:
        p = MappedParameter(Expression("foo + bar * hugo"))
        self.assertIsInstance(repr(p), str)


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

    def test_str(self):
        self.assertEqual(str(ParameterConstraint('a < b')), 'a < b')

        self.assertEqual(str(ParameterConstraint('a==b')), 'a==b')


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

