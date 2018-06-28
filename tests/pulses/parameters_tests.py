import unittest

from qctoolkit.expressions import Expression
from qctoolkit.pulses.parameters import ConstantParameter, MappedParameter, ParameterNotProvidedException,\
    ParameterConstraint, InvalidParameterNameException, ParameterConstrainer, ParameterConstraintViolation

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


class ParameterConstrainerTest(unittest.TestCase):
    def __init__(self, *args, to_test_constructor=None, **kwargs):
        super().__init__(*args, **kwargs)

        if to_test_constructor is None:
            self.to_test_constructor = lambda parameter_constraints=None:\
                ParameterConstrainer(parameter_constraints=parameter_constraints)
        else:
            self.to_test_constructor = to_test_constructor

    def test_parameter_constraints(self):
        to_test = self.to_test_constructor()
        self.assertEqual(to_test.parameter_constraints, {})

        to_test = self.to_test_constructor(['a < b'])
        self.assertEqual(to_test.parameter_constraints, {ParameterConstraint('a < b')})

        to_test = self.to_test_constructor(['a < b', 'c < 1'])
        self.assertEqual(to_test.parameter_constraints, {ParameterConstraint('a < b'), ParameterConstraint('c < 1')})

    def test_validate_parameter_constraints(self):
        to_test = self.to_test_constructor()
        to_test.validate_parameter_constraints(dict())
        to_test.validate_parameter_constraints(dict(a=1))

        to_test = self.to_test_constructor(['a < b'])
        with self.assertRaises(ParameterNotProvidedException):
            to_test.validate_parameter_constraints(dict())
        with self.assertRaises(ParameterConstraintViolation):
            to_test.validate_parameter_constraints(dict(a=1, b=0.8))
        to_test.validate_parameter_constraints(dict(a=1, b=2))

        to_test = self.to_test_constructor(['a < b', 'c < 1'])
        with self.assertRaises(ParameterNotProvidedException):
            to_test.validate_parameter_constraints(dict(a=1, b=2))
        with self.assertRaises(ParameterNotProvidedException):
            to_test.validate_parameter_constraints(dict(c=0.5))

        with self.assertRaises(ParameterConstraintViolation):
            to_test.validate_parameter_constraints(dict(a=1, b=0.8, c=0.5))
        with self.assertRaises(ParameterConstraintViolation):
            to_test.validate_parameter_constraints(dict(a=0.5, b=0.8, c=1))
        to_test.validate_parameter_constraints(dict(a=0.5, b=0.8, c=0.1))

    def test_constrained_parameters(self):
        to_test = self.to_test_constructor()
        self.assertEqual(to_test.constrained_parameters, set())

        to_test = self.to_test_constructor(['a < b'])
        self.assertEqual(to_test.constrained_parameters, {'a', 'b'})

        to_test = self.to_test_constructor(['a < b', 'c < 1'])
        self.assertEqual(to_test.constrained_parameters, {'a', 'b', 'c'})

    def test_constraint_order_invariance(self) -> None:
        c1 = ParameterConstrainer(parameter_constraints=['bla > blub', 'foo == bar'])
        c2 = ParameterConstrainer(parameter_constraints=['bla > blub', 'foo == bar'])
        c3 = ParameterConstrainer(parameter_constraints=['foo == bar', 'bla > blub'])

        self.assertEqual(c1.parameter_constraints, c2.parameter_constraints)
        self.assertEqual(c2.parameter_constraints, c3.parameter_constraints)
        self.assertEqual(c3.parameter_constraints, c1.parameter_constraints)

        self.assertEqual(c1.constrained_parameters, c2.constrained_parameters)
        self.assertEqual(c2.constrained_parameters, c3.constrained_parameters)
        self.assertEqual(c3.constrained_parameters, c1.constrained_parameters)


if __name__ == "__main__":
    unittest.main(verbosity=2)

