import unittest

from qctoolkit.expressions import Expression
from qctoolkit.pulses.parameters import ConstantParameter, MappedParameter, ParameterNotProvidedException,\
    ParameterConstraint, InvalidParameterNameException, ParameterLibrary

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


class ParameterLibraryTests(unittest.TestCase):

    def setUp(self) -> None:
        self.high_level_params = {
            'global': {'foo': 0.32, 'bar': 3.156, 'ilse': -1.2365, 'hugo': -15.236},
            'test_pulse': {'foo': -2.4},
            'tast_pulse': {'foo': 0, 'ilse': 0}
        }
        self.intermediate_level_params = {
            'global': {'foo': -1.176, 'hugo': 0.151}
        }
        self.low_level_params = {
            'test_pulse': {'bar': -2.75},
            'tast_pulse': {'foo': 0.12}
        }
        self.param_sources = [self.high_level_params, self.intermediate_level_params, self.low_level_params]

    def test_get_parameters_1(self) -> None:
        identifier = 'test_pulse'
        expected_params = {
            'foo': self.intermediate_level_params['global']['foo'],
            'bar': self.low_level_params[identifier]['bar'],
            'ilse': self.high_level_params['global']['ilse'],
            'hugo': self.intermediate_level_params['global']['hugo']
        }

        composer = ParameterLibrary(self.param_sources)
        params = composer.get_parameters(identifier)

        self.assertEqual(expected_params, params)

    def test_get_parameters_2(self) -> None:
        identifier = 'tast_pulse'
        expected_params = {
            'foo': self.low_level_params[identifier]['foo'],
            'bar': self.high_level_params['global']['bar'],
            'hugo': self.intermediate_level_params['global']['hugo'],
            'ilse': self.high_level_params[identifier]['ilse']
        }

        composer = ParameterLibrary(self.param_sources)
        params = composer.get_parameters(identifier)

        self.assertEqual(expected_params, params)

    def test_get_parameters_with_local_subst(self) -> None:
        identifier='tast_pulse'
        local_param_subst = {
            'ilse': -12.5,
            'hugo': 7.25
        }

        expected_params = {
            'foo': self.low_level_params[identifier]['foo'],
            'bar': self.high_level_params['global']['bar'],
            'ilse': local_param_subst['ilse'],
            'hugo': local_param_subst['hugo']
        }

        composer = ParameterLibrary(self.param_sources)
        params = composer.get_parameters(identifier, local_param_subst)

        self.assertEqual(expected_params, params)

    def test_no_pulse_params(self) -> None:
        identifier='unknown_pulse'
        expected_params = {
            'foo': self.intermediate_level_params['global']['foo'],
            'bar': self.high_level_params['global']['bar'],
            'ilse': self.high_level_params['global']['ilse'],
            'hugo': self.intermediate_level_params['global']['hugo']
        }

        composer = ParameterLibrary(self.param_sources)
        params = composer.get_parameters(identifier)

        self.assertEqual(expected_params, params)

    def test_no_pulse_id(self) -> None:
        identifier = None
        expected_params = {
            'foo': self.intermediate_level_params['global']['foo'],
            'bar': self.high_level_params['global']['bar'],
            'ilse': self.high_level_params['global']['ilse'],
            'hugo': self.intermediate_level_params['global']['hugo']
        }

        composer = ParameterLibrary(self.param_sources)
        params = composer.get_parameters(identifier)

        self.assertEqual(expected_params, params)

    def test_no_pulse_params_subst(self) -> None:
        identifier = 'unknown_pulse'
        local_param_subst = {
            'ilse': -12.5,
            'hugo': 7.25
        }
        expected_params = {
            'foo': self.intermediate_level_params['global']['foo'],
            'bar': self.high_level_params['global']['bar'],
            'ilse': local_param_subst['ilse'],
            'hugo': local_param_subst['hugo']
        }

        composer = ParameterLibrary(self.param_sources)
        params = composer.get_parameters(identifier, local_param_subst)

        self.assertEqual(expected_params, params)

    def test_global_or_pulse_params(self) -> None:
        high_level_params = {
            'test_pulse': {'foo': -2.4},
            'tast_pulse': {'foo': 0, 'ilse': 0}
        }
        low_level_params = {
            'test_pulse': {'bar': -2.75},
            'tast_pulse': {'foo': 0.12}
        }
        identifier = 'unknown_pulse'
        local_param_subst = {
            'ilse': -12.5,
            'hugo': 7.25
        }
        expected_params = {
            'ilse': local_param_subst['ilse'],
            'hugo': local_param_subst['hugo']
        }

        composer = ParameterLibrary([high_level_params, low_level_params])
        params = composer.get_parameters(identifier, local_param_subst)

        self.assertEqual(expected_params, params)

    def test_empty_pulse_context(self) -> None:
        lib = ParameterLibrary([self.high_level_params, self.intermediate_level_params, self.low_level_params])
        expected_params = { 'foo': -1.176, 'bar': 3.156, 'ilse': -1.2365, 'hugo': 0.151}
        self.assertEqual(expected_params, lib.get_parameters(""))


if __name__ == "__main__":
    unittest.main(verbosity=2)

