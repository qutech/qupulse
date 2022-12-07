import unittest

from qupulse.pulses.parameters import ParameterConstraint, ParameterConstraintViolation,\
    ParameterNotProvidedException, ParameterConstrainer, ConstrainedParameterIsVolatileWarning
from qupulse.pulses.measurement import MeasurementDefiner


class MeasurementDefinerTest(unittest.TestCase):
    def __init__(self, *args, to_test_constructor=None, **kwargs):
        super().__init__(*args, **kwargs)

        if to_test_constructor is None:
            self.to_test_constructor = lambda measurements=None: MeasurementDefiner(measurements=measurements)
        else:
            self.to_test_constructor = to_test_constructor

    def test_measurement_windows(self) -> None:
        pulse = self.to_test_constructor(measurements=[('mw', 0, 5)])
        with self.assertRaises(KeyError):
            pulse.get_measurement_windows(parameters=dict(), measurement_mapping=dict())
        windows = pulse.get_measurement_windows(parameters=dict(), measurement_mapping={'mw': 'asd'})
        self.assertEqual([('asd', 0, 5)], windows)
        self.assertEqual(pulse.measurement_declarations, [('mw', 0, 5)])

    def test_multiple_windows(self):
        pulse = self.to_test_constructor(measurements=[('mw', 0, 5), ('H', 'a', 'b')])
        with self.assertRaises(KeyError):
            pulse.get_measurement_windows(parameters=dict(), measurement_mapping=dict())
        windows = pulse.get_measurement_windows(parameters=dict(a=0.5, b=1), measurement_mapping={'mw': 'asd', 'H': 'H'})
        self.assertEqual([('asd', 0, 5), ('H', 0.5, 1)], windows)
        self.assertEqual(pulse.measurement_declarations, [('mw', 0, 5), ('H', 'a', 'b')])

    def test_no_measurement_windows(self) -> None:
        pulse = self.to_test_constructor()
        windows = pulse.get_measurement_windows(dict(), {'mw': 'asd'})
        self.assertEqual([], windows)
        self.assertEqual([], pulse.measurement_declarations)

    def test_measurement_windows_with_parameters(self) -> None:
        pulse = self.to_test_constructor(measurements=[('mw', 1, '(1+length)/2')])
        parameters = dict(length=100)
        windows = pulse.get_measurement_windows(parameters, measurement_mapping={'mw': 'asd'})
        self.assertEqual(windows, [('asd', 1, 101 / 2)])
        self.assertEqual(pulse.measurement_declarations, [('mw', 1, '(1+length)/2')])

    def test_measurement_windows_invalid(self) -> None:
        pulse = self.to_test_constructor(measurements=[('mw', 'a', 'd')])
        measurement_mapping = {'mw': 'mw'}

        with self.assertRaises(ValueError):
            pulse.get_measurement_windows(measurement_mapping=measurement_mapping,
                                          parameters=dict(length=10, a=-1, d=3))
        with self.assertRaises(ValueError):
            pulse.get_measurement_windows(measurement_mapping=measurement_mapping,
                                          parameters=dict(length=10, a=3, d=-1))

    def test_none_mappings(self):
        pulse = self.to_test_constructor(measurements=[('mw', 'a', 'd'), ('asd', 0, 1.)])

        parameters = dict(length=100, a=4, d=5)

        windows = pulse.get_measurement_windows(parameters, measurement_mapping={'mw': 'mw', 'asd': None})
        self.assertEqual(windows, [('mw', 4, 5)])

        windows = pulse.get_measurement_windows(dict(length=100), measurement_mapping={'mw': None, 'asd': None})
        self.assertEqual(windows, [])


class ParameterConstrainerTest(unittest.TestCase):
    def __init__(self, *args, to_test_constructor=None, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Figure out what is going on here
        if to_test_constructor is None:
            self.to_test_constructor = lambda parameter_constraints=None:\
                ParameterConstrainer(parameter_constraints=parameter_constraints)
        else:
            self.to_test_constructor = to_test_constructor

    def test_parameter_constraints(self):
        to_test = self.to_test_constructor()
        self.assertEqual(to_test.parameter_constraints, [])

        to_test = self.to_test_constructor(['a < b'])
        self.assertEqual(to_test.parameter_constraints, [ParameterConstraint('a < b')])

        to_test = self.to_test_constructor(['a < b', 'c < 1'])
        self.assertEqual(to_test.parameter_constraints, [ParameterConstraint('a < b'), ParameterConstraint('c < 1')])

    def test_validate_parameter_constraints(self):
        to_test = self.to_test_constructor()
        to_test.validate_parameter_constraints(dict(), set())
        to_test.validate_parameter_constraints(dict(a=1), set())

        to_test = self.to_test_constructor(['a < b'])
        with self.assertRaises(ParameterNotProvidedException):
            to_test.validate_parameter_constraints(dict(), set())
        with self.assertRaises(ParameterConstraintViolation):
            to_test.validate_parameter_constraints(dict(a=1, b=0.8), set())
        to_test.validate_parameter_constraints(dict(a=1, b=2), set())

        to_test = self.to_test_constructor(['a < b', 'c < 1'])
        with self.assertRaises(ParameterNotProvidedException):
            to_test.validate_parameter_constraints(dict(a=1, b=2), set())
        with self.assertRaises(ParameterNotProvidedException):
            to_test.validate_parameter_constraints(dict(c=0.5), set())

        with self.assertRaises(ParameterConstraintViolation):
            to_test.validate_parameter_constraints(dict(a=1, b=0.8, c=0.5), set())
        with self.assertRaises(ParameterConstraintViolation):
            to_test.validate_parameter_constraints(dict(a=0.5, b=0.8, c=1), set())
        to_test.validate_parameter_constraints(dict(a=0.5, b=0.8, c=0.1), {'j'})

        with self.assertWarns(ConstrainedParameterIsVolatileWarning):
            to_test.validate_parameter_constraints(dict(a=0.5, b=0.8, c=0.1), {'a'})

    def test_constrained_parameters(self):
        to_test = self.to_test_constructor()
        self.assertEqual(to_test.constrained_parameters, set())

        to_test = self.to_test_constructor(['a < b'])
        self.assertEqual(to_test.constrained_parameters, {'a', 'b'})

        to_test = self.to_test_constructor(['a < b', 'c < 1'])
        self.assertEqual(to_test.constrained_parameters, {'a', 'b', 'c'})