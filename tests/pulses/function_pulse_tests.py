import unittest
import sympy
import numpy as np

from qupulse.utils.types import TimeType
from qupulse.pulses.function_pulse_template import FunctionPulseTemplate
from qupulse.serialization import Serializer, Serializable, PulseStorage
from qupulse.expressions import Expression
from qupulse.pulses.parameters import ParameterConstraintViolation, ParameterConstraint
from qupulse._program.waveforms import FunctionWaveform

from tests.serialization_dummies import DummySerializer, DummyStorageBackend
from tests.pulses.measurement_tests import MeasurementDefinerTest, ParameterConstrainerTest
from tests.serialization_tests import SerializableTests


class FunctionPulseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None
        self.s = 'a + b * t'
        self.s2 = 'c'

        self.meas_list = [('mw', 1, 1), ('mw', 'x', 'z'), ('drup', 'j', 'u')]

        self.constraints = ['a < b', 'c > 1', 'd > c']

        self.valid_par_vals = dict(a=1,
                             b=2,
                             c=14.5,
                             d=15,
                             x=0.1,
                             z=0.2,
                             j=0.3,
                             u=0.4)

        self.invalid_par_vals = dict(a=1,
                             b=2,
                             c=14.5,
                             d=14,
                             x=0.1,
                             z=0.2,
                             j=0.3,
                             u=0.4)

        self.fpt = FunctionPulseTemplate(self.s, self.s2, channel='A',
                                         measurements=self.meas_list,
                                         parameter_constraints=self.constraints)


class FunctionPulsePropertyTest(FunctionPulseTest):
    def test_expression(self):
        self.assertEqual(self.fpt.expression, self.s)

    def test_function_parameters(self):
        self.assertEqual(self.fpt.function_parameters, {'a', 'b', 'c'})

    def test_defined_channels(self) -> None:
        self.assertEqual({'A'}, self.fpt.defined_channels)

    def test_parameter_names(self):
        fpt = FunctionPulseTemplate('k*r*sin(x)', '1.14*d', parameter_constraints={'x < hugo'}, measurements=[('meas', 'f', 1)])
        self.assertEqual(fpt.parameter_names, {'k', 'r', 'x', 'd', 'hugo', 'f'})

    def test_duration(self):
        self.assertEqual(self.fpt.duration, self.s2)

    def test_measurement_names(self):
        self.assertEqual(self.fpt.measurement_names, {'mw', 'drup'})

    def test_parameter_names_and_declarations_expression_input(self) -> None:
        template = FunctionPulseTemplate(Expression("3 * foo + bar * t"), Expression("5 * hugo"))
        expected_parameter_names = {'foo', 'bar', 'hugo'}
        self.assertEqual(expected_parameter_names, template.parameter_names)

    def test_parameter_names_and_declarations_string_input(self) -> None:
        template = FunctionPulseTemplate("3 * foo + bar * t", "5 * hugo",channel='A')
        expected_parameter_names = {'foo', 'bar', 'hugo'}
        self.assertEqual(expected_parameter_names, template.parameter_names)

    def test_integral(self) -> None:
        pulse = FunctionPulseTemplate('sin(0.5*t+b)', '2*Tmax')
        self.assertEqual({'default': Expression('2.0*cos(b) - 2.0*cos(1.0*Tmax+b)')}, pulse.integral)

    def test_initial_values(self):
        fpt = FunctionPulseTemplate('3 + exp(t * a)', 'pi', channel='A')
        self.assertEqual({'A': 4}, fpt.initial_values)

    def test_final_values(self):
        fpt = FunctionPulseTemplate('3 + exp(t * a)', 'pi', channel='A')
        self.assertEqual({'A': Expression('3 + exp(pi*a)')}, fpt.final_values)

    def test_as_expression(self):
        pulse = FunctionPulseTemplate('sin(0.5*t+b)', '2*Tmax')
        expr = sympy.sin(0.5 * pulse._AS_EXPRESSION_TIME + sympy.sympify('b'))
        self.assertEqual({'default': Expression.make(expr)}, pulse._as_expression())


class FunctionPulseSerializationTest(SerializableTests, unittest.TestCase):

    @property
    def class_to_test(self):
        return FunctionPulseTemplate

    def make_kwargs(self):
        return {
            'expression': Expression('a + b * t'),
            'duration_expression': Expression('c'),
            'channel': 'A',
            'measurements': [('mw', 1, 1), ('mw', 'x', 'z'), ('drup', 'j', 'u')],
            'parameter_constraints': [str(ParameterConstraint('a < b')), str(ParameterConstraint('c > 1')),
                                      str(ParameterConstraint('d > c'))]
        }

    def assert_equal_instance_except_id(self, lhs: FunctionPulseTemplate, rhs: FunctionPulseTemplate):
        self.assertIsInstance(lhs, FunctionPulseTemplate)
        self.assertIsInstance(rhs, FunctionPulseTemplate)
        self.assertEqual(lhs.parameter_names, rhs.parameter_names)
        self.assertEqual(lhs.measurement_declarations, rhs.measurement_declarations)
        self.assertEqual(lhs.parameter_constraints, rhs.parameter_constraints)
        self.assertEqual(lhs.duration, rhs.duration)
        self.assertEqual(lhs.expression, rhs.expression)


class FunctionPulseOldSerializationTests(FunctionPulseTest):

    def test_get_serialization_data_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="FunctionPT does not issue warning for old serialization routines."):
            expected_data = dict(duration_expression=str(self.s2),
                                 expression=str(self.s),
                                 channel='A',
                                 measurement_declarations=self.meas_list,
                                 parameter_constraints=self.constraints)
            self.assertEqual(expected_data, self.fpt.get_serialization_data(
                DummySerializer(serialize_callback=lambda x: x.original_expression)))

    def test_deserialize_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="FunctionPT does not issue warning for old serialization routines."):
            basic_data = dict(duration_expression=str(self.s2),
                              expression=self.s,
                              channel='A',
                              identifier='hugo',
                              measurement_declarations=self.meas_list,
                              parameter_constraints=self.constraints)
            serializer = DummySerializer(serialize_callback=lambda x: x.original_expression)
            serializer.subelements[self.s2] = Expression(self.s2)
            serializer.subelements[self.s] = Expression(self.s)
            template = FunctionPulseTemplate.deserialize(serializer, **basic_data)
            self.assertEqual('hugo', template.identifier)
            self.assertEqual({'a', 'b', 'c', 'x', 'z', 'j', 'u', 'd'}, template.parameter_names)
            self.assertEqual(template.measurement_declarations,
                             self.meas_list)
            serialized_data = template.get_serialization_data(serializer)
            del basic_data['identifier']
            self.assertEqual(basic_data, serialized_data)

    def test_serializer_integration_old(self):
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="FunctionPT does not issue warning for old serialization routines."):
            before = FunctionPulseTemplate(expression=self.s,
                                           duration_expression=self.s2,
                                           channel='A',
                                           measurements=self.meas_list,
                                           parameter_constraints=self.constraints,
                                           identifier='my_tpt', registry=dict())
            serializer = Serializer(DummyStorageBackend())
            serializer.serialize(before)
            after = serializer.deserialize('my_tpt')

            self.assertIsInstance(after, FunctionPulseTemplate)
            self.assertEqual(before.expression, after.expression)
            self.assertEqual(before.duration, after.duration)
            self.assertEqual(before.defined_channels, after.defined_channels)

            self.assertEqual(before.measurement_declarations, after.measurement_declarations)
            self.assertEqual(before.parameter_constraints, after.parameter_constraints)


class FunctionPulseSequencingTest(FunctionPulseTest):
    def test_build_waveform(self) -> None:
        with self.assertRaises(ParameterConstraintViolation):
            self.fpt.build_waveform(self.invalid_par_vals, channel_mapping={'A': 'B'})

        wf = self.fpt.build_waveform(self.valid_par_vals, channel_mapping={'A': 'B'})
        self.assertIsNotNone(wf)
        self.assertIsInstance(wf, FunctionWaveform)

        expression = Expression(self.s).evaluate_symbolic(self.valid_par_vals)
        duration = Expression(self.s2).evaluate_numeric(c=self.valid_par_vals['c'])

        expected_waveform = FunctionWaveform(expression, duration=duration, channel='B')
        self.assertEqual(expected_waveform, wf)

    def test_build_waveform_none(self):
        self.assertIsNone(self.fpt.build_waveform(self.valid_par_vals, channel_mapping={'A': None}))


class TablePulseTemplateConstraintTest(ParameterConstrainerTest):
    def __init__(self, *args, **kwargs):

        def tpt_constructor(parameter_constraints=None):
            return FunctionPulseTemplate('a*t', 'duration',
                                         parameter_constraints=parameter_constraints, measurements=[('M', 'n', 1)])

        super().__init__(*args,
                         to_test_constructor=tpt_constructor, **kwargs)


class TablePulseTemplateMeasurementTest(MeasurementDefinerTest):
    def __init__(self, *args, **kwargs):

        def tpt_constructor(measurements=None):
            return FunctionPulseTemplate('a*t', 'duration',
                                         parameter_constraints=['a < b'], measurements=measurements)

        super().__init__(*args,
                         to_test_constructor=tpt_constructor, **kwargs)




class FunctionPulseMeasurementTest(unittest.TestCase):
    def assert_window_equal(self, w1, w2):
        self.assertEqual(len(w1), len(w2))
        self.assertEqual(type(w1), type(w2))
        for x, y in zip(w1, w2):
            self.assertEqual(type(y), type(y))
            if isinstance(x, str):
                self.assertEqual(sympy.sympify(x), sympy.sympify(y))
            else:
                self.assertEqual(x, y)

    def assert_declaration_dict_equal(self, d1, d2):
        self.assertEqual(set(d1.keys()), set(d2.keys()))

        for k in d1.keys():
            self.assertEqual(len(d1[k]), len(d2[k]))
            for w1, w2 in zip(d1[k], d2[k]):
                self.assert_window_equal(w1, w2)

    def test_measurement_windows(self) -> None:
        pulse = FunctionPulseTemplate(5, 5, measurements=[('mw', 0, 5)])

        windows = pulse.get_measurement_windows(parameters={}, measurement_mapping={'mw': 'asd'})
        self.assertEqual([('asd', 0, 5)], windows)
        self.assertEqual(pulse.measurement_declarations, [('mw', 0, 5)])

    def test_no_measurement_windows(self) -> None:
        pulse = FunctionPulseTemplate(5, 5)

        windows = pulse.get_measurement_windows({}, {'mw': 'asd'})
        self.assertEqual([], windows)
        self.assertEqual([], pulse.measurement_declarations)

    def test_measurement_windows_with_parameters(self) -> None:
        pulse = FunctionPulseTemplate(5, 'length', measurements=[('mw', 1, '(1+length)/2')])

        parameters = dict(length=100)
        windows = pulse.get_measurement_windows(parameters, measurement_mapping={'mw': 'asd'})
        self.assertEqual(windows, [('asd', 1, 101/2)])

        declared = pulse.measurement_declarations
        self.assertEqual(declared, [('mw', 1, '(1+length)/2')])

    def test_multiple_measurement_windows(self) -> None:
        pulse = FunctionPulseTemplate(5, 'length',
                                      measurements=[('A', 0, '(1+length)/2'),
                                                    ('A', 1, 3),
                                                    ('B', 'begin', 2)])

        parameters = dict(length=5, begin=1)
        measurement_mapping = dict(A='A', B='C')
        windows = pulse.get_measurement_windows(parameters=parameters,
                                                measurement_mapping=measurement_mapping)
        expected = [('A', 0, 3), ('A', 1, 3), ('C', 1, 2)]
        self.assertEqual(sorted(windows), sorted(expected))
        self.assertEqual(pulse.measurement_declarations,
                         [('A', 0, '(1+length)/2'),
                          ('A', 1, 3),
                          ('B', 'begin', 2)])