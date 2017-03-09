import unittest
import sympy

from qctoolkit.pulses.function_pulse_template import FunctionPulseTemplate,\
    FunctionWaveform
from qctoolkit.pulses.parameters import ParameterDeclaration, ParameterNotProvidedException
from qctoolkit.expressions import Expression
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform
import numpy as np

from tests.serialization_dummies import DummySerializer
from tests.pulses.sequencing_dummies import DummyParameter


class FunctionPulseTest(unittest.TestCase):

    def setUp(self) -> None:
        self.maxDiff = None
        self.s = 'a + b * t'
        self.s2 = 'c'

        self.meas_list = [('mw', 1, 1), ('mw', 'x', 'z'), ('drup', 'j', 'u')]
        self.meas_dict = {'mw': [(1, 1), ('x', 'z')], 'drup': [('j', 'u')]}

        self.fpt = FunctionPulseTemplate(self.s, self.s2,channel='A')
        for mw in self.meas_list:
            self.fpt.add_measurement_declaration(*mw)

        self.pars = dict(a=DummyParameter(1), b=DummyParameter(2), c=DummyParameter(136.78))

    def test_is_interruptable(self) -> None:
        self.assertFalse(self.fpt.is_interruptable)

    def test_defined_channels(self) -> None:
        self.assertEqual({'A'}, self.fpt.defined_channels)

    def test_parameter_names_and_declarations_expression_input(self) -> None:
        template = FunctionPulseTemplate(Expression("3 * foo + bar * t"), Expression("5 * hugo"))
        expected_parameter_names = {'foo', 'bar', 'hugo'}
        self.assertEqual(expected_parameter_names, template.parameter_names)
        self.assertEqual({ParameterDeclaration(name) for name in expected_parameter_names}, template.parameter_declarations)

    def test_parameter_names_and_declarations_string_input(self) -> None:
        template = FunctionPulseTemplate("3 * foo + bar * t", "5 * hugo",channel='A')
        expected_parameter_names = {'foo', 'bar', 'hugo'}
        self.assertEqual(expected_parameter_names, template.parameter_names)
        self.assertEqual({ParameterDeclaration(name) for name in expected_parameter_names},
                         template.parameter_declarations)

    def test_serialization_data(self) -> None:
        expected_data = dict(duration_expression=str(self.s2),
                             expression=str(self.s),
                             channel='A',
                             measurement_declarations=self.meas_dict)
        self.assertEqual(expected_data, self.fpt.get_serialization_data(
            DummySerializer(serialize_callback=lambda x: str(x))))

    def test_deserialize(self) -> None:
        basic_data = dict(duration_expression=str(self.s2),
                          expression=str(self.s),
                          channel='A',
                          identifier='hugo',
                          measurement_declarations=self.meas_dict)
        serializer = DummySerializer(serialize_callback=lambda x: str(x))
        serializer.subelements[str(self.s2)] = Expression(self.s2)
        serializer.subelements[str(self.s)] = Expression(self.s)
        template = FunctionPulseTemplate.deserialize(serializer, **basic_data)
        self.assertEqual('hugo', template.identifier)
        self.assertEqual({'a', 'b', 'c', 'x', 'z', 'j', 'u'}, template.parameter_names)
        self.assertEqual({ParameterDeclaration(name) for name in template.parameter_names},
                         template.parameter_declarations)
        self.assertEqual(template.measurement_declarations,
                         self.meas_dict)
        serialized_data = template.get_serialization_data(serializer)
        del basic_data['identifier']
        self.assertEqual(basic_data, serialized_data)


class FunctionPulseSequencingTest(unittest.TestCase):

    def setUp(self) -> None:
        unittest.TestCase.setUp(self)
        self.f = "a * t"
        self.duration = "y"
        self.args = dict(a=DummyParameter(3),y=DummyParameter(1))
        self.fpt = FunctionPulseTemplate(self.f, self.duration)

    @unittest.skip
    def test_build_waveform(self) -> None:
        wf = self.fpt.build_waveform(self.args, {}, channel_mapping={'default': 'default'})
        self.assertIsNotNone(wf)
        self.assertIsInstance(wf, MultiChannelWaveform)
        expected_waveform = MultiChannelWaveform({'default': FunctionWaveform(Expression(self.f),
                                                                              Expression(self.duration))})
        self.assertEqual(expected_waveform, wf)

    def test_requires_stop(self) -> None:
        parameters = dict(a=DummyParameter(36.126), y=DummyParameter(247.9543))
        self.assertFalse(self.fpt.requires_stop(parameters, dict()))
        parameters = dict(a=DummyParameter(36.126), y=DummyParameter(247.9543, requires_stop=True))
        self.assertTrue(self.fpt.requires_stop(parameters, dict()))


class FunctionWaveformTest(unittest.TestCase):

    def test_equality(self) -> None:
        wf1a = FunctionWaveform(Expression('2*t'), 3, measurement_windows=[], channel='A')
        wf1b = FunctionWaveform(Expression('2*t'), 3, measurement_windows=[], channel='A')
        wf2 = FunctionWaveform(Expression('2*t'), 3, measurement_windows=[('K', 1, 2)], channel='A')
        wf3 = FunctionWaveform(Expression('2*t+2'), 3, measurement_windows=[], channel='A')
        wf4 = FunctionWaveform(Expression('2*t'), 4, measurement_windows=[], channel='A')
        self.assertEqual(wf1a, wf1a)
        self.assertEqual(wf1a, wf1b)
        self.assertNotEqual(wf1a, wf2)
        self.assertNotEqual(wf1a, wf3)
        self.assertNotEqual(wf1a, wf4)

    def test_defined_channels(self) -> None:
        wf = FunctionWaveform(Expression('t'), 4, measurement_windows=[], channel='A')
        self.assertEqual({'A'}, wf.defined_channels)

    def test_duration(self) -> None:
        wf = FunctionWaveform(expression=Expression('2*t'), duration=4/5, measurement_windows=[],
                              channel='A')
        self.assertEqual(4/5, wf.duration)

    def test_unsafe_sample(self):
        fw = FunctionWaveform(Expression('sin(2*pi*t) + 3'), 5, channel='A', measurement_windows=[])

        t = np.linspace(0, 5, dtype=float)
        expected_result = np.sin(2*np.pi*t) + 3
        result = fw.unsafe_sample(channel='A', sample_times=t)
        np.testing.assert_equal(result, expected_result)

        out_array = np.empty_like(t)
        result = fw.unsafe_sample(channel='A', sample_times=t, output_array=out_array)
        np.testing.assert_equal(result, expected_result)
        self.assertIs(result, out_array)


    @unittest.skip
    def test_sample(self) -> None:
        f = Expression("(t+1)**b")
        length = Expression("c**b")
        par = {"b":2,"c":10}
        fw = FunctionWaveform(f, length, measurement_windows=[], channel='A')
        a = np.arange(4)
        expected_result = [[1, 4, 9, 16]]
        result = fw.sample(a)
        self.assertTrue(np.all(result == expected_result))


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
        pulse = FunctionPulseTemplate(5, 5)

        pulse.add_measurement_declaration('mw', 0, 5)
        windows = pulse.get_measurement_windows(parameters={}, measurement_mapping={'mw': 'asd'})
        self.assertEqual([('asd', 0, 5)], windows)
        self.assertEqual(pulse.measurement_declarations, dict(mw=[(0, 5)]))

    def test_no_measurement_windows(self) -> None:
        pulse = FunctionPulseTemplate(5, 5)

        windows = pulse.get_measurement_windows({}, {'mw': 'asd'})
        self.assertEqual([], windows)
        self.assertEqual(dict(), pulse.measurement_declarations)

    def test_measurement_windows_with_parameters(self) -> None:
        pulse = FunctionPulseTemplate(5, 'length')

        pulse.add_measurement_declaration('mw',1,'(1+length)/2')
        parameters = dict(length=100)
        windows = pulse.get_measurement_windows(parameters, measurement_mapping={'mw': 'asd'})
        self.assertEqual(windows, [('asd', 1, 101/2)])

        declared = pulse.measurement_declarations
        expected = dict(mw=[(1, '(1+length)/2')])

        self.assert_declaration_dict_equal(declared, expected)

    def test_multiple_measurement_windows(self) -> None:
        pulse = FunctionPulseTemplate(5, 'length')

        pulse.add_measurement_declaration('A', 0, '(1+length)/2')
        pulse.add_measurement_declaration('A', 1, 3)
        pulse.add_measurement_declaration('B', 'begin', 2)

        parameters = dict(length=5, begin=1)
        measurement_mapping = dict(A='A', B='C')
        windows = pulse.get_measurement_windows(parameters=parameters,
                                                measurement_mapping=measurement_mapping)
        expected = [('A', 0, 3), ('A', 1, 3), ('C', 1, 2)]
        self.assertEqual(sorted(windows), sorted(expected))

        self.assert_declaration_dict_equal(pulse.measurement_declarations,
                                           dict(A=[(0, '(1+length)/2'), (1, 3)],
                                                B=[('begin', 2)]))