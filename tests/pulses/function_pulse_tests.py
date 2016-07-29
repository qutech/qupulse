import unittest

from qctoolkit.pulses.function_pulse_template import FunctionPulseTemplate,\
    FunctionWaveform
from qctoolkit.pulses.parameters import ParameterDeclaration, ParameterNotProvidedException
from qctoolkit.expressions import Expression
import numpy as np

from tests.serialization_dummies import DummySerializer
from tests.pulses.sequencing_dummies import DummyParameter


class FunctionPulseTest(unittest.TestCase):

    def setUp(self) -> None:
        self.maxDiff = None
        self.s = 'a + b * t'
        self.s2 = 'c'
        self.fpt = FunctionPulseTemplate(self.s, self.s2)
        self.pars = dict(a=DummyParameter(1), b=DummyParameter(2), c=DummyParameter(136.78))

    def test_get_pulse_length(self) -> None:
        self.assertEqual(136.78, self.fpt.get_pulse_length(self.pars))

    def test_get_pulse_length_missing_parameter(self) -> None:
        with self.assertRaises(ParameterNotProvidedException):
            self.fpt.get_pulse_length(dict(b=DummyParameter(26.3267)))

    def test_is_interruptable(self) -> None:
        self.assertFalse(self.fpt.is_interruptable)

    def test_num_channels(self) -> None:
        self.assertEqual(1, self.fpt.num_channels)

    def test_parameter_names_and_declarations_expression_input(self) -> None:
        template = FunctionPulseTemplate(Expression("3 * foo + bar * t"), Expression("5 * hugo"))
        expected_parameter_names = {'foo', 'bar', 'hugo'}
        self.assertEqual(expected_parameter_names, template.parameter_names)
        self.assertEqual({ParameterDeclaration(name) for name in expected_parameter_names}, template.parameter_declarations)

    def test_parameter_names_and_declarations_string_input(self) -> None:
        template = FunctionPulseTemplate("3 * foo + bar * t", "5 * hugo")
        expected_parameter_names = {'foo', 'bar', 'hugo'}
        self.assertEqual(expected_parameter_names, template.parameter_names)
        self.assertEqual({ParameterDeclaration(name) for name in expected_parameter_names},
                         template.parameter_declarations)

    def test_serialization_data(self) -> None:
        expected_data = dict(duration_expression=str(self.s2),
                             expression=str(self.s),
                             measurement=False)
        self.assertEqual(expected_data, self.fpt.get_serialization_data(
            DummySerializer(serialize_callback=lambda x: str(x))))

    def test_deserialize(self) -> None:
        basic_data = dict(duration_expression=str(self.s2),
                          expression=str(self.s),
                          measurement=False,
                          identifier='hugo')
        serializer = DummySerializer(serialize_callback=lambda x: str(x))
        serializer.subelements[str(self.s2)] = Expression(self.s2)
        serializer.subelements[str(self.s)] = Expression(self.s)
        template = FunctionPulseTemplate.deserialize(serializer, **basic_data)
        self.assertEqual('hugo', template.identifier)
        self.assertEqual({'a', 'b', 'c'}, template.parameter_names)
        self.assertEqual({ParameterDeclaration(name) for name in {'a', 'b', 'c'}},
                         template.parameter_declarations)
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

    def test_build_waveform(self) -> None:
        wf = self.fpt.build_waveform(self.args)
        self.assertIsNotNone(wf)
        self.assertIsInstance(wf, FunctionWaveform)
        expected_waveform = FunctionWaveform(dict(a=3, y=1), Expression(self.f), Expression(self.duration))
        self.assertEqual(expected_waveform, wf)

    def test_requires_stop(self) -> None:
        parameters = dict(a=DummyParameter(36.126), y=DummyParameter(247.9543))
        self.assertFalse(self.fpt.requires_stop(parameters, dict()))
        parameters = dict(a=DummyParameter(36.126), y=DummyParameter(247.9543, requires_stop=True))
        self.assertTrue(self.fpt.requires_stop(parameters, dict()))


class FunctionWaveformTest(unittest.TestCase):

    def test_equality(self) -> None:
        wf1a = FunctionWaveform(dict(a=2, b=1), Expression('a*t'), Expression('b'))
        wf1b = FunctionWaveform(dict(a=2, b=1), Expression('a*t'), Expression('b'))
        wf2 = FunctionWaveform(dict(a=3, b=1), Expression('a*t'), Expression('b'))
        wf3 = FunctionWaveform(dict(a=2, b=1), Expression('a*t+2'), Expression('b'))
        wf4 = FunctionWaveform(dict(a=2, c=2), Expression('a*t'), Expression('c'))
        self.assertEqual(wf1a, wf1a)
        self.assertEqual(wf1a, wf1b)
        self.assertNotEqual(wf1a, wf2)
        self.assertNotEqual(wf1a, wf3)
        self.assertNotEqual(wf1a, wf4)

    def test_num_channels(self) -> None:
        wf = FunctionWaveform(dict(), Expression('t'), Expression('4'))
        self.assertEqual(1, wf.num_channels)

    def test_duration(self) -> None:
        wf = FunctionWaveform(dict(foo=2.5), Expression('2*t'), Expression('4*foo/5'))
        self.assertEqual(2, wf.duration)

    def test_sample(self) -> None:
        f = Expression("(t+1)**b")
        length = Expression("c**b")
        par = {"b":2,"c":10}
        fw = FunctionWaveform(par,f,length)
        a = np.arange(4)
        self.assertEqual(list(fw.sample(a)), [1,4,9,16])
        