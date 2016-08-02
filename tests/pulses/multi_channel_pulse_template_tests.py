import unittest

import numpy

from qctoolkit.pulses.pulse_template_parameter_mapping import MissingMappingException, MissingParameterDeclarationException
from qctoolkit.pulses.parameters import ParameterDeclaration, ParameterNotProvidedException, MappedParameter, ConstantParameter
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelPulseTemplate, MultiChannelWaveform
from qctoolkit.expressions import Expression

from tests.pulses.sequencing_dummies import DummySequencer, DummyInstructionBlock, DummyPulseTemplate, DummyWaveform
from tests.serialization_dummies import DummySerializer


class MultiChannelWaveformTest(unittest.TestCase):

    def test_init_no_args(self) -> None:
        with self.assertRaises(ValueError):
            MultiChannelWaveform([])
        with self.assertRaises(ValueError):
            MultiChannelWaveform(None)

    def test_init_single_channel(self) -> None:
        dwf = DummyWaveform(duration=1.3)
        with self.assertRaises(ValueError):
            MultiChannelWaveform([(dwf, [1])])
        with self.assertRaises(ValueError):
            MultiChannelWaveform([(dwf, [-1])])
        waveform = MultiChannelWaveform([(dwf, [0])])
        self.assertEqual(1, waveform.num_channels)
        self.assertEqual(1.3, waveform.duration)

    def test_init_several_channels(self) -> None:
        dwfa = DummyWaveform(duration=4.2, num_channels=2)
        dwfb = DummyWaveform(duration=4.2, num_channels=3)
        dwfc = DummyWaveform(duration=2.3)
        with self.assertRaises(ValueError):
            MultiChannelWaveform([(dwfa, [2, 4]), (dwfb, [3, 5, 1])])
        with self.assertRaises(ValueError):
            MultiChannelWaveform([(dwfa, [2, 4]), (dwfb, [3, -1, 1])])
        with self.assertRaises(ValueError):
            MultiChannelWaveform([(dwfa, [0, 1]), (dwfc, [2])])
        with self.assertRaises(ValueError):
            MultiChannelWaveform([(dwfa, [0, 0]), (dwfb, [3, 4, 1])])
        with self.assertRaises(ValueError):
            MultiChannelWaveform([(dwfa, [2, 4]), (dwfb, [3, 4, 1])])
        waveform = MultiChannelWaveform([(dwfa, [2, 4]), (dwfb, [3, 0, 1])])
        self.assertEqual(5, waveform.num_channels)
        self.assertEqual(4.2, waveform.duration)

    def test_sample(self) -> None:
        sample_times = numpy.linspace(98.5, 103.5, num=11)
        samples_a = numpy.array([
            [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10],
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
#            [0, 0.5], [1, 0.6], [2, 0.7], [3, 0.8], [4, 0.9], [5, 1.0],
#            [6, 1.1], [7, 1.2], [8, 1.3], [9, 1.4], [10, 1.5]
        ])
        samples_b = numpy.array([
            [-10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20]
#           [-10], [-11], [-12], [-13], [-14], [-15], [-16], [-17], [-18], [-19], [-20]
        ])
        dwf_a = DummyWaveform(duration=3.2, sample_output=samples_a, num_channels=2)
        dwf_b = DummyWaveform(duration=3.2, sample_output=samples_b, num_channels=1)
        waveform = MultiChannelWaveform([(dwf_a, [2, 0]), (dwf_b, [1])])
        self.assertEqual(3, waveform.num_channels)
        self.assertEqual(3.2, waveform.duration)

        result = waveform.sample(sample_times, 0.7)
        self.assertEqual([(list(sample_times), 0.7)], dwf_a.sample_calls)
        self.assertEqual([(list(sample_times), 0.7)], dwf_b.sample_calls)

        # expected = numpy.array([
        #     [0.5, -10, 0],
        #     [0.6, -11, 1],
        #     [0.7, -12, 2],
        #     [0.8, -13, 3],
        #     [0.9, -14, 4],
        #     [1.0, -15, 5],
        #     [1.1, -16, 6],
        #     [1.2, -17, 7],
        #     [1.3, -18, 8],
        #     [1.4, -19, 9],
        #     [1.5, -20, 10],
        # ])
        expected = numpy.array([
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            [-10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20],
            [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10]
        ])
        self.assertTrue(numpy.all(expected == result))

    def test_equality(self) -> None:
        dwf_a = DummyWaveform(duration=246.2, num_channels=2)
        waveform_a1 = MultiChannelWaveform([(dwf_a, [0, 1])])
        waveform_a2 = MultiChannelWaveform([(dwf_a, [0, 1])])
        waveform_a3 = MultiChannelWaveform([(dwf_a, [1, 0])])
        self.assertEqual(waveform_a1, waveform_a1)
        self.assertEqual(waveform_a1, waveform_a2)
        self.assertNotEqual(waveform_a1, waveform_a3)


class MultiChannelPulseTemplateTest(unittest.TestCase):

    def test_init_empty(self) -> None:
        template = MultiChannelPulseTemplate([], {}, identifier='foo')
        self.assertEqual('foo', template.identifier)
        self.assertFalse(template.parameter_names)
        self.assertFalse(template.parameter_declarations)
        self.assertTrue(template.is_interruptable)
        self.assertFalse(template.requires_stop(dict(), dict()))
        self.assertEqual(0, template.num_channels)

    def test_init_single_subtemplate_no_external_params(self) -> None:
        subtemplate = DummyPulseTemplate(parameter_names={'foo'}, num_channels=2, duration=1.3)
        template = MultiChannelPulseTemplate([(subtemplate, {'foo': "2.3"}, [1, 0])], {})
        self.assertFalse(template.parameter_names)
        self.assertFalse(template.parameter_declarations)
        self.assertFalse(template.is_interruptable)
        self.assertFalse(template.requires_stop(dict(), dict()))
        self.assertEqual(2, template.num_channels)

    def test_init_single_subtemplate_requires_stop_external_params(self) -> None:
        subtemplate = DummyPulseTemplate(parameter_names={'foo'}, requires_stop=True, num_channels=2, duration=1.3)
        template = MultiChannelPulseTemplate([(subtemplate, {'foo': "2.3 ** bar"}, [1, 0])], {'bar'})
        self.assertEqual({'bar'}, template.parameter_names)
        self.assertEqual({ParameterDeclaration('bar')}, template.parameter_declarations)
        self.assertFalse(template.is_interruptable)
        self.assertTrue(template.requires_stop(dict(bar=ConstantParameter(3.5)), dict()))
        self.assertEqual(2, template.num_channels)

    def test_init_single_subtemplate_invalid_channel_mapping(self) -> None:
        subtemplate = DummyPulseTemplate(parameter_names={'foo'}, num_channels=2, duration=1.3)
        with self.assertRaises(ValueError):
            MultiChannelPulseTemplate([(subtemplate, {'foo': "2.3"}, [3, 0])], {})
        with self.assertRaises(ValueError):
            MultiChannelPulseTemplate([(subtemplate, {'foo': "2.3"}, [-1, 0])], {})

    def test_init_multi_subtemplates_not_interruptable_requires_stop(self) -> None:
        st1 = DummyPulseTemplate(parameter_names={'foo'}, requires_stop=True, num_channels=2,
                                 duration=1.3)
        st2 = DummyPulseTemplate(parameter_names={'bar'}, is_interruptable=True, num_channels=1,
                                 duration=6.34)
        template = MultiChannelPulseTemplate(
            [
                (st1, {'foo': "2.3 ** bar"}, [0, 2]),
                (st2, {'bar': "bar"}, [1])
            ],
            {'bar'}
        )
        self.assertEqual({'bar'}, template.parameter_names)
        self.assertEqual({ParameterDeclaration('bar')}, template.parameter_declarations)
        self.assertFalse(template.is_interruptable)
        self.assertTrue(template.requires_stop(dict(bar=ConstantParameter(52.6)), dict()))
        self.assertEqual(3, template.num_channels)

    def test_init_multi_subtemplates_interruptable_no_requires_stop(self) -> None:
        st1 = DummyPulseTemplate(parameter_names={'foo'}, is_interruptable=True, num_channels=2,
                                 duration=1.3)
        st2 = DummyPulseTemplate(parameter_names={'bar'}, is_interruptable=True, num_channels=1,
                                 duration=6.34)
        template = MultiChannelPulseTemplate(
            [
                (st1, {'foo': "2.3 ** bar"}, [0, 2]),
                (st2, {'bar': "bar"}, [1])
            ],
            {'bar'}
        )
        self.assertEqual({'bar'}, template.parameter_names)
        self.assertEqual({ParameterDeclaration('bar')}, template.parameter_declarations)
        self.assertTrue(template.is_interruptable)
        self.assertFalse(template.requires_stop(dict(bar=ConstantParameter(4.1)), dict()))
        self.assertEqual(3, template.num_channels)

    def test_init_multi_subtemplates_wrong_channel_mapping(self) -> None:
        st1 = DummyPulseTemplate(parameter_names={'foo'}, is_interruptable=True, num_channels=2,
                                 duration=1.3)
        st2 = DummyPulseTemplate(parameter_names={'bar'}, is_interruptable=True, num_channels=1,
                                 duration=6.34)
        with self.assertRaises(ValueError):
            MultiChannelPulseTemplate(
                [
                    (st1, {'foo': "2.3 ** bar"}, [0, 3]),
                    (st2, {'bar': "bar"}, [1])
                ],
                {'bar'}
            )
        with self.assertRaises(ValueError):
            MultiChannelPulseTemplate(
                [
                    (st1, {'foo': "2.3 ** bar"}, [0, -1]),
                    (st2, {'bar': "bar"}, [1])
                ],
                {'bar'}
            )
        with self.assertRaises(ValueError):
            MultiChannelPulseTemplate(
                [
                    (st1, {'foo': "2.3 ** bar"}, [0, 2]),
                    (st2, {'bar': "bar"}, [-1])
                ],
                {'bar'}
            )
        with self.assertRaises(ValueError):
            MultiChannelPulseTemplate(
                [
                    (st1, {'foo': "2.3 ** bar"}, [0, 2]),
                    (st2, {'bar': "bar"}, [3])
                ],
                {'bar'}
            )
        with self.assertRaises(ValueError):
            MultiChannelPulseTemplate(
                [
                    (st1, {'foo': "2.3 ** bar"}, [0, 0]),
                    (st2, {'bar': "bar"}, [1])
                ],
                {'bar'}
            )
        with self.assertRaises(ValueError):
            MultiChannelPulseTemplate(
                [
                    (st1, {'foo': "2.3 ** bar"}, [0, 2]),
                    (st2, {'bar': "bar"}, [2])
                ],
                {'bar'}
            )

    def test_init_broken_mappings(self) -> None:
        st1 = DummyPulseTemplate(parameter_names={'foo'}, is_interruptable=True, num_channels=2,
                                 duration=1.3)
        st2 = DummyPulseTemplate(parameter_names={'bar'}, is_interruptable=True, num_channels=1,
                                 duration=6.34)
        with self.assertRaises(MissingMappingException):
            MultiChannelPulseTemplate([(st1, {'foo': "bar"}, [0, 2]), (st2, {}, [1])], {'bar'})
        with self.assertRaises(MissingMappingException):
            MultiChannelPulseTemplate([(st1, {}, [0, 2]), (st2, {'bar': "bar"}, [1])], {'bar'})
        with self.assertRaises(MissingParameterDeclarationException):
            MultiChannelPulseTemplate(
                [
                    (st1, {'foo': "2.3 ** bar"}, [0, 2]),
                    (st2, {'bar': "bar"}, [1])
                ],
                {}
            )


class MultiChannelPulseTemplateSequencingTests(unittest.TestCase):

    def test_requires_stop_false_mapped_parameters(self) -> None:
        dummy = DummyPulseTemplate(parameter_names={'foo'})
        pulse = MultiChannelPulseTemplate([(dummy, dict(foo='2*bar'), [0]),
                                           (dummy, dict(foo='rab-5'), [1])],
                                          {'bar', 'rab'})
        self.assertEqual({'bar', 'rab'}, pulse.parameter_names)
        self.assertEqual({ParameterDeclaration('bar'), ParameterDeclaration('rab')},
                         pulse.parameter_declarations)
        parameters = dict(bar=ConstantParameter(-3.6), rab=ConstantParameter(35.26))
        self.assertFalse(pulse.requires_stop(parameters, dict()))

    def test_requires_stop_true_mapped_parameters(self) -> None:
        dummy = DummyPulseTemplate(parameter_names={'foo'}, requires_stop=True)
        pulse = MultiChannelPulseTemplate([(dummy, dict(foo='2*bar'), [0]),
                                           (dummy, dict(foo='rab-5'), [1])],
                                          {'bar', 'rab'})
        self.assertEqual({'bar', 'rab'}, pulse.parameter_names)
        self.assertEqual({ParameterDeclaration('bar'), ParameterDeclaration('rab')},
                         pulse.parameter_declarations)
        parameters = dict(bar=ConstantParameter(-3.6), rab=ConstantParameter(35.26))
        self.assertTrue(pulse.requires_stop(parameters, dict()))

    def test_build_sequence_no_params(self) -> None:
        dummy1 = DummyPulseTemplate(parameter_names={'foo'})
        pulse = MultiChannelPulseTemplate([(dummy1, {'foo': '2*bar'}, [1]),
                                           (dummy1, {'foo': '3'}, [0])], {'bar'})

        self.assertEqual({'bar'}, pulse.parameter_names)
        self.assertEqual({ParameterDeclaration('bar')}, pulse.parameter_declarations)

        with self.assertRaises(ParameterNotProvidedException):
            pulse.build_waveform({})

        with self.assertRaises(ParameterNotProvidedException):
            pulse.build_sequence(DummySequencer(), dict(), dict(), DummyInstructionBlock())

    def test_build_sequence(self) -> None:
        dummy_wf1 = DummyWaveform(duration=2.3, num_channels=2)
        dummy_wf2 = DummyWaveform(duration=2.3, num_channels=1)
        dummy1 = DummyPulseTemplate(parameter_names={'foo'}, num_channels=2, waveform=dummy_wf1)
        dummy2 = DummyPulseTemplate(parameter_names={}, num_channels=1, waveform=dummy_wf2)

        pulse = MultiChannelPulseTemplate([(dummy1, {'foo': '2*bar'}, [2, 1]),
                                           (dummy2, {}, [0])], {'bar'})

        result = pulse.build_waveform({'bar': ConstantParameter(3)})
        expected = MultiChannelWaveform([(dummy_wf1, [2, 1]), (dummy_wf2, [0])])
        self.assertEqual(expected, result)
        self.assertEqual([{'foo': MappedParameter(Expression("2*bar"), {'bar': ConstantParameter(3)})}], dummy1.build_waveform_calls)
        self.assertEqual([{}], dummy2.build_waveform_calls)

    def test_integration_table_and_function_template(self) -> None:
        from qctoolkit.pulses import TablePulseTemplate, FunctionPulseTemplate, Sequencer, EXECInstruction, STOPInstruction

        table_template = TablePulseTemplate(channels=2)
        table_template.add_entry(1, 4, channel=0)
        table_template.add_entry('foo', 'bar', channel=0)
        table_template.add_entry(10, 0, channel=0)
        table_template.add_entry('foo', 2.7, interpolation='linear', channel=1)
        table_template.add_entry(9, 'bar', interpolation='linear', channel=1)

        function_template = FunctionPulseTemplate('sin(t)', '10')

        template = MultiChannelPulseTemplate(
            [(function_template, dict(), [1]),
             (table_template, dict(foo='5', bar='2 * hugo'), [2, 0])],
            {'hugo'}
        )

        sample_times = numpy.linspace(98.5, 103.5, num=11)
        function_template_samples = function_template.build_waveform(dict()).sample(sample_times)
        table_template_samples = table_template.build_waveform(dict(foo=ConstantParameter(5), bar=ConstantParameter(2*(-1.3)))).sample(sample_times)

        template_waveform = template.build_waveform(dict(hugo=ConstantParameter(-1.3)))
        template_samples = template_waveform.sample(sample_times)

        self.assertTrue(numpy.all(table_template_samples[0] == template_samples[2]))
        self.assertTrue(numpy.all(table_template_samples[1] == template_samples[0]))
        self.assertTrue(numpy.all(function_template_samples[0] == template_samples[1]))

        sequencer = Sequencer()
        sequencer.push(template, parameters=dict(hugo=-1.3), conditions=dict())
        instructions = sequencer.build()
        self.assertEqual(2, len(instructions))
        self.assertIsInstance(instructions[0], EXECInstruction)
        self.assertIsInstance(instructions[1], STOPInstruction)


class MutliChannelPulseTemplateSerializationTests(unittest.TestCase):

    def __init__(self, methodName) -> None:
        super().__init__(methodName=methodName)
        self.maxDiff = None

    def test_get_serialization_data(self) -> None:
        serializer = DummySerializer(
            serialize_callback=lambda x: str(x) if isinstance(x, Expression) else str(id(x)))
        dummy1 = DummyPulseTemplate(parameter_names={'foo'}, num_channels=2)
        dummy2 = DummyPulseTemplate(parameter_names={}, num_channels=1)
        template = MultiChannelPulseTemplate(
            [
                (dummy1, {'foo': "bar+3"}, [0, 2]),
                (dummy2, {}, [1])
             ],
            {'bar'},
            identifier='herbert'
        )
        expected_data = dict(
            external_parameters=['bar'],
            subtemplates = [
                dict(template=str(id(dummy1)),
                     parameter_mappings=dict(foo=str(Expression("bar+3"))),
                     channel_mappings=[0, 2]),
                dict(template=str(id(dummy2)),
                     parameter_mappings=dict(),
                     channel_mappings=[1])
            ]
        )
        data = template.get_serialization_data(serializer)
        self.assertEqual(expected_data, data)

    def test_deserialize(self) -> None:
        dummy1 = DummyPulseTemplate(parameter_names={'foo'}, num_channels=2)
        dummy2 = DummyPulseTemplate(parameter_names={}, num_channels=1)
        exp = Expression("bar - 35")

        data = dict(
            external_parameters=['bar'],
            subtemplates=[
                dict(template=str(id(dummy1)),
                     parameter_mappings=dict(foo=str(exp)),
                     channel_mappings=[0, 2]),
                dict(template=str(id(dummy2)),
                     parameter_mappings=dict(),
                     channel_mappings=[1])
            ]
        )

        serializer = DummySerializer(serialize_callback=lambda x: str(x) if isinstance(x, Expression) else str(id(x)))
        serializer.subelements[str(id(dummy1))] = dummy1
        serializer.subelements[str(id(dummy2))] = dummy2
        serializer.subelements[str(exp)] = exp

        template = MultiChannelPulseTemplate.deserialize(serializer, **data)
        self.assertEqual(set(data['external_parameters']), template.parameter_names)
        self.assertEqual({ParameterDeclaration('bar')}, template.parameter_declarations)

        recovered_data = template.get_serialization_data(serializer)
        self.assertEqual(data, recovered_data)

