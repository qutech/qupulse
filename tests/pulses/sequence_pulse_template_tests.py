import unittest
import copy

import numpy as np

from qctoolkit.pulses.pulse_template import DoubleParameterNameException
from qctoolkit.expressions import Expression
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate, SequenceWaveform
from qctoolkit.pulses.pulse_template_parameter_mapping import MissingMappingException, UnnecessaryMappingException, MissingParameterDeclarationException, MappingPulseTemplate
from qctoolkit.pulses.parameters import ParameterNotProvidedException, ConstantParameter, ParameterConstraint, ParameterConstraintViolation

from tests.pulses.sequencing_dummies import DummySequencer, DummyInstructionBlock, DummyPulseTemplate,\
    DummyNoValueParameter, DummyWaveform
from tests.serialization_dummies import DummySerializer


class SequenceWaveformTest(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def test_init(self):
        dwf_ab = DummyWaveform(duration=1.1, defined_channels={'A', 'B'})
        dwf_abc = DummyWaveform(duration=2.2, defined_channels={'A', 'B', 'C'})

        with self.assertRaises(ValueError):
            SequenceWaveform([])

        with self.assertRaises(ValueError):
            SequenceWaveform((dwf_ab, dwf_abc))

        swf1 = SequenceWaveform((dwf_ab, dwf_ab))
        self.assertEqual(swf1.duration, 2*dwf_ab.duration)
        self.assertEqual(len(swf1.compare_key), 2)

        swf2 = SequenceWaveform((swf1, dwf_ab))
        self.assertEqual(swf2.duration, 3 * dwf_ab.duration)

        self.assertEqual(len(swf2.compare_key), 3)

    def test_unsafe_sample(self):
        dwfs = (DummyWaveform(duration=1., sample_output=np.linspace(5, 6, num=10)),
                DummyWaveform(duration=3., sample_output=np.linspace(1, 2, num=30)),
                DummyWaveform(duration=2., sample_output=np.linspace(8, 9, num=20)))

        swf = SequenceWaveform(dwfs)

        sample_times = np.arange(0, 60)*0.1
        expected_output = np.concatenate(tuple(dwf.sample_output for dwf in dwfs))

        output = swf.unsafe_sample('A', sample_times=sample_times)
        np.testing.assert_equal(expected_output, output)

        output_2 = swf.unsafe_sample('A', sample_times=sample_times, output_array=output)
        self.assertIs(output_2, output)

    def test_unsafe_get_subset_for_channels(self):
        dwf_1 = DummyWaveform(duration=2.2, defined_channels={'A', 'B', 'C'})
        dwf_2 = DummyWaveform(duration=3.3, defined_channels={'A', 'B', 'C'})

        wf = SequenceWaveform([dwf_1, dwf_2])

        subset = {'A', 'C'}
        sub_wf = wf.unsafe_get_subset_for_channels(subset)
        self.assertIsInstance(sub_wf, SequenceWaveform)

        self.assertEqual(len(sub_wf.compare_key), 2)
        self.assertEqual(sub_wf.compare_key[0].defined_channels, subset)
        self.assertEqual(sub_wf.compare_key[1].defined_channels, subset)

        self.assertEqual(sub_wf.compare_key[0].duration, 2.2)
        self.assertEqual(sub_wf.compare_key[1].duration, 3.3)



    def test_get_measurement_windows(self):
        dwfs = (DummyWaveform(duration=1., measurement_windows=[('M', 0.2, 0.5)]),
                DummyWaveform(duration=3., measurement_windows=[('N', 0.6, 0.7)]),
                DummyWaveform(duration=2., measurement_windows=[('M', 0.1, 0.2), ('N', 0.5, 0.6)]))
        swf = SequenceWaveform(dwfs)

        expected_windows = sorted((('M', 0.2, 0.5), ('N', 1.6, 0.7), ('M', 4.1, 0.2), ('N', 4.5, 0.6)))
        received_windows = sorted(tuple(swf.get_measurement_windows()))
        self.assertEqual(received_windows, expected_windows)


class SequencePulseTemplateTest(unittest.TestCase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Setup test data
        self.square = TablePulseTemplate({'default': [(0, 0),
                                                      ('up', 'v', 'hold'),
                                                      ('down', 0, 'hold'),
                                                      ('length', 0)]},
                                         measurements=[('mw1', 'up', 'length-up')])
        self.mapping1 = {
            'up': 'uptime',
            'down': 'uptime + length',
            'v': 'voltage',
            'length': '0.5 * pulse_length'
        }

        self.window_name_mapping = {'mw1' : 'test_window'}

        self.outer_parameters = {'uptime', 'length', 'pulse_length', 'voltage'}

        self.parameters = dict()
        self.parameters['uptime'] = ConstantParameter(5)
        self.parameters['length'] = ConstantParameter(10)
        self.parameters['pulse_length'] = ConstantParameter(100)
        self.parameters['voltage'] = ConstantParameter(10)

        self.sequence = SequencePulseTemplate(MappingPulseTemplate(self.square,
                                                                   parameter_mapping=self.mapping1,
                                                                   measurement_mapping=self.window_name_mapping),
                                              external_parameters=self.outer_parameters)

    def test_init(self):
        with self.assertRaises(MissingParameterDeclarationException):
            SequencePulseTemplate(DummyPulseTemplate(parameter_names={'a', 'b'}), external_parameters={'a'})

        with self.assertRaises(MissingParameterDeclarationException):
            SequencePulseTemplate(DummyPulseTemplate(parameter_names={'a'}),
                                  parameter_constraints=['b < 4'],
                                  external_parameters={'a'})

        with self.assertRaises(MissingMappingException):
            SequencePulseTemplate(DummyPulseTemplate(parameter_names={'a'}),
                                  parameter_constraints=['b < 4'],
                                  external_parameters={'a', 'b', 'c'})

    def test_duration(self):
        pt = SequencePulseTemplate(DummyPulseTemplate(duration='a'),
                                   DummyPulseTemplate(duration='a'),
                                   DummyPulseTemplate(duration='b'))
        self.assertEqual(pt.duration, Expression('a+a+b'))

    def test_build_waveform(self):
        wfs = [DummyWaveform(), DummyWaveform()]
        pts = [DummyPulseTemplate(waveform=wf) for wf in wfs]

        spt = SequencePulseTemplate(*pts, parameter_constraints=['a < 3'])
        with self.assertRaises(ParameterConstraintViolation):
            spt.build_waveform(dict(a=4), dict(), dict())

        parameters = dict(a=2)
        channel_mapping = dict()
        measurement_mapping = dict()
        wf = spt.build_waveform(parameters, channel_mapping=channel_mapping, measurement_mapping=measurement_mapping)

        for wfi, pt in zip(wfs, pts):
            self.assertEqual(pt.build_waveform_calls, [(parameters, dict(), dict())])
            self.assertIs(pt.build_waveform_calls[0][0], parameters)

        self.assertIsInstance(wf, SequenceWaveform)
        for wfa, wfb in zip(wf.compare_key, wfs):
            self.assertIs(wfa, wfb)

    def test_identifier(self) -> None:
        identifier = 'some name'
        pulse = SequencePulseTemplate(DummyPulseTemplate(), external_parameters=set(), identifier=identifier)
        self.assertEqual(identifier, pulse.identifier)

    def test_multiple_channels(self) -> None:
        dummy = DummyPulseTemplate(parameter_names={'hugo'}, defined_channels={'A', 'B'})
        subtemplates = [(dummy, {'hugo': 'foo'}, {}), (dummy, {'hugo': '3'}, {})]
        sequence = SequencePulseTemplate(*subtemplates, external_parameters={'foo'})
        self.assertEqual({'A', 'B'}, sequence.defined_channels)

    def test_multiple_channels_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            SequencePulseTemplate(DummyPulseTemplate(defined_channels={'A'}),
                                  DummyPulseTemplate(defined_channels={'B'}))

        with self.assertRaises(ValueError):
            SequencePulseTemplate(
                DummyPulseTemplate(defined_channels={'A'}), DummyPulseTemplate(defined_channels={'A', 'B'})
                , external_parameters=set())


class SequencePulseTemplateSerializationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.serializer = DummySerializer()

        self.table_foo = TablePulseTemplate({'default': [('hugo', 2),
                                                         ('albert', 'voltage')]},
                                            parameter_constraints=['albert<9.1'],
                                            measurements=[('mw_foo','hugo','albert')],
                                            identifier='foo')

        self.foo_param_mappings = dict(hugo='ilse', albert='albert', voltage='voltage')
        self.foo_meas_mappings = dict(mw_foo='mw_bar')

    def test_get_serialization_data(self) -> None:
        dummy1 = DummyPulseTemplate()
        dummy2 = DummyPulseTemplate()

        sequence = SequencePulseTemplate(dummy1, dummy2, parameter_constraints=['a<b'])
        serializer = DummySerializer(serialize_callback=lambda x: str(x))

        expected_data = dict(
            subtemplates=[str(dummy1), str(dummy2)],
            parameter_constraints=[ParameterConstraint('a<b')]
        )
        data = sequence.get_serialization_data(serializer)
        self.assertEqual(expected_data, data)

    def test_deserialize(self) -> None:
        dummy1 = DummyPulseTemplate()
        dummy2 = DummyPulseTemplate()

        serializer = DummySerializer(serialize_callback=lambda x: str(id(x)))

        data = dict(
            subtemplates=[serializer.dictify(dummy1), serializer.dictify(dummy2)],
            identifier='foo',
            parameter_constraints=['a<b']
        )

        template = SequencePulseTemplate.deserialize(serializer, **data)
        self.assertEqual(template.subtemplates, [dummy1, dummy2])
        self.assertEqual(template.parameter_constraints, [ParameterConstraint('a<b')])


class SequencePulseTemplateSequencingTests(SequencePulseTemplateTest):
    def test_build_sequence(self) -> None:
        sub1 = DummyPulseTemplate(requires_stop=False)
        sub2 = DummyPulseTemplate(requires_stop=True, parameter_names={'foo'})
        parameters = {'foo': DummyNoValueParameter()}

        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        seq = SequencePulseTemplate(sub1, (sub2, {'foo': 'foo'}), external_parameters={'foo'})
        seq.build_sequence(sequencer, parameters, {}, {}, {}, block)
        self.assertEqual(2, len(sequencer.sequencing_stacks[block]))

        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        seq = SequencePulseTemplate((sub2, {'foo': 'foo'}), sub1, external_parameters={'foo'})
        seq.build_sequence(sequencer, parameters, {}, {}, {}, block)
        self.assertEqual(2, len(sequencer.sequencing_stacks[block]))

    @unittest.skip("Was this test faulty before? Why should the three last cases return false?")
    def test_requires_stop(self) -> None:
        sub1 = (DummyPulseTemplate(requires_stop=False), {}, {})
        sub2 = (DummyPulseTemplate(requires_stop=True, parameter_names={'foo'}), {'foo': 'foo'}, {})
        parameters = {'foo': DummyNoValueParameter()}

        seq = SequencePulseTemplate(sub1)
        self.assertFalse(seq.requires_stop(parameters, {}))

        seq = SequencePulseTemplate(sub2, external_parameters={'foo'})
        self.assertFalse(seq.requires_stop(parameters, {}))

        seq = SequencePulseTemplate(sub1, sub2, external_parameters={'foo'})
        self.assertFalse(seq.requires_stop(parameters, {}))

        seq = SequencePulseTemplate(sub2, sub1, external_parameters={'foo'})
        self.assertFalse(seq.requires_stop(parameters, {}))

    def test_missing_parameter_declaration_exception(self):
        mapping = copy.deepcopy(self.mapping1)
        mapping['up'] = "foo"

        subtemplates = [(self.square, mapping,{})]
        with self.assertRaises(MissingParameterDeclarationException):
            SequencePulseTemplate(*subtemplates, external_parameters=self.outer_parameters)

    def test_crash(self) -> None:
        table = TablePulseTemplate({'default': [('ta', 'va', 'hold'),
                                                ('tb', 'vb', 'linear'),
                                                ('tend', 0, 'jump')]}, identifier='foo')

        external_parameters = ['ta', 'tb', 'tc', 'td', 'va', 'vb', 'tend']
        first_mapping = {
            'ta': 'ta',
            'tb': 'tb',
            'va': 'va',
            'vb': 'vb',
            'tend': 'tend'
        }
        second_mapping = {
            'ta': 'tc',
            'tb': 'td',
            'va': 'vb',
            'vb': 'va + vb',
            'tend': '2 * tend'
        }
        sequence = SequencePulseTemplate((table, first_mapping, {}), (table, second_mapping, {}),
                                         external_parameters=external_parameters)

        parameters = {
            'ta': ConstantParameter(2),
            'va': ConstantParameter(2),
            'tb': ConstantParameter(4),
            'vb': ConstantParameter(3),
            'tc': ConstantParameter(5),
            'td': ConstantParameter(11),
            'tend': ConstantParameter(6)}

        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        self.assertFalse(sequence.requires_stop(parameters, {}))
        sequence.build_sequence(sequencer,
                                parameters=parameters,
                                conditions={},
                                measurement_mapping={},
                                channel_mapping={'default': 'default'},
                                instruction_block=block)
        from qctoolkit.pulses.sequencing import Sequencer
        s = Sequencer()
        s.push(sequence, parameters, channel_mapping={'default': 'EXAMPLE_A'})
        s.build()

    def test_missing_parameter_declaration_exception(self) -> None:
        mapping = copy.deepcopy(self.mapping1)
        mapping['up'] = "foo"

        subtemplates = [(self.square, mapping)]
        with self.assertRaises(MissingParameterDeclarationException):
            SequencePulseTemplate(*subtemplates, external_parameters=self.outer_parameters)


class SequencePulseTemplateTestProperties(SequencePulseTemplateTest):
    def test_is_interruptable(self):

        self.assertTrue(
            SequencePulseTemplate(DummyPulseTemplate(is_interruptable=True),
                                  DummyPulseTemplate(is_interruptable=True)).is_interruptable)
        self.assertTrue(
            SequencePulseTemplate(DummyPulseTemplate(is_interruptable=True),
                                  DummyPulseTemplate(is_interruptable=False)).is_interruptable)
        self.assertFalse(
            SequencePulseTemplate(DummyPulseTemplate(is_interruptable=False),
                                  DummyPulseTemplate(is_interruptable=False)).is_interruptable)



class PulseTemplateConcatenationTest(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def test_concatenation_pulse_template(self):
        a = DummyPulseTemplate(parameter_names={'foo'}, defined_channels={'A'})
        b = DummyPulseTemplate(parameter_names={'bar'}, defined_channels={'A'})
        c = DummyPulseTemplate(parameter_names={'snu'}, defined_channels={'A'})
        d = DummyPulseTemplate(parameter_names={'snu'}, defined_channels={'A'})

        seq = a @ a
        self.assertTrue(len(seq.subtemplates) == 2)
        for st in seq.subtemplates:
            self.assertEqual(st, a)

        seq = a @ b
        self.assertTrue(len(seq.subtemplates)==2)
        for st, expected in zip(seq.subtemplates,[a, b]):
            self.assertTrue(st, expected)

        with self.assertRaises(DoubleParameterNameException):
            a @ b @ a
        with self.assertRaises(DoubleParameterNameException):
            a @ b @ c @ d

        seq = a @ b @ c
        self.assertTrue(len(seq.subtemplates) == 3)
        for st, expected in zip(seq.subtemplates, [a, b, c]):
            self.assertTrue(st, expected)


    def test_concatenation_sequence_table_pulse(self):
        a = DummyPulseTemplate(parameter_names={'foo'}, defined_channels={'A'})
        b = DummyPulseTemplate(parameter_names={'bar'}, defined_channels={'A'})
        c = DummyPulseTemplate(parameter_names={'snu'}, defined_channels={'A'})
        d = DummyPulseTemplate(parameter_names={'snu'}, defined_channels={'A'})

        seq1 = SequencePulseTemplate(a, b, external_parameters=['foo', 'bar'])
        seq2 = SequencePulseTemplate(c, d, external_parameters=['snu'])

        seq = seq1 @ c
        self.assertTrue(len(seq.subtemplates) == 3)
        for st, expected in zip(seq.subtemplates,[a, b, c]):
            self.assertTrue(st, expected)

        seq = c @ seq1
        self.assertTrue(len(seq.subtemplates) == 3)
        for st, expected in zip(seq.subtemplates, [c, a, b]):
            self.assertTrue(st, expected)

        seq = seq1 @ seq2
        self.assertTrue(len(seq.subtemplates) == 4)
        for st, expected in zip(seq.subtemplates, [a, b, c, d]):
            self.assertTrue(st, expected)

        with self.assertRaises(DoubleParameterNameException):
            seq2 @ c

if __name__ == "__main__":
    unittest.main(verbosity=2)
