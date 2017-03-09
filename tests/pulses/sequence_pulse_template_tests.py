import unittest
import copy

import numpy as np

from qctoolkit.pulses.pulse_template import DoubleParameterNameException
from qctoolkit.expressions import Expression
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate, SequenceWaveform
from qctoolkit.pulses.pulse_template_parameter_mapping import MissingMappingException, UnnecessaryMappingException, MissingParameterDeclarationException, MappingTemplate
from qctoolkit.pulses.parameters import ParameterDeclaration, ParameterNotProvidedException, ConstantParameter

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
        self.square = TablePulseTemplate()
        self.square.add_entry('up', 'v', 'hold')
        self.square.add_entry('down', 0, 'hold')
        self.square.add_entry('length', 0)
        self.square.add_measurement_declaration('mw1', 'up', 'down+length')

        self.mapping1 = {
            'up': 'uptime',
            'down': 'uptime + length',
            'v': 'voltage',
            'length': '0.5 * pulse_length'
        }

        self.window_name_mapping = {'mw1' : 'test_window'}

        self.outer_parameters = {'uptime', 'length', 'pulse_length', 'voltage'}

        self.parameters = {}
        self.parameters['uptime'] = ConstantParameter(5)
        self.parameters['length'] = ConstantParameter(10)
        self.parameters['pulse_length'] = ConstantParameter(100)
        self.parameters['voltage'] = ConstantParameter(10)

        self.sequence = SequencePulseTemplate([MappingTemplate(self.square, self.mapping1, measurement_mapping=self.window_name_mapping)], self.outer_parameters)

    def test_missing_mapping(self) -> None:
        mapping = self.mapping1
        mapping.pop('v')

        subtemplates = [(self.square, mapping, {})]
        with self.assertRaises(MissingMappingException):
            SequencePulseTemplate(subtemplates, self.outer_parameters)

    def test_unnecessary_mapping(self) -> None:
        mapping = self.mapping1
        mapping['unnecessary'] = 'voltage'

        subtemplates = [(self.square, mapping, {})]
        with self.assertRaises(UnnecessaryMappingException):
            SequencePulseTemplate(subtemplates, self.outer_parameters)

    def test_identifier(self) -> None:
        identifier = 'some name'
        pulse = SequencePulseTemplate([DummyPulseTemplate()], {}, identifier=identifier)
        self.assertEqual(identifier, pulse.identifier)

    def test_multiple_channels(self) -> None:
        dummy = DummyPulseTemplate(parameter_names={'hugo'}, defined_channels={'A', 'B'})
        subtemplates = [(dummy, {'hugo': 'foo'}, {}), (dummy, {'hugo': '3'}, {})]
        sequence = SequencePulseTemplate(subtemplates, {'foo'})
        self.assertEqual({'A', 'B'}, sequence.defined_channels)

    def test_multiple_channels_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            SequencePulseTemplate(
                [DummyPulseTemplate(defined_channels={'A'}), DummyPulseTemplate(defined_channels={'B'})]
                , set())

        with self.assertRaises(ValueError):
            SequencePulseTemplate(
                [DummyPulseTemplate(defined_channels={'A'}), DummyPulseTemplate(defined_channels={'A', 'B'})]
                , set())


class SequencePulseTemplateSerializationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.serializer = DummySerializer()

        self.table_foo = TablePulseTemplate(identifier='foo')
        self.table_foo.add_entry('hugo', 2)
        self.table_foo.add_entry(ParameterDeclaration('albert', max=9.1), 'voltage')
        self.table_foo.add_measurement_declaration('mw_foo','hugo','albert')

        self.foo_param_mappings = dict(hugo='ilse', albert='albert', voltage='voltage')
        self.foo_meas_mappings = dict(mw_foo='mw_bar')

        self.table = TablePulseTemplate()

    def test_get_serialization_data(self) -> None:
        dummy1 = DummyPulseTemplate()
        dummy2 = DummyPulseTemplate()

        sequence = SequencePulseTemplate([dummy1, dummy2], [])
        serializer = DummySerializer(serialize_callback=lambda x: str(x))

        expected_data = dict(
            type=serializer.get_type_identifier(sequence),
            subtemplates = [str(dummy1), str(dummy2)]
        )
        data = sequence.get_serialization_data(serializer)
        self.assertEqual(expected_data, data)

    def test_deserialize(self) -> None:
        dummy1 = DummyPulseTemplate()
        dummy2 = DummyPulseTemplate()

        serializer = DummySerializer(serialize_callback=lambda x: str(id(x)))

        data = dict(
            subtemplates = [serializer.dictify(dummy1), serializer.dictify(dummy2)],
            identifier='foo'
        )

        template = SequencePulseTemplate.deserialize(serializer,**data)
        self.assertEqual(template.subtemplates, [dummy1, dummy2])


class SequencePulseTemplateSequencingTests(SequencePulseTemplateTest):
    def test_build_sequence(self) -> None:
        sub1 = DummyPulseTemplate(requires_stop=False)
        sub2 = DummyPulseTemplate(requires_stop=True, parameter_names={'foo'})
        parameters = {'foo': DummyNoValueParameter()}

        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        seq = SequencePulseTemplate([(sub1, {}, {}), (sub2, {'foo': 'foo'}, {})], {'foo'})
        seq.build_sequence(sequencer, parameters, {}, {}, {}, block)
        self.assertEqual(2, len(sequencer.sequencing_stacks[block]))

        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        seq = SequencePulseTemplate([(sub2, {'foo': 'foo'}, {}), (sub1, {}, {})], {'foo'})
        seq.build_sequence(sequencer, parameters, {}, {}, {}, block)
        self.assertEqual(2, len(sequencer.sequencing_stacks[block]))

    @unittest.skip("Was this test faulty before? Why should the three last cases return false?")
    def test_requires_stop(self) -> None:
        sub1 = (DummyPulseTemplate(requires_stop=False), {}, {})
        sub2 = (DummyPulseTemplate(requires_stop=True, parameter_names={'foo'}), {'foo': 'foo'}, {})
        parameters = {'foo': DummyNoValueParameter()}

        seq = SequencePulseTemplate([sub1], {})
        self.assertFalse(seq.requires_stop(parameters, {}))

        seq = SequencePulseTemplate([sub2], {'foo'})
        self.assertFalse(seq.requires_stop(parameters, {}))

        seq = SequencePulseTemplate([sub1, sub2], {'foo'})
        self.assertFalse(seq.requires_stop(parameters, {}))

        seq = SequencePulseTemplate([sub2, sub1], {'foo'})
        self.assertFalse(seq.requires_stop(parameters, {}))

    def test_missing_parameter_declaration_exception(self):
        mapping = copy.deepcopy(self.mapping1)
        mapping['up'] = "foo"

        subtemplates = [(self.square, mapping,{})]
        with self.assertRaises(MissingParameterDeclarationException):
            SequencePulseTemplate(subtemplates, self.outer_parameters)

    def test_crash(self) -> None:
        table = TablePulseTemplate(identifier='foo')
        table.add_entry('ta', 'va', interpolation='hold')
        table.add_entry('tb', 'vb', interpolation='linear')
        table.add_entry('tend', 0, interpolation='jump')

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
        sequence = SequencePulseTemplate([(table, first_mapping, {}), (table, second_mapping, {})], external_parameters)

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
        sequence.build_sequence(sequencer, parameters, {}, {}, {'default', 'default'}, block)
        from qctoolkit.pulses.sequencing import Sequencer
        s = Sequencer()
        s.push(sequence, parameters, channel_mapping={'default': 'EXAMPLE_A'})
        s.build()

    def test_missing_parameter_declaration_exception(self) -> None:
        mapping = copy.deepcopy(self.mapping1)
        mapping['up'] = "foo"

        subtemplates = [(self.square, mapping)]
        with self.assertRaises(MissingParameterDeclarationException):
            SequencePulseTemplate(subtemplates, self.outer_parameters)


class SequencePulseTemplateTestProperties(SequencePulseTemplateTest):
    def test_is_interruptable(self):

        self.assertTrue(
            SequencePulseTemplate([DummyPulseTemplate(is_interruptable=True),
                                   DummyPulseTemplate(is_interruptable=True)], []).is_interruptable)
        self.assertTrue(
            SequencePulseTemplate([DummyPulseTemplate(is_interruptable=True),
                                   DummyPulseTemplate(is_interruptable=False)], []).is_interruptable)
        self.assertFalse(
            SequencePulseTemplate([DummyPulseTemplate(is_interruptable=False),
                                   DummyPulseTemplate(is_interruptable=False)], []).is_interruptable)
        
    def test_parameter_declarations(self):
        decl = self.sequence.parameter_declarations
        self.assertEqual(decl, set([ParameterDeclaration(i) for i in self.outer_parameters]))


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

        seq1 = SequencePulseTemplate([a, b], ['foo', 'bar'])
        seq2 = SequencePulseTemplate([c, d], ['snu'])

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
