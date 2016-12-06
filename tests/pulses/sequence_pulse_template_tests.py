import unittest
import copy

from qctoolkit.pulses.pulse_template import DoubleParameterNameException
from qctoolkit.expressions import Expression
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate
from qctoolkit.pulses.pulse_template_parameter_mapping import MissingMappingException, UnnecessaryMappingException, MissingParameterDeclarationException, MappingTemplate
from qctoolkit.pulses.parameters import ParameterDeclaration, ParameterNotProvidedException, ConstantParameter

from tests.pulses.sequencing_dummies import DummySequencer, DummyInstructionBlock, DummyPulseTemplate, DummyNoValueParameter
from tests.serialization_dummies import DummySerializer

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
    @unittest.skip("The test for missing parameters is performed on the lowest level.")
    def test_missing_parameter(self):
        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        parameters = copy.deepcopy(self.parameters)
        parameters.pop('uptime')
        with self.assertRaises(ParameterNotProvidedException):
            self.sequence.build_sequence(sequencer, parameters, {}, {}, {}, block)

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
