import unittest
import copy

from qctoolkit.expressions import Expression
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate
from qctoolkit.pulses.pulse_template_parameter_mapping import MissingMappingException, UnnecessaryMappingException, MissingParameterDeclarationException
from qctoolkit.pulses.parameters import ParameterDeclaration, ParameterNotProvidedException, ConstantParameter

from tests.pulses.sequencing_dummies import DummySequencer, DummyInstructionBlock, DummyPulseTemplate, DummyNoValueParameter
from tests.serialization_dummies import DummySerializer

class SequencePulseTemplateTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Setup test data
        self.square = TablePulseTemplate()
        self.square.add_entry('up', 'v', 'hold')
        self.square.add_entry('down', 0, 'hold')
        self.square.add_entry('length', 0)

        self.mapping1 = {
            'up': 'uptime',
            'down': 'uptime + length',
            'v': 'voltage',
            'length': '0.5 * pulse_length'
        }

        self.outer_parameters = ['uptime', 'length', 'pulse_length', 'voltage']

        self.parameters = {}
        self.parameters['uptime'] = ConstantParameter(5)
        self.parameters['length'] = ConstantParameter(10)
        self.parameters['pulse_length'] = ConstantParameter(100)
        self.parameters['voltage'] = ConstantParameter(10)

        self.sequence = SequencePulseTemplate([(self.square, self.mapping1)], self.outer_parameters)

    def test_missing_mapping(self):
        mapping = self.mapping1
        mapping.pop('v')

        subtemplates = [(self.square, mapping)]
        with self.assertRaises(MissingMappingException):
            SequencePulseTemplate(subtemplates, self.outer_parameters)

    def test_unnecessary_mapping(self):
        mapping = self.mapping1
        mapping['unnecessary'] = 'voltage'

        subtemplates = [(self.square, mapping)]
        with self.assertRaises(UnnecessaryMappingException):
            SequencePulseTemplate(subtemplates, self.outer_parameters)

    def test_identifier(self):
        identifier = 'some name'
        pulse = SequencePulseTemplate([], [], identifier=identifier)
        self.assertEqual(identifier, pulse.identifier)


class SequencePulseTemplateSerializationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.serializer = DummySerializer()

        self.table_foo = TablePulseTemplate(identifier='foo')
        self.table_foo.add_entry('hugo', 2)
        self.table_foo.add_entry(ParameterDeclaration('albert', max=9.1), 'voltage')

        self.table = TablePulseTemplate(measurement=True)
        self.foo_mappings = dict(hugo='ilse', albert='albert', voltage='voltage')

    def test_get_serialization_data(self) -> None:
        serializer = DummySerializer(serialize_callback=lambda x: str(x))
        foo_mappings = {k: Expression(v) for k, v in self.foo_mappings.items()}
        sequence = SequencePulseTemplate([(self.table_foo, self.foo_mappings), (self.table, {})],
                                         ['ilse', 'albert', 'voltage'],
                                         identifier='foo')
        expected_data = dict(
            type=serializer.get_type_identifier(sequence),
            external_parameters=['albert', 'ilse', 'voltage'],
            is_interruptable=True,
            subtemplates = [
                dict(template=str(self.table_foo), mappings={k: str(v) for k, v in foo_mappings.items()}),
                dict(template=str(self.table), mappings=dict())
            ]
        )
        data = sequence.get_serialization_data(serializer)
        self.assertEqual(expected_data, data)

    def test_deserialize(self) -> None:
        foo_mappings = {k: Expression(v) for k, v in self.foo_mappings.items()}
        data = dict(
            external_parameters={'ilse', 'albert', 'voltage'},
            is_interruptable=True,
            subtemplates = [
                dict(template=str(id(self.table_foo)), mappings={k: str(id(v)) for k, v in foo_mappings.items()}),
                dict(template=str(id(self.table)), mappings=dict())
            ],
            identifier='foo'
        )

        # prepare dependencies for deserialization
        self.serializer.subelements[str(id(self.table_foo))] = self.table_foo
        self.serializer.subelements[str(id(self.table))] = self.table
        for v in foo_mappings.values():
            self.serializer.subelements[str(id(v))] = v

        # deserialize
        sequence = SequencePulseTemplate.deserialize(self.serializer, **data)

        # compare!
        self.assertEqual(data['external_parameters'], sequence.parameter_names)
        self.assertEqual({ParameterDeclaration('ilse'), ParameterDeclaration('albert'), ParameterDeclaration('voltage')},
                         sequence.parameter_declarations)
        self.assertIs(self.table_foo, sequence.subtemplates[0][0])
        self.assertIs(self.table, sequence.subtemplates[1][0])
        #self.assertEqual(self.foo_mappings, {k: m.string for k,m in sequence.subtemplates[0][1].items()})
        self.assertEqual(foo_mappings, sequence.subtemplates[0][1])
        self.assertEqual(dict(), sequence.subtemplates[1][1])
        self.assertEqual(data['identifier'], sequence.identifier)


class SequencePulseTemplateSequencingTests(SequencePulseTemplateTest):

    def test_missing_parameter(self):
        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        parameters = copy.deepcopy(self.parameters)
        parameters.pop('uptime')
        with self.assertRaises(ParameterNotProvidedException):
            self.sequence.build_sequence(sequencer, parameters, {}, block)

    def test_build_sequence(self) -> None:
        sub1 = DummyPulseTemplate(requires_stop=False)
        sub2 = DummyPulseTemplate(requires_stop=True, parameter_names={'foo'})
        parameters = {'foo': DummyNoValueParameter()}

        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        seq = SequencePulseTemplate([(sub1, {}), (sub2, {'foo': 'foo'})], {'foo'})
        seq.build_sequence(sequencer, parameters, {}, block)
        self.assertEqual(2, len(sequencer.sequencing_stacks[block]))

        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        seq = SequencePulseTemplate([(sub2, {'foo': 'foo'}), (sub1, {})], {'foo'})
        seq.build_sequence(sequencer, parameters, {}, block)
        self.assertEqual(2, len(sequencer.sequencing_stacks[block]))

    def test_requires_stop(self) -> None:
        sub1 = (DummyPulseTemplate(requires_stop=False), {})
        sub2 = (DummyPulseTemplate(requires_stop=True, parameter_names={'foo'}), {'foo': 'foo'})
        parameters = {'foo': DummyNoValueParameter()}

        seq = SequencePulseTemplate([],[])
        self.assertFalse(seq.requires_stop(parameters, {}))

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

        subtemplates = [(self.square, mapping)]
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
        sequence = SequencePulseTemplate([(table, first_mapping), (table, second_mapping)], external_parameters)

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
        sequence.build_sequence(sequencer, parameters, {}, block)
        from qctoolkit.pulses.sequencing import Sequencer
        s = Sequencer()
        s.push(sequence, parameters)
        s.build()

    def test_missing_parameter_declaration_exception(self) -> None:
        mapping = copy.deepcopy(self.mapping1)
        mapping['up'] = "foo"

        subtemplates = [(self.square, mapping)]
        with self.assertRaises(MissingParameterDeclarationException):
            SequencePulseTemplate(subtemplates, self.outer_parameters)


class SequencePulseTemplateTestProperties(SequencePulseTemplateTest):
    def test_is_interruptable(self):
        self.assertTrue(self.sequence.is_interruptable)
        self.sequence.is_interruptable = False
        self.assertFalse(self.sequence.is_interruptable)
        
    def test_parameter_declarations(self):
        decl = self.sequence.parameter_declarations
        self.assertEqual(decl, set([ParameterDeclaration(i) for i in self.outer_parameters]))

if __name__ == "__main__":
    unittest.main(verbosity=2)
