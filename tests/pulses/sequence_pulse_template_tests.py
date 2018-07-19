import unittest

import numpy as np

from qctoolkit.utils.types import time_from_float
from qctoolkit.expressions import Expression, ExpressionScalar
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate, SequenceWaveform
from qctoolkit.pulses.pulse_template_parameter_mapping import MappingPulseTemplate
from qctoolkit.pulses.parameters import ConstantParameter, ParameterConstraint, ParameterConstraintViolation
from qctoolkit._program.instructions import MEASInstruction

from tests.pulses.sequencing_dummies import DummySequencer, DummyInstructionBlock, DummyPulseTemplate,\
    DummyNoValueParameter, DummyWaveform
from tests.serialization_dummies import DummySerializer
from tests.serialization_tests import SerializableTests



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
                                                                   measurement_mapping=self.window_name_mapping))

    def test_external_parameters_warning(self):
        dummy = DummyPulseTemplate()
        with self.assertWarnsRegex(DeprecationWarning, "external_parameters",
                                   msg="SequencePT did not issue a warning for argument external_parameters"):
            SequencePulseTemplate(dummy, external_parameters={'a'})

    def test_duration(self):
        pt = SequencePulseTemplate(DummyPulseTemplate(duration='a'),
                                   DummyPulseTemplate(duration='a'),
                                   DummyPulseTemplate(duration='b'))
        self.assertEqual(pt.duration, Expression('a+a+b'))

    def test_parameter_names_param_only_in_constraint(self) -> None:
        pt = SequencePulseTemplate(DummyPulseTemplate(parameter_names={'a'}), DummyPulseTemplate(parameter_names={'b'}), parameter_constraints=['a==b', 'a<c'])
        self.assertEqual(pt.parameter_names, {'a','b','c'})

    def test_build_waveform(self):
        wfs = [DummyWaveform(), DummyWaveform()]
        pts = [DummyPulseTemplate(waveform=wf) for wf in wfs]

        spt = SequencePulseTemplate(*pts, parameter_constraints=['a < 3'])
        with self.assertRaises(ParameterConstraintViolation):
            spt.build_waveform(dict(a=4), dict())

        parameters = dict(a=2)
        channel_mapping = dict()
        wf = spt.build_waveform(parameters, channel_mapping=channel_mapping)

        for wfi, pt in zip(wfs, pts):
            self.assertEqual(pt.build_waveform_calls, [(parameters, dict())])
            self.assertIs(pt.build_waveform_calls[0][0], parameters)

        self.assertIsInstance(wf, SequenceWaveform)
        for wfa, wfb in zip(wf.compare_key, wfs):
            self.assertIs(wfa, wfb)

    def test_identifier(self) -> None:
        identifier = 'some name'
        pulse = SequencePulseTemplate(DummyPulseTemplate(), identifier=identifier)
        self.assertEqual(identifier, pulse.identifier)

    def test_multiple_channels(self) -> None:
        dummy = DummyPulseTemplate(parameter_names={'hugo'}, defined_channels={'A', 'B'})
        subtemplates = [(dummy, {'hugo': 'foo'}, {}), (dummy, {'hugo': '3'}, {})]
        sequence = SequencePulseTemplate(*subtemplates)
        self.assertEqual({'A', 'B'}, sequence.defined_channels)
        self.assertEqual({'foo'}, sequence.parameter_names)

    def test_multiple_channels_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            SequencePulseTemplate(DummyPulseTemplate(defined_channels={'A'}),
                                  DummyPulseTemplate(defined_channels={'B'}))

        with self.assertRaises(ValueError):
            SequencePulseTemplate(
                DummyPulseTemplate(defined_channels={'A'}), DummyPulseTemplate(defined_channels={'A', 'B'})
            )

    def test_integral(self) -> None:
        dummy1 = DummyPulseTemplate(defined_channels={'A', 'B'},
                                    integrals={'A': ExpressionScalar('k+2*b'), 'B': ExpressionScalar('3')})
        dummy2 = DummyPulseTemplate(defined_channels={'A', 'B'},
                                    integrals={'A': ExpressionScalar('7*(b-f)'), 'B': ExpressionScalar('0.24*f-3.0')})
        pulse = SequencePulseTemplate(dummy1, dummy2)

        self.assertEqual({'A': ExpressionScalar('k+2*b+7*(b-f)'), 'B': ExpressionScalar('0.24*f')}, pulse.integral)


class SequencePulseTemplateSerializationTests(SerializableTests, unittest.TestCase):

    @property
    def class_to_test(self):
        return SequencePulseTemplate

    def make_kwargs(self):
        return {
            'subtemplates': [DummyPulseTemplate(), DummyPulseTemplate()],
            'parameter_constraints': [str(ParameterConstraint('a<b'))],
            'measurements': [('m', 0, 1)]
        }

    def make_instance(self, identifier=None, registry=None):
        kwargs = self.make_kwargs()
        subtemplates = kwargs['subtemplates']
        del kwargs['subtemplates']
        return self.class_to_test(identifier=identifier, *subtemplates, **kwargs, registry=registry)

    def assert_equal_instance_except_id(self, lhs: SequencePulseTemplate, rhs: SequencePulseTemplate):
        self.assertIsInstance(lhs, SequencePulseTemplate)
        self.assertIsInstance(rhs, SequencePulseTemplate)
        self.assertEqual(lhs.subtemplates, rhs.subtemplates)
        self.assertEqual(lhs.parameter_constraints, rhs.parameter_constraints)
        self.assertEqual(lhs.measurement_declarations, rhs.measurement_declarations)


class SequencePulseTemplateOldSerializationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.table_foo = TablePulseTemplate({'default': [('hugo', 2),
                                                         ('albert', 'voltage')]},
                                            parameter_constraints=['albert<9.1'],
                                            measurements=[('mw_foo','hugo','albert')],
                                            identifier='foo',
                                            registry=dict())

        self.foo_param_mappings = dict(hugo='ilse', albert='albert', voltage='voltage')
        self.foo_meas_mappings = dict(mw_foo='mw_bar')

    def test_get_serialization_data_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="SequencePT does not issue warning for old serialization routines."):
            dummy1 = DummyPulseTemplate()
            dummy2 = DummyPulseTemplate()

            sequence = SequencePulseTemplate(dummy1, dummy2, parameter_constraints=['a<b'], measurements=[('m', 0, 1)],
                                             registry=dict())
            serializer = DummySerializer(serialize_callback=lambda x: str(x))

            expected_data = dict(
                subtemplates=[str(dummy1), str(dummy2)],
                parameter_constraints=['a < b'],
                measurements=[('m', 0, 1)]
            )
            data = sequence.get_serialization_data(serializer)
            self.assertEqual(expected_data, data)

    def test_deserialize_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="SequencePT does not issue warning for old serialization routines."):
            dummy1 = DummyPulseTemplate()
            dummy2 = DummyPulseTemplate()

            serializer = DummySerializer(serialize_callback=lambda x: str(id(x)))

            data = dict(
                subtemplates=[serializer.dictify(dummy1), serializer.dictify(dummy2)],
                identifier='foo',
                parameter_constraints=['a < b'],
                measurements=[('m', 0, 1)]
            )

            template = SequencePulseTemplate.deserialize(serializer, **data)
            self.assertEqual(template.subtemplates, [dummy1, dummy2])
            self.assertEqual(template.parameter_constraints, [ParameterConstraint('a<b')])
            self.assertEqual(template.measurement_declarations, [('m', 0, 1)])


class SequencePulseTemplateSequencingTests(SequencePulseTemplateTest):
    def test_build_sequence(self) -> None:
        sub1 = DummyPulseTemplate(requires_stop=False)
        sub2 = DummyPulseTemplate(requires_stop=True, parameter_names={'foo'})
        parameters = {'foo': DummyNoValueParameter()}

        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        seq = SequencePulseTemplate(sub1, (sub2, {'foo': 'foo'}), measurements=[('a', 0, 1)])
        seq.build_sequence(sequencer, parameters,
                           conditions=dict(),
                           channel_mapping={'default': 'a'},
                           measurement_mapping={'a': 'b'},
                           instruction_block=block)
        self.assertEqual(2, len(sequencer.sequencing_stacks[block]))

        self.assertEqual(block.instructions[0], MEASInstruction([('b', 0, 1)]))

        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        seq = SequencePulseTemplate((sub2, {'foo': 'foo'}), sub1)
        seq.build_sequence(sequencer, parameters, {}, {}, {}, block)
        self.assertEqual(2, len(sequencer.sequencing_stacks[block]))

    @unittest.skip("Was this test faulty before? Why should the three last cases return false?")
    def test_requires_stop(self) -> None:
        sub1 = (DummyPulseTemplate(requires_stop=False), {}, {})
        sub2 = (DummyPulseTemplate(requires_stop=True, parameter_names={'foo'}), {'foo': 'foo'}, {})
        parameters = {'foo': DummyNoValueParameter()}

        seq = SequencePulseTemplate(sub1)
        self.assertFalse(seq.requires_stop(parameters, {}))

        seq = SequencePulseTemplate(sub2)
        self.assertFalse(seq.requires_stop(parameters, {}))

        seq = SequencePulseTemplate(sub1, sub2)
        self.assertFalse(seq.requires_stop(parameters, {}))

        seq = SequencePulseTemplate(sub2, sub1)
        self.assertFalse(seq.requires_stop(parameters, {}))

    def test_crash(self) -> None:
        table = TablePulseTemplate({'default': [('ta', 'va', 'hold'),
                                                ('tb', 'vb', 'linear'),
                                                ('tend', 0, 'jump')]}, identifier='foo')

        expected_parameters = {'ta', 'tb', 'tc', 'td', 'va', 'vb', 'tend'}
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
        sequence = SequencePulseTemplate((table, first_mapping, {}), (table, second_mapping, {}))
        self.assertEqual(expected_parameters, sequence.parameter_names)

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

    def test_measurement_names(self):
        d1 = DummyPulseTemplate(measurement_names={'a'})
        d2 = DummyPulseTemplate(measurement_names={'b'})

        spt = SequencePulseTemplate(d1, d2, measurements=[('c', 0, 1)])

        self.assertEqual(spt.measurement_names, {'a', 'b', 'c'})


class PulseTemplateConcatenationTest(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def test_concatenation_pulse_template(self):
        a = DummyPulseTemplate(parameter_names={'foo'}, defined_channels={'A'})
        b = DummyPulseTemplate(parameter_names={'bar'}, defined_channels={'A'})
        c = DummyPulseTemplate(parameter_names={'snu'}, defined_channels={'A'})
        d = (c, {'snu': 'bar'})

        seq = a @ a
        self.assertTrue(len(seq.subtemplates) == 2)
        for st in seq.subtemplates:
            self.assertEqual(st, a)

        seq = a @ b
        self.assertTrue(len(seq.subtemplates)==2)
        for st, expected in zip(seq.subtemplates,[a, b]):
            self.assertTrue(st, expected)

        seq = a @ b @ c
        self.assertTrue(len(seq.subtemplates) == 3)
        for st, expected in zip(seq.subtemplates, [a, b, c]):
            self.assertTrue(st, expected)

        seq = a @ d
        self.assertTrue(len(seq.subtemplates) == 2)
        self.assertIs(seq.subtemplates[0], a)
        self.assertIsInstance(seq.subtemplates[1], MappingPulseTemplate)
        self.assertIs(seq.subtemplates[1].template, c)

        seq = d @ a
        self.assertTrue(len(seq.subtemplates) == 2)
        self.assertIs(seq.subtemplates[1], a)
        self.assertIsInstance(seq.subtemplates[0], MappingPulseTemplate)
        self.assertIs(seq.subtemplates[0].template, c)

    def test_concatenation_sequence_table_pulse(self):
        a = DummyPulseTemplate(parameter_names={'foo'}, defined_channels={'A'})
        b = DummyPulseTemplate(parameter_names={'bar'}, defined_channels={'A'})
        c = DummyPulseTemplate(parameter_names={'snu'}, defined_channels={'A'})
        d = DummyPulseTemplate(parameter_names={'snu'}, defined_channels={'A'})

        seq1 = SequencePulseTemplate(a, b)
        self.assertEqual({'foo', 'bar'}, seq1.parameter_names)
        seq2 = SequencePulseTemplate(c, d)
        self.assertEqual({'snu'}, seq2.parameter_names)

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

if __name__ == "__main__":
    unittest.main(verbosity=2)
