import unittest

import numpy as np

from qctoolkit.pulses.repetition_pulse_template import RepetitionPulseTemplate,ParameterNotIntegerException, RepetitionWaveform
from qctoolkit.pulses.parameters import ParameterNotProvidedException, ParameterConstraintViolation, ConstantParameter, \
    ParameterConstraint
from qctoolkit.pulses.instructions import REPJInstruction, InstructionPointer

from tests.pulses.sequencing_dummies import DummyPulseTemplate, DummySequencer, DummyInstructionBlock, DummyParameter,\
    DummyCondition, DummyWaveform
from tests.serialization_dummies import DummySerializer


class RepetitionWaveformTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_unsafe_sample(self):
        body_wf = DummyWaveform(duration=7)

        rwf = RepetitionWaveform(body=body_wf, repetition_count=10)

        sample_times = np.arange(80) * 70./80.
        np.testing.assert_equal(rwf.unsafe_sample(channel='A', sample_times=sample_times), sample_times)

        output_expected = np.empty_like(sample_times)
        output_received = rwf.unsafe_sample(channel='A', sample_times=sample_times, output_array=output_expected)
        self.assertIs(output_expected, output_received)
        np.testing.assert_equal(output_received, sample_times)

    def test_get_measurement_windows(self):
        body_wf = DummyWaveform(duration=7, measurement_windows=[('M', .1, .2), ('N', .5, .7)])

        rwf = RepetitionWaveform(body=body_wf, repetition_count=3)

        expected_windows = [('M', .1, .2), ('N', .5, .7),
                            ('M', 7.1, .2), ('N', 7.5, .7),
                            ('M', 14.1, .2), ('N', 14.5, .7)]
        received_windows = list(rwf.get_measurement_windows())
        self.assertEqual(received_windows, expected_windows)


class RepetitionPulseTemplateTest(unittest.TestCase):

    def test_init(self) -> None:
        body = DummyPulseTemplate()
        repetition_count = 3
        t = RepetitionPulseTemplate(body, repetition_count)
        self.assertEqual(repetition_count, t.repetition_count)
        self.assertEqual(body, t.body)

        repetition_count = 'foo'
        t = RepetitionPulseTemplate(body, repetition_count)
        self.assertEqual(repetition_count, t.repetition_count)
        self.assertEqual(body, t.body)

    def test_parameter_names_and_declarations(self) -> None:
        body = DummyPulseTemplate()
        t = RepetitionPulseTemplate(body, 5)
        self.assertEqual(body.parameter_names, t.parameter_names)

        body.parameter_names_ = {'foo', 't', 'bar'}
        self.assertEqual(body.parameter_names, t.parameter_names)

    @unittest.skip('is interruptable not implemented for loops')
    def test_is_interruptable(self) -> None:
        body = DummyPulseTemplate(is_interruptable=False)
        t = RepetitionPulseTemplate(body, 6)
        self.assertFalse(t.is_interruptable)

        body.is_interruptable_ = True
        self.assertTrue(t.is_interruptable)

    def test_str(self) -> None:
        body = DummyPulseTemplate()
        t = RepetitionPulseTemplate(body, 9)
        self.assertIsInstance(str(t), str)
        t = RepetitionPulseTemplate(body, 'foo')
        self.assertIsInstance(str(t), str)


class RepetitionPulseTemplateSequencingTests(unittest.TestCase):

    def test_requires_stop_constant(self) -> None:
        body = DummyPulseTemplate(requires_stop=False)
        t = RepetitionPulseTemplate(body, 2)
        self.assertFalse(t.requires_stop({}, {}))
        body.requires_stop_ = True
        self.assertFalse(t.requires_stop({}, {}))

    def test_requires_stop_declaration(self) -> None:
        body = DummyPulseTemplate(requires_stop=False)
        t = RepetitionPulseTemplate(body, 'foo')

        parameter = DummyParameter()
        parameters = dict(foo=parameter)
        condition = DummyCondition()
        conditions = dict(foo=condition)

        for body_requires_stop in [True, False]:
            for condition_requires_stop in [True, False]:
                for parameter_requires_stop in [True, False]:
                    body.requires_stop_ = body_requires_stop
                    condition.requires_stop_ = condition_requires_stop
                    parameter.requires_stop_ = parameter_requires_stop
                    self.assertEqual(parameter_requires_stop, t.requires_stop(parameters, conditions))

    def setUp(self) -> None:
        self.body = DummyPulseTemplate()
        self.repetitions = 'foo'
        self.template = RepetitionPulseTemplate(self.body, self.repetitions, parameter_constraints=['foo<9'])
        self.sequencer = DummySequencer()
        self.block = DummyInstructionBlock()

    def test_build_sequence_constant(self) -> None:
        repetitions = 3
        t = RepetitionPulseTemplate(self.body, repetitions)
        parameters = {}
        measurement_mapping = {'my': 'thy'}
        conditions = dict(foo=DummyCondition(requires_stop=True))
        channel_mapping = {}
        t.build_sequence(self.sequencer, parameters, conditions, measurement_mapping, channel_mapping, self.block)

        self.assertTrue(self.block.embedded_blocks)
        body_block = self.block.embedded_blocks[0]
        self.assertEqual({body_block}, set(self.sequencer.sequencing_stacks.keys()))
        self.assertEqual([(self.body, parameters, conditions, measurement_mapping, channel_mapping)], self.sequencer.sequencing_stacks[body_block])
        self.assertEqual([REPJInstruction(repetitions, InstructionPointer(body_block, 0))], self.block.instructions)

    def test_build_sequence_declaration_success(self) -> None:
        parameters = dict(foo=ConstantParameter(3))
        conditions = dict(foo=DummyCondition(requires_stop=True))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')
        self.template.build_sequence(self.sequencer, parameters, conditions, measurement_mapping, channel_mapping, self.block)

        self.assertTrue(self.block.embedded_blocks)
        body_block = self.block.embedded_blocks[0]
        self.assertEqual({body_block}, set(self.sequencer.sequencing_stacks.keys()))
        self.assertEqual([(self.body, parameters, conditions, measurement_mapping, channel_mapping)],
                         self.sequencer.sequencing_stacks[body_block])
        self.assertEqual([REPJInstruction(3, InstructionPointer(body_block, 0))], self.block.instructions)

    def test_build_sequence_declaration_exceeds_bounds(self) -> None:
        parameters = dict(foo=ConstantParameter(9))
        conditions = dict(foo=DummyCondition(requires_stop=True))
        with self.assertRaises(ParameterConstraintViolation):
            self.template.build_sequence(self.sequencer, parameters, conditions, {}, {}, self.block)
        self.assertFalse(self.sequencer.sequencing_stacks)

    def test_build_sequence_declaration_parameter_missing(self) -> None:
        parameters = {}
        conditions = dict(foo=DummyCondition(requires_stop=True))
        with self.assertRaises(ParameterNotProvidedException):
            self.template.build_sequence(self.sequencer, parameters, conditions, {}, {}, self.block)
        self.assertFalse(self.sequencer.sequencing_stacks)

    def test_build_sequence_declaration_parameter_value_not_whole(self) -> None:
        parameters = dict(foo=ConstantParameter(3.3))
        conditions = dict(foo=DummyCondition(requires_stop=True))
        with self.assertRaises(ParameterNotIntegerException):
            self.template.build_sequence(self.sequencer, parameters, conditions, {}, {}, self.block)
        self.assertFalse(self.sequencer.sequencing_stacks)


class RepetitionPulseTemplateSerializationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.serializer = DummySerializer(deserialize_callback=lambda x: x['name'])
        self.body = DummyPulseTemplate()

    def test_get_serialization_data_constant(self) -> None:
        repetition_count = 3
        template = RepetitionPulseTemplate(self.body, repetition_count)
        expected_data = dict(
            body=str(id(self.body)),
            repetition_count=repetition_count,
            parameter_constraints=[]
        )
        data = template.get_serialization_data(self.serializer)
        self.assertEqual(expected_data, data)

    def test_get_serialization_data_declaration(self) -> None:
        template = RepetitionPulseTemplate(self.body, 'foo', parameter_constraints=['foo<3'])
        expected_data = dict(
            body=str(id(self.body)),
            repetition_count='foo',
            parameter_constraints=[ParameterConstraint('foo<3')]
        )
        data = template.get_serialization_data(self.serializer)
        self.assertEqual(expected_data, data)

    def test_deserialize_constant(self) -> None:
        repetition_count = 3
        data = dict(
            repetition_count=repetition_count,
            body=dict(name=str(id(self.body))),
            identifier='foo',
            parameter_constraints=['bar<3']
        )
        # prepare dependencies for deserialization
        self.serializer.subelements[str(id(self.body))] = self.body
        # deserialize
        template = RepetitionPulseTemplate.deserialize(self.serializer, **data)
        # compare!
        self.assertEqual(self.body, template.body)
        self.assertEqual(repetition_count, template.repetition_count)
        self.assertEqual([str(c) for c in template.parameter_constraints], ['bar < 3'])

    def test_deserialize_declaration(self) -> None:
        data = dict(
            repetition_count='foo',
            body=dict(name=str(id(self.body))),
            identifier='foo',
            parameter_constraints=['foo<3']
        )
        # prepare dependencies for deserialization
        self.serializer.subelements[str(id(self.body))] = self.body

        # deserialize
        template = RepetitionPulseTemplate.deserialize(self.serializer, **data)

        # compare!
        self.assertEqual(self.body, template.body)
        self.assertEqual('foo', template.repetition_count)
        self.assertEqual(template.parameter_constraints, [ParameterConstraint('foo < 3')])


class ParameterNotIntegerExceptionTests(unittest.TestCase):

    def test(self) -> None:
        exception = ParameterNotIntegerException('foo', 3)
        self.assertIsInstance(str(exception), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)