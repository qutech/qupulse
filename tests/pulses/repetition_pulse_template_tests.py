import unittest

import numpy as np

from qctoolkit.expressions import Expression
from qctoolkit.pulses.repetition_pulse_template import RepetitionPulseTemplate,ParameterNotIntegerException, RepetitionWaveform
from qctoolkit.pulses.parameters import ParameterNotProvidedException, ParameterConstraintViolation, ConstantParameter, \
    ParameterConstraint
from qctoolkit.pulses.instructions import REPJInstruction, InstructionPointer
from qctoolkit.utils.types import time_from_float

from tests.pulses.sequencing_dummies import DummyPulseTemplate, DummySequencer, DummyInstructionBlock, DummyParameter,\
    DummyCondition, DummyWaveform
from tests.serialization_dummies import DummySerializer
from tests.serialization_tests import SerializableTests


class RepetitionWaveformTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init(self):
        body_wf = DummyWaveform()

        with self.assertRaises(ValueError):
            RepetitionWaveform(body_wf, -1)

        with self.assertRaises(ValueError):
            RepetitionWaveform(body_wf, 1.1)

        wf = RepetitionWaveform(body_wf, 3)
        self.assertIs(wf._body, body_wf)
        self.assertEqual(wf._repetition_count, 3)

    def test_duration(self):
        wf = RepetitionWaveform(DummyWaveform(duration=2.2), 3)
        self.assertEqual(wf.duration, time_from_float(2.2)*3)

    def test_defined_channels(self):
        body_wf = DummyWaveform(defined_channels={'a'})
        self.assertIs(RepetitionWaveform(body_wf, 2).defined_channels, body_wf.defined_channels)

    def test_compare_key(self):
        body_wf = DummyWaveform(defined_channels={'a'})
        wf = RepetitionWaveform(body_wf, 2)
        self.assertEqual(wf.compare_key, (body_wf.compare_key, 2))

    def test_unsafe_get_subset_for_channels(self):
        body_wf = DummyWaveform(defined_channels={'a', 'b'})

        chs = {'a'}

        subset = RepetitionWaveform(body_wf, 3).get_subset_for_channels(chs)
        self.assertIsInstance(subset, RepetitionWaveform)
        self.assertIsInstance(subset._body, DummyWaveform)
        self.assertIs(subset._body.defined_channels, chs)
        self.assertEqual(subset._repetition_count, 3)

    def test_unsafe_sample(self):
        body_wf = DummyWaveform(duration=7)

        rwf = RepetitionWaveform(body=body_wf, repetition_count=10)

        sample_times = np.arange(80) * 70./80.
        inner_sample_times = (sample_times.reshape((10, -1)) - (7 * np.arange(10))[:, np.newaxis]).ravel()

        result = rwf.unsafe_sample(channel='A', sample_times=sample_times)
        np.testing.assert_equal(result, inner_sample_times)

        output_expected = np.empty_like(sample_times)
        output_received = rwf.unsafe_sample(channel='A', sample_times=sample_times, output_array=output_expected)
        self.assertIs(output_expected, output_received)
        np.testing.assert_equal(output_received, inner_sample_times)


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

        with self.assertRaises(ValueError):
            RepetitionPulseTemplate(body, Expression(-1))

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

    def test_measurement_names(self):
        measurement_names = {'M'}
        body = DummyPulseTemplate(measurement_names=measurement_names)
        t = RepetitionPulseTemplate(body, 9)

        self.assertEqual(measurement_names, t.measurement_names)

        t = RepetitionPulseTemplate(body, 9, measurements=[('N', 1, 2)])
        self.assertEqual({'M', 'N'}, t.measurement_names)

    def test_duration(self):
        body = DummyPulseTemplate(duration='foo')
        t = RepetitionPulseTemplate(body, 'bar')

        self.assertEqual(t.duration, Expression('foo*bar'))

    def test_integral(self) -> None:
        dummy = DummyPulseTemplate(integrals=['foo+2', 'k*3+x**2'])
        template = RepetitionPulseTemplate(dummy, 7)
        self.assertEqual([Expression('7*(foo+2)'), Expression('7*(k*3+x**2)')], template.integral)

        template = RepetitionPulseTemplate(dummy, '2+m')
        self.assertEqual([Expression('(2+m)*(foo+2)'), Expression('(2+m)*(k*3+x**2)')], template.integral)

        template = RepetitionPulseTemplate(dummy, Expression('2+m'))
        self.assertEqual([Expression('(2+m)*(foo+2)'), Expression('(2+m)*(k*3+x**2)')], template.integral)


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

    def test_parameter_not_provided(self):
        parameters = dict(foo=ConstantParameter(4))
        conditions = dict(foo=DummyCondition(requires_stop=True))
        measurement_mapping = dict(moth='fire')
        channel_mapping = dict(asd='f')

        template = RepetitionPulseTemplate(self.body, 'foo*bar', parameter_constraints=['foo<9'])

        with self.assertRaises(ParameterNotProvidedException):
            template.build_sequence(self.sequencer, parameters, conditions, measurement_mapping, channel_mapping,
                                     self.block)

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

    def test_parameter_names_param_only_in_constraint(self) -> None:
        pt = RepetitionPulseTemplate(DummyPulseTemplate(parameter_names={'a'}), 'n', parameter_constraints=['a<c'])
        self.assertEqual(pt.parameter_names, {'a','c', 'n'})


class RepetitionPulseTemplateSerializationTests(SerializableTests, unittest.TestCase):

    @property
    def class_to_test(self):
        return RepetitionPulseTemplate

    def make_kwargs(self):
        return {
            'body': DummyPulseTemplate(),
            'repetition_count': 3,
            'parameter_constraints': [str(ParameterConstraint('a<b'))],
            'measurements': [('m', 0, 1)]
        }

    def assert_equal_instance(self, lhs: RepetitionPulseTemplate, rhs: RepetitionPulseTemplate):
        self.assertIsInstance(lhs, RepetitionPulseTemplate)
        self.assertIsInstance(rhs, RepetitionPulseTemplate)
        self.assertEqual(lhs.body, rhs.body)
        self.assertEqual(lhs.parameter_constraints, rhs.parameter_constraints)
        self.assertEqual(lhs.measurement_declarations, rhs.measurement_declarations)


class RepetitionPulseTemplateOldSerializationTests(unittest.TestCase):

    def test_get_serialization_data_minimal_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="RepetitionPT does not issue warning for old serialization routines."):
            serializer = DummySerializer(deserialize_callback=lambda x: x['name'])
            body = DummyPulseTemplate()
            repetition_count = 3
            template = RepetitionPulseTemplate(body, repetition_count)
            expected_data = dict(
                body=str(id(body)),
                repetition_count=repetition_count,
            )
            data = template.get_serialization_data(serializer)
            self.assertEqual(expected_data, data)

    def test_get_serialization_data_all_features_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="RepetitionPT does not issue warning for old serialization routines."):
            serializer = DummySerializer(deserialize_callback=lambda x: x['name'])
            body = DummyPulseTemplate()
            repetition_count = 'foo'
            measurements = [('a', 0, 1), ('b', 1, 1)]
            parameter_constraints = ['foo < 3']
            template = RepetitionPulseTemplate(body, repetition_count,
                                               measurements=measurements,
                                               parameter_constraints=parameter_constraints)
            expected_data = dict(
                body=str(id(body)),
                repetition_count=repetition_count,
                measurements=measurements,
                parameter_constraints=parameter_constraints
            )
            data = template.get_serialization_data(serializer)
            self.assertEqual(expected_data, data)

    def test_deserialize_minimal_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="RepetitionPT does not issue warning for old serialization routines."):
            serializer = DummySerializer(deserialize_callback=lambda x: x['name'])
            body = DummyPulseTemplate()
            repetition_count = 3
            data = dict(
                repetition_count=repetition_count,
                body=dict(name=str(id(body))),
                identifier='foo'
            )
            # prepare dependencies for deserialization
            serializer.subelements[str(id(body))] = body
            # deserialize
            template = RepetitionPulseTemplate.deserialize(serializer, **data)
            # compare!
            self.assertIs(body, template.body)
            self.assertEqual(repetition_count, template.repetition_count)
            #self.assertEqual([str(c) for c in template.parameter_constraints], ['bar < 3'])

    def test_deserialize_all_features_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="RepetitionPT does not issue warning for old serialization routines."):
            serializer = DummySerializer(deserialize_callback=lambda x: x['name'])
            body = DummyPulseTemplate()
            data = dict(
                repetition_count='foo',
                body=dict(name=str(id(body))),
                identifier='foo',
                parameter_constraints=['foo < 3'],
                measurements=[('a', 0, 1), ('b', 1, 1)]
            )
            # prepare dependencies for deserialization
            serializer.subelements[str(id(body))] = body

            # deserialize
            template = RepetitionPulseTemplate.deserialize(serializer, **data)

            # compare!
            self.assertIs(body, template.body)
            self.assertEqual('foo', template.repetition_count)
            self.assertEqual(template.parameter_constraints, [ParameterConstraint('foo < 3')])
            self.assertEqual(template.measurement_declarations, data['measurements'])


class ParameterNotIntegerExceptionTests(unittest.TestCase):

    def test(self) -> None:
        exception = ParameterNotIntegerException('foo', 3)
        self.assertIsInstance(str(exception), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)