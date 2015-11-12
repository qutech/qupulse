import unittest

from qctoolkit.pulses.loop_pulse_template import LoopPulseTemplate, ConditionMissingException

from tests.pulses.sequencing_dummies import DummyCondition, DummyPulseTemplate, DummySequencer, DummyInstructionBlock
from tests.serialization_dummies import DummySerializer

class LoopPulseTemplateTest(unittest.TestCase):

    def test_parameter_names_and_declarations(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate()
        t = LoopPulseTemplate(condition, body)
        self.assertEqual(body.parameter_names, t.parameter_names)
        self.assertEqual(body.parameter_declarations, t.parameter_declarations)

        body.parameter_names_ = {'foo', 't', 'bar'}
        self.assertEqual(body.parameter_names, t.parameter_names)
        self.assertEqual(body.parameter_declarations, t.parameter_declarations)

    def test_is_interruptable(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate(is_interruptable=False)
        t = LoopPulseTemplate(condition, body)
        self.assertFalse(t.is_interruptable)

        body.is_interruptable_ = True
        self.assertTrue(t.is_interruptable)

    def test_str(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate()
        t = LoopPulseTemplate(condition, body)
        self.assertIsInstance(str(t), str)


class LoopPulseTemplateSequencingTests(unittest.TestCase):

    def test_requires_stop(self) -> None:
        condition = DummyCondition(requires_stop=False)
        conditions = {'foo_cond': condition}
        body = DummyPulseTemplate(requires_stop=False)
        t = LoopPulseTemplate('foo_cond', body)
        self.assertFalse(t.requires_stop({}, conditions))

        condition.requires_stop_ = True
        self.assertTrue(t.requires_stop({}, conditions))

        body.requires_stop_ = True
        condition.requires_stop_ = False
        self.assertFalse(t.requires_stop({}, conditions))

    def test_build_sequence(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate()
        t = LoopPulseTemplate('foo_cond', body)
        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        parameters = {}
        conditions = {'foo_cond': condition}
        t.build_sequence(sequencer, parameters, conditions, block)
        expected_data = dict(
            delegator=t,
            body=body,
            sequencer=sequencer,
            parameters=parameters,
            conditions=conditions,
            instruction_block=block
        )
        self.assertEqual(expected_data, condition.loop_call_data)
        self.assertFalse(condition.branch_call_data)
        self.assertFalse(sequencer.sequencing_stacks)

    def test_condition_missing(self) -> None:
        body = DummyPulseTemplate(requires_stop=False)
        t = LoopPulseTemplate('foo_cond', body)
        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        with self.assertRaises(ConditionMissingException):
            t.requires_stop({}, {})
            t.build_sequence(sequencer, {}, {}, block)


class LoopPulseTemplateSerializationTests(unittest.TestCase):

    def test_get_serialization_data(self) -> None:
        body = DummyPulseTemplate()
        condition_name = 'foo_cond'
        identifier = 'foo_loop'
        t = LoopPulseTemplate(condition_name, body, identifier=identifier)

        serializer = DummySerializer()
        expected_data = dict(type=serializer.get_type_identifier(t),
                             body=str(id(body)),
                             condition=condition_name)

        data = t.get_serialization_data(serializer)
        self.assertEqual(expected_data, data)

    def test_deserialize(self) -> None:
        data = dict(
            identifier='foo_loop',
            condition='foo_cond',
            body='bodyDummyPulse'
        )

        # prepare dependencies for deserialization
        serializer = DummySerializer()
        serializer.subelements[data['body']] = DummyPulseTemplate()

        # deserialize
        result = LoopPulseTemplate.deserialize(serializer, **data)

        # compare
        self.assertIs(serializer.subelements[data['body']], result.body)
        self.assertEqual(data['condition'], result.condition)
        self.assertEqual(data['identifier'], result.identifier)


class ConditionMissingExceptionTest(unittest.TestCase):

    def test(self) -> None:
        exc = ConditionMissingException('foo')
        self.assertIsInstance(str(exc), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)