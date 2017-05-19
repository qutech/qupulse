import unittest

from sympy import sympify

from qctoolkit.pulses.loop_pulse_template import ForLoopPulseTemplate, WhileLoopPulseTemplate,\
    ConditionMissingException, ParametrizedRange, LoopIndexNotUsedException
from qctoolkit.pulses.parameters import ConstantParameter

from tests.pulses.sequencing_dummies import DummyCondition, DummyPulseTemplate, DummySequencer, DummyInstructionBlock
from tests.serialization_dummies import DummySerializer


class ParametrizedRangeTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init(self):
        self.assertEqual(ParametrizedRange(7).to_tuple(),
                         (0, 7, 1))
        self.assertEqual(ParametrizedRange(4, 7).to_tuple(),
                         (4, 7, 1))
        self.assertEqual(ParametrizedRange(4, 'h', 5).to_tuple(),
                         (4, 'h', 5))

        self.assertEqual(ParametrizedRange(start=7, stop=1, step=-1).to_tuple(),
                         (7, 1, -1))

        with self.assertRaises(TypeError):
            ParametrizedRange()
        with self.assertRaises(TypeError):
            ParametrizedRange(1, 2, 3, 4)

    def test_to_range(self):
        pr = ParametrizedRange(4, 'l*k', 'k')

        self.assertEqual(pr.to_range({'l': 5, 'k': 2}),
                         range(4, 10, 2))

    def test_parameter_names(self):
        self.assertEqual(ParametrizedRange(5).parameter_names, set())
        self.assertEqual(ParametrizedRange('g').parameter_names, {'g'})
        self.assertEqual(ParametrizedRange('g*h', 'h', 'l/m').parameter_names, {'g', 'h', 'l', 'm'})


class ForLoopPulseTemplateTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init(self):
        dt = DummyPulseTemplate(parameter_names={'i', 'k'})
        self.assertEqual(ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=5).loop_range.to_tuple(),
                         (0, 5, 1))
        self.assertEqual(ForLoopPulseTemplate(body=dt, loop_index='i', loop_range='s').loop_range.to_tuple(),
                         (0, 's', 1))
        self.assertEqual(ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=(2, 5)).loop_range.to_tuple(),
                         (2, 5, 1))
        self.assertEqual(ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=range(1, 2, 5)).loop_range.to_tuple(),
                         (1, 2, 5))
        self.assertEqual(ForLoopPulseTemplate(body=dt, loop_index='i',
                                              loop_range=ParametrizedRange('a', 'b', 'c')).loop_range.to_tuple(),
                         ('a', 'b', 'c'))

        with self.assertRaises(ValueError):
            ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=slice(None))

        with self.assertRaises(LoopIndexNotUsedException):
            ForLoopPulseTemplate(body=DummyPulseTemplate(), loop_index='i', loop_range=1)

    def test_body_parameter_generator(self):
        dt = DummyPulseTemplate(parameter_names={'i', 'k'})
        flt = ForLoopPulseTemplate(body=dt, loop_index='i', loop_range=('a', 'b', 'c'))

        expected_range = range(2, 17, 3)

        outer_params = dict(k=ConstantParameter(5),
                            a=ConstantParameter(expected_range.start),
                            b=ConstantParameter(expected_range.stop),
                            c=ConstantParameter(expected_range.step))
        forward_parameter_dicts = list(flt._body_parameter_generator(outer_params, forward=True))
        backward_parameter_dicts = list(flt._body_parameter_generator(outer_params, forward=False))

        self.assertEqual(forward_parameter_dicts, list(reversed(backward_parameter_dicts)))
        for local_params, i in zip(forward_parameter_dicts, expected_range):
            expected_local_params = dict(k=ConstantParameter(5), i=ConstantParameter(i))
            self.assertEqual(expected_local_params, local_params)


class WhileLoopPulseTemplateTest(unittest.TestCase):

    def test_parameter_names_and_declarations(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate()
        t = WhileLoopPulseTemplate(condition, body)
        self.assertEqual(body.parameter_names, t.parameter_names)

        body.parameter_names_ = {'foo', 't', 'bar'}
        self.assertEqual(body.parameter_names, t.parameter_names)

    @unittest.skip
    def test_is_interruptable(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate(is_interruptable=False)
        t = WhileLoopPulseTemplate(condition, body)
        self.assertFalse(t.is_interruptable)

        body.is_interruptable_ = True
        self.assertTrue(t.is_interruptable)

    def test_str(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate()
        t = WhileLoopPulseTemplate(condition, body)
        self.assertIsInstance(str(t), str)


class LoopPulseTemplateSequencingTests(unittest.TestCase):

    def test_requires_stop(self) -> None:
        condition = DummyCondition(requires_stop=False)
        conditions = {'foo_cond': condition}
        body = DummyPulseTemplate(requires_stop=False)
        t = WhileLoopPulseTemplate('foo_cond', body)
        self.assertFalse(t.requires_stop({}, conditions))

        condition.requires_stop_ = True
        self.assertTrue(t.requires_stop({}, conditions))

        body.requires_stop_ = True
        condition.requires_stop_ = False
        self.assertFalse(t.requires_stop({}, conditions))

    def test_build_sequence(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate()
        t = WhileLoopPulseTemplate('foo_cond', body)
        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        parameters = {}
        conditions = {'foo_cond': condition}
        measurement_mapping = {'swag': 'aufdrehen'}
        channel_mapping = {}
        t.build_sequence(sequencer, parameters, conditions, measurement_mapping, channel_mapping, block)
        expected_data = dict(
            delegator=t,
            body=body,
            sequencer=sequencer,
            parameters=parameters,
            conditions=conditions,
            measurement_mapping=measurement_mapping,
            channel_mapping=channel_mapping,
            instruction_block=block
        )
        self.assertEqual(expected_data, condition.loop_call_data)
        self.assertFalse(condition.branch_call_data)
        self.assertFalse(sequencer.sequencing_stacks)

    def test_condition_missing(self) -> None:
        body = DummyPulseTemplate(requires_stop=False)
        t = WhileLoopPulseTemplate('foo_cond', body)
        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        with self.assertRaises(ConditionMissingException):
            t.requires_stop({}, {})
            t.build_sequence(sequencer, {}, {}, {}, block)


class LoopPulseTemplateSerializationTests(unittest.TestCase):

    def test_get_serialization_data(self) -> None:
        body = DummyPulseTemplate()
        condition_name = 'foo_cond'
        identifier = 'foo_loop'
        t = WhileLoopPulseTemplate(condition_name, body, identifier=identifier)

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
        result = WhileLoopPulseTemplate.deserialize(serializer, **data)

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