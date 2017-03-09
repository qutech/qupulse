import unittest

from qctoolkit.pulses.branch_pulse_template import BranchPulseTemplate
from qctoolkit.pulses.parameters import ParameterDeclaration
from qctoolkit.pulses.conditions import ConditionMissingException

from tests.pulses.sequencing_dummies import DummyPulseTemplate, DummyParameter, DummyCondition, DummySequencer, DummyInstructionBlock
from tests.serialization_dummies import DummySerializer


class BranchPulseTemplateTest(unittest.TestCase):

    def test_wrong_channel_composition(self) -> None:
        if_dummy = DummyPulseTemplate(defined_channels={'A', 'B'})
        else_dummy = DummyPulseTemplate(defined_channels={'A', 'C'})
        with self.assertRaises(ValueError):
            BranchPulseTemplate('foo_condition', if_dummy, else_dummy)

    def test_identifier(self) -> None:
        if_dummy = DummyPulseTemplate(defined_channels={'A', 'B'})
        else_dummy = DummyPulseTemplate(defined_channels={'A', 'B'})
        template = BranchPulseTemplate('foo_condition', if_dummy, else_dummy, identifier='hugo')
        self.assertEqual('hugo', template.identifier)

    def test_parameter_names_and_declarations(self) -> None:
        if_dummy = DummyPulseTemplate(defined_channels={'A', 'B'}, parameter_names={'foo', 'bar'})
        else_dummy = DummyPulseTemplate(defined_channels={'A', 'B'}, parameter_names={'foo', 'hugo'})
        template = BranchPulseTemplate('foo_condition', if_dummy, else_dummy)
        self.assertEqual({'foo', 'bar', 'hugo'}, template.parameter_names)
        self.assertEqual({ParameterDeclaration(name) for name in {'foo', 'bar', 'hugo'}}, template.parameter_declarations)

    def test_defined_channels(self) -> None:
        if_dummy = DummyPulseTemplate(defined_channels={'A', 'B'})
        else_dummy = DummyPulseTemplate(defined_channels={'A', 'B'})
        template = BranchPulseTemplate('foo_condition', if_dummy, else_dummy)
        self.assertEqual({'A', 'B'}, template.defined_channels)

    def test_measurement_names(self) -> None:
        if_dummy = DummyPulseTemplate(measurement_names={'if_meas'})
        else_dummy = DummyPulseTemplate(measurement_names={'else_meas'})
        template = BranchPulseTemplate('foo_condition', if_dummy, else_dummy)
        self.assertEqual({'if_meas','else_meas'}, template.measurement_names)

    def test_is_interruptable(self) -> None:
        if_dummy = DummyPulseTemplate(defined_channels={'A', 'B'}, is_interruptable=True)
        else_dummy = DummyPulseTemplate(defined_channels={'A', 'B'})
        template = BranchPulseTemplate('foo_condition', if_dummy, else_dummy)
        self.assertFalse(template.is_interruptable)

        if_dummy = DummyPulseTemplate(defined_channels={'A', 'B'}, is_interruptable=True)
        else_dummy = DummyPulseTemplate(defined_channels={'A', 'B'}, is_interruptable=True)
        template = BranchPulseTemplate('foo_condition', if_dummy, else_dummy)
        self.assertTrue(template.is_interruptable)

    def test_str(self) -> None:
        if_dummy = DummyPulseTemplate(defined_channels={'A', 'B'}, is_interruptable=True)
        else_dummy = DummyPulseTemplate(defined_channels={'A', 'B'})
        template = BranchPulseTemplate('foo_condition', if_dummy, else_dummy)
        self.assertIsInstance(str(template), str)
        

class BranchPulseTemplateSequencingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.maxDiff = None
        self.if_dummy   = DummyPulseTemplate(defined_channels={'A', 'B'}, parameter_names={'foo', 'bar'},
                                             measurement_names={'if_meas'})
        self.else_dummy = DummyPulseTemplate(defined_channels={'A', 'B'}, parameter_names={'foo', 'hugo'},
                                             measurement_names={'else_meas'})
        self.template   = BranchPulseTemplate('foo_condition', self.if_dummy, self.else_dummy)
        self.sequencer  = DummySequencer()
        self.block      = DummyInstructionBlock()

    def test_requires_stop_parameters_dont_conditions_dont(self) -> None:
        conditions = dict(foo_condition=DummyCondition(requires_stop=False))
        parameters = dict(foo=DummyParameter(326.272),
                          bar=DummyParameter(-2624.23),
                          hugo=DummyParameter(3.532))
        self.assertFalse(self.template.requires_stop(parameters, conditions))

    def test_requires_stop_parameters_do_conditions_dont(self) -> None:
        conditions = dict(foo_condition=DummyCondition(requires_stop=False))
        parameters = dict(foo=DummyParameter(326.272),
                          bar=DummyParameter(-2624.23, requires_stop=True),
                          hugo=DummyParameter(3.532))
        self.assertFalse(self.template.requires_stop(parameters, conditions))

    def test_requires_stop_parameters_dont_conditions_do(self) -> None:
        conditions = dict(foo_condition=DummyCondition(requires_stop=True))
        parameters = dict(foo=DummyParameter(326.272),
                          bar=DummyParameter(-2624.23),
                          hugo=DummyParameter(3.532))
        self.assertTrue(self.template.requires_stop(parameters, conditions))

    def test_requires_stop_parameters_do_conditions_do(self) -> None:
        conditions = dict(foo_condition=DummyCondition(requires_stop=True))
        parameters = dict(foo=DummyParameter(326.272, requires_stop=True),
                          bar=DummyParameter(-2624.23),
                          hugo=DummyParameter(3.532))
        self.assertTrue(self.template.requires_stop(parameters, conditions))

    def test_requires_stop_condition_missing(self) -> None:
        conditions = dict(bar_condition=DummyCondition(requires_stop=True))
        parameters = dict(foo=DummyParameter(326.272),
                          bar=DummyParameter(-2624.23),
                          hugo=DummyParameter(3.532))
        with self.assertRaises(ConditionMissingException):
            self.template.requires_stop(parameters, conditions)

    def test_requires_stop_parameters_missing(self) -> None:
        conditions = dict(foo_condition=DummyCondition(requires_stop=False))
        parameters = dict(foo=DummyParameter(326.272),
                          hugo=DummyParameter(3.532))
        self.assertFalse(self.template.requires_stop(parameters, conditions))

    def test_build_sequence(self) -> None:
        foo_condition = DummyCondition()
        conditions = dict(foo_condition=foo_condition)
        parameters = dict(foo=DummyParameter(326.272),
                          bar=DummyParameter(-2624.23),
                          hugo=DummyParameter(3.532))
        window_mapping = dict(else_meas='my_meas',if_meas='thy_meas')
        channel_mapping = dict()

        self.template.build_sequence(self.sequencer, parameters, conditions, window_mapping, channel_mapping, self.block)
        self.assertFalse(foo_condition.loop_call_data)
        self.assertEqual(
            dict(
                delegator=self.template,
                if_branch=self.if_dummy,
                else_branch=self.else_dummy,
                sequencer=self.sequencer,
                parameters=parameters,
                conditions=conditions,
                measurement_mapping=window_mapping,
                channel_mapping=channel_mapping,
                instruction_block=self.block
            ),
            foo_condition.branch_call_data
        )
        self.assertFalse(self.sequencer.sequencing_stacks)
        self.assertFalse(self.block.instructions)

    def test_build_sequence_condition_missing(self) -> None:
        conditions = dict(bar_condition=DummyCondition(requires_stop=True))
        parameters = dict(foo=DummyParameter(326.272),
                          bar=DummyParameter(-2624.23),
                          hugo=DummyParameter(3.532))
        window_mapping = dict()
        channel_mapping = dict()
        with self.assertRaises(ConditionMissingException):
            self.template.build_sequence(self.sequencer, parameters, conditions, window_mapping, channel_mapping, self.block)

    def test_build_sequence_parameter_missing(self) -> None:
        foo_condition = DummyCondition()
        conditions = dict(foo_condition=foo_condition)
        parameters = dict(foo=DummyParameter(326.272),
                          bar=DummyParameter(-2624.23))
        window_mapping = dict(else_meas='my_meas',if_meas='thy_meas')
        channel_mapping = dict()
        self.template.build_sequence(self.sequencer, parameters, conditions, window_mapping, channel_mapping, self.block)
        self.assertFalse(foo_condition.loop_call_data)
        self.assertEqual(
            dict(
                delegator=self.template,
                if_branch=self.if_dummy,
                else_branch=self.else_dummy,
                sequencer=self.sequencer,
                parameters=parameters,
                conditions=conditions,
                measurement_mapping=window_mapping,
                channel_mapping=channel_mapping,
                instruction_block=self.block
            ),
            foo_condition.branch_call_data
        )
        self.assertFalse(self.sequencer.sequencing_stacks)
        self.assertFalse(self.block.instructions)


class BranchPulseTemplateSerializationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.maxDiff = None
        self.if_dummy = DummyPulseTemplate(defined_channels={'A', 'B'}, parameter_names={'foo', 'bar'}, measurement_names={'if_mease'})
        self.else_dummy = DummyPulseTemplate(defined_channels={'A', 'B'}, parameter_names={'foo', 'hugo'}, measurement_names={'else_meas'})
        self.template = BranchPulseTemplate('foo_condition', self.if_dummy, self.else_dummy)

    def test_get_serialization_data(self) -> None:
        expected_data = dict(
            if_branch_template=str(id(self.if_dummy)),
            else_branch_template=str(id(self.else_dummy)),
            condition='foo_condition'
        )
        serializer = DummySerializer()
        serialized_data = self.template.get_serialization_data(serializer)
        self.assertEqual(expected_data, serialized_data)

    def test_deserialize(self) -> None:
        base_data = dict(
            if_branch_template=str(id(self.if_dummy)),
            else_branch_template=str(id(self.else_dummy)),
            condition='foo_condition',
            identifier='hugo'
        )
        serializer = DummySerializer()
        serializer.subelements[str(id(self.if_dummy))] = self.if_dummy
        serializer.subelements[str(id(self.else_dummy))] = self.else_dummy
        template = BranchPulseTemplate.deserialize(serializer, **base_data)
        self.assertEqual('hugo', template.identifier)
        serialized_data = template.get_serialization_data(serializer)
        del base_data['identifier']
        self.assertEqual(base_data, serialized_data)