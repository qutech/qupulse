import unittest
import os
import sys

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from tests.pulses.SequencingDummies import DummyCondition, DummyPulseTemplate, DummySequencer, DummySequencingHardware, DummyInstructionBlock

from pulses.LoopPulseTemplate import LoopPulseTemplate
from pulses.Sequencer import Sequencer
from pulses.Instructions import EXECInstruction


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

    def test_create_constant_for_loop(self) -> None:
        body = DummyPulseTemplate()
        t = LoopPulseTemplate.create_constant_for_loop(10, body)
        hardware = DummySequencingHardware()
        sequencer = Sequencer(hardware)
        block = DummyInstructionBlock()
        sequencer.push(t, {}, block)
        sequencer.build()
        self.assertEqual(10, body.build_sequence_calls)

    def test_str(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate()
        t = LoopPulseTemplate(condition, body)
        self.assertIsInstance(str(t), str)


class LoopPulseTemplateSequencingTests(unittest.TestCase):

    def test_requires_stop(self) -> None:
        condition = DummyCondition(requires_stop=False)
        body = DummyPulseTemplate(requires_stop=False)
        t = LoopPulseTemplate(condition, body)
        self.assertFalse(t.requires_stop({}))

        condition.requires_stop_ = True
        self.assertTrue(t.requires_stop({}))

        body.requires_stop_ = True
        condition.requires_stop_ = False
        self.assertFalse(t.requires_stop({}))

    def test_build_sequence(self) -> None:
        condition = DummyCondition()
        body = DummyPulseTemplate()
        t = LoopPulseTemplate(condition, body)
        sequencer = DummySequencer(DummySequencingHardware())
        block = DummyInstructionBlock()
        parameters = {}
        t.build_sequence(sequencer, parameters, block)
        expected_data = dict(
            delegator=t,
            body=body,
            sequencer=sequencer,
            parameters=parameters,
            instruction_block=block
        )
        self.assertEqual(expected_data, condition.loop_call_data)
        self.assertFalse(condition.branch_call_data)
        self.assertFalse(sequencer.sequencing_stacks)
        self.assertFalse(sequencer.hardware.waveforms)


class GenericLoopPulseTemplateTest(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main(verbosity=2)