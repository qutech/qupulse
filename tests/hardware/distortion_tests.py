import unittest
import numpy as np

from tests.pulses.sequencing_dummies import DummyInstructionBlock, DummyWaveform
from qctoolkit.pulses.instructions import EXECInstruction, REPJInstruction, STOPInstruction, ImmutableInstructionBlock, InstructionPointer
from qctoolkit.hardware.distortion import ControlFlowEmulator, DistortionCompensator


class ControlFlowEmulatorTests(unittest.TestCase):

    def test_stuff(self):
        exec_in1 = EXECInstruction(DummyWaveform())
        exec_in2 = EXECInstruction(DummyWaveform())

        outer_block = DummyInstructionBlock()
        outer_block.add_instruction(exec_in1)

        inner_block = DummyInstructionBlock()
        inner_block.add_instruction(exec_in1)
        inner_block.add_instruction(exec_in2)
        inner_block.return_ip = InstructionPointer(outer_block, len(outer_block) - 1)

        repj_in = REPJInstruction(4, InstructionPointer(inner_block, 0))
        outer_block.add_instruction(repj_in)
        outer_block.add_instruction(exec_in2)

        expected_exec_instructions = [exec_in1, exec_in1, exec_in2, exec_in1, exec_in2, exec_in1, exec_in2, exec_in1, exec_in2, exec_in2]
        observed_exec_instructions = []

        emulator = ControlFlowEmulator(ImmutableInstructionBlock(outer_block))
        while not isinstance(emulator.instruction, STOPInstruction):
            instruction = emulator.instruction
            if isinstance(instruction, EXECInstruction):
                observed_exec_instructions.append(instruction)
            emulator.make_step()

        self.assertEqual(expected_exec_instructions, observed_exec_instructions)


class DistortionCompensatorTests(unittest.TestCase):

    def test_more_stuff(self):
        exec_in1 = EXECInstruction(DummyWaveform(sample_output=np.linspace(0, 10, endpoint=False)))
        exec_in2 = EXECInstruction(DummyWaveform(sample_output=np.linspace(-10, 10, endpoint=False)))

        outer_block = DummyInstructionBlock()
        outer_block.add_instruction(exec_in1)

        inner_block = DummyInstructionBlock()
        inner_block.add_instruction(exec_in1)
        inner_block.add_instruction(exec_in2)
        inner_block.return_ip = InstructionPointer(outer_block, len(outer_block) - 1)

        repj_in = REPJInstruction(4, InstructionPointer(inner_block, 0))
        outer_block.add_instruction(repj_in)
        outer_block.add_instruction(exec_in2)

        #todo: fill tests
        c = DistortionCompensator()


