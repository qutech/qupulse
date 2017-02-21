import unittest
import itertools
from copy import deepcopy

from string import Formatter

from qctoolkit.hardware.program import Loop, MultiChannelProgram
from qctoolkit.pulses.instructions import REPJInstruction, InstructionBlock, ImmutableInstructionBlock
from tests.pulses.sequencing_dummies import DummyWaveform
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform


class LoopTests(unittest.TestCase):
    def __init__(self, *args, waveform_data_generator=itertools.repeat(None), waveform_duration=None, num_channels=2, **kwargs):
        super().__init__(*args, **kwargs)

        names = 'ABCDEFGH'[:num_channels]

        def generate_waveform(chan):
            return DummyWaveform(sample_output=next(waveform_data_generator),
                                 duration=waveform_duration,
                                 defined_channels={chan})

        def generate_multi_channel_waveform():
            return MultiChannelWaveform([generate_waveform(names[ch_i]) for ch_i in range(num_channels)])


        self.old_description = \
"""\
LOOP 1 times:
  ->EXEC 1 times
  ->LOOP 10 times:
      ->LOOP 5 times:
          ->EXEC 1 times
  ->LOOP 17 times:
      ->LOOP 2 times:
          ->EXEC 1 times
          ->EXEC 1 times
      ->EXEC 1 times
  ->LOOP 3 times:
      ->EXEC 1 times
      ->EXEC 1 times
  ->LOOP 4 times:
      ->LOOP 6 times:
          ->LOOP 7 times:
              ->EXEC 1 times
          ->LOOP 8 times:
              ->EXEC 1 times
      ->LOOP 9 times:
          ->LOOP 10 times:
              ->EXEC 1 times
          ->LOOP 11 times:
              ->EXEC 1 times"""

        self.new_description = \
"""\
LOOP 1 times:
  ->EXEC {} 1 times
  ->EXEC {} 50 times
  ->LOOP 17 times:
      ->LOOP 2 times:
          ->EXEC {} 1 times
          ->EXEC {} 1 times
      ->EXEC {} 1 times
  ->LOOP 3 times:
      ->EXEC {} 1 times
      ->EXEC {} 1 times
  ->LOOP 4 times:
      ->LOOP 6 times:
          ->EXEC {} 7 times
          ->EXEC {} 8 times
      ->LOOP 9 times:
          ->EXEC {} 10 times
          ->EXEC {} 11 times"""
        self.root_block = InstructionBlock()
        self.loop_block11 = InstructionBlock()
        self.loop_block1 = InstructionBlock()
        self.loop_block21 = InstructionBlock()
        self.loop_block2 = InstructionBlock()
        self.loop_block3 = InstructionBlock()
        self.loop_block411 = InstructionBlock()
        self.loop_block412 = InstructionBlock()
        self.loop_block41 = InstructionBlock()
        self.loop_block4 = InstructionBlock()
        self.loop_block421 = InstructionBlock()
        self.loop_block422 = InstructionBlock()
        self.loop_block42 = InstructionBlock()

        self.root_block.add_instruction_exec(generate_multi_channel_waveform())

        self.loop_block11.add_instruction_exec(generate_multi_channel_waveform())
        self.loop_block1.add_instruction_repj(5, ImmutableInstructionBlock(self.loop_block11))

        self.loop_block21.add_instruction_exec(generate_multi_channel_waveform())
        self.loop_block21.add_instruction_exec(generate_multi_channel_waveform())

        self.loop_block2.add_instruction_repj(2, ImmutableInstructionBlock(self.loop_block21))
        self.loop_block2.add_instruction_exec(generate_multi_channel_waveform())

        self.loop_block3.add_instruction_exec(generate_multi_channel_waveform())
        self.loop_block3.add_instruction_exec(generate_multi_channel_waveform())

        self.loop_block411.add_instruction_exec(generate_multi_channel_waveform())
        self.loop_block412.add_instruction_exec(generate_multi_channel_waveform())

        self.loop_block41.add_instruction_repj(7, ImmutableInstructionBlock(self.loop_block411))
        self.loop_block41.add_instruction_repj(8, ImmutableInstructionBlock(self.loop_block412))

        self.loop_block421.add_instruction_exec(generate_multi_channel_waveform())
        self.loop_block422.add_instruction_exec(generate_multi_channel_waveform())

        self.loop_block42.add_instruction_repj(10, ImmutableInstructionBlock(self.loop_block421))
        self.loop_block42.add_instruction_repj(11, ImmutableInstructionBlock(self.loop_block422))

        self.loop_block4.add_instruction_repj(6, ImmutableInstructionBlock(self.loop_block41))
        self.loop_block4.add_instruction_repj(9, ImmutableInstructionBlock(self.loop_block42))

        self.root_block.add_instruction_repj(10, ImmutableInstructionBlock(self.loop_block1))
        self.root_block.add_instruction_repj(17, ImmutableInstructionBlock(self.loop_block2))
        self.root_block.add_instruction_repj(3,  ImmutableInstructionBlock(self.loop_block3))
        self.root_block.add_instruction_repj(4, ImmutableInstructionBlock(self.loop_block4))

        self.maxDiff = None

    def get_root_loop(self):
        program = MultiChannelProgram(self.root_block, {'A', 'B'})
        return program[{'A', 'B'}]

    def test_repr(self):
        root_loop = self.get_root_loop()
        repres = root_loop.__repr__()
        expected = self.new_description.format(*(loop.waveform
                                                 for loop in root_loop.get_depth_first_iterator() if loop.is_leaf()))
        self.assertEqual(repres, expected)

    def test_is_leaf(self):
        root_loop = self.get_root_loop()

        for loop in root_loop.get_depth_first_iterator():
            self.assertTrue(bool(loop.is_leaf()) != bool(loop.waveform is None))

        for loop in root_loop.get_breadth_first_iterator():
            self.assertTrue(bool(loop.is_leaf()) != bool(loop.waveform is None))

    def test_depth(self):
        root_loop = self.get_root_loop()
        self.assertEqual(root_loop.depth(), 3)
        self.assertEqual(root_loop[-1].depth(), 2)
        self.assertEqual(root_loop[-1][-1].depth(), 1)
        self.assertEqual(root_loop[-1][-1][-1].depth(), 0)
        with self.assertRaises(IndexError):
            root_loop[-1][-1][-1][-1].depth()

    def test_is_balanced(self):
        root_loop = self.get_root_loop()
        self.assertFalse(root_loop.is_balanced())

        self.assertFalse(root_loop[2].is_balanced())
        self.assertTrue(root_loop[0].is_balanced())
        self.assertTrue(root_loop[1].is_balanced())
        self.assertTrue(root_loop[3].is_balanced())
        self.assertTrue(root_loop[4].is_balanced())


class MultiChannelTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wf = DummyWaveform()
        self.descriptionA = \
"""\
LOOP 1 times:
  ->EXEC {} 1 times
  ->EXEC {} 50 times
  ->LOOP 17 times:
      ->LOOP 2 times:
          ->EXEC {} 1 times
          ->EXEC {} 1 times
      ->EXEC {} 1 times
  ->LOOP 3 times:
      ->EXEC {} 1 times
      ->EXEC {} 1 times
  ->LOOP 24 times:
      ->EXEC {} 7 times
      ->EXEC {} 8 times"""
        self.descriptionB = \
"""\
LOOP 1 times:
  ->EXEC {} 1 times
  ->EXEC {} 50 times
  ->LOOP 17 times:
      ->LOOP 2 times:
          ->EXEC {} 1 times
          ->EXEC {} 1 times
      ->EXEC {} 1 times
  ->LOOP 3 times:
      ->EXEC {} 1 times
      ->EXEC {} 1 times
  ->LOOP 36 times:
      ->EXEC {} 10 times
      ->EXEC {} 11 times"""

        def generate_waveform(channel):
            return DummyWaveform(sample_output=None, duration=None, defined_channels={channel})

        def generate_multi_channel_waveform():
            return MultiChannelWaveform([generate_waveform('A'), generate_waveform('B')])

        self.loop_block11 = InstructionBlock()
        self.loop_block11.add_instruction_exec(generate_multi_channel_waveform())

        self.loop_block1 = InstructionBlock()
        self.loop_block1.add_instruction_repj(5, ImmutableInstructionBlock(self.loop_block11))

        self.loop_block21 = InstructionBlock()
        self.loop_block21.add_instruction_exec(generate_multi_channel_waveform())
        self.loop_block21.add_instruction_exec(generate_multi_channel_waveform())

        self.loop_block2 = InstructionBlock()
        self.loop_block2.add_instruction_repj(2, ImmutableInstructionBlock(self.loop_block21))
        self.loop_block2.add_instruction_exec(generate_multi_channel_waveform())

        self.loop_block3 = InstructionBlock()
        self.loop_block3.add_instruction_exec(generate_multi_channel_waveform())
        self.loop_block3.add_instruction_exec(generate_multi_channel_waveform())

        self.loop_block411 = InstructionBlock()
        self.loop_block411.add_instruction_exec(MultiChannelWaveform([generate_waveform('A')]))
        self.loop_block412 = InstructionBlock()
        self.loop_block412.add_instruction_exec(MultiChannelWaveform([generate_waveform('A')]))

        self.loop_block41 = InstructionBlock()
        self.loop_block41.add_instruction_repj(7, ImmutableInstructionBlock(self.loop_block411))
        self.loop_block41.add_instruction_repj(8, ImmutableInstructionBlock(self.loop_block412))

        self.loop_block421 = InstructionBlock()
        self.loop_block421.add_instruction_exec(MultiChannelWaveform([generate_waveform('B')]))
        self.loop_block422 = InstructionBlock()
        self.loop_block422.add_instruction_exec(MultiChannelWaveform([generate_waveform('B')]))

        self.loop_block42 = InstructionBlock()
        self.loop_block42.add_instruction_repj(10, ImmutableInstructionBlock(self.loop_block421))
        self.loop_block42.add_instruction_repj(11, ImmutableInstructionBlock(self.loop_block422))

        self.chan_block4A = InstructionBlock()
        self.chan_block4A.add_instruction_repj(6, ImmutableInstructionBlock(self.loop_block41))

        self.chan_block4B = InstructionBlock()
        self.chan_block4B.add_instruction_repj(9, ImmutableInstructionBlock(self.loop_block42))

        self.loop_block4 = InstructionBlock()
        self.loop_block4.add_instruction_chan({frozenset('A'): ImmutableInstructionBlock(self.chan_block4A),
                                              frozenset('B'): ImmutableInstructionBlock(self.chan_block4B)})

        self.root_block = InstructionBlock()
        self.root_block.add_instruction_exec(generate_multi_channel_waveform())
        self.root_block.add_instruction_repj(10, ImmutableInstructionBlock(self.loop_block1))
        self.root_block.add_instruction_repj(17, ImmutableInstructionBlock(self.loop_block2))
        self.root_block.add_instruction_repj(3, ImmutableInstructionBlock(self.loop_block3))
        self.root_block.add_instruction_repj(4, ImmutableInstructionBlock(self.loop_block4))

        self.maxDiff = None

    def get_root_loop(self, channels):
        program = MultiChannelProgram(self.root_block, ['A', 'B'])
        return program[channels]

    def test_via_repr(self):
        root_loopA = self.get_root_loop('A')
        root_loopB = self.get_root_loop('B')
        waveformsA = tuple(loop.waveform
                                           for loop in root_loopA.get_depth_first_iterator() if loop.is_leaf())
        reprA = self.descriptionA.format(*waveformsA)
        reprB = self.descriptionB.format(*(loop.waveform
                                           for loop in root_loopB.get_depth_first_iterator() if loop.is_leaf()))
        self.assertEqual(root_loopA.__repr__(), reprA)
        self.assertEqual(root_loopB.__repr__(), reprB)
