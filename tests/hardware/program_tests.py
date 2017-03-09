import unittest
import itertools
from copy import deepcopy

import numpy as np

from string import Formatter, ascii_uppercase

from qctoolkit.hardware.program import Loop, MultiChannelProgram
from qctoolkit.pulses.instructions import REPJInstruction, InstructionBlock, ImmutableInstructionBlock
from tests.pulses.sequencing_dummies import DummyWaveform
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform


class WaveformGenerator:
    def __init__(self, num_channels,
                 duration_generator=itertools.repeat(None),
                 waveform_data_generator=itertools.repeat(None), channel_names=ascii_uppercase):
        self.num_channels = num_channels
        self.duration_generator = duration_generator
        self.waveform_data_generator = waveform_data_generator
        self.channel_names = channel_names[:num_channels]

    def generate_single_channel_waveform(self, channel):
        return DummyWaveform(sample_output=next(self.waveform_data_generator),
                             duration=next(self.duration_generator),
                             defined_channels={channel})

    def generate_multi_channel_waveform(self):
        return MultiChannelWaveform([self.generate_single_channel_waveform(self.channel_names[ch_i])
                                     for ch_i in range(self.num_channels)])

    def __call__(self):
        return self.generate_multi_channel_waveform()


def get_two_chan_test_block(wfg=WaveformGenerator(2)):
    generate_waveform = wfg.generate_single_channel_waveform
    generate_multi_channel_waveform = wfg.generate_multi_channel_waveform

    loop_block11 = InstructionBlock()
    loop_block11.add_instruction_exec(generate_multi_channel_waveform())

    loop_block1 = InstructionBlock()
    loop_block1.add_instruction_repj(5, ImmutableInstructionBlock(loop_block11))

    loop_block21 = InstructionBlock()
    loop_block21.add_instruction_exec(generate_multi_channel_waveform())
    loop_block21.add_instruction_exec(generate_multi_channel_waveform())

    loop_block2 = InstructionBlock()
    loop_block2.add_instruction_repj(2, ImmutableInstructionBlock(loop_block21))
    loop_block2.add_instruction_exec(generate_multi_channel_waveform())

    loop_block3 = InstructionBlock()
    loop_block3.add_instruction_exec(generate_multi_channel_waveform())
    loop_block3.add_instruction_exec(generate_multi_channel_waveform())

    loop_block411 = InstructionBlock()
    loop_block411.add_instruction_exec(MultiChannelWaveform([generate_waveform('A')]))
    loop_block412 = InstructionBlock()
    loop_block412.add_instruction_exec(MultiChannelWaveform([generate_waveform('A')]))

    loop_block41 = InstructionBlock()
    loop_block41.add_instruction_repj(7, ImmutableInstructionBlock(loop_block411))
    loop_block41.add_instruction_repj(8, ImmutableInstructionBlock(loop_block412))

    loop_block421 = InstructionBlock()
    loop_block421.add_instruction_exec(MultiChannelWaveform([generate_waveform('B')]))
    loop_block422 = InstructionBlock()
    loop_block422.add_instruction_exec(MultiChannelWaveform([generate_waveform('B')]))

    loop_block42 = InstructionBlock()
    loop_block42.add_instruction_repj(10, ImmutableInstructionBlock(loop_block421))
    loop_block42.add_instruction_repj(11, ImmutableInstructionBlock(loop_block422))

    chan_block4A = InstructionBlock()
    chan_block4A.add_instruction_repj(6, ImmutableInstructionBlock(loop_block41))

    chan_block4B = InstructionBlock()
    chan_block4B.add_instruction_repj(9, ImmutableInstructionBlock(loop_block42))

    loop_block4 = InstructionBlock()
    loop_block4.add_instruction_chan({frozenset('A'): ImmutableInstructionBlock(chan_block4A),
                                           frozenset('B'): ImmutableInstructionBlock(chan_block4B)})

    root_block = InstructionBlock()
    root_block.add_instruction_exec(generate_multi_channel_waveform())
    root_block.add_instruction_repj(10, ImmutableInstructionBlock(loop_block1))
    root_block.add_instruction_repj(17, ImmutableInstructionBlock(loop_block2))
    root_block.add_instruction_repj(3, ImmutableInstructionBlock(loop_block3))
    root_block.add_instruction_repj(4, ImmutableInstructionBlock(loop_block4))

    return root_block


class LoopTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.maxDiff = None

        self.test_loop_repr = \
"""\
LOOP 1 times:
  ->EXEC {} 1 times
  ->LOOP 10 times:
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

    @staticmethod
    def get_test_loop(waveform_generator=None):
        if waveform_generator is None:
            waveform_generator = lambda: None

        return Loop(repetition_count=1, children=[Loop(repetition_count=1, waveform=waveform_generator()),
                                                  Loop(repetition_count=10, children=[Loop(repetition_count=50, waveform=waveform_generator())]),
                                                  Loop(repetition_count=17, children=[Loop(repetition_count=2, children=[Loop(repetition_count=1, waveform=waveform_generator()),
                                                                                                                         Loop(repetition_count=1, waveform=waveform_generator())]),
                                                                                      Loop(repetition_count=1, waveform=waveform_generator())]),
                                                  Loop(repetition_count=3, children=[Loop(repetition_count=1, waveform=waveform_generator()),
                                                                                     Loop(repetition_count=1, waveform=waveform_generator())]),
                                                  Loop(repetition_count=4, children=[Loop(repetition_count=6, children=[Loop(repetition_count=7, waveform=waveform_generator()),
                                                                                                                        Loop(repetition_count=8, waveform=waveform_generator())]),
                                                                                     Loop(repetition_count=9, children=[Loop(repetition_count=10, waveform=waveform_generator()),
                                                                                                                        Loop(repetition_count=11, waveform=waveform_generator())])])])

    def test_compare_key(self):
        wf_gen = WaveformGenerator(num_channels=1)

        wf_1 = wf_gen()
        wf_2 = wf_gen()

        tree1 = Loop(children=[Loop(waveform=wf_1, repetition_count=5)])
        tree2 = Loop(children=[Loop(waveform=wf_1, repetition_count=4)])
        tree3 = Loop(children=[Loop(waveform=wf_2, repetition_count=5)])
        tree4 = Loop(children=[Loop(waveform=wf_1, repetition_count=5)])

        self.assertNotEqual(tree1, tree2)
        self.assertNotEqual(tree1, tree3)
        self.assertNotEqual(tree2, tree3)
        self.assertEqual(tree1, tree4)

        tree1 = Loop(children=[Loop(waveform=wf_1, repetition_count=5),
                               Loop(waveform=wf_2, repetition_count=7)], repetition_count=2)
        tree2 = Loop(children=[Loop(waveform=wf_1, repetition_count=5),
                               Loop(waveform=wf_2, repetition_count=5)], repetition_count=2)
        tree3 = Loop(children=[Loop(waveform=wf_1, repetition_count=5),
                               Loop(waveform=wf_1, repetition_count=7)], repetition_count=2)
        tree4 = Loop(children=[Loop(waveform=wf_1, repetition_count=5),
                               Loop(waveform=wf_2, repetition_count=7)], repetition_count=3)
        tree5 = Loop(children=[Loop(waveform=wf_1, repetition_count=5),
                               Loop(waveform=wf_2, repetition_count=7)], repetition_count=2)
        self.assertNotEqual(tree1, tree2)
        self.assertNotEqual(tree1, tree3)
        self.assertNotEqual(tree1, tree4)
        self.assertEqual(tree1, tree5)

    def test_repr(self):
        wf_gen = WaveformGenerator(num_channels=1)
        wfs = [wf_gen() for _ in range(11)]

        expected = self.test_loop_repr.format(*wfs)

        tree = self.get_test_loop()
        for loop in tree.get_depth_first_iterator():
            if loop.is_leaf():
                loop.waveform = wfs.pop(0)
        self.assertEqual(len(wfs), 0)

        self.assertEqual(repr(tree), expected)

    def test_is_leaf(self):
        root_loop = self.get_test_loop(waveform_generator=WaveformGenerator(1))

        for loop in root_loop.get_depth_first_iterator():
            self.assertTrue(bool(loop.is_leaf()) != bool(loop.waveform is None))

        for loop in root_loop.get_breadth_first_iterator():
            self.assertTrue(bool(loop.is_leaf()) != bool(loop.waveform is None))

    def test_depth(self):
        root_loop = self.get_test_loop()
        self.assertEqual(root_loop.depth(), 3)
        self.assertEqual(root_loop[-1].depth(), 2)
        self.assertEqual(root_loop[-1][-1].depth(), 1)
        self.assertEqual(root_loop[-1][-1][-1].depth(), 0)
        with self.assertRaises(IndexError):
            root_loop[-1][-1][-1][-1].depth()

    def test_is_balanced(self):
        root_loop = self.get_test_loop()
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

    def get_mcp(self, channels):
        program = MultiChannelProgram(self.root_block, ['A', 'B'])
        return program[channels]

    def test_init(self):
        with self.assertRaises(ValueError):
            MultiChannelProgram(InstructionBlock())

        mcp = MultiChannelProgram(self.root_block, ['A', 'B'])
        self.assertEqual(mcp.channels, {'A', 'B'})

        with self.assertRaises(KeyError):
            mcp['C']

    def test_via_repr(self):
        root_loopA = self.get_mcp('A')
        root_loopB = self.get_mcp('B')
        waveformsA = tuple(loop.waveform
                                           for loop in root_loopA.get_depth_first_iterator() if loop.is_leaf())
        reprA = self.descriptionA.format(*waveformsA)
        reprB = self.descriptionB.format(*(loop.waveform
                                           for loop in root_loopB.get_depth_first_iterator() if loop.is_leaf()))
        self.assertEqual(root_loopA.__repr__(), reprA)
        self.assertEqual(root_loopB.__repr__(), reprB)
