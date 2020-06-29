import unittest
from unittest import mock
import itertools

from string import ascii_uppercase

from qupulse.expressions import ExpressionScalar
from qupulse.parameter_scope import DictScope

from qupulse.utils.types import TimeType, time_from_float
from qupulse._program.volatile import VolatileRepetitionCount
from qupulse._program._loop import Loop, _make_compatible, _is_compatible, _CompatibilityLevel,\
    RepetitionWaveform, SequenceWaveform, make_compatible, MakeCompatibleWarning, DroppedMeasurementWarning, VolatileModificationWarning
from qupulse._program._loop import Loop, _make_compatible, _is_compatible, _CompatibilityLevel,\
    RepetitionWaveform, SequenceWaveform, make_compatible, MakeCompatibleWarning
from tests.pulses.sequencing_dummies import DummyWaveform
from qupulse.pulses.multi_channel_pulse_template import MultiChannelWaveform


class WaveformGenerator:
    def __init__(self, num_channels,
                 duration_generator=itertools.repeat(1),
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


@mock.patch.object(Loop, 'MAX_REPR_SIZE', 10000)
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

        with mock.patch.object(Loop, 'MAX_REPR_SIZE', 1):
            self.assertEqual(repr(tree), '...')

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

    def test_flatten_and_balance(self):
        """This test was written before Loop was a Comparable and works based on __repr__"""
        before = LoopTests.get_test_loop(lambda: DummyWaveform())
        before[1][0].encapsulate()

        after = before.copy_tree_structure()
        after.flatten_and_balance(2)

        wf_reprs = dict(zip(ascii_uppercase,
                            (repr(loop.waveform)
                             for loop in before.get_depth_first_iterator()
                             if loop.is_leaf())))

        before_repr = """\
LOOP 1 times:
  ->EXEC {A} 1 times
  ->LOOP 10 times:
      ->LOOP 1 times:
          ->EXEC {B} 50 times
  ->LOOP 17 times:
      ->LOOP 2 times:
          ->EXEC {C} 1 times
          ->EXEC {D} 1 times
      ->EXEC {E} 1 times
  ->LOOP 3 times:
      ->EXEC {F} 1 times
      ->EXEC {G} 1 times
  ->LOOP 4 times:
      ->LOOP 6 times:
          ->EXEC {H} 7 times
          ->EXEC {I} 8 times
      ->LOOP 9 times:
          ->EXEC {J} 10 times
          ->EXEC {K} 11 times""".format(**wf_reprs)
        self.assertEqual(repr(before), before_repr)

        expected_after_repr = """\
LOOP 1 times:
  ->LOOP 1 times:
      ->EXEC {A} 1 times
  ->LOOP 10 times:
      ->EXEC {B} 50 times
  ->LOOP 17 times:
      ->EXEC {C} 1 times
      ->EXEC {D} 1 times
      ->EXEC {C} 1 times
      ->EXEC {D} 1 times
      ->EXEC {E} 1 times
  ->LOOP 3 times:
      ->EXEC {F} 1 times
      ->EXEC {G} 1 times
  ->LOOP 6 times:
      ->EXEC {H} 7 times
      ->EXEC {I} 8 times
  ->LOOP 9 times:
      ->EXEC {J} 10 times
      ->EXEC {K} 11 times
  ->LOOP 6 times:
      ->EXEC {H} 7 times
      ->EXEC {I} 8 times
  ->LOOP 9 times:
      ->EXEC {J} 10 times
      ->EXEC {K} 11 times
  ->LOOP 6 times:
      ->EXEC {H} 7 times
      ->EXEC {I} 8 times
  ->LOOP 9 times:
      ->EXEC {J} 10 times
      ->EXEC {K} 11 times
  ->LOOP 6 times:
      ->EXEC {H} 7 times
      ->EXEC {I} 8 times
  ->LOOP 9 times:
      ->EXEC {J} 10 times
      ->EXEC {K} 11 times""".format(**wf_reprs)

        self.assertEqual(expected_after_repr, repr(after))

    def test_flatten_and_balance_comparison_based(self):
        wfs = [DummyWaveform(duration=i) for i in range(2)]

        root = Loop(children=[Loop(children=[
            Loop(waveform=wfs[0]),
            Loop(children=[Loop(waveform=wfs[1], repetition_count=2)])
        ])])

        expected = Loop(children=[
            Loop(waveform=wfs[0]),
            Loop(waveform=wfs[1], repetition_count=2)
        ])

        root.flatten_and_balance(1)
        self.assertEqual(root, expected)

    def test_unroll(self):
        wf = DummyWaveform(duration=1)
        wf2 = DummyWaveform(duration=2)
        wf3 = DummyWaveform(duration=3)
        root = Loop(waveform=wf)

        with self.assertRaisesRegex(RuntimeError, 'Leaves cannot be unrolled'):
            root.unroll()

        root = Loop(children=[Loop(waveform=wf),
                              Loop(children=[Loop(waveform=wf2), Loop(waveform=wf3)], repetition_count=2)])
        root.children[1].unroll()

        expected = Loop(children=[Loop(waveform=wf),
                                  Loop(waveform=wf2),
                                  Loop(waveform=wf3),
                                  Loop(waveform=wf2),
                                  Loop(waveform=wf3)])
        self.assertEqual(expected, root)

    def test_cleanup(self):
        wfs = [DummyWaveform(duration=i) for i in range(3)]

        root = Loop(children=[
            Loop(waveform=wfs[0]),
            Loop(waveform=None),
            Loop(children=[Loop(waveform=None)]),
            Loop(children=[Loop(waveform=wfs[1], repetition_count=2, measurements=[('m', 0, 1)])], repetition_count=3),
            Loop(children=[Loop(waveform=wfs[2], repetition_count=2)], repetition_count=3, measurements=[('n', 0, 1)])
        ])

        expected = Loop(children=[
            Loop(waveform=wfs[0]),
            Loop(waveform=wfs[1], repetition_count=6, measurements=[('m', 0, 1)]),
            Loop(children=[Loop(waveform=wfs[2], repetition_count=2)], repetition_count=3, measurements=[('n', 0, 1)])
        ])

        root.cleanup()

        self.assertEqual(expected, root)

    def test_cleanup_single_rep(self):
        wf = DummyWaveform(duration=1)
        measurements = [('n', 0, 1)]

        root = Loop(children=[Loop(waveform=wf, repetition_count=1)],
                    measurements=measurements, repetition_count=10)

        expected = Loop(waveform=wf, repetition_count=10, measurements=measurements)
        root.cleanup()
        self.assertEqual(expected, root)

    def test_cleanup_warnings(self):
        root = Loop(children=[
            Loop(measurements=[('m', 0, 1)])
        ])

        with self.assertWarnsRegex(DroppedMeasurementWarning, 'Dropping measurement'):
            root.cleanup()

        root = Loop(children=[
            Loop(measurements=[('m', 0, 1)], children=[Loop()])
        ])
        with self.assertWarnsRegex(DroppedMeasurementWarning, 'Dropping measurement since there is no waveform in children'):
            root.cleanup()


class ProgramWaveformCompatibilityTest(unittest.TestCase):
    def test_is_compatible_warnings(self):
        wf = DummyWaveform(duration=1)
        volatile_repetition_count = VolatileRepetitionCount(ExpressionScalar('x'),
                                                            DictScope.from_kwargs(x=3, volatile={'x'}))

        volatile_leaf = Loop(waveform=wf, repetition_count=volatile_repetition_count)
        with self.assertWarns(VolatileModificationWarning):
            self.assertEqual(_CompatibilityLevel.action_required, _is_compatible(volatile_leaf, min_len=3, quantum=1,
                                                                                 sample_rate=time_from_float(1.)))

        volatile_node = Loop(children=[Loop(waveform=wf)], repetition_count=volatile_repetition_count)
        with self.assertWarns(VolatileModificationWarning):
            self.assertEqual(_CompatibilityLevel.action_required, _is_compatible(volatile_node, min_len=3, quantum=1,
                                                                                 sample_rate=time_from_float(1.)))

    def test_is_compatible_incompatible(self):
        wf = DummyWaveform(duration=1.1)

        self.assertEqual(_is_compatible(Loop(waveform=wf), min_len=1, quantum=1, sample_rate=time_from_float(1.)),
                         _CompatibilityLevel.incompatible_fraction)

        self.assertEqual(_is_compatible(Loop(waveform=wf, repetition_count=10), min_len=20, quantum=1, sample_rate=time_from_float(1.)),
                         _CompatibilityLevel.incompatible_too_short)

        self.assertEqual(_is_compatible(Loop(waveform=wf, repetition_count=10), min_len=10, quantum=3, sample_rate=time_from_float(1.)),
                         _CompatibilityLevel.incompatible_quantum)

    def test_is_compatible_leaf(self):
        self.assertEqual(_is_compatible(Loop(waveform=DummyWaveform(duration=1.1), repetition_count=10),
                                        min_len=11, quantum=1, sample_rate=TimeType.from_float(1.)),
                         _CompatibilityLevel.action_required)

        self.assertEqual(_is_compatible(Loop(waveform=DummyWaveform(duration=1.1), repetition_count=10),
                                        min_len=11, quantum=1, sample_rate=TimeType.from_float(10.)),
                         _CompatibilityLevel.compatible)

    def test_is_compatible_node(self):
        program = Loop(children=[Loop(waveform=DummyWaveform(duration=1.5), repetition_count=2),
                                 Loop(waveform=DummyWaveform(duration=2.0))])

        self.assertEqual(_is_compatible(program, min_len=1, quantum=1, sample_rate=TimeType.from_float(2.)),
                         _CompatibilityLevel.compatible)

        self.assertEqual(_is_compatible(program, min_len=1, quantum=1, sample_rate=TimeType.from_float(1.)),
                         _CompatibilityLevel.action_required)

    def test_make_compatible_repetition_count(self):
        wf1 = DummyWaveform(duration=1.5)
        wf2 = DummyWaveform(duration=2.0)

        program = Loop(children=[Loop(waveform=wf1, repetition_count=2),
                                 Loop(waveform=wf2)])
        duration = program.duration
        _make_compatible(program, min_len=1, quantum=1, sample_rate=time_from_float(1.))
        self.assertEqual(program.duration, duration)

        wf2 = DummyWaveform(duration=2.5)
        program = Loop(children=[Loop(waveform=wf1, repetition_count=3),
                                 Loop(waveform=wf2)])
        duration = program.duration
        with self.assertWarns(MakeCompatibleWarning):
            make_compatible(program, minimal_waveform_length=1, waveform_quantum=1, sample_rate=time_from_float(1.))
        self.assertEqual(program.duration, duration)

        program = Loop(children=[Loop(waveform=wf1, repetition_count=3),
                                 Loop(waveform=wf2)], repetition_count=3)
        duration = program.duration
        _make_compatible(program, min_len=1, quantum=3, sample_rate=time_from_float(1.))
        self.assertEqual(program.duration, duration)

    def test_make_compatible_partial_unroll(self):
        wf1 = DummyWaveform(duration=1.5)
        wf2 = DummyWaveform(duration=2.0)

        program = Loop(children=[Loop(waveform=wf1, repetition_count=2),
                                 Loop(waveform=wf2)])

        _make_compatible(program, min_len=1, quantum=1, sample_rate=TimeType.from_float(1.))

        self.assertIsNone(program.waveform)
        self.assertEqual(len(program), 2)
        self.assertIsInstance(program[0].waveform, RepetitionWaveform)
        self.assertIs(program[0].waveform._body, wf1)
        self.assertEqual(program[0].waveform._repetition_count, 2)
        self.assertIs(program[1].waveform, wf2)

        program = Loop(children=[Loop(waveform=wf1, repetition_count=2),
                                 Loop(waveform=wf2)], repetition_count=2)
        _make_compatible(program, min_len=5, quantum=1, sample_rate=TimeType.from_float(1.))

        self.assertIsInstance(program.waveform, SequenceWaveform)
        self.assertEqual(list(program.children), [])
        self.assertEqual(program.repetition_count, 2)

        self.assertEqual(len(program.waveform._sequenced_waveforms), 2)
        self.assertIsInstance(program.waveform._sequenced_waveforms[0], RepetitionWaveform)
        self.assertIs(program.waveform._sequenced_waveforms[0]._body, wf1)
        self.assertEqual(program.waveform._sequenced_waveforms[0]._repetition_count, 2)
        self.assertIs(program.waveform._sequenced_waveforms[1], wf2)

    def test_make_compatible_complete_unroll(self):
        wf1 = DummyWaveform(duration=1.5)
        wf2 = DummyWaveform(duration=2.0)

        program = Loop(children=[Loop(waveform=wf1, repetition_count=2),
                                 Loop(waveform=wf2, repetition_count=1)], repetition_count=2)

        _make_compatible(program, min_len=5, quantum=10, sample_rate=TimeType.from_float(1.))

        self.assertIsInstance(program.waveform, RepetitionWaveform)
        self.assertEqual(list(program.children), [])
        self.assertEqual(program.repetition_count, 1)

        self.assertIsInstance(program.waveform, RepetitionWaveform)

        self.assertIsInstance(program.waveform._body, SequenceWaveform)
        body_wf = program.waveform._body
        self.assertEqual(len(body_wf._sequenced_waveforms), 2)
        self.assertIsInstance(body_wf._sequenced_waveforms[0], RepetitionWaveform)
        self.assertIs(body_wf._sequenced_waveforms[0]._body, wf1)
        self.assertEqual(body_wf._sequenced_waveforms[0]._repetition_count, 2)
        self.assertIs(body_wf._sequenced_waveforms[1], wf2)

    def test_make_compatible(self):
        program = Loop()
        pub_kwargs = dict(minimal_waveform_length=5,
                          waveform_quantum=10,
                          sample_rate=TimeType.from_float(1.))
        priv_kwargs = dict(min_len=5, quantum=10, sample_rate=TimeType.from_float(1.))

        with mock.patch('qupulse._program._loop._is_compatible',
                        return_value=_CompatibilityLevel.incompatible_too_short) as mocked:
            with self.assertRaisesRegex(ValueError, 'too short'):
                make_compatible(program, **pub_kwargs)
            mocked.assert_called_once_with(program, **priv_kwargs)

        with mock.patch('qupulse._program._loop._is_compatible',
                        return_value=_CompatibilityLevel.incompatible_fraction) as mocked:
            with self.assertRaisesRegex(ValueError, 'not an integer'):
                make_compatible(program, **pub_kwargs)
            mocked.assert_called_once_with(program, **priv_kwargs)

        with mock.patch('qupulse._program._loop._is_compatible',
                        return_value=_CompatibilityLevel.incompatible_quantum) as mocked:
            with self.assertRaisesRegex(ValueError, 'not a multiple of quantum'):
                make_compatible(program, **pub_kwargs)
            mocked.assert_called_once_with(program, **priv_kwargs)

        with mock.patch('qupulse._program._loop._is_compatible',
                        return_value=_CompatibilityLevel.action_required) as is_compat:
            with mock.patch('qupulse._program._loop._make_compatible') as make_compat:
                make_compatible(program, **pub_kwargs)

                is_compat.assert_called_once_with(program, **priv_kwargs)
                make_compat.assert_called_once_with(program, **priv_kwargs)
