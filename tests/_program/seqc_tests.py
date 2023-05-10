import unittest
from unittest import TestCase, mock
import time
from itertools import zip_longest, islice
import sys
import tempfile
import pathlib
import hashlib
import random

import numpy as np

from qupulse.expressions import ExpressionScalar
from qupulse.parameter_scope import DictScope

from qupulse._program._loop import Loop
from qupulse._program.waveforms import ConstantWaveform
from qupulse._program.seqc import BinaryWaveform, loop_to_seqc, WaveformPlayback, Repeat, SteppingRepeat, Scope,\
    to_node_clusters, find_sharable_waveforms, mark_sharable_waveforms, UserRegisterManager, HDAWGProgramManager,\
    UserRegister, WaveformFileSystem
from qupulse._program.volatile import VolatileRepetitionCount

from tests.pulses.sequencing_dummies import DummyWaveform

try:
    import zhinst
except ImportError:
    zhinst = None


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def dummy_loop_to_seqc(loop, **kwargs):
    return loop


class BinaryWaveformTest(unittest.TestCase):
    MAX_RATE = 14

    def test_dynamic_rate_reduction(self):

        ones = np.ones(2**(self.MAX_RATE + 2) * 3, np.uint16)

        for n in (2, 3, 5):
            self.assertEqual(BinaryWaveform(ones[:n * 16 * 3]).dynamic_rate(), 0, f"Reducing {n}")
        for n in (4, 6):
            self.assertEqual(BinaryWaveform(ones[:16 * n * 3]).dynamic_rate(), 1)

        irreducibles = [
            np.array([0, 0, 1, 1, 0, 1] * 16, dtype=np.uint16),
            np.array([0, 0, 0] * 16 + [0, 1, 0] + [0, 0, 0] * 15, dtype=np.uint16),
            np.array([0, 0, 0] * 16 + [1, 0, 0] + [0, 0, 0] * 15, dtype=np.uint16),
        ]
        for max_rate in range(self.MAX_RATE):
            for n in range(self.MAX_RATE):
                for irreducible in irreducibles:
                    data = np.tile(np.tile(irreducible.reshape(-1, 1, 3), (1, 2**n, 1)).ravel(), (16,))

                    dyn_n = BinaryWaveform(data).dynamic_rate(max_rate=max_rate)

                    self.assertEqual(min(max_rate, n), dyn_n)


def make_binary_waveform(waveform):
    if zhinst is None:
        # TODO: mock used function
        raise unittest.SkipTest("zhinst not present")

    if waveform.duration == 0:
        data = np.asarray(3 * [1, 2, 3, 4, 5], dtype=np.uint16)
        return (BinaryWaveform(data),)
    else:
        chs = sorted(waveform.defined_channels)
        t = np.arange(0., float(waveform.duration), 1.)

        sampled = [None if ch is None else waveform.get_sampled(ch, t)
                   for _, ch in zip_longest(range(6), take(6, chs), fillvalue=None)]
        ch1, ch2, *markers = sampled
        return (BinaryWaveform.from_sampled(ch1, ch2, markers),)


def _key_to_int(n: int, duration: int, defined_channels: frozenset):
    key_bytes = str((n, duration, sorted(defined_channels))).encode('ascii')
    key_int64 = int(hashlib.sha256(key_bytes).hexdigest()[:2*8], base=16) // 2
    return key_int64


def get_unique_wfs(n=10000, duration=32, defined_channels=frozenset(['A'])):
    if not hasattr(get_unique_wfs, 'cache'):
        get_unique_wfs.cache = {}

    key = (n, duration, defined_channels)

    if key not in get_unique_wfs.cache:
        # positive deterministic int64
        h = _key_to_int(n, duration, defined_channels)
        base = np.bitwise_xor(np.linspace(-h, h, num=duration + n, dtype=np.int64), h)
        base = base / np.max(np.abs(base))

        get_unique_wfs.cache[key] = [
            DummyWaveform(duration=duration, sample_output=base[idx:idx+duration],
                          defined_channels=defined_channels)
            for idx in range(n)
        ]
    return get_unique_wfs.cache[key]


def get_constant_unique_wfs(n=10000, duration=192, defined_channels=frozenset(['A'])):
    if not hasattr(get_unique_wfs, 'cache'):
        get_unique_wfs.cache = {}

    key = (n, duration, defined_channels)

    if key not in get_unique_wfs.cache:
        bit_gen = np.random.PCG64(_key_to_int(n, duration, defined_channels))
        rng = np.random.Generator(bit_gen)

        random_values = rng.random(size=(n, len(defined_channels)))

        sorted_channels = sorted(defined_channels)
        get_unique_wfs.cache[key] = [
            ConstantWaveform.from_mapping(duration, {ch: ch_value
                                                     for ch, ch_value in zip(sorted_channels, wf_values)})
            for wf_values in random_values
        ]
    return get_unique_wfs.cache[key]


def complex_program_as_loop(unique_wfs, wf_same):
    root = Loop(repetition_count=12)

    for wf_unique in unique_wfs:
        root.append_child(children=[Loop(repetition_count=42, waveform=wf_unique),
                                    Loop(repetition_count=98, waveform=wf_same)],
                          repetition_count=10)

    root.append_child(waveform=unique_wfs[0], repetition_count=21)
    root.append_child(waveform=wf_same, repetition_count=23)

    volatile_repetition = VolatileRepetitionCount(ExpressionScalar('n + 4'),
                                                  DictScope.from_kwargs(n=3, volatile={'n'}))
    root.append_child(waveform=wf_same, repetition_count=volatile_repetition)

    return root


def complex_program_as_seqc(unique_wfs, wf_same):
    return Repeat(12,
           Scope([
               SteppingRepeat([
                   Repeat(repetition_count=10, scope=Scope([
                       Repeat(42, WaveformPlayback(make_binary_waveform(unique_wf))),
                       Repeat(98, WaveformPlayback(make_binary_waveform(wf_same), shared=True)),
                   ]))
                   for unique_wf in unique_wfs
               ]),
               Repeat(21, WaveformPlayback(make_binary_waveform(unique_wfs[0]))),
               Repeat(23, WaveformPlayback(make_binary_waveform(wf_same))),
               Repeat('test_14', WaveformPlayback(make_binary_waveform(wf_same)))
           ])
           )


class DummyWfManager:
    def __init__(self):
        self.shared = {}
        self.concatenated = []

    def request_shared(self, wf):
        return self.shared.setdefault(wf, len(self.shared) + 1)

    def request_concatenated(self, wf):
        self.concatenated.append(wf)
        return 0


class WaveformFileSystemTests(TestCase):
    def setUp(self) -> None:
        clients = [mock.Mock(), mock.Mock()]
        bin_waveforms = [mock.Mock(), mock.Mock(), mock.Mock()]
        table_data = [np.ones(1, dtype=np.uint16) * i for i, _ in enumerate(bin_waveforms)]
        for bin_wf, tab in zip(bin_waveforms, table_data):
            bin_wf.to_csv_compatible_table.return_value = tab

        self.temp_dir = tempfile.TemporaryDirectory()
        self.table_data = table_data
        self.clients = clients
        self.waveforms = [
            {'0': bin_waveforms[0], '1': bin_waveforms[1]},
            {'1': bin_waveforms[1], '2': bin_waveforms[2]}
        ]
        self.fs = WaveformFileSystem(pathlib.Path(self.temp_dir.name))

    def read_files(self) -> dict:
        return {
            p.name: p.read_text().strip() for p in self.fs._path.iterdir()
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_pub_sync(self):
        with mock.patch.object(self.fs, '_sync') as mock_sync:
            self.fs.sync(self.clients[0], self.waveforms[0], hallo=0)
            mock_sync.assert_called_once_with(hallo=0)

            self.assertEqual({id(self.clients[0]): self.waveforms[0]}, self.fs._required)

    def test_sync(self):
        self.fs.sync(self.clients[0], self.waveforms[0])
        self.assertEqual({'0': '0', '1': '1'}, self.read_files())

        self.fs.sync(self.clients[0], self.waveforms[1])
        self.assertEqual({'2': '2', '1': '1'}, self.read_files())

        self.fs.sync(self.clients[1], self.waveforms[0])
        self.assertEqual({'2': '2', '1': '1', '0': '0'}, self.read_files())

    def test_sync_write_all(self):
        self.fs.sync(self.clients[0], self.waveforms[0])
        self.assertEqual({'0': '0', '1': '1'}, self.read_files())

        self.table_data[0][:] = 7
        self.fs.sync(self.clients[0], self.waveforms[0])
        self.assertEqual({'0': '0', '1': '1'}, self.read_files())

        self.fs.sync(self.clients[0], self.waveforms[0], write_all=True)
        self.assertEqual({'0': '7', '1': '1'}, self.read_files())

    def test_sync_no_delete(self):
        self.fs.sync(self.clients[0], self.waveforms[0])
        self.assertEqual({'0': '0', '1': '1'}, self.read_files())

        self.fs.sync(self.clients[0], self.waveforms[1], delete=False)
        self.assertEqual({'2': '2', '1': '1', '0': '0'}, self.read_files())


class SEQCNodeTests(TestCase):
    """Test everything besides source code generation"""
    @unittest.skipIf(zhinst is None, "test requires zhinst")
    def test_visit_nodes(self):
        wf, wf_shared = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(2)))
        wf_shared.shared = True

        waveform_manager = mock.Mock(wraps=DummyWfManager())
        wf._visit_nodes(waveform_manager)
        waveform_manager.request_concatenated.assert_called_once_with(wf.waveform)

        waveform_manager = mock.Mock(wraps=DummyWfManager())
        wf_shared._visit_nodes(waveform_manager)
        waveform_manager.request_concatenated.assert_not_called()

        scope = Scope([mock.Mock(wraps=wf), mock.Mock(wraps=wf_shared)])
        scope._visit_nodes(waveform_manager)
        scope.nodes[0]._visit_nodes.assert_called_once_with(waveform_manager)
        scope.nodes[1]._visit_nodes.assert_called_once_with(waveform_manager)
        waveform_manager.request_concatenated.assert_called_once_with(wf.waveform)

        waveform_manager = mock.Mock(wraps=DummyWfManager())
        repeat = Repeat(12, mock.Mock(wraps=wf))
        repeat._visit_nodes(waveform_manager)
        repeat.scope._visit_nodes.assert_called_once_with(waveform_manager)
        waveform_manager.request_concatenated.assert_called_once_with(wf.waveform)

        waveform_manager = mock.Mock(wraps=DummyWfManager())
        stepping_repeat = SteppingRepeat([mock.Mock(wraps=wf), mock.Mock(wraps=wf), mock.Mock(wraps=wf)])
        stepping_repeat._visit_nodes(waveform_manager)
        for node in stepping_repeat.node_cluster:
            node._visit_nodes.assert_called_once_with(waveform_manager)

    @unittest.skipIf(zhinst is None, "test requires zhinst")
    def test_same_stepping(self):
        wf1, wf2 = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(2, 32)))
        wf3, wf_shared = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(2, 64)))
        wf_shared.shared = True

        scope1 = Scope([wf1, wf1, wf2])
        scope2 = Scope([wf1, wf2, wf2])
        scope3 = Scope([wf1, wf2, wf3])
        scope4 = Scope([wf1, wf2, wf2, wf2])

        repeat1 = Repeat(13, wf1)
        repeat2 = Repeat(13, wf2)
        repeat3 = Repeat(15, wf2)
        repeat4 = Repeat(13, wf3)

        stepping_repeat1 = SteppingRepeat([wf1, wf1, wf2])
        stepping_repeat2 = SteppingRepeat([wf2, wf2, wf2])
        stepping_repeat3 = SteppingRepeat([wf3, wf3, wf3])
        stepping_repeat4 = SteppingRepeat([wf1, wf1, wf2, wf1])

        self.assertTrue(wf1.same_stepping(wf1))
        self.assertTrue(wf1.same_stepping(wf2))
        self.assertFalse(wf1.same_stepping(wf3))
        self.assertFalse(wf3.same_stepping(wf_shared))
        self.assertFalse(wf_shared.same_stepping(wf3))

        self.assertFalse(scope1.same_stepping(wf1))
        self.assertTrue(scope1.same_stepping(scope2))
        self.assertFalse(scope1.same_stepping(scope3))
        self.assertFalse(scope1.same_stepping(scope4))

        self.assertFalse(repeat1.same_stepping(scope1))
        self.assertTrue(repeat1.same_stepping(repeat2))
        self.assertFalse(repeat1.same_stepping(repeat3))
        self.assertFalse(repeat1.same_stepping(repeat4))

        self.assertFalse(stepping_repeat1.same_stepping(scope1))
        self.assertTrue(stepping_repeat1.same_stepping(stepping_repeat2))
        self.assertFalse(stepping_repeat1.same_stepping(stepping_repeat3))
        self.assertFalse(stepping_repeat1.same_stepping(stepping_repeat4))

    @unittest.skipIf(zhinst is None, "test requires zhinst")
    def test_iter_waveform_playback(self):
        wf1, wf2 = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(2, 32)))
        wf3, wf_shared = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(2, 64)))

        for wf in (wf1, wf2, wf3, wf_shared):
            pb, = wf.iter_waveform_playbacks()
            self.assertIs(pb, wf)

        repeat = Repeat(13, wf1)
        self.assertEqual(list(repeat.iter_waveform_playbacks()), [wf1])

        scope = Scope([wf1, repeat, wf2, wf3, wf_shared])
        self.assertEqual(list(scope.iter_waveform_playbacks()), [wf1, wf1, wf2, wf3, wf_shared])

        stepping_repeat = SteppingRepeat([wf1, repeat, wf2, wf3, wf_shared])
        self.assertEqual(list(stepping_repeat.iter_waveform_playbacks()), [wf1, wf1, wf2, wf3, wf_shared])

    @unittest.skipIf(zhinst is None, "test requires zhinst")
    def test_get_single_indexed_playback(self):
        wf1, wf_shared = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(2, 32)))
        wf_shared.shared = True
        self.assertIs(wf1._get_single_indexed_playback(), wf1)
        self.assertIsNone(wf_shared._get_single_indexed_playback())

        self.assertIs(Scope([wf1, wf_shared])._get_single_indexed_playback(), wf1)
        self.assertIsNone(Scope([wf1, wf_shared, wf1])._get_single_indexed_playback(), wf1)

    def test_get_position_advance_strategy(self):
        node = mock.Mock()
        node.samples.return_value = 0
        node._get_single_indexed_playback.return_value.samples.return_value = 128
        repeat = Repeat(10, node)

        # no samples at all
        self.assertIs(repeat._get_position_advance_strategy(), repeat._AdvanceStrategy.IGNORE)
        node.samples.assert_called_once_with()
        node._get_single_indexed_playback.assert_not_called()

        node.reset_mock()
        node.samples.return_value = 64

        # samples do differ
        self.assertIs(repeat._get_position_advance_strategy(), repeat._AdvanceStrategy.INITIAL_RESET)
        node.samples.assert_called_once_with()
        node._get_single_indexed_playback.assert_called_once_with()
        node._get_single_indexed_playback.return_value.samples.assert_called_once_with()

        node.reset_mock()
        node.samples.return_value = 128

        # samples are the same
        self.assertIs(repeat._get_position_advance_strategy(), repeat._AdvanceStrategy.POST_ADVANCE)
        node.samples.assert_called_once_with()
        node._get_single_indexed_playback.assert_called_once_with()
        node._get_single_indexed_playback.return_value.samples.assert_called_once_with()

        node.reset_mock()
        node._get_single_indexed_playback.return_value = None
        # multiple indexed playbacks
        self.assertIs(repeat._get_position_advance_strategy(), repeat._AdvanceStrategy.INITIAL_RESET)
        node.samples.assert_called_once_with()
        node._get_single_indexed_playback.assert_called_once_with()


@unittest.skipIf(zhinst is None, "test requires zhinst")
class LoopToSEQCTranslationTests(TestCase):
    def test_loop_to_seqc_leaf(self):
        """Test the translation of leaves"""
        # we use None because it is not used in this test
        user_registers = None

        wf = DummyWaveform(duration=32, sample_output=lambda x: np.sin(x))
        loop = Loop(waveform=wf)

        # with wrapping repetition
        loop.repetition_count = 15
        waveform_to_bin = mock.Mock(wraps=make_binary_waveform)
        expected = Repeat(loop.repetition_count, WaveformPlayback(waveform=make_binary_waveform(wf)))
        result = loop_to_seqc(loop, 1, 1, waveform_to_bin, user_registers=user_registers)
        waveform_to_bin.assert_called_once_with(wf)
        self.assertEqual(expected, result)

        # without wrapping repetition
        loop.repetition_count = 1
        waveform_to_bin = mock.Mock(wraps=make_binary_waveform)
        expected = WaveformPlayback(waveform=make_binary_waveform(wf))
        result = loop_to_seqc(loop, 1, 1, waveform_to_bin, user_registers=user_registers)
        waveform_to_bin.assert_called_once_with(wf)
        self.assertEqual(expected, result)

    def test_loop_to_seqc_len_1(self):
        """Test the translation of loops with len(loop) == 1"""
        # we use None because it is not used in this test
        user_registers = None

        loop = Loop(children=[Loop()])
        waveform_to_bin = mock.Mock(wraps=make_binary_waveform)
        loop_to_seqc_kwargs = dict(min_repetitions_for_for_loop=2,
                                   min_repetitions_for_shared_wf=3,
                                   waveform_to_bin=waveform_to_bin,
                                   user_registers=user_registers)

        expected = 'asdf'
        with mock.patch('qupulse._program.seqc.loop_to_seqc', return_value=expected) as mocked_loop_to_seqc:
            result = loop_to_seqc(loop, **loop_to_seqc_kwargs)
            self.assertEqual(result, expected)
            mocked_loop_to_seqc.assert_called_once_with(loop[0], **loop_to_seqc_kwargs)

        loop.repetition_count = 14
        expected = Repeat(14, 'asdfg')
        with mock.patch('qupulse._program.seqc.loop_to_seqc', return_value=expected.scope) as mocked_loop_to_seqc:
            result = loop_to_seqc(loop, **loop_to_seqc_kwargs)
            self.assertEqual(result, expected)
            mocked_loop_to_seqc.assert_called_once_with(loop[0], **loop_to_seqc_kwargs)

        waveform_to_bin.assert_not_called()

    def test_to_node_clusters(self):
        """Test cluster generation"""
        wf1, wf2 = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(2, 32)))
        wf3, wf_shared = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(2, 64)))

        loop_to_seqc_kwargs = {'my': 'kwargs'}

        loops = [wf1, wf2, wf1, wf1, wf3, wf1, wf1, wf1, wf3, wf1, wf3, wf1, wf3]
        expected_calls = [mock.call(loop, **loop_to_seqc_kwargs) for loop in loops]
        expected_result = [[wf1, wf2, wf1, wf1], [wf3], [wf1, wf1, wf1], [Scope([wf3, wf1]), Scope([wf3, wf1])], [wf3]]

        with mock.patch('qupulse._program.seqc.loop_to_seqc', wraps=dummy_loop_to_seqc) as mock_loop_to_seqc:
            result = to_node_clusters(loops, loop_to_seqc_kwargs)
            self.assertEqual(mock_loop_to_seqc.mock_calls, expected_calls)
        self.assertEqual(expected_result, result)

    def test_to_node_clusters_crash(self):
        wf1 = WaveformPlayback(make_binary_waveform(*get_unique_wfs(1, 32)))
        wf2 = WaveformPlayback(make_binary_waveform(*get_unique_wfs(1, 64)))
        wf3 = WaveformPlayback(make_binary_waveform(*get_unique_wfs(1, 128)))
        wf4 = WaveformPlayback(make_binary_waveform(*get_unique_wfs(1, 256)))

        loop_to_seqc_kwargs = {'my': 'kwargs'}

        loops = [wf1, wf2, wf3] * 3 + [wf1] + [wf2, wf4] * 3 + [wf1]
        with mock.patch('qupulse._program.seqc.loop_to_seqc', wraps=dummy_loop_to_seqc) as mock_loop_to_seqc:
            result = to_node_clusters(loops, loop_to_seqc_kwargs)
        expected_result = [[Scope([wf1, wf2, wf3])]*3, [wf1], [Scope([wf2, wf4])]*3, [wf1]]
        self.assertEqual(expected_result, result)

    def test_find_sharable_waveforms(self):
        wf1, wf2 = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(2, 32)))
        wf3, wf_shared = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(2, 64)))

        scope1 = Scope([wf1, wf1, wf_shared, wf1])
        scope2 = Scope([wf1, wf2, wf_shared, wf2])
        scope3 = Scope([wf2, wf2, wf_shared, wf3])
        scope4 = Scope([wf2, wf2, wf3,       wf3])

        self.assertIsNone(find_sharable_waveforms([scope1, scope2, scope3, scope4]))

        shareable = find_sharable_waveforms([scope1, scope2, scope3])
        self.assertEqual([False, False, True, False], shareable)

    def test_mark_sharable_waveforms(self):
        shareable = [False, False, True, False]
        
        pb_gen = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(12, 32)))
        
        nodes = [Scope([mock.Mock(wraps=next(pb_gen)), mock.Mock(wraps=next(pb_gen)), mock.Mock(wraps=next(pb_gen)), mock.Mock(wraps=next(pb_gen))]),
                 Scope([mock.Mock(wraps=next(pb_gen)), mock.Mock(wraps=next(pb_gen)), mock.Mock(wraps=next(pb_gen)), mock.Mock(wraps=next(pb_gen))]),
                 Scope([mock.Mock(wraps=next(pb_gen)), mock.Mock(wraps=next(pb_gen)), mock.Mock(wraps=next(pb_gen)), mock.Mock(wraps=next(pb_gen))])]

        mocks = [mock.Mock(wraps=scope) for scope in nodes]

        mark_sharable_waveforms(mocks, shareable)

        for mock_scope, scope in zip(mocks, nodes):
            mock_scope.iter_waveform_playbacks.assert_called_once_with()
            m1, m2, m3, m4 = scope.nodes
            self.assertIsInstance(m1.shared, mock.Mock)
            m1.iter_waveform_playbacks.assert_called_once_with()
            self.assertIsInstance(m2.shared, mock.Mock)
            m2.iter_waveform_playbacks.assert_called_once_with()
            self.assertTrue(m3.shared)
            m3.iter_waveform_playbacks.assert_called_once_with()
            self.assertIsInstance(m4.shared, mock.Mock)
            m4.iter_waveform_playbacks.assert_called_once_with()

    def test_loop_to_seqc_cluster_handling(self):
        """Test handling of clusters"""

        # we use None because it is not used in this test
        user_registers = None

        with self.assertRaises(AssertionError):
            loop_to_seqc(Loop(repetition_count=12, children=[Loop()]),
                         min_repetitions_for_for_loop=3, min_repetitions_for_shared_wf=2,
                         waveform_to_bin=make_binary_waveform, user_registers=user_registers)

        loop_to_seqc_kwargs = dict(min_repetitions_for_for_loop=3,
                                   min_repetitions_for_shared_wf=4,
                                   waveform_to_bin=make_binary_waveform, user_registers=user_registers)

        wf_same = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(100000, 32)))
        wf_sep, = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(1, 64)))

        node_clusters = [take(2, wf_same), [wf_sep],
                         take(3, wf_same), [wf_sep],
                         take(4, wf_same), take(4, wf_same)]
        root = Loop(repetition_count=12, children=[Loop() for _ in range(2 + 1 + 3 + 1 + 4 + 1 + 4)])

        expected = Repeat(12, Scope([
            *node_clusters[0],
            wf_sep,
            SteppingRepeat(node_clusters[2]),
            wf_sep,
            SteppingRepeat(node_clusters[4]),
            SteppingRepeat(node_clusters[5])
        ]))

        def dummy_find_sharable_waveforms(cluster):
            if cluster is node_clusters[4]:
                return [True]
            else:
                return None

        p1 = mock.patch('qupulse._program.seqc.to_node_clusters', return_value=node_clusters)
        p2 = mock.patch('qupulse._program.seqc.find_sharable_waveforms', wraps=dummy_find_sharable_waveforms)
        p3 = mock.patch('qupulse._program.seqc.mark_sharable_waveforms')

        with p1 as to_node_clusters_mock, p2 as find_share_mock, p3 as mark_share_mock:
            result = loop_to_seqc(root, **loop_to_seqc_kwargs)
            self.assertEqual(expected, result)

            to_node_clusters_mock.assert_called_once_with(root, loop_to_seqc_kwargs)
            self.assertEqual(find_share_mock.mock_calls,
                             [mock.call(node_clusters[4]), mock.call(node_clusters[5])])
            mark_share_mock.assert_called_once_with(node_clusters[4], [True])

    def test_program_translation(self):
        """Integration test"""
        user_registers = UserRegisterManager(range(14, 15), 'test_{register}')

        unique_wfs = get_unique_wfs()
        same_wf = DummyWaveform(duration=32, sample_output=np.ones(32))
        root = complex_program_as_loop(unique_wfs, wf_same=same_wf)

        t0 = time.perf_counter()

        seqc = loop_to_seqc(root, 50, 100, make_binary_waveform, user_registers=user_registers)

        t1 = time.perf_counter()
        print('took', t1 - t0, 's')

        expected = complex_program_as_seqc(unique_wfs, wf_same=same_wf)
        self.assertEqual(expected, seqc)


@unittest.skipIf(zhinst is None, "test requires zhinst")
class SEQCToCodeTranslationTests(TestCase):
    def setUp(self) -> None:
        self.line_prefix = '   '
        self.node_name_generator = map(str, range(10000000000000000000))
        self.pos_var_name = 'foo'
        self.waveform_manager = DummyWfManager()

    def test_shared_playback(self):
        wf, = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(1, 32)))
        wf.shared = True

        expected = ['   playWave(1);']
        result = list(wf.to_source_code(self.waveform_manager, self.node_name_generator, self.line_prefix, self.pos_var_name, True))
        self.assertEqual(expected, result)

    def test_indexed_playback(self):
        wf, = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(1, 32)))

        expected = ['   playWaveIndexed(0, foo, 32); foo = foo + 32;']
        result = list(
            wf.to_source_code(self.waveform_manager, self.node_name_generator, self.line_prefix, self.pos_var_name,
                              True))
        self.assertEqual(expected, result)

        expected = ['   playWaveIndexed(0, foo, 32);' + wf.ADVANCE_DISABLED_COMMENT]
        result = list(
            wf.to_source_code(self.waveform_manager, self.node_name_generator, self.line_prefix, self.pos_var_name,
                              False))
        self.assertEqual(expected, result)

    def test_scope(self):
        nodes = [mock.Mock(), mock.Mock(), mock.Mock()]
        for idx, node in enumerate(nodes):
            node.to_source_code = mock.Mock(return_value=map(str, [idx + 100, idx + 200]))

        scope = Scope(nodes)
        expected = ['100', '200', '101', '201', '102', '202']
        result = list(scope.to_source_code(self.waveform_manager, self.node_name_generator,
                                           self.line_prefix, self.pos_var_name, False))
        self.assertEqual(expected, result)
        for node in nodes:
            node.to_source_code.assert_called_once_with(self.waveform_manager,
                                                        line_prefix=self.line_prefix,
                                                        pos_var_name=self.pos_var_name,
                                                        node_name_generator=self.node_name_generator,
                                                        advance_pos_var=False)

    def test_stepped_repeat(self):
        nodes = [mock.Mock(), mock.Mock(), mock.Mock()]
        for idx, node in enumerate(nodes):
            node.to_source_code = mock.Mock(return_value=map(str, [idx + 100, idx + 200]))

        stepping_repeat = SteppingRepeat(nodes)

        body_prefix = self.line_prefix + stepping_repeat.INDENTATION
        expected = [
            '   repeat(3) {' + stepping_repeat.STEPPING_REPEAT_COMMENT,
            '100',
            '200',
            '   }'
        ]
        result = list(stepping_repeat.to_source_code(self.waveform_manager, self.node_name_generator,
                                                     self.line_prefix, self.pos_var_name, False))
        self.assertEqual(expected, result)
        nodes[0].to_source_code.assert_called_once_with(self.waveform_manager,
                                                        line_prefix=body_prefix,
                                                        pos_var_name=self.pos_var_name,
                                                        node_name_generator=self.node_name_generator,
                                                        advance_pos_var=False)
        nodes[1].to_source_code.assert_not_called()
        nodes[2].to_source_code.assert_not_called()
        nodes[0]._visit_nodes.assert_not_called()
        nodes[1]._visit_nodes.assert_called_once_with(self.waveform_manager)
        nodes[2]._visit_nodes.assert_called_once_with(self.waveform_manager)

    def test_repeat(self):
        node = mock.Mock()
        node.to_source_code = mock.Mock(return_value=['asd', 'jkl'])
        node._get_single_indexed_playback = mock.Mock(return_value=None)
        node.samples = mock.Mock(return_value=64)

        repeat = Repeat(12, node)

        body_prefix = self.line_prefix + repeat.INDENTATION
        expected = ['   var init_pos_0 = foo;',
                    '   repeat(12) {',
                    '     foo = init_pos_0;',
                    'asd', 'jkl', '   }']

        result = list(repeat.to_source_code(self.waveform_manager,
                                            node_name_generator=self.node_name_generator,
                                            line_prefix=self.line_prefix, pos_var_name=self.pos_var_name,
                                            advance_pos_var=True))
        self.assertEqual(expected, result)
        node.to_source_code.assert_called_once_with(self.waveform_manager, node_name_generator=self.node_name_generator,
                                                    line_prefix=body_prefix,
                                                    pos_var_name=self.pos_var_name,
                                                    advance_pos_var=True)
        node._get_single_indexed_playback.assert_called_once_with()
        node.samples.assert_called_once_with()

    def test_repeat_detect_no_advance(self):
        node = mock.Mock()
        node.to_source_code = mock.Mock(return_value=['asd', 'jkl'])
        node._get_single_indexed_playback = mock.Mock(return_value=None)
        node.samples = mock.Mock(return_value=0)

        repeat = Repeat(12, node)
        body_prefix = self.line_prefix + repeat.INDENTATION

        expected = ['   repeat(12) {',
                    'asd', 'jkl', '   }']
        result_no_advance = list(repeat.to_source_code(self.waveform_manager,
                                                       node_name_generator=self.node_name_generator,
                                                       line_prefix=self.line_prefix, pos_var_name=self.pos_var_name,
                                                       advance_pos_var=True))
        self.assertEqual(expected, result_no_advance)
        node.to_source_code.assert_called_once_with(self.waveform_manager, node_name_generator=self.node_name_generator,
                                                    line_prefix=body_prefix,
                                                    pos_var_name=self.pos_var_name,
                                                    advance_pos_var=False)
        node._get_single_indexed_playback.assert_not_called()
        node.samples.assert_called_once_with()

    def test_repeat_extern_no_advance(self):
        node = mock.Mock()
        node.to_source_code = mock.Mock(return_value=['asd', 'jkl'])
        node._get_single_indexed_playback = mock.Mock(return_value=None)
        node.samples = mock.Mock(return_value=64)

        repeat = Repeat(12, node)

        body_prefix = self.line_prefix + repeat.INDENTATION

        expected = ['   repeat(12) {',
                    'asd', 'jkl', '   }']
        result_no_advance = list(repeat.to_source_code(self.waveform_manager,
                                                       node_name_generator=self.node_name_generator,
                                                       line_prefix=self.line_prefix, pos_var_name=self.pos_var_name,
                                                       advance_pos_var=False))
        self.assertEqual(expected, result_no_advance)
        node.to_source_code.assert_called_once_with(self.waveform_manager, node_name_generator=self.node_name_generator,
                                                    line_prefix=body_prefix,
                                                    pos_var_name=self.pos_var_name,
                                                    advance_pos_var=False)
        node._get_single_indexed_playback.assert_not_called()
        node.samples.assert_not_called()

    def test_program_to_code_translation(self):
        """Integration test"""
        unique_wfs = get_unique_wfs()
        same_wf = DummyWaveform(duration=48, sample_output=np.ones(48))
        seqc_nodes = complex_program_as_seqc(unique_wfs, wf_same=same_wf)

        wf_manager = DummyWfManager()
        def node_name_gen():
            for i in range(100):
                yield str(i)

        seqc_code = '\n'.join(seqc_nodes.to_source_code(wf_manager,
                                                        line_prefix='',
                                                        pos_var_name='pos',
                                                        node_name_generator=node_name_gen()))
        # this is just copied from the result...
        expected = """var init_pos_0 = pos;
repeat(12) {
  pos = init_pos_0;
  repeat(10000) { // stepping repeat
    repeat(10) {
      repeat(42) {
        playWaveIndexed(0, pos, 32); // advance disabled do to parent repetition
      }
      repeat(98) {
        playWave(1);
      }
    }
    pos = pos + 32;
  }
  repeat(21) {
    playWaveIndexed(0, pos, 32); // advance disabled do to parent repetition
  }
  pos = pos + 32;
  repeat(23) {
    playWaveIndexed(0, pos, 48); // advance disabled do to parent repetition
  }
  pos = pos + 48;
  var idx_1;
  for(idx_1 = 0; idx_1 < test_14; idx_1 = idx_1 + 1) {
    playWaveIndexed(0, pos, 48); // advance disabled do to parent repetition
  }
  pos = pos + 48;
}"""
        self.assertEqual(expected, seqc_code)


class UserRegisterTest(unittest.TestCase):
    def test_conversions(self):
        reg = UserRegister(zero_based_value=3)
        self.assertEqual(3, reg.to_seqc())
        self.assertEqual(3, reg.to_labone())
        self.assertEqual(4, reg.to_web_interface())

        reg = UserRegister(one_based_value=4)
        self.assertEqual(3, reg.to_seqc())
        self.assertEqual(3, reg.to_labone())
        self.assertEqual(4, reg.to_web_interface())

        self.assertEqual(reg, UserRegister.from_seqc(3))
        self.assertEqual(reg, UserRegister.from_labone(3))
        self.assertEqual(reg, UserRegister.from_web_interface(4))

    def test_formatting(self):
        reg = UserRegister.from_seqc(3)

        with self.assertRaises(ValueError):
            '{}'.format(reg)

        self.assertEqual('3', '{:seqc}'.format(reg))
        self.assertEqual('4', '{:web}'.format(reg))
        self.assertEqual('UserRegister(zero_based_value=3)', repr(reg))
        self.assertEqual(repr(reg), '{:r}'.format(reg))


class UserRegisterManagerTest(unittest.TestCase):
    def test_require(self):
        manager = UserRegisterManager([7, 8, 9], 'test{register}')

        required = [manager.request(0), manager.request(1), manager.request(2)]

        self.assertEqual({'test7', 'test8', 'test9'}, set(required))
        self.assertEqual(required[1], manager.request(1))

        with self.assertRaisesRegex(ValueError, "No register"):
            manager.request(3)


class HDAWGProgramManagerTest(unittest.TestCase):
    @unittest.skipIf(sys.version_info.minor < 6, "This test requires dict to be ordered.")
    def test_full_run(self):
        defined_channels = frozenset(['A', 'B', 'C'])

        unique_n = 1000
        unique_duration = 32

        unique_wfs = get_unique_wfs(n=unique_n, duration=unique_duration, defined_channels=defined_channels)
        same_wf = DummyWaveform(duration=48, sample_output=np.ones(48), defined_channels=defined_channels)

        channels = ('A', 'B')
        markers = ('C', None, 'A', None)
        amplitudes = (1., 1.)
        offsets = (0., 0.)
        volatage_transformations = (lambda x: x, lambda x: x)
        sample_rate = 1

        root = complex_program_as_loop(unique_wfs, wf_same=same_wf)
        seqc_nodes = complex_program_as_seqc(unique_wfs, wf_same=same_wf)

        manager = HDAWGProgramManager()

        manager.add_program('test', root, channels, markers, amplitudes, offsets, volatage_transformations, sample_rate)

        # 0: Program selection
        # 1: Trigger
        self.assertEqual({UserRegister(zero_based_value=2): 7}, manager.get_register_values('test'))
        seqc_program = manager.to_seqc_program()
        expected_program = """const PROG_SEL_REGISTER = 0;
const TRIGGER_REGISTER = 1;
const TRIGGER_RESET_MASK = 0b10000000000000000000000000000000;
const PROG_SEL_NONE = 0;
const NO_RESET_MASK = 0b10000000000000000000000000000000;
const PLAYBACK_FINISHED_MASK = 0b1000000000000000000000000000000;
const PROG_SEL_MASK = 0b111111111111111111111111111111;
const INVERTED_PROG_SEL_MASK = 0b11000000000000000000000000000000;
const IDLE_WAIT_CYCLES = 300;
wave test_concatenated_waveform_0 = "c45d955d9dc472d46bf74f7eb9ae2ed4d159adea7d6fe9ce3f48c95423535333";
wave test_shared_waveform_121f5c6e8822793b3836fb3098fa4591b91d4c205cc2d8afd01ee1bf6956e518 = "121f5c6e8822793b3836fb3098fa4591b91d4c205cc2d8afd01ee1bf6956e518";

// function used by manually triggered programs
void waitForSoftwareTrigger() {
  while (true) {
    var trigger_register = getUserReg(TRIGGER_REGISTER);
    if (trigger_register & TRIGGER_RESET_MASK) setUserReg(TRIGGER_REGISTER, 0);
    if (trigger_register) return;
  }
}


// program definitions
void test_function() {
  var pos = 0;
  var user_reg_2 = getUserReg(2);
  waitForSoftwareTrigger();
  var init_pos_1 = pos;
  repeat(12) {
    pos = init_pos_1;
    repeat(1000) { // stepping repeat
      repeat(10) {
        repeat(42) {
          playWaveIndexed(test_concatenated_waveform_0, pos, 32); // advance disabled do to parent repetition
        }
        repeat(98) {
          playWave(test_shared_waveform_121f5c6e8822793b3836fb3098fa4591b91d4c205cc2d8afd01ee1bf6956e518);
        }
      }
      pos = pos + 32;
    }
    repeat(21) {
      playWaveIndexed(test_concatenated_waveform_0, pos, 32); // advance disabled do to parent repetition
    }
    pos = pos + 32;
    repeat(23) {
      playWaveIndexed(test_concatenated_waveform_0, pos, 48); // advance disabled do to parent repetition
    }
    pos = pos + 48;
    var idx_2;
    for(idx_2 = 0; idx_2 < user_reg_2; idx_2 = idx_2 + 1) {
      playWaveIndexed(test_concatenated_waveform_0, pos, 48); // advance disabled do to parent repetition
    }
    pos = pos + 48;
  }
}

// Declare and initialize global variables
// Selected program index (0 -> None)
var prog_sel = 0;

// Value that gets written back to program selection register.
// Used to signal that at least one program was played completely.
var new_prog_sel = 0;

// Is OR'ed to new_prog_sel.
// Set to PLAYBACK_FINISHED_MASK if a program was played completely.
var playback_finished = 0;


// runtime block
while (true) {
  // read program selection value
  prog_sel = getUserReg(PROG_SEL_REGISTER);
  
  // calculate value to write back to PROG_SEL_REGISTER
  new_prog_sel = prog_sel | playback_finished;
  if (!(prog_sel & NO_RESET_MASK)) new_prog_sel &= INVERTED_PROG_SEL_MASK;
  setUserReg(PROG_SEL_REGISTER, new_prog_sel);
  
  // reset playback flag
  playback_finished = 0;
  
  // only use part of prog sel that does not mean other things to select the program.
  prog_sel &= PROG_SEL_MASK;
  
  switch (prog_sel) {
    case 1:
      test_function();
      waitWave();
      playback_finished = PLAYBACK_FINISHED_MASK;
    default:
      wait(IDLE_WAIT_CYCLES);
  }
}"""
        self.assertEqual(expected_program, seqc_program)

    @unittest.skipIf(sys.version_info.minor < 6, "This test requires dict to be ordered.")
    def test_full_run_with_dynamic_rate_reduction(self):
        defined_channels = frozenset(['A', 'B', 'C'])

        unique_n = 1000
        unique_duration = 192

        unique_wfs = get_constant_unique_wfs(n=unique_n, duration=unique_duration,
                                             defined_channels=defined_channels)
        same_wf = DummyWaveform(duration=48, sample_output=np.ones(48), defined_channels=defined_channels)

        channels = ('A', 'B')
        markers = ('C', None, 'A', None)
        amplitudes = (1., 1.)
        offsets = (0., 0.)
        volatage_transformations = (lambda x: x, lambda x: x)
        sample_rate = 1

        old_value, WaveformPlayback.ENABLE_DYNAMIC_RATE_REDUCTION = WaveformPlayback.ENABLE_DYNAMIC_RATE_REDUCTION, True
        try:
            root = complex_program_as_loop(unique_wfs, wf_same=same_wf)
            seqc_nodes = complex_program_as_seqc(unique_wfs, wf_same=same_wf)

            manager = HDAWGProgramManager()

            manager.add_program('test', root, channels, markers, amplitudes, offsets, volatage_transformations,
                                sample_rate)
        finally:
            WaveformPlayback.ENABLE_DYNAMIC_RATE_REDUCTION = old_value



        # 0: Program selection
        # 1: Trigger
        self.assertEqual({UserRegister(zero_based_value=2): 7}, manager.get_register_values('test'))
        seqc_program = manager.to_seqc_program()
        expected_program = """const PROG_SEL_REGISTER = 0;
const TRIGGER_REGISTER = 1;
const TRIGGER_RESET_MASK = 0b10000000000000000000000000000000;
const PROG_SEL_NONE = 0;
const NO_RESET_MASK = 0b10000000000000000000000000000000;
const PLAYBACK_FINISHED_MASK = 0b1000000000000000000000000000000;
const PROG_SEL_MASK = 0b111111111111111111111111111111;
const INVERTED_PROG_SEL_MASK = 0b11000000000000000000000000000000;
const IDLE_WAIT_CYCLES = 300;
wave test_concatenated_waveform_0 = "7fd412eb866ad371f717857ea33b309ec458c6c3469c7e51dcffcdce9a8c2679";
wave test_shared_waveform_121f5c6e8822793b3836fb3098fa4591b91d4c205cc2d8afd01ee1bf6956e518 = "121f5c6e8822793b3836fb3098fa4591b91d4c205cc2d8afd01ee1bf6956e518";

// function used by manually triggered programs
void waitForSoftwareTrigger() {
  while (true) {
    var trigger_register = getUserReg(TRIGGER_REGISTER);
    if (trigger_register & TRIGGER_RESET_MASK) setUserReg(TRIGGER_REGISTER, 0);
    if (trigger_register) return;
  }
}


// program definitions
void test_function() {
  var pos = 0;
  var user_reg_2 = getUserReg(2);
  waitForSoftwareTrigger();
  var init_pos_1 = pos;
  repeat(12) {
    pos = init_pos_1;
    repeat(1000) { // stepping repeat
      repeat(10) {
        repeat(42) {
          playWaveIndexed(test_concatenated_waveform_0, pos, 48, 2); // advance disabled do to parent repetition
        }
        repeat(98) {
          playWave(test_shared_waveform_121f5c6e8822793b3836fb3098fa4591b91d4c205cc2d8afd01ee1bf6956e518, 0);
        }
      }
      pos = pos + 48;
    }
    repeat(21) {
      playWaveIndexed(test_concatenated_waveform_0, pos, 48, 2); // advance disabled do to parent repetition
    }
    pos = pos + 48;
    repeat(23) {
      playWaveIndexed(test_concatenated_waveform_0, pos, 48, 0); // advance disabled do to parent repetition
    }
    pos = pos + 48;
    var idx_2;
    for(idx_2 = 0; idx_2 < user_reg_2; idx_2 = idx_2 + 1) {
      playWaveIndexed(test_concatenated_waveform_0, pos, 48, 0); // advance disabled do to parent repetition
    }
    pos = pos + 48;
  }
}

// Declare and initialize global variables
// Selected program index (0 -> None)
var prog_sel = 0;

// Value that gets written back to program selection register.
// Used to signal that at least one program was played completely.
var new_prog_sel = 0;

// Is OR'ed to new_prog_sel.
// Set to PLAYBACK_FINISHED_MASK if a program was played completely.
var playback_finished = 0;


// runtime block
while (true) {
  // read program selection value
  prog_sel = getUserReg(PROG_SEL_REGISTER);
  
  // calculate value to write back to PROG_SEL_REGISTER
  new_prog_sel = prog_sel | playback_finished;
  if (!(prog_sel & NO_RESET_MASK)) new_prog_sel &= INVERTED_PROG_SEL_MASK;
  setUserReg(PROG_SEL_REGISTER, new_prog_sel);
  
  // reset playback flag
  playback_finished = 0;
  
  // only use part of prog sel that does not mean other things to select the program.
  prog_sel &= PROG_SEL_MASK;
  
  switch (prog_sel) {
    case 1:
      test_function();
      waitWave();
      playback_finished = PLAYBACK_FINISHED_MASK;
    default:
      wait(IDLE_WAIT_CYCLES);
  }
}"""
        self.assertEqual(expected_program, seqc_program)

    def test_shuttle_pulse(self):
        from qupulse.pulses import PointPT, RepetitionPT, ParallelChannelPT, MappingPT, ForLoopPT, AtomicMultiChannelPT, FunctionPT, TimeReversalPT
        import sympy

        read_pls = PointPT([
            ('t_read_0', 'V_read_0'),
            ('t_read_1', 'V_read_1'),
            ('t_read_2', 'V_read_2'),
            ('t_read_3', 'V_read_3'),
        ], tuple('ABCDEFGHIJKLMNOP'), measurements=[('read', 0, 't_read_3')])

        arbitrary_load = PointPT([
            ('t_load_0', 'V_load_0 * bit_flag + V_empty_0 * (1 - bit_flag)'),
            ('t_load_1', 'V_load_1 * bit_flag + V_empty_1 * (1 - bit_flag)'),
            ('t_load_2', 'V_load_2 * bit_flag + V_empty_2 * (1 - bit_flag)'),
            ('t_load_3', 'V_load_3 * bit_flag + V_empty_3 * (1 - bit_flag)'),
        ], tuple('ABCDEFGHIJKLMNOP'), measurements=[('load', 0, 't_load_3')])

        load_bit = MappingPT(arbitrary_load, parameter_mapping={'bit_flag': 'pattern[bit]'})

        reduce_amp = PointPT([
            (0, 'V_reduce_amp_0'),
            ('t_reduce_amp', 'V_reduce_amp_1', 'linear'),
            ('t_amp_holdon', 'V_reduce_amp_1', 'hold'),
            ('t_recover_amp', 'V_reduce_amp_2', 'linear'),
        ], tuple('ABCDEFGHIJKLMNOP'), measurements=[('reduce_amp', 0, 't_recover_amp')])

        #  define sinewaves by FunctionPT

        sample_rate, f, n_segments = sympy.symbols('sample_rate, f, n_segments')
        n_oct = sympy.symbols('n_oct')
        segment_time = n_segments / sample_rate  # in ns

        shuttle_period = sympy.ceiling(1 / f / segment_time) * segment_time  # in ns
        shuttle_oct = shuttle_period / 8
        actual_frequency = 1 / shuttle_period

        # Make a shuttle pulse including 4 clavier gates + 2 individual gates on each side.
        arbitrary_shuttle = AtomicMultiChannelPT(
            FunctionPT(f'amp_A * cos(2*pi*{actual_frequency}*t + phi[0]) + offset[0]', duration_expression='duration',
                       channel='A'),
            FunctionPT(f'amp_B * cos(2*pi*{actual_frequency}*t + phi[1]) + offset[1]', duration_expression='duration',
                       channel='B'),
            FunctionPT(f'amp_C * cos(2*pi*{actual_frequency}*t + phi[2]) + offset[2]', duration_expression='duration',
                       channel='C'),
            FunctionPT(f'amp_D * cos(2*pi*{actual_frequency}*t + phi[3]) + offset[3]', duration_expression='duration',
                       channel='D'),

            # >> Add T gates for both ends with consistent channel names ('F': TLB2 <- S4, 'K': TRB2 <- S4)
            FunctionPT(f'amp_F * cos(2*pi*{actual_frequency}*t + phi[5]) + offset[5]', duration_expression='duration',
                       channel='F'),
            FunctionPT(f'amp_K * cos(2*pi*{actual_frequency}*t + phi[10]) + offset[10]', duration_expression='duration',
                       channel='K'),
            measurements=[('shuttle', 0, 'duration')]
        )

        shuttle_in = MappingPT(arbitrary_shuttle,
                               parameter_mapping={'duration': shuttle_period},
                               measurement_mapping={'shuttle': 'shuttle_in'})

        shuttle_out = MappingPT(TimeReversalPT(arbitrary_shuttle),
                                parameter_mapping={'duration': shuttle_period},
                                measurement_mapping={'shuttle': 'shuttle_out'})

        shuttle_fract_in = MappingPT(arbitrary_shuttle,
                                     parameter_mapping={'duration': shuttle_oct * n_oct},
                                     measurement_mapping={'shuttle': 'shuttle_in'}
                                     )

        shuttle_fract_out = shuttle_fract_in.with_time_reversal()

        flush_out = MappingPT(RepetitionPT(TimeReversalPT(arbitrary_shuttle), '2 * len(pattern)'),
                              parameter_mapping={'duration': shuttle_period},
                              measurement_mapping={'shuttle': 'flush_out'},
                              parameter_constraints=['len(pattern) > 0'])

        wobble_shuttle = MappingPT(arbitrary_shuttle @ arbitrary_shuttle.with_time_reversal(),
                                   parameter_mapping={'duration': shuttle_period},
                                   measurement_mapping={'shuttle': 'shuttle_in'})

        # Plug load, shuttle in and read together => PL1 --> S+n --> DL1
        channels_onhold = load_bit.defined_channels - shuttle_in.defined_channels
        bit_in = load_bit @ ParallelChannelPT(shuttle_in,
                                              overwritten_channels={ch: f"offset[{ord(ch) - ord('A')}]" for ch in
                                                                    channels_onhold})
        pattern_in = ForLoopPT(bit_in, 'bit', 'len(pattern)')

        # Plug fractional shuttle in, reduce amplitude and fractional out together
        channels_onhold = pattern_in.defined_channels - shuttle_fract_in.defined_channels
        fract_in_n_reduce_amp = ParallelChannelPT(
            shuttle_fract_in,
            overwritten_channels={ch: f"offset[{ord(ch) - ord('A')}]" for ch in channels_onhold}
        ) @ reduce_amp @ ParallelChannelPT(
            shuttle_fract_out,
            overwritten_channels={ch: f"offset[{ord(ch) - ord('A')}]" for ch in channels_onhold})

        # plug read and shuttle out together => S(-n' per read) --> DL1
        channels_onhold = reduce_amp.defined_channels - shuttle_in.defined_channels
        period_out = ParallelChannelPT(shuttle_out, overwritten_channels={ch: f"offset[{ord(ch) - ord('A')}]" for ch in
                                                                          channels_onhold}) @ read_pls

        #
        channels_onhold = read_pls.defined_channels - shuttle_in.defined_channels
        wobble = ParallelChannelPT(RepetitionPT(wobble_shuttle, 3),
                                   overwritten_channels={ch: f"offset[{ord(ch) - ord('A')}]" for ch in channels_onhold})

        # repeated read and shuttle out => [S(-n' per read) --> DL1] x (n_in + n_extra)
        tot_period_out = RepetitionPT(period_out, 'len(pattern) + n_period_extra')

        # make a flush pulse according to the last read.
        channels_onhold = read_pls.defined_channels - flush_out.defined_channels
        flush = ParallelChannelPT(flush_out,
                                  overwritten_channels={ch: f"flush[{ord(ch) - ord('A')}]" for ch in channels_onhold})

        pattern_in_n_out = pattern_in @ fract_in_n_reduce_amp @ tot_period_out

        default_params = {}

        for n in ('load', 'read', 'empty'):
            for ii in range(4):
                default_params[f't_{n}_{ii}'] = ii * 25e6

        default_params = {**default_params,
                          'amp_A': .14 * 2,
                          'amp_B': .14,
                          'amp_C': .14 * 2,
                          'amp_D': .14,
                          'amp_F': .14,
                          'amp_K': .14,
                          }

        default_params = {**default_params,
                          #              A,   B,   C,   D,   E,    F,    G,    H,    I,   J,    K,   L,   M,   N,   O,   P
                          #             S1,  S2,  S3,  S4, TRP, TRB2,  RB2, TRB1, TLB1, TLP, TLB2, LB2, LB1, RB1, EMP, CLK

                          'V_load_0': [-.3, 0., .3, 0., -.2, 0., -.1, -.4, 0., 0., 0, -.3, 0., 0., 0., 0.],
                          'V_load_1': [-.3, 0., .3, 0., -.2, 0., -.1, -.4, 0., .5, 0, -.3, 0., 0., 0., 0.],
                          'V_load_2': [-.3, 0., .3, 0., -.2, 0., -.1, -.4, -.5, .5, 0, -.3, 0., 0., 0., 0.],
                          'V_load_3': [-.3, 0., .3, 0., -.2, 0., -.1, -.4, -.5, .5, 0, -.3, 0., 0., 0., 0.],

                          'V_read_0': [-.3, 0., .3, 0., -.2, 0., -.1, -.4, -.5, .5, 0, -.3, 0., 0., 0., 0.],
                          'V_read_1': [-.3, 0., .3, 0., -.2, 0., -.1, -.4, -.5, .5, 0, -.3, 0., 0., 0., 0.],
                          'V_read_2': [-.3, 0., .3, 0., -.2, 0., -.1, -.4, 0., .5, 0, -.3, 0., 0., 0., 0.],
                          'V_read_3': [-.3, 0., .3, 0., -.2, 0., -.1, -.4, 0., 0., 0, -.3, 0., 0., 0., 0.],

                          'V_empty_0': [-.3, 0., .3, 0., -.2, 0., 0, -.4, 0., 0., 0, -.3, 0., 0., 0., 0.],
                          'V_empty_1': [-.3, 0., .3, 0., -.2, 0., 0, -.4, 0., 0., 0, -.3, 0., 0., 0., 0.],
                          'V_empty_2': [-.3, 0., .3, 0., -.2, 0., 0, -.4, -.5, 0., 0, -.3, 0., 0., 0., 0.],
                          'V_empty_3': [-.3, 0., .3, 0., -.2, 0., 0, -.4, -.5, 0., 0, -.3, 0., 0., 0., 0.],

                          'offset': [0., 0., 0., 0., -.2, 0., -.1, -.2, 0., 0., 0, -.3, 0., 0., 0., 0.],
                          'flush': [0., 0., 0., 0., -.2, 0., -.1, -.2, 0., 0., 0, -.3, 0., 0., 0., 0.],

                          'f': 100e-9,

                          'pattern': [1, 0, 1, 0, 1, 1],
                          'n_oct': 1,
                          'n_period_extra': 3,
                          # number of period (extra reading wrt shuttle in. Default = 3 -> 3 additiional reading)

                          't_reduce_amp': 30e6,
                          't_amp_holdon': 60e6,
                          't_recover_amp': 90e6,

                          'V_reduce_amp_0': [-.3, 0., .3, 0., -.2, 0., -.1, -.4, 0., 0., 0, -.3, 0., 0., 0., 0.],
                          'V_reduce_amp_1': [-.3, -.1, .3, .1, -.2, 0., -.1, -.4, 0., 0., 0, -.3, 0., 0., 0., 0.],
                          'V_reduce_amp_2': [-.3, 0., .3, 0., -.2, 0., -.1, -.4, 0., 0., 0, -.3, 0., 0., 0., 0.],

                          'phi': [-3.1416, -4.7124, -6.2832, -7.8540, -1.5708, -7.8540, -1.5708, -1.5708, -1.5708,
                                  -1.5708, -1.5708, -1.5708],
                          'n_segments': 192,
                          'sample_rate': 0.1 / 2 ** 5,
                          }

        program = pattern_in_n_out.create_program(parameters=default_params)

        manager = HDAWGProgramManager()

        manager.add_program('test', program,
                            channels=('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'),
                            markers=(None,) * 16,
                            amplitudes=(4.,)*8,
                            offsets=(0.,)*8,
                            voltage_transformations=(None,)*8,
                            sample_rate=default_params['sample_rate'])
        seqc_program = manager.to_seqc_program()

        return seqc_program

    @unittest.skipIf(sys.version_info.minor < 6, "This test requires dict to be ordered.")
    def test_DigTrigger(self):
        defined_channels = frozenset(['A', 'B', 'C'])

        unique_n = 1000
        unique_duration = 32

        unique_wfs = get_unique_wfs(n=unique_n, duration=unique_duration, defined_channels=defined_channels)
        same_wf = DummyWaveform(duration=48, sample_output=np.ones(48), defined_channels=defined_channels)

        channels = ('A', 'B')
        markers = ('C', None, 'A', None)
        amplitudes = (1., 1.)
        offsets = (0., 0.)
        volatage_transformations = (lambda x: x, lambda x: x)
        sample_rate = 1

        triggerIn = 8
        DigTriggerIndex = 1

        root = complex_program_as_loop(unique_wfs, wf_same=same_wf)
        seqc_nodes = complex_program_as_seqc(unique_wfs, wf_same=same_wf)

        manager = HDAWGProgramManager()
        compiler_settings = manager.DEFAULT_COMPILER_SETTINGS
        compiler_settings['trigger_wait_code'] = f'waitDigTrigger({DigTriggerIndex});'
        manager.add_program('test', root, channels, markers, amplitudes, offsets, volatage_transformations, sample_rate)

        # 0: Program selection
        # 1: Trigger

        seqc_program = manager.to_seqc_program()

        return seqc_program




