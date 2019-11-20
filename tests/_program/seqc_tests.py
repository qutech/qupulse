from unittest import TestCase, mock
import time
from more_itertools import take

import numpy as np

from qupulse._program._loop import Loop
from qupulse._program.seqc import BinaryWaveform, loop_to_seqc, WaveformPlayback, Repeat, SteppingRepeat, Scope,\
    to_node_clusters, find_sharable_waveforms, mark_sharable_waveforms

from tests.pulses.sequencing_dummies import DummyWaveform

try:
    import zhinst
except ImportError:
    zhinst = None


def make_binary_waveform(waveform):
    if waveform.duration == 0:
        data = 3 * [1, 2, 3, 4, 5]
        return BinaryWaveform(data, (True, True), (True, True))
    else:
        ch = next(iter(sorted(waveform.defined_channels)))
        t = np.arange(0., waveform.duration, 1.)
        data = waveform.get_sampled(ch, t)
        return BinaryWaveform(data, (True, False), (False, False))


def get_unique_wfs(n=10000, duration=8):
    if not hasattr(get_unique_wfs, 'cache'):
        get_unique_wfs.cache = {}

    key = (n, duration)
    if key not in get_unique_wfs.cache:
        get_unique_wfs.cache[key] = [
            DummyWaveform(duration=duration, sample_output=float(idx) * np.arange(duration))
            for idx in range(n)
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


class SEQCNodeTests(TestCase):
    """Test everything besides source code generation"""
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

    def test_get_single_indexed_playback(self):
        wf1, wf_shared = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(2, 32)))
        wf_shared.shared = True
        self.assertIs(wf1._get_single_indexed_playback(), wf1)
        self.assertIsNone(wf_shared._get_single_indexed_playback())

        self.assertIs(Scope([wf1, wf_shared])._get_single_indexed_playback(), wf1)
        self.assertIsNone(Scope([wf1, wf_shared, wf1])._get_single_indexed_playback(), wf1)

    def test_stores_initial_pos(self):
        wf1, wf_shared = map(WaveformPlayback, map(make_binary_waveform, get_unique_wfs(2, 32)))

        scope = Scope([wf1, wf1, wf1])
        stepping_repeat = SteppingRepeat([wf1, wf1, wf1])

        raise NotImplementedError()


class LoopToSEQCTranslationTests(TestCase):
    def test_loop_to_seqc_leaf(self):
        """Test the translation of leaves"""
        wf = DummyWaveform(duration=10)
        loop = Loop(waveform=wf)

        # with wrapping repetition
        loop.repetition_count = 15
        waveform_to_bin = mock.Mock(wraps=make_binary_waveform)
        expected = Repeat(loop.repetition_count, WaveformPlayback(waveform=make_binary_waveform(wf)))
        result = loop_to_seqc(loop, 1, 1, waveform_to_bin)
        waveform_to_bin.assert_called_once_with(wf)
        self.assertEqual(expected, result)

        # without wrapping repetition
        loop.repetition_count = 1
        waveform_to_bin = mock.Mock(wraps=make_binary_waveform)
        expected = WaveformPlayback(waveform=make_binary_waveform(wf))
        result = loop_to_seqc(loop, 1, 1, waveform_to_bin)
        waveform_to_bin.assert_called_once_with(wf)
        self.assertEqual(expected, result)

    def test_loop_to_seqc_len_1(self):
        """Test the translation of loops with len(loop) == 1"""
        loop = Loop(children=[Loop()])
        waveform_to_bin = mock.Mock(wraps=make_binary_waveform)
        loop_to_seqc_kwargs = dict(min_repetitions_for_for_loop=2,
                                   min_repetitions_for_shared_wf=3,
                                   waveform_to_bin=waveform_to_bin)

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

        def dummy_loop_to_seqc(loop, **kwargs):
            return loop

        loops = [wf1, wf2, wf1, wf1, wf3, wf1, wf1, wf1]
        expected_calls = [mock.call(loop, **loop_to_seqc_kwargs) for loop in loops]
        expected_result = [[wf1, wf2, wf1, wf1], [wf3], [wf1, wf1, wf1]]

        with mock.patch('qupulse._program.seqc.loop_to_seqc', wraps=dummy_loop_to_seqc) as mock_loop_to_seqc:
            result = to_node_clusters(loops, loop_to_seqc_kwargs)
            self.assertEqual(mock_loop_to_seqc.mock_calls, expected_calls)
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
        with self.assertRaises(AssertionError):
            loop_to_seqc(Loop(repetition_count=12, children=[Loop()]),
                         min_repetitions_for_for_loop=3, min_repetitions_for_shared_wf=2,
                         waveform_to_bin=make_binary_waveform)

        loop_to_seqc_kwargs = dict(min_repetitions_for_for_loop=3,
                                   min_repetitions_for_shared_wf=4,
                                   waveform_to_bin=make_binary_waveform)

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
        unique_wfs = get_unique_wfs()
        same_wf = DummyWaveform(duration=15, sample_output=np.ones(15))
        root = complex_program_as_loop(unique_wfs, wf_same=same_wf)

        t0 = time.perf_counter()

        seqc = loop_to_seqc(root, 50, 100, make_binary_waveform)

        t1 = time.perf_counter()
        print('took', t1 - t0, 's')

        expected = complex_program_as_seqc(unique_wfs, wf_same=same_wf)
        self.assertEqual(expected, seqc)


class SEQCToCodeTranslationTests(TestCase):
    def test_shared_playback(self):
        raise NotImplementedError()

    def test_indexed_playback(self):
        raise NotImplementedError()

    def test_scope(self):
        raise NotImplementedError()

    def test_stepped_repeat(self):
        raise NotImplementedError()

    def test_repeat(self):
        raise NotImplementedError()

    def test_program_to_code_translation(self):
        """Integration test"""
        unique_wfs = get_unique_wfs()
        same_wf = DummyWaveform(duration=15, sample_output=np.ones(15))
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
    var init_pos_1 = pos;
    repeat(10) {
      pos = init_pos_1;
      var init_pos_2 = pos;
      repeat(42) {
        pos = init_pos_2;
        playWaveformIndexed(0, pos, 8); pos = pos + 8;
      }
      repeat(98) {
        playWaveform(1);
      }
    }
  }
  var init_pos_3 = pos;
  repeat(21) {
    pos = init_pos_3;
    playWaveformIndexed(0, pos, 8); pos = pos + 8;
  }
  var init_pos_4 = pos;
  repeat(23) {
    pos = init_pos_4;
    playWaveformIndexed(0, pos, 15); pos = pos + 15;
  }
}"""
        self.assertEqual(expected, seqc_code)

