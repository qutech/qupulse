from unittest import TestCase, mock
import time

import numpy as np

from qupulse._program._loop import Loop
from qupulse.hardware.awgs.seqc import BinaryWaveform, loop_to_seqc, WaveformPlayback, Repeat, SteppingRepeat, Scope

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


class SEQCNodeTests(TestCase):
    """Test everything besides source code generation"""
    def test_visit_nodes(self):
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
        with mock.patch('qupulse.hardware.awgs.seqc.loop_to_seqc', return_value=expected) as mocked_loop_to_seqc:
            result = loop_to_seqc(loop, **loop_to_seqc_kwargs)
            self.assertEqual(result, expected)
            mocked_loop_to_seqc.assert_called_once_with(loop[0], **loop_to_seqc_kwargs)

        loop.repetition_count = 14
        expected = Repeat(14, 'asdfg')
        with mock.patch('qupulse.hardware.awgs.seqc.loop_to_seqc', return_value=expected.scope) as mocked_loop_to_seqc:
            result = loop_to_seqc(loop, **loop_to_seqc_kwargs)
            self.assertEqual(result, expected)
            mocked_loop_to_seqc.assert_called_once_with(loop[0], **loop_to_seqc_kwargs)

        waveform_to_bin.assert_not_called()

    def test_to_node_clusters(self):
        """Test cluster generation"""
        raise NotImplementedError()

    def test_find_sharable_waveforms(self):
        raise NotImplementedError()

    def test_mark_sharable_waveforms(self):
        raise NotImplementedError()

    def test_loop_to_seqc_cluster_handling(self):
        """Test handling of clusters"""
        raise NotImplementedError()

    def test_program_translation(self):
        """Integration test"""
        unique_wfs = get_unique_wfs()
        same_wf = DummyWaveform(duration=15, sample_output=np.ones(15))
        root, same_wf = complex_program_as_loop(unique_wfs, wf_same=same_wf)

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

        class DummyWfManager:
            def __init__(self):
                self.shared = {}
                self.concatenated = []

            def request_shared(self, wf):
                return self.shared.setdefault(wf, len(self.shared) + 1)

            def request_concatenated(self, wf):
                self.concatenated.append(wf)
                return 0

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

