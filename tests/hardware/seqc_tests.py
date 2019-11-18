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


class SEQCNodeTests(TestCase):
    """Test everything besides source code generation"""
    def test_visit_nodes(self):
        raise NotImplementedError()


class SEQCTranslationTests(TestCase):
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
        root = Loop(repetition_count=12)

        unique_wfs = []

        wf_same = DummyWaveform(duration=10, sample_output=np.ones(10))
        for idx in range(10000):
            wf_unique = DummyWaveform(duration=8, sample_output=float(idx) * np.arange(8))
            unique_wfs.append(wf_unique)
            root.append_child(children=[Loop(repetition_count=42, waveform=wf_unique),
                                        Loop(repetition_count=98, waveform=wf_same)],
                              repetition_count=10)

        root.append_child(waveform=unique_wfs[0], repetition_count=21)
        root.append_child(waveform=wf_same, repetition_count=23)

        t0 = time.perf_counter()

        seqc = loop_to_seqc(root, 50, 100, make_binary_waveform)

        t1 = time.perf_counter()
        print('took', t1 - t0, 's')

        expected = Repeat(12,
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
        self.assertEqual(expected, seqc)
