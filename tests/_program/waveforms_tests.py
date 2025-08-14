import unittest
from unittest import mock

import numpy
import numpy as np

from qupulse.utils.types import TimeType
from qupulse.pulses.interpolation import HoldInterpolationStrategy, LinearInterpolationStrategy,\
    JumpInterpolationStrategy
from qupulse.program.waveforms import MultiChannelWaveform, RepetitionWaveform, SequenceWaveform,\
    TableWaveformEntry, TableWaveform, TransformingWaveform, SubsetWaveform, ArithmeticWaveform, ConstantWaveform,\
    Waveform, FunctorWaveform, FunctionWaveform, ReversedWaveform
from qupulse.program.transformation import LinearTransformation
from qupulse.expressions import ExpressionScalar, Expression

from tests.pulses.sequencing_dummies import DummyWaveform, DummyInterpolationStrategy
from tests._program.transformation_tests import TransformationStub


def assert_constant_consistent(test_case: unittest.TestCase, wf: Waveform):
    if wf.is_constant():
        cvs = wf.constant_value_dict()
        test_case.assertEqual(wf.defined_channels, cvs.keys())
        for ch in wf.defined_channels:
            test_case.assertEqual(cvs[ch], wf.constant_value(ch))
    else:
        test_case.assertIsNone(wf.constant_value_dict())
        test_case.assertIn(None, {wf.constant_value(ch) for ch in wf.defined_channels})


class WaveformStub(Waveform):
    """Not a slot class to allow easier mocking"""
    def __init__(self):
        super(WaveformStub, self).__init__(duration=None)
    
    @property
    def defined_channels(self):
        raise NotImplementedError()

    def unsafe_get_subset_for_channels(self, channels) -> 'Waveform':
        raise NotImplementedError()

    def unsafe_sample(self,
                      channel,
                      sample_times,
                      output_array=None) -> np.ndarray:
        raise NotImplementedError()

    @property
    def compare_key(self):
        raise NotImplementedError()


class WaveformTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)

    def test_get_sampled_exceptions(self):
        wf = DummyWaveform(duration=2., sample_output=[1, 2], defined_channels={'A', 'B'})

        with self.assertRaises(ValueError):
            wf.get_sampled(channel='A',
                           sample_times=numpy.asarray([2, 1], dtype=float))
        with self.assertRaises(ValueError):
            wf.get_sampled(channel='A',
                           sample_times=numpy.asarray([-12, 1], dtype=float))
        with self.assertRaises(KeyError):
            wf.get_sampled(channel='C',
                           sample_times=numpy.asarray([0.5, 1], dtype=float))
        with self.assertRaises(ValueError):
            wf.get_sampled(channel='A',
                           sample_times=numpy.asarray([0.5, 1], dtype=float),
                           output_array=numpy.empty(1))

    def test_get_sampled_caching(self):
        wf = DummyWaveform(duration=2., sample_output=[1, 2], defined_channels={'A', 'B'})

        self.assertIs(wf.get_sampled('A', sample_times=numpy.arange(2)),
                      wf.get_sampled('A', sample_times=numpy.arange(2)))

    def test_get_sampled_empty(self):
        wf = DummyWaveform(duration=2., defined_channels={'A', 'B'})

        sample_times = numpy.zeros(0)
        output_array = numpy.zeros(0)

        sampled = wf.get_sampled('A', sample_times=sample_times)
        self.assertIsInstance(sampled, numpy.ndarray)
        self.assertEqual(len(sampled), 0)

        self.assertIs(wf.get_sampled('A', sample_times=sample_times, output_array=output_array), output_array)
        self.assertEqual(len(output_array), 0)

        with self.assertRaises(ValueError):
            wf.get_sampled('A', sample_times=sample_times, output_array=numpy.zeros(1))

    def test_get_sampled_argument_forwarding(self):
        wf = DummyWaveform(duration=2., sample_output=[1, 2], defined_channels={'A', 'B'})

        out_expected = numpy.empty(2)

        out_received = wf.get_sampled('A', sample_times=numpy.arange(2), output_array=out_expected)

        self.assertIs(out_expected, out_received)
        self.assertEqual(len(wf.sample_calls), 1)
        self.assertIs(wf.sample_calls[0][-1], out_expected)
        self.assertEqual(out_received.tolist(), [1, 2])

    def test_get_subset_for_channels(self):
        wf_ab = DummyWaveform(defined_channels={'A', 'B'})
        wf_a = DummyWaveform(defined_channels={'A'})

        with self.assertRaises(KeyError):
            wf_ab.get_subset_for_channels({'C'})
        with self.assertRaises(KeyError):
            wf_ab.get_subset_for_channels({'A', 'C'})
        with self.assertRaises(KeyError):
            wf_a.get_subset_for_channels({'C'})
        with self.assertRaises(KeyError):
            wf_a.get_subset_for_channels({'A', 'C'})

        self.assertIs(wf_ab, wf_ab.get_subset_for_channels({'A', 'B'}))
        self.assertIs(wf_a, wf_a.get_subset_for_channels({'A'}))

        wf_sub = wf_ab.get_subset_for_channels({'A'})
        self.assertEqual(wf_sub.defined_channels, {'A'})

    def test_constant_default_impl(self):
        wf = DummyWaveform(defined_channels={'A', 'B'})
        self.assertFalse(wf.is_constant())

        values = {'A': 4., 'B': 5.}
        wf.constant_value = lambda ch: values[ch]
        self.assertEqual(values, wf.constant_value_dict())
        assert_constant_consistent(self, wf)

    def test_negation(self):
        wf = DummyWaveform(defined_channels={'A', 'B'})
        self.assertIs(wf, +wf)

        expected_neg = FunctorWaveform(wf, {'A': np.negative, 'B': np.negative})
        self.assertEqual(expected_neg, -wf)

    def test_slot(self):
        wf = ConstantWaveform.from_mapping(1, {'f': 3})
        with self.assertRaises(AttributeError):
            wf.asd = 5


class MultiChannelWaveformTest(unittest.TestCase):
    def test_init_no_args(self) -> None:
        with self.assertRaises(ValueError):
            MultiChannelWaveform(dict())
        with self.assertRaises(ValueError):
            MultiChannelWaveform(None)

    def test_from_parallel(self):
        dwf_a = DummyWaveform(duration=2.2, defined_channels={'A'})
        dwf_b = DummyWaveform(duration=2.2, defined_channels={'B'})
        dwf_c = DummyWaveform(duration=2.2, defined_channels={'C'})

        self.assertIs(dwf_a, MultiChannelWaveform.from_parallel([dwf_a]))

        wf_ab = MultiChannelWaveform.from_parallel([dwf_a, dwf_b])
        self.assertEqual(wf_ab, MultiChannelWaveform([dwf_a, dwf_b]))

        wf_abc = MultiChannelWaveform.from_parallel([wf_ab, dwf_c])
        self.assertEqual(wf_abc, MultiChannelWaveform([dwf_a, dwf_b, dwf_c]))

    def test_get_item(self):
        dwf_a = DummyWaveform(duration=2.2, defined_channels={'A'})
        dwf_b = DummyWaveform(duration=2.2, defined_channels={'B'})
        dwf_c = DummyWaveform(duration=2.2, defined_channels={'C'})

        wf = MultiChannelWaveform([dwf_a, dwf_b, dwf_c])

        self.assertIs(wf['A'], dwf_a)
        self.assertIs(wf['B'], dwf_b)
        self.assertIs(wf['C'], dwf_c)

        with self.assertRaises(KeyError):
            wf['D']

    def test_init_single_channel(self) -> None:
        dwf = DummyWaveform(duration=1.3, defined_channels={'A'})

        waveform = MultiChannelWaveform([dwf])
        self.assertEqual({'A'}, waveform.defined_channels)
        self.assertEqual(TimeType.from_float(1.3), waveform.duration)

    def test_init_several_channels(self) -> None:
        dwf_a = DummyWaveform(duration=2.2, defined_channels={'A'})
        dwf_b = DummyWaveform(duration=2.2, defined_channels={'B'})
        dwf_c = DummyWaveform(duration=2.3, defined_channels={'C'})

        waveform = MultiChannelWaveform([dwf_a, dwf_b])
        self.assertEqual({'A', 'B'}, waveform.defined_channels)
        self.assertEqual(TimeType.from_float(2.2), waveform.duration)

        with self.assertRaises(ValueError):
            MultiChannelWaveform([dwf_a, dwf_c])
        with self.assertRaises(ValueError):
            MultiChannelWaveform([waveform, dwf_c])
        with self.assertRaises(ValueError):
            MultiChannelWaveform((dwf_a, dwf_a))

        dwf_c_valid = DummyWaveform(duration=2.2, defined_channels={'C'})
        waveform_flat = MultiChannelWaveform.from_parallel((waveform, dwf_c_valid))
        self.assertEqual(
            MultiChannelWaveform([dwf_a, dwf_b, dwf_c_valid]),
            waveform_flat
        )

    def test_unsafe_sample(self) -> None:
        sample_times = numpy.linspace(98.5, 103.5, num=11)
        samples_a = numpy.linspace(4, 5, 11)
        samples_b = numpy.linspace(2, 3, 11)
        dwf_a = DummyWaveform(duration=3.2, sample_output=samples_a, defined_channels={'A'})
        dwf_b = DummyWaveform(duration=3.2, sample_output=samples_b, defined_channels={'B', 'C'})
        waveform = MultiChannelWaveform((dwf_a, dwf_b))

        result_a = waveform.unsafe_sample('A', sample_times)
        numpy.testing.assert_equal(result_a, samples_a)

        result_b = waveform.unsafe_sample('B', sample_times)
        numpy.testing.assert_equal(result_b, samples_b)

        self.assertEqual(len(dwf_a.sample_calls), 1)
        self.assertEqual(len(dwf_b.sample_calls), 1)

        numpy.testing.assert_equal(sample_times, dwf_a.sample_calls[0][1])
        numpy.testing.assert_equal(sample_times, dwf_b.sample_calls[0][1])

        self.assertEqual('A', dwf_a.sample_calls[0][0])
        self.assertEqual('B', dwf_b.sample_calls[0][0])

        self.assertIs(dwf_a.sample_calls[0][2], None)
        self.assertIs(dwf_b.sample_calls[0][2], None)

        reuse_output = numpy.empty_like(samples_a)
        result_a = waveform.unsafe_sample('A', sample_times, reuse_output)
        self.assertEqual(len(dwf_a.sample_calls), 2)
        self.assertIs(result_a, reuse_output)
        self.assertIs(result_a, dwf_a.sample_calls[1][2])
        numpy.testing.assert_equal(result_b, samples_b)

    def test_equality(self) -> None:
        dwf_a = DummyWaveform(duration=246.2, defined_channels={'A'})
        dwf_b = DummyWaveform(duration=246.2, defined_channels={'B'})
        dwf_c = DummyWaveform(duration=246.2, defined_channels={'C'})
        waveform_a1 = MultiChannelWaveform([dwf_a, dwf_b])
        waveform_a2 = MultiChannelWaveform([dwf_a, dwf_b])
        waveform_a3 = MultiChannelWaveform([dwf_a, dwf_c])
        self.assertEqual(waveform_a1, waveform_a1)
        self.assertEqual(waveform_a1, waveform_a2)
        self.assertNotEqual(waveform_a1, waveform_a3)

    def test_unsafe_get_subset_for_channels(self):
        dwf_a = DummyWaveform(duration=246.2, defined_channels={'A'})
        dwf_b = DummyWaveform(duration=246.2, defined_channels={'B'})
        dwf_c = DummyWaveform(duration=246.2, defined_channels={'C'})

        mcwf = MultiChannelWaveform((dwf_a, dwf_b, dwf_c))
        with self.assertRaises(KeyError):
            mcwf.unsafe_get_subset_for_channels({'D'})
        with self.assertRaises(KeyError):
            mcwf.unsafe_get_subset_for_channels({'A', 'D'})

        self.assertIs(mcwf.unsafe_get_subset_for_channels({'A'}), dwf_a)
        self.assertIs(mcwf.unsafe_get_subset_for_channels({'B'}), dwf_b)
        self.assertIs(mcwf.unsafe_get_subset_for_channels({'C'}), dwf_c)

        sub_ab = mcwf.unsafe_get_subset_for_channels({'A', 'B'})
        self.assertEqual(sub_ab.defined_channels, {'A', 'B'})
        self.assertIsInstance(sub_ab, MultiChannelWaveform)
        self.assertIs(sub_ab.unsafe_get_subset_for_channels({'A'}), dwf_a)
        self.assertIs(sub_ab.unsafe_get_subset_for_channels({'B'}), dwf_b)

    def test_constant_default_impl(self):
        wf_non_const_a = DummyWaveform(defined_channels={'A'}, duration=3)
        wf_non_const_b = DummyWaveform(defined_channels={'B'}, duration=3)
        wf_const_c = ConstantWaveform(channel='C', amplitude=2.2, duration=3)
        wf_const_d = ConstantWaveform(channel='D', amplitude=3.3, duration=3)

        wf_const = MultiChannelWaveform.from_parallel((wf_const_c, wf_const_d))
        wf_non_const = MultiChannelWaveform.from_parallel((wf_non_const_b, wf_non_const_a))
        wf_mixed = MultiChannelWaveform.from_parallel((wf_non_const_a, wf_const_c))

        assert_constant_consistent(self, wf_const)
        assert_constant_consistent(self, wf_non_const)
        assert_constant_consistent(self, wf_mixed)

        self.assertEqual(wf_const.constant_value_dict(), {'C': 2.2, 'D': 3.3})
        self.assertIsNone(wf_non_const.constant_value_dict())
        self.assertIsNone(wf_mixed.constant_value_dict())
        self.assertEqual(wf_mixed.constant_value('C'), 2.2)


class RepetitionWaveformTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init(self):
        body_wf = DummyWaveform()

        with self.assertRaises(ValueError):
            RepetitionWaveform(body_wf, -1)

        with self.assertRaises(ValueError):
            RepetitionWaveform(body_wf, 1.1)

        wf = RepetitionWaveform(body_wf, 3)
        self.assertIs(wf._body, body_wf)
        self.assertEqual(wf._repetition_count, 3)

        assert_constant_consistent(self, wf)

    def test_from_repetition_count(self):
        dwf = DummyWaveform()
        self.assertEqual(RepetitionWaveform(dwf, 3), RepetitionWaveform.from_repetition_count(dwf, 3))

        cwf = ConstantWaveform(duration=3, amplitude=2.2, channel='A')
        with mock.patch.object(ConstantWaveform, 'from_mapping', return_value=mock.sentinel) as from_mapping:
            self.assertIs(from_mapping.return_value, RepetitionWaveform.from_repetition_count(cwf, 5))
            from_mapping.assert_called_once_with(15, {'A': 2.2})

    def test_duration(self):
        wf = RepetitionWaveform(DummyWaveform(duration=2.2), 3)
        self.assertEqual(wf.duration, TimeType.from_float(2.2)*3)

    def test_defined_channels(self):
        body_wf = DummyWaveform(defined_channels={'a'})
        self.assertIs(RepetitionWaveform(body_wf, 2).defined_channels, body_wf.defined_channels)

    def test_equality(self):
        body_wf_1 = DummyWaveform(defined_channels={'a'})
        wf_1 = RepetitionWaveform(body_wf_1, 2)
        body_wf_2 = DummyWaveform(defined_channels={'a'})
        wf_2 = RepetitionWaveform(body_wf_2, 2)
        wf_3 = RepetitionWaveform(body_wf_1, 3)
        wf_1_equal = RepetitionWaveform(body_wf_1, 2)
        self.assertEqual(wf_1_equal, wf_1)
        self.assertNotEqual(wf_1, wf_2)
        self.assertNotEqual(wf_1, wf_3)
        self.assertEqual({wf_1, wf_2, wf_3}, {wf_1, wf_2, wf_3, wf_1_equal})

    def test_unsafe_get_subset_for_channels(self):
        body_wf = DummyWaveform(defined_channels={'a', 'b'})

        chs = {'a'}

        subset = RepetitionWaveform(body_wf, 3).get_subset_for_channels(chs)
        self.assertIsInstance(subset, RepetitionWaveform)
        self.assertIsInstance(subset._body, DummyWaveform)
        self.assertIs(subset._body.defined_channels, chs)
        self.assertEqual(subset._repetition_count, 3)

    def test_unsafe_sample(self):
        body_wf = DummyWaveform(duration=7)

        rwf = RepetitionWaveform(body=body_wf, repetition_count=10)

        sample_times = np.arange(80) * 70./80.
        inner_sample_times = (sample_times.reshape((10, -1)) - (7 * np.arange(10))[:, np.newaxis]).ravel()

        result = rwf.unsafe_sample(channel='A', sample_times=sample_times)
        np.testing.assert_equal(result, inner_sample_times)

        output_expected = np.empty_like(sample_times)
        output_received = rwf.unsafe_sample(channel='A', sample_times=sample_times, output_array=output_expected)
        self.assertIs(output_expected, output_received)
        np.testing.assert_equal(output_received, inner_sample_times)

    def test_float_sample_time(self):
        # issue 624
        body_wf = FunctionWaveform.from_expression(ExpressionScalar('sin(t)'), 1./3., channel='a')
        rwf = RepetitionWaveform(body_wf, 2)

        sample_times = np.arange(160) / 80. / 3.
        sampled = rwf.unsafe_sample(sample_times=sample_times, channel='a')
        inner_sample_times = np.concatenate((sample_times[:80], sample_times[80:] - 1./3.))
        np.testing.assert_equal(sampled, np.sin(inner_sample_times))

    def test_repr(self):
        body_wf = ConstantWaveform(amplitude=1.1, duration=1.3, channel='3')
        wf = RepetitionWaveform(body_wf, 3)
        r = repr(wf)
        self.assertEqual(wf, eval(r))


class SequenceWaveformTest(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def test_init(self):
        dwf_ab = DummyWaveform(duration=1.1, defined_channels={'A', 'B'})
        dwf_abc = DummyWaveform(duration=2.2, defined_channels={'A', 'B', 'C'})

        with self.assertRaises(ValueError):
            SequenceWaveform([])

        with self.assertRaises(ValueError):
            SequenceWaveform((dwf_ab, dwf_abc))

        swf1 = SequenceWaveform((dwf_ab, dwf_ab))
        self.assertEqual(swf1.duration, 2*dwf_ab.duration)
        self.assertEqual(swf1.sequenced_waveforms, (dwf_ab, dwf_ab))

        swf2 = SequenceWaveform((swf1, dwf_ab))
        self.assertEqual(swf2.duration, 3 * dwf_ab.duration)
        self.assertEqual(swf2.sequenced_waveforms, (swf1, dwf_ab))

    def test_from_sequence(self):
        dwf = DummyWaveform(duration=1.1, defined_channels={'A'})

        self.assertIs(dwf, SequenceWaveform.from_sequence((dwf,)))

        swf1 = SequenceWaveform.from_sequence((dwf, dwf))
        swf2 = SequenceWaveform.from_sequence((swf1, dwf))

        assert_constant_consistent(self, swf1)
        assert_constant_consistent(self, swf2)

        self.assertEqual(3*(dwf,), swf2.sequenced_waveforms)

        cwf_2_a = ConstantWaveform(duration=1.1, amplitude=2.2, channel='A')
        cwf_3 = ConstantWaveform(duration=1.1, amplitude=3.3, channel='A')
        cwf_2_b = ConstantWaveform(duration=1.1, amplitude=2.2, channel='A')

        with mock.patch.object(ConstantWaveform, 'from_mapping', return_value=mock.sentinel) as from_mapping:
            new_constant = SequenceWaveform.from_sequence((cwf_2_a, cwf_2_b))
            self.assertIs(from_mapping.return_value, new_constant)
            from_mapping.assert_called_once_with(2*TimeType.from_float(1.1), {'A': 2.2})

        swf3 = SequenceWaveform.from_sequence((cwf_2_a, dwf))
        self.assertEqual((cwf_2_a, dwf), swf3.sequenced_waveforms)
        self.assertIsNone(swf3.constant_value('A'))
        assert_constant_consistent(self, swf3)

        swf3 = SequenceWaveform.from_sequence((cwf_2_a, cwf_3))
        self.assertEqual((cwf_2_a, cwf_3), swf3.sequenced_waveforms)
        self.assertIsNone(swf3.constant_value('A'))
        assert_constant_consistent(self, swf3)

    def test_sample_times_type(self) -> None:
        with mock.patch.object(DummyWaveform, 'unsafe_sample') as unsafe_sample_patch:
            dwfs = (DummyWaveform(duration=1.),
                    DummyWaveform(duration=3.),
                    DummyWaveform(duration=2.))

            swf = SequenceWaveform(dwfs)

            sample_times = np.arange(0, 60) * 0.1
            expected_output = np.concatenate((sample_times[:10], sample_times[10:40] - 1, sample_times[40:] - 4))
            expected_inputs = sample_times[0:10], sample_times[10:40] - 1, sample_times[40:] - 4

            swf.unsafe_sample('A', sample_times=sample_times)
            inputs = [call_args[1]['sample_times']
                      for call_args in unsafe_sample_patch.call_args_list]  # type: List[np.ndarray]
            np.testing.assert_equal(expected_inputs, inputs)
            self.assertEqual([input.dtype for input in inputs], [np.float64 for _ in inputs])

    def test_unsafe_sample(self):
        dwfs = (DummyWaveform(duration=1.),
                DummyWaveform(duration=3.),
                DummyWaveform(duration=2.))

        swf = SequenceWaveform(dwfs)

        sample_times = np.arange(0, 60)*0.1
        expected_output = np.concatenate((sample_times[:10], sample_times[10:40]-1, sample_times[40:]-4))

        output = swf.unsafe_sample('A', sample_times=sample_times)
        np.testing.assert_equal(expected_output, output)

        output_2 = swf.unsafe_sample('A', sample_times=sample_times, output_array=output)
        self.assertIs(output_2, output)

    def test_unsafe_get_subset_for_channels(self):
        dwf_1 = DummyWaveform(duration=2.2, defined_channels={'A', 'B', 'C'})
        dwf_2 = DummyWaveform(duration=3.3, defined_channels={'A', 'B', 'C'})

        wf = SequenceWaveform([dwf_1, dwf_2])

        subset = {'A', 'C'}
        sub_wf = wf.unsafe_get_subset_for_channels(subset)
        self.assertIsInstance(sub_wf, SequenceWaveform)

        self.assertEqual(len(sub_wf.sequenced_waveforms), 2)
        self.assertEqual(sub_wf.sequenced_waveforms[0].defined_channels, subset)
        self.assertEqual(sub_wf.sequenced_waveforms[1].defined_channels, subset)

        self.assertEqual(sub_wf.sequenced_waveforms[0].duration, TimeType.from_float(2.2))
        self.assertEqual(sub_wf.sequenced_waveforms[1].duration, TimeType.from_float(3.3))

    def test_repr(self):
        cwf_2_a = ConstantWaveform(duration=1.1, amplitude=2.2, channel='A')
        cwf_3 = ConstantWaveform(duration=1.1, amplitude=3.3, channel='A')
        swf = SequenceWaveform([cwf_2_a, cwf_3])
        r = repr(swf)
        self.assertEqual(swf, eval(r))


class ConstantWaveformTests(unittest.TestCase):
    def test_waveform_duration(self):
        waveform = ConstantWaveform(10, 1., 'P1')
        self.assertEqual(waveform.duration, 10)

    def test_waveform_sample(self):
        waveform = ConstantWaveform(10, .1, 'P1')
        sample_times = [-1, 0, 1, 2]
        result = waveform.unsafe_sample('P1', sample_times)
        self.assertTrue(np.all(result == .1))

        self.assertIs(waveform, waveform.unsafe_get_subset_for_channels({'A'}))

    def test_from_mapping(self):
        from_single = ConstantWaveform.from_mapping(1., {'A': 2.})
        expected_single = ConstantWaveform(duration=1., amplitude=2., channel='A')
        self.assertEqual(expected_single, from_single)

        from_multi = ConstantWaveform.from_mapping(1., {'A': 2., 'B': 3.})
        expected_from_multi = MultiChannelWaveform([ConstantWaveform(duration=1., amplitude=2., channel='A'),
                                                    ConstantWaveform(duration=1., amplitude=3., channel='B')])
        self.assertEqual(expected_from_multi, from_multi)

    def test_constness(self):
        waveform = ConstantWaveform(10, .1, 'P1')
        self.assertTrue(waveform.is_constant())
        assert_constant_consistent(self, waveform)


class TableWaveformTests(unittest.TestCase):

    def test_from_table(self):
        expected = ConstantWaveform(0.1, 0.2, 'A')

        for interp in (HoldInterpolationStrategy(), JumpInterpolationStrategy(), LinearInterpolationStrategy()):
            wf = TableWaveform.from_table('A',
                                          [TableWaveformEntry(0.0, 0.2, interp),
                                           TableWaveformEntry(0.1, 0.2, interp)])
            self.assertEqual(expected, wf)

    def test_validate_input_errors(self):
        with self.assertRaises(ValueError):
            TableWaveform._validate_input([TableWaveformEntry(0.0, 0.2, HoldInterpolationStrategy())])

        with self.assertRaises(ValueError):
            TableWaveform._validate_input([TableWaveformEntry(0.0, 0.2, HoldInterpolationStrategy()),
                                           TableWaveformEntry(0.0, 0.3, HoldInterpolationStrategy())])

        with self.assertRaises(ValueError):
            TableWaveform._validate_input([TableWaveformEntry(0.1, 0.2, HoldInterpolationStrategy()),
                                           TableWaveformEntry(0.2, 0.2, HoldInterpolationStrategy())])

        with self.assertRaisesRegex(ValueError, "not increasing"):
            TableWaveform._validate_input([TableWaveformEntry(0.0, 0.2, HoldInterpolationStrategy()),
                                           TableWaveformEntry(0.2, 0.2, HoldInterpolationStrategy()),
                                           TableWaveformEntry(0.1, 0.2, HoldInterpolationStrategy())])

        with self.assertRaisesRegex(ValueError, "Negative"):
            TableWaveform._validate_input([TableWaveformEntry(0.0, 0.2, HoldInterpolationStrategy()),
                                           TableWaveformEntry(-0.2, 0.2, HoldInterpolationStrategy()),
                                           TableWaveformEntry(0.1, 0.2, HoldInterpolationStrategy())])

    def test_validate_input_const_detection(self):
        constant_table = [TableWaveformEntry(0.0, 2.5, HoldInterpolationStrategy()),
                          (1.4, 2.5, LinearInterpolationStrategy())]
        linear_table = [TableWaveformEntry(0.0, 0.0, HoldInterpolationStrategy()),
                        TableWaveformEntry(1.4, 2.5, LinearInterpolationStrategy())]

        self.assertEqual((1.4, 2.5), TableWaveform._validate_input(constant_table))
        self.assertEqual(linear_table,
                         TableWaveform._validate_input(linear_table))

    def test_const_detection_regression(self):
        # regression test 707
        from qupulse.pulses import PointPT
        second_point_pt = PointPT([(0, 'v_0+v_1'),
                                   ('t_2', 'v_0', 'linear')],
                                  channel_names=('A',),
                                  measurements=[('M', 0, 1)])
        parameters = dict(t=3,
                          t_2=2,
                          v_0=1,
                          v_1=1.4)
        channel_mapping = {'A': 'A'}
        wf = second_point_pt.build_waveform(parameters=parameters, channel_mapping=channel_mapping)
        self.assertIsInstance(wf, TableWaveform)

    def test_validate_input_duplicate_removal(self):
        validated = TableWaveform._validate_input([TableWaveformEntry(0.0, 0.2, HoldInterpolationStrategy()),
                                                   TableWaveformEntry(0.1, 0.2, LinearInterpolationStrategy()),
                                                   TableWaveformEntry(0.1, 0.3, JumpInterpolationStrategy()),
                                                   TableWaveformEntry(0.1, 0.3, HoldInterpolationStrategy()),
                                                   TableWaveformEntry(0.2, 0.3, LinearInterpolationStrategy()),
                                                   TableWaveformEntry(0.3, 0.3, JumpInterpolationStrategy())])

        self.assertEqual(validated, [TableWaveformEntry(0.0, 0.2, HoldInterpolationStrategy()),
                                     TableWaveformEntry(0.1, 0.2, LinearInterpolationStrategy()),
                                     TableWaveformEntry(0.1, 0.3, HoldInterpolationStrategy()),
                                     TableWaveformEntry(0.3, 0.3, JumpInterpolationStrategy())])

    def test_duration(self) -> None:
        entries = [TableWaveformEntry(0, 0, HoldInterpolationStrategy()),
                   TableWaveformEntry(5, 1, HoldInterpolationStrategy())]
        waveform = TableWaveform.from_table('A', entries)
        self.assertEqual(5, waveform.duration)

    def test_duration_no_entries_exception(self) -> None:
        with self.assertRaises(ValueError):
            TableWaveform.from_table('A', [])

    def test_few_entries(self) -> None:
        with self.assertRaises(ValueError):
            TableWaveform.from_table('A', [])
        with self.assertRaises(ValueError):
            TableWaveform.from_table('A', [TableWaveformEntry(0, 0, HoldInterpolationStrategy())])

    def test_unsafe_get_subset_for_channels(self):
        interp = DummyInterpolationStrategy()
        entries = (TableWaveformEntry(0, 0, interp),
                   TableWaveformEntry(2.1, -33.2, interp),
                   TableWaveformEntry(5.7, 123.4, interp))
        waveform = TableWaveform('A', entries)
        self.assertIs(waveform.unsafe_get_subset_for_channels({'A'}), waveform)

    def test_unsafe_sample(self) -> None:
        interp = DummyInterpolationStrategy()
        entries = (TableWaveformEntry(0, 0, interp),
                   TableWaveformEntry(2.1, -33.2, interp),
                   TableWaveformEntry(5.7, 123.4, interp))
        waveform = TableWaveform('A', entries)
        sample_times = numpy.linspace(.5, 5.5, num=11)

        expected_interp_arguments = [((0, 0), (2.1, -33.2), [0.5, 1.0, 1.5, 2.0]),
                                     ((2.1, -33.2), (5.7, 123.4), [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])]
        expected_result = numpy.copy(sample_times)

        result = waveform.unsafe_sample('A', sample_times)

        self.assertEqual(expected_interp_arguments, interp.call_arguments)
        numpy.testing.assert_equal(expected_result, result)

        output_expected = numpy.empty_like(expected_result)
        output_received = waveform.unsafe_sample('A', sample_times, output_array=output_expected)
        self.assertIs(output_expected, output_received)
        numpy.testing.assert_equal(expected_result, output_received)

    def test_simple_properties(self):
        interp = DummyInterpolationStrategy()
        entries = [TableWaveformEntry(0, 0, interp),
                   TableWaveformEntry(2.1, -33.2, interp),
                   TableWaveformEntry(5.7, 123.4, interp)]
        chan = 'A'
        waveform = TableWaveform.from_table(chan, entries)

        self.assertEqual(waveform.defined_channels, {chan})
        self.assertIs(waveform.unsafe_get_subset_for_channels({'A'}), waveform)
        assert_constant_consistent(self, waveform)

        evaled = eval(repr(waveform))
        self.assertEqual(evaled, waveform)


class WaveformEntryTest(unittest.TestCase):
    def test_interpolation_exception(self):
        with self.assertRaises(TypeError):
            TableWaveformEntry(1, 2, 3)

    def test_repr(self):
        interpolation = DummyInterpolationStrategy()
        self.assertEqual(f"TableWaveformEntry(t={1.}, v={2.}, interp={interpolation})",
                         repr(TableWaveformEntry(t=1., v=2., interp=interpolation)))


class TransformationDummy(TransformationStub):
    def __init__(self, output_channels=None, transformed=None, input_channels=None, constant_invariant=None):
        if output_channels:
            self.get_output_channels = mock.MagicMock(return_value=output_channels)

        if input_channels:
            self.get_input_channels = mock.MagicMock(return_value=input_channels)

        if transformed is not None:
            type(self).__call__ = mock.MagicMock(return_value=transformed)

        if constant_invariant is not None:
            self.is_constant_invariant = mock.MagicMock(return_value=constant_invariant)


class TransformingWaveformTest(unittest.TestCase):
    def test_from_transformation(self):
        const_output = {'c': 4.4, 'd': 5.5, 'e': 6.6}
        trafo = TransformationDummy(output_channels=const_output.keys(), constant_invariant=False)
        const_trafo = TransformationDummy(output_channels=const_output.keys(), constant_invariant=True,
                                          transformed=const_output)
        dummy_wf = DummyWaveform(duration=1.5, defined_channels={'a', 'b'})
        const_wf = ConstantWaveform.from_mapping(3, {'a': 2.2, 'b': 3.3})

        self.assertEqual(TransformingWaveform(inner_waveform=dummy_wf, transformation=trafo),
                         TransformingWaveform.from_transformation(inner_waveform=dummy_wf, transformation=trafo))

        self.assertEqual(TransformingWaveform(inner_waveform=dummy_wf, transformation=const_trafo),
                         TransformingWaveform.from_transformation(inner_waveform=dummy_wf, transformation=const_trafo))

        self.assertEqual(TransformingWaveform(inner_waveform=const_wf, transformation=trafo),
                         TransformingWaveform.from_transformation(inner_waveform=const_wf, transformation=trafo))

        with mock.patch.object(ConstantWaveform, 'from_mapping', return_value=mock.sentinel) as from_mapping:
            self.assertIs(from_mapping.return_value,
                          TransformingWaveform.from_transformation(inner_waveform=const_wf, transformation=const_trafo))
            from_mapping.assert_called_once_with(const_wf.duration, const_output)

    def test_simple_properties(self):
        output_channels = {'c', 'd', 'e'}

        trafo = TransformationDummy(output_channels=output_channels)

        inner_wf = DummyWaveform(duration=1.5, defined_channels={'a', 'b'})
        trafo_wf = TransformingWaveform(inner_waveform=inner_wf, transformation=trafo)

        self.assertIs(trafo_wf.inner_waveform, inner_wf)
        self.assertIs(trafo_wf.transformation, trafo)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(trafo_wf.compare_key, (inner_wf, trafo))
        self.assertIs(trafo_wf.duration, inner_wf.duration)
        self.assertIs(trafo_wf.defined_channels, output_channels)
        trafo.get_output_channels.assert_called_once_with(inner_wf.defined_channels)

    def test_get_subset_for_channels(self):
        output_channels = {'c', 'd', 'e'}

        trafo = TransformationDummy(output_channels=output_channels)

        inner_wf = DummyWaveform(duration=1.5, defined_channels={'a', 'b'})
        trafo_wf = TransformingWaveform(inner_waveform=inner_wf, transformation=trafo)

        subset_wf = trafo_wf.get_subset_for_channels({'c', 'd'})
        self.assertIsInstance(subset_wf, SubsetWaveform)
        self.assertIs(subset_wf.inner_waveform, trafo_wf)
        self.assertEqual(subset_wf.defined_channels, {'c', 'd'})

    def test_unsafe_sample(self):
        time = np.linspace(10, 20, num=25)
        ch_a = np.exp(time)
        ch_b = np.exp(-time)
        ch_c = np.sinh(time)
        ch_d = np.cosh(time)
        ch_e = np.arctan(time)

        sample_output = {'a': ch_a, 'b': ch_b}
        expected_call_data = sample_output

        transformed = {'c': ch_c, 'd': ch_d, 'e': ch_e}

        trafo = TransformationDummy(transformed=transformed, input_channels={'a', 'b'})
        inner_wf = DummyWaveform(duration=1.5, defined_channels={'a', 'b'}, sample_output=sample_output)
        trafo_wf = TransformingWaveform(inner_waveform=inner_wf, transformation=trafo)

        np.testing.assert_equal(ch_c, trafo_wf.unsafe_sample('c', time))
        np.testing.assert_equal(ch_d, trafo_wf.unsafe_sample('d', time))
        np.testing.assert_equal(ch_e, trafo_wf.unsafe_sample('e', time))

        output = np.empty_like(time)
        ch_d_out = trafo_wf.unsafe_sample('d', time, output_array=output)
        self.assertIs(output, ch_d_out)
        np.testing.assert_equal(ch_d_out, ch_d)

        call_list = TransformationDummy.__call__.call_args_list
        self.assertEqual(len(call_list), 1)

        (pos_args, kw_args), = call_list
        self.assertEqual(kw_args, {})

        c_time, c_data = pos_args
        np.testing.assert_equal((time, expected_call_data), pos_args)

    def test_const_value(self):
        output_channels = {'c', 'd', 'e'}
        trafo = TransformationStub()
        inner_wf = WaveformStub()

        trafo_wf = TransformingWaveform(inner_wf, trafo)

        self.assertFalse(trafo_wf.is_constant())
        self.assertIsNone(trafo_wf.constant_value_dict())

        with mock.patch.object(trafo, 'is_constant_invariant', return_value=False) as is_constant_invariant:
            self.assertIsNone(trafo_wf.constant_value('A'))
            is_constant_invariant.assert_called_once_with()

        with mock.patch.object(trafo, 'is_constant_invariant', return_value=True):
            # all inputs constant
            inner_const_values = {'A': 1.1, 'B': 2.2}

            with mock.patch.object(trafo, 'get_input_channels', return_value=inner_const_values.keys()):
                with mock.patch.object(inner_wf, 'constant_value', side_effect=inner_const_values.values()) as constant_value:
                    with mock.patch.object(TransformationStub, '__call__', return_value={'C': mock.sentinel}) as call:
                        self.assertIs(trafo_wf.constant_value('C'), call.return_value['C'])
                        call.assert_called_once_with(0., inner_const_values)
                    self.assertEqual([mock.call(ch) for ch in inner_const_values], constant_value.call_args_list)

                inner_const_values['B'] = None
                with mock.patch.object(inner_wf, 'constant_value', side_effect=inner_const_values.values()) as constant_value:
                    self.assertIsNone(trafo_wf.constant_value('C'))


class SubsetWaveformTest(unittest.TestCase):
    def test_simple_properties(self):
        inner_wf = DummyWaveform(duration=1.5, defined_channels={'a', 'b', 'c'})

        subset_wf = SubsetWaveform(inner_wf, {'a', 'c'})

        self.assertIs(subset_wf.inner_waveform, inner_wf)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(subset_wf.compare_key, (frozenset(['a', 'c']), inner_wf))
        self.assertIs(subset_wf.duration, inner_wf.duration)
        self.assertEqual(subset_wf.defined_channels, {'a', 'c'})

    def test_get_subset_for_channels(self):
        subsetted = DummyWaveform(defined_channels={'a'})
        with mock.patch.object(DummyWaveform,
                               'get_subset_for_channels',
                               mock.Mock(return_value=subsetted)) as get_subset_for_channels:
            inner_wf = DummyWaveform(defined_channels={'a', 'b', 'c'})
            subset_wf = SubsetWaveform(inner_wf, {'a', 'c'})

            actual_subsetted = subset_wf.get_subset_for_channels({'a'})
            get_subset_for_channels.assert_called_once_with({'a'})
            self.assertIs(subsetted, actual_subsetted)

    def test_unsafe_sample(self):
        """Test perfect forwarding"""
        time = {'time'}
        output = {'output'}
        expected_data = {'data'}

        with mock.patch.object(DummyWaveform,
                               'unsafe_sample',
                               mock.Mock(return_value=expected_data)) as unsafe_sample:
            inner_wf = DummyWaveform(defined_channels={'a', 'b', 'c'})
            subset_wf = SubsetWaveform(inner_wf, {'a', 'c'})

            actual_data = subset_wf.unsafe_sample('g', time, output)
            self.assertIs(expected_data, actual_data)
            unsafe_sample.assert_called_once_with('g', time, output)


class ArithmeticWaveformTest(unittest.TestCase):
    def test_from_operator(self):
        lhs = DummyWaveform(duration=1.5, defined_channels={'a', 'b', 'c'})
        rhs = DummyWaveform(duration=1.5, defined_channels={'a', 'b', 'd'})

        lhs_const = ConstantWaveform.from_mapping(1.5, {'a': 1.1, 'b': 2.2, 'c': 3.3})
        rhs_const = ConstantWaveform.from_mapping(1.5, {'a': 1.2, 'b': 2.4, 'd': 3.4})

        self.assertEqual(ArithmeticWaveform(lhs, '+', rhs), ArithmeticWaveform.from_operator(lhs, '+', rhs))
        self.assertEqual(ArithmeticWaveform(lhs_const, '+', rhs), ArithmeticWaveform.from_operator(lhs_const, '+', rhs))
        self.assertEqual(ArithmeticWaveform(lhs, '+', rhs_const), ArithmeticWaveform.from_operator(lhs, '+', rhs_const))

        expected = ConstantWaveform.from_mapping(1.5, {'a': 1.1-1.2, 'b': 2.2-2.4, 'c': 3.3, 'd': -3.4})
        consted = ArithmeticWaveform.from_operator(lhs_const, '-', rhs_const)
        self.assertEqual(expected, consted)

    def test_const_propagation(self):
        lhs = MultiChannelWaveform([
            DummyWaveform(duration=1.5, defined_channels={'a', 'c', 'd', 'i'}),
            ConstantWaveform.from_mapping(1.5, {'e': 1.2, 'f': 1.3, 'h': 4.6})
        ])
        rhs = MultiChannelWaveform([
            DummyWaveform(duration=1.5, defined_channels={'a', 'b', 'e'}),
            ConstantWaveform.from_mapping(1.5, {'f': 2.5, 'g': 3.5, 'i': 6.4})
        ])

        wf = ArithmeticWaveform(lhs, '-', rhs)

        assert_constant_consistent(self, wf)

        expected = {'a': None,
                    'b': None,
                    'c': None,
                    'd': None,
                    'e': None,
                    'f': 1.3-2.5,
                    'g': -3.5,
                    'h': 4.6,
                    'i': None}

        actual = {ch: wf.constant_value(ch) for ch in wf.defined_channels}
        self.assertEqual(expected, actual)

    def test_simple_properties(self):
        lhs = DummyWaveform(duration=1.5, defined_channels={'a', 'b', 'c'})
        rhs = DummyWaveform(duration=1.5, defined_channels={'a', 'b', 'd'})

        arith = ArithmeticWaveform(lhs, '-', rhs)

        self.assertEqual(set('abcd'), arith.defined_channels)
        self.assertIs(lhs, arith.lhs)
        self.assertIs(rhs, arith.rhs)
        self.assertEqual('-', arith.arithmetic_operator)
        self.assertEqual(lhs.duration, arith.duration)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(('-', lhs, rhs), arith.compare_key)

    def test_unsafe_get_subset_for_channels(self):
        lhs = DummyWaveform(duration=1.5, defined_channels={'a', 'b', 'c'})
        rhs = DummyWaveform(duration=1.5, defined_channels={'a', 'b', 'd'})

        arith = ArithmeticWaveform(lhs, '-', rhs)

        self.assertEqual(SubsetWaveform(arith, {'a', 'c'}), arith.unsafe_get_subset_for_channels({'a', 'c'}))

    def test_unsafe_sample(self):
        sample_times = np.linspace(0, 10, num=20)
        rhs_a = np.sin(sample_times)
        rhs_b = np.cos(sample_times)
        rhs_c = np.tan(sample_times)
        rhs_out = dict(a=rhs_a,
                       b=rhs_b,
                       c=rhs_c)

        lhs_a = np.sinh(sample_times)
        lhs_b = np.cosh(sample_times)
        lhs_d = np.tanh(sample_times)
        lhs_out = dict(a=lhs_a,
                       b=lhs_b,
                       d=lhs_d)

        for op_str, op_fn in (('+', np.add), ('-', np.subtract)):
            for output_array in (None, np.zeros_like(sample_times)):
                lhs = DummyWaveform(sample_output=lhs_out)
                rhs = DummyWaveform(sample_output=rhs_out)
                arith = ArithmeticWaveform(lhs, op_str, rhs)

                expected = dict(a=op_fn(lhs_a, rhs_a),
                                b=op_fn(lhs_b, rhs_b),
                                c=op_fn(0, rhs_c),
                                d=op_fn(lhs_d, 0))

                for ch, data in expected.items():
                    result = arith.unsafe_sample(ch, sample_times=sample_times, output_array=output_array)

                    np.testing.assert_equal(data, result)
                    if output_array is not None:
                        self.assertIs(result, output_array)


class FunctionWaveformTest(unittest.TestCase):

    def test_equality(self) -> None:
        wf1a = FunctionWaveform(ExpressionScalar('2*t'), 3, channel='A')
        wf1b = FunctionWaveform(ExpressionScalar('2*t'), 3, channel='A')
        wf3 = FunctionWaveform(ExpressionScalar('2*t+2'), 3, channel='A')
        wf4 = FunctionWaveform(ExpressionScalar('2*t'), 4, channel='A')
        wf5 = FunctionWaveform(ExpressionScalar('2*t'), 3, channel='B')
        self.assertEqual(wf1a, wf1a)
        self.assertEqual(wf1a, wf1b)
        self.assertNotEqual(wf1a, wf3)
        self.assertNotEqual(wf1a, wf4)
        self.assertNotEqual(wf1a, wf5)

    def test_defined_channels(self) -> None:
        wf = FunctionWaveform(ExpressionScalar('t'), 4, channel='A')
        self.assertEqual({'A'}, wf.defined_channels)

    def test_duration(self) -> None:
        wf = FunctionWaveform(expression=ExpressionScalar('2*t'), duration=4/5,
                              channel='A')
        self.assertEqual(TimeType.from_float(4/5), wf.duration)

    def test_unsafe_sample(self):
        fw = FunctionWaveform(ExpressionScalar('sin(2*pi*t) + 3'), 5, channel='A')

        t = np.linspace(0, 5, dtype=float)
        expected_result = np.sin(2*np.pi*t) + 3
        result = fw.unsafe_sample(channel='A', sample_times=t)
        np.testing.assert_equal(result, expected_result)

        out_array = np.empty_like(t)
        result = fw.unsafe_sample(channel='A', sample_times=t, output_array=out_array)
        np.testing.assert_equal(result, expected_result)
        self.assertIs(result, out_array)

    def test_constant_evaluation(self):
        # cause for 596
        fw = FunctionWaveform(ExpressionScalar(3), 5, channel='A')
        t = np.linspace(0, 5, dtype=float)
        expected_result = np.full_like(t, fill_value=3.)
        out_array = np.full_like(t, fill_value=np.nan)
        result = fw.unsafe_sample(channel='A', sample_times=t, output_array=out_array)
        self.assertIs(result, out_array)
        np.testing.assert_equal(result, expected_result)

        result = fw.unsafe_sample(channel='A', sample_times=t)
        np.testing.assert_equal(result, expected_result)

        assert_constant_consistent(self, fw)

    def test_unsafe_get_subset_for_channels(self):
        fw = FunctionWaveform(ExpressionScalar('sin(2*pi*t) + 3'), 5, channel='A')
        self.assertIs(fw.unsafe_get_subset_for_channels({'A'}), fw)

    def test_construction(self):
        with self.assertRaises(ValueError):
            FunctionWaveform(ExpressionScalar('sin(omega*t)'), duration=5, channel='A')

        const = FunctionWaveform.from_expression(ExpressionScalar('4.'), duration=5, channel='A')
        expected_const = ConstantWaveform(duration=5, amplitude=4., channel='A')
        self.assertEqual(expected_const, const)

        linear = FunctionWaveform.from_expression(ExpressionScalar('4.*t'), 5, 'A')
        expected_linear = FunctionWaveform(ExpressionScalar('4.*t'), 5, 'A')
        self.assertEqual(expected_linear, linear)

    def test_repr(self):
        wf = FunctionWaveform(ExpressionScalar('sin(2*pi*t) + 3'), 5, channel='A')
        r = repr(wf)
        self.assertEqual(wf, eval(r))


class FunctorWaveformTests(unittest.TestCase):
    def test_duration(self):
        dummy_wf = DummyWaveform(1.5, defined_channels={'A', 'B'})
        f_wf = FunctorWaveform.from_functor(dummy_wf, {'A': np.negative, 'B': np.positive})
        self.assertIs(dummy_wf.duration, f_wf.duration)

    def test_from_functor(self):
        dummy_wf = DummyWaveform(1.5, defined_channels={'A', 'B'})
        const_wf = ConstantWaveform.from_mapping(1.5, {'A': 1.1, 'B': 2.2})

        wf = FunctorWaveform.from_functor(dummy_wf, {'A': np.negative, 'B': np.positive})
        self.assertEqual(FunctorWaveform(dummy_wf, {'A': np.negative, 'B': np.positive}), wf)
        self.assertFalse(wf.is_constant())
        assert_constant_consistent(self, wf)

        wf = FunctorWaveform.from_functor(const_wf, {'A': np.negative, 'B': np.positive})
        self.assertEqual(ConstantWaveform.from_mapping(1.5, {'A': -1.1, 'B': 2.2}), wf)
        assert_constant_consistent(self, wf)

    def test_const_value(self):
        mixed_wf = MultiChannelWaveform([DummyWaveform(1.5, defined_channels={'A'}),
                                         ConstantWaveform(1.5, 1.1, 'B')])
        wf = FunctorWaveform(mixed_wf, {'A': np.negative, 'B': np.negative})
        self.assertIsNone(wf.constant_value('A'))
        self.assertEqual(-1.1, wf.constant_value('B'))

    def test_unsafe_sample(self):
        inner_wf = DummyWaveform(defined_channels={'A', 'B'})
        functors = dict(A=mock.Mock(return_value=1.), B=mock.Mock(return_value=2.))
        wf = FunctorWaveform(inner_wf, functors)

        with mock.patch.object(inner_wf, 'unsafe_sample', return_value=mock.sentinel) as inner_sample:
            self.assertEqual(wf.unsafe_sample('A', 3.14, 6.75), 1.)
            inner_sample.assert_called_once_with('A', 3.14, 6.75)
            functors['A'].assert_called_once_with(inner_sample.return_value, out=inner_sample.return_value)

    def test_unsafe_get_subset_for_channels(self):
        inner_wf = DummyWaveform(defined_channels={'A', 'B'})
        inner_subset_wf = DummyWaveform(defined_channels={'A'})
        functors = dict(A=mock.Mock(return_value=1.), B=mock.Mock(return_value=2.))
        inner_functors = {'A': functors['A']}
        wf = FunctorWaveform(inner_wf, functors)

        with mock.patch.object(inner_wf, 'unsafe_get_subset_for_channels', return_value=inner_subset_wf) as inner_subset:
            self.assertEqual(FunctorWaveform(inner_subset_wf, inner_functors),
                             wf.unsafe_get_subset_for_channels({'A'}))
            inner_subset.assert_called_once_with({'A'})

    def test_comparison(self):
        inner_wf_1 = DummyWaveform(defined_channels={'A', 'B'})
        inner_wf_2 = DummyWaveform(defined_channels={'A', 'B'})
        functors_1 = dict(A=np.positive, B=np.negative)
        functors_2 = dict(A=np.negative, B=np.negative)

        wf11 = FunctorWaveform(inner_wf_1, functors_1)
        wf12 = FunctorWaveform(inner_wf_1, functors_2)
        wf21 = FunctorWaveform(inner_wf_2, functors_1)
        wf22 = FunctorWaveform(inner_wf_2, functors_2)

        with self.assertWarns(DeprecationWarning):
            self.assertEqual((inner_wf_1, frozenset(functors_1.items())), wf11.compare_key)
        self.assertEqual(wf11, wf11)
        self.assertEqual(wf11, FunctorWaveform(inner_wf_1, functors_1))

        self.assertNotEqual(wf11, wf12)
        self.assertNotEqual(wf11, wf21)
        self.assertNotEqual(wf11, wf22)


class ReversedWaveformTest(unittest.TestCase):
    def test_simple_properties(self):
        dummy_wf = DummyWaveform(1.5, defined_channels={'A', 'B'})
        reversed_wf = ReversedWaveform(dummy_wf)

        self.assertEqual(dummy_wf.duration, reversed_wf.duration)
        self.assertEqual(dummy_wf.defined_channels, reversed_wf.defined_channels)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(dummy_wf.compare_key, reversed_wf.compare_key)
        self.assertNotEqual(reversed_wf, dummy_wf)

    def test_reversed_sample(self):
        time_array = np.array([0.1, 0.2, 0.21, 0.3])
        sample_output = np.array([1.1, 1.2, 1.3, 0.9])

        dummy_wf = DummyWaveform(1.5, defined_channels={'A', 'B'}, sample_output=sample_output.copy())
        reversed_wf = ReversedWaveform(dummy_wf)

        output = reversed_wf.unsafe_sample('A', time_array)
        np.testing.assert_equal(output, sample_output[::-1])
        self.assertEqual(dummy_wf.sample_calls, [('A', list(1.5 - time_array[::-1]), None)])

        mem = np.full_like(time_array, fill_value=np.nan)
        output = reversed_wf.unsafe_sample('A', time_array, output_array=mem)
        self.assertIs(output, mem)
        np.testing.assert_equal(output, sample_output[::-1])
        np.testing.assert_equal(dummy_wf.sample_calls, [
            ('A', list(1.5 - time_array[::-1]), None),
            ('A', list(1.5 - time_array[::-1]), mem[::-1])])