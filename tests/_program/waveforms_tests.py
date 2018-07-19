import unittest

import numpy
import numpy as np

from qctoolkit.utils.types import time_from_float
from qctoolkit.pulses.interpolation import HoldInterpolationStrategy, LinearInterpolationStrategy,\
    JumpInterpolationStrategy
from qctoolkit._program.waveforms import MultiChannelWaveform, RepetitionWaveform, SequenceWaveform,\
    TableWaveformEntry, TableWaveform

from tests.pulses.sequencing_dummies import DummyWaveform, DummyInterpolationStrategy


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


class MultiChannelWaveformTest(unittest.TestCase):
    def test_init_no_args(self) -> None:
        with self.assertRaises(ValueError):
            MultiChannelWaveform(dict())
        with self.assertRaises(ValueError):
            MultiChannelWaveform(None)

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
        self.assertEqual(time_from_float(1.3), waveform.duration)

    def test_init_several_channels(self) -> None:
        dwf_a = DummyWaveform(duration=2.2, defined_channels={'A'})
        dwf_b = DummyWaveform(duration=2.2, defined_channels={'B'})
        dwf_c = DummyWaveform(duration=2.3, defined_channels={'C'})

        waveform = MultiChannelWaveform([dwf_a, dwf_b])
        self.assertEqual({'A', 'B'}, waveform.defined_channels)
        self.assertEqual(time_from_float(2.2), waveform.duration)

        with self.assertRaises(ValueError):
            MultiChannelWaveform([dwf_a, dwf_c])
        with self.assertRaises(ValueError):
            MultiChannelWaveform([waveform, dwf_c])
        with self.assertRaises(ValueError):
            MultiChannelWaveform((dwf_a, dwf_a))

        dwf_c_valid = DummyWaveform(duration=2.2, defined_channels={'C'})
        waveform_flat = MultiChannelWaveform((waveform, dwf_c_valid))
        self.assertEqual(len(waveform_flat.compare_key), 3)

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

    def test_duration(self):
        wf = RepetitionWaveform(DummyWaveform(duration=2.2), 3)
        self.assertEqual(wf.duration, time_from_float(2.2)*3)

    def test_defined_channels(self):
        body_wf = DummyWaveform(defined_channels={'a'})
        self.assertIs(RepetitionWaveform(body_wf, 2).defined_channels, body_wf.defined_channels)

    def test_compare_key(self):
        body_wf = DummyWaveform(defined_channels={'a'})
        wf = RepetitionWaveform(body_wf, 2)
        self.assertEqual(wf.compare_key, (body_wf.compare_key, 2))

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
        self.assertEqual(len(swf1.compare_key), 2)

        swf2 = SequenceWaveform((swf1, dwf_ab))
        self.assertEqual(swf2.duration, 3 * dwf_ab.duration)

        self.assertEqual(len(swf2.compare_key), 3)

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

        self.assertEqual(len(sub_wf.compare_key), 2)
        self.assertEqual(sub_wf.compare_key[0].defined_channels, subset)
        self.assertEqual(sub_wf.compare_key[1].defined_channels, subset)

        self.assertEqual(sub_wf.compare_key[0].duration, time_from_float(2.2))
        self.assertEqual(sub_wf.compare_key[1].duration, time_from_float(3.3))



class TableWaveformTests(unittest.TestCase):

    def test_validate_input_errors(self):
        with self.assertRaises(ValueError):
            TableWaveform._validate_input([TableWaveformEntry(0.0, 0.2, HoldInterpolationStrategy())])

        with self.assertRaises(ValueError):
            TableWaveform._validate_input([TableWaveformEntry(0.0, 0.2, HoldInterpolationStrategy()),
                                           TableWaveformEntry(0.0, 0.3, HoldInterpolationStrategy())])

        with self.assertRaises(ValueError):
            TableWaveform._validate_input([TableWaveformEntry(0.1, 0.2, HoldInterpolationStrategy()),
                                           TableWaveformEntry(0.2, 0.2, HoldInterpolationStrategy())])

        with self.assertRaises(ValueError):
            TableWaveform._validate_input([TableWaveformEntry(0.0, 0.2, HoldInterpolationStrategy()),
                                           TableWaveformEntry(0.2, 0.2, HoldInterpolationStrategy()),
                                           TableWaveformEntry(0.1, 0.2, HoldInterpolationStrategy())])

    def test_validate_input_duplicate_removal(self):
        validated = TableWaveform._validate_input([TableWaveformEntry(0.0, 0.2, HoldInterpolationStrategy()),
                                                   TableWaveformEntry(0.1, 0.2, LinearInterpolationStrategy()),
                                                   TableWaveformEntry(0.1, 0.3, JumpInterpolationStrategy()),
                                                   TableWaveformEntry(0.1, 0.3, HoldInterpolationStrategy()),
                                                   TableWaveformEntry(0.2, 0.3, LinearInterpolationStrategy()),
                                                   TableWaveformEntry(0.3, 0.3, JumpInterpolationStrategy())])

        self.assertEqual(validated, (TableWaveformEntry(0.0, 0.2, HoldInterpolationStrategy()),
                                     TableWaveformEntry(0.1, 0.2, LinearInterpolationStrategy()),
                                     TableWaveformEntry(0.1, 0.3, HoldInterpolationStrategy()),
                                     TableWaveformEntry(0.3, 0.3, JumpInterpolationStrategy())))



    def test_duration(self) -> None:
        entries = [TableWaveformEntry(0, 0, HoldInterpolationStrategy()), TableWaveformEntry(5, 1, HoldInterpolationStrategy())]
        waveform = TableWaveform('A', entries)
        self.assertEqual(5, waveform.duration)

    def test_duration_no_entries_exception(self) -> None:
        with self.assertRaises(ValueError):
            waveform = TableWaveform('A', [])
            self.assertEqual(0, waveform.duration)

    def test_few_entries(self) -> None:
        with self.assertRaises(ValueError):
            TableWaveform('A', [[]])
        with self.assertRaises(ValueError):
            TableWaveform('A', [TableWaveformEntry(0, 0, HoldInterpolationStrategy())])

    def test_unsafe_get_subset_for_channels(self):
        interp = DummyInterpolationStrategy()
        entries = [TableWaveformEntry(0, 0, interp),
                   TableWaveformEntry(2.1, -33.2, interp),
                   TableWaveformEntry(5.7, 123.4, interp)]
        waveform = TableWaveform('A', entries)
        self.assertIs(waveform.unsafe_get_subset_for_channels({'A'}), waveform)

    def test_unsafe_sample(self) -> None:
        interp = DummyInterpolationStrategy()
        entries = [TableWaveformEntry(0, 0, interp),
                   TableWaveformEntry(2.1, -33.2, interp),
                   TableWaveformEntry(5.7, 123.4, interp)]
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
        waveform = TableWaveform(chan, entries)

        self.assertEqual(waveform.defined_channels, {chan})
        self.assertIs(waveform.unsafe_get_subset_for_channels({'A'}), waveform)


class WaveformEntryTest(unittest.TestCase):
    def test_interpolation_exception(self):
        with self.assertRaises(TypeError):
            TableWaveformEntry(1, 2, 3)