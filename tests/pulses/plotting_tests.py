import unittest
import numpy

from qctoolkit.pulses.plotting import PlottingNotPossibleException, render
from qctoolkit.pulses.instructions import InstructionBlock
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate
from qctoolkit.pulses.sequencing import Sequencer

from tests.pulses.sequencing_dummies import DummyWaveform, DummyInstruction, DummyPulseTemplate


class PlotterTests(unittest.TestCase):

    def test_render_unsupported_instructions(self) -> None:
        block = InstructionBlock()
        block.add_instruction(DummyInstruction())

        with self.assertRaises(NotImplementedError):
            render(block)

    def test_render_no_waveforms(self) -> None:
        self.assertEqual(([], []), render(InstructionBlock()))

    def test_render(self) -> None:
        wf1 = DummyWaveform(duration=19)
        wf2 = DummyWaveform(duration=21)

        block = InstructionBlock()
        block.add_instruction_exec(wf1)
        block.add_instruction_exec(wf2)

        wf1_expected = ('A', [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
        wf2_expected = ('A', [x-19 for x in [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]])
        wf1_output_array_len_expected = len(wf1_expected[1])
        wf2_output_array_len_expected = len(wf2_expected[1])

        wf1.sample_output = numpy.linspace(start=4, stop=5, num=len(wf1_expected[1]))
        wf2.sample_output = numpy.linspace(6, 7, num=len(wf2_expected[1]))

        expected_times = numpy.arange(start=0, stop=42, step=2)
        expected_result = numpy.concatenate((wf1.sample_output, wf2.sample_output))

        times, voltages = render(block, sample_rate=0.5)

        self.assertEqual(len(wf1.sample_calls), 1)
        self.assertEqual(len(wf2.sample_calls), 1)

        self.assertEqual(wf1_expected[0], wf1.sample_calls[0][0])
        self.assertEqual(wf2_expected[0], wf2.sample_calls[0][0])

        numpy.testing.assert_almost_equal(wf1_expected[1], wf1.sample_calls[0][1])
        numpy.testing.assert_almost_equal(wf2_expected[1], wf2.sample_calls[0][1])

        self.assertEqual(wf1_output_array_len_expected, len(wf1.sample_calls[0][2]))
        self.assertEqual(wf2_output_array_len_expected, len(wf2.sample_calls[0][2]))

        self.assertEqual(voltages.keys(), dict(A=0).keys())

        numpy.testing.assert_almost_equal(expected_times, times)
        numpy.testing.assert_almost_equal(expected_result, voltages['A'])
        self.assertEqual(expected_result.shape, voltages['A'].shape)

    def integrated_test_with_sequencer_and_pulse_templates(self) -> None:
        # Setup test data
        square = TablePulseTemplate()
        square.add_entry('up', 'v', 'hold')
        square.add_entry('down', 0, 'hold')
        square.add_entry('length', 0)

        mapping1 = {
            'up': 'uptime',
            'down': 'uptime + length',
            'v': 'voltage',
            'length': '0.5 * pulse_length'
        }

        outer_parameters = ['uptime', 'length', 'pulse_length', 'voltage']

        parameters = {}
        parameters['uptime'] = 5
        parameters['length'] = 10
        parameters['pulse_length'] = 100
        parameters['voltage'] = 10

        sequence = SequencePulseTemplate([(square, mapping1), (square, mapping1)], outer_parameters)

        # run the sequencer and render the plot
        sample_rate = 20
        sequencer = Sequencer()
        sequencer.push(sequence, parameters)
        block = sequencer.build()
        times, voltages = render(block, sample_rate=sample_rate)

        # compute expected values
        expected_times = numpy.linspace(0, 100, sample_rate)
        expected_voltages = numpy.zeros_like(expected_times)
        expected_voltages[100:300] = numpy.ones(200) * parameters['voltage']

        # compare
        self.assertEqual(expected_times, times)
        self.assertEqual(expected_voltages, voltages)


class PlottingNotPossibleExceptionTests(unittest.TestCase):

    def test(self) -> None:
        t = DummyPulseTemplate()
        exception = PlottingNotPossibleException(t)
        self.assertIs(t, exception.pulse)
        self.assertIsInstance(str(exception), str)