import unittest
import numpy

from src.pulses.Plotting import Plotter, PlottingNotPossibleException
from src.pulses.Instructions import InstructionBlock
from tests.pulses.SequencingDummies import DummyWaveform
from src.pulses.TablePulseTemplate import TablePulseTemplate
from src.pulses.SequencePulseTemplate import SequencePulseTemplate
from src.pulses.Sequencer import Sequencer


class PlotterTests(unittest.TestCase):

    def test_render_unsupported_instructions(self) -> None:
        block = InstructionBlock()
        block.add_instruction_stop()
        plotter = Plotter()
        with self.assertRaises(NotImplementedError):
            plotter.render(block)

    def test_render_no_waveforms(self) -> None:
        self.assertEqual(([], []), Plotter().render(InstructionBlock()))

    def test_render(self) -> None:
        wf1 = DummyWaveform(duration=19)
        wf2 = DummyWaveform(duration=21)

        block = InstructionBlock()
        block.add_instruction_exec(wf1)
        block.add_instruction_exec(wf2)

        plotter = Plotter(sample_rate=0.5)
        times, voltages = plotter.render(block)

        wf1_expected = [([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], 0)]
        wf2_expected = [([20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40], 1)]
        expected_result = list(range(0, 41, 2))
        self.assertEqual(wf1_expected, wf1.sample_calls)
        self.assertEqual(wf2_expected, wf2.sample_calls)
        self.assertEqual(expected_result, list(times))
        self.assertEqual(expected_result, list(voltages))

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
        plotter = Plotter(sample_rate=sample_rate)
        sequencer = Sequencer(plotter)
        sequencer.push(sequence, parameters)
        block = sequencer.build()
        times, voltages = plotter.render(block)

        # compute expected values
        expected_times = numpy.linspace(0, 100, sample_rate)
        expected_voltages = numpy.zeros_like(expected_times)
        expected_voltages[100:300] = numpy.ones(200) * parameters['voltage']

        # compare
        self.assertEqual(expected_times, times)
        self.assertEqual(expected_voltages, voltages)


class PlottingNotPossibleExceptionTests(unittest.TestCase):

    def test(self) -> None:
        t = TablePulseTemplate()
        exception = PlottingNotPossibleException(t)
        self.assertIs(t, exception.pulse)
        self.assertIsInstance(str(exception), str)