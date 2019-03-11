import unittest
import numpy

from qupulse.pulses.plotting import PlottingNotPossibleException, render, iter_waveforms, iter_instruction_block, plot
from qupulse._program.instructions import InstructionBlock
from qupulse.pulses.table_pulse_template import TablePulseTemplate
from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate
from qupulse.pulses.sequencing import Sequencer
from qupulse._program._loop import MultiChannelProgram, Loop

from tests.pulses.sequencing_dummies import DummyWaveform, DummyInstruction, DummyPulseTemplate


class PlotterTests(unittest.TestCase):

    def test_render_unsupported_instructions(self) -> None:
        block = InstructionBlock()
        block.add_instruction(DummyInstruction())

        with self.assertRaises(NotImplementedError):
            render(block)

    def test_render_no_waveforms(self) -> None:
        time, channel_data = render(InstructionBlock())
        self.assertEqual(channel_data, dict())
        numpy.testing.assert_equal(time, numpy.empty(0))

    def test_iter_waveforms(self) -> None:
        wf1 = DummyWaveform(duration=7)
        wf2 = DummyWaveform(duration=5)
        wf3 = DummyWaveform(duration=3)

        repeated_block = InstructionBlock()
        repeated_block.add_instruction_meas([('m', 1, 2)])
        repeated_block.add_instruction_exec(wf2)
        repeated_block.add_instruction_exec(wf1)

        main_block = InstructionBlock()
        main_block.add_instruction_exec(wf1)
        main_block.add_instruction_repj(2, repeated_block)
        main_block.add_instruction_exec(wf3)

        for idx, (expected, received) in enumerate(zip([wf1, wf2, wf1, wf2, wf1, wf3], iter_waveforms(main_block))):
            self.assertIs(expected, received, msg="Waveform {} is wrong".format(idx))

    def test_iter_waveform_exceptions(self) -> None:
        wf1 = DummyWaveform(duration=7)
        wf2 = DummyWaveform(duration=5)
        wf3 = DummyWaveform(duration=3)

        repeated_block = InstructionBlock()
        repeated_block.add_instruction_meas([('m', 1, 2)])
        repeated_block.add_instruction_exec(wf2)
        repeated_block.add_instruction_exec(wf1)

        main_block = InstructionBlock()
        main_block.add_instruction_exec(wf1)
        main_block.add_instruction_repj(2, repeated_block)
        main_block.add_instruction_exec(wf3)
        main_block.add_instruction_goto(repeated_block)

        with self.assertRaises(NotImplementedError):
            list(iter_waveforms(main_block))

        repeated_block.add_instruction(DummyInstruction())
        with self.assertRaises(NotImplementedError):
            list(iter_waveforms(main_block))

        main_block = InstructionBlock()
        main_block.add_instruction_stop()

        with self.assertRaises(StopIteration):
            next(iter_waveforms(main_block))

    def test_iter_instruction_block(self) -> None:
        wf1 = DummyWaveform(duration=7)
        wf2 = DummyWaveform(duration=5)
        wf3 = DummyWaveform(duration=3)

        repeated_block = InstructionBlock()
        repeated_block.add_instruction_meas([('m', 1, 2)])
        repeated_block.add_instruction_exec(wf2)
        repeated_block.add_instruction_exec(wf1)

        main_block = InstructionBlock()
        main_block.add_instruction_exec(wf1)
        main_block.add_instruction_repj(2, repeated_block)
        main_block.add_instruction_exec(wf3)

        waveforms, measurements, total_time = iter_instruction_block(main_block, True)

        for idx, (expected, received) in enumerate(zip([wf1, wf2, wf1, wf2, wf1, wf3], waveforms)):
            self.assertIs(expected, received, msg="Waveform {} is wrong".format(idx))
        self.assertEqual([('m', 8, 2), ('m', 20, 2)], measurements)
        self.assertEqual(total_time, 34)

    def test_iter_instruction_block_exceptions(self) -> None:
        wf1 = DummyWaveform(duration=7)
        wf2 = DummyWaveform(duration=5)
        wf3 = DummyWaveform(duration=3)

        repeated_block = InstructionBlock()
        repeated_block.add_instruction_meas([('m', 1, 2)])
        repeated_block.add_instruction_exec(wf2)

        main_block = InstructionBlock()
        main_block.add_instruction_exec(wf1)
        main_block.add_instruction_repj(2, repeated_block)
        main_block.add_instruction_exec(wf3)

        repeated_block.add_instruction_goto(main_block)

        with self.assertRaises(NotImplementedError):
            iter_instruction_block(main_block, False)

        repeated_block = InstructionBlock()
        repeated_block.add_instruction_meas([('m', 1, 2)])
        repeated_block.add_instruction_exec(wf2)
        repeated_block.add_instruction(DummyInstruction())

        main_block = InstructionBlock()
        main_block.add_instruction_exec(wf1)
        main_block.add_instruction_repj(2, repeated_block)
        main_block.add_instruction_exec(wf3)

        with self.assertRaises(NotImplementedError):
            iter_instruction_block(main_block, False)

    def test_render(self) -> None:
        with self.assertWarnsRegex(DeprecationWarning, ".*InstructionBlock.*"):
            wf1 = DummyWaveform(duration=19)
            wf2 = DummyWaveform(duration=21)

            block = InstructionBlock()
            block.add_instruction_exec(wf1)
            block.add_instruction_meas([('asd', 0, 1)])
            block.add_instruction_exec(wf2)

            wf1_expected = ('A', [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
            wf2_expected = ('A', [x-19 for x in [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]])
            wf1_output_array_len_expected = len(wf1_expected[1])
            wf2_output_array_len_expected = len(wf2_expected[1])

            wf1.sample_output = numpy.linspace(start=4, stop=5, num=len(wf1_expected[1]))
            wf2.sample_output = numpy.linspace(6, 7, num=len(wf2_expected[1]))

            expected_times = numpy.arange(start=0, stop=42, step=2)
            expected_result = numpy.concatenate((wf1.sample_output, wf2.sample_output))

            times, voltages, _ = render(block, sample_rate=0.5)

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

            times, voltages, measurements = render(block, sample_rate=0.5, render_measurements=True)
            self.assertEqual(voltages.keys(), dict(A=0).keys())

            numpy.testing.assert_almost_equal(expected_times, times)
            numpy.testing.assert_almost_equal(expected_result, voltages['A'])
            self.assertEqual(expected_result.shape, voltages['A'].shape)

            self.assertEqual(measurements, [('asd', 19, 1)])

    def test_render_block_time_slice(self) -> None:
        with self.assertWarnsRegex(DeprecationWarning, ".*InstructionBlock.*"):
            with self.assertRaises(ValueError):
                wf1 = DummyWaveform(duration=19)
                wf2 = DummyWaveform(duration=21)

                block = InstructionBlock()
                block.add_instruction_exec(wf1)
                block.add_instruction_exec(wf2)

                times, voltages, _ = render(block, sample_rate=0.5, time_slice=(1, 16))

    def test_render_loop_compare(self) -> None:
        wf1 = DummyWaveform(duration=19)
        wf2 = DummyWaveform(duration=21)

        block = InstructionBlock()
        block.add_instruction_exec(wf1)
        block.add_instruction_meas([('asd', 0, 1), ('asd', 1, 1)])
        block.add_instruction_exec(wf2)

        mcp = MultiChannelProgram(block)
        loop = next(iter(mcp.programs.values()))

        block_times, block_voltages, _ = render(block, sample_rate=0.5)
        loop_times, loop_voltages, _ = render(loop, sample_rate=0.5)

        numpy.testing.assert_equal(block_times, loop_times)
        numpy.testing.assert_equal(block_voltages, loop_voltages)

        block_times, block_voltages, block_measurements = render(block, sample_rate=0.5, render_measurements=True)
        loop_times, loop_voltages, loop_measurements = render(loop, sample_rate=0.5, render_measurements=True)
        numpy.testing.assert_equal(block_times, loop_times)
        numpy.testing.assert_equal(block_voltages, loop_voltages)
        self.assertEqual(block_measurements, loop_measurements)

    def test_render_loop_sliced(self) -> None:
        wf = DummyWaveform(duration=19)

        full_times = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        wf.sample_output = numpy.linspace(start=4, stop=5, num=len(full_times))[3:7]

        full_measurements = [('foo', 1, 3), ('bar', 5, 2), ('foo', 7, 3), ('bar', 11, 2), ('foo', 15, 3)]
        loop = Loop(waveform=wf, measurements=full_measurements)

        expected_times = full_times[3:7]
        expected_measurements = full_measurements[1:4]

        times, voltages, measurements = render(loop, sample_rate=0.5, render_measurements=True, time_slice=(6, 12))

        numpy.testing.assert_almost_equal(expected_times, times)
        numpy.testing.assert_almost_equal(wf.sample_output, voltages['A'])
        self.assertEqual(expected_measurements, measurements)

    def test_render_loop_invalid_slice(self) -> None:
        with self.assertRaises(ValueError):
            render(Loop(waveform=DummyWaveform()), time_slice=(5, 1))
            render(Loop(waveform=DummyWaveform()), time_slice=(-1, 4))
            render(Loop(waveform=DummyWaveform()), time_slice=(-7, -3))

    def test_render_warning(self) -> None:
        wf1 = DummyWaveform(duration=19)
        wf2 = DummyWaveform(duration=21)

        block = InstructionBlock()
        block.add_instruction_exec(wf1)
        block.add_instruction_meas([('asd', 0, 1)])
        block.add_instruction_exec(wf2)

        with self.assertWarns(UserWarning):
            render(block, sample_rate=0.51314323423)

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

    def test_plot_empty_pulse(self) -> None:
        import matplotlib
        matplotlib.use('svg') # use non-interactive backend so that test does not fail on travis

        pt = DummyPulseTemplate()
        with self.assertWarnsRegex(UserWarning, "empty", msg="plot() did not issue a warning for an empty pulse"):
            plot(pt, dict(), show=False)


class PlottingNotPossibleExceptionTests(unittest.TestCase):

    def test(self) -> None:
        t = DummyPulseTemplate()
        exception = PlottingNotPossibleException(t)
        self.assertIs(t, exception.pulse)
        self.assertIsInstance(str(exception), str)
