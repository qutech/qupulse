import unittest
from unittest import mock
import sys
import importlib

import numpy

from qupulse.pulses import ConstantPT
from qupulse.plotting import PlottingNotPossibleException, render, plot
from qupulse.pulses.table_pulse_template import TablePulseTemplate
from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate
from qupulse._program._loop import Loop

from tests.pulses.sequencing_dummies import DummyWaveform, DummyPulseTemplate


class PlotterTests(unittest.TestCase):
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

        program = Loop(children=[Loop(waveform=wf1), Loop(waveform=wf2)])

        with self.assertWarns(UserWarning):
            render(program, sample_rate=0.51314323423)

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

        # render the plot
        sample_rate = 20
        program = sequence.create_program(parameters=parameters)
        times, voltages = render(program, sample_rate=sample_rate)

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

    def test_plot_pulse_automatic_sample_rate(self) -> None:
        import matplotlib
        matplotlib.use('svg') # use non-interactive backend so that test does not fail on travis
        pt=ConstantPT(100, {'a': 1})
        plot(pt, sample_rate=None)

    def test_bug_447(self):
        """Adapted code from https://github.com/qutech/qupulse/issues/447"""
        TablePT = TablePulseTemplate
        SequencePT = SequencePulseTemplate

        period = 8.192004194306148e-05
        repetitions = 80
        sampling_rate = 1e7
        sec_to_ns = 1e9

        table_pt = TablePT({'test': [(0, 0), (period * sec_to_ns, 0, 'linear')]})

        template = SequencePT(*((table_pt,) * repetitions))

        program = template.create_program()

        with self.assertWarns(UserWarning):
            (_, voltages, _) = render(program, sampling_rate / sec_to_ns)


class PlottingNotPossibleExceptionTests(unittest.TestCase):

    def test(self) -> None:
        t = DummyPulseTemplate()
        exception = PlottingNotPossibleException(t)
        self.assertIs(t, exception.pulse)
        self.assertIsInstance(str(exception), str)


class PlottingIsinstanceTests(unittest.TestCase):
    @unittest.skip("Breaks other tests")
    def test_bug_422(self):
        import matplotlib
        matplotlib.use('svg')  # use non-interactive backend so that test does not fail on travis

        to_reload = ['qupulse._program._loop',
                     'qupulse.pulses.pulse_template',
                     'qupulse.pulses.table_pulse_template']

        with mock.patch.dict(sys.modules, sys.modules.copy()):
            for module in to_reload:
                sys.modules.pop(module, None)
            for module in to_reload:
                sys.modules[module] = importlib.reload(importlib.import_module(module))

            from qupulse.pulses.table_pulse_template import TablePulseTemplate

            pt = TablePulseTemplate({'X': [(0, 1), (1, 1)]})

            plot(pt, parameters={})

    def test_bug_422_mock(self):
        pt = TablePulseTemplate({'X': [(0, 1), (100, 1)]})
        program = pt.create_program()

        mock_program = mock.Mock(spec=dir(program))

        for attr in dir(Loop):
            if not attr.endswith('_'):
                setattr(mock_program, attr, getattr(program, attr))
        mock_program.__len__ = lambda x: 1
        mock_program.__iter__ = lambda x: iter(program)
        mock_program.__getitem__ = lambda x, idx: program[idx]

        self.assertNotIsInstance(mock_program, Loop)

        render(mock_program, sample_rate=1)
