import unittest
from unittest import mock

import numpy as np

from qupulse.hardware.setup import HardwareSetup, PlaybackChannel, MarkerChannel, MeasurementMask
from qupulse._program._loop import Loop

from tests.pulses.sequencing_dummies import DummyWaveform

from tests.hardware.dummy_devices import DummyAWG, DummyDAC


class SingleChannelTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.awg1 = DummyAWG(num_channels=2, num_markers=2)
        self.awg2 = DummyAWG(num_channels=2, num_markers=2)

    def test_eq_play_play(self):
        self.assertEqual(PlaybackChannel(self.awg1, 0),
                         PlaybackChannel(self.awg1, 0))
        self.assertEqual(PlaybackChannel(self.awg1, 0, lambda x: 0),
                         PlaybackChannel(self.awg1, 0))
        self.assertEqual(PlaybackChannel(self.awg1, 0),
                         PlaybackChannel(self.awg1, 0, lambda x: 0))

        self.assertNotEqual(PlaybackChannel(self.awg1, 0),
                            PlaybackChannel(self.awg1, 1))
        self.assertNotEqual(PlaybackChannel(self.awg1, 0),
                            PlaybackChannel(self.awg2, 0))

    def test_eq_play_mark(self):
        self.assertNotEqual(MarkerChannel(self.awg1, 0),
                         PlaybackChannel(self.awg1, 0))
        self.assertNotEqual(PlaybackChannel(self.awg1, 0),
                         MarkerChannel(self.awg1, 0))
        self.assertNotEqual(PlaybackChannel(self.awg1, 0, lambda x: 0),
                         MarkerChannel(self.awg1, 0))
        self.assertNotEqual(MarkerChannel(self.awg1, 0),
                         PlaybackChannel(self.awg1, 0, lambda x: 0))

        self.assertNotEqual(MarkerChannel(self.awg1, 0),
                            PlaybackChannel(self.awg1, 1))
        self.assertNotEqual(PlaybackChannel(self.awg1, 0),
                            MarkerChannel(self.awg2, 0))

    def test_eq_mark_mark(self):
        self.assertEqual(MarkerChannel(self.awg1, 0),
                         MarkerChannel(self.awg1, 0))

        self.assertNotEqual(MarkerChannel(self.awg1, 0),
                            MarkerChannel(self.awg1, 1))
        self.assertNotEqual(MarkerChannel(self.awg1, 0),
                            MarkerChannel(self.awg2, 0))

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            MarkerChannel(self.awg1, 2)

        with self.assertRaises(ValueError):
            PlaybackChannel(self.awg1, 2)


class HardwareSetupTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_test_loops(self):
        wf_1 = DummyWaveform(duration=1.1, defined_channels={'A', 'B'})
        wf_2 = DummyWaveform(duration=1.1, defined_channels={'A', 'C'})
        wf_3 = DummyWaveform(duration=1.1, defined_channels={'C'})
        loop1 = Loop(repetition_count=1, children=[Loop(repetition_count=1, waveform=wf_1),
                                                   Loop(repetition_count=10, children=[
                                                      Loop(repetition_count=50, waveform=wf_1)]),
                                                   Loop(repetition_count=17, children=[Loop(repetition_count=2,
                                                                                            waveform=wf_1),
                                                                                       Loop(repetition_count=1,
                                                                                            waveform=wf_1)])],
                     measurements=[('m1', 0., 1.)])
        loop2 = Loop(repetition_count=1, children=[Loop(repetition_count=1, waveform=wf_2),
                                                   Loop(repetition_count=10, children=[
                                                      Loop(repetition_count=50, waveform=wf_2)]),
                                                   Loop(repetition_count=17, children=[Loop(repetition_count=2,
                                                                                            waveform=wf_2),
                                                                                       Loop(repetition_count=1,
                                                                                            waveform=wf_2)])],
                     measurements=[('m2', 0., 1.)])
        loop3 = Loop(repetition_count=1, children=[Loop(repetition_count=1, waveform=wf_3),
                                                   Loop(repetition_count=10, children=[
                                                       Loop(repetition_count=50, waveform=wf_3)]),
                                                   Loop(repetition_count=17, children=[Loop(repetition_count=2,
                                                                                            waveform=wf_3),
                                                                                       Loop(repetition_count=1,
                                                                                            waveform=wf_3)])])
        return loop1, loop2, loop3

    def test_set_channel(self):
        awg1 = DummyAWG()
        awg2 = DummyAWG(num_channels=4)

        setup = HardwareSetup()

        setup.set_channel('A', PlaybackChannel(awg1, 0))
        setup.set_channel('B', PlaybackChannel(awg2, 0))
        self.assertEqual(setup.registered_channels(),
                         dict(A={PlaybackChannel(awg1, 0)},
                              B={PlaybackChannel(awg2, 0)}))

        with self.assertRaises(ValueError):
            setup.set_channel('C', PlaybackChannel(awg1, 0))

        with self.assertWarns(DeprecationWarning):
            setup.set_channel('A', PlaybackChannel(awg2, 1), True)

        self.assertEqual(setup.registered_channels(),
                         dict(A={PlaybackChannel(awg2, 1), PlaybackChannel(awg1, 0)},
                              B={PlaybackChannel(awg2, 0)}))

        setup.set_channel('A', {PlaybackChannel(awg2, 3), PlaybackChannel(awg2, 2)})
        self.assertEqual(setup.registered_channels(),
                         dict(A={PlaybackChannel(awg2, 3), PlaybackChannel(awg2, 2)},
                              B={PlaybackChannel(awg2, 0)}))


    def test_rm_channel(self):
        awg1 = DummyAWG()
        awg2 = DummyAWG(num_channels=2)

        setup = HardwareSetup()

        setup.set_channel('A', PlaybackChannel(awg1, 0))
        setup.set_channel('B', PlaybackChannel(awg2, 0))

        with self.assertRaises(KeyError):
            setup.rm_channel('b')
        setup.rm_channel('B')

        self.assertEqual(setup.registered_channels(),
                         dict(A={PlaybackChannel(awg1, 0)}))

    def test_arm_program(self):
        loop1, loop2, _ = self.get_test_loops()

        awg1 = DummyAWG()
        awg2 = DummyAWG()
        awg3 = DummyAWG()

        dac1 = DummyDAC()
        dac2 = DummyDAC()

        setup = HardwareSetup()

        setup.set_channel('A', PlaybackChannel(awg1, 0))
        setup.set_channel('B', MarkerChannel(awg2, 0))
        setup.set_channel('C', MarkerChannel(awg3, 0))

        setup.set_measurement('m1', MeasurementMask(dac1, 'DAC_1'))
        setup.set_measurement('m2', MeasurementMask(dac2, 'DAC_2'))

        setup.register_program('test_1', loop1)

        self.assertIsNone(awg1._armed)
        self.assertIsNone(awg2._armed)
        self.assertIsNone(awg3._armed)
        self.assertIsNone(dac1._armed_program)
        self.assertIsNone(dac2._armed_program)

        setup.arm_program('test_1')

        self.assertEqual(awg1._armed, 'test_1')
        self.assertEqual(awg2._armed, 'test_1')
        self.assertIsNone(awg3._armed)
        self.assertEqual(dac1._armed_program, 'test_1')
        self.assertIsNone(dac2._armed_program)

        setup.register_program('test_2', loop2)

        self.assertEqual(awg1._armed, 'test_1')
        self.assertEqual(awg2._armed, 'test_1')
        self.assertIsNone(awg3._armed)
        self.assertEqual(dac1._armed_program, 'test_1')
        self.assertIsNone(dac2._armed_program)

        setup.arm_program('test_2')

        self.assertEqual(awg1._armed, 'test_2')
        self.assertIsNone(awg2._armed)
        self.assertEqual(awg3._armed, 'test_2')
        # currently not defined
        # self.assertEqual(dac1._armed_program, 'test_1')
        self.assertEqual(dac2._armed_program, 'test_2')

    def test_register_program(self):
        awg1 = DummyAWG()
        awg2 = DummyAWG(num_channels=2, num_markers=5)

        dac = DummyDAC()

        setup = HardwareSetup()

        loop, *rest = self.get_test_loops()

        class ProgStart:
            def __init__(self):
                self.was_started = False

            def __call__(self):
                self.was_started = True
        program_started = ProgStart()

        trafo1 = lambda x: x

        setup.set_channel('A', PlaybackChannel(awg1, 0, trafo1))
        setup.set_channel('B', MarkerChannel(awg2, 1))

        setup.set_measurement('m1', MeasurementMask(dac, 'DAC'))

        setup.register_program('p1', loop, program_started)

        self.assertEqual(tuple(setup.registered_programs.keys()), ('p1',))
        self.assertEqual(setup.registered_programs['p1'].run_callback,  program_started)
        self.assertEqual(setup.registered_programs['p1'].awgs_to_upload_to, {awg1, awg2})

        self.assertFalse(program_started.was_started)

        self.assertEqual(len(awg1._programs), 1)
        self.assertEqual(len(awg2._programs), 1)

        _, channels, markers, trafos = awg1._programs['p1']
        self.assertEqual(channels, ('A', ))
        self.assertEqual(markers, (None,))
        self.assertEqual(trafos, (trafo1,))

        _, channels, markers, trafos = awg2._programs['p1']
        self.assertEqual(channels, (None, None))
        self.assertEqual(markers, (None, 'B', None, None, None))

        self.assertEqual(awg1._armed, None)
        self.assertEqual(awg2._armed, None)

        expected_measurement_windows = {'p1':
                                            {'DAC':
                                                 (np.array([0,]), np.array([1,]))
                                            }
                                        }
        np.testing.assert_equal(dac._measurement_windows,
                                expected_measurement_windows)

    def test_remove_program(self):
        loop1, loop2, _ = self.get_test_loops()

        awg1 = DummyAWG()
        awg2 = DummyAWG()
        awg3 = DummyAWG()

        dac1 = DummyDAC()
        dac2 = DummyDAC()

        setup = HardwareSetup()

        setup.set_channel('A', PlaybackChannel(awg1, 0))
        setup.set_channel('B', MarkerChannel(awg2, 0))
        setup.set_channel('C', MarkerChannel(awg3, 0))

        setup.set_measurement('m1', MeasurementMask(dac1, 'DAC_1'))
        setup.set_measurement('m2', MeasurementMask(dac2, 'DAC_2'))

        setup.register_program('test_1', loop1)

        setup.register_program('test_2', loop2)

        setup.arm_program('test_1')

        setup.remove_program('test_1')

        self.assertEqual(setup.registered_programs.keys(), {'test_2'})

        self.assertIsNone(awg1._armed)
        self.assertIsNone(awg2._armed)

    def test_run_program(self):

        loop, *rest = self.get_test_loops()

        awg1 = DummyAWG()
        awg2 = DummyAWG()
        awg3 = DummyAWG()

        dac1 = DummyDAC()
        dac2 = DummyDAC()

        setup = HardwareSetup()

        setup.set_channel('A', PlaybackChannel(awg1, 0))
        setup.set_channel('B', MarkerChannel(awg2, 0))
        setup.set_channel('C', MarkerChannel(awg3, 0))

        setup.set_measurement('m1', MeasurementMask(dac1, 'DAC_1'))
        setup.set_measurement('m2', MeasurementMask(dac2, 'DAC_2'))

        class ProgStart:
            def __init__(self):
                self.was_started = False

            def __call__(self):
                self.was_started = True
        program_started = ProgStart()

        setup.register_program('test', loop, run_callback=program_started)

        self.assertIsNone(awg1._armed)
        self.assertIsNone(awg2._armed)
        self.assertIsNone(awg3._armed)
        self.assertIsNone(dac1._armed_program)
        self.assertIsNone(dac2._armed_program)

        setup.run_program('test')

        self.assertEqual(awg1._armed, 'test')
        self.assertEqual(awg2._armed, 'test')
        self.assertIsNone(awg3._armed)
        self.assertEqual(dac1._armed_program, 'test')
        self.assertIsNone(dac2._armed_program)

        self.assertTrue(program_started.was_started)

    def test_register_program_exceptions(self):
        setup = HardwareSetup()

        loop, *rest = self.get_test_loops()

        with self.assertRaises(TypeError):
            setup.register_program('p1', loop, 4)

        with self.assertRaises(KeyError):
            setup.register_program('p1', loop, lambda: None)

        awg = DummyAWG(num_channels=2, num_markers=5)

        setup.set_channel('A', PlaybackChannel(awg, 0, lambda x: x))
        setup.set_channel('B', MarkerChannel(awg, 1))

        with self.assertRaises(KeyError):
            setup.register_program('p1', loop, lambda: None)

    def test_known_dacs(self) -> None:
        setup = HardwareSetup()
        dac1 = DummyDAC()
        dac2 = DummyDAC()
        setup.set_measurement('m1', MeasurementMask(dac1, 'mask_1'))
        setup.set_measurement('m2', MeasurementMask(dac2, 'mask_2'))
        expected = {dac1, dac2}
        self.assertEqual(expected, setup.known_dacs)

    def test_known_awgs(self) -> None:
        setup = HardwareSetup()
        awg1 = DummyAWG(num_channels=2, num_markers=0)
        awg2 = DummyAWG(num_channels=0, num_markers=1)
        setup.set_channel('A', PlaybackChannel(awg1, 0))
        setup.set_channel('B', PlaybackChannel(awg1, 1))
        setup.set_channel('M1', MarkerChannel(awg2, 0))
        expected = {awg1, awg2}
        self.assertEqual(expected, setup.known_awgs)

    def test_update_parameters(self) -> None:
        setup = HardwareSetup()
        awg1 = DummyAWG(num_channels=2, num_markers=0)
        awg2 = DummyAWG(num_channels=1, num_markers=1)
        dac1 = DummyDAC()
        dac2 = DummyDAC()
        setup.set_channel('A', PlaybackChannel(awg1, 0))
        setup.set_channel('B', PlaybackChannel(awg1, 1))
        setup.set_channel('C', PlaybackChannel(awg2, 0))
        setup.set_measurement('m1', MeasurementMask(dac1, 'DAC_1'))
        setup.set_measurement('m2', MeasurementMask(dac2, 'DAC_2'))
        loop1, loop2, _ = self.get_test_loops()
        name1 = 'prog1'
        name2 = 'prog2'
        setup.register_program(name1, loop1)
        setup.register_program(name2, loop2)

        parameters = dict(a=1, b=5)

        with mock.patch.object(DummyAWG, 'set_volatile_parameters') as setpara:
            setup.update_parameters(name1, parameters)
            setpara.assert_called_once_with(name1, parameters)
        with mock.patch.object(DummyAWG, 'set_volatile_parameters') as setpara:
            setup.update_parameters(name2, parameters)
            assert setpara.call_count == 2
            setpara.assert_called_with(name2, parameters)

    def test_clear_programs(self) -> None:
        setup = HardwareSetup()
        awg1 = DummyAWG(num_channels=2, num_markers=0)
        awg2 = DummyAWG(num_channels=1, num_markers=1)
        dac1 = DummyDAC()
        dac2 = DummyDAC()
        setup.set_channel('A', PlaybackChannel(awg1, 0))
        setup.set_channel('B', PlaybackChannel(awg1, 1))
        setup.set_channel('C', PlaybackChannel(awg2, 0))
        setup.set_channel('M1', MarkerChannel(awg2, 0))

        setup.set_measurement('m1', MeasurementMask(dac1, 'DAC_1'))
        setup.set_measurement('m2', MeasurementMask(dac2, 'DAC_2'))

        loop1, loop2, loop3 = self.get_test_loops()

        setup.register_program('prog1', loop1)
        setup.register_program('prog2', loop2)
        setup.register_program('prog3', loop3)

        self.assertTrue(setup.registered_programs)

        setup.clear_programs()

        self.assertFalse(setup.registered_programs)
