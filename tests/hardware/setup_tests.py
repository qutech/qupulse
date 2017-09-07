import unittest
import itertools

import numpy as np

from qctoolkit.pulses.instructions import InstructionBlock, EXECInstruction
from qctoolkit.hardware.setup import HardwareSetup, ChannelID, PlaybackChannel, _SingleChannel, MarkerChannel

from tests.pulses.sequencing_dummies import DummyWaveform

from tests.hardware.dummy_devices import DummyAWG, DummyDAC
from tests.hardware.program_tests import get_two_chan_test_block, WaveformGenerator


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

    def test_set_channel(self):
        awg1 = DummyAWG()
        awg2 = DummyAWG(num_channels=2)

        setup = HardwareSetup()

        setup.set_channel('A', PlaybackChannel(awg1, 0))
        setup.set_channel('B', PlaybackChannel(awg2, 0))
        self.assertEqual(setup.registered_channels(),
                         dict(A={PlaybackChannel(awg1, 0)},
                              B={PlaybackChannel(awg2, 0)}))

        with self.assertRaises(ValueError):
            setup.set_channel('C', PlaybackChannel(awg1, 0))
        setup.set_channel('A', PlaybackChannel(awg2, 1))
        self.assertEqual(setup.registered_channels(),
                         dict(A={PlaybackChannel(awg2, 1), PlaybackChannel(awg1, 0)},
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
        wf = DummyWaveform(duration=1.1, defined_channels={'A', 'B'})

        block = InstructionBlock()
        block.add_instruction_exec(wf)

        awg1 = DummyAWG()
        awg2 = DummyAWG()
        awg3 = DummyAWG()

        dac1 = DummyDAC()
        dac2 = DummyDAC()

        setup = HardwareSetup()

        setup.set_channel('A', PlaybackChannel(awg1, 0))
        setup.set_channel('B', MarkerChannel(awg2, 0))
        setup.set_channel('C', MarkerChannel(awg3, 0))

        setup.register_dac(dac1)
        setup.register_dac(dac2)

        setup.register_program('test', block)

        self.assertIsNone(awg1._armed)
        self.assertIsNone(awg2._armed)
        self.assertIsNone(awg3._armed)
        self.assertIsNone(dac1._armed_program)
        self.assertIsNone(dac2._armed_program)

        setup.arm_program('test')

        self.assertEqual(awg1._armed, 'test')
        self.assertEqual(awg2._armed, 'test')
        self.assertIsNone(awg3._armed)
        self.assertEqual(dac1._armed_program, 'test')
        self.assertEqual(dac2._armed_program, 'test')

    def test_register_program(self):
        awg1 = DummyAWG()
        awg2 = DummyAWG(num_channels=2, num_markers=5)

        dac = DummyDAC()

        setup = HardwareSetup()

        wfg = WaveformGenerator(num_channels=2, duration_generator=itertools.repeat(1))
        block = get_two_chan_test_block(wfg)

        block.instructions[0].waveform._sub_waveforms[0].measurement_windows_ = [('M', 0.1, 0.2)]

        class ProgStart:
            def __init__(self):
                self.was_started = False

            def __call__(self):
                self.was_started = True
        program_started = ProgStart()

        trafo1 = lambda x: x

        setup.set_channel('A', PlaybackChannel(awg1, 0, trafo1))
        setup.set_channel('B', MarkerChannel(awg2, 1))

        setup.register_dac(dac)

        setup.register_program('p1', block, program_started)

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

        np.testing.assert_equal(dac._measurement_windows,
                                dict(p1=dict(M=(np.array([0.1, 0.1]), np.array([0.2, 0.2])))))

    def test_run_program(self):
        wf = DummyWaveform(duration=1.1, defined_channels={'A', 'B'})

        block = InstructionBlock()
        block.add_instruction_exec(wf)

        awg1 = DummyAWG()
        awg2 = DummyAWG()
        awg3 = DummyAWG()

        dac1 = DummyDAC()
        dac2 = DummyDAC()

        setup = HardwareSetup()

        setup.set_channel('A', PlaybackChannel(awg1, 0))
        setup.set_channel('B', MarkerChannel(awg2, 0))
        setup.set_channel('C', MarkerChannel(awg3, 0))

        setup.register_dac(dac1)
        setup.register_dac(dac2)

        class ProgStart:
            def __init__(self):
                self.was_started = False

            def __call__(self):
                self.was_started = True
        program_started = ProgStart()

        setup.register_program('test', block, run_callback=program_started)

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
        self.assertEqual(dac2._armed_program, 'test')

        self.assertTrue(program_started.was_started)

    def test_register_program_exceptions(self):
        setup = HardwareSetup()

        wfg = WaveformGenerator(num_channels=2, duration_generator=itertools.repeat(1))
        block = get_two_chan_test_block(wfg)

        with self.assertRaises(TypeError):
            setup.register_program('p1', block, 4)

        with self.assertRaises(KeyError):
            setup.register_program('p1', block, lambda: None)

        awg = DummyAWG(num_channels=2, num_markers=5)

        setup.set_channel('A', PlaybackChannel(awg, 0, lambda x: x))
        setup.set_channel('B', MarkerChannel(awg, 1))

        with self.assertRaises(ValueError):
            setup.register_program('p1', block, lambda: None)

