import unittest
import itertools


from qctoolkit.hardware.setup import HardwareSetup, ChannelID, PlaybackChannel, _SingleChannel, MarkerChannel
from qctoolkit.hardware.awgs.base import DummyAWG

from.program_tests import get_two_chan_test_block, WaveformGenerator


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

    def test_register_program(self):
        awg1 = DummyAWG()
        awg2 = DummyAWG(num_channels=2, num_markers=5)

        setup = HardwareSetup()

        wfg = WaveformGenerator(num_channels=2, duration_generator=itertools.repeat(1))
        block = get_two_chan_test_block(wfg)

        class ProgStart:
            def __init__(self):
                self.was_started = False

            def __call__(self):
                self.was_started = True
        program_started = ProgStart()

        trafo1 = lambda x: x
        trafo2 = lambda x: x

        setup.set_channel('A', PlaybackChannel(awg1, 0, trafo1))
        setup.set_channel('B', PlaybackChannel(awg2, 1, trafo2))
        setup.register_program('p1', block, program_started)

        self.assertEqual(tuple(setup.registered_programs.keys()), ('p1',))
        self.assertEqual(setup.registered_programs['p1'].run_callback,  program_started)
        self.assertEqual(setup.registered_programs['p1'].awgs_to_upload_to, {awg1, awg2})

        self.assertEqual(len(awg1._programs), 1)
        self.assertEqual(len(awg2._programs), 1)

        _, channels, markers, trafos = awg1._programs['p1']
        self.assertEqual(channels, ('A', ))
        self.assertEqual(markers, (None,))
        self.assertEqual(trafos, (trafo1,))

        _, channels, markers, trafos = awg2._programs['p1']
        self.assertEqual(channels, (None, 'B'))
        self.assertEqual(markers, (None,)*5)
        self.assertEqual(trafos, (None, trafo2))

        self.assertEqual(awg1._armed, None)
        self.assertEqual(awg2._armed, None)
        setup.arm_program('p1')
        self.assertEqual(awg1._armed, 'p1')
        self.assertEqual(awg2._armed, 'p1')



