import unittest
from unittest import mock
import math

import numpy as np

from qupulse.hardware.awgs.virtual import VirtualAWG
from qupulse._program._loop import Loop

from tests.pulses.sequencing_dummies import DummyWaveform


class VirtualAWGTests(unittest.TestCase):

    def test_init(self):
        vawg = VirtualAWG('asd', 5)

        self.assertEqual(vawg.identifier, 'asd')
        self.assertEqual(vawg.num_channels, 5)

    def test_no_markers(self):
        vawg = VirtualAWG('asd', 5)

        self.assertEqual(vawg.num_markers, 0)

    def test_sample_rate(self):
        vawg = VirtualAWG('asd', 5)

        self.assertTrue(math.isnan(vawg.sample_rate))

    def test_arm(self):
        name = 'prognam'
        vawg = VirtualAWG('asd', 5)

        vawg.arm(name)

        self.assertEqual(vawg._current_program, name)

    def test_function_handle_callback(self):
        callback = mock.MagicMock()

        vawg = VirtualAWG('asd', 3)

        vts = (lambda x: x, lambda x: 2*x, None)

        dummy_program = Loop()
        dummy_waveform = DummyWaveform(sample_output={'X': np.sin(np.arange(10)),
                                                      'Y': np.cos(np.arange(10))}, duration=42,
                                       defined_channels={'X', 'Y'})
        vawg.upload('test', dummy_program, ('X', 'Y', None), (), vts)
        vawg.arm('test')
        vawg.set_function_handle_callback(callback)
        with mock.patch('qupulse.hardware.awgs.virtual.to_waveform', autospec=True, return_value=dummy_waveform) as dummy_to_waveform:
            vawg.run_current_program()

            dummy_to_waveform.assert_called_once_with(dummy_program)

        callback.assert_called_once()
        (duration, sample_callbacks), kwargs = callback.call_args
        self.assertEqual(kwargs, {})

        self.assertEqual(duration, dummy_waveform.duration)
        x, y, n = sample_callbacks
        self.assertIsNone(n)

        t = np.arange(10)*1.
        np.testing.assert_equal(x(t), dummy_waveform.sample_output['X'])
        np.testing.assert_equal(y(t), 2*dummy_waveform.sample_output['Y'])
