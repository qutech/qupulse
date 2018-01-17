import unittest
import subprocess
import time

import pytabor

from qctoolkit.hardware.awgs.tabor import TaborAWGRepresentation, TaborException


class TaborSimulatorBasedTest(unittest.TestCase):
    simulator_executable = 'WX2184C.exe'

    simulator_process = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instrument = None

    @classmethod
    def killRunningSimulators(cls):
        subprocess.run(['Taskkill', '/IM WX2184C.exe'])

    @classmethod
    def setUpClass(cls):
        cls.killRunningSimulators()

        cls.simulator_process = subprocess.Popen([cls.simulator_executable, '/switch-on', '/gui-in-tray'])

        start = time.time()
        while pytabor.open_session('127.0.0.1') is None:
            if cls.simulator_process.returncode:
                raise RuntimeError('Simulator exited with return code {}'.format(cls.simulator_process.returncode))
            if time.time() - start > 20.:
                raise RuntimeError('Could not connect to simulator')
            time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        if cls.simulator_process is not None:
            cls.simulator_process.kill()

    def setUp(self):
        self.instrument = TaborAWGRepresentation('127.0.0.1',
                                                 reset=True,
                                                 paranoia_level=2)

        if self.instrument.main_instrument.visa_inst is None:
            raise RuntimeError('Could not connect to simulator')

    def tearDown(self):
        self.instrument.reset()
        for device in self.instrument.all_devices:
            device.close()
        self.instrument = None


class TaborAWGRepresentationTests(TaborSimulatorBasedTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_sample_rate(self):
        for ch in (1, 2, 3, 4):
            self.assertIsInstance(self.instrument.sample_rate(ch), int)

        with self.assertRaises(TaborException):
            self.instrument.sample_rate(0)

        self.instrument.send_cmd(':INST:SEL 1;')
        self.instrument.send_cmd(':FREQ:RAST 2.3e9')

        self.assertEqual(2300000000, self.instrument.sample_rate(1))

    def test_amplitude(self):
        for ch in (1, 2, 3, 4):
            self.assertIsInstance(self.instrument.amplitude(ch), float)

        with self.assertRaises(TaborException):
            self.instrument.amplitude(0)

        self.instrument.send_cmd(':INST:SEL 1; :OUTP:COUP DC')
        self.instrument.send_cmd(':VOLT 0.7')

        self.assertAlmostEqual(.7, self.instrument.amplitude(1))

    def test_select_marker(self):
        with self.assertRaises(TaborException):
            self.instrument.select_marker(6)

        self.instrument.select_marker(2)
        selected = self.instrument.send_query(':SOUR:MARK:SEL?')
        self.assertEqual(selected, '2')

        self.instrument.select_marker(1)
        selected = self.instrument.send_query(':SOUR:MARK:SEL?')
        self.assertEqual(selected, '1')

    def test_select_channel(self):
        with self.assertRaises(TaborException):
            self.instrument.select_channel(6)

        self.instrument.select_channel(1)
        self.assertEqual(self.instrument.send_query(':INST:SEL?'), '1')

        self.instrument.select_channel(4)
        self.assertEqual(self.instrument.send_query(':INST:SEL?'), '4')

