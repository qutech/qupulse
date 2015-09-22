import unittest
import numpy
from tempfile import TemporaryDirectory
import os

from src.pulses.PulseControlInterface import PulseControlInterface
from src.pulses.Sequencer import InstructionBlock

from tests.pulses.SequencingDummies import DummyPulseTemplate, DummyWaveform, DummySequencer


class PulseControlInterfaceTests(unittest.TestCase):

    def setUp(self) -> None:
        #self.tmpdir = TemporaryDirectory()
        #self.cwd = os.getcwd()
        #os.chdir(self.tmpdir.name)
        dirname = 'pulsecontrolinterfacetest'
        #os.mkdir(dirname) # replace by temporary directory
        self.pulse_control_interface = PulseControlInterface(lambda x: 1, 1.2e9)

    def tearDown(self) -> None:
        pass
        #os.chdir(self.cwd)
        #self.tmpdir.cleanup()

    def test_register_waveform(self) -> None:
        waveform = DummyWaveform(duration=4)
        #sequencer = DummySequencer(pulse_control_interface)
        self.pulse_control_interface.register_waveform(waveform)

    def test_create_pulse_group(self) -> None:
        waveform = DummyWaveform(duration=4)
        block = InstructionBlock()
        block.add_instruction_exec(waveform)
        block.add_instruction_exec(waveform)
        self.pulse_control_interface.register_waveform(waveform)
        code = self.pulse_control_interface.create_pulse_group(block, 'main')
        print(code)
        print(3)