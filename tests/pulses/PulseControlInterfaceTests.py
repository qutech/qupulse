import unittest
import numpy
from tempfile import TemporaryDirectory
import os

from src.pulses.PulseControlInterface import PulseControlInterface
from src.pulses.Sequencer import InstructionBlock

from tests.pulses.SequencingDummies import DummyPulseTemplate, DummyWaveform, DummySequencer


class PulseControlInterfaceTests(unittest.TestCase):

    def setUp(self) -> None:
        self.pulse_control_interface = PulseControlInterface(lambda x: 1, 1.2e9)

    def test_nothing(self) -> None:
        pass