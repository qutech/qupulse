import unittest
import os
import json
import typing
import importlib.util
import sys
import warnings

from tests.hardware.tabor_simulator_based_tests import TaborSimulatorManager
from tests.hardware.dummy_devices import DummyDAC
from tests.backward_compatibility.hardware_test_helper import LoadingAndSequencingHelper

from qupulse.serialization import Serializer, FilesystemBackend, PulseStorage
from qupulse.pulses.pulse_template import PulseTemplate
from qupulse.hardware.setup import HardwareSetup, PlaybackChannel, MarkerChannel, MeasurementMask
try:
    import tabor_control
except ImportError:
    tabor_control = None

if tabor_control is not None:
    from qupulse.hardware.awgs.tabor import PlottableProgram, TaborAWGRepresentation

def do_not_skip(test_class):
    if hasattr(test_class, '__unittest_skip__'):
        test_class.__unittest_skip__ = False
    return test_class


def is_test_skipped(test):
    if hasattr(test, '__unittest_skip__'):
        return test.__unittest_skip__
    else:
        return False


class DummyTest(unittest.TestCase):
    def test_dummy(self):
        self.assertTrue(True)


@unittest.skipIf(tabor_control is None, "tabor_control not available")
class TaborLoadingAndSequencingHelper(LoadingAndSequencingHelper):
    def __init__(self, data_folder, pulse_name):
        super().__init__(data_folder=data_folder, pulse_name=pulse_name)

        self.preparation_commands = self.load_json('tabor_preparation_commands.json')

        expected_binary_programs = self.load_json('binary_programs.json')
        if expected_binary_programs:
            self.expected_binary_programs = [PlottableProgram.from_builtin(prog) if prog else None
                                             for prog in expected_binary_programs]
        else:
            self.expected_binary_programs = None

        self.validate_programs = self.load_function_from_file('binary_program_validation.py', 'validate_programs')
        self.validation_data = self.load_json('binary_program_validation.json')

        self.program_AB = None
        self.program_CD = None

    def initialize_hardware_setup(self):
        self.simulator_manager = TaborSimulatorManager(TaborAWGRepresentation, 'instr_addr',
                                                       dict(paranoia_level=2, reset=True))

        try:
            self.simulator_manager.start_simulator()
        except RuntimeError as err:
            raise unittest.SkipTest(*err.args) from err

        self.awg = self.simulator_manager.connect()
        if self.preparation_commands:
            for cmd in self.preparation_commands:
                self.awg.send_cmd(cmd)

        self.dac = DummyDAC()
        self.hardware_setup = HardwareSetup()

        self.hardware_setup.set_channel('TABOR_A', PlaybackChannel(self.awg.channel_pair_AB, 0))
        self.hardware_setup.set_channel('TABOR_B', PlaybackChannel(self.awg.channel_pair_AB, 1))
        self.hardware_setup.set_channel('TABOR_A_MARKER', MarkerChannel(self.awg.channel_pair_AB, 0))
        self.hardware_setup.set_channel('TABOR_B_MARKER', MarkerChannel(self.awg.channel_pair_AB, 1))

        self.hardware_setup.set_channel('TABOR_C', PlaybackChannel(self.awg.channel_pair_CD, 0))
        self.hardware_setup.set_channel('TABOR_D', PlaybackChannel(self.awg.channel_pair_CD, 1))
        self.hardware_setup.set_channel('TABOR_C_MARKER', MarkerChannel(self.awg.channel_pair_CD, 0))
        self.hardware_setup.set_channel('TABOR_D_MARKER', MarkerChannel(self.awg.channel_pair_CD, 1))

        self.hardware_setup.set_measurement("MEAS_A", MeasurementMask(self.dac, "MASK_A"))
        self.hardware_setup.set_measurement("MEAS_B", MeasurementMask(self.dac, "MASK_B"))
        self.hardware_setup.set_measurement("MEAS_C", MeasurementMask(self.dac, "MASK_C"))
        self.hardware_setup.set_measurement("MEAS_D", MeasurementMask(self.dac, "MASK_D"))

    def read_program(self):
        self.program_AB = self.awg.channel_pair_AB.read_complete_program()
        self.program_CD = self.awg.channel_pair_CD.read_complete_program()
        return self.program_AB, self.program_CD


class CompleteIntegrationTestHelper(unittest.TestCase):
    data_folder = None
    pulse_name = None

    @classmethod
    def setUpClass(cls):
        if cls.data_folder is None:
            raise unittest.SkipTest("Base class")
        cls.test_state = TaborLoadingAndSequencingHelper(cls.data_folder, cls.pulse_name)

    def test_1_1_deserialization(self):
        with self.assertWarns(DeprecationWarning):
            self.test_state.deserialize_pulse()

    def test_1_2_deserialization_2018(self) -> None:
        self.test_state.deserialize_pulse_2018()

    def test_2_1_sequencing(self):
        if self.test_state.pulse is None:
            self.skipTest("deserialization failed")
        self.test_state.sequence_pulse()

    def test_3_1_initialize_hardware_setup(self):
        self.test_state.initialize_hardware_setup()

    def test_4_1_register_program(self):
        if self.test_state.hardware_setup is None:
            self.skipTest("No hardware setup")
        self.test_state.register_program()
        self.assertIn(self.pulse_name, self.test_state.hardware_setup.registered_programs)

    def test_5_1_arm_program(self):
        if self.test_state.hardware_setup is None:
            self.skipTest("No hardware setup")
        if self.pulse_name not in self.test_state.hardware_setup.registered_programs:
            self.skipTest("Program is not registered")
        self.test_state.hardware_setup.arm_program(self.pulse_name)
        self.assertEqual(self.test_state.awg.channel_pair_AB._current_program, self.pulse_name,
                         "Program not armed on AB")
        self.assertEqual(self.test_state.awg.channel_pair_CD._current_program, self.pulse_name,
                         "Program not armed on CD")

    def test_6_1_read_program(self):
        if self.test_state.hardware_setup is None:
            self.skipTest("No hardware setup")
        if self.test_state.awg.channel_pair_AB._current_program != self.pulse_name:
            self.skipTest("Program not armed on AB")
        if self.test_state.awg.channel_pair_CD._current_program != self.pulse_name:
            self.skipTest("Program not armed on CD")
        self.test_state.read_program()

    def test_7_1_verify_program(self):
        if self.test_state.hardware_setup is None:
            self.skipTest("No hardware setup")
        if self.test_state.expected_binary_programs is not None:
            self.assertEqual(self.test_state.expected_binary_programs[0], self.test_state.program_AB)
            self.assertEqual(self.test_state.expected_binary_programs[1], self.test_state.program_CD)
        elif self.test_state.validate_programs:
            self.test_state.validate_programs(self.test_state.program_AB,
                                              self.test_state.program_CD,
                                              self.test_state.validation_data,
                                              self.test_state.parameters)
        else:
            self.skipTest("No expected programs given.")


class ChargeScan1Tests(CompleteIntegrationTestHelper):
    data_folder = os.path.join(os.path.dirname(__file__), 'charge_scan_1')
    pulse_name = 'charge_scan'
