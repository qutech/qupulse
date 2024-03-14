import unittest

import unittest
import os
import json
import typing
import importlib.util
import sys


from tests.hardware.dummy_devices import DummyDAC

from qupulse.serialization import Serializer, FilesystemBackend, PulseStorage
from qupulse.pulses.pulse_template import PulseTemplate
from qupulse.hardware.setup import HardwareSetup, PlaybackChannel, MarkerChannel, MeasurementMask

from tests.backward_compatibility.hardware_test_helper import LoadingAndSequencingHelper

try:
    from qupulse.hardware.awgs import zihdawg
except ImportError:
    zihdawg = None


def get_test_hdawg():
    assert zihdawg is not None

    serial_env_var = 'QUPULSE_HDAWG_TEST_DEV'
    interface_env_var = 'QUPULSE_HDAWG_TEST_INTERFACE'
    device_serial = os.environ.get(serial_env_var, None)
    if device_serial is None:
        raise unittest.SkipTest(f"No test HDAWG specified via environment variable {serial_env_var}")
    kwargs = dict(device_serial=device_serial)
    device_interface = os.environ.get(interface_env_var, None)
    if device_interface is not None:
        kwargs['device_interface'] = device_interface

    return zihdawg.HDAWGRepresentation(**kwargs)


class HDAWGLoadingAndSequencingHelper(LoadingAndSequencingHelper):
    def __init__(self, data_folder, pulse_name):
        if zihdawg is None:
            raise unittest.SkipTest("zhinst import failed")

        super().__init__(data_folder=data_folder, pulse_name=pulse_name)

        self.preparation_commands = self.load_json('tabor_preparation_commands.json')

        self.awg: zihdawg.HDAWGRepresentation = None
        self.channel_group: zihdawg.HDAWGChannelGroup = None

    def initialize_hardware_setup(self):
        self.awg = get_test_hdawg()

        if self.preparation_commands:
            preparation_commands = [(key.format(device_serial=self.awg.serial), value)
                                    for key, value in self.preparation_commands.items()
                                    ]
            self.awg.api_session.set(preparation_commands)

        for idx in range(1, 9):
            # switch off all outputs
            self.awg.output(idx, False)

        self.awg.channel_grouping = zihdawg.HDAWGChannelGrouping.CHAN_GROUP_1x8

        self.channel_group, = self.awg.channel_tuples

        self.dac = DummyDAC()

        hardware_setup = HardwareSetup()

        hardware_setup.set_channel('TABOR_A', PlaybackChannel(self.channel_group, 0))
        hardware_setup.set_channel('TABOR_B', PlaybackChannel(self.channel_group, 1))
        hardware_setup.set_channel('TABOR_A_MARKER', MarkerChannel(self.channel_group, 0))
        hardware_setup.set_channel('TABOR_B_MARKER', MarkerChannel(self.channel_group, 1))

        hardware_setup.set_channel('TABOR_C', PlaybackChannel(self.channel_group, 2))
        hardware_setup.set_channel('TABOR_D', PlaybackChannel(self.channel_group, 3))
        hardware_setup.set_channel('TABOR_C_MARKER', MarkerChannel(self.channel_group, 3))
        hardware_setup.set_channel('TABOR_D_MARKER', MarkerChannel(self.channel_group, 4))

        hardware_setup.set_measurement("MEAS_A", MeasurementMask(self.dac, "MASK_A"))
        hardware_setup.set_measurement("MEAS_B", MeasurementMask(self.dac, "MASK_B"))
        hardware_setup.set_measurement("MEAS_C", MeasurementMask(self.dac, "MASK_C"))
        hardware_setup.set_measurement("MEAS_D", MeasurementMask(self.dac, "MASK_D"))

        self.hardware_setup = hardware_setup


class CompleteIntegrationTestHelper(unittest.TestCase):
    data_folder = None
    pulse_name = None

    @classmethod
    def setUpClass(cls):
        if cls.data_folder is None:
            raise unittest.SkipTest("Base class")
        cls.test_state = HDAWGLoadingAndSequencingHelper(cls.data_folder, cls.pulse_name)

    def test_1_1_deserialization(self):
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
        self.assertEqual(self.test_state.channel_group._current_program, self.pulse_name,
                         "Program not armed")


class ChargeScan1Tests(CompleteIntegrationTestHelper):
    data_folder = os.path.join(os.path.dirname(__file__), 'charge_scan_1')
    pulse_name = 'charge_scan'
