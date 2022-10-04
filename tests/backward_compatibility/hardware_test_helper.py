import unittest
import os
import json
import typing
import importlib.util
import sys

from qupulse.serialization import Serializer, FilesystemBackend, PulseStorage
from qupulse.pulses.pulse_template import PulseTemplate

class LoadingAndSequencingHelper:
    def __init__(self, data_folder, pulse_name):
        self.data_folder = data_folder
        self.pulse_name = pulse_name

        self.parameters = self.load_json('parameters.json')
        self.window_mapping = self.load_json('measurement_mapping.json')
        self.channel_mapping = self.load_json('channel_mapping.json')

        self.validate_programs = self.load_function_from_file('binary_program_validation.py', 'validate_programs')
        self.validation_data = self.load_json('binary_program_validation.json')

        self.pulse = None
        self.program = None

        self.simulator_manager = None

        self.hardware_setup = None  # type: HardwareSetup
        self.dac = None  # type: DummyDAC

    def load_json(self, file_name):
        complete_file_name = os.path.join(self.data_folder, file_name)
        if os.path.exists(complete_file_name):
            with open(complete_file_name, 'r') as file_handle:
                return json.load(file_handle)
        else:
            return None

    def load_function_from_file(self, file_name, function_name):
        full_file_name = os.path.join(self.data_folder, file_name)
        if not os.path.exists(full_file_name):
            return None
        module_name = os.path.normpath(os.path.splitext(full_file_name)[0]).replace(os.sep, '.')

        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            try:
                spec = importlib.util.spec_from_file_location(module_name, full_file_name)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except ImportError:
                return None
        return getattr(module, function_name, None)

    def deserialize_pulse(self):
        serializer = Serializer(FilesystemBackend(os.path.join(self.data_folder, 'pulse_storage')))
        self.pulse = typing.cast(PulseTemplate, serializer.deserialize(self.pulse_name))

    def deserialize_pulse_2018(self) -> None:
        pulse_storage = PulseStorage(FilesystemBackend(os.path.join(self.data_folder, 'pulse_storage_converted_2018')))
        self.pulse = typing.cast(PulseTemplate, pulse_storage[self.pulse_name])

    def sequence_pulse(self):
        self.program = self.pulse.create_program(
            parameters=self.parameters,
            measurement_mapping=self.window_mapping,
            channel_mapping=self.channel_mapping)

    def register_program(self):
        self.hardware_setup.register_program(self.pulse_name, self.program)

    def arm_program(self):
        self.hardware_setup.arm_program(self.pulse_name)

