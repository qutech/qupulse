# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Tuple, Set, Dict
from collections import deque

from qupulse.hardware.dacs.dac_base import DAC

class DummyDAC(DAC):
    """Dummy DAC for automated testing, debugging and usage in examples. """

    def __init__(self):
        self._measurement_windows = dict()
        self._operations = dict()
        self.measured_data = deque([])
        self._meas_masks = {}
        self._armed_program = None

    @property
    def armed_program(self):
        return self._armed_program

    def register_measurement_windows(self, program_name: str, windows: Dict[str, Tuple['numpy.ndarray',
                                                                                       'numpy.ndarray']]):
        self._measurement_windows[program_name] = windows

    def register_operations(self, program_name: str, operations):
        self._operations[program_name] = operations

    def arm_program(self, program_name: str):
        self._armed_program = program_name

    def delete_program(self, program_name):
        if program_name in self._operations:
            self._operations.pop(program_name)
        if program_name in self._measurement_windows:
            self._measurement_windows.pop(program_name)

    def clear(self) -> None:
        self._measurement_windows = dict()
        self._operations = dict()
        self._armed_program = None

    def measure_program(self, channels):
        return self.measured_data.pop()

    def set_measurement_mask(self, program_name, mask_name, begins, lengths) -> Tuple['numpy.ndarray', 'numpy.ndarray']:
        self._meas_masks.setdefault(program_name, {})[mask_name] = (begins, lengths)
        return begins, lengths
