from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, Iterable

import numpy

__all__ = ['DAC']


class DAC(metaclass=ABCMeta):
    """Representation of a data acquisition card"""

    @abstractmethod
    def register_measurement_windows(self, program_name: str, windows: Dict[str, Tuple[numpy.ndarray,
                                                                                       numpy.ndarray]]) -> None:
        """Register measurement windows for a given program. Overwrites previously defined measurement windows for
        this program.

        Args:
            program_name: Name of the program
            windows: Measurement windows by name.
                     First array are the start points of measurement windows in nanoseconds.
                     Second array are the corresponding measurement window's lengths in nanoseconds.
        """

    @abstractmethod
    def set_measurement_mask(self, program_name: str, mask_name: str,
                             begins: numpy.ndarray,
                             lengths: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Set/overwrite a single the measurement mask for a program. Begins and lengths are in nanoseconds.

        Args:
            program_name: Name of the program
            mask_name: Name of the mask/measurement windows
            begins: Staring points in nanoseconds
            lengths: Lengths in nanoseconds

        Returns:
            Measurement windows in DAC samples (begins, lengths)
        """

    @abstractmethod
    def register_operations(self, program_name: str, operations) -> None:
        """Register operations that are to be applied to the measurement results.

        Args:
            program_name: Name of the program
            operations: DAC specific instructions what to do with the data recorded by the device.
        """
    
    @abstractmethod
    def arm_program(self, program_name: str) -> None:
        """Prepare the device for measuring the given program and wait for a trigger event."""

    @abstractmethod
    def delete_program(self, program_name) -> None:
        """Delete program from internal memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clears all registered programs."""

    @abstractmethod
    def measure_program(self, channels: Iterable[str]) -> Dict[str, numpy.ndarray]:
        """Get the last measurement's results of the specified operations/channels"""
