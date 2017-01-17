from abc import ABCMeta, abstractmethod
from typing import Dict


__all__ = ['DAC']


class DAC(ABCMeta):
    """Representation of a data acquisition card"""

    @abstractmethod
    def register_measurement_windows(self, program_name: str, windows: Dict[str, 'numpy.ndarray']):
        """"""

    def register_operations(self, program_name: str, operations):
        """"""

    def arm_program(self, program_name: str):
        """"""

    def delete_program(self, program_name):
        """"""
