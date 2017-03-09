from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

__all__ = ['DAC']


class DAC(metaclass=ABCMeta):
    """Representation of a data acquisition card"""

    @abstractmethod
    def register_measurement_windows(self, program_name: str, windows: Dict[str, Tuple['numpy.ndarray',
                                                                                       'numpy.ndarray']]) -> None:
        """"""

    @abstractmethod
    def register_operations(self, program_name: str, operations) -> None:
        """"""

    @abstractmethod
    def arm_program(self, program_name: str) -> None:
        """"""

    def delete_program(self, program_name) -> None:
        """"""
