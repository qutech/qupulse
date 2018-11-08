from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, Iterable

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

    @abstractmethod
    def delete_program(self, program_name) -> None:
        """"""

    @abstractmethod
    def clear(self) -> None:
        """Clears all registered programs.

        Caution: This affects all programs and waveforms on the AWG, not only those uploaded using qupulse!
        """

    @abstractmethod
    def measure_program(self, channels: Iterable[str]) -> Dict:
        """Get all measurements at once and write them in a dictionary"""