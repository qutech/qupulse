"""This is a private package meaning there are no stability guarantees."""
from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, ContextManager, Mapping

import numpy as np

from qupulse._program.waveforms import Waveform
from qupulse.utils.types import MeasurementWindow
from qupulse._program.volatile import VolatileRepetitionCount

try:
    from typing import Protocol
except ImportError:
    Protocol = object


RepetitionCount = Union[int, VolatileRepetitionCount]


class Program(Protocol):
    """This protocol is used to inspect and or manipulate programs"""

    def to_single_waveform(self) -> Waveform:
        pass

    def get_measurement_windows(self) -> Mapping[str, np.ndarray]:
        pass


class ProgramBuilder(Protocol):
    """This protocol is used by PulseTemplate to build the program."""

    def append_leaf(self, waveform: Waveform,
                    measurements: Optional[Sequence[MeasurementWindow]] = None,
                    repetition_count: int = 1):
        pass

    def potential_child(self, measurements: Optional[Sequence[MeasurementWindow]],
                        repetition_count: Union[VolatileRepetitionCount, int] = 1) -> ContextManager['ProgramBuilder']:
        """

        Args:
            measurements: Measurements to attach to the potential child. Is not repeated with repetition_count.
            repetition_count:

        Returns:

        """

    def to_program(self) -> Optional[Program]:
        pass


def default_program_builder() -> ProgramBuilder:
    from qupulse._program._loop import Loop
    return Loop()
