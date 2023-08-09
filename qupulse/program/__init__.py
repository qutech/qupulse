from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Sequence, ContextManager, Mapping, Tuple, Generic, TypeVar
from numbers import Real

import numpy as np

from qupulse._program.waveforms import Waveform
from qupulse.utils.types import MeasurementWindow, TimeType
from qupulse._program.volatile import VolatileRepetitionCount

from typing import Protocol, runtime_checkable


NumVal = TypeVar('NumVal', bound=Real)


@dataclass
class SimpleExpression(Generic[NumVal]):
    """This is a potential hardware evaluable expression of the form

    C + C1*R1 + C2*R2 + ...
    where R1, R2, ... are potential runtime parameters.

    The main use case is the expression of for loop dependent variables where the Rs are loop indices. There the
    expressions can be calculated via simple increments.
    """

    base: NumVal
    offsets: Sequence[Tuple[str, NumVal]]

    def value(self, scope: Mapping[str, NumVal]) -> NumVal:
        value = self.base
        for name, factor in self.offsets:
            value += scope[name] * factor
        return value


RepetitionCount = Union[int, VolatileRepetitionCount]
Value = Union[Real, SimpleExpression]




@runtime_checkable
class Program(Protocol):
    """This protocol is used to inspect and or manipulate programs"""

    @property
    def duration(self) -> TimeType:
        raise NotImplementedError()


class ProgramBuilder(Protocol):
    """This protocol is used by PulseTemplate to build the program.

    There is a default implementation which is the loop class.

    Other hardware backends can use this protocol to implement easy translation of pulse templates.

    """

    def hold_voltage(self, duration: Value, voltages: Mapping[str, Value]):
        """Supports dynamic i.e. for loop generated offsets and duration"""

    # further specialized commandos like play_harmoic might be added here

    def play_arbitrary_waveform(self, waveform: Waveform):
        """"""

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        """Add given measurements at the current position"""

    def with_repetition(self, repetition_count: RepetitionCount) -> ContextManager['ProgramBuilder']:
        """Measurements that are added to the new builder are dropped if the builder is empty upon exit"""

    def with_sequence(self) -> ContextManager['ProgramBuilder']:
        """

        Measurements that are added in to the returned program builder are discarded if the sequence is empty on exit.

        Args:
            measurements: Measurements to attach to the potential child. Is not repeated with repetition_count.
            repetition_count:
        Returns:
        """

    def new_subprogram(self) -> ContextManager['ProgramBuilder']:
        """Create a context managed program builder whose contents are translated into a single waveform upon exit if
        it is not empty."""

    def with_iteration(self, index_name: str, rng: range) -> ContextManager['ProgramBuilder']:
        pass

    def to_program(self) -> Optional[Program]:
        """Further addition of new elements might fail after finalizing the program."""
