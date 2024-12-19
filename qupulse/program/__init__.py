from typing import Optional, Union, Sequence, ContextManager, Mapping, Tuple, Generic, TypeVar, Iterable, Dict
from typing import Protocol, runtime_checkable, Set

from qupulse._program.waveforms import Waveform
from qupulse.utils.types import MeasurementWindow, TimeType
from qupulse._program.volatile import VolatileRepetitionCount
from qupulse.parameter_scope import Scope
from qupulse.expressions import sympy as sym_expr
from qupulse.expressions.simple import SimpleExpression
from qupulse import ChannelID

RepetitionCount = Union[int, VolatileRepetitionCount, SimpleExpression[int]]
HardwareTime = Union[TimeType, SimpleExpression[TimeType]]
HardwareVoltage = Union[float, SimpleExpression[float]]


@runtime_checkable
class Program(Protocol):
    """This protocol is used to inspect and or manipulate programs"""

    @property
    def duration(self) -> TimeType:
        raise NotImplementedError()


class ProgramBuilder(Protocol):
    """This protocol is used by PulseTemplate to build the program via the visitor pattern.

    There is a default implementation which is the loop class.

    Other hardware backends can use this protocol to implement easy translation of pulse templates."""

    def inner_scope(self, scope: Scope) -> Scope:
        """This function is necessary to inject program builder specific parameter implementations into the build
        process."""

    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[str, HardwareVoltage]):
        """Supports dynamic i.e. for loop generated offsets and duration"""

    # further specialized commandos like play_harmoic might be added here

    def play_arbitrary_waveform(self, waveform: Waveform):
        """"""

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        """Unconditionally add given measurements relative to the current position."""

    def with_repetition(self, repetition_count: RepetitionCount,
                        measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        """Measurements that are added to the new builder are dropped if the builder is empty upon exit"""

    def with_sequence(self,
                      measurements: Optional[Sequence[MeasurementWindow]] = None) -> ContextManager['ProgramBuilder']:
        """

        Measurements that are added in to the returned program builder are discarded if the sequence is empty on exit.

        Args:
            measurements: Measurements to attach to the potential child. Is not repeated with repetition_count.
            repetition_count:
        Returns:
        """

    def new_subprogram(self, global_transformation: 'Transformation' = None) -> ContextManager['ProgramBuilder']:
        """Create a context managed program builder whose contents are translated into a single waveform upon exit if
        it is not empty."""

    def with_iteration(self, index_name: str, rng: range,
                       pt_obj: 'ForLoopPT', #hack this in for now.
                       # can be placed more suitably, like in pulsemetadata later on, but need some working thing now.
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        pass
    
    def evaluate_nested_stepping(self, scope: Scope, parameter_names: set[str]) -> bool:
        return False
    
    def time_reversed(self) -> ContextManager['ProgramBuilder']:
        pass
    
    def to_program(self, defined_channels: Set[ChannelID]) -> Optional[Program]:
        """Further addition of new elements might fail after finalizing the program."""


def default_program_builder() -> ProgramBuilder:
    from qupulse.program.loop import LoopBuilder
    return LoopBuilder()


# TODO: hackedy, hackedy
sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES = sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES + (SimpleExpression,)
