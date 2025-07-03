from abc import abstractmethod
from typing import runtime_checkable, Protocol, Mapping, Optional, Sequence, Iterable, ContextManager

from qupulse import MeasurementWindow
from qupulse.parameter_scope import Scope
from qupulse.program import HardwareTime, HardwareVoltage, Waveform, RepetitionCount, Transformation

from qupulse.utils.types import TimeType


@runtime_checkable
class Program(Protocol):
    """This protocol is used to inspect and or manipulate programs. As you can see the functionality is very limited
    because most of a program class' capability are specific to the implementation."""

    @property
    @abstractmethod
    def duration(self) -> TimeType:
        """The duration of the program in nanoseconds."""


@runtime_checkable
class ProgramBuilder(Protocol):
    """This protocol is used by :py:meth:`.PulseTemplate.create_program` to build a program via a variation of the
    visitor pattern.

    The pulse templates call the methods that correspond to their functionality on the program builder. For example,
    :py:class:`.ConstantPulseTemplate` translates itself into a simple :py:meth:`.ProgramBuilder.hold_voltage` call while
    :class:`SequencePulseTemplate` uses :py:meth:`.ProgramBuilder.with_sequence` to signify a logical unit with
    attached measurements and passes the resulting object to the sequenced sub-templates.

    Due to backward compatibility the handling of measurements is a bit weird since they have to be omitted in certain
    cases. However, this is not relevant for HDAWG specific implementations because these are expected to ignore
    :py:meth:`.ProgramBuilder.measure` calls.

    This interface makes heavy use of context managers and generators/iterators which allows for flexible iteration
    and repetition implementation.
    """

    @abstractmethod
    def inner_scope(self, scope: Scope) -> Scope:
        """This function is part of the iteration protocol and necessary to inject program builder specific parameter
        implementations into the build process. :py:meth:`.ProgramBuilder.with_iteration` and
        `.ProgramBuilder.with_iteration` callers *must* call this function inside the iteration.

        Args:
            scope: The parameter scope outside the iteration.

        Returns:
            The parameter scope inside the iteration.
        """

    @abstractmethod
    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[str, HardwareVoltage]):
        """Hold the specified voltage for a given time. Advances the current time by ``duration``. The values are
        hardware dependent type which are inserted into the parameter scope via :py:meth:`.ProgramBuilder.with_iteration`.

        Args:
            duration: Duration of voltage hold
            voltages: Voltages for each channel
        """

    # further specialized commandos like play_harmonic might be added here

    @abstractmethod
    def play_arbitrary_waveform(self, waveform: Waveform):
        """Insert the playback of an arbitrary waveform. If possible pulse templates should use more specific commands
        like :py:meth:`.ProgramBuilder.hold_voltage` (the only more specific command at the time of this writing).

        Args:
            waveform: The waveform to play
        """

    @abstractmethod
    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        """Unconditionally add given measurements relative to the current position.

        Args:
            measurements: Measurements to add.
        """

    @abstractmethod
    def with_repetition(self, repetition_count: RepetitionCount,
                        measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        """Start a new repetition context with given repetition count. The caller has to iterate over the return value
        and call `:py:meth:`.ProgramBuilder.inner_scope` inside the iteration context.

        Args:
            repetition_count: Repetition count
            measurements: These measurements are added relative to the position at the start of the iteration iff the
                          iteration is not empty.

        Returns:
            An iterable of :py:class:`ProgramBuilder` instances.
        """

    @abstractmethod
    def with_sequence(self,
                      measurements: Optional[Sequence[MeasurementWindow]] = None) -> ContextManager['ProgramBuilder']:
        """Start a new sequence context. The caller has to enter the returned context manager and add the sequenced
        elements there.

        Measurements that are added in to the returned program builder are discarded if the sequence is empty on exit.

        Args:
            measurements: These measurements are added relative to the position at the start of the sequence iff the
            sequence is not empty.

        Returns:
            A context manager that returns a :py:class:`ProgramBuilder` on entering.
        """

    @abstractmethod
    def new_subprogram(self, global_transformation: 'Transformation' = None) -> ContextManager['ProgramBuilder']:
        """Create a context managed program builder whose contents are translated into a single waveform upon exit if
        it is not empty.

        Args:
            global_transformation: This transformation is applied to the waveform

        Returns:
            A context manager that returns a :py:class:`ProgramBuilder` on entering.
        """

    @abstractmethod
    def with_iteration(self, index_name: str, rng: range,
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        """Create an iterable that represent the body of the iteration. This can be an iterable with an element for each
        step in the iteration or a single object that represents the complete iteration.

        Args:
            index_name: The name of index
            rng: The range if the index
            measurements: Measurements to add iff the iteration body is not empty.
        """

    @abstractmethod
    def time_reversed(self) -> ContextManager['ProgramBuilder']:
        """This returns a new context manager that will reverse everything added to it in time upon exit.

        Returns:
            A context manager that returns a :py:class:`ProgramBuilder` on entering.
        """

    @abstractmethod
    def to_program(self) -> Optional[Program]:
        """Generate the final program. This is allowed to invalidate the program builder.

        Returns:
            A program implementation. None if nothing was added to this program builder.
        """
