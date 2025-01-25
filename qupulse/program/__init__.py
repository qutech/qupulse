# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""This package contains the means to construct a program from a pulse template.

A program is an un-parameterized multichannel time to voltage relation. They are constructed by sequencing playback
commands which typically mean that an arbitrary waveform is played.

The arbitrary waveforms are defined in the :py:mod:`.waveforms` module.

:py:mod:`.transformation` contains useful transformations for waveforms which for example allow the
construction of virtual channels, i.e. linear combinations of channels from a set of other channes.

:py:mod:`.loop` contains the legacy program representation with is an aribtrariliy nested sequence/repetition structure
of waveform playbacks.

:py:mod:`.linspace` contains a more modern program representation to efficiently execute linearly spaced voltage sweeps
even if interleaved with constant segments.

:py:mod:`.volatile` contains the machinery to declare quickly changable program parameters. This functionality is stale
and was not used by the library authors for a long term. It is very useful for dynamic nuclear polarization which is not
used/required/possible with (purified) silicon samples.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Sequence, ContextManager, Mapping, Tuple, Generic, TypeVar, Iterable, Dict
from numbers import Real

from qupulse.program.waveforms import Waveform
from qupulse.program.transformation import Transformation
from qupulse.program.volatile import VolatileRepetitionCount
from qupulse.utils.types import MeasurementWindow, TimeType
from qupulse.parameter_scope import Scope
from qupulse.expressions import sympy as sym_expr
from qupulse.utils.sympy import _lambdify_modules

from typing import Protocol, runtime_checkable


NumVal = TypeVar('NumVal', bound=Real)


@dataclass
class SimpleExpression(Generic[NumVal]):
    """This is a potential hardware evaluable expression of the form

    C + C1*R1 + C2*R2 + ...
    where R1, R2, ... are potential runtime parameters.

    The main use case is the expression of for loop dependent variables where the Rs are loop indices. There the
    expressions can be calculated via simple increments.

    This class tries to pass a number and a :py:class:`sympy.expr.Expr` on
    best effort basis.

    Attributes:
        base: The part of this expression which is not runtime parameter dependent
        offsets: Factors would have been a better name in hindsight. A mapping of inner parameter names to the factor
        with which they contribute to the final value.
    """

    base: NumVal
    offsets: Mapping[str, NumVal]

    def __post_init__(self):
        assert isinstance(self.offsets, Mapping)

    def value(self, scope: Mapping[str, NumVal]) -> NumVal:
        """Numeric value of the expression with the given scope.
        Args:
            scope: Scope in which the expression is evaluated.
        Returns:
            The numeric value.
        """
        value = self.base
        for name, factor in self.offsets:
            value += scope[name] * factor
        return value

    def __add__(self, other):
        if isinstance(other, (float, int, TimeType)):
            return SimpleExpression(self.base + other, self.offsets)

        if type(other) == type(self):
            offsets = dict(self.offsets)
            for name, value in other.offsets.items():
                offsets[name] = value + offsets.get(name, 0)
            return SimpleExpression(self.base + other.base, offsets)

        # this defers evaluation when other is still a symbolic expression
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return SimpleExpression(-self.base, {name: -value for name, value in self.offsets.items()})

    def __mul__(self, other: NumVal):
        if isinstance(other, (float, int, TimeType)):
            return SimpleExpression(self.base * other, {name: other * value for name, value in self.offsets.items()})

        # this defers evaluation when other is still a symbolic expression
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        inv = 1 / other
        return self.__mul__(inv)

    @property
    def free_symbols(self):
        """This is required for the :py:class:`sympy.expr.Expr` interface compliance. Since the keys of
        :py:attr:`.offsets` are internal parameters we do not have free symbols.

        Returns:
            An empty tuple
        """
        return ()

    def _sympy_(self):
        """This method is used by :py:`sympy.sympify`. This class tries to "just work" in the sympy evaluation pipelines.

        Returns:
            self
        """
        return self

    def replace(self, r, s):
        """We mock :class:`sympy.Expr.replace` here. This class does not support inner parameters so there is nothing
        to replace. Importantly, the keys of the offsets are no runtime variables!

        Returns:
            self
        """
        return self


# this keeps the simple expression in lambdified results
_lambdify_modules.append({'SimpleExpression': SimpleExpression})


RepetitionCount = Union[int, VolatileRepetitionCount, SimpleExpression[int]]
HardwareTime = Union[TimeType, SimpleExpression[TimeType]]
HardwareVoltage = Union[float, SimpleExpression[float]]


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


def default_program_builder() -> ProgramBuilder:
    """This function returns an instance of the default program builder class :class:`.LoopBuilder` in the default
    configuration.

    Returns:
        A program builder instance.
    """
    from qupulse.program.loop import LoopBuilder
    return LoopBuilder()


# TODO: hackedy, hackedy
sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES = sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES + (SimpleExpression,)
