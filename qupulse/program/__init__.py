import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Sequence, ContextManager, Mapping, Tuple, Generic, TypeVar, Iterable
from numbers import Real

import numpy as np
import sympy
from frozendict import frozendict

from qupulse._program.waveforms import Waveform
from qupulse.utils.types import MeasurementWindow, TimeType
from qupulse._program.volatile import VolatileRepetitionCount
from qupulse.parameter_scope import Scope
from qupulse.expressions import sympy as sym_expr
from qupulse.utils import sympy as sym_util

from typing import Protocol, runtime_checkable


NumVal = TypeVar('NumVal', bound=Real)


def _to_numeric_scalar(val) -> Union[int, float, TimeType, type(None)]:
    if isinstance(val, (float, int, TimeType)):
        return val

    elif isinstance(val, sympy.Integer):
        return int(val)

    elif isinstance(val, sympy.Rational):
        return TimeType(val)

    elif isinstance(val, sympy.Float):
        return float(val)


@dataclass(frozen=True, init=False)
class SimpleExpression(Generic[NumVal]):
    """This is a potential hardware evaluable expression of the form

    C + C1*R1 + C2*R2 + ...
    where R1, R2, ... are potential runtime parameters.

    The main use case is the expression of for loop dependent variables where the Rs are loop indices. There the
    expressions can be calculated via simple increments.
    """

    base: NumVal
    offsets: Mapping[str, NumVal]

    def __init__(self, base: NumVal = 0, offsets: Union[Mapping[str, NumVal], Sequence[Tuple[str, NumVal]]] = ()):
        object.__setattr__(self, 'base', base)
        if not isinstance(offsets, Mapping):
            sequence = offsets
            offsets = {}
            for name, factor in sequence:
                offsets[name] = factor + offsets.get(name, 0)
        object.__setattr__(self, 'offsets', frozendict(offsets))

    def value(self, scope: Mapping[str, NumVal]) -> NumVal:
        value = self.base
        for name, factor in self.offsets.items():
            value += scope[name] * factor
        return value

    def __add__(self, other):
        if isinstance(other, SimpleExpression):
            offsets = dict(self.offsets)
            for name, factor in other.offsets.items():
                offsets[name] = offsets.get(name, 0) + factor
            return SimpleExpression(self.base + other.base, frozendict(offsets))

        base = self.base.__add__(other)
        if base is NotImplemented:
            base = other.__add__(base)
        if base is NotImplemented:
            return NotImplemented
        return SimpleExpression(base, self.offsets)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        (-self).__add__(other)

    def __neg__(self):
        return SimpleExpression(-self.base, frozendict((name, -value) for name, value in self.offsets.items()))

    def __mul__(self, other: NumVal):
        if isinstance(other, SimpleExpression):
            return NotImplemented
        return SimpleExpression(self.base * other, frozendict((name, value * other)
                                                              for name, value in self.offsets.items()))

    def __rmul__(self, other):
        return self.__mul__(other)

    def evaluate_in_scope(self, *args, **kwargs):
        # TODO: remove. It is currently required to avoid nesting this class in an expression for the MappedScope
        # We can maybe replace is with a HardwareScope or something along those lines
        return self

    def _sympy_(self):
        return self

    @property
    def free_symbols(self):
        return tuple(self.offsets.keys())


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
    """This protocol is used by PulseTemplate to build the program.

    There is a default implementation which is the loop class.

    Other hardware backends can use this protocol to implement easy translation of pulse templates.

    """

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
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        pass

    def to_program(self) -> Optional[Program]:
        """Further addition of new elements might fail after finalizing the program."""


def default_program_builder() -> ProgramBuilder:
    from qupulse.program.loop import LoopBuilder
    return LoopBuilder()


# TODO: hackedy, hackedy
sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES = sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES + (SimpleExpression,)
sym_util.MANUAL_ARITHMETICS = sym_util.MANUAL_ARITHMETICS + (SimpleExpression,)
