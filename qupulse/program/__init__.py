# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: GPL-3.0-or-later

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Sequence, ContextManager, Mapping, Tuple, Generic, TypeVar, Iterable, Dict
from numbers import Real, Number

import numpy as np

from qupulse._program.waveforms import Waveform
from qupulse.utils.types import MeasurementWindow, TimeType, FrozenMapping
from qupulse._program.volatile import VolatileRepetitionCount
from qupulse.parameter_scope import Scope
from qupulse.expressions import sympy as sym_expr, Expression
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
    """

    base: NumVal
    offsets: Mapping[str, NumVal]

    def __post_init__(self):
        assert isinstance(self.offsets, Mapping)

    def value(self, scope: Mapping[str, NumVal]) -> NumVal:
        value = self.base
        for name, factor in self.offsets:
            value += scope[name] * factor
        return value

    def __add__(self, other):
        if isinstance(other, (float, int, TimeType)):
            return SimpleExpression(self.base + other, self.offsets)

        if type(other) == type(self):
            offsets = self.offsets.copy()
            for name, value in other.offsets.items():
                offsets[name] = value + offsets.get(name, 0)
            return SimpleExpression(self.base + other.base, offsets)

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

        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        inv = 1 / other
        return self.__mul__(inv)

    @property
    def free_symbols(self):
        return ()

    def _sympy_(self):
        return self

    def replace(self, r, s):
        return self

    def evaluate_in_scope_(self, *args, **kwargs):
        # TODO: remove. It is currently required to avoid nesting this class in an expression for the MappedScope
        # We can maybe replace is with a HardwareScope or something along those lines
        return self


_lambdify_modules.append({'SimpleExpression': SimpleExpression})


RepetitionCount = Union[int, VolatileRepetitionCount, SimpleExpression[int]]
HardwareTime = Union[TimeType, SimpleExpression[TimeType]]
HardwareVoltage = Union[float, SimpleExpression[float]]


@runtime_checkable
class Program(Protocol):
    """This protocol is used to inspect and or manipulate programs. As you can see the functionality is very limited
    because most of a program class' capability are specific to the implementation."""

    @property
    def duration(self) -> TimeType:
        raise NotImplementedError()


class ProgramBuilder(Protocol):
    """This protocol is used by :py:meth:`.PulseTemplate.create_program` to build a program via the visitor pattern.

    There is a default implementation which is the loop class.

    Other hardware backends can use this protocol to implement easy translation of pulse templates into a hardware
    compatible format."""

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

    def time_reversed(self) -> ContextManager['ProgramBuilder']:
        pass

    def to_program(self) -> Optional[Program]:
        """Further addition of new elements might fail after finalizing the program."""


def default_program_builder() -> ProgramBuilder:
    """This function returns an instance of the default program builder class `LoopBuilder`"""
    from qupulse.program.loop import LoopBuilder
    return LoopBuilder()


# TODO: hackedy, hackedy
sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES = sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES + (SimpleExpression,)
