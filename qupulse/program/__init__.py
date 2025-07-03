# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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

from dataclasses import dataclass
from typing import Union, Mapping, Generic, TypeVar
from numbers import Real

from qupulse.program.protocol import Program, ProgramBuilder
from qupulse.program.waveforms import Waveform
from qupulse.program.transformation import Transformation
from qupulse.program.volatile import VolatileRepetitionCount
from qupulse.utils.types import TimeType
from qupulse.expressions import sympy as sym_expr
from qupulse.utils.sympy import _lambdify_modules

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
