from dataclasses import dataclass
from numbers import Real
from typing import TypeVar, Generic, Mapping, Union

from qupulse.program.volatile import VolatileRepetitionCount
from qupulse.utils.types import TimeType

from qupulse.expressions import sympy as sym_expr
from qupulse.utils.sympy import _lambdify_modules


NumVal = TypeVar('NumVal', bound=Real)


@dataclass
class DynamicLinearValue(Generic[NumVal]):
    """This is a potential runtime-evaluable expression of the form

    C + C1*R1 + C2*R2 + ...
    where R1, R2, ... are potential runtime parameters.

    The main use case is the expression of for loop-dependent variables where the Rs are loop indices. There the
    expressions can be calculated via simple increments.

    This class tries to pass a number and a :py:class:`sympy.expr.Expr` on best effort basis.

    Attributes:
        base: The part of this expression which is not runtime parameter-dependent
        factors: A mapping of inner parameter names to the factor with which they contribute to the final value.
    """

    base: NumVal
    factors: Mapping[str, NumVal]

    def __post_init__(self):
        assert isinstance(self.factors, Mapping)

    def value(self, scope: Mapping[str, NumVal]) -> NumVal:
        """Numeric value of the expression with the given scope.
        Args:
            scope: Scope in which the expression is evaluated.
        Returns:
            The numeric value.
        """
        value = self.base
        for name, factor in self.factors:
            value += scope[name] * factor
        return value

    def __add__(self, other):
        if isinstance(other, (float, int, TimeType)):
            return DynamicLinearValue(self.base + other, self.factors)

        if type(other) == type(self):
            offsets = dict(self.factors)
            for name, value in other.factors.items():
                offsets[name] = value + offsets.get(name, 0)
            return DynamicLinearValue(self.base + other.base, offsets)

        # this defers evaluation when other is still a symbolic expression
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return DynamicLinearValue(-self.base, {name: -value for name, value in self.factors.items()})

    def __mul__(self, other: NumVal):
        if isinstance(other, (float, int, TimeType)):
            return DynamicLinearValue(self.base * other, {name: other * value for name, value in self.factors.items()})

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


# TODO: hackedy, hackedy
sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES = sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES + (DynamicLinearValue,)

# this keeps the simple expression in lambdified results
_lambdify_modules.append({'DynamicLinearValue': DynamicLinearValue})

RepetitionCount = Union[int, VolatileRepetitionCount, DynamicLinearValue[int]]
HardwareTime = Union[TimeType, DynamicLinearValue[TimeType]]
HardwareVoltage = Union[float, DynamicLinearValue[float]]
