import numpy as np
from numbers import Real, Number
from typing import Optional, Union, Sequence, ContextManager, Mapping, Tuple, Generic, TypeVar, Iterable, Dict, List
from dataclasses import dataclass

from functools import total_ordering
from qupulse.utils.sympy import _lambdify_modules
from qupulse.expressions import sympy as sym_expr, Expression
from qupulse.utils.types import MeasurementWindow, TimeType, FrozenMapping


NumVal = TypeVar('NumVal', bound=Real)


@total_ordering
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
        for name, factor in self.offsets.items():
            value += scope[name] * factor
        return value
    
    def __abs__(self):
        return abs(self.base)+sum([abs(o) for o in self.offsets.values()])
    
    def __eq__(self, other):
        #there is no good way to compare it without having a value,
        #but cannot require more parameters in magic method?
        #so have this weird full equality for now which doesn logically make sense
        #in most cases to catch unintended consequences
        
        if isinstance(other, (float, int, TimeType)):
            return self.base==other and all([o==other for o in self.offsets])

        if type(other) == type(self):
            if len(self.offsets)!=len(other.offsets): return False
            return self.base==other.base and all([o1==o2 for o1,o2 in zip(self.offsets,other.offsets)])

        return NotImplemented
    
    def __gt__(self, other):
        return all([b for b in self._return_greater_comparison_bools(other)])
    
    def __lt__(self, other):
        return all([not b for b in self._return_greater_comparison_bools(other)])
    
    def _return_greater_comparison_bools(self, other) -> List[bool]:
        #there is no good way to compare it without having a value,
        #but cannot require more parameters in magic method?
        #so have this weird full equality for now which doesn logically make sense
        #in most cases to catch unintended consequences
        if isinstance(other, (float, int, TimeType)):
            return [self.base>other] + [o>other for o in self.offsets.values()]

        if type(other) == type(self):
            if len(self.offsets)!=len(other.offsets): return [False]
            return [self.base>other.base] + [o1>o2 for o1,o2 in zip(self.offsets.values(),other.offsets.values())]

        return NotImplemented
    
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
        (-self).__add__(other)

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
