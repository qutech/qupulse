"""Runtime variable value implementations."""

from dataclasses import dataclass, field
from numbers import Real
from typing import TypeVar, Generic, Mapping, Union, List, Tuple, Optional
from types import NotImplementedType, MappingProxyType
import operator

import numpy as np

from qupulse.program.volatile import VolatileRepetitionCount
from qupulse.utils.types import TimeType
from qupulse.expressions import sympy as sym_expr
from qupulse.utils.sympy import _lambdify_modules


NumVal = TypeVar('NumVal', bound=Real)


@dataclass(frozen=True)
class DynamicLinearValue(Generic[NumVal]):
    """This is a potential runtime-evaluable expression of the form

    C + C1*R1 + C2*R2 + ...
    where R1, R2, ... are potential runtime parameters.

    The main use case is the expression of for loop-dependent variables where the Rs are loop indices. There the
    expressions can be calculated via simple increments.

    This class tries to pass a number and a :py:class:`sympy.expr.Expr` on best effort basis.
    """

    #: The part of this expression which is not runtime parameter-dependent
    base: NumVal

    #: A mapping of inner parameter names to the factor with which they contribute to the final value.
    factors: Mapping[str, NumVal]

    def __post_init__(self):
        assert isinstance(self.factors, Mapping)
        immutable = MappingProxyType(dict(self.factors))
        object.__setattr__(self, 'factors', immutable)
        
    def value(self, scope: Mapping[str, NumVal]) -> NumVal:
        """Numeric value of the expression with the given scope.
        Args:
            scope: Scope in which the expression is evaluated.
        Returns:
            The numeric value.
        """
        value = self.base
        for name, factor in self.factors.items():
            value += scope[name] * factor
        return value
    
    def __abs__(self):
        # The deifnition of an absolute value is ambiguous, but there is a case
        # to define it as sum_i abs(f_i) + abs(base) for certain conveniences.
        # return abs(self.base)+sum([abs(o) for o in self.factors.values()])
        raise NotImplementedError(f'abs({self.__class__.__name__}) is ambiguous')
    
    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.base == other.base and self.factors == other.factors

        if (base_eq := self.base.__eq__(other)) is NotImplemented:
            return NotImplemented

        return base_eq and not self.factors
    
    def __add__(self, other):
        if isinstance(other, (float, int, TimeType)):
            return DynamicLinearValue(self.base + other, self.factors)

        if type(other) == type(self):
            factors = dict(self.factors)
            for name, value in other.factors.items():
                factors[name] = value + factors.get(name, 0)
            return DynamicLinearValue(self.base + other.base, factors)

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
    
    def __hash__(self):
        return hash((self.base,frozenset(sorted(self.factors.items()))))
    
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


# is there any way to cast the numpy cumprod to int?
int_type = Union[np.int64,np.int32,int]

@dataclass(frozen=True)
class ResolutionDependentValue(Generic[NumVal]):
    """This is a potential runtime-evaluable expression of the form
    
    o + sum_i  b_i*m_i
    
    with (potential) float o, b_i and integers m_i. o and b_i are rounded(gridded)
    to a resolution given upon __call__.
    
    The main use case is the correct rounding of increments used in command-based
    voltage scans on some hardware devices, where an imprecise numeric value is
    looped over m_i times and could, if not rounded, accumulate a proportional
    error leading to unintended drift in output voltages when jump-back commands
    afterwards do not account for the deviations.
    Rounding the value preemptively and supplying corrected values to jump-back
    commands prevents this.
    """
    
    bases: Tuple[NumVal, ...]
    multiplicities: Tuple[int, ...]
    offset: NumVal
    __is_time_or_int: bool = field(init=False, repr=False)
    
    def __post_init__(self):

        flag = all(isinstance(b,(TimeType,int_type)) for b in self.bases)\
            and isinstance(self.offset,(TimeType,int_type))
        object.__setattr__(self, '_ResolutionDependentValue__is_time_or_int', flag)

    #this is not to circumvent float errors in python, but rounding errors from awg-increment commands.
    #python float are thereby accurate enough
    def __call__(self, resolution: Optional[float]) -> Union[NumVal,TimeType]:
        #with resolution = None handle TimeType/int case?
        if resolution is None:
            assert self.__is_time_or_int
            return sum(b*m for b,m in zip(self.bases,self.multiplicities)) + self.offset
        #resolution as float value of granularity of base val.
        #to avoid conflicts between positive and negative vals from casting
        #half to even, use abs val
        return sum(np.sign(b) * round(abs(b) / resolution) * m * resolution for b,m in zip(self.bases,self.multiplicities))\
             + np.sign(self.offset) * round(abs(self.offset) / resolution) * resolution

    def __bool__(self):
        #if any value is not zero - this helps for some checks
        return any(bool(b) for b in self.bases) or bool(self.offset)

    def __add__(self, other):
        # this should happen in the context of an offset being added to it, not the bases being modified.
        if isinstance(other, (float, int, TimeType)):
            return ResolutionDependentValue(self.bases,self.multiplicities,self.offset+other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        # this should happen when the amplitude is being scaled
        # multiplicities are not affected
        if isinstance(other, (float, int, TimeType)):
            return ResolutionDependentValue(tuple(b*other for b in self.bases),self.multiplicities,self.offset*other)
        return NotImplemented

    def __rmul__(self,other):
        return self.__mul__(other)

    def __truediv__(self,other):
        return self.__mul__(1/other)

    def __float__(self):
        return float(self(resolution=None))

    def __str__(self):
        return f"RDP of {sum(b*m for b,m in zip(self.bases,self.multiplicities)) + self.offset}"

    def __repr__(self):
        return "RDP("+",".join([f"{k}="+v.__str__() for k,v in vars(self).items()])+")"

    def __eq__(self,o):
        if not isinstance(o,ResolutionDependentValue):
            return False
        return self.__dict__ == o.__dict__

    def __hash__(self):
        return hash((self.bases,self.offset,self.multiplicities,self.__is_time_or_int))


#This is a simple dervide class to allow better isinstance checks in the HDAWG driver
@dataclass(frozen=True)
class DynamicLinearValueStepped(DynamicLinearValue):
    step_nesting_level: int
    rng: range
    reverse: int|bool


# TODO: hackedy, hackedy
sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES = sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES + (DynamicLinearValue,)

# this keeps the simple expression in lambdified results
_lambdify_modules.append({'DynamicLinearValue': DynamicLinearValue,
                          'DynamicLinearValueStepped': DynamicLinearValueStepped})

RepetitionCount = Union[int, VolatileRepetitionCount, DynamicLinearValue[int]]
HardwareTime = Union[TimeType, DynamicLinearValue[TimeType]]
HardwareVoltage = Union[float, DynamicLinearValue[float]]
