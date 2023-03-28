import functools
import inspect
import math
from typing import Sequence, Any, Mapping, Union
from numbers import Real

import numpy as np

from qupulse.expressions import protocol, sympy


def make_wrappers(expr, expr_scalar, expr_vector):
    class ExpressionWrapper(protocol.Expression):
        def __init__(self, x):
            self._wrapped: protocol.Expression = expr(x)

        @classmethod
        def make(cls, expression_or_dict, numpy_evaluation=None) -> 'Expression':
            return cls(expression_or_dict)

        @property
        def underlying_expression(self) -> Any:
            return self._wrapped.underlying_expression

        def __hash__(self) -> int:
            return hash(self._wrapped)

        def __eq__(self, other):
            return self._wrapped == getattr(other, '_wrapped', other)

        @property
        def variables(self) -> Sequence[str]:
            return self._wrapped.variables

        def evaluate_in_scope(self, scope: Mapping) -> Union[Real, np.ndarray]:
            return self._wrapped.evaluate_in_scope(scope)

        def evaluate_symbolic(self, substitutions: Mapping[str, Any]) -> 'ExpressionWrapper':
            """Substitute a part of the expression for another"""
            return ExpressionWrapper(self._wrapped.evaluate_symbolic(substitutions))

        def evaluate_time_dependent(self, scope: Mapping) -> Union['Expression', Real, np.ndarray]:
            """Evaluate to a time dependent expression or a constant."""
            return self._wrapped.evaluate_time_dependent(scope)

        def get_serialization_data(self):
            return self._wrapped.get_serialization_data()

    class ExpressionScalarWrapper(ExpressionWrapper, protocol.ExpressionScalar):
        def __init__(self, x):
            ExpressionWrapper.__init__(self, 0)
            self._wrapped: protocol.ExpressionScalar = expr_scalar(x)

        # Scalar
        def __add__(self, other):
            return ExpressionScalarWrapper(self._wrapped + getattr(other, '_wrapped', other))

        def __sub__(self, other):
            return ExpressionScalarWrapper(self._wrapped - getattr(other, '_wrapped', other))

        def __mul__(self, other):
            return ExpressionScalarWrapper(self._wrapped * getattr(other, '_wrapped', other))

        def __truediv__(self, other):
            return ExpressionScalarWrapper(self._wrapped / getattr(other, '_wrapped', other))

        def __floordiv__(self, other):
            return ExpressionScalarWrapper(self._wrapped // getattr(other, '_wrapped', other))

        def __ceil__(self):
            return ExpressionScalarWrapper(math.ceil(self._wrapped))

        def __floor__(self):
            return ExpressionScalarWrapper(math.floor(self._wrapped))

        def __float__(self):
            return float(self._wrapped)

        def __int__(self):
            return int(self._wrapped)

        def __abs__(self):
            return ExpressionScalarWrapper(abs(self._wrapped))

        # Ordered
        def __lt__(self, other):
            return self._wrapped < getattr(other, '_wrapped', other)

        def __le__(self, other):
            return self._wrapped <= getattr(other, '_wrapped', other)

        def __gt__(self, other):
            return self._wrapped > getattr(other, '_wrapped', other)

        def __ge__(self, other):
            return self._wrapped >= getattr(other, '_wrapped', other)

    class ExpressionVectorWrapper(ExpressionWrapper):
        pass

    return ExpressionWrapper, ExpressionScalarWrapper, ExpressionVectorWrapper
