# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""This module contains the function :py:``make_wrappers`` to define wrapper classes for expression protocol implementations
which only implements methods of the protocol.
It is used for finding code that relies on expression implementation details."""

import math
from typing import Sequence, Any, Mapping, Union, Tuple
from numbers import Real

import numpy as np

from qupulse.expressions import protocol, sympy


def make_wrappers(expr: type, expr_scalar: type, expr_vector: type) -> Tuple[type, type, type]:
    """Create wrappers for expression base, scalar and vector types that only expose the methods defined in the
    corresponding expression protocol classes.

    The vector is currently not implemented.

    Args:
        expr: Expression base type of the implementation
        expr_scalar: Expression scalar type of the implementation
        expr_vector: Expression vector type of the implementation

    Returns:
        A tuple of (base, scalar, vector) types that wrap the given types.
    """

    class ExpressionWrapper(protocol.Expression):
        def __init__(self, x):
            self._wrapped: protocol.Expression = expr(x)

        @classmethod
        def make(cls, expression_or_dict, numpy_evaluation=None) -> 'ExpressionWrapper':
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
