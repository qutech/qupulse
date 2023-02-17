"""This module contains the interface / protocol descriptions."""

try:
    from typing import Protocol
except ImportError:
    # python version < 3.8
    from typing_extensions import Protocol

from typing import Mapping, Union, Sequence, Hashable, Any

from numbers import Real

import numpy as np


class Ordered(Protocol):
    def __lt__(self, other):
        pass

    def __le__(self, other):
        pass

    def __gt__(self, other):
        pass

    def __ge__(self, other):
        pass


class Scalar(Protocol):
    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __floordiv__(self, other):
        pass

    def __ceil__(self):
        pass

    def __floor__(self):
        pass

    def __float__(self):
        pass

    def __int__(self):
        pass

    def __abs__(self):
        pass




class Expression(Hashable, Protocol):
    def evaluate_in_scope(self, scope: Mapping) -> Union[Real, np.ndarray]:
        """Evaluate the expression by taking the variables from the given scope (typically of type Scope but it can be
        any mapping.)
        Args:
            scope:

        Returns:

        """

    def evaluate_symbolic(self, substitutions: Mapping[str, Any]) -> 'Expression':
        """Substitute a part of the expression for another"""

    def evaluate_time_dependent(self, scope: Mapping) -> Union['Expression', Real, np.ndarray]:
        """Evaluate to a time dependent expression or a constant."""
    @property
    def variables(self) -> Sequence[str]:
        """ Get all free variables in the expression.

        Returns:
            A collection of all free variables occurring in the expression.
        """
        raise NotImplementedError()

    @classmethod
    def make(cls,
             expression_or_dict,
             numpy_evaluation=None) -> 'Expression':
        """Backward compatible expression generation to allow creation from dict."""
        raise NotImplementedError()

    @property
    def underlying_expression(self) -> Any:
        """Return some internal unspecified representation"""
        raise NotImplementedError()

    def get_serialization_data(self):
        pass


class ExpressionScalar(Expression, Scalar, Ordered, Protocol):
    pass


class ExpressionVector(Expression, Protocol):
    pass
