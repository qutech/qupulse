"""This module defines strategies for interpolation between points in a pulse table or similar.

Classes:
    - InterpolationStrategy: Interface for interpolation strategies.
    - LinearInterpolationStrategy: Interpolates linearly between two points.
    - HoldInterpolationStrategy: Interpolates by holding the first point's value.
    - JumpInterpolationStrategy: Interpolates by holding the second point's value.
"""


from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Optional
import numpy as np
from sympy.utilities import lambdify

from qupulse.expressions import ExpressionScalar

__all__ = ["InterpolationStrategy", "HoldInterpolationStrategy",
           "JumpInterpolationStrategy", "LinearInterpolationStrategy"]


class InterpolationStrategy(metaclass=ABCMeta):
    """Defines a strategy to interpolate values between two points."""

    @abstractmethod
    def __call__(self,
                 start: Tuple[float, float],
                 end: Tuple[float, float],
                 times: np.ndarray) -> np.ndarray:
        """Return a sequence of voltage values for the time slot between the start and the
        end point (given as (time, value) pairs) according to the interpolation strategy.

        Args:
            start ((float, float)): The start point of the interpolation as (time, value) pair.
            end ((float, float)): The end point of the interpolation as (time, value) pair.
            times (numpy.ndarray): An array of sample times for which values will be computed. All
                values in this array must lie within the boundaries defined by start and end.
        Returns:
            A numpy.ndarray containing the interpolated values.
        """

    @property
    @abstractmethod
    def integral(self) -> ExpressionScalar:
        """Returns the symbolic integral of this interpolation strategy using (v0,t0) and (v1,t1)
        to represent start and end point."""

    @property
    @abstractmethod
    def expression(self) -> ExpressionScalar:
        """Returns a symbolic expression of the interpolation strategy using (v0,t0) and (v1, t1)
        to represent start and end point and t as free variable. Note that the expression is only valid for values of t
        between t0 and t1."""

    @abstractmethod
    def evaluate_integral(self, t0, v0, t1, v1):
        """ Evaluate integral using arguments v0, t0, v1, t1 """

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the Interpolation Strategy Class"""

    def __eq__(self, other: Any) -> bool:
        # Interpolations are the same, if their type is the same
        return type(self) == type(other)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def constant_value(self, start: Tuple[float, float], end: Tuple[float, float]) -> Optional[float]:
        """The value of the interpolation if it is constant.

        Args:
            start: The start point of the interpolation as (time, value) pair.
            end: The end point of the interpolation as (time, value) pair.

        Returns:
            The value of the interpolation if it is constant
        """
        return None


class LinearInterpolationStrategy(InterpolationStrategy):
    """An InterpolationStrategy that interpolates linearly between two points."""

    _integral_expression = ExpressionScalar('(t1-t0) * (v0 + v1) / 2')
    _expression = ExpressionScalar('v0 + (v1-v0) * (t-t0)/(t1-t0)')
    
    def __call__(self,
                 start: Tuple[float, float],
                 end: Tuple[float, float],
                 times: np.ndarray) -> np.ndarray:
        m = (end[1] - start[1])/(end[0] - start[0])
        return m * (times - start[0]) + start[1]

    @property
    def integral(self) -> ExpressionScalar:
        return self._integral_expression

    def evaluate_integral(self, t0, v0, t1, v1):
        return (t1-t0) * (v0 + v1) / 2

    @property
    def expression(self) -> ExpressionScalar:
        return self._expression

    def __str__(self) -> str:
        return 'linear'

    def __repr__(self) -> str:
        return "<Linear Interpolation>"

    def constant_value(self, start: Tuple[float, float], end: Tuple[float, float]) -> Optional[float]:
        return start[1] if start[1] == end[1] else None

    
class HoldInterpolationStrategy(InterpolationStrategy):
    """An InterpolationStrategy that interpolates by holding the value of the start point for the
    entire intermediate space."""

    _integral_expression = ExpressionScalar('v0*(t1-t0)')
    _expression = ExpressionScalar('v0')
    
    def __call__(self,
                 start: Tuple[float, float],
                 end: Tuple[float, float],
                 times: np.ndarray) -> np.ndarray:
        if np.any(times < start[0]) or np.any(times > end[0]):
            raise ValueError(
                "Time Value for interpolation out of bounds. Must be between {0} and {1}.".format(
                    start[0], end[0]
                )
            )
        return np.full_like(times, fill_value=start[1], dtype=float)

    @property
    def integral(self) -> ExpressionScalar:
        return self._integral_expression

    def evaluate_integral(self, t0, v0, t1, v1):
        return v0*(t1-t0)

    @property
    def expression(self) -> ExpressionScalar:
        return self._expression

    def __str__(self) -> str:
        return 'hold'

    def __repr__(self) -> str:
        return "<Hold Interpolation>"

    def constant_value(self, start: Tuple[float, float], end: Tuple[float, float]) -> Optional[float]:
        return start[1]


class JumpInterpolationStrategy(InterpolationStrategy):
    """An InterpolationStrategy that interpolates by holding the value of the end point for the
    entire intermediate space."""

    _integral_expression = ExpressionScalar('v1*(t1-t0)')
    _expression = ExpressionScalar('v1')

    def __call__(self,
                 start: Tuple[float, float],
                 end: Tuple[float, float],
                 times: np.ndarray) -> np.ndarray:
        if np.any(times < start[0]) or np.any(times > end[0]):
            raise ValueError(
                "Time Value for interpolation out of bounds. Must be between {0} and {1}.".format(
                    start[0], end[0]
                )
            )
        return np.full_like(times, fill_value=end[1], dtype=float)

    @property
    def integral(self) -> ExpressionScalar:
        return self._integral_expression

    def evaluate_integral(self, t0, v0, t1, v1):
        return v1*(t1-t0)

    @property
    def expression(self) -> ExpressionScalar:
        return self._expression

    def __str__(self) -> str:
        return 'jump'

    def __repr__(self) -> str:
        return "<Jump Interpolation>"

    def constant_value(self, start: Tuple[float, float], end: Tuple[float, float]) -> Optional[float]:
        return end[1]
