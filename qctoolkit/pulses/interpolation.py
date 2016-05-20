"""This module defines strategies for interpolation between points in a pulse table or similar.

Classes:
    InterpolationStrategy: Interface for interpolation strategies.
    LinearInterpolationStrategy: Interpolates linearly between two points.
    HoldInterpolationStrategy: Interpolates by holding the first point's value.
    JumpInterpolationStrategy: Interpolates by holding the second point's value.
"""


from abc import ABCMeta, abstractmethod
from typing import Any, Tuple
import numpy as np


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
            start ((float, float): The start point of the interpolation as (time, value) pair.
            end ((float, float): The end point of the interpolation as (time, value) pair.
            times (numpy.ndarray): An array of sample times for which values will be computed. All
                values in this array must lie within the boundaries defined by start and end.
        Returns:
            A numpy.ndarray containing the interpolated values.
        """

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

    
class LinearInterpolationStrategy(InterpolationStrategy):
    """An InterpolationStrategy that interpolates linearly between two points."""
    
    def __call__(self,
                 start: Tuple[float, float],
                 end: Tuple[float, float],
                 times: np.ndarray) -> np.ndarray:
        m = (end[1] - start[1])/(end[0] - start[0])
        interpolator = lambda t: m * (t - start[0]) + start[1]
        return interpolator(times)

    def __str__(self) -> str:
        return 'linear'

    def __repr__(self) -> str:
        return "<Linear Interpolation>"

    
class HoldInterpolationStrategy(InterpolationStrategy):
    """An InterpolationStrategy that interpolates by holding the value of the start point for the
    entire intermediate space."""

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

        voltages = np.ones_like(times) * start[1]
        return voltages

    def __str__(self) -> str:
        return 'hold'

    def __repr__(self) -> str:
        return "<Hold Interpolation>"


class JumpInterpolationStrategy(InterpolationStrategy):
    """An InterpolationStrategy that interpolates by holding the value of the end point for the
    entire intermediate space."""

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

        voltages = np.ones_like(times) * end[1]
        return voltages

    def __str__(self) -> str:
        return 'jump'

    def __repr__(self) -> str:
        return "<Jump Interpolation>"
