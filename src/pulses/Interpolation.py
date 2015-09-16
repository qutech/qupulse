from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Tuple

from .Serializer import Serializable

# We could think about using an Enum from the enum package (standard library) instead and separate code and data
class InterpolationStrategy(Serializable, metaclass = ABCMeta):
    
    @abstractmethod
    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: np.ndarray) -> np.ndarray:
        """Return a sequence of voltage values for the time slot between the previous and the current point (given as (time, value) pairs)
        according to the interpolation strategy.
        
        The resulting sequence includes the sample for the time of the current point and start at the sample just after the previous point, i.e., 
        is of the form [f(sample(previous_point_time)+1), f(sample(previous_point_time)+2), ... f(sample(current_point_time))].
        """
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def __eq__(self, other):
        # Interpolations are the same, if their type is the same
        return type(self) == type(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__repr__())
    
    @property
    def identifier(self):
        return self.__repr__()

    def get_serialization_data(self):
        return dict(type='Interpolation', interpolation=self.__repr__())
    
class LinearInterpolationStrategy(InterpolationStrategy):
    """Interpolates linearly."""
    
    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: np.ndarray) -> np.ndarray:
        m = (end[1] - start[1])/(end[0] - start[0])
        interpolator = lambda t: m * (t - start[0]) + start[1]
        return interpolator(times)

    def to_json(self):
        return 'linear'

    def __repr__(self):
        return "<Linear Interpolation>"

    
class HoldInterpolationStrategy(InterpolationStrategy):
    """Holds previous value and jumps to the current value at the last sample."""

    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: np.ndarray) -> np.ndarray:
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]

        if np.any(times < start[0]) or np.any(times > end[0]):
            raise ValueError("Time Value for interpolation out of bounds. Must be between {0} and {1}.".format(start[0], end[0]))

        voltages = np.ones_like(times) * start[1]
        return voltages

    def to_json(self):
        return 'hold'

    def __repr__(self):
        return "<Hold Interpolation>"

class JumpInterpolationStrategy(InterpolationStrategy):
    """Jumps to the current value at the first sample and holds."""
    # TODO: better name than jump

    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: np.ndarray) -> np.ndarray:
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]

        if np.any(times < start[0]) or np.any(times > end[0]):
           raise ValueError("Time Value for interpolation out of bounds. Must be between {0} and {1}.".format(start[0], end[0]))

        voltages = np.ones_like(times) * end[1]
        return voltages

    def to_json(self):
        return 'jump'

    def __repr__(self):
        return "<Jump Interpolation>"
