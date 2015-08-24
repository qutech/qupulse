import numpy as np
from scipy.interpolate import interp1d


class InterpolationStrategy(metaclass = ABCMeta):
    
    @abstractmethod
    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: np.ndarray) -> np.ndarray:
        """Return a sequence of voltage values for the time slot between the previous and the current point (given as (time, value) pairs)
        according to the interpolation strategy.
        
        The resulting sequence includes the sample for the time of the current point and start at the sample just after the previous point, i.e., 
        is of the form [f(sample(previous_point_time)+1), f(sample(previous_point_time)+2), ... f(sample(current_point_time))].
        """
        pass
    
class LinearInterpolationStrategy(InterpolationStrategy):
    """Interpolates linearly."""
    
    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: np.ndarray) -> np.ndarray:
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]

        interpolator = interp1d(xs, ys, kind='linear', copy=False) # No extra error checking needed, interp1d throws errors for times out of bounds
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
