from typing import Optional, List, Tuple, Union, Dict, Set, Mapping
from numbers import Real
import itertools

from qupulse.expressions import ExpressionScalar, ExpressionLike
from qupulse.utils.types import MeasurementWindow
from qupulse.parameter_scope import Scope
from qupulse.pulses.range import ParametrizedRange, RangeLike, RangeScope

MeasurementRangeDeclaration = Tuple[str, ExpressionLike, ExpressionLike, Tuple[str, RangeLike]]

MeasurementDeclaration = Union[
    Tuple[str, ExpressionLike, ExpressionLike],
    MeasurementRangeDeclaration
]


class MeasurementDefiner:
    def __init__(self, measurements: Optional[List[MeasurementDeclaration]]):
        self._measurement_windows: Tuple[Tuple[str, ExpressionScalar, ExpressionScalar], ...] = ()
        self._measurement_ranges: Tuple[Tuple[str, ExpressionScalar, ExpressionScalar, Tuple[str, RangeLike]], ...] = ()

        if measurements is not None:
            measurement_windows = []
            measurement_ranges = []

            for declaration in measurements:
                if len(declaration) == 3:
                    name, begin, length = declaration
                    measurement_windows.append((name, ExpressionScalar(begin), ExpressionScalar(length)))
                else:
                    name, begin, length, (idx_name, param_range) = declaration
                    measurement_ranges.append((name, ExpressionScalar(begin), ExpressionScalar(length),
                                               (idx_name, ParametrizedRange.from_range_like(param_range))))

            self._measurement_windows = tuple(measurement_windows)
            self._measurement_ranges = tuple(measurement_ranges)
        for _, _, length in self._measurement_windows:
            if (length < 0) is True:
                raise ValueError('Measurement window length may not be negative')

    def get_measurement_windows(self,
                                parameters: Union[Mapping[str, Real], Scope],
                                measurement_mapping: Dict[str, Optional[str]]) -> List[MeasurementWindow]:
        """Calculate measurement windows with the given parameter set and rename them with the measurement mapping. This
        method only returns the measurement windows that are defined on `self`. It does _not_ collect the measurement
        windows defined on eventual child objects that `self` has/is composed of.

        Args:
            parameters: Used to calculate the numeric begins and lengths of symbolically defined measurement windows.
            measurement_mapping: Used to rename/drop measurement windows. Windows mapped to None are dropped.

        Returns:
            List of measurement windows directly defined on self
        """
        try:
            volatile = parameters.get_volatile_parameters().keys()
        except AttributeError:
            volatile = frozenset()

        resulting_windows = []
        for name, begin, length in self._measurement_windows:
            name = measurement_mapping[name]
            if name is None:
                continue

            assert volatile.isdisjoint(begin.variables) and volatile.isdisjoint(length.variables), "volatile measurement parameters are not supported"

            begin_val = begin.evaluate_in_scope(parameters)
            length_val = length.evaluate_in_scope(parameters)
            if begin_val < 0 or length_val < 0:
                raise ValueError('Measurement window with negative begin or length: {}, {}'.format(begin, length))

            resulting_windows.append(
                (name,
                 begin_val,
                 length_val)
            )

        for name, begin, length, (idx_name, idx_range) in self._measurement_ranges:
            name = measurement_mapping[name]
            if name is None:
                continue

            for idx_val in idx_range.to_range(parameters):
                scope = RangeScope(parameters, idx_name, idx_val)

                begin_val = begin.evaluate_in_scope(scope)
                length_val = length.evaluate_in_scope(scope)

                resulting_windows.append(
                    (name,
                     begin_val,
                     length_val)
                )

        return resulting_windows

    @property
    def measurement_parameters(self) -> Set[str]:
        """Return the parameters of measurements that are directly declared on `self`.
        Does _not_ visit eventual child objects."""
        return set(var
                   for _, begin, length in self._measurement_windows
                   for var in itertools.chain(begin.variables, length.variables))

    @property
    def measurement_declarations(self) -> List[MeasurementDeclaration]:
        """Return the measurements that are directly declared on `self`. Does _not_ visit eventual child objects."""
        measurements = []
        measurements.extend((name,
                             begin.original_expression,
                             length.original_expression)
                            for name, begin, length in self._measurement_windows)
        measurements.extend((name, begin.original_expression, length.original_expression,
                             (idx_name, idx_range.to_tuple()))
                            for name, begin, length, (idx_name, idx_range) in self._measurement_ranges)
        return measurements

    @property
    def measurement_names(self) -> Set[str]:
        """Return the names of measurements that are directly declared on `self`.
        Does _not_ visit eventual child objects."""
        return {name for name, *_ in itertools.chain(self._measurement_windows, self._measurement_ranges)}

    def __hash__(self):
        return hash((self._measurement_windows, self._measurement_ranges))

    def __eq__(self, other):
        return (self._measurement_windows == getattr(other, '_measurement_windows', None) and
                self._measurement_ranges == getattr(other, '_measurement_ranges', None))
