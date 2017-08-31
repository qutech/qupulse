from typing import Optional, List, Tuple, Union, Dict, Set
from numbers import Real
import itertools

from qctoolkit.expressions import Expression


MeasurementDeclaration = Tuple[str, Union[Expression, str, Real], Union[Expression, str, Real]]
MeasurementWindow = Tuple[str, Real, Real]


class MeasurementDefiner:
    def __init__(self, measurements: Optional[List[MeasurementDeclaration]]):
        if measurements is None:
            self._measurement_windows = []
        else:
            self._measurement_windows = [(name,
                                          begin if isinstance(begin, Expression) else Expression(begin),
                                          length if isinstance(length, Expression) else Expression(length))
                                         for name, begin, length in measurements]
        for _, _, length in self._measurement_windows:
            if (length < 0) is True:
                raise ValueError('Measurement window length may not be negative')

    def get_measurement_windows(self,
                                parameters: Dict[str, Real],
                                measurement_mapping: Dict[str, str]) -> List[MeasurementWindow]:
        """Calculate measurement windows with the given parameter set and rename them woth the measurement mapping"""
        def get_val(v):
            return v.evaluate_numeric(**parameters)

        resulting_windows = [(measurement_mapping[name], get_val(begin), get_val(length))
                             for name, begin, length in self._measurement_windows]

        for _, begin, length in resulting_windows:
            if begin < 0 or length < 0:
                raise ValueError('Measurement window with negative begin or length: {}, {}'.format(begin, length))
        return resulting_windows

    @property
    def measurement_parameters(self) -> Set[str]:
        return set(var
                   for _, begin, length in self._measurement_windows
                   for var in itertools.chain(begin.variables, length.variables))

    @property
    def measurement_declarations(self) -> List[MeasurementDeclaration]:
        return [(name,
                 begin.original_expression,
                 length.original_expression)
                for name, begin, length in self._measurement_windows]

    @property
    def measurement_names(self) -> Set[str]:
        return {name for name, *_ in self._measurement_windows}
