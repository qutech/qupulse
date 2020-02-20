from typing import Optional, List, Tuple, Union, Dict, Set, Mapping
from numbers import Real
import itertools

from qupulse.expressions import Expression
from qupulse.utils.types import MeasurementWindow
from qupulse.parameter_scope import Scope

MeasurementDeclaration = Tuple[str, Union[Expression, str, Real], Union[Expression, str, Real]]


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
                                parameters: Union[Mapping[str, Real], Scope],
                                measurement_mapping: Dict[str, Optional[str]]) -> List[MeasurementWindow]:
        """Calculate measurement windows with the given parameter set and rename them woth the measurement mapping"""
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
