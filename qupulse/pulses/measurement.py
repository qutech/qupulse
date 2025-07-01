# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import warnings
from typing import Optional, List, Tuple, Union, Dict, Set, Mapping, AbstractSet
from numbers import Real
import itertools

from qupulse.expressions import Expression, ExpressionScalar
from qupulse.utils.types import MeasurementWindow
from qupulse.parameter_scope import Scope

MeasurementDeclaration = Tuple[str, Union[Expression, str, Real], Union[Expression, str, Real]]


class MeasurementDefiner:
    def __init__(self, measurements: Optional[List[MeasurementDeclaration]]):
        if measurements is None:
            self._measurement_windows = []
        else:
            self._measurement_windows = [(name,
                                          begin if isinstance(begin, Expression) else ExpressionScalar(begin),
                                          length if isinstance(length, Expression) else ExpressionScalar(length))
                                         for name, begin, length in measurements]
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
            try:
                if begin_val < 0 or length_val < 0:
                    warnings.warn('Measurement window with negative begin or length: {}, {}'.format(begin, length))
            except TypeError:
                pass

            resulting_windows.append(
                (name,
                 begin_val,
                 length_val)
            )
        return resulting_windows

    @property
    def measurement_parameters(self) -> AbstractSet[str]:
        """Return the parameters of measurements that are directly declared on `self`.
        Does _not_ visit eventual child objects."""
        return set(var
                   for _, begin, length in self._measurement_windows
                   for var in itertools.chain(begin.variables, length.variables))

    @property
    def measurement_declarations(self) -> List[MeasurementDeclaration]:
        """Return the measurements that are directly declared on `self`. Does _not_ visit eventual child objects."""
        return [(name,
                 begin,
                 length)
                for name, begin, length in self._measurement_windows]

    @property
    def measurement_names(self) -> Set[str]:
        """Return the names of measurements that are directly declared on `self`.
        Does _not_ visit eventual child objects."""
        return {name for name, *_ in self._measurement_windows}
