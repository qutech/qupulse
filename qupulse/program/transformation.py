# SPDX-FileCopyrightText: 2014-2024 Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Transformations to be applied to sampled waveform data (offset/scale/virtual gates)."""

from typing import Any, Mapping, Set, Tuple, Sequence, AbstractSet, Union, TYPE_CHECKING, Hashable
from abc import abstractmethod
from numbers import Real
import warnings

import numpy as np

from qupulse import ChannelID
from qupulse.utils.types import SingletonABCMeta, frozendict, DocStringABCMeta
from qupulse.expressions import ExpressionScalar
from qupulse.program.values import DynamicLinearValue


_TrafoValue = Union[Real, ExpressionScalar, DynamicLinearValue]


__all__ = ['Transformation', 'IdentityTransformation', 'LinearTransformation', 'ScalingTransformation',
           'OffsetTransformation', 'ParallelChannelTransformation', 'ChainedTransformation',
           'chain_transformations']


class Transformation(metaclass=DocStringABCMeta):
    __slots__ = ()

    _identity_singleton = None
    """Transforms numeric time-voltage values for multiple channels to other time-voltage values. The number and names
     of input and output channels might differ."""

    @abstractmethod
    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        """Apply transformation to data
        Args:
            time:
            data:

        Returns:
            transformed: A DataFrame that has been transformed with index == output_channels
        """

    @abstractmethod
    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        """Return the channel identifiers"""

    @abstractmethod
    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        """Channels that are required for getting data for the requested output channel"""

    def chain(self, next_transformation: 'Transformation') -> 'Transformation':
        if next_transformation is IdentityTransformation():
            return self
        else:
            return chain_transformations(self, next_transformation)

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return False

    def get_constant_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return frozenset()
    
    def contains_dynamic_value(self) -> bool:
        raise NotImplementedError()
    

class IdentityTransformation(Transformation, metaclass=SingletonABCMeta):
    __slots__ = ()

    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        return data

    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return input_channels

    @property
    def compare_key(self) -> type(None):
        warnings.warn("Transformation.compare_key is deprecated since 0.11 and will be removed in 0.12",
                      DeprecationWarning, stacklevel=2)
        return None

    def __hash__(self):
        return 0x1234991

    def __eq__(self, other):
        return self is other

    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return output_channels

    def chain(self, next_transformation: Transformation) -> Transformation:
        return next_transformation

    def __repr__(self):
        return 'IdentityTransformation()'

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return True

    def get_constant_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return input_channels


class ChainedTransformation(Transformation):
    __slots__ = ('_transformations',)

    def __init__(self, *transformations: Transformation):
        self._transformations = transformations

    @property
    def transformations(self) -> Tuple[Transformation, ...]:
        return self._transformations

    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        for transformation in self._transformations:
            input_channels = transformation.get_output_channels(input_channels)
        return input_channels

    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        for transformation in reversed(self._transformations):
            output_channels = transformation.get_input_channels(output_channels)
        return output_channels

    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        for transformation in self._transformations:
            data = transformation(time, data)
        return data

    @property
    def compare_key(self) -> Tuple[Transformation, ...]:
        warnings.warn("Transformation.compare_key is deprecated since 0.11 and will be removed in 0.12",
                      DeprecationWarning, stacklevel=2)
        return self._transformations

    def __hash__(self):
        return hash(self._transformations)

    def __eq__(self, other):
        if isinstance(other, ChainedTransformation):
            return self._transformations == other._transformations
        return NotImplemented

    def chain(self, next_transformation) -> Transformation:
        return chain_transformations(*self.transformations, next_transformation)

    def __repr__(self):
        return f'{type(self).__name__}{self._transformations!r}'

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return all(trafo.is_constant_invariant() for trafo in self._transformations)

    def get_constant_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        for trafo in self._transformations:
            input_channels = trafo.get_constant_output_channels(input_channels)
        return input_channels


class LinearTransformation(Transformation):
    def __init__(self,
                 transformation_matrix: np.ndarray,
                 input_channels: Sequence[ChannelID],
                 output_channels: Sequence[ChannelID]):
        """

        Args:
            transformation_matrix: Matrix describing the transformation with shape (output_channels, input_channels)
            input_channels: Channel ids of the columns
            output_channels: Channel ids of the rows
        """
        transformation_matrix = np.asarray(transformation_matrix)

        if transformation_matrix.shape != (len(output_channels), len(input_channels)):
            raise ValueError('Shape of transformation matrix does not match to the given channels')

        output_sorter = np.argsort(output_channels)
        transformation_matrix = transformation_matrix[output_sorter, :]

        input_sorter = np.argsort(input_channels)
        transformation_matrix = transformation_matrix[:, input_sorter]

        self._matrix = transformation_matrix
        self._input_channels = tuple(sorted(input_channels))
        self._output_channels = tuple(sorted(output_channels))
        self._input_channels_set = frozenset(self._input_channels)
        self._output_channels_set = frozenset(self._output_channels)

    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        data_out = {forwarded_channel: data[forwarded_channel]
                    for forwarded_channel in set(data.keys()).difference(self._input_channels)}

        if len(data_out) == len(data):
            # only forwarded data
            return data_out

        try:
            data_in = np.stack([data[in_channel] for in_channel in self._input_channels])
        except KeyError as error:
            raise KeyError('Invalid input channels', set(data.keys()), set(self._input_channels)) from error

        transformed_data = self._matrix @ data_in

        assert transformed_data.shape[0] == len(self._output_channels)
        for out_channel, transformed_channel_data in zip(self._output_channels, transformed_data):
            data_out[out_channel] = transformed_channel_data

        return data_out

    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        if not input_channels >= self._input_channels_set:
            # input_channels is not a superset of the required input channels
            raise KeyError('Invalid input channels', input_channels, self._input_channels_set)

        return (input_channels - self._input_channels_set) | self._output_channels_set

    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        forwarded = output_channels - self._output_channels_set
        if not forwarded.isdisjoint(self._input_channels):
            raise KeyError('Is input channel', forwarded & self._input_channels_set)
        elif output_channels.isdisjoint(self._output_channels):
            return output_channels
        else:
            return forwarded | self._input_channels_set

    def __hash__(self):
        return hash((self._input_channels, self._output_channels, self._matrix.tobytes()))

    def __eq__(self, other):
        if isinstance(other, LinearTransformation):
            return (self._input_channels == other._input_channels and
                    self._output_channels == other._output_channels and
                    np.array_equal(self._matrix, other._matrix))
        return NotImplemented

    @property
    def compare_key(self) -> Tuple[Tuple[ChannelID], Tuple[ChannelID], bytes]:
        warnings.warn("Transformation.compare_key is deprecated since 0.11 and will be removed in 0.12",
                      DeprecationWarning, stacklevel=2)
        return self._input_channels, self._output_channels, self._matrix.tobytes()

    def __repr__(self):
        return ('LinearTransformation('
                'transformation_matrix={transformation_matrix},'
                'input_channels={input_channels},'
                'output_channels={output_channels})').format(transformation_matrix=self._matrix.tolist(),
                                                             input_channels=self._input_channels,
                                                             output_channels=self._output_channels)

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return True

    def get_constant_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return input_channels


class OffsetTransformation(Transformation):
    __slots__ = ('_offsets',)

    def __init__(self, offsets: Mapping[ChannelID, _TrafoValue]):
        """Adds an offset to each channel specified in offsets.

        Channels not in offsets are forewarded

        Args:
            offsets: Channel -> offset mapping
        """
        self._offsets = frozendict(offsets)
        assert _are_valid_transformation_expressions(self._offsets), f"Not valid transformation expressions: {self._offsets}"

    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        offsets = _instantiate_expression_dict(time, self._offsets, default_sweepval=0.)
        return {channel: channel_values + offsets[channel] if channel in offsets else channel_values
                for channel, channel_values in data.items()}

    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return output_channels

    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return input_channels

    def __eq__(self, other):
        if isinstance(other, OffsetTransformation):
            return self._offsets == other._offsets
        return NotImplemented

    def __hash__(self):
        return hash(self._offsets)

    @property
    def compare_key(self) -> Hashable:
        warnings.warn("Transformation.compare_key is deprecated since 0.11 and will be removed in 0.12",
                      DeprecationWarning, stacklevel=2)
        return self._offsets

    def __repr__(self):
        return f'{type(self).__name__}({dict(self._offsets)!r})'

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return not _has_time_dependent_values(self._offsets)

    def get_constant_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return _get_constant_output_channels(self._offsets, input_channels)
    
    def contains_dynamic_value(self) -> bool:
        return any(isinstance(o,DynamicLinearValue) for o in self._offsets.values())
    

class ScalingTransformation(Transformation):
    __slots__ = ('_factors',)

    def __init__(self, factors: Mapping[ChannelID, _TrafoValue]):
        self._factors = frozendict(factors)
        assert _are_valid_transformation_expressions(self._factors), f"Not valid transformation expressions: {self._factors}"

    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        factors = _instantiate_expression_dict(time, self._factors, default_sweepval=1.)
        return {channel: channel_values * factors[channel] if channel in factors else channel_values
                for channel, channel_values in data.items()}

    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return output_channels

    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return input_channels

    def __eq__(self, other):
        if isinstance(other, ScalingTransformation):
            return self._factors == other._factors
        return NotImplemented

    def __hash__(self):
        return hash(self._factors)

    @property
    def compare_key(self) -> Hashable:
        warnings.warn("Transformation.compare_key is deprecated since 0.11 and will be removed in 0.12",
                      DeprecationWarning, stacklevel=2)
        return self._factors

    def __repr__(self):
        return f'{type(self).__name__}({dict(self._factors)!r})'

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return not _has_time_dependent_values(self._factors)

    def get_constant_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return _get_constant_output_channels(self._factors, input_channels)
    
    def contains_dynamic_value(self) -> bool:
        return any(isinstance(o,DynamicLinearValue) for o in self._factors.values())
    

try:
    if TYPE_CHECKING:
        import pandas
        PandasDataFrameType = pandas.DataFrame
    else:
        PandasDataFrameType = Any

    def linear_transformation_from_pandas(transformation: PandasDataFrameType) -> LinearTransformation:
        """ Creates a LinearTransformation object out of a pandas data frame.

        Args:
            transformation (pandas.DataFrame): The pandas.DataFrame object out of which a LinearTransformation will be formed.

        Returns:
            the created LinearTransformation instance
        """
        return LinearTransformation(transformation.values, transformation.columns, transformation.index)

    LinearTransformation.from_pandas = linear_transformation_from_pandas
except ImportError:
    pass


class ParallelChannelTransformation(Transformation):
    __slots__ = ('_channels', )

    def __init__(self, channels: Mapping[ChannelID, _TrafoValue]):
        """Set channel values to given values regardless their former existence. The values can be time dependent
        expressions.

        Args:
            channels: Channels present in this map are set to the given value.
        """
        self._channels: Mapping[ChannelID, _TrafoValue] = frozendict(channels.items())
        assert _are_valid_transformation_expressions(self._channels), f"Not valid transformation expressions: {self._channels}"

    def __call__(self, time: Union[np.ndarray, float],
                 data: Mapping[ChannelID, Union[np.ndarray, float]]) -> Mapping[ChannelID, Union[np.ndarray, float]]:
        overwritten = self._instantiated_values(time)
        return {**data, **overwritten}

    def _instantiated_values(self, time):
        scope = {'t': time}
        array_or_float = lambda x: np.full_like(time, fill_value=x, dtype=float) if hasattr(time, '__len__') else x
        return {channel: value.evaluate_in_scope(scope) if hasattr(value, 'evaluate_in_scope') else array_or_float(value)
                for channel, value in self._channels.items()}

    def __hash__(self):
        return hash(self._channels)

    def __eq__(self, other):
        if isinstance(other, ParallelChannelTransformation):
            return self._channels == other._channels
        return NotImplemented

    @property
    def compare_key(self) -> Hashable:
        warnings.warn("Transformation.compare_key is deprecated since 0.11 and will be removed in 0.12",
                      DeprecationWarning, stacklevel=2)
        return self._channels

    def get_input_channels(self, output_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return output_channels - self._channels.keys()

    def get_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        return input_channels | self._channels.keys()

    def __repr__(self):
        return f'{type(self).__name__}({dict(self._channels)!r})'

    def is_constant_invariant(self):
        """Signals if the transformation always maps constants to constants."""
        return not _has_time_dependent_values(self._channels)

    def get_constant_output_channels(self, input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
        output_channels = set(input_channels)
        for ch, value in self._channels.items():
            if hasattr(value, 'variables'):
                output_channels.discard(ch)
            else:
                output_channels.add(ch)

        return output_channels
    
    def contains_dynamic_value(self) -> bool:
        return any(isinstance(o,DynamicLinearValue) for o in self._channels.values())
    

def chain_transformations(*transformations: Transformation) -> Transformation:
    parsed_transformations = []
    for transformation in transformations:
        if transformation is IdentityTransformation() or transformation is None:
            pass
        elif isinstance(transformation, ChainedTransformation):
            parsed_transformations.extend(transformation.transformations)
        else:
            parsed_transformations.append(transformation)
    if len(parsed_transformations) == 0:
        return IdentityTransformation()
    elif len(parsed_transformations) == 1:
        return parsed_transformations[0]
    else:
        return ChainedTransformation(*parsed_transformations)


def _instantiate_expression_dict(time,
                                 expressions: Mapping[str, _TrafoValue],
                                 default_sweepval: Real,
                                 ) -> Mapping[str, Union[Real, np.ndarray]]:
    scope = {'t': time}
    modified_expressions = {}
    for name, value in expressions.items():
        if hasattr(value, 'evaluate_in_scope'):
            modified_expressions[name] = value.evaluate_in_scope(scope)
        if isinstance(value, DynamicLinearValue):
            # it is assumed that swept parameters will be handled by the ProgramBuilder accordingly
            # such that here only an "identity" trafo is to be applied and the
            # trafos are set in the program internally.
            modified_expressions[name] = default_sweepval
    if modified_expressions:
        return {**expressions, **modified_expressions}
    else:
        return expressions


def _has_time_dependent_values(expressions: Mapping[ChannelID, _TrafoValue]) -> bool:
    return any(hasattr(value, 'variables')
               for value in expressions.values())


def _get_constant_output_channels(expressions: Mapping[ChannelID, _TrafoValue],
                                  constant_input_channels: AbstractSet[ChannelID]) -> AbstractSet[ChannelID]:
    return {ch
            for ch in constant_input_channels
            if not hasattr(expressions.get(ch, None), 'variables')}


def _are_valid_transformation_expressions(expressions: Mapping[ChannelID, _TrafoValue]) -> bool:
    return all(expr.variables == ('t',)
               for expr in expressions.values()
               if hasattr(expr, 'variables'))
