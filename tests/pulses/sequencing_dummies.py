"""STANDARD LIBRARY IMPORTS"""
import numbers
from typing import Tuple, List, Dict, Optional, Set, Any, Union, Mapping
import copy

import numpy
import unittest

"""LOCAL IMPORTS"""
from qupulse.parameter_scope import Scope
from qupulse._program._loop import Loop
from qupulse.utils.types import MeasurementWindow, ChannelID, TimeType, time_from_float
from qupulse.serialization import Serializer
from qupulse._program.waveforms import Waveform
from qupulse.pulses.pulse_template import AtomicPulseTemplate
from qupulse.pulses.interpolation import InterpolationStrategy
from qupulse.expressions import Expression, ExpressionScalar


class MeasurementWindowTestCase(unittest.TestCase):

    def assert_measurement_windows_equal(self, expected, actual) -> bool:
        self.assertEqual(expected.keys(), actual.keys())
        for k in expected:
            self.assertEqual(list(expected[k][0]), list(actual[k][0]))
            self.assertEqual(list(expected[k][1]), list(actual[k][1]))


class DummyWaveform(Waveform):

    def __init__(self, duration: Union[float, TimeType]=0, sample_output: Union[numpy.ndarray, dict]=None, defined_channels=None) -> None:
        super().__init__(duration=duration if isinstance(duration, TimeType) else TimeType.from_float(duration))
        self.sample_output = sample_output
        if defined_channels is None:
            if isinstance(sample_output, dict):
                defined_channels = set(sample_output.keys())
            else:
                defined_channels = {'A'}
        self.defined_channels_ = defined_channels
        self.sample_calls = []

    @property
    def compare_key(self) -> Any:
        if self.sample_output is not None:
            try:
                return hash(self.sample_output.tobytes())
            except AttributeError:
                pass
            return hash(
                tuple(sorted((channel, output.tobytes()) for channel, output in self.sample_output.items()))
            )
        else:
            return id(self)

    @property
    def measurement_windows(self):
        return []

    def unsafe_sample(self,
                      channel: ChannelID,
                      sample_times: numpy.ndarray,
                      output_array: numpy.ndarray = None) -> numpy.ndarray:
        self.sample_calls.append((channel, list(sample_times), output_array))
        if output_array is None:
            output_array = numpy.empty_like(sample_times)
        if self.sample_output is not None:
            if isinstance(self.sample_output, dict):
                output_array[:] = self.sample_output[channel]
            elif callable(self.sample_output):
                output_array[:] = self.sample_output(sample_times)
            else:
                output_array[:] = self.sample_output
        else:
            output_array[:] = sample_times
        return output_array

    def unsafe_get_subset_for_channels(self, channels: Set[ChannelID]) -> 'Waveform':
        if not channels <= self.defined_channels_:
            raise KeyError('channels not in defined_channels')

        if isinstance(self.sample_output, dict):
            sample_output = {ch: self.sample_output[ch] for ch in channels}
        else:
            sample_output = copy.copy(self.sample_output)
        duration = self.duration
        defined_channels = channels
        return DummyWaveform(sample_output=sample_output,
                             duration=duration,
                             defined_channels=defined_channels)

    @property
    def defined_channels(self):
        return self.defined_channels_

    def last_value(self, channel) -> float:
        if self.sample_output is None:
            return 0.
        elif isinstance(self.sample_output, dict):
            sample_output = self.sample_output[channel]
        else:
            sample_output = self.sample_output
        return sample_output[-1]


class DummyInterpolationStrategy(InterpolationStrategy):

    def __init__(self, id_ = None) -> None:
        self.call_arguments = []
        self._id = id(self) if id_ is None else id_

    def __call__(self, start: Tuple[float, float], end: Tuple[float, float], times: numpy.ndarray) -> numpy.ndarray:
        self.call_arguments.append((start, end, list(times)))
        return times

    def __repr__(self) -> str:
        return f"DummyInterpolationStrategy({id(self)})"

    @property
    def integral(self) -> ExpressionScalar:
        raise NotImplementedError()

    @property
    def expression(self) -> ExpressionScalar:
        raise NotImplementedError()

    def evaluate_integral(self, t0, v0, t1, v1):
        """ Evaluate integral using arguments v0, t0, v1, t1 """
        raise

class DummyPulseTemplate(AtomicPulseTemplate):

    def __init__(self,
                 requires_stop: bool=False,
                 parameter_names: Set[str]=set(),
                 defined_channels: Set[ChannelID]=None,
                 duration: Any=0,
                 waveform: Waveform=tuple(),
                 measurement_names: Set[str] = set(),
                 measurements: list=list(),
                 integrals: Dict[ChannelID, ExpressionScalar]=None,
                 initial_values: Dict[ChannelID, Any]=None,
                 final_values: Dict[ChannelID, Any]=None,
                 program: Optional[Loop]=None,
                 identifier=None,
                 registry=None) -> None:
        super().__init__(identifier=identifier, measurements=measurements)
        self.requires_stop_ = requires_stop
        self.requires_stop_arguments = []

        if defined_channels is None:
            defined_channels = {'default'}
        if integrals is None:
            integrals = {ch: ExpressionScalar(0) for ch in defined_channels}

        self.parameter_names_ = parameter_names
        self.defined_channels_ = defined_channels
        self._duration = Expression(duration)
        self.waveform = waveform
        self.build_waveform_calls = []
        self.measurement_names_ = set(measurement_names)
        self._integrals = integrals
        self.create_program_calls = []
        self._program = program
        self._register(registry=registry)

        if initial_values is None:
            self._initial_values = {ch: ExpressionScalar(0) for ch in self.defined_channels}
        else:
            self._initial_values = {ch: ExpressionScalar(val) for ch, val in initial_values.items()}

        if final_values is None:
            self._final_values = {ch: ExpressionScalar(0) for ch in self.defined_channels}
        else:
            self._final_values = {ch: ExpressionScalar(val) for ch, val in final_values.items()}

        if integrals is not None:
            assert isinstance(integrals, Mapping)

    @property
    def duration(self):
        return self._duration

    @property
    def parameter_names(self) -> Set[str]:
        return set(self.parameter_names_)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        return set(self.defined_channels_)

    @property
    def measurement_names(self) -> Set[str]:
        return self.measurement_names_

    def _internal_create_program(self, *,
                                 scope: Scope,
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional['Transformation'],
                                 to_single_waveform: Set[Union[str, 'PulseTemplate']],
                                 parent_loop: Loop) -> None:
        measurements = self.get_measurement_windows(scope, measurement_mapping)
        self.create_program_calls.append((scope, measurement_mapping, channel_mapping, parent_loop))
        if self._program:
            parent_loop.add_measurements(measurements)
            parent_loop.append_child(waveform=self._program.waveform, children=self._program.children)
        elif self.waveform:
            parent_loop.add_measurements(measurements)
            parent_loop.append_child(waveform=self.waveform)

    def build_waveform(self,
                       parameters: Dict[str, numbers.Real],
                       channel_mapping: Dict[ChannelID, ChannelID]):
        self.build_waveform_calls.append((parameters, channel_mapping))
        if self.waveform or self.waveform is None:
            return self.waveform
        return DummyWaveform(duration=self.duration.evaluate_numeric(**parameters), defined_channels=self.defined_channels)

    def get_serialization_data(self, serializer: Optional['Serializer']=None) -> Dict[str, Any]:
        data = super().get_serialization_data(serializer=serializer)
        if serializer: # compatibility with old serialization routines
            data = dict()
        data['parameter_names'] = self.parameter_names
        data['defined_channels'] = self.defined_channels
        data['duration'] = self.duration
        data['measurement_names'] = self.measurement_names
        data['integrals'] = self.integral
        return data

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._integrals

    @property
    def compare_key(self) -> Tuple[Any, ...]:
        return (self.requires_stop_, self.parameter_names,
                self.defined_channels, self.duration, self.waveform, self.measurement_names, self.integral)

    def _as_expression(self) -> Dict[ChannelID, ExpressionScalar]:
        assert self.duration != 0
        t = self._AS_EXPRESSION_TIME
        duration = self.duration.underlying_expression
        return {ch: ExpressionScalar(integral.underlying_expression*t/duration**2 * 2)
                for ch, integral in self.integral.items()}

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._initial_values

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._final_values
