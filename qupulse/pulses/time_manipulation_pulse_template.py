import numbers
from typing import Dict, Optional, Set, Union

from qupulse import ChannelID
from qupulse._program._loop import Loop
from qupulse._program.transformation import Transformation
from qupulse.parameter_scope import Scope
from qupulse.pulses.pulse_template import PulseTemplate
from qupulse.expressions import ExpressionLike, ExpressionScalar
from qupulse._program.waveforms import ConstantWaveform


def _evaluate_expression_dict(expression_dict: Dict[str, ExpressionScalar], scope: Scope) -> Dict[str, float]:
    return {ch: value.evaluate_in_scope(scope)
            for ch, value in expression_dict.items()}


class TimeExtensionPulseTemplate(PulseTemplate):
    """Extend the given pulse template with a constant(?) prefix and/or suffix"""
    
    @property
    def parameter_names(self) -> Set[str]:
        return self._inner.parameter_names | set(self._stop.variables) | set(self._start.variables)

    @property
    def measurement_names(self) -> Set[str]:
        return set()

    @property
    def duration(self) -> ExpressionScalar:
        return self._stop - self._start

    @property
    def defined_channels(self) -> Set['ChannelID']:
        return self._inner.defined_channels

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        integral = self._inner.integral
        initial_values = self._inner.initial_values
        final_values = self._inner.final_values
        for ch, inner_value in integral.items():
            integral[ch] = inner_value + initial_values[ch] * self._start + final_values[ch] * self._stop
        return integral

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._inner.initial_values

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        return self._inner.final_values

    def _start_waveform(self, scope) -> Optional[ConstantWaveform]:
        start = self._start.evaluate_in_scope(scope)
        if start > 0:
            return ConstantWaveform(start, )

    def _internal_create_program(self, *, scope: Scope, measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional[Transformation],
                                 to_single_waveform: Set[Union[str, 'PulseTemplate']], parent_loop: Loop) -> None:
        start = self._start.evaluate_in_scope(scope)
        stop = self._stop.evaluate_in_scope(scope)

        if start > 0:
            start_wf = ConstantWaveform.from_mapping(start,
                                                     _evaluate_expression_dict(self._inner.initial_values, scope))
            parent_loop.append_child(Loop(waveform=start_wf, repetition_count=1))

        self._inner._create_program(
            scope=scope,
            measurement_mapping=measurement_mapping,
            channel_mapping=channel_mapping,
            global_transformation=global_transformation,
            to_single_waveform=to_single_waveform,
            parent_loop=parent_loop)

        if stop > 0:
            stop_wf = ConstantWaveform.from_mapping(start,
                                                    _evaluate_expression_dict(self._inner.final_values, scope))
            parent_loop.append_child(Loop(waveform=stop_wf, repetition_count=1))

    def __init__(self, inner: PulseTemplate, start: ExpressionLike, stop: ExpressionLike,
                 *,
                 identifier=None):
        PulseTemplate.__init__(self, identifier=identifier)

        self._inner = inner
        self._start = ExpressionScalar(start)
        self._stop = ExpressionScalar(stop)
