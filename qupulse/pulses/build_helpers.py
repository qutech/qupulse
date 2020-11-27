from typing import List, Optional, Dict, Mapping, Union
import abc
from dataclasses import dataclass, field
from collections import defaultdict
import contextlib

import sympy

from qupulse import ChannelID
from qupulse.pulses.pulse_template import PulseTemplate, Buildable, BuildContext, BuildRequirement
from qupulse.pulses.interpolation import InterpolationStrategy
from qupulse.expressions import ExpressionLike, ExpressionScalar


class Interpolating(Buildable):
    def __init__(self,
                 interpolation_strategy: Union[InterpolationStrategy, Mapping[ChannelID, InterpolationStrategy]],
                 duration: ExpressionLike):
        self._interpolation_strategy = interpolation_strategy
        self._duration = ExpressionScalar(duration)

    def required_context(self) -> BuildRequirement:
        return BuildRequirement(
            parent=False,
            previous=True,
            next=True
        )

    def _interpolation_strategy_mapping(self) -> Mapping[ChannelID, InterpolationStrategy]:
        if isinstance(self._interpolation_strategy, Mapping):
            return self._interpolation_strategy
        else:
            return defaultdict(default_factory=lambda: self._interpolation_strategy)

    def build(self, context: BuildContext) -> PulseTemplate:
        from qupulse.pulses.function_pulse_template import FunctionPulseTemplate
        from qupulse.pulses.multi_channel_pulse_template import AtomicMultiChannelPulseTemplate
        assert context.previous is not None and context.next is not None

        initial_values = context.previous.final_values
        final_values = context.next.initial_values
        interpolation_strategies = self._interpolation_strategy_mapping()

        fpts = []
        for ch, initial in initial_values.items():
            final = final_values[ch]
            interp_expr = interpolation_strategies[ch].expression.evaluate_symbolic({
                't0': 0,
                't1': self._duration,
                'v0': initial,
                'v1': final
            })
            fpts.append(FunctionPulseTemplate(
                interp_expr,
                duration_expression=self._duration,
                channel=ch
            ))
        return AtomicMultiChannelPulseTemplate(*fpts)


class FillHelper(Buildable):
    def __init__(self, target_identifier: str, duration: ExpressionLike, values: Optional[Dict[ChannelID, ExpressionLike]]):
        self._target_identifier = target_identifier
        self._target_duration = ExpressionScalar(duration)
        self._duration_dummy = sympy.Dummy()
        self._values = None if values is None else dict(values)

    def required_context(self) -> BuildRequirement:
        return BuildRequirement(
            parent=True,
            previous=self._values is None,
            next=False
        )

    @property
    def duration(self):
        return self._duration_dummy

    def build(self, context: BuildContext) -> PulseTemplate:
        from qupulse.pulses.table_pulse_template import TablePulseTemplate

        for parent in context.parents:
            if parent.identifier == self._target_identifier:
                target = parent
                break
        else:
            raise ValueError(f'Could not find target "{self._target_identifier}" in context', context)

        current_duration = target.duration.sympified_expression
        self_duration = sympy.solve(sympy.Eq(current_duration, self._target_duration.sympified_expression),
                                    self._duration_dummy)

        if self._values is None:
            values = context.previous.final_values
        else:
            values = self._values

        return TablePulseTemplate({ch: [(0, value), (self_duration, value)] for ch, value in values.items()})
