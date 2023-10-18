import contextlib
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, ContextManager, Iterable, Tuple, Union

from qupulse import ChannelID, MeasurementWindow
from qupulse.parameter_scope import Scope, MappedScope, FrozenDict
from qupulse.program import (ProgramBuilder, HardwareTime, HardwareVoltage, Waveform, RepetitionCount, TimeType,
                             SimpleExpression)
from qupulse.expressions import sympy as sym_expr


@dataclass
class LinSpaceNode:
    """AST node for a program that supports linear spacing of set points as well as nested sequencing and repetitions"""


@dataclass
class LinSpaceSet(LinSpaceNode):
    bases: Tuple[float, ...]
    factors: Tuple[Optional[Tuple[float, ...]], ...]

    duration_base: TimeType
    duration_factors: Optional[Tuple[TimeType, ...]]


@dataclass
class LinSpaceRepeat(LinSpaceNode):
    body: Tuple[LinSpaceNode, ...]
    count: int


@dataclass
class LinSpaceIter(LinSpaceNode):
    """Iteration in linear space are restricted to range 0 to length. Offsets and spacing are stored in the set node."""
    body: Tuple[LinSpaceNode, ...]
    length: int


class LinSpaceBuilder(ProgramBuilder):
    def __init__(self, channels: Tuple[Optional[ChannelID], ...]):
        super().__init__()
        self._name_to_idx = {name: idx for idx, name in enumerate(channels) if name is not None}
        self._idx_to_name = channels

        self._stack = [[]]
        self._ranges = []

    @classmethod
    def from_channel_dict(cls, channels: Mapping[ChannelID, int]):
        assert len(set(channels.values())) == len(channels), "no duplicate target channels"
        channel_list = [None] * 20
        for ch_name, ch_idx in channels.items():
            channel_list[ch_idx] = ch_name
        return cls(tuple(channel_list))

    def _root(self):
        return self._stack[0]

    def _get_rng(self, idx_name: str) -> range:
        return self._get_ranges()[idx_name]

    def inner_scope(self, scope: Scope) -> Scope:
        """This function is necessary to inject program builder specific parameter implementations into the build
        process."""
        if self._ranges:
            name, _ = self._ranges[-1]
            return MappedScope(scope, FrozenDict({name: SimpleExpression(base=0, offsets=[(name, 1)])}))
        else:
            return scope

    def _get_ranges(self):
        return dict(self._ranges)

    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[ChannelID, HardwareVoltage]):
        voltages = sorted((self._name_to_idx[ch_name], value) for ch_name, value in voltages.items())
        voltages = [value for _, value in voltages]

        ranges = self._get_ranges()
        factors = []
        bases = []
        for value in voltages:
            if isinstance(value, float):
                bases.append(value)
                factors.append(None)
                continue
            offsets = value.offsets
            base = value.base
            incs = []
            for rng_name, rng in ranges.items():
                start = 0.
                step = 0.
                for off_name, offset in offsets:
                    if off_name == rng_name:
                        start += rng.start * offset
                        step += rng.step * offset
                base += start
                incs.append(step)
            factors.append(tuple(incs))
            bases.append(base)

        if isinstance(duration, SimpleExpression):
            duration_factors = duration.offsets
            duration_base = duration.base
        else:
            duration_base = duration
            duration_factors = None

        set_cmd = LinSpaceSet(bases=tuple(bases),
                              factors=tuple(factors),
                              duration_base=duration_base,
                              duration_factors=duration_factors)

        self._stack[-1].append(set_cmd)

    def play_arbitrary_waveform(self, waveform: Waveform):
        raise NotImplementedError('Not implemented yet (postponed)')

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        """Ignores measurements"""
        pass

    def with_repetition(self, repetition_count: RepetitionCount,
                        measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        if repetition_count == 0:
            return
        self._stack.append([])
        yield self
        blocks = self._stack.pop()
        if blocks:
            self._stack[-1].append(LinSpaceRepeat(body=tuple(blocks), count=repetition_count))

    @contextlib.contextmanager
    def with_sequence(self,
                      measurements: Optional[Sequence[MeasurementWindow]] = None) -> ContextManager['ProgramBuilder']:
        yield self

    def new_subprogram(self, global_transformation: 'Transformation' = None) -> ContextManager['ProgramBuilder']:
        raise NotImplementedError('Not implemented yet (postponed)')

    def with_iteration(self, index_name: str, rng: range,
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        if len(rng) == 0:
            return
        self._stack.append([])
        self._ranges.append((index_name, rng))
        yield self
        cmds = self._stack.pop()
        self._ranges.pop()
        if cmds:
            self._stack[-1].append(LinSpaceIter(body=tuple(cmds), length=len(rng)))

    def to_program(self) -> Optional[Sequence[LinSpaceNode]]:
        if self._root():
            return self._root()
