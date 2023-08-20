from typing import *
from collections import defaultdict
from contextlib import contextmanager

from ..utils.types import MeasurementWindow
from . import ProgramBuilder, RepetitionCount, HardwareTime, HardwareVoltage, iterate_context_managers
from .._program._loop import Loop, to_waveform
from .._program.waveforms import ConstantWaveform, Waveform
from ..pulses.range import RangeScope
from ..parameter_scope import Scope


class LoopBuilder(ProgramBuilder):
    def __init__(self):
        self._root = Loop()
        self._top = self._root

        self._duration = 0
        self._stack: List[Tuple[Loop, Optional[Tuple[str, int]]]] = [(self._top, None)]
        self._measurements = {}

    def inner_scope(self, scope: Scope) -> Scope:
        local_vars = self._stack[-1][1]
        if local_vars is None:
            return scope
        else:
            return RangeScope(scope, *local_vars)

    def _push(self, loop, index):
        self._top = loop
        self._stack.append((loop, index))

    def _pop(self):
        stack = self._stack
        loop, _ = stack.pop()
        parent = self._top = self._stack[-1][0]
        if loop.parent is None and (len(loop.children) != 0 or loop.waveform is not None):
            parent.append_child(loop=loop)

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        self._top.add_measurements(measurements)

    def with_repetition(self, repetition_count: RepetitionCount) -> Iterable['ProgramBuilder']:

        new_top = Loop(repetition_count=repetition_count)
        self._push(new_top, None)

        yield self

        self._pop()

    def with_iteration(self, index_name: str, rng: range) -> Iterable['ProgramBuilder']:
        for value in rng:
            loop = Loop()
            self._push(loop, (index_name, value))
            yield self
            self._pop()

    @contextmanager
    def with_sequence(self) -> ContextManager['ProgramBuilder']:
        loop = Loop()
        self._push(loop, None)
        yield self
        self._pop()

    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[str, HardwareVoltage]):
        self.play_arbitrary_waveform(ConstantWaveform.from_mapping(duration=duration, amplitude=voltages))

    def play_arbitrary_waveform(self, waveform: Waveform):
        self._top.append_child(waveform=waveform)

    @contextmanager
    def new_subprogram(self) -> ContextManager['ProgramBuilder']:
        loop = Loop()
        self._push(loop, None)
        yield self
        if loop.children or (loop.waveform and loop.repetition_count != 1):
            loop._waveform = to_waveform(loop)
            loop._repetition_definition = 1
        self._pop()
