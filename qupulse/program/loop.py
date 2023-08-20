import warnings
from typing import *
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager

import dataclasses

from qupulse.utils.types import MeasurementWindow
from qupulse.program import ProgramBuilder, RepetitionCount, HardwareTime, HardwareVoltage, TimeType
from qupulse._program._loop import Loop, to_waveform
from qupulse._program.waveforms import ConstantWaveform, Waveform
from qupulse.pulses.range import RangeScope
from qupulse.parameter_scope import Scope


@dataclass
class LoopGuard:
    loop: Loop
    measurements: Optional[List[MeasurementWindow]]

    def append_child(self, **kwargs):
        if self.measurements:
            self.loop.add_measurements(self.measurements)
            self.measurements = None
        self.loop.append_child(**kwargs)

    def add_measurements(self, measurements: List[MeasurementWindow]):
        if self.measurements is None:
            self.measurements = measurements
        else:
            self.measurements.extend(measurements)


@dataclass
class StackFrame:
    loop: Union[Loop, LoopGuard]

    iterating: Optional[Tuple[str, int]]


class LoopBuilder(ProgramBuilder):
    """

    Notes fduring implementation:
     - This program builder does not use the Loop class to generate the measurements

    """

    def __init__(self):
        self._root: Loop = Loop()
        self._top: Union[Loop, LoopGuard] = self._root

        self._stack: List[StackFrame] = [StackFrame(self._root, None)]

    def inner_scope(self, scope: Scope) -> Scope:
        local_vars = self._stack[-1].iterating
        if local_vars is None:
            return scope
        else:
            return RangeScope(scope, *local_vars)

    def _push(self, stack_entry: StackFrame):
        self._top = stack_entry.loop
        self._stack.append(stack_entry)

    def _pop(self):
        stack = self._stack
        stack.pop()
        self._top = stack[-1].loop

    def _try_append(self, loop, measurements):
        if loop.waveform or len(loop) != 0:
            if measurements is not None:
                self._top.add_measurements(measurements)
            self._top.append_child(loop=loop)

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        if measurements:
            self._top.add_measurements(measurements)

    def with_repetition(self, repetition_count: RepetitionCount,
                        measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        repetition_loop = Loop(repetition_count=repetition_count)
        self._push(StackFrame(repetition_loop, None))
        yield self
        self._pop()
        self._try_append(repetition_loop, measurements)

    def with_iteration(self, index_name: str, rng: range,
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        with self.with_sequence():
            top_frame = self._stack[-1]
            for value in rng:
                top_frame.iterating = (index_name, value)
                yield self

    @contextmanager
    def with_sequence(self, measurements: Optional[Sequence[MeasurementWindow]] = None) -> ContextManager['ProgramBuilder']:
        top_frame = StackFrame(LoopGuard(self._top, measurements), None)
        self._push(top_frame)
        yield self
        self._pop()

    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[str, HardwareVoltage]):
        self.play_arbitrary_waveform(ConstantWaveform.from_mapping(duration, voltages))

    def play_arbitrary_waveform(self, waveform: Waveform):
        self._top.append_child(waveform=waveform)

    @contextmanager
    def new_subprogram(self) -> ContextManager['ProgramBuilder']:
        inner_builder = LoopBuilder()
        yield inner_builder
        inner_program = inner_builder.to_program()

        if inner_program is not None:
            for name, (begins, lengths) in inner_program.get_measurement_windows().items():
                for begin, length in zip(begins, lengths):
                    self._top.add_measurements((name, begin, length))
            self.play_arbitrary_waveform(to_waveform(inner_program))

    def to_program(self) -> Optional[Loop]:
        if len(self._stack) != 1:
            warnings.warn("Creating program with active build stack.")
        if self._root.waveform or len(self._root.children) != 0:
            return self._root

    @classmethod
    def _testing_dummy(cls, stack):
        builder = cls()
        builder._stack = [StackFrame(loop, None) for loop in stack]
        builder._root = builder._stack[0].loop
        builder._top = builder._stack[-1].loop
        return builder
