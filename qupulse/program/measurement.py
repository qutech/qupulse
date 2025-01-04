from typing import Sequence, Mapping, Iterable, Optional, Union, ContextManager
from dataclasses import dataclass

import numpy
from rich.measure import Measurement

from qupulse.utils.types import TimeType
from qupulse.program import (ProgramBuilder, Program, HardwareVoltage, HardwareTime,
                             MeasurementWindow, Waveform, RepetitionCount, SimpleExpression)
from qupulse.parameter_scope import Scope


@dataclass
class MeasurementNode:
    windows: Sequence[MeasurementWindow]
    duration: HardwareTime


@dataclass
class MeasurementRepetition(MeasurementNode):
    body: MeasurementNode
    count: RepetitionCount

@dataclass
class MeasurementSequence(MeasurementNode):
    nodes: Sequence[tuple[HardwareTime, MeasurementNode]]


@dataclass
class MeasurementFrame:
    commands: list['Command']
    has_duration: bool

MeasurementID = str | int


class MeasurementBuilder(ProgramBuilder):
    def __init__(self):
        super().__init__()

        self._frames = [MeasurementFrame([], False)]
        self._ranges: list[tuple[str, range]] = []
        self._repetitions = []
        self._measurements = []
        self._label_counter = 0

    def _with_new_frame(self, measurements):
        self._frames.append(MeasurementFrame([], False))
        yield self
        frame = self._frames.pop()
        if not frame.has_duration:
            return
        parent = self._frames[-1]
        parent.has_duration = True
        if measurements:
            parent.commands.extend(map(Measure, measurements))
        return frame.commands

    def inner_scope(self, scope: Scope) -> Scope:
        """This function is necessary to inject program builder specific parameter implementations into the build
        process."""
        if self._ranges:
            name, rng = self._ranges[-1]
            return scope.overwrite({name: SimpleExpression(base=rng.start, offsets={name: rng.step})})
        else:
            return scope

    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[str, HardwareVoltage]):
        """Supports dynamic i.e. for loop generated offsets and duration"""
        self._frames[-1].commands.append(Wait(duration))
        self._frames[-1].has_duration = True

    def play_arbitrary_waveform(self, waveform: Waveform):
        """"""
        self._frames[-1].commands.append(Wait(waveform.duration))
        self._frames[-1].has_duration = True

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        """Unconditionally add given measurements relative to the current position."""
        if measurements:
            commands = self._frames[-1].commands
            commands.extend(Measure(*meas) for meas in measurements)
            self._frames[-1].has_duration = True

    def with_repetition(self, repetition_count: RepetitionCount,
                        measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        """Measurements that are added to the new builder are dropped if the builder is empty upon exit"""
        new_commands = yield from self._with_new_frame(measurements)
        if new_commands is None:
            return
        parent = self._frames[-1]

        self._label_counter += 1
        label_idx = self._label_counter
        parent.commands.append(LoopLabel(idx=label_idx, runtime_name=None, count=RepetitionCount))
        parent.commands.extend(new_commands)
        parent.commands.append(LoopJmp(idx=label_idx))

    def with_sequence(self,
                      measurements: Optional[Sequence[MeasurementWindow]] = None) -> ContextManager['ProgramBuilder']:
        """

        Measurements that are added in to the returned program builder are discarded if the sequence is empty on exit.

        Args:
            measurements: Measurements to attach to the potential child.
        Returns:
        """
        new_commands = yield from self._with_new_frame(measurements)
        if new_commands is None:
            return
        parent = self._frames[-1]
        parent.commands.extend(new_commands)

    def new_subprogram(self, global_transformation: 'Transformation' = None) -> ContextManager['ProgramBuilder']:
        """Create a context managed program builder whose contents are translated into a single waveform upon exit if
        it is not empty."""
        yield self

    def with_iteration(self, index_name: str, rng: range,
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        self._ranges.append((index_name, rng))
        new_commands = yield from self._with_new_frame(measurements)
        self._ranges.pop()
        if new_commands is None:
            return
        parent = self._frames[-1]

        self._label_counter += 1
        label_idx = self._label_counter
        parent.commands.append(LoopLabel(idx=label_idx, runtime_name=index_name, count=len(rng)))
        parent.commands.extend(new_commands)
        parent.commands.append(LoopJmp(idx=label_idx))

    def time_reversed(self) -> ContextManager['ProgramBuilder']:
        self._frames.append(MeasurementFrame([], False))
        yield self
        frame = self._frames.pop()
        if not frame.has_duration:
            return

        self._frames[-1].has_duration = True
        self._frames[-1].commands.extend(_reversed_commands(frame.commands))

    def to_program(self) -> Optional[Program]:
        """Further addition of new elements might fail after finalizing the program."""
        if self._frames[0].has_duration:
            return self._frames[0].commands


@dataclass
class LoopLabel:
    idx: int
    runtime_name: str | None
    count: RepetitionCount


@dataclass
class Measure:
    meas_id: MeasurementID
    delay: HardwareTime
    length: HardwareTime


@dataclass
class Wait:
    duration: HardwareTime


@dataclass
class LoopJmp:
    idx: int


Command = Union[LoopLabel, LoopJmp, Wait, Measure]


def _reversed_commands(cmds: Sequence[Command]) -> Sequence[Command]:
    reversed_cmds = []
    jumps = {}
    for cmd in reversed(cmds):
        if isinstance(cmd, LoopJmp):
            jumps[cmd.idx] = len(reversed_cmds)
            reversed_cmds.append(cmd)
        elif isinstance(cmd, LoopLabel):
            jump_idx = jumps[cmd.idx]
            jump = reversed_cmds[jump_idx]
            reversed_cmds[jump_idx] = cmd
            reversed_cmds.append(jump)

        elif isinstance(cmd, Measure):
            if isinstance(cmd.delay, SimpleExpression) or isinstance(cmd.delay, SimpleExpression):
                raise NotImplementedError("TODO")
            reversed_cmds.append(Measure(meas_id=cmd.meas_id,
                                         delay=-(cmd.delay + cmd.length),
                                         length=cmd.length,))
        elif isinstance(cmd, Wait):
            reversed_cmds.append(cmd)
        else:
            raise ValueError("Not a command", cmd)

    return reversed_cmds


def to_table(commands: Sequence[Command]) -> dict[str, numpy.ndarray]:
    time = TimeType(0)

    memory = {}
    counts = [None]

    tables = {}

    def eval_hardware_time(t: HardwareTime):
        if isinstance(t, SimpleExpression):
            value = t.base
            for (factor_name, factor_val) in t.offsets.items():
                count = counts[memory[factor_name]]
                value += factor_val * count
            return value
        else:
            return t

    def execute(sequence: Sequence[Command]) -> int:
        nonlocal time
        nonlocal tables
        nonlocal memory
        nonlocal counts

        skip = 0
        for idx, cmd in enumerate(sequence):
            if idx < skip:
                continue
            if isinstance(cmd, LoopJmp):
                return idx
            elif isinstance(cmd, LoopLabel):
                if cmd.runtime_name:
                    memory[cmd.runtime_name] = cmd.idx
                if cmd.idx == len(counts):
                    counts.append(0)
                assert cmd.idx < len(counts)

                for iter_val in range(cmd.count):
                    counts[cmd.idx] = iter_val
                    pos = execute(sequence[idx + 1:])
                skip = idx + pos + 2
            elif isinstance(cmd, Measure):
                meas_time = float(eval_hardware_time(cmd.delay) + time)
                meas_len = float(eval_hardware_time(cmd.length))
                tables.setdefault(cmd.meas_id, []).append((meas_time, meas_len))
            elif isinstance(cmd, Wait):
                time += eval_hardware_time(cmd.duration)

    execute(commands)
    return {
        name: numpy.array(values) for name, values in tables.items()
    }
