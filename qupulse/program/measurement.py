import contextlib
from typing import Sequence, Mapping, Iterable, Optional, Union, ContextManager, Callable
from dataclasses import dataclass
from functools import cached_property

import numpy

from qupulse.utils.types import TimeType
from qupulse.program import (ProgramBuilder, Program, HardwareVoltage, HardwareTime,
                             MeasurementWindow, Waveform, RepetitionCount, SimpleExpression)
from qupulse.parameter_scope import Scope


MeasurementID = str | int


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


@dataclass
class MeasurementInstructions(Program):
    commands: Sequence[Command]

    @cached_property
    def duration(self) -> float:
        latest = 0.

        def process(_, begin, length):
            nonlocal latest
            end = begin + length
            latest = max(latest, end)

        vm = MeasurementVM(process)
        vm.execute(commands=self.commands)
        return latest


@dataclass
class MeasurementFrame:
    commands: list['Command']
    keep: bool


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
        if not frame.keep:
            return
        self.measure(measurements)
        # measure does not keep if there are no measurements
        self._frames[-1].keep = True
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
        self._frames[-1].keep = True

    def play_arbitrary_waveform(self, waveform: Waveform):
        """"""
        self._frames[-1].commands.append(Wait(waveform.duration))
        self._frames[-1].keep = True

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        """Unconditionally add given measurements relative to the current position."""
        if measurements:
            commands = self._frames[-1].commands
            commands.extend(Measure(*meas) for meas in measurements)
            self._frames[-1].keep = True

    def with_repetition(self, repetition_count: RepetitionCount,
                        measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        """Measurements that are added to the new builder are dropped if the builder is empty upon exit"""
        new_commands = yield from self._with_new_frame(measurements)
        if new_commands is None:
            return
        parent = self._frames[-1]

        self._label_counter += 1
        label_idx = self._label_counter
        parent.commands.append(LoopLabel(idx=label_idx, runtime_name=None, count=repetition_count))
        parent.commands.extend(new_commands)
        parent.commands.append(LoopJmp(idx=label_idx))

    @contextlib.contextmanager
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

    @contextlib.contextmanager
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
        if not frame.keep:
            return

        self._frames[-1].keep = True
        self._frames[-1].commands.extend(_reversed_commands(frame.commands))

    def to_program(self) -> Optional[Program]:
        """Further addition of new elements might fail after finalizing the program."""
        if self._frames[0].keep:
            return MeasurementInstructions(self._frames[0].commands)


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


class MeasurementVM:
    """A VM that is capable of executing the measurement commands"""

    def __init__(self, callback: Callable[[str, float, float], None]):
        self._time = TimeType(0)
        self._memory = {}
        self._counts = {}
        self._callback = callback

    def _eval_hardware_time(self, t: HardwareTime):
        if isinstance(t, SimpleExpression):
            value = t.base
            for (factor_name, factor_val) in t.offsets.items():
                count = self._counts[self._memory[factor_name]]
                value += factor_val * count
            return value
        else:
            return t

    def _execute_after_label(self, sequence: Sequence[Command]) -> int:
        skip = 0
        for idx, cmd in enumerate(sequence):
            if idx < skip:
                continue
            if isinstance(cmd, LoopJmp):
                return idx
            elif isinstance(cmd, LoopLabel):
                if cmd.runtime_name:
                    self._memory[cmd.runtime_name] = cmd.idx

                for iter_val in range(cmd.count):
                    self._counts[cmd.idx] = iter_val
                    pos = self._execute_after_label(sequence[idx + 1:])
                skip = idx + pos + 2

            elif isinstance(cmd, Measure):
                meas_time = float(self._eval_hardware_time(cmd.delay) + self._time)
                meas_len = float(self._eval_hardware_time(cmd.length))
                self._callback(cmd.meas_id, meas_time, meas_len)

            elif isinstance(cmd, Wait):
                self._time += self._eval_hardware_time(cmd.duration)

    def execute(self, commands: Sequence[Command]):
        self._execute_after_label(commands)


def to_table(commands: Sequence[Command]) -> dict[str, numpy.ndarray]:
    tables = {}

    vm = MeasurementVM(lambda name, begin, length: tables.setdefault(name, []).append((begin, length)))
    vm.execute(commands)
    return {
        name: numpy.array(values) for name, values in tables.items()
    }
