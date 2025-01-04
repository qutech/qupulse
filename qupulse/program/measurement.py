import collections
import contextlib
from typing import Sequence, Mapping, Iterable, Optional, Union, ContextManager, Callable
from dataclasses import dataclass
from functools import cached_property

import numba as nb
import numpy as np

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

            return MeasurementInstructions(_merge_waits(self._frames[0].commands))


def _merge_waits(sequence: Sequence[Command]) -> Sequence[Command]:
    """Merges consecutive waits and removes trailing waits"""
    merged = []
    wait_duration = 0
    for command in sequence:
        if isinstance(command, Wait):
            wait_duration = wait_duration + command.duration
        else:
            if wait_duration != 0:
                merged.append(Wait(wait_duration))
                wait_duration = 0
            merged.append(command)
    if wait_duration != 0:
        # This is just here to document that we remove trailing waits
        pass
    return merged


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
        self._memory = {}
        self._counts = {}
        self._stack = []
        self._callback = callback

    def _eval_hardware_time(self, t: HardwareTime):
        if isinstance(t, SimpleExpression):
            value = t.base
            for (factor_name, factor_val) in t.offsets.items():
                count = self._stack[self._memory[factor_name]]
                value += factor_val * count
            return value
        else:
            return t

    def _execute(self, sequence: Sequence[Command]):
        time = TimeType(0)
        labels = {cmd.idx: pos for pos, cmd in enumerate(sequence) if isinstance(cmd, LoopLabel)}

        memory_map = self._memory
        stack = self._stack

        current = 0
        while current < len(sequence):
            cmd = sequence[current]
            if isinstance(cmd, LoopJmp):
                stack[-1] += 1
                pos = labels[cmd.idx]
                if stack[-1] < sequence[pos].count:
                    current = pos + 1
                    continue
                else:
                    stack.pop()

            elif isinstance(cmd, LoopLabel):
                if cmd.runtime_name:
                    memory_map[cmd.runtime_name] = len(stack)
                stack.append(0)

            elif isinstance(cmd, Measure):
                meas_time = float(self._eval_hardware_time(cmd.delay) + time)
                meas_len = float(self._eval_hardware_time(cmd.length))
                self._callback(cmd.meas_id, meas_time, meas_len)

            elif isinstance(cmd, Wait):
                time += self._eval_hardware_time(cmd.duration)
            current += 1

    def execute(self, commands: Sequence[Command]):
        self._execute(commands)


def to_table(commands: Sequence[Command]) -> dict[str, np.ndarray]:
    tables = {}

    vm = MeasurementVM(lambda name, begin, length: tables.setdefault(name, []).append((begin, length)))
    vm.execute(commands)
    return {
        name: np.array(values) for name, values in tables.items()
    }


_FastCommand = np.dtype([
    ('op_code', 'u4'),
    ('loop_count', 'u4'),
    ('meas_index', 'u4'),
    ('payload_1', 'i8', (8,)),
    ('payload_2', 'i8', (8,)),
])


@dataclass
class FastInstructions:
    time_base: int
    measurement_names: tuple[str, ...]
    measurement_lengths: tuple[int, ...]

    commands: np.ndarray

    def create_tables(self) -> dict[str, np.ndarray]:
        tables = _execute_fast(self.commands, self.measurement_lengths)
        return {
            name: arr.reshape(-1, 2) for name, arr in zip(self.measurement_names, tables)
        }

    @classmethod
    def from_commands(cls, time_base: int, sequence: Sequence[Command],
                      measurement_opcodes: Mapping[MeasurementID, int]) -> 'FastInstructions':
        """By default all measurements get the opcode ???. measuzrement opcodes allows to spcify that"""
        measurement_names = {}
        measurement_stack = [collections.defaultdict(lambda: 0)]
        
        loop_counts = {}

        commands = np.zeros_like(sequence, dtype=_FastCommand)
        stack = []
        for idx, command in enumerate(sequence):
            if isinstance(command, LoopJmp):
                stack.pop()

                for meas_id, meas_count in measurement_stack.pop().items():
                    measurement_stack[-1][meas_id] += meas_count * loop_counts[command.idx]

                commands['op_code'][idx] = 1
                commands['loop_count'][idx] = loop_counts[command.idx]

            elif isinstance(command, LoopLabel):
                stack.append(command.runtime_name)
                measurement_stack.append(collections.defaultdict(lambda: 0))
                
                commands['op_code'][idx] = 2
                loop_counts[command.idx] = commands['loop_count'][idx] =  command.count

            elif isinstance(command, Measure):
                measurement_stack[-1][command.meas_id] += 1

                commands['op_code'][idx] = measurement_opcodes.get(command.meas_id, 3)
                commands['meas_index'][idx] = measurement_names.setdefault(command.meas_id, len(measurement_names))
                delay_payload = _make_idx(time_base, stack, command.delay)
                length_payload = _make_idx(time_base, stack, command.length)

                commands['payload_1'][idx] = delay_payload
                commands['payload_2'][idx] = length_payload

            elif isinstance(command, Wait):
                commands['op_code'][idx] = 4
                commands['payload_1'][idx] = _make_idx(time_base, stack, command.duration)
            else:
                raise NotImplementedError("Unkonwn command", command)

        measurement_lengths, = measurement_stack
        meas_lengths = []
        meas_names = []
        for meas_id, pos in measurement_names.items():
            meas_lengths.append(measurement_lengths[meas_id])
            meas_names.append(meas_id)

        return cls(time_base, tuple(meas_names), tuple(meas_lengths), commands)


@nb.njit(inline='always')
def _evaluate_payload(stack: list[int], payload):
    value = np.int64(payload[0])
    for idx, factor in zip(stack, payload[1:]):
        value += factor * idx
    return value


tuple_type = nb.types.Tuple((nb.types.uint32, nb.types.uint32))


@nb.njit
def _execute_fast(commands,
                  measurement_lengths: Sequence[int]):
    tables = [np.zeros(1 + 2 * meas_len, dtype=np.int64)
              for meas_len in measurement_lengths]

    time = nb.int64(0)

    positions = [0] * 7
    values = [0] * 7
    stack_len = 0

    current = nb.uint32(0)
    while current < len(commands):
        op_code = commands['op_code'][current]
        if op_code == 0:
            # no op
            current += 1

        elif op_code == 1:
            # loop jump
            stack_len -= 1
            value = values[stack_len] + 1
            count = commands['loop_count'][current]
            if values[stack_len] < count:
                values[stack_len] += value
                current = positions[stack_len] + 1
                stack_len += 1
            else:
                current += 1

        elif op_code == 2:
            # loop label
            positions[stack_len] = current
            values[stack_len] = 0
            stack_len += 1
            current += 1

        elif op_code == 3:
            # create table entry
            meas_idx = commands['meas_index'][current]
            delay_payload = commands['payload_1'][current]
            length_payload = commands['payload_2'][current]

            delay = _evaluate_payload(values, delay_payload)
            length = _evaluate_payload(values, length_payload)
            begin = time + delay

            table = tables[meas_idx]
            meas_len = table[0]
            table[1 + meas_len * 2] = begin
            table[2 + meas_len * 2] = length
            table[0] += 1
            current += 1

        elif op_code == 4:
            duration_payload = commands['payload_1'][current]
            time += _evaluate_payload(values, duration_payload)
            current += 1

    return [t[1:].reshape(-1, 2) for t in tables]


def _to_int(v) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        assert v.is_integer()
        return int(v)
    assert v.denominator == 1
    return v.numerator


def _make_idx(time_base: int, stack: Sequence[str], val: HardwareTime) -> tuple[int, int, int, int, int, int, int, int]:
    scaled_val = val * time_base
    if isinstance(scaled_val, SimpleExpression):
        result = [_to_int(scaled_val.base)] + 7 * [0]
        for runtime_name, offset in scaled_val.offsets.items():
            idx = stack.index(runtime_name)
            assert result[idx] == 0
            result[idx] = _to_int(offset)
        return tuple(result)
    else:
        return _to_int(scaled_val), 0, 0, 0, 0, 0, 0, 0
