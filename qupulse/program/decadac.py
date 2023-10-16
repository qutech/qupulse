from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, ContextManager, Iterable, Tuple

from qupulse import ChannelID, MeasurementWindow
from qupulse.parameter_scope import Scope, MappedScope, FrozenDict
from qupulse.program import (ProgramBuilder, HardwareTime, HardwareVoltage, Waveform, RepetitionCount, TimeType,
                             SimpleExpression)
from qupulse.expressions import sympy as sym_expr


# TODO: hackedy, hackedy
sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES = sym_expr.ALLOWED_NUMERIC_SCALAR_TYPES + (SimpleExpression,)

def _channel_to_voltage_location(ch: int) -> int:
    return 1545 + ch * 16


SAFE_MEMORY_LOCATIONS = tuple(range(40693, 40704))


@dataclass
class DecaDACASCIIProgram:
    script: str
    duration: TimeType


@dataclass
class StackFrame:
    iterating: Tuple[str, int]


class HoldFixed:
    channels: Sequence[Tuple[int, int]]


def _some_code():
    command = f'*{self._block_number}:'
    command += 'X' + str(768 + self._block_number) + ';'  # wait for the previous count to finish
    for channel in voltages:
        channel_number = self._name_to_idx(channel)
        dac_value = self.convert_voltage(voltages[channel])
        B_value = int(channel_number / 5)
        C_value = channel_number % 5

        command += f'B{B_value};C{C_value};D{dac_value};'

    duration = int(duration * 1000000)
    command += f'${duration}'

    self._block_number += 1


class DecaDACASCIIBuilder:
    def __init__(self, channels: Tuple[Optional[ChannelID], ...]):
        assert len(channels) in (20,), "Only 5 slots are supported for now"
        self._name_to_idx = {idx: name for idx, name in enumerate(channels) if name is not None}
        self._idx_to_name = channels

        self._block_number = 1
        self._iteration_count = 0

        self._stack = [[]]
        self._program = '{'

    @classmethod
    def from_channel_dict(cls, channels: Mapping[ChannelID, int]):
        assert len(set(channels.values())) == len(channels), "no duplicate target channels"
        channel_list = [None] * 20
        for ch_name, ch_idx in channels.items():
            channel_list[ch_idx] = ch_name
        return cls(tuple(channel_list))

    def convert_voltage_to_digit(self, voltage: float) -> int:
        # hardcoded range for now
        return int((voltage + 10) / 20 * 65535)

    def inner_scope(self, scope: Scope) -> Scope:
        """This function is necessary to inject program builder specific parameter implementations into the build
        process."""
        if self._stack:
            name, _ = self._stack[-1]
            return MappedScope(scope, FrozenDict({name: SimpleExpression(base=0, offsets=[(name, 1)])}))
        else:
            return scope

    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[ChannelID, HardwareVoltage]):
        fixed = {}
        variable = {}
        dependents = set()
        for ch_name, value in voltages.items():
            ch_idx = self._name_to_idx[ch_name]
            if isinstance(value, float):
                fixed[ch_idx] = self.convert_voltage_to_digit(value)
            else:
                base = self.convert_voltage_to_digit(value.base)
                offsets = [(name, self.convert_voltage_to_digit(offset)) for name, offset in value.offsets]
                dependents.update(name for name, _ in offsets)
                variable[ch_idx] = SimpleExpression(base, tuple(offsets))

        if isinstance(duration, SimpleExpression):
            raise NotImplementedError('TODO: support for swept durations')

        if variable:
            raise NotImplementedError('TODO: support for swept voltages')

        channels = sorted(fixed.values())

        self._stack[-1].append(HoldFixed(channels))

    def play_arbitrary_waveform(self, waveform: Waveform):
        raise NotImplementedError('Not implemented yet (postponed)')

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        """Ignores measurements"""
        pass

    def with_repetition(self, repetition_count: RepetitionCount,
                        measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        self._stack.append((repetition_count, []))
        yield self
        _, blocks = self._stack.pop()

        raise NotImplementedError('TODO: handle repetition block')
        command = f'*{self._block_number}:'
        location = str(self._safe_memory_locations[self._iteration_count])
        iteration_command = 'A' + location + ';P' + str(repetition_count) + ';'
        block_command = f'*{self._block_number}:'
        block_command += 'X' + str(768 + self._block_number) + ';'
        command = iteration_command + block_command

    def end_repetition(self):
        location = str(self._safe_memory_locations[self._iteration_count])
        X_command = 1280 + self._block_number + 1
        rep_flag = f'A{location};+-1;X' + str(X_command) + ';'
        self._block_number += 1
        self._iteration_count += 1
        self._program += rep_flag

        # TODO: nested repetitions are impossible with this implementation

    def with_sequence(self,
                      measurements: Optional[Sequence[MeasurementWindow]] = None) -> ContextManager['ProgramBuilder']:
        raise NotImplementedError('TODO')

    def new_subprogram(self, global_transformation: 'Transformation' = None) -> ContextManager['ProgramBuilder']:
        raise NotImplementedError('Not implemented yet (postponed)')

    def with_iteration(self, index_name: str, rng: range,
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        self._stack.append((index_name, rng, []))
        yield self
        _, _, cmds = self._stack.pop()
        if cmds:
            raise NotImplementedError('Process the generated commands')
        return
        channel_number = self._name_to_idx(index_name)
        dac_value = self.convert_voltage(rng.step)

        command = 'A' + str(self._channel_voltage_location[channel_number]) + ';'  # point to voltage location
        command += '+' + str(dac_value) + ';'
        self._program += command
        # TODO : no way to tell how long each iteration should last

    def to_program(self) -> Optional[DecaDACASCIIProgram]:
        """Further addition of new elements might fail after finalizing the program."""

        # TODO: implement proper triggering
        # TODO: implement additional compression if necessary

        raise NotImplementedError('TODO')
