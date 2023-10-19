import abc
import contextlib
import dataclasses
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, ContextManager, Iterable, Tuple, Union, Dict, List

from qupulse import ChannelID, MeasurementWindow
from qupulse.parameter_scope import Scope, MappedScope, FrozenDict
from qupulse.program import (ProgramBuilder, HardwareTime, HardwareVoltage, Waveform, RepetitionCount, TimeType,
                             SimpleExpression)
from qupulse.expressions import sympy as sym_expr


DEFAULT_RESOLUTION: float = 1e-9


@dataclass
class LinSpaceNode:
    """AST node for a program that supports linear spacing of set points as well as nested sequencing and repetitions"""
    def dependencies(self) -> Mapping[int, set]:
        raise NotImplementedError


@dataclass
class LinSpaceSet:
    channel: int
    base: float
    factors: Optional[Tuple[float, ...]]



@dataclass
class Wait:
    duration: TimeType


@dataclass
class LinSpaceHold(LinSpaceNode):
    bases: Tuple[float, ...]
    factors: Tuple[Optional[Tuple[float, ...]], ...]

    duration_base: TimeType
    duration_factors: Optional[Tuple[TimeType, ...]]

    def dependencies(self) -> Mapping[int, set]:
        return {idx: {factors}
                for idx, factors in enumerate(self.factors)
                if factors}

    def to_atomic_commands(self):
        if self.duration_factors:
            raise NotImplementedError('Variable durations are not implemented for  commands yet')
        return [LinSpaceSet(idx, base, factors)
                for idx, (base, factors) in enumerate(zip(self.bases, self.factors))] + [Wait(self.duration_base)]

    def to_increment_commands(self, previous: Tuple[float, ...], iter_advance: Sequence[bool]):
        if self.duration_factors:
            raise NotImplementedError('Variable durations are not implemented for increment commands yet')
        set_vals = []
        inc_vals = []
        for prev, base, factors in zip(previous, self.bases, self.factors):
            set_val = base
            if set_val == prev:
                # TODO: epsilon
                set_val = None

            if factors:
                inc_val = 0.
                for advance, factor in zip(iter_advance, factors):
                    if advance:
                        inc_val += factor

                inc_val = None
                if base != prev:

                    set_vals.append(base)
                else:
                    pass
            assert inc_val is None or set_val is None
            inc_vals.append(inc_val)
            set_vals.append(set_val)


@dataclass
class LinSpaceRepeat(LinSpaceNode):
    body: Tuple[LinSpaceNode, ...]
    count: int

    def dependencies(self):
        dependencies = {}
        for node in self.body:
            for idx, deps in node.dependencies().items():
                dependencies.setdefault(idx, set()).update(deps)
        return dependencies


@dataclass
class LinSpaceIter(LinSpaceNode):
    """Iteration in linear space are restricted to range 0 to length. Offsets and spacing are stored in the set node."""
    body: Tuple[LinSpaceNode, ...]
    length: int

    def dependencies(self):
        dependencies = {}
        for node in self.body:
            for idx, deps in node.dependencies().items():
                shortened = {dep[:-1] for dep in deps}
                if shortened != {()}:
                    dependencies.setdefault(idx, set()).update(deps)
        return dependencies


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

        set_cmd = LinSpaceHold(bases=tuple(bases),
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



@dataclass
class LoopLabel:
    idx: int
    count: int


@dataclass
class Increment:
    channel: int
    value: float
    dependency_key: 'DepKey'


@dataclass
class Set:
    channel: int
    value: float
    key: 'DepKey' = dataclasses.field(default_factory=lambda: DepKey(()))


@dataclass
class LoopJmp:
    idx: int


@dataclass(frozen=True)
class DepState:
    base: float
    iterations: Tuple[int, ...]


@dataclass(frozen=True)
class DepKey:
    """The key that identifies how a certain set command depends on iteration indices."""
    factors: Tuple[int, ...]

    @classmethod
    def from_voltages(cls, voltages: Sequence[float], resolution: float):
        # remove trailing zeros
        while voltages and voltages[-1] == 0:
            voltages = voltages[:-1]
        return cls(tuple(int(round(voltage / resolution)) for voltage in voltages))


@dataclass
class TranslationState:
    label_num: int
    commands: list
    iterations: list
    active_dep: Dict[int, DepKey]
    dep_states: Dict[int, Dict[DepKey, DepState]]
    plain_voltage: Dict[int, float]
    resolution: float = dataclasses.field(default_factory=lambda: DEFAULT_RESOLUTION)

    def new_loop(self, count: int):
        label = LoopLabel(self.label_num, count)
        jmp = LoopJmp(self.label_num)
        self.label_num += 1
        return label, jmp

    def get_dependency_state(self, dependencies: Mapping[int, set]):
        return {
            self.dep_states.get(ch, {}).get(DepKey.from_voltages(dep, self.resolution), None)
            for ch, deps in dependencies.items()
            for dep in deps
        }

    def set_voltage(self, channel: int, value: float):
        key = DepKey(())
        if self.active_dep.get(channel, None) != key or self.plain_voltage.get(channel, None) != value:
            self.commands.append(Set(channel, value, key))
            self.active_dep[channel] = key
            self.plain_voltage[channel] = value


def to_atomic_commands(node: Union[LinSpaceNode, Sequence[LinSpaceNode]], state: TranslationState):
    """This step replaces iterations with """
    if isinstance(node, Sequence):
        for lin_node in node:
            to_atomic_commands(lin_node, state)

    if isinstance(node, LinSpaceRepeat):
        pre_dep_state = state.get_dependency_state(node.dependencies())
        label, jmp = state.new_loop(node.count)
        initial_position = len(state.commands)
        state.commands.append(label)
        to_atomic_commands(node.body, state)
        post_dep_state = state.get_dependency_state(node.dependencies())
        if pre_dep_state != post_dep_state:
            # hackedy
            state.commands.pop(initial_position)
            state.commands.append(label)
            label.count -= 1
            to_atomic_commands(node.body, state)
        state.commands.append(jmp)

    elif isinstance(node, LinSpaceIter):
        state.iterations.append(0)
        to_atomic_commands(node.body, state)

        if node.length > 1:
            state.iterations[-1] = node.length
            label, jmp = state.new_loop(node.length - 1)
            state.commands.append(label)
            to_atomic_commands(node.body, state)
            state.commands.append(jmp)
        state.iterations.pop()

    elif isinstance(node, LinSpaceHold):
        if node.duration_factors:
            raise NotImplementedError("TODO")

        for ch, (base, factors) in enumerate(zip(node.bases, node.factors)):
            if factors is None:
                state.set_voltage(ch, base)
                continue

            dep_key = DepKey.from_voltages(voltages=factors, resolution=state.resolution)
            new_dep_state = DepState(
                base,
                iterations=tuple(state.iterations)
            )

            current_dep_state = state.dep_states.setdefault(ch, {}).get(dep_key, None)
            if current_dep_state is None:
                assert all(it == 0 for it in state.iterations)
                state.commands.append(Set(ch, base, dep_key))
                state.active_dep[ch] = dep_key

            else:
                inc = current_dep_state.base - new_dep_state.base
                for i, j, factor in zip(current_dep_state.iterations, new_dep_state.iterations, factors):
                    if i == j:
                        continue
                    if i < j:
                        assert i == 0
                        # regular iteration
                        inc += factor
                    else:
                        assert j == 0
                        inc -= factor * i
                # we insert all inc here (also inc == 0) because it signals to activate this amplitude register
                if inc or state.active_dep.get(ch, None) != dep_key:
                    state.commands.append(Increment(ch, inc, dep_key))
                state.active_dep[ch] = dep_key
            state.dep_states[ch][dep_key] = new_dep_state
        state.commands.append(Wait(node.duration_base))


def to_increment_commands(linspace_nodes: Sequence[LinSpaceNode]) -> list:
    state = TranslationState(0, [], [], active_dep={}, dep_states={}, plain_voltage={})
    to_atomic_commands(linspace_nodes, state)
    return state.commands





