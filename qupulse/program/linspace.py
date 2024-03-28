import abc
import contextlib
import dataclasses
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, ContextManager, Iterable, Tuple, Union, Dict, List

from qupulse import ChannelID, MeasurementWindow
from qupulse.parameter_scope import Scope, MappedScope, FrozenDict
from qupulse.program import (ProgramBuilder, HardwareTime, HardwareVoltage, Waveform, RepetitionCount, TimeType,
                             SimpleExpression)
from qupulse.program.waveforms import MultiChannelWaveform

# this resolution is used to unify increments
# the increments themselves remain floats
DEFAULT_INCREMENT_RESOLUTION: float = 1e-9


@dataclass(frozen=True)
class DepKey:
    """The key that identifies how a certain set command depends on iteration indices. The factors are rounded with a
    given resolution to be independent on rounding errors.

    These objects allow backends which support it to track multiple amplitudes at once.
    """
    factors: Tuple[int, ...]

    @classmethod
    def from_voltages(cls, voltages: Sequence[float], resolution: float):
        # remove trailing zeros
        while voltages and voltages[-1] == 0:
            voltages = voltages[:-1]
        return cls(tuple(int(round(voltage / resolution)) for voltage in voltages))


@dataclass
class LinSpaceNode:
    """AST node for a program that supports linear spacing of set points as well as nested sequencing and repetitions"""

    def dependencies(self) -> Mapping[int, set]:
        raise NotImplementedError


@dataclass
class LinSpaceHold(LinSpaceNode):
    """Hold voltages for a given time. The voltages and the time may depend on the iteration index."""

    bases: Tuple[float, ...]
    factors: Tuple[Optional[Tuple[float, ...]], ...]

    duration_base: TimeType
    duration_factors: Optional[Tuple[TimeType, ...]]

    def dependencies(self) -> Mapping[int, set]:
        return {idx: {factors}
                for idx, factors in enumerate(self.factors)
                if factors}


@dataclass
class LinSpaceArbitraryWaveform(LinSpaceNode):
    """This is just a wrapper to pipe arbitrary waveforms through the system."""
    waveform: Waveform
    channels: Tuple[ChannelID, ...]


@dataclass
class LinSpaceRepeat(LinSpaceNode):
    """Repeat the body count times."""
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
    """Iteration in linear space are restricted to range 0 to length.

    Offsets and spacing are stored in the hold node."""
    body: Tuple[LinSpaceNode, ...]
    length: int

    def dependencies(self):
        dependencies = {}
        for node in self.body:
            for idx, deps in node.dependencies().items():
                # remove the last elemt in index because this iteration sets it -> no external dependency
                shortened = {dep[:-1] for dep in deps}
                if shortened != {()}:
                    dependencies.setdefault(idx, set()).update(shortened)
        return dependencies


class LinSpaceBuilder(ProgramBuilder):
    """This program builder supports efficient translation of pulse templates that use symbolic linearly
    spaced voltages and durations.

    The channel identifiers are reduced to their index in the given channel tuple.

    Arbitrary waveforms are not implemented yet
    """

    def __init__(self, channels: Tuple[ChannelID, ...]):
        super().__init__()
        self._name_to_idx = {name: idx for idx, name in enumerate(channels)}
        self._idx_to_name = channels

        self._stack = [[]]
        self._ranges = []

    def _root(self):
        return self._stack[0]

    def _get_rng(self, idx_name: str) -> range:
        return self._get_ranges()[idx_name]

    def inner_scope(self, scope: Scope) -> Scope:
        """This function is necessary to inject program builder specific parameter implementations into the build
        process."""
        if self._ranges:
            name, _ = self._ranges[-1]
            return scope.overwrite({name: SimpleExpression(base=0, offsets=[(name, 1)])})
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
        return self._stack[-1].append(LinSpaceArbitraryWaveform(waveform, self._idx_to_name))

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
    dependency_key: DepKey


@dataclass
class Set:
    channel: int
    value: float
    key: DepKey = dataclasses.field(default_factory=lambda: DepKey(()))


@dataclass
class Wait:
    duration: TimeType


@dataclass
class LoopJmp:
    idx: int


@dataclass
class Play:
    waveform: Waveform
    channels: Tuple[ChannelID]


Command = Union[Increment, Set, LoopLabel, LoopJmp, Wait, Play]


@dataclass(frozen=True)
class DepState:
    base: float
    iterations: Tuple[int, ...]

    def required_increment_from(self, previous: 'DepState', factors: Sequence[float]) -> float:
        assert len(self.iterations) == len(previous.iterations)
        assert len(self.iterations) == len(factors)

        increment = self.base - previous.base
        for old, new, factor in zip(previous.iterations, self.iterations, factors):
            # By convention there are only two possible values for each integer here: 0 or the last index
            # The three possible increments are none, regular and jump to next line

            if old == new:
                # we are still in the same iteration of this sweep
                pass

            elif old < new:
                assert old == 0
                # regular iteration, although the new value will probably be > 1, the resulting increment will be
                # applied multiple times so only one factor is needed.
                increment += factor

            else:
                assert new == 0
                # we need to jump back. The old value gives us the number of increments to reverse
                increment -= factor * old
        return increment


@dataclass
class _TranslationState:
    """This is the state of a translation of a LinSpace program to a command sequence."""

    label_num: int = dataclasses.field(default=0)
    commands: List[Command] = dataclasses.field(default_factory=list)
    iterations: List[int] = dataclasses.field(default_factory=list)
    active_dep: Dict[int, DepKey] = dataclasses.field(default_factory=dict)
    dep_states: Dict[int, Dict[DepKey, DepState]] = dataclasses.field(default_factory=dict)
    plain_voltage: Dict[int, float] = dataclasses.field(default_factory=dict)
    resolution: float = dataclasses.field(default_factory=lambda: DEFAULT_INCREMENT_RESOLUTION)

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

    def _add_repetition_node(self, node: LinSpaceRepeat):
        pre_dep_state = self.get_dependency_state(node.dependencies())
        label, jmp = self.new_loop(node.count)
        initial_position = len(self.commands)
        self.commands.append(label)
        self.add_node(node.body)
        post_dep_state = self.get_dependency_state(node.dependencies())
        if pre_dep_state != post_dep_state:
            # hackedy
            self.commands.pop(initial_position)
            self.commands.append(label)
            label.count -= 1
            self.add_node(node.body)
        self.commands.append(jmp)

    def _add_iteration_node(self, node: LinSpaceIter):
        self.iterations.append(0)
        self.add_node(node.body)

        if node.length > 1:
            self.iterations[-1] = node.length
            label, jmp = self.new_loop(node.length - 1)
            self.commands.append(label)
            self.add_node(node.body)
            self.commands.append(jmp)
        self.iterations.pop()

    def _set_indexed_voltage(self, channel: int, base: float, factors: Sequence[float]):
        dep_key = DepKey.from_voltages(voltages=factors, resolution=self.resolution)
        new_dep_state = DepState(
            base,
            iterations=tuple(self.iterations)
        )

        current_dep_state = self.dep_states.setdefault(channel, {}).get(dep_key, None)
        if current_dep_state is None:
            assert all(it == 0 for it in self.iterations)
            self.commands.append(Set(channel, base, dep_key))
            self.active_dep[channel] = dep_key

        else:
            inc = new_dep_state.required_increment_from(previous=current_dep_state, factors=factors)

            # we insert all inc here (also inc == 0) because it signals to activate this amplitude register
            if inc or self.active_dep.get(channel, None) != dep_key:
                self.commands.append(Increment(channel, inc, dep_key))
            self.active_dep[channel] = dep_key
        self.dep_states[channel][dep_key] = new_dep_state

    def _add_hold_node(self, node: LinSpaceHold):
        if node.duration_factors:
            raise NotImplementedError("TODO")

        for ch, (base, factors) in enumerate(zip(node.bases, node.factors)):
            if factors is None:
                self.set_voltage(ch, base)
                continue

            else:
                self._set_indexed_voltage(ch, base, factors)

        self.commands.append(Wait(node.duration_base))

    def add_node(self, node: Union[LinSpaceNode, Sequence[LinSpaceNode]]):
        """Translate a (sequence of) linspace node(s) to commands and add it to the internal command list."""
        if isinstance(node, Sequence):
            for lin_node in node:
                self.add_node(lin_node)

        elif isinstance(node, LinSpaceRepeat):
            self._add_repetition_node(node)

        elif isinstance(node, LinSpaceIter):
            self._add_iteration_node(node)

        elif isinstance(node, LinSpaceHold):
            self._add_hold_node(node)

        elif isinstance(node, LinSpaceArbitraryWaveform):
            self.commands.append(Play(node.waveform, node.channels))

        else:
            raise TypeError("The node type is not handled", type(node), node)


def to_increment_commands(linspace_nodes: Sequence[LinSpaceNode]) -> List[Command]:
    """translate the given linspace node tree to a minimal sequence of set and increment commands as well as loops."""
    state = _TranslationState()
    state.add_node(linspace_nodes)
    return state.commands
