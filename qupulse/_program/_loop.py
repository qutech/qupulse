import itertools
from typing import Union, Dict, Set, Iterable, FrozenSet, Tuple, cast, List, Optional, DefaultDict, Generator
from collections import defaultdict, deque
from copy import deepcopy
from enum import Enum
import warnings

import numpy as np

from qupulse.utils.types import ChannelID, TimeType
from qupulse._program.instructions import AbstractInstructionBlock, EXECInstruction, REPJInstruction, GOTOInstruction,\
    STOPInstruction, CHANInstruction, Waveform, MEASInstruction, Instruction
from qupulse.utils.tree import Node, is_tree_circular
from qupulse.utils.types import MeasurementWindow
from qupulse.utils import is_integer

from qupulse._program.waveforms import SequenceWaveform, RepetitionWaveform

__all__ = ['Loop', 'MultiChannelProgram', 'make_compatible']


class Loop(Node):
    MAX_REPR_SIZE = 2000

    """Build a loop tree. The leaves of the tree are loops with one element."""
    def __init__(self,
                 parent: Union['Loop', None]=None,
                 children: Iterable['Loop']=list(),
                 waveform: Optional[Waveform]=None,
                 measurements: Optional[List[MeasurementWindow]]=None,
                 repetition_count=1):
        super().__init__(parent=parent, children=children)

        self._waveform = waveform
        self._measurements = measurements
        self._repetition_count = int(repetition_count)
        self._cached_body_duration = None

        if abs(self._repetition_count - repetition_count) > 1e-10:
            raise ValueError('Repetition count was not an integer')

        if not isinstance(waveform, (type(None), Waveform)):
            raise Exception()

    @property
    def compare_key(self) -> Tuple:
        return self._waveform, self.repetition_count, self._measurements if self._measurements else None,\
               super().compare_key

    def append_child(self, loop: Optional['Loop']=None, **kwargs) -> None:
        # do not invalidate but update cached duration
        if loop is not None:
            if kwargs:
                raise ValueError("Cannot pass a Loop object and Loop constructor arguments at the same time in append_child")
            arg = (loop,)
        else:
            arg = (kwargs,)
        super().__setitem__(slice(len(self), len(self)), arg)
        self._invalidate_duration(body_duration_increment=self[-1].duration)

    def _invalidate_duration(self, body_duration_increment=None):
        if self._cached_body_duration is not None:
            if body_duration_increment is not None:
                self._cached_body_duration += body_duration_increment
            else:
                self._cached_body_duration = None
        if self.parent:
            if body_duration_increment is not None:
                self.parent._invalidate_duration(body_duration_increment=body_duration_increment*self.repetition_count)
            else:
                self.parent._invalidate_duration()

    def add_measurements(self, measurements: List[MeasurementWindow]):
        body_duration = float(self.body_duration)
        if body_duration == 0:
            measurements = measurements
        else:
            measurements = ((mw_name, begin+body_duration, length) for mw_name, begin, length in measurements)

        if self._measurements is None:
            self._measurements = list(measurements)
        else:
            self._measurements.extend(measurements)

    @property
    def waveform(self) -> Waveform:
        return self._waveform

    @waveform.setter
    def waveform(self, val) -> None:
        self._waveform = val
        self._invalidate_duration()

    @property
    def body_duration(self) -> TimeType:
        if self._cached_body_duration is None:
            if self.is_leaf():
                if self.waveform:
                    self._cached_body_duration = self.waveform.duration
                else:
                    self._cached_body_duration = TimeType(0)
            else:
                self._cached_body_duration = sum(child.duration for child in self)
        return self._cached_body_duration

    @property
    def duration(self) -> TimeType:
        return self.repetition_count*self.body_duration

    @property
    def repetition_count(self) -> int:
        return self._repetition_count

    @repetition_count.setter
    def repetition_count(self, val) -> None:
        new_repetition = int(val)
        if abs(new_repetition - val) > 1e-10:
            raise ValueError('Repetition count was not an integer')
        self._repetition_count = new_repetition

    def unroll(self) -> None:
        if self.is_leaf():
            raise RuntimeError('Leaves cannot be unrolled')

        i = self.parent_index
        self.parent[i:i+1] = (child.copy_tree_structure(new_parent=self.parent)
                              for _ in range(self.repetition_count)
                              for child in self)
        self.parent.assert_tree_integrity()

    def __setitem__(self, idx, value):
        super().__setitem__(idx, value)
        self._invalidate_duration()

    def unroll_children(self) -> None:
        old_children = self.children
        self[:] = (child.copy_tree_structure()
                   for _ in range(self.repetition_count)
                   for child in old_children)
        self.repetition_count = 1
        self.assert_tree_integrity()

    def encapsulate(self) -> None:
        self[:] = [Loop(children=self,
                        repetition_count=self.repetition_count,
                        waveform=self._waveform,
                        measurements=self._measurements)]
        self.repetition_count = 1
        self._waveform = None
        self._measurements = None
        self.assert_tree_integrity()

    def _get_repr(self, first_prefix, other_prefixes) -> Generator[str, None, None]:
        if self.is_leaf():
            yield '%sEXEC %r %d times' % (first_prefix, self._waveform, self.repetition_count)
        else:
            yield '%sLOOP %d times:' % (first_prefix, self.repetition_count)

            for elem in self:
                yield from cast(Loop, elem)._get_repr(other_prefixes + '  ->', other_prefixes + '    ')

    def __repr__(self) -> str:
        is_circular = is_tree_circular(self)
        if is_circular:
            return '{}: Circ {}'.format(id(self), is_circular)

        str_len = 0
        repr_list = []
        for sub_repr in self._get_repr('', ''):
            str_len += len(sub_repr)

            if self.MAX_REPR_SIZE and str_len > self.MAX_REPR_SIZE:
                repr_list.append('...')
                break
            else:
                repr_list.append(sub_repr)
        return '\n'.join(repr_list)

    def copy_tree_structure(self, new_parent: Union['Loop', bool]=False) -> 'Loop':
        return type(self)(parent=self.parent if new_parent is False else new_parent,
                          waveform=self._waveform,
                          repetition_count=self.repetition_count,
                          measurements=self._measurements,
                          children=(child.copy_tree_structure() for child in self))

    def _get_measurement_windows(self) -> DefaultDict[str, np.ndarray]:
        temp_meas_windows = defaultdict(list)
        if self._measurements:
            for (mw_name, begin, length) in self._measurements:
                temp_meas_windows[mw_name].append((begin, length))

            for mw_name, begin_length_list in temp_meas_windows.items():
                temp_meas_windows[mw_name] = [np.asarray(begin_length_list, dtype=float)]

        # calculate duration together with meas windows in the same iteration
        if self.is_leaf():
            body_duration = float(self.body_duration)
        else:
            offset = TimeType(0)
            for child in self:
                for mw_name, begins_length_array in child._get_measurement_windows().items():
                    begins_length_array[:, 0] += float(offset)
                    temp_meas_windows[mw_name].append(begins_length_array)
                offset += child.duration

            body_duration = float(offset)

        # repeat and add repetition based offset
        for mw_name, begin_length_list in temp_meas_windows.items():
            temp_begin_length_array = np.concatenate(begin_length_list)

            begin_length_array = np.tile(temp_begin_length_array, (self.repetition_count, 1))

            shaped_begin_length_array = np.reshape(begin_length_array, (self.repetition_count, -1, 2))

            shaped_begin_length_array[:, :, 0] += (np.arange(self.repetition_count) * body_duration)[:, np.newaxis]

            temp_meas_windows[mw_name] = begin_length_array

        return temp_meas_windows

    def get_measurement_windows(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {mw_name: (begin_length_list[:, 0], begin_length_list[:, 1])
                for mw_name, begin_length_list in self._get_measurement_windows().items()}

    def split_one_child(self, child_index=None) -> None:
        """Take the last child that has a repetition count larger one, decrease it's repetition count and insert a copy
        with repetition cout one after it"""
        if child_index:
            if self[child_index].repetition_count < 2:
                raise ValueError('Cannot split child {} as the repetition count is not larger 1')
        else:
            try:
                child_index = next(i for i in reversed(range(len(self)))
                                   if self[i].repetition_count > 1)
            except StopIteration:
                raise RuntimeError('There is no child with repetition count > 1')

        new_child = self[child_index].copy_tree_structure()
        new_child.repetition_count = 1

        self[child_index].repetition_count -= 1

        self[child_index+1:child_index+1] = (new_child,)
        self.assert_tree_integrity()

    def flatten_and_balance(self, depth: int) -> None:
        """
        Modifies the program so all tree branches have the same depth
        :param depth: Target depth of the program
        :return:
        """
        i = 0
        while i < len(self):
            # only used by type checker
            sub_program = cast(Loop, self[i])

            if sub_program.depth() < depth - 1:
                sub_program.encapsulate()

            elif not sub_program.is_balanced():
                sub_program.flatten_and_balance(depth - 1)

            elif sub_program.depth() == depth - 1:
                i += 1

            elif len(sub_program) == 1 and len(sub_program[0]) == 1:
                sub_sub_program = cast(Loop, sub_program[0])

                sub_program.repetition_count = sub_program.repetition_count * sub_sub_program.repetition_count
                sub_program[:] = sub_sub_program[:]
                sub_program.waveform = sub_sub_program.waveform

            elif not sub_program.is_leaf():
                sub_program.unroll()

            else:
                # we land in this case if the function gets called with depth == 0 and the current subprogram is a leaf
                i += 1

    def remove_empty_loops(self):
        new_children = []
        for child in self:
            if child.is_leaf():
                if child.waveform is None:
                    if child._measurements:
                        warnings.warn("Dropping measurement since there is no waveform attached")
                else:
                    new_children.append(child)
            else:
                child.remove_empty_loops()
                if not child.is_leaf():
                    new_children.append(child)
                else:
                    # all children of child were empty
                    pass
        self[:] = new_children

    def cleanup(self):
        """Remove empty loops and merge nested loops with single child"""
        new_children = []
        for child in self:
            if child.is_leaf():
                if child.waveform is None:
                    if child._measurements:
                        warnings.warn("Dropping measurement since there is no waveform attached")
                else:
                    new_children.append(child)

            else:
                child.cleanup()
                if child.waveform or not child.is_leaf():
                    new_children.append(child)

                elif child._measurements:
                    warnings.warn("Dropping measurement since there is no waveform in children")

        if len(new_children) == 1 and not self._measurements:
            assert not self._waveform
            only_child = new_children[0]

            self._measurements = only_child._measurements
            self.waveform = only_child.waveform
            self.repetition_count = self.repetition_count * only_child.repetition_count
            self[:] = only_child[:]

        elif len(self) != len(new_children):
            self[:] = new_children


class ChannelSplit(Exception):
    def __init__(self, channel_sets):
        self.channel_sets = channel_sets


class MultiChannelProgram:
    def __init__(self, instruction_block: Union[AbstractInstructionBlock, Loop], channels: Iterable[ChannelID] = None):
        """Channels with identifier None are ignored."""
        self._programs = dict()
        if isinstance(instruction_block, AbstractInstructionBlock):
            self._init_from_instruction_block(instruction_block, channels)
        elif isinstance(instruction_block, Loop):
            assert channels is None
            self._init_from_loop(loop=instruction_block)
        else:
            raise TypeError('Invalid program type', type(instruction_block), instruction_block)

        for program in self.programs.values():
            program.cleanup()

    def _init_from_loop(self, loop: Loop):
        first_waveform = next(loop.get_depth_first_iterator()).waveform

        assert first_waveform is not None

        self._programs[frozenset(first_waveform.defined_channels)] = loop

    def _init_from_instruction_block(self, instruction_block, channels):
        if channels is None:
            def find_defined_channels(instruction_list):
                for instruction in instruction_list:
                    if isinstance(instruction, EXECInstruction):
                        yield instruction.waveform.defined_channels
                    elif isinstance(instruction, REPJInstruction):
                        yield from find_defined_channels(
                            instruction.target.block.instructions[instruction.target.offset:])
                    elif isinstance(instruction, GOTOInstruction):
                        yield from find_defined_channels(instruction.target.block.instructions[instruction.target.offset:])
                    elif isinstance(instruction, CHANInstruction):
                        yield itertools.chain(*instruction.channel_to_instruction_block.keys())
                    elif isinstance(instruction, STOPInstruction):
                        return
                    elif isinstance(instruction, MEASInstruction):
                        pass
                    else:
                        raise TypeError('Unhandled instruction type', type(instruction))

            try:
                channels = next(find_defined_channels(instruction_block.instructions))
            except StopIteration:
                raise ValueError('Instruction block has no defined channels')
        else:
            channels = set(channels)

        channels = frozenset(channels - {None})

        root = Loop()
        stacks = {channels: (root, [((), deque(instruction_block.instructions))])}

        while len(stacks) > 0:
            chans, (root_loop, stack) = stacks.popitem()
            try:
                self._programs[chans] = MultiChannelProgram.__split_channels(chans, root_loop, stack)
            except ChannelSplit as split:
                for new_channel_set in split.channel_sets:
                    assert (new_channel_set not in stacks)
                    assert (chans.issuperset(new_channel_set))

                    stacks[new_channel_set] = (root_loop.copy_tree_structure(), deepcopy(stack))

        def repeat_measurements(child_loop, rep_count):
            duration_float = float(child_loop.duration)
            if child_loop._measurements:
                for r in range(rep_count):
                    for name, begin, length in child_loop._measurements:
                        yield (name, begin+r*duration_float, length)

    @property
    def programs(self) -> Dict[FrozenSet[ChannelID], Loop]:
        return self._programs

    @property
    def channels(self) -> Set[ChannelID]:
        return set(itertools.chain(*self._programs.keys()))

    @staticmethod
    def __split_channels(channels: FrozenSet[ChannelID],
                         root_loop: Loop,
                         block_stack: List[Tuple[Tuple[int, ...],
                                                 deque]]) -> Loop:
        while block_stack:
            current_loop_location, current_instruction_block = block_stack.pop()
            current_loop = root_loop.locate(current_loop_location)

            while current_instruction_block:
                instruction = current_instruction_block.popleft()

                if isinstance(instruction, EXECInstruction):
                    if not instruction.waveform.defined_channels.issuperset(channels):
                        raise Exception(instruction.waveform.defined_channels, channels)
                    current_loop.append_child(waveform=instruction.waveform)

                elif isinstance(instruction, REPJInstruction):
                    if current_instruction_block:
                        block_stack.append((current_loop_location, current_instruction_block))

                    current_loop.append_child(repetition_count=instruction.count)
                    block_stack.append(
                        (current_loop[-1].get_location(),
                         deque(instruction.target.block[instruction.target.offset:-1]))
                    )
                    break

                elif isinstance(instruction, CHANInstruction):
                    if channels in instruction.channel_to_instruction_block.keys():
                        # push to front
                        new_instruction_ptr = instruction.channel_to_instruction_block[channels]
                        new_instruction_list = [*new_instruction_ptr.block[new_instruction_ptr.offset:-1]]
                        current_instruction_block.extendleft(new_instruction_list)

                    else:
                        block_stack.append((current_loop_location, deque([instruction]) + current_instruction_block))

                        raise ChannelSplit(instruction.channel_to_instruction_block.keys())

                elif isinstance(instruction, MEASInstruction):
                    current_loop.add_measurements(instruction.measurements)

                else:
                    raise Exception('Encountered unhandled instruction {} on channel(s) {}'.format(instruction, channels))
        return root_loop

    def __getitem__(self, item: Union[ChannelID, Set[ChannelID], FrozenSet[ChannelID]]) -> Loop:
        if not isinstance(item, (set, frozenset)):
            item = frozenset((item,))
        elif isinstance(item, set):
            item = frozenset(item)

        for channels, program in self._programs.items():
            if item.issubset(channels):
                return program
        raise KeyError(item)


def to_waveform(program: Loop) -> Waveform:
    if program.is_leaf():
        if program.repetition_count == 1:
            return program.waveform
        else:
            return RepetitionWaveform(program.waveform, program.repetition_count)
    else:
        if len(program) == 1:
            sequenced_waveform = to_waveform(cast(Loop, program[0]))
        else:
            sequenced_waveform = SequenceWaveform([to_waveform(cast(Loop, sub_program))
                                                   for sub_program in program])
        if program.repetition_count > 1:
            return RepetitionWaveform(sequenced_waveform, program.repetition_count)
        else:
            return sequenced_waveform


class _CompatibilityLevel(Enum):
    compatible = 0
    action_required = 1
    incompatible = 2


def _is_compatible(program: Loop, min_len: int, quantum: int, sample_rate: TimeType) -> _CompatibilityLevel:
    program_duration_in_samples = program.duration * sample_rate

    if program_duration_in_samples.denominator != 1:
        return _CompatibilityLevel.incompatible

    if program_duration_in_samples < min_len or program_duration_in_samples % quantum > 0:
        return _CompatibilityLevel.incompatible

    if program.is_leaf():
        waveform_duration_in_samples = program.body_duration * sample_rate
        if waveform_duration_in_samples < min_len or (waveform_duration_in_samples / quantum).denominator != 1:
            return _CompatibilityLevel.action_required
        else:
            return _CompatibilityLevel.compatible
    else:
        if all(_is_compatible(cast(Loop, sub_program), min_len, quantum, sample_rate) == _CompatibilityLevel.compatible
               for sub_program in program):
            return _CompatibilityLevel.compatible
        else:
            return _CompatibilityLevel.action_required


def _make_compatible(program: Loop, min_len: int, quantum: int, sample_rate: TimeType) -> None:

    if program.is_leaf():
        program.waveform = to_waveform(program.copy_tree_structure())
        program.repetition_count = 1

    else:
        comp_levels = np.array([_is_compatible(cast(Loop, sub_program), min_len, quantum, sample_rate)
                                for sub_program in program])
        incompatible = comp_levels == _CompatibilityLevel.incompatible
        if np.any(incompatible):
            single_run = program.duration * sample_rate / program.repetition_count
            if is_integer(single_run / quantum) and single_run >= min_len:
                new_repetition_count = program.repetition_count
                program.repetition_count = 1
            else:
                new_repetition_count = 1
            program.waveform = to_waveform(program.copy_tree_structure())
            program.repetition_count = new_repetition_count
            program[:] = []
            return
        else:
            for sub_program, comp_level in zip(program, comp_levels):
                if comp_level == _CompatibilityLevel.action_required:
                    _make_compatible(sub_program, min_len, quantum, sample_rate)


def make_compatible(program: Loop, minimal_waveform_length: int, waveform_quantum: int, sample_rate: TimeType):
    comp_level = _is_compatible(program,
                                min_len=minimal_waveform_length,
                                quantum=waveform_quantum,
                                sample_rate=sample_rate)
    if comp_level == _CompatibilityLevel.incompatible:
        raise ValueError('The program cannot be made compatible to restrictions')
    elif comp_level == _CompatibilityLevel.action_required:
        _make_compatible(program,
                         min_len=minimal_waveform_length,
                         quantum=waveform_quantum,
                         sample_rate=sample_rate)
