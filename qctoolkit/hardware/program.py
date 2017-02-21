import itertools
from typing import Union, Dict, Set, Iterable, FrozenSet, List, NamedTuple, Any, Callable, Tuple
from collections import deque
from copy import deepcopy
from ctypes import c_int64 as MutableInt

from qctoolkit import MeasurementWindow, ChannelID
from qctoolkit.pulses.instructions import AbstractInstructionBlock, EXECInstruction, REPJInstruction, GOTOInstruction, STOPInstruction, InstructionPointer, CHANInstruction, Waveform
from qctoolkit.comparable import Comparable
from qctoolkit.utils.tree import Node, is_tree_circular


__all__ = ['Loop', 'MultiChannelProgram', '']


class Loop(Comparable, Node):
    """Build a loop tree. The leaves of the tree are loops with one element."""
    def __init__(self,
                 parent: Union['Loop', None]=None,
                 children: Iterable['Loop']=list(),
                 waveform: Union[EXECInstruction, Waveform]=None,
                 repetition_count=1):
        super().__init__(parent=parent, children=children)

        self._waveform = waveform
        self._repetition_count = repetition_count

        if not isinstance(waveform, (type(None), Waveform)):
            raise Exception()

    @property
    def compare_key(self) -> Tuple:
        return self._waveform, self._repetition_count, tuple(c.compare_key for c in self)

    def append_child(self, **kwargs) -> None:
        self[len(self):len(self)] = (kwargs, )

    @property
    def waveform(self) -> Waveform:
        return self._waveform

    @waveform.setter
    def waveform(self, val) -> None:
        self._waveform = val

    @property
    def repetition_count(self) -> int:
        return self._repetition_count

    @repetition_count.setter
    def repetition_count(self, val) -> None:
        self._repetition_count = val

    def unroll(self) -> None:
        for i, e in enumerate(self.parent):
            if id(e) == id(self):
                self.parent[i:i+1] = (child.copy_tree_structure(new_parent=self.parent)
                                      for _ in range(self.repetition_count)
                                      for child in self)
                self.parent.assert_tree_integrity()
                return
        raise Exception('self not found in parent')

    def unroll_children(self) -> None:
        old_children = self.children
        self[:] = (child.copy_tree_structure()
                   for _ in range(self.repetition_count)
                   for child in old_children)
        self._repetition_count = 1
        self.assert_tree_integrity()

    def encapsulate(self) -> None:
        self[:] = [Loop(children=self.children,
                        repetition_count=self._repetition_count,
                        waveform=self._waveform)]
        self._repetition_count = 1
        self._waveform = None
        self.assert_tree_integrity()

    def __repr__(self) -> str:
        is_circular = is_tree_circular(self)
        if is_circular:
            return '{}: Circ {}'.format(id(self), is_circular)

        if self.is_leaf():
            return 'EXEC {} {} times'.format(self._waveform, self._repetition_count)
        else:
            repr = ['LOOP {} times:'.format(self._repetition_count)]
            for elem in self:
                sub_repr = elem.__repr__().splitlines()
                sub_repr = ['  ->' + sub_repr[0]] + ['    ' + line for line in sub_repr[1:]]
                repr += sub_repr
            return '\n'.join(repr)

    def copy_tree_structure(self, new_parent: Union['Loop', bool]=False) -> 'Loop':
        return type(self)(parent=self.parent if new_parent is False else new_parent,
                          waveform=self._waveform,
                          repetition_count=self.repetition_count,
                          children=(child.copy_tree_structure() for child in self))

    def get_measurement_windows(self, offset: MutableInt = MutableInt(0), measurement_windows=dict()) -> Dict[str,
                                                                                                              deque]:
        if self.is_leaf():
            for _ in range(self.repetition_count):
                for (mw_name, begin, length) in self.waveform.get_measurement_windows():
                    measurement_windows.get(mw_name, default=deque()).append((begin + offset.value, length))
                offset.value += self.waveform.duration
        else:
            for _ in range(self.repetition_count):
                for child in self:
                    child.get_measurement_windows(offset, measurement_windows=measurement_windows)
        return measurement_windows

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


class ChannelSplit(Exception):
    def __init__(self, channel_sets):
        self.channel_sets = channel_sets


class MultiChannelProgram:
    def __init__(self, instruction_block: AbstractInstructionBlock, channels: Iterable[ChannelID] = None):
        if channels is None:
            def find_defined_channels(instruction_list):
                for instruction in instruction_list:
                    if isinstance(instruction, EXECInstruction):
                        return instruction.waveform.defined_channels
                    elif isinstance(instruction, REPJInstruction):
                        for _ in range(instruction.count):
                            return find_defined_channels(
                                instruction.target.block.instructions[instruction.target.offset:])
                    elif isinstance(instruction, GOTOInstruction):
                        return find_defined_channels(instruction.target.block.instructions[instruction.target.offset:])
                    elif isinstance(instruction, CHANInstruction):
                        return itertools.chain(*instruction.channel_to_instruction_block.keys())
                    elif isinstance(instruction, STOPInstruction):
                        break
                    else:
                        raise TypeError('Unhandled instruction type', type(instruction))
                raise ValueError('Instruction block has no defined channels')

            channels = find_defined_channels(instruction_block.instructions)

        channels = frozenset(channels)

        root = Loop()
        stacks = {channels: (root, [([], [*instruction_block[:-1]])])}
        self.__programs = dict()

        while len(stacks) > 0:
            chans, (root_loop, stack) = stacks.popitem()
            try:
                self.__programs[chans] = MultiChannelProgram.__split_channels(chans, root_loop, stack)
            except ChannelSplit as split:
                for new_channel_set in split.channel_sets:
                    assert (new_channel_set not in stacks)
                    assert (chans.issuperset(new_channel_set))

                    stacks[new_channel_set] = (root_loop.copy_tree_structure(), deepcopy(stack))

        for channels, program in self.__programs.items():
            iterable = program.get_breadth_first_iterator()
            while True:
                try:
                    loop = next(iterable)
                    if len(loop) == 1:
                        loop.waveform = loop[0].waveform
                        loop.repetition_count = loop.repetition_count * loop[0].repetition_count
                        loop[:] = loop[0][:]

                        iterable = itertools.chain((loop,), iterable)
                except StopIteration:
                    break

    @property
    def programs(self) -> Dict[FrozenSet[ChannelID], Loop]:
        return self.__programs

    @property
    def channels(self) -> Set[ChannelID]:
        return set(itertools.chain(*self.__programs.keys()))

    @staticmethod
    def __split_channels(channels, root_loop, block_stack) -> Loop:
        while block_stack:
            current_loop_location, current_instruction_block = block_stack.pop()
            current_loop = root_loop.locate(current_loop_location)

            while current_instruction_block:
                instruction = current_instruction_block.pop(0)

                if isinstance(instruction, EXECInstruction):
                    if not instruction.waveform.defined_channels.issuperset(channels):
                        raise Exception(instruction.waveform.defined_channels, channels)
                    current_loop.append_child(waveform=instruction.waveform)

                elif isinstance(instruction, REPJInstruction):
                    current_loop.append_child(repetition_count=instruction.count)
                    block_stack.append(
                        (current_loop[-1].get_location(),
                         [*instruction.target.block[instruction.target.offset:-1]])
                    )

                elif isinstance(instruction, CHANInstruction):
                    if channels in instruction.channel_to_instruction_block.keys():
                        # push to front
                        new_instruction_ptr = instruction.channel_to_instruction_block[channels]
                        new_instruction_list = [*new_instruction_ptr.block[new_instruction_ptr.offset:-1]]
                        current_instruction_block[0:0] = new_instruction_list

                    else:
                        block_stack.append((current_loop_location, [instruction] + current_instruction_block))

                        raise ChannelSplit(instruction.channel_to_instruction_block.keys())

                else:
                    raise Exception('Encountered unhandled instruction {} on channel(s) {}'.format(instruction, channels))
        return root_loop

    def __getitem__(self, item: Union[ChannelID, Set[ChannelID], FrozenSet[ChannelID]]) -> Loop:
        if not isinstance(item, (set, frozenset)):
            item = frozenset((item,))
        elif isinstance(item, set):
            item = frozenset(item)

        for channels, program in self.__programs.items():
            if item.issubset(channels):
                return program
        raise KeyError(item)
