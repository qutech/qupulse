from qctoolkit.pulses.instructions import AbstractInstructionBlock, InstructionBlock, EXECInstruction, REPJInstruction, GOTOInstruction, STOPInstruction, InstructionPointer, CHANInstruction
from typing import Union, Dict, Set, Iterable, FrozenSet, List, NamedTuple, Any, Callable
from qctoolkit.hardware.awgs import AWG
from qctoolkit.comparable import Comparable

from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform
import itertools
from collections import namedtuple
from copy import deepcopy, copy as shallowcopy


ChannelID = int

__all__ = ['ChannelID', 'Loop']


class Loop(Comparable):
    """Build a loop tree. The leaves of the tree are loops with one element."""
    def __init__(self, parent=None, instruction=None, children=list(), repetition_count=1):
        super().__init__()

        self.__parent = parent
        self.__children = [self.__parse_child(child) for child in children]
        self.__instruction = instruction
        self.__repetition_count = repetition_count

        if not (instruction is None or isinstance(instruction, (EXECInstruction, MultiChannelWaveform))):
            raise Exception()

    def unroll(self):
        for i, e in enumerate(self.__parent):
            if e is self:
                self.__parent[i:i+1] = (child.copy_tree_structure(new_parent=self.__parent)
                                        for _ in range(self.repetition_count)
                                        for child in self.children)
                self.__parent.assert_tree_integrity()
                self.__parent = None
                return
        raise Exception('self not found in parent')

    def unroll_children(self):
        self.__children = [Loop(parent=self, instruction=child.instruction, children=child.children)
                           for _ in range(self.repetition_count)
                           for child in self.children]
        self.__repetition_count = 1
        self.assert_tree_integrity()

    def encapsulate(self):
        self.__children = [Loop(children=self.__children,
                                parent=self,
                                repetition_count=self.__repetition_count,
                                instruction=self.__instruction)]
        self.__repetition_count = 1
        self.__instruction = None
        self.assert_tree_integrity()

    def split_one_child(self):
        self.assert_tree_integrity()
        for i in reversed(range(len(self))):
            if self[i].repetition_count > 1:
                self[i].repetition_count -= 1
                self[i+1:i+1] = (self[i].copy_tree_structure(),)
                self[i+1].repetition_count = 1
                self.assert_tree_integrity()
                return
        raise Exception('Could not split of one child', self)

    def is_leaf(self) -> bool:
        return len(self.__children) == 0

    def depth(self) -> int:
        return 0 if self.is_leaf() else (1 + max((e.depth() for e in self)))

    def is_balanced(self) -> bool:
        if self.is_leaf():
            return True
        return all((e.depth() == self.__children[0].depth() and e.is_balanced()) for e in self)

    def merge(self):
        """Merge successive loops that are repeated once and are no leafs into one"""
        raise Exception("Not tested.")
        if self.depth() < 2:
            return
        # TODO: make this pythonic
        i = 0
        while i < len(self.__children):
            if self.__children[i].repetition_count == 1 and not self.__children[i].is_leaf():
                j = i + 1
                while j < len(self.__children) and self.__children[j].repetition_count == 1 and not self.__children[j].is_leaf():
                    j += 1
                if j > i + 1:
                    self.__children[i:j] = Loop(parent=self,
                                                children=[cc for child in self.__children[i:j] for cc in child],
                                                repetition_count=1)
            i += 1
        self.assert_tree_integrity()

    def compare_key(self):
        return self.__instruction, self.__repetition_count, tuple(c.compare_key() for c in self.__children)

    @property
    def children(self) -> List['Loop']:
        """
        :return: shallow copy of children
        """
        return shallowcopy(self.__children)

    def append_child(self, **kwargs):
        self.__children.append(Loop(parent=self, **kwargs))
        self.assert_tree_integrity()

    def __check_circular(self, visited: List['Loop']):
        for v in visited:
            if self is v:
                raise Exception(self, visited)
        visited.append(self)
        for c in self.__children:
            c.__check_circular(shallowcopy(visited))

    def check_circular(self):
        self.__check_circular([])

    @property
    def instruction(self):
        return self.__instruction

    @instruction.setter
    def instruction(self, val):
        self.__instruction = val

    @property
    def repetition_count(self):
        return self.__repetition_count

    @repetition_count.setter
    def repetition_count(self, val):
        self.__repetition_count = val

    def get_root(self):
        if self.__parent:
            return self.__parent.get_root()
        else:
            return self

    def get_depth_first_iterator(self):
        if not self.is_leaf():
            for e in self.__children:
                yield from e.get_depth_first_iterator()
        yield self

    def get_breadth_first_iterator(self, queue: List['Loop']=[]):
        yield self
        if not self.is_leaf():
            queue += self.__children
        if queue:
            yield from queue.pop(0).get_breadth_first_iterator(queue)

    def __iter__(self) -> Iterable['Loop']:
        return iter(self.children)

    def __setitem__(self, idx, value):
        if isinstance(idx, slice):
            if isinstance(value, Loop):
                raise TypeError('can only assign an iterable (Loop does not count)')
            value = (self.__parse_child(child) for child in value)
        else:
            value = self.__parse_child(value)
        self.__children.__setitem__(idx, value)

    def __getitem__(self, *args, **kwargs) ->Union['Loop', List['Loop']]:
        return self.__children.__getitem__(*args, **kwargs)

    def __len__(self):
        return len(self.__children)

    def __repr__(self):
        try:
            self.check_circular()
        except Exception as e:
            if len(e.args) == 2:
                return '{}: Circ {}'.format(id(self), len(e.args[1]))

        if self.is_leaf():
            return 'EXEC {} {} times'.format(self.__instruction, self.__repetition_count)
        else:
            repr = ['LOOP {} times:'.format(self.__repetition_count)]
            for elem in self.__children:
                sub_repr = elem.__repr__().splitlines()
                sub_repr = ['  ->' + sub_repr[0]] + ['    ' + line for line in sub_repr[1:]]
                repr += sub_repr
            return '\n'.join(repr)

    def __parse_child(self, child):
        if isinstance(child, dict):
            return Loop(parent=self, **child)
        elif isinstance(child, Loop):
            child.__parent = self
            return child
        else:
            raise TypeError('Invalid child type', type(child))

    def assert_tree_integrity(self):
        if self.__parent:
            children_ids = [id(c) for c in self.__parent.children]
            if id(self) not in children_ids:
                raise Exception()
        for child in self.__children:
            child.assert_tree_integrity()

    def copy_tree_structure(self, new_parent: Union['Loop', bool]=False):
        return type(self)(parent=self.__parent if new_parent is not False else new_parent,
                          instruction=self.__instruction,
                          repetition_count=self.repetition_count,
                          children=(child.copy_tree_structure() for child in self.__children))


class ChannelSplit(Exception):
    def __init__(self, channels_and_blocks):
        self.channels_and_stacks = channels_and_blocks


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

        stacks = {channels: [(Loop(), [*instruction_block[:-1]])]}
        self.__programs = dict()

        while len(stacks) > 0:
            chans, stack = stacks.popitem()
            try:
                self.__programs[chans] = MultiChannelProgram.__split_channels(stack, chans)
            except ChannelSplit as c:
                for new_chans, new_stack in c.channels_and_stacks.items():
                    assert (new_chans not in stacks)
                    assert (chans.issuperset(new_chans))
                    stacks[new_chans] = new_stack

        for channels, program in self.__programs.items():
            iterable = program.get_breadth_first_iterator()
            while True:
                try:
                    loop = next(iterable)
                    if len(loop) == 1:
                        loop.instruction = loop[0].instruction
                        loop.repetition_count = loop.repetition_count * loop[0].repetition_count
                        loop[:] = loop[0][:]

                        iterable = itertools.chain((loop,), iterable)
                except StopIteration:
                    break

            for loop in program.get_breadth_first_iterator():
                loop.assert_tree_integrity()

    @property
    def programs(self):
        return self.__programs

    @property
    def channels(self):
        return set(itertools.chain(*self.__programs.keys()))

    @staticmethod
    def __split_channels(block_stack, channels):
        while block_stack:
            current_loop, current_instruction_block = block_stack.pop()
            while current_instruction_block:
                instruction = current_instruction_block.pop(0)
                if isinstance(instruction, EXECInstruction):
                    if not instruction.waveform.defined_channels.issuperset(channels):
                        raise Exception(instruction.waveform.defined_channels, channels)
                    current_loop.append_child(instruction=instruction)

                elif isinstance(instruction, REPJInstruction):
                    current_loop.append_child(repetition_count=instruction.count)
                    block_stack.append(
                        (current_loop.children[-1],
                         [*instruction.target.block[instruction.target.offset:-1]])
                    )

                elif isinstance(instruction, CHANInstruction):
                    if channels in instruction.channel_to_instruction_block.keys():
                        # push to front
                        new_instruction_ptr = instruction.channel_to_instruction_block[channels]
                        new_instruction_list = [*new_instruction_ptr.block[new_instruction_ptr.offset:-1]]
                        current_instruction_block[0:0] = new_instruction_list

                    else:
                        block_stack.append((current_loop, current_instruction_block))

                        channel_to_stack = dict()
                        for (chs, instruction_ptr) in instruction.channel_to_instruction_block.items():
                            channel_to_stack[chs] = deepcopy(block_stack)
                            channel_to_stack[chs][-1][1][0:0] = [*instruction_ptr.block[instruction_ptr.offset:-1]]
                        raise ChannelSplit(channel_to_stack)
                else:
                    raise Exception('Encountered unhandled instruction {} on channel(s) {}'.format(instruction, channels))
        return current_loop.get_root()

    def __getitem__(self, item: Union[ChannelID, Set[ChannelID], FrozenSet[ChannelID]]):
        if not isinstance(item, (set, frozenset)):
            item = frozenset((item,))
        elif isinstance(item, set):
            item = frozenset(item)

        for channels, program in self.__programs.items():
            if item.issubset(channels):
                return program
        raise KeyError(item)
