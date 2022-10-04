from typing import Union, Dict, Iterable, Tuple, cast, List, Optional, Generator, Mapping
from collections import defaultdict
from enum import Enum
import warnings
import bisect

import numpy as np
import sympy.ntheory

from qupulse._program.waveforms import Waveform, ConstantWaveform
from qupulse._program.volatile import VolatileRepetitionCount, VolatileProperty

from qupulse.utils import is_integer
from qupulse.utils.types import TimeType, MeasurementWindow
from qupulse.utils.tree import Node, is_tree_circular
from qupulse.utils.numeric import smallest_factor_ge

from qupulse._program.waveforms import SequenceWaveform, RepetitionWaveform

__all__ = ['Loop', 'make_compatible', 'MakeCompatibleWarning']


class Loop(Node):
    MAX_REPR_SIZE = 2000
    __slots__ = ('_waveform', '_measurements', '_repetition_definition', '_cached_body_duration')

    """This class represents a initialized (sub-)program as a tree. Each Loop of a valid program has a repetition count
    and either a waveform or a sequence of loops as children.
    
    A Loop can have associated measurements which are also repeated.
    """
    def __init__(self,
                 parent: Union['Loop', None] = None,
                 children: Iterable['Loop'] = (),
                 waveform: Optional[Waveform] = None,
                 measurements: Optional[List[MeasurementWindow]] = None,
                 repetition_count: Union[int, VolatileRepetitionCount] = 1):
        """Initialize a new loop

        Args:
            parent: Forwarded to Node.__init__
            children: Forwarded to Node.__init__
            waveform: "Payload"
            measurements: Associated measurements
            repetition_count: The children / waveform are repeated this often
        """
        super().__init__(parent=parent, children=children)

        self._waveform = waveform
        self._measurements = measurements
        self._repetition_definition = repetition_count
        self._cached_body_duration = None
        assert isinstance(repetition_count, VolatileRepetitionCount) or is_integer(repetition_count)
        assert isinstance(waveform, (type(None), Waveform))

    def __eq__(self, other: 'Loop') -> bool:
        if type(self) == type(other):
            return (self._repetition_definition == other._repetition_definition and
                    self.waveform == other.waveform and
                    (self._measurements or None) == (other._measurements or None) and
                    len(self) == len(other) and
                    all(self_child == other_child for self_child, other_child in zip(self, other)))
        else:
            return NotImplemented

    def append_child(self, loop: Optional['Loop'] = None, **kwargs) -> None:
        """Append a child to this loop. Either an existing Loop object or a newly created from kwargs

        Args:
            loop: loop to append
            **kwargs: Child is constructed with these kwargs

        Raises:
            ValueError: if called with loop and kwargs
        """
        if loop is not None:
            if kwargs:
                raise ValueError("Cannot pass a Loop object and Loop constructor arguments at the same time in "
                                 "append_child")
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

    def add_measurements(self, measurements: Iterable[MeasurementWindow]):
        """Add measurements offset by the current body duration i.e. to the END of the current loop

        Args:
            measurements: Measurements to add
        """
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
                    self._cached_body_duration = TimeType.from_fraction(0, 1)
            else:
                self._cached_body_duration = sum(child.duration for child in self)
        return self._cached_body_duration

    @property
    def duration(self) -> TimeType:
        return self.body_duration * TimeType.from_fraction(self.repetition_count, 1)

    @property
    def volatile_repetition(self) -> Optional[VolatileProperty]:
        return getattr(self._repetition_definition, 'volatile_property', None)

    @property
    def repetition_definition(self) -> Union[int, VolatileRepetitionCount]:
        return self._repetition_definition

    @repetition_definition.setter
    def repetition_definition(self, new_definition: Union[int, VolatileRepetitionCount]):
        self._repetition_definition = new_definition

    @property
    def repetition_count(self) -> int:
        return int(self._repetition_definition)

    @repetition_count.setter
    def repetition_count(self, val: int) -> None:
        assert isinstance(val, (int, float))
        new_repetition = int(val)
        if abs(new_repetition - val) > 1e-10:
            raise ValueError('Repetition count was not an integer')
        self._repetition_definition = new_repetition

    def unroll(self) -> None:
        if self.is_leaf():
            raise RuntimeError('Leaves cannot be unrolled')
        if self.volatile_repetition:
            warnings.warn("Unrolling a Loop with volatile repetition count", VolatileModificationWarning)

        i = self.parent_index
        self.parent[i:i+1] = (child.copy_tree_structure(new_parent=self.parent)
                              for _ in range(self.repetition_count)
                              for child in self)
        self.parent.assert_tree_integrity()

    def __setitem__(self, idx, value):
        super().__setitem__(idx, value)
        self._invalidate_duration()

    def unroll_children(self) -> None:
        if self.volatile_repetition:
            warnings.warn("Unrolling a Loop with volatile repetition count", VolatileModificationWarning)
        old_children = self.children
        self[:] = (child.copy_tree_structure()
                   for _ in range(self.repetition_count)
                   for child in old_children)
        self.repetition_count = 1
        self.assert_tree_integrity()

    def encapsulate(self) -> None:
        """Add a nesting level by moving self to its children."""
        self[:] = [Loop(children=self,
                        repetition_count=self._repetition_definition,
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
                          repetition_count=self._repetition_definition,
                          measurements=None if self._measurements is None else list(self._measurements),
                          children=(child.copy_tree_structure() for child in self))

    def _get_measurement_windows(self) -> Mapping[str, np.ndarray]:
        """Private implementation of get_measurement_windows with a slightly different data format for easier tiling.

        Returns:
             A dictionary (measurement_name -> array) with begin == array[:, 0] and length == array[:, 1]
        """
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

        # this gives us regular dict behaviour of the returned object
        temp_meas_windows.default_factory = None

        # repeat and add repetition based offset
        for mw_name, begin_length_list in temp_meas_windows.items():
            temp_begin_length_array = np.concatenate(begin_length_list)

            begin_length_array = np.tile(temp_begin_length_array, (self.repetition_count, 1))

            shaped_begin_length_array = np.reshape(begin_length_array, (self.repetition_count, -1, 2))

            shaped_begin_length_array[:, :, 0] += (np.arange(self.repetition_count) * body_duration)[:, np.newaxis]

            temp_meas_windows[mw_name] = begin_length_array

        # the cast is here because static type analysis struggles to detect that we replace _all_ values by ndarray in
        # the previous loop
        return cast(Mapping[str, np.ndarray], temp_meas_windows)

    def get_measurement_windows(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Iterates over all children and collect the begin and length arrays of each measurement window.

        Returns:
            A dictionary (measurement_name -> (begin, length)) with begin and length being :class:`numpy.ndarray`
        """
        return {mw_name: (begin_length_list[:, 0], begin_length_list[:, 1])
                for mw_name, begin_length_list in self._get_measurement_windows().items()}

    def split_one_child(self, child_index=None) -> None:
        """Take the last child that has a repetition count larger one, decrease it's repetition count and insert a copy
        with repetition cout one after it"""
        if child_index is not None:
            if self[child_index].repetition_count < 2:
                raise ValueError('Cannot split child {} as the repetition count is not larger 1')

        else:
            # we cannot reverse enumerate
            n_child = len(self) - 1
            for reverse_idx, child in enumerate(reversed(self)):
                if child.repetition_count > 1:
                    forward_idx = n_child - reverse_idx
                    if not child.volatile_repetition:
                        child_index = forward_idx
                        break
                    elif child_index is None:
                        child_index = forward_idx
            else:
                if child_index is None:
                    raise RuntimeError('There is no child with repetition count > 1')

        if self[child_index].volatile_repetition:
            warnings.warn("Splitting a child with volatile repetition count", VolatileModificationWarning)

        new_child = self[child_index].copy_tree_structure()
        new_child.repetition_count = 1

        self[child_index].repetition_count -= 1

        self[child_index+1:child_index+1] = (new_child,)
        self.assert_tree_integrity()

    def flatten_and_balance(self, depth: int) -> None:
        """Modifies the program so all tree branches have the same depth.

        Args:
            depth: Target depth of the program
        """
        i = 0
        while i < len(self):
            # only used by type checker
            sub_program = cast(Loop, self[i])

            if sub_program.depth() < depth - 1:
                # increase nesting because the subprogram is not deep enough
                sub_program.encapsulate()

            elif not sub_program.is_balanced():
                # balance the sub program. We revisit it in the next iteration (no change of i )
                # because it might modify self. While writing this comment I am not sure this is true. 14.01.2020 Simon
                sub_program.flatten_and_balance(depth - 1)

            elif sub_program.depth() == depth - 1:
                # subprogram is balanced with the correct depth
                i += 1

            elif sub_program._has_single_child_that_can_be_merged():
                # subprogram is balanced but to deep and has no measurements -> we can "lift" the sub-sub-program
                # TODO: There was a len(sub_sub_program) == 1 check here that I cannot explain
                sub_program._merge_single_child()

            elif not sub_program.is_leaf():
                # subprogram is balanced but too deep
                sub_program.unroll()

            else:
                # we land in this case if the function gets called with depth == 0 and the current subprogram is a leaf
                i += 1

    def _has_single_child_that_can_be_merged(self) -> bool:
        if len(self) == 1:
            child = cast(Loop, self[0])
            return not self._measurements or (child.repetition_count == 1 and not child.volatile_repetition)
        else:
            return False

    def _merge_single_child(self):
        """Lift the single child to current level. Requires _has_single_child_that_can_be_merged to be true"""
        assert len(self) == 1, "bug: _merge_single_child called on loop with len != 1"
        child = cast(Loop, self[0])

        # if the child has a fixed repetition count of 1 the measurements can be merged
        mergable_measurements = child.repetition_count == 1 and not child.volatile_repetition

        assert not self._measurements or mergable_measurements, "bug: _merge_single_child called on loop with measurements"
        assert not self._waveform, "bug: _merge_single_child called on loop with children and waveform"

        measurements = child._measurements
        if self._measurements:
            if measurements:
                measurements.extend(self._measurements)
            else:
                measurements = self._measurements

        if not self.volatile_repetition and not child.volatile_repetition:
            # simple integer multiplication
            repetition_definition = self.repetition_count * child.repetition_count
        elif not self.volatile_repetition:
            repetition_definition = child._repetition_definition * self.repetition_count
        elif not child.volatile_repetition:
            repetition_definition = self._repetition_definition * child.repetition_count
        else:
            # create a new expression that depends on both
            expression = 'parent_repetition_count * child_repetition_count'
            repetition_definition = VolatileRepetitionCount.operation(
                expression=expression,
                parent_repetition_count=self._repetition_definition,
                child_repetition_count=child._repetition_definition)

        self[:] = iter(child)
        self._waveform = child._waveform
        self._repetition_definition = repetition_definition
        self._measurements = measurements
        self._invalidate_duration()
        return True

    def cleanup(self, actions=('remove_empty_loops', 'merge_single_child')):
        """Apply the specified actions to cleanup the Loop.

        remove_empty_loops: Remove loops with no children and no waveform (a DroppedMeasurementWarning is issued)
        merge_single_child: see `_try_merge_single_child` documentation

        Warnings:
            DroppedMeasurementWarning: Likely a bug in qupulse. TODO: investigate whether there are usecases
        """
        if 'remove_empty_loops' in actions:
            new_children = []
            for child in self:
                child = cast(Loop, child)
                if child.is_leaf():
                    if child.waveform is None:
                        if child._measurements:
                            warnings.warn("Dropping measurement since there is no waveform attached",
                                          category=DroppedMeasurementWarning)
                    else:
                        new_children.append(child)

                else:
                    child.cleanup(actions)
                    if child.waveform or not child.is_leaf():
                        new_children.append(child)

                    elif child._measurements:
                        warnings.warn("Dropping measurement since there is no waveform in children",
                                      category=DroppedMeasurementWarning)

            if len(self) != len(new_children):
                self[:] = new_children

        else:
            # only do the recursive call
            for child in self:
                child.cleanup(actions)

        if 'merge_single_child' in actions and self._has_single_child_that_can_be_merged():
            self._merge_single_child()
    
    def get_duration_structure(self) -> Tuple[int, Union[TimeType, tuple]]:
        if self.is_leaf():
            return self.repetition_count, self.waveform.duration
        else:
            return self.repetition_count, tuple(child.get_duration_structure() for child in self)

    def reverse_inplace(self):
        if self.is_leaf():
            self._waveform = self._waveform.reversed()
        else:
            self._reverse_children()
            for child in self:
                child.reverse_inplace()
        if self._measurements:
            duration = self.duration
            self._measurements = [
                (name, duration - (begin + length), length)
                for name, begin, length in self._measurements
            ]


class ChannelSplit(Exception):
    def __init__(self, channel_sets):
        self.channel_sets = channel_sets


def to_waveform(program: Loop) -> Waveform:
    if program.is_leaf():
        if program.repetition_count == 1:
            return program.waveform
        else:
            return RepetitionWaveform.from_repetition_count(program.waveform, program.repetition_count)
    else:
        if len(program) == 1:
            sequenced_waveform = to_waveform(cast(Loop, program[0]))
        else:
            sequenced_waveform = SequenceWaveform.from_sequence(
                [to_waveform(cast(Loop, sub_program))
                 for sub_program in program])
        if program.repetition_count > 1:
            return RepetitionWaveform.from_repetition_count(sequenced_waveform, program.repetition_count)
        else:
            return sequenced_waveform


class _CompatibilityLevel(Enum):
    compatible = 0
    action_required = 1
    incompatible_too_short = 2
    incompatible_fraction = 3
    incompatible_quantum = 4

    def is_incompatible(self) -> bool:
        return self in (self.incompatible_fraction, self.incompatible_quantum, self.incompatible_too_short)


def _is_compatible(program: Loop, min_len: int, quantum: int, sample_rate: TimeType) -> _CompatibilityLevel:
    """ check whether program loop is compatible with awg requirements
        possible reasons for incompatibility:
            program shorter than minimum length
            program duration not an integer
            program duration not a multiple of quantum """
    program_duration_in_samples = program.duration * sample_rate

    if program_duration_in_samples.denominator != 1:
        return _CompatibilityLevel.incompatible_fraction

    if program_duration_in_samples < min_len:
        return _CompatibilityLevel.incompatible_too_short

    if program_duration_in_samples % quantum > 0:
        return _CompatibilityLevel.incompatible_quantum

    if program.is_leaf():
        waveform_duration_in_samples = program.body_duration * sample_rate
        if waveform_duration_in_samples < min_len or (waveform_duration_in_samples / quantum).denominator != 1:
            if program.volatile_repetition:
                warnings.warn("_is_compatible requires an action which drops volatility.",
                              category=VolatileModificationWarning)
            return _CompatibilityLevel.action_required
        else:
            return _CompatibilityLevel.compatible
    else:
        if all(_is_compatible(cast(Loop, sub_program), min_len, quantum, sample_rate) == _CompatibilityLevel.compatible
               for sub_program in program):
            return _CompatibilityLevel.compatible
        else:
            if program.volatile_repetition:
                warnings.warn("_is_compatible requires an action which drops volatility.",
                              category=VolatileModificationWarning)
            return _CompatibilityLevel.action_required


def _make_compatible(program: Loop, min_len: int, quantum: int, sample_rate: TimeType) -> None:
    if program.is_leaf():
        program.waveform = to_waveform(program.copy_tree_structure())
        program.repetition_count = 1
    else:
        comp_levels = [_is_compatible(cast(Loop, sub_program), min_len, quantum, sample_rate)
                       for sub_program in program]

        if any(comp_level.is_incompatible() for comp_level in comp_levels):
            single_run = program.duration * sample_rate / program.repetition_count
            if (single_run / quantum).denominator == 1 and single_run >= min_len:
                # it is enough to concatenate all children
                new_repetition_definition = program.repetition_definition
                program.repetition_count = 1
            else:
                # we need to concatenate all children and unroll
                new_repetition_definition = 1

            program.waveform = to_waveform(program.copy_tree_structure())
            program.repetition_definition = new_repetition_definition
            program[:] = []
            return
        else:
            for sub_program, comp_level in zip(program, comp_levels):
                if comp_level == _CompatibilityLevel.action_required:
                    _make_compatible(sub_program, min_len, quantum, sample_rate)


def make_compatible(program: Loop, minimal_waveform_length: int, waveform_quantum: int, sample_rate: TimeType):
    """ check program for compatibility to AWG requirements, make it compatible if necessary and  possible"""
    comp_level = _is_compatible(program,
                                min_len=minimal_waveform_length,
                                quantum=waveform_quantum,
                                sample_rate=sample_rate)
    if comp_level == _CompatibilityLevel.incompatible_fraction:
        raise ValueError('The program duration in samples {} is not an integer'.format(program.duration * sample_rate))
    if comp_level == _CompatibilityLevel.incompatible_too_short:
        raise ValueError('The program is too short to be a valid waveform. \n'
                         ' program duration in samples: {} \n'
                         ' minimal length: {}'.format(program.duration * sample_rate, minimal_waveform_length))
    if comp_level == _CompatibilityLevel.incompatible_quantum:
        raise ValueError('The program duration in samples {} '
                         'is not a multiple of quantum {}'.format(program.duration * sample_rate, waveform_quantum))

    elif comp_level == _CompatibilityLevel.action_required:
        warnings.warn("qupulse will now concatenate waveforms to make the pulse/program compatible with the chosen AWG."
                      " This might take some time. If you need this pulse more often it makes sense to write it in a "
                      "way which is more AWG friendly.", MakeCompatibleWarning)

        _make_compatible(program,
                         min_len=minimal_waveform_length,
                         quantum=waveform_quantum,
                         sample_rate=sample_rate)

    else:
        assert comp_level == _CompatibilityLevel.compatible


def roll_constant_waveforms(program: Loop, minimal_waveform_quanta: int, waveform_quantum: int, sample_rate: TimeType):
    """This function finds waveforms in program that can be replaced with repetitions of shorter waveforms and replaces
    them. Complexity O(N_waveforms)

    This is possible if:
     - The waveform is constant on all channels
     - waveform.duration * sample_rate / waveform_quantum has a factor that is bigger than minimal_waveform_quanta

    Args:
        program:
        minimal_waveform_quanta:
        waveform_quantum:
        sample_rate:
    """
    waveform = program.waveform

    if waveform is None:
        for child in program:
            roll_constant_waveforms(child, minimal_waveform_quanta, waveform_quantum, sample_rate)
    else:
        waveform_quanta = (waveform.duration * sample_rate) // waveform_quantum

        # example
        # waveform_quanta = 15
        # minimal_waveform_quanta = 2
        # => repetition_count = 5, new_waveform_quanta = 3
        if waveform_quanta < minimal_waveform_quanta * 2:
            # there is no way to roll this waveform because it is too short
            return

        const_values = waveform.constant_value_dict()
        if const_values is None:
            # The waveform is not constant
            return

        new_waveform_quanta = smallest_factor_ge(waveform_quanta, min_factor=minimal_waveform_quanta)
        if new_waveform_quanta == waveform_quanta:
            # the waveform duration in samples has no suitable factor
            # TODO: Option to insert multiple Loop objects
            return

        additional_repetition_count = waveform_quanta // new_waveform_quanta

        new_waveform = ConstantWaveform.from_mapping(
            duration=waveform_quantum * new_waveform_quanta / sample_rate,
            constant_values=const_values)

        # use the private properties to avoid invalidating the duration cache of the parent loop
        program._repetition_definition = program.repetition_definition * additional_repetition_count
        program._waveform = new_waveform


class MakeCompatibleWarning(ResourceWarning):
    pass


class VolatileModificationWarning(RuntimeWarning):
    """This warning is emitted if the colatile part of a program gets modified. This might imply that the volatile
    parameter cannot be change anymore."""


class DroppedMeasurementWarning(RuntimeWarning):
    """This warning is emitted if a measurement was dropped because there was no waveform attached."""
