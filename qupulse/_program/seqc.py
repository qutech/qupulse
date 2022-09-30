"""This module contains the ZI HDAWG compatible description of programs. There is no code in here that interacts with
hardware directly.

The public interface to all functionality is given by `HDAWGProgramManager`. This class can create seqc source code
which contains multiple programs and allows switching between these with the user registers of a device,

Furthermore:
- `SEQCNode`: AST of a subset of sequencing C
- `loop_to_seqc`: conversion of `Loop` objects to this subset in a clever way
- `BinaryWaveform`: Bundles functionality of handling segments in a native way.
- `WaveformMemory`: Functionality to sync waveforms to the device (via the LabOne user folder)
- `ProgramWaveformManager` and `HDAWGProgramEntry`: Program wise handling of waveforms and seqc-code
classes that convert `Loop` objects"""
import warnings
from typing import Optional, Union, Sequence, Dict, Iterator, Tuple, Callable, NamedTuple, MutableMapping, Mapping,\
    Iterable, Any, List, Deque
from types import MappingProxyType
import abc
import itertools
import inspect
import logging
import hashlib
from weakref import WeakValueDictionary
from collections import OrderedDict
import re
import collections
import numbers
import string
import functools

import numpy as np
from pathlib import Path

from qupulse.utils.types import ChannelID, TimeType
from qupulse.utils import replace_multiple, grouper
from qupulse._program.waveforms import Waveform
from qupulse._program._loop import Loop
from qupulse._program.volatile import VolatileRepetitionCount, VolatileProperty
from qupulse.hardware.awgs.base import ProgramEntry
from qupulse.hardware.util import zhinst_voltage_to_uint16
from qupulse.pulses.parameters import MappedParameter, ConstantParameter

try:
    # zhinst fires a DeprecationWarning from its own code in some versions...
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        import zhinst.utils
except ImportError:
    zhinst = None


__all__ = ["HDAWGProgramManager"]


def make_valid_identifier(name: str) -> str:
    # replace all invalid characters and conactenate with hash of original name
    name_hash = hashlib.sha256(name.encode('utf-8')).hexdigest()
    valid_chars = string.ascii_letters + string.digits + '_'
    namestub = ''.join(c for c in name if c in valid_chars)
    return f'renamed_{namestub}_{name_hash}'


class BinaryWaveform:
    """This class represents a sampled waveform in the native HDAWG format as returned
    by zhinst.utils.convert_awg_waveform.

    BinaryWaveform.data can be uploaded directly to {device]/awgs/{awg}/waveform/waves/{wf}

    `to_csv_compatible_table` can be used to create a compatible compact csv file (with marker data included)
    """
    __slots__ = ('data',)

    PLAYBACK_QUANTUM = 16
    PLAYBACK_MIN_QUANTA = 2

    def __init__(self, data: np.ndarray):
        """ TODO: always use both channels?

        Args:
            data: data as returned from zhinst.utils.convert_awg_waveform
        """
        n_quantum, remainder = divmod(data.size, 3 * self.PLAYBACK_QUANTUM)
        assert n_quantum > 1, "Waveform too short (min len is 32)"
        assert remainder == 0, "Waveform has not a valid length"
        assert data.dtype is np.dtype('uint16')
        assert np.all(data[2::3] < 16), "invalid marker data"
        assert data.ndim == 1, "Data not one dimensional"

        self.data = data
        self.data.flags.writeable = False

    @property
    def ch1(self):
        return self.data[::3]

    @property
    def ch2(self):
        return self.data[1::3]

    @property
    def marker_data(self):
        return self.data[2::3]

    @property
    def markers_ch1(self):
        return np.bitwise_and(self.marker_data, 0b0011)

    @property
    def markers_ch2(self):
        return np.bitwise_and(self.marker_data, 0b1100)

    @classmethod
    def from_sampled(cls, ch1: Optional[np.ndarray], ch2: Optional[np.ndarray],
                     markers: Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                    Optional[np.ndarray], Optional[np.ndarray]]) -> 'BinaryWaveform':
        """Combines the sampled and scaled waveform data into a single binary compatible waveform

        Args:
            ch1: sampled waveform scaled to full range (-1., 1.)
            ch2: sampled waveform scaled to full range (-1., 1.)
            markers: (ch1_front_marker, ch1_dio_marker, ch2_front_marker, ch2_dio_marker)

        Returns:

        """
        return cls(zhinst_voltage_to_uint16(ch1, ch2, markers))

    @classmethod
    def zeroed(cls, size):
        return cls(zhinst.utils.convert_awg_waveform(np.zeros(size), np.zeros(size), np.zeros(size, dtype=np.uint16)))

    def __len__(self):
        return self.data.size // 3

    def __eq__(self, other):
        return np.array_equal(self.data, other.data)

    def __hash__(self):
        return hash(bytes(self.data))

    def fingerprint(self) -> str:
        """This fingerprint is runtime independent"""
        return hashlib.sha256(self.data).hexdigest()

    def to_csv_compatible_table(self) -> np.ndarray:
        """The integer values in that file should be 18-bit unsigned integers with the two least significant bits
        being the markers. The values are mapped to 0 => -FS, 262143 => +FS, with FS equal to the full scale.

        >>> np.savetxt(waveform_dir, binary_waveform.to_csv_compatible_table(), fmt='%u')
        """
        assert self.data.size % self.PLAYBACK_QUANTUM == 0, "conversion to csv requires a valid length"

        table = np.zeros((len(self), 2), dtype=np.uint32)
        table[:, 0] = self.ch1
        table[:, 1] = self.ch2
        np.left_shift(table, 2, out=table)
        table[:, 0] += self.markers_ch1
        table[:, 1] += self.markers_ch2

        return table

    def playback_possible(self) -> bool:
        """Returns if the waveform can be played without padding"""
        return self.data.size % self.PLAYBACK_QUANTUM == 0

    def dynamic_rate(self, max_rate: int = 12) -> int:
        min_pre_division_quanta = 2 * self.PLAYBACK_QUANTUM

        reduced = self.data.reshape(-1, 3)
        for n in range(max_rate):
            n_quantum, remainder = divmod(reduced.shape[0], min_pre_division_quanta)
            if remainder != 0 or n_quantum < self.PLAYBACK_MIN_QUANTA or np.any(reduced[::2, :] != reduced[1::2, :]):
                return n
            reduced = reduced[::2, :]
        return max_rate


class ConcatenatedWaveform:
    def __init__(self):
        """Handle the concatenation of multiple binary waveforms to create a big indexable waveform."""
        self._concatenated: Optional[List[Tuple[BinaryWaveform, ...]]] = []
        self._as_binary: Optional[Tuple[BinaryWaveform, ...]] = None

    def __bool__(self):
        return bool(self._concatenated)

    def is_finalized(self):
        return self._as_binary is not None or self._concatenated is None

    def as_binary(self) -> Optional[Tuple[BinaryWaveform, ...]]:
        assert self.is_finalized()
        return self._as_binary

    def append(self, binary_waveform: Tuple[BinaryWaveform, ...]):
        assert not self.is_finalized()
        assert not self._concatenated or len(self._concatenated[-1]) == len(binary_waveform)
        self._concatenated.append(binary_waveform)

    def finalize(self):
        assert not self.is_finalized()
        if self._concatenated:
            n_groups = len(self._concatenated[0])
            as_binary = [[] for _ in range(n_groups)]
            for wf_tuple in self._concatenated:
                for grp, wf in enumerate(wf_tuple):
                    as_binary[grp].append(wf.data)
            self._as_binary = tuple(BinaryWaveform(np.concatenate(as_bin)) for as_bin in as_binary)
        else:
            self._concatenated = None

    def clear(self):
        if self._concatenated is None:
            self._concatenated = []
        else:
            self._concatenated.clear()
        self._as_binary = None


class WaveformFileSystem:
    logger = logging.getLogger('qupulse.hdawg.waveforms')
    _by_path = WeakValueDictionary()

    def __init__(self, path: Path):
        """This class coordinates multiple AWGs (channel pairs) using the same file system to store the waveforms.

        Args:
            path: Waveforms are stored here
        """
        self._required = {}
        self._path = path
    
    @classmethod
    def get_waveform_file_system(cls, path: Path) -> 'WaveformFileSystem':
        """Get the instance for the given path. Multiple instances that access the same path lead to inconsistencies."""
        return cls._by_path.setdefault(path, cls(path))

    def sync(self, client: 'WaveformMemory', waveforms: Mapping[str, BinaryWaveform], **kwargs):
        """Write the required waveforms to the filesystem."""
        self._required[id(client)] = waveforms
        self._sync(**kwargs)

    def _sync(self, delete=True, write_all=False):
        to_save = {self._path.joinpath(file_name): binary
                   for d in self._required.values()
                   for file_name, binary in d.items()}

        for existing_file in self._path.iterdir():
            if not existing_file.is_file():
                pass
            elif existing_file in to_save:
                if not write_all:
                    self.logger.debug('Skipping %r', existing_file.name)
                    to_save.pop(existing_file)
            elif delete:
                try:
                    self.logger.debug('Deleting %r', existing_file.name)
                    existing_file.unlink()
                except OSError:
                    self.logger.exception("Error deleting: %r", existing_file.name)

        for file_name, binary_waveform in to_save.items():
            table = binary_waveform.to_csv_compatible_table()
            np.savetxt(file_name, table, '%u')
            self.logger.debug('Wrote %r', file_name)


class WaveformMemory:
    """Global waveform "memory" representation (currently the file system)"""
    CONCATENATED_WAVEFORM_TEMPLATE = '{program_name}_concatenated_waveform_{group_index}'
    SHARED_WAVEFORM_TEMPLATE = '{program_name}_shared_waveform_{hash}'
    WF_PLACEHOLDER_TEMPLATE = '*{id}*'
    FILE_NAME_TEMPLATE = '{hash}.csv'

    _WaveInfo = NamedTuple('_WaveInfo', [('wave_name', str),
                                         ('file_name', str),
                                         ('binary_waveform', BinaryWaveform)])

    def __init__(self):
        self.shared_waveforms = OrderedDict()  # type: MutableMapping[BinaryWaveform, set]
        self.concatenated_waveforms = OrderedDict()  # type: MutableMapping[str, ConcatenatedWaveform]

    def clear(self):
        self.shared_waveforms.clear()
        self.concatenated_waveforms.clear()

    def _shared_waveforms_iter(self) -> Iterator[Tuple[str, _WaveInfo]]:
        for wf, program_set in self.shared_waveforms.items():
            if program_set:
                wave_hash = wf.fingerprint()
                wave_name = self.SHARED_WAVEFORM_TEMPLATE.format(program_name='_'.join(program_set),
                                                                 hash=wave_hash)
                wave_placeholder = self.WF_PLACEHOLDER_TEMPLATE.format(id=id(program_set))
                file_name = self.FILE_NAME_TEMPLATE.format(hash=wave_hash)
                yield wave_placeholder, self._WaveInfo(wave_name, file_name, wf)

    def _concatenated_waveforms_iter(self) -> Iterator[Tuple[str, Tuple[_WaveInfo, ...]]]:
        for program_name, concatenated_waveform in self.concatenated_waveforms.items():
            # we assume that if the first entry is not empty the rest also isn't
            if concatenated_waveform:
                infos = []
                for group_index, binary in enumerate(concatenated_waveform.as_binary()):
                    wave_hash = binary.fingerprint()
                    wave_name = self.CONCATENATED_WAVEFORM_TEMPLATE.format(program_name=program_name,
                                                                           group_index=group_index)
                    file_name = self.FILE_NAME_TEMPLATE.format(hash=wave_hash)
                    infos.append(self._WaveInfo(wave_name, file_name, binary))

                wave_placeholder = self.WF_PLACEHOLDER_TEMPLATE.format(id=id(concatenated_waveform))
                yield wave_placeholder, tuple(infos)

    def _all_info_iter(self) -> Iterator[_WaveInfo]:
        for _, infos in self._concatenated_waveforms_iter():
            yield from infos
        for _, info in self._shared_waveforms_iter():
            yield info

    def waveform_name_replacements(self) -> Dict[str, str]:
        """replace place holders of complete seqc program with

        >>> waveform_name_translation = waveform_memory.waveform_name_replacements()
        >>> seqc_program = qupulse.utils.replace_multiple(seqc_program, waveform_name_translation)
        """
        translation = {}
        for wave_placeholder, wave_info in self._shared_waveforms_iter():
            translation[wave_placeholder] = wave_info.wave_name

        for wave_placeholder, wave_infos in self._concatenated_waveforms_iter():
            translation[wave_placeholder] = ','.join(info.wave_name for info in wave_infos)
        return translation

    def waveform_declaration(self) -> str:
        """Produces a string that declares all needed waveforms.
        It is needed to know the waveform index in case we want to update a waveform during playback."""
        declarations = []
        for wave_info in self._all_info_iter():
            declarations.append(
                'wave {wave_name} = "{file_name}";'.format(wave_name=wave_info.wave_name,
                                                           file_name=wave_info.file_name.replace('.csv', ''))
            )
        return '\n'.join(declarations)

    def sync_to_file_system(self, file_system: WaveformFileSystem):
        to_save = {wave_info.file_name: wave_info.binary_waveform
                   for wave_info in self._all_info_iter()}
        file_system.sync(self, to_save)


class ProgramWaveformManager:
    """Manages waveforms of a program"""
    def __init__(self, name: str, memory: WaveformMemory):
        if not name.isidentifier():
            waveform_name = make_valid_identifier(name)
        else:
            waveform_name = name
        
        self._waveform_name = waveform_name
        self._program_name = name
        self._memory = memory

        assert self._program_name not in self._memory.concatenated_waveforms
        assert all(self._program_name not in programs for programs in self._memory.shared_waveforms.values())
        self._memory.concatenated_waveforms[waveform_name] = ConcatenatedWaveform()

    @property
    def program_name(self) -> str:
        return self._program_name
    
    @property
    def main_waveform_name(self) -> str:
        self._waveform_name

    def clear_requested(self):
        for programs in self._memory.shared_waveforms.values():
            programs.discard(self._program_name)
        self._memory.concatenated_waveforms[self._waveform_name].clear()

    def request_shared(self, binary_waveform: Tuple[BinaryWaveform, ...]) -> str:
        """Register waveform if not already registered and return a unique identifier placeholder.

        The unique identifier currently is computed from the id of the set which stores all programs using this
        waveform.
        """
        placeholders = []
        for wf in binary_waveform:
            program_set = self._memory.shared_waveforms.setdefault(wf, set())
            program_set.add(self._program_name)
            placeholders.append(self._memory.WF_PLACEHOLDER_TEMPLATE.format(id=id(program_set)))
        return ",".join(placeholders)

    def request_concatenated(self, binary_waveform: Tuple[BinaryWaveform, ...]) -> str:
        """Append the waveform to the concatenated waveform"""
        bin_wf_list = self._memory.concatenated_waveforms[self._waveform_name]
        bin_wf_list.append(binary_waveform)
        return self._memory.WF_PLACEHOLDER_TEMPLATE.format(id=id(bin_wf_list))

    def finalize(self):
        self._memory.concatenated_waveforms[self._waveform_name].finalize()

    def prepare_delete(self):
        """Delete all references in waveform memory to this program. Cannot be used afterwards."""
        self.clear_requested()
        del self._memory.concatenated_waveforms[self._waveform_name]


class UserRegister:
    """This class is a helper class to avoid errors due to 0 and 1 based register indexing"""
    __slots__ = ('_zero_based_value',)

    def __init__(self, *, zero_based_value: int = None, one_based_value: int = None):
        assert None in (zero_based_value, one_based_value)
        assert isinstance(zero_based_value, int) or isinstance(one_based_value, int)

        if one_based_value is not None:
            assert one_based_value > 0, "A one based value needs to be larger zero"
            self._zero_based_value = one_based_value - 1
        else:
            self._zero_based_value = zero_based_value

    @classmethod
    def from_seqc(cls, value: int) -> 'UserRegister':
        return cls(zero_based_value=value)

    def to_seqc(self) -> int:
        return self._zero_based_value

    @classmethod
    def from_labone(cls, value: int) -> 'UserRegister':
        return cls(zero_based_value=value)

    def to_labone(self) -> int:
        return self._zero_based_value

    @classmethod
    def from_web_interface(cls, value: int) -> 'UserRegister':
        return cls(one_based_value=value)

    def to_web_interface(self) -> int:
        return self._zero_based_value + 1

    def __hash__(self):
        return hash(self._zero_based_value)

    def __eq__(self, other):
        return self._zero_based_value == getattr(other, '_zero_based_value', None)

    def __repr__(self):
        return 'UserRegister(zero_based_value={zero_based_value})'.format(zero_based_value=self._zero_based_value)

    def __format__(self, format_spec: str) -> str:
        if format_spec in ('zero_based', 'seqc', 'labone', 'lab_one'):
            return str(self.to_seqc())
        elif format_spec in ('one_based', 'web', 'web_interface'):
            return str(self.to_web_interface())
        elif format_spec in ('repr', 'r'):
            return repr(self)
        else:
            raise ValueError('Invalid format spec for UserRegister: ', format_spec)


class UserRegisterManager:
    """This class keeps track of the user registered that are used in a certain context"""
    def __init__(self, available: Iterable[UserRegister], name_template: str):
        assert 'register' in (x[1] for x in string.Formatter().parse(name_template))

        self._available = set(available)
        self._name_template = name_template
        self._used = {}

    def request(self, obj) -> str:
        """Request a user register name to store object. If an object that evaluates equal to obj was requested before
        the name name is returned.

        Args:
            obj: Object to store

        Returns:
            Name of the variable with the user register

        Raises:
            Value error if no register is available
        """
        for register, registered_obj in self._used.items():
            if obj == registered_obj:
                return self._name_template.format(register=register)
        if self._available:
            register = self._available.pop()
            self._used[register] = obj
            return self._name_template.format(register=register)
        else:
            raise ValueError("No register available for %r" % obj)

    def iter_used_register_names(self) -> Iterator[Tuple[UserRegister, str]]:
        """

        Returns:
            An iterator over (register index, register name) pairs
        """
        return ((register, self._name_template.format(register=register)) for register in self._used.keys())

    def iter_used_register_values(self) -> Iterable[Tuple[UserRegister, Any]]:
        return self._used.items()


class HDAWGProgramEntry(ProgramEntry):
    USER_REG_NAME_TEMPLATE = 'user_reg_{register:seqc}'

    def __init__(self, loop: Loop, selection_index: int, waveform_memory: WaveformMemory, program_name: str,
                 channels: Tuple[Optional[ChannelID], ...],
                 markers: Tuple[Optional[ChannelID], ...],
                 amplitudes: Tuple[float, ...],
                 offsets: Tuple[float, ...],
                 voltage_transformations: Tuple[Optional[Callable], ...],
                 sample_rate: TimeType):
        super().__init__(loop, channels=channels, markers=markers,
                         amplitudes=amplitudes,
                         offsets=offsets,
                         voltage_transformations=voltage_transformations,
                         sample_rate=sample_rate)
        for waveform, (all_sampled_channels, all_sampled_markers) in self._waveforms.items():
            size = int(waveform.duration * sample_rate)

            # group in channel pairs for binary waveform
            binary_waveforms = []
            for (sampled_channels, sampled_markers) in zip(grouper(all_sampled_channels, 2),
                                                           grouper(all_sampled_markers, 4)):
                if all(x is None for x in (*sampled_channels, *sampled_markers)):
                    # empty channel pairs
                    binary_waveforms.append(BinaryWaveform.zeroed(size))
                else:
                    binary_waveforms.append(BinaryWaveform.from_sampled(*sampled_channels, sampled_markers))
            self._waveforms[waveform] = tuple(binary_waveforms)

        self._waveform_manager = ProgramWaveformManager(program_name, waveform_memory)
        self.selection_index = selection_index
        self._trigger_wait_code = None
        self._seqc_node = None
        self._seqc_source = None
        self._var_declarations = None
        self._user_registers = None
        self._user_register_source = None

    def compile(self,
                min_repetitions_for_for_loop: int,
                min_repetitions_for_shared_wf: int,
                indentation: str,
                trigger_wait_code: str,
                available_registers: Iterable[UserRegister]):
        """Compile the loop representation to an internal sequencing c one using `loop_to_seqc`

        Args:
            min_repetitions_for_for_loop: See `loop_to_seqc`
            min_repetitions_for_shared_wf: See `loop_to_seqc`
            indentation: Each line is prefixed with this
            trigger_wait_code: The code is put before the playback start
            available_registers
        Returns:

        """
        pos_var_name = 'pos'

        if self._seqc_node:
            self._waveform_manager.clear_requested()

        user_registers = UserRegisterManager(available_registers, self.USER_REG_NAME_TEMPLATE)

        self._seqc_node = loop_to_seqc(self._loop,
                                       min_repetitions_for_for_loop=min_repetitions_for_for_loop,
                                       min_repetitions_for_shared_wf=min_repetitions_for_shared_wf,
                                       waveform_to_bin=self.get_binary_waveform,
                                       user_registers=user_registers)

        self._user_register_source = '\n'.join(
            '{indentation}var {user_reg_name} = getUserReg({register});'.format(indentation=indentation,
                                                                                user_reg_name=user_reg_name,
                                                                                register=register.to_seqc())
            for register, user_reg_name in user_registers.iter_used_register_names()
        )
        self._user_registers = user_registers

        self._var_declarations = '{indentation}var {pos_var_name} = 0;'.format(pos_var_name=pos_var_name,
                                                                               indentation=indentation)
        self._trigger_wait_code = indentation + trigger_wait_code
        self._seqc_source = '\n'.join(self._seqc_node.to_source_code(self._waveform_manager,
                                                                     map(str, itertools.count(1)),
                                                                     line_prefix=indentation,
                                                                     pos_var_name=pos_var_name))
        self._waveform_manager.finalize()

    @property
    def seqc_node(self) -> 'SEQCNode':
        assert self._seqc_node is not None, "compile not called"
        return self._seqc_node

    @property
    def seqc_source(self) -> str:
        assert self._seqc_source is not None, "compile not called"
        return '\n'.join([self._var_declarations,
                          self._user_register_source,
                          self._trigger_wait_code,
                          self._seqc_source])

    def volatile_repetition_counts(self) -> Iterable[Tuple[UserRegister, VolatileRepetitionCount]]:
        """
        Returns:
            An iterator over the register and parameter
        """
        assert self._user_registers is not None, "compile not called"
        return self._user_registers.iter_used_register_values()

    @property
    def name(self) -> str:
        return self._waveform_manager.program_name

    def parse_to_seqc(self, waveform_memory):
        raise NotImplementedError()

    def get_binary_waveform(self, waveform: Waveform) -> Tuple[BinaryWaveform, ...]:
        return self._waveforms[waveform]

    def prepare_delete(self):
        """Delete all references to this program. Cannot be used afterwards"""
        self._waveform_manager.prepare_delete()
        self._seqc_node = None
        self._seqc_source = None


class HDAWGProgramManager:
    """This class contains everything that is needed to create the final seqc program and provides an interface to write
    the required waveforms to the file system. It does not talk to the device."""

    class Constants:
        PROG_SEL_REGISTER = UserRegister(zero_based_value=0)
        TRIGGER_REGISTER = UserRegister(zero_based_value=1)
        TRIGGER_RESET_MASK = bin(1 << 31)
        PROG_SEL_NONE = 0
        # if not set the register is set to PROG_SEL_NONE
        NO_RESET_MASK = bin(1 << 31)
        # set to one if playback finished
        PLAYBACK_FINISHED_MASK = bin(1 << 30)
        PROG_SEL_MASK = bin((1 << 30) - 1)
        INVERTED_PROG_SEL_MASK = bin(((1 << 32) - 1) ^ int(PROG_SEL_MASK, 2))
        IDLE_WAIT_CYCLES = 300

        @classmethod
        def as_dict(cls) -> Dict[str, Any]:
            return {name: value
                    for name, value in vars(cls).items()
                    if name[0] in string.ascii_uppercase}

    class GlobalVariables:
        """Global variables of the program together with their (multiline) doc string.
        The python names are uppercase."""

        PROG_SEL = (['Selected program index (0 -> None)'], 0)
        NEW_PROG_SEL = (('Value that gets written back to program selection register.',
                         'Used to signal that at least one program was played completely.'), 0)
        PLAYBACK_FINISHED = (('Is OR\'ed to new_prog_sel.',
                              'Set to PLAYBACK_FINISHED_MASK if a program was played completely.',), 0)

        @classmethod
        def as_dict(cls) -> Dict[str, Tuple[Sequence[str], int]]:
            return {name: value
                    for name, value in vars(cls).items()
                    if name[0] in string.ascii_uppercase}

        @classmethod
        def get_init_block(cls) -> str:
            lines = ['// Declare and initialize global variables']
            for var_name, (comment, initial_value) in cls.as_dict().items():
                lines.extend(f'// {comment_line}' for comment_line in comment)
                lines.append(f'var {var_name.lower()} = {initial_value};')
                lines.append('')
            return '\n'.join(lines)

    _PROGRAM_FUNCTION_NAME_TEMPLATE = '{program_name}_function'
    WAIT_FOR_SOFTWARE_TRIGGER = "waitForSoftwareTrigger();"
    SOFTWARE_WAIT_FOR_TRIGGER_FUNCTION_DEFINITION = (
        'void waitForSoftwareTrigger() {\n'
        '  while (true) {\n'
        '    var trigger_register = getUserReg(TRIGGER_REGISTER);\n'
        '    if (trigger_register & TRIGGER_RESET_MASK) setUserReg(TRIGGER_REGISTER, 0);\n'
        '    if (trigger_register) return;\n'
        '  }\n'
        '}\n'
    )
    DEFAULT_COMPILER_SETTINGS = {
        'trigger_wait_code': WAIT_FOR_SOFTWARE_TRIGGER,
        'min_repetitions_for_for_loop': 20,
        'min_repetitions_for_shared_wf': 1000,
        'indentation': '  '
    }
    
    @classmethod
    def get_program_function_name(cls, program_name: str):
        if not program_name.isidentifier():
            program_name = make_valid_identifier(program_name)
        return cls._PROGRAM_FUNCTION_NAME_TEMPLATE.format(program_name=program_name)

    def __init__(self):
        self._waveform_memory = WaveformMemory()
        self._programs = OrderedDict()  # type: MutableMapping[str, HDAWGProgramEntry]
        self._compiler_settings = [
            # default settings: None -> take cls value
            (re.compile('.*'), {'trigger_wait_code': None,
                                'min_repetitions_for_for_loop': None,
                                'min_repetitions_for_shared_wf': None,
                                'indentation': None})]

    def _get_compiler_settings(self, program_name: str) -> dict:
        arg_spec = inspect.getfullargspec(HDAWGProgramEntry.compile)
        required_compiler_args = (set(arg_spec.args) | set(arg_spec.kwonlyargs)) - {'self', 'available_registers'}

        settings = {}
        for regex, settings_dict in self._compiler_settings:
            if regex.match(program_name):
                settings.update(settings_dict)
        if required_compiler_args - set(settings):
            raise ValueError('Not all compiler arguments for program have been defined.'
                             ' (the default catch all has been removed)'
                             f'Missing: {required_compiler_args - set(settings)}')
        for k, v in settings.items():
            if v is None:
                settings[k] = self.DEFAULT_COMPILER_SETTINGS[k]
        return settings

    @property
    def waveform_memory(self):
        return self._waveform_memory

    def _get_low_unused_index(self):
        existing = {entry.selection_index for entry in self._programs.values()}
        for idx in itertools.count():
            if idx not in existing and idx != self.Constants.PROG_SEL_NONE:
                return idx

    def add_program(self, name: str, loop: Loop,
                    channels: Tuple[Optional[ChannelID], ...],
                    markers: Tuple[Optional[ChannelID], ...],
                    amplitudes: Tuple[float, ...],
                    offsets: Tuple[float, ...],
                    voltage_transformations: Tuple[Optional[Callable], ...],
                    sample_rate: TimeType):
        """Register the given program and translate it to seqc.

        TODO: Add an interface to change the trigger mode

        Args:
            name: Human readable name of the program (used f.i. for the function name)
            loop: The program to upload
            channels: see AWG.upload
            markers: see AWG.upload
            amplitudes: Used to sample the waveforms
            offsets: Used to sample the waveforms
            voltage_transformations: see AWG.upload
            sample_rate: Used to sample the waveforms
        """
        assert name not in self._programs

        selection_index = self._get_low_unused_index()

        # TODO: verify total number of registers
        available_registers = [UserRegister.from_seqc(idx) for idx in range(2, 16)]

        program_entry = HDAWGProgramEntry(loop, selection_index, self._waveform_memory, name,
                                          channels, markers, amplitudes, offsets, voltage_transformations, sample_rate)

        compiler_settings = self._get_compiler_settings(program_name=name)

        # TODO: put compilation in seperate function
        program_entry.compile(**compiler_settings,
                              available_registers=available_registers)

        self._programs[name] = program_entry

    def get_register_values(self, name: str) -> Mapping[UserRegister, int]:
        return {register: int(parameter)
                for register, parameter in self._programs[name].volatile_repetition_counts()}

    def get_register_values_to_update_volatile_parameters(self, name: str,
                                                          parameters: Mapping[str,
                                                                              numbers.Number]) -> Mapping[UserRegister,
                                                                                                          int]:
        """

        Args:
            name: Program name
            parameters: new values for volatile parameters

        Returns:
            A dict user_register->value that reflects the new parameter values
        """
        program_entry = self._programs[name]
        result = {}
        for register, volatile_repetition in program_entry.volatile_repetition_counts():
            new_value = volatile_repetition.update_volatile_dependencies(parameters)
            result[register] = new_value
        return result

    @property
    def programs(self) -> Mapping[str, HDAWGProgramEntry]:
        return MappingProxyType(self._programs)

    def remove(self, name: str) -> None:
        self._programs.pop(name).prepare_delete()

    def clear(self) -> None:
        self._waveform_memory.clear()
        self._programs.clear()

    def name_to_index(self, name: str) -> int:
        assert self._programs[name].name == name
        return self._programs[name].selection_index

    def _get_sub_program_source_code(self, program_name: str) -> str:
        program = self.programs[program_name]
        program_function_name = self.get_program_function_name(program_name)
        return "\n".join(
            [
                f"void {program_function_name}() {{",
                program.seqc_source,
                "}\n"
            ]
        )

    def _get_program_selection_code(self) -> str:
        return _make_program_selection_block((program.selection_index, self.get_program_function_name(program_name))
                                             for program_name, program in self.programs.items())

    def to_seqc_program(self, single_program: Optional[str] = None) -> str:
        """Generate sequencing c source code that is either capable of playing pack all uploaded programs where the
        program is selected at runtime without re-compile or always will play the same program if `single_program`
        is specified.

         The program selection is based on a user register in the first case.

        Args:
            single_program: The seqc source only contains this program if not None

        Returns:
            SEQC source code.
        """
        lines = []
        for const_name, const_val in self.Constants.as_dict().items():
            if isinstance(const_val, (int, str)):
                const_repr = str(const_val)
            else:
                const_repr = const_val.to_seqc()
            lines.append('const {const_name} = {const_repr};'.format(const_name=const_name, const_repr=const_repr))

        lines.append(self._waveform_memory.waveform_declaration())

        lines.append('\n// function used by manually triggered programs')
        lines.append(self.SOFTWARE_WAIT_FOR_TRIGGER_FUNCTION_DEFINITION)

        replacements = self._waveform_memory.waveform_name_replacements()

        lines.append('\n// program definitions')
        if single_program:
            lines.append(
                replace_multiple(self._get_sub_program_source_code(single_program), replacements)
            )

        else:
            for program_name, program in self.programs.items():
                lines.append(replace_multiple(self._get_sub_program_source_code(program_name), replacements))

        lines.append(self.GlobalVariables.get_init_block())

        lines.append('\n// runtime block')
        if single_program:
            lines.append(f"{self.get_program_function_name(single_program)}();")
        else:
            lines.append(self._get_program_selection_code())
        
        return '\n'.join(lines)


def find_sharable_waveforms(node_cluster: Sequence['SEQCNode']) -> Optional[Sequence[bool]]:
    """Expects nodes to have a compatible stepping

    TODO: encode in type system?
    """
    waveform_playbacks = list(node_cluster[0].iter_waveform_playbacks())

    candidates = [True] * len(waveform_playbacks)

    for node in itertools.islice(node_cluster, 1, None):
        candidates_left = False
        for idx, (wf, node_wf) in enumerate(zip(waveform_playbacks, node.iter_waveform_playbacks())):
            if candidates[idx]:
                candidates[idx] = wf == node_wf
            candidates_left = candidates_left or candidates[idx]

        if not candidates_left:
            return None

    return candidates


def mark_sharable_waveforms(node_cluster: Sequence['SEQCNode'], sharable_waveforms: Sequence[bool]):
    for node in node_cluster:
        for sharable, wf_playback in zip(sharable_waveforms, node.iter_waveform_playbacks()):
            if sharable:
                wf_playback.shared = True


def _find_repetition(nodes: Deque['SEQCNode'],
                     hashes: Deque[int],
                     cluster_dump: List[List['SEQCNode']]) -> Tuple[
    Tuple['SEQCNode', ...],
    Tuple[int, ...],
    List['SEQCNode']
]:
    """Finds repetitions of stepping patterns in nodes. Assumes hashes contains the stepping_hash of each node. If a
    pattern is """
    assert len(nodes) == len(hashes)

    max_cluster_size = len(nodes) // 2
    for cluster_size in range(max_cluster_size, 0, -1):
        n_repetitions = len(nodes) // cluster_size
        for c_idx in range(cluster_size):
            idx_a = -1 - c_idx

            for n in range(1, n_repetitions):
                idx_b = idx_a - n * cluster_size
                if hashes[idx_a] != hashes[idx_b] or not nodes[idx_a].same_stepping(nodes[idx_b]):
                    n_repetitions = n
                    break

            if n_repetitions < 2:
                break

        else:
            assert n_repetitions > 1
            # found a stepping pattern repetition of length cluster_size!
            to_dump = len(nodes) - (n_repetitions * cluster_size)
            for _ in range(to_dump):
                cluster_dump.append([nodes.popleft()])
                hashes.popleft()

            assert len(nodes) == n_repetitions * cluster_size

            if cluster_size == 1:
                current_cluster = list(nodes)

                cluster_template_hashes = (hashes.popleft(),)
                cluster_template: Tuple[SEQCNode] = (nodes.popleft(),)

                nodes.clear()
                hashes.clear()

            else:
                cluster_template_hashes = tuple(hashes.popleft() for _ in range(cluster_size))
                cluster_template = tuple(
                    nodes.popleft() for _ in range(cluster_size)
                )

                current_cluster: List[SEQCNode] = [Scope(list(cluster_template))]

                for n in range(1, n_repetitions):
                    current_cluster.append(Scope([
                        nodes.popleft() for _ in range(cluster_size)
                    ]))
                assert not nodes
                hashes.clear()

            return cluster_template, cluster_template_hashes, current_cluster
    return (), (), []


def to_node_clusters(loop: Union[Sequence[Loop], Loop], loop_to_seqc_kwargs: dict) -> Sequence[Sequence['SEQCNode']]:
    """transform to seqc recursively noes and cluster them if they have compatible stepping"""
    assert len(loop) > 1

    # complexity: O( len(loop) * MAX_SUB_CLUSTER * loop.depth() )
    # I hope...
    MAX_SUB_CLUSTER = 4

    node_clusters: List[List[SEQCNode]] = []

    # this is the period that we currently are collecting
    current_period: List[SEQCNode] = []

    # list of already collected periods. Each period is transformed into a SEQCNode
    current_cluster: List[SEQCNode] = []

    # this is a template for what we are currently collecting
    current_template: Tuple[SEQCNode, ...] = ()
    current_template_hashes: Tuple[int, ...] = ()

    # only populated if we are looking for a node template
    last_node = loop_to_seqc(loop[0], **loop_to_seqc_kwargs)
    last_hashes = collections.deque([last_node.stepping_hash()], maxlen=MAX_SUB_CLUSTER*2)
    last_nodes = collections.deque([last_node], maxlen=MAX_SUB_CLUSTER*2)

    # compress all nodes in clusters of the same stepping
    for child in itertools.islice(loop, 1, None):
        current_node = loop_to_seqc(child, **loop_to_seqc_kwargs)
        current_hash = current_node.stepping_hash()

        if current_template:
            # we are currently collecting something
            idx = len(current_period)
            if current_template_hashes[idx] == current_hash and current_node.same_stepping(current_template[idx]):
                current_period.append(current_node)

                if len(current_period) == len(current_template):
                    if idx == 0:
                        node = current_period.pop()
                    else:
                        node = Scope(current_period)
                        current_period = []
                    current_cluster.append(node)

            else:
                # current template became invalid
                assert len(current_cluster) > 1
                node_clusters.append(current_cluster)

                assert not last_nodes
                assert not last_hashes
                last_nodes.extend(current_period)
                last_hashes.extend(current_template_hashes[:len(current_period)])

                current_period.clear()

                last_nodes.append(current_node)
                last_hashes.append(current_hash)

                (current_template,
                 current_template_hashes,
                 current_cluster) = _find_repetition(last_nodes, last_hashes,
                                                     node_clusters)
        else:
            assert not current_period
            if len(last_nodes) == last_nodes.maxlen:
                # lookup deque is full
                node_clusters.append([last_nodes.popleft()])
                last_hashes.popleft()

            last_nodes.append(current_node)
            last_hashes.append(current_hash)

            (current_template,
             current_template_hashes,
             current_cluster) = _find_repetition(last_nodes, last_hashes,
                                                 node_clusters)

    assert not (current_cluster and last_nodes)
    if current_cluster:
        node_clusters.append(current_cluster)
    node_clusters.extend([node] for node in current_period)
    node_clusters.extend([node] for node in last_nodes)

    return node_clusters


def loop_to_seqc(loop: Loop,
                 min_repetitions_for_for_loop: int,
                 min_repetitions_for_shared_wf: int,
                 waveform_to_bin: Callable[[Waveform], Tuple[BinaryWaveform, ...]],
                 user_registers: UserRegisterManager) -> 'SEQCNode':
    assert min_repetitions_for_for_loop <= min_repetitions_for_shared_wf
    # At which point do we switch from indexed to shared

    if loop.is_leaf():
        node = WaveformPlayback(waveform_to_bin(loop.waveform))

    elif len(loop) == 1:
        node = loop_to_seqc(loop[0],
                            min_repetitions_for_for_loop=min_repetitions_for_for_loop,
                            min_repetitions_for_shared_wf=min_repetitions_for_shared_wf,
                            waveform_to_bin=waveform_to_bin, user_registers=user_registers)

    else:
        node_clusters = to_node_clusters(loop, dict(min_repetitions_for_for_loop=min_repetitions_for_for_loop,
                                                    min_repetitions_for_shared_wf=min_repetitions_for_shared_wf,
                                                    waveform_to_bin=waveform_to_bin,
                                                    user_registers=user_registers))

        seqc_nodes = []

        # identify shared waveforms in node clusters
        for node_cluster in node_clusters:
            if len(node_cluster) < min_repetitions_for_for_loop:
                seqc_nodes.extend(node_cluster)

            else:
                if len(node_cluster) >= min_repetitions_for_shared_wf:
                    sharable_waveforms = find_sharable_waveforms(node_cluster)
                    if sharable_waveforms:
                        mark_sharable_waveforms(node_cluster, sharable_waveforms)

                seqc_nodes.append(SteppingRepeat(node_cluster))

        node = Scope(seqc_nodes)

    if loop.volatile_repetition:
        register_var = user_registers.request(loop.repetition_definition)
        return Repeat(scope=node, repetition_count=register_var)

    elif loop.repetition_count != 1:
        return Repeat(scope=node, repetition_count=loop.repetition_count)
    else:
        return node


class SEQCNode(metaclass=abc.ABCMeta):
    __slots__ = ()

    INDENTATION = '  '

    @abc.abstractmethod
    def samples(self) -> int:
        pass

    @abc.abstractmethod
    def stepping_hash(self) -> int:
        """hash of the stepping properties of this node"""

    @abc.abstractmethod
    def same_stepping(self, other: 'SEQCNode'):
        pass

    @abc.abstractmethod
    def iter_waveform_playbacks(self) -> Iterator['WaveformPlayback']:
        pass

    def _get_single_indexed_playback(self) -> Optional['WaveformPlayback']:
        """Returns None if there is no or if there are more than one indexed playbacks"""
        # detect if there is only a single indexed playback
        single_indexed_playback = None
        for playback in self.iter_waveform_playbacks():
            if not playback.shared:
                if single_indexed_playback is None:
                    single_indexed_playback = playback
                else:
                    break
        else:
            return single_indexed_playback
        return None

    @abc.abstractmethod
    def _visit_nodes(self, waveform_manager: ProgramWaveformManager):
        """push all concatenated waveforms in the waveform manager"""

    @abc.abstractmethod
    def to_source_code(self, waveform_manager: ProgramWaveformManager, node_name_generator: Iterator[str], line_prefix: str, pos_var_name: str,
                        advance_pos_var: bool = True):
        """besides creating the source code, this function registers all needed waveforms to the program manager
        1. shared waveforms
        2. concatenated waveforms in the correct order

        Args:
            waveform_manager:
            node_name_generator: generates unique names of nodes
            line_prefix:
            pos_var_name:
            advance_pos_var: Indexed playback will not advance the position if set to False. This is used internally
            to optimize repeat statements with a single indexed playback.
        Returns:

        """

    def __eq__(self, other):
        """Compare objects based on __slots__"""
        assert getattr(self, '__dict__', None) is None
        return type(self) == type(other) and all(getattr(self, attr) == getattr(other, attr)
                                                 for base_class in inspect.getmro(type(self))
                                                 for attr in getattr(base_class, '__slots__', ()))


class Scope(SEQCNode):
    """Sequence of nodes"""

    __slots__ = ('nodes',)

    def __init__(self, nodes: Sequence[SEQCNode] = ()):
        self.nodes = list(nodes)

    def samples(self):
        return sum(node.samples() for node in self.nodes)

    def iter_waveform_playbacks(self) -> Iterator['WaveformPlayback']:
        for node in self.nodes:
            yield from node.iter_waveform_playbacks()

    def stepping_hash(self) -> int:
        return functools.reduce(int.__xor__, (node.stepping_hash() for node in self.nodes), hash(type(self)))

    def same_stepping(self, other: 'Scope'):
        return (type(other) is Scope and
                len(self.nodes) == len(other.nodes) and
                all(n1.same_stepping(n2) for n1, n2 in zip(self.nodes, other.nodes)))

    def _visit_nodes(self, waveform_manager: ProgramWaveformManager):
        for node in self.nodes:
            node._visit_nodes(waveform_manager)

    def to_source_code(self, waveform_manager: ProgramWaveformManager, node_name_generator: Iterator[str],
                       line_prefix: str, pos_var_name: str,
                       advance_pos_var: bool = True):
        for node in self.nodes:
            yield from node.to_source_code(waveform_manager,
                                           line_prefix=line_prefix,
                                           pos_var_name=pos_var_name,
                                           node_name_generator=node_name_generator,
                                           advance_pos_var=advance_pos_var)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.nodes == other.nodes
        else:
            return NotImplemented

    def __repr__(self):
        return f"Scope(nodes={self.nodes!r})"


class Repeat(SEQCNode):
    """"""
    __slots__ = ('repetition_count', 'scope')
    INITIAL_POSITION_NAME_TEMPLATE = 'init_pos_{node_name}'
    FOR_LOOP_NAME_TEMPLATE = 'idx_{node_name}'

    class _AdvanceStrategy:
        """describes what happens how this node interacts with the position variable"""
        INITIAL_RESET = 'initial_reset'
        POST_ADVANCE = 'post_advance'
        IGNORE = 'ignore'

    def __init__(self, repetition_count: Union[int, str], scope: SEQCNode):
        """
        Args:
            repetition_count: A const integer value or a string that is expected to be a "var"
            scope: The repeated scope
        """
        if isinstance(repetition_count, int):
            assert repetition_count > 1
        else:
            assert isinstance(repetition_count, str) and repetition_count.isidentifier()

        self.repetition_count = repetition_count
        self.scope = scope

    def samples(self):
        return self.scope.samples()

    def same_stepping(self, other: 'Repeat'):
        return (type(self) == type(other) and
                self.repetition_count == other.repetition_count and
                self.scope.same_stepping(other.scope))

    def stepping_hash(self) -> int:
        return hash((type(self), self.repetition_count, self.scope.stepping_hash()))

    def iter_waveform_playbacks(self) -> Iterator['WaveformPlayback']:
        return self.scope.iter_waveform_playbacks()

    def _visit_nodes(self, waveform_manager: ProgramWaveformManager):
        self.scope._visit_nodes(waveform_manager)

    def _get_position_advance_strategy(self):
        """Deduct the optimal position advance strategy:

        There is more than one indexed playback -> position needs to be advanced during each iteration and set back to
        initial value at the begin of each new iteration
        There is exactly one indexed playback -> The position is not advanced in the body but needs to be advanced after
        all repetitions are done
        There is no indexed playback -> We do not care about the position at all
        """
        self_samples = self.samples()
        if self_samples > 0:
            single_playback = self.scope._get_single_indexed_playback()
            if single_playback is None or single_playback.samples() != self_samples:
                # TODO: I am not sure whether the 'single_playback.samples() != self_samples' is necessary
                # there is more than one indexed playback
                return self._AdvanceStrategy.INITIAL_RESET
            else:
                # there is only a single indexed playback
                return self._AdvanceStrategy.POST_ADVANCE
        else:
            # there is no indexed playback
            return self._AdvanceStrategy.IGNORE

    def to_source_code(self, waveform_manager: ProgramWaveformManager, node_name_generator: Iterator[str],
                       line_prefix: str, pos_var_name: str, advance_pos_var: bool = True):
        body_prefix = line_prefix + self.INDENTATION

        advance_strategy = self._get_position_advance_strategy() if advance_pos_var else self._AdvanceStrategy.IGNORE
        inner_advance_pos_var = advance_strategy == self._AdvanceStrategy.INITIAL_RESET

        def get_node_name():
            """Helper to assert node name only generated when needed and only generated once"""
            if getattr(get_node_name, 'node_name', None) is None:
                get_node_name.node_name = next(node_name_generator)
            return get_node_name.node_name

        if advance_strategy == self._AdvanceStrategy.INITIAL_RESET:
            initial_position_name = self.INITIAL_POSITION_NAME_TEMPLATE.format(node_name=get_node_name())

            # store initial position
            yield '{line_prefix}var {init_pos_name} = {pos_var_name};'.format(line_prefix=line_prefix,
                                                                              init_pos_name=initial_position_name,
                                                                              pos_var_name=pos_var_name)

        if isinstance(self.repetition_count, int):
            yield '{line_prefix}repeat({repetition_count}) {{'.format(line_prefix=line_prefix,
                                                                      repetition_count=self.repetition_count)
        else:
            # repeat requires a const-expression so we need to use a for loop for user reg vars
            assert isinstance(self.repetition_count, str)
            loop_var = self.FOR_LOOP_NAME_TEMPLATE.format(node_name=get_node_name())
            yield '{line_prefix}var {loop_var};'.format(line_prefix=line_prefix, loop_var=loop_var)
            yield ('{line_prefix}for({loop_var} = 0; '
                   '{loop_var} < {repetition_count}; '
                   '{loop_var} = {loop_var} + 1) {{').format(line_prefix=line_prefix,
                                                             loop_var=loop_var,
                                                             repetition_count=self.repetition_count)

        if advance_strategy == self._AdvanceStrategy.INITIAL_RESET:
            yield ('{body_prefix}{pos_var_name} = {init_pos_name};'
                   '').format(body_prefix=body_prefix,
                              pos_var_name=pos_var_name,
                              init_pos_name=initial_position_name)
        yield from self.scope.to_source_code(waveform_manager,
                                             line_prefix=body_prefix, pos_var_name=pos_var_name,
                                             node_name_generator=node_name_generator,
                                             advance_pos_var=inner_advance_pos_var)
        yield '{line_prefix}}}'.format(line_prefix=line_prefix)

        if advance_strategy == self._AdvanceStrategy.POST_ADVANCE:
            yield '{line_prefix}{pos_var_name} = {pos_var_name} + {samples};'.format(line_prefix=line_prefix,
                                                                                     pos_var_name=pos_var_name,
                                                                                     samples=self.samples())


class SteppingRepeat(SEQCNode):
    STEPPING_REPEAT_COMMENT = ' // stepping repeat'
    __slots__ = ('node_cluster',)

    def __init__(self, node_cluster: Sequence[SEQCNode]):
        self.node_cluster = node_cluster

    def samples(self) -> int:
        return self.repetition_count * self.node_cluster[0].samples()

    @property
    def repetition_count(self):
        return len(self.node_cluster)

    def iter_waveform_playbacks(self) -> Iterator['WaveformPlayback']:
        for node in self.node_cluster:
            yield from node.iter_waveform_playbacks()

    def stepping_hash(self) -> int:
        return hash((type(self), self.node_cluster[0].stepping_hash()))

    def same_stepping(self, other: 'SteppingRepeat'):
        return (type(other) is SteppingRepeat and
                len(self.node_cluster) == len(other.node_cluster) and
                self.node_cluster[0].same_stepping(other.node_cluster[0]))

    def _visit_nodes(self, waveform_manager: ProgramWaveformManager):
        for node in self.node_cluster:
            node._visit_nodes(waveform_manager)

    def to_source_code(self, waveform_manager: ProgramWaveformManager, node_name_generator: Iterator[str],
                       line_prefix: str, pos_var_name: str,
                       advance_pos_var: bool = True):
        body_prefix = line_prefix + self.INDENTATION
        repeat_open = '{line_prefix}repeat({repetition_count}) {{' + self.STEPPING_REPEAT_COMMENT
        yield repeat_open.format(line_prefix=line_prefix,
                                 repetition_count=self.repetition_count)
        yield from self.node_cluster[0].to_source_code(waveform_manager,
                                                       line_prefix=body_prefix, pos_var_name=pos_var_name,
                                                       node_name_generator=node_name_generator,
                                                       advance_pos_var=advance_pos_var)

        # register remaining concatenated waveforms
        for node in itertools.islice(self.node_cluster, 1, None):
            node._visit_nodes(waveform_manager)

        yield '{line_prefix}}}'.format(line_prefix=line_prefix)


class WaveformPlayback(SEQCNode):
    ADVANCE_DISABLED_COMMENT = ' // advance disabled do to parent repetition'
    ENABLE_DYNAMIC_RATE_REDUCTION = False

    __slots__ = ('waveform', 'shared', 'rate')

    def __init__(self, waveform: Tuple[BinaryWaveform, ...], shared: bool = False, rate: int = None):
        assert isinstance(waveform, tuple)
        if self.ENABLE_DYNAMIC_RATE_REDUCTION and rate is None:
            for wf in waveform:
                rate = wf.dynamic_rate(12 if rate is None else rate)
        self.waveform = waveform
        self.shared = shared
        self.rate = rate

    def __repr__(self):
        return f"WaveformPlayback(<{id(self)}>)"

    def samples(self) -> int:
        """Samples consumed in the big concatenated waveform"""
        if self.shared:
            return 0
        else:
            wf_lens = set(map(len, self.waveform))
            assert len(wf_lens) == 1
            wf_len, = wf_lens
            if self.rate is not None:
                wf_len //= (1 << self.rate)
            return wf_len

    def rate_reduced_waveform(self) -> Tuple[BinaryWaveform]:
        if self.rate is None:
            return self.waveform
        else:
            return tuple(BinaryWaveform(wf.data.reshape((-1, 3))[::(1 << self.rate), :].ravel())
                         for wf in self.waveform)

    def stepping_hash(self) -> int:
        if self.shared:
            return hash((type(self), self.waveform))
        else:
            return hash((type(self), self.samples()))

    def same_stepping(self, other: 'WaveformPlayback') -> bool:
        same_type = type(self) is type(other) and self.shared == other.shared
        if self.shared:
            return same_type and self.rate == other.rate and self.waveform == other.waveform
        else:
            return same_type and self.samples() == other.samples()

    def iter_waveform_playbacks(self) -> Iterator['WaveformPlayback']:
        yield self

    def _visit_nodes(self, waveform_manager: ProgramWaveformManager):
        if not self.shared:
            waveform_manager.request_concatenated(self.rate_reduced_waveform())

    def to_source_code(self, waveform_manager: ProgramWaveformManager,
                       node_name_generator: Iterator[str], line_prefix: str, pos_var_name: str,
                       advance_pos_var: bool = True):
        rate_adjustment = "" if self.rate is None else f", {self.rate}"
        if self.shared:
            yield f'{line_prefix}playWave(' \
                  f'{waveform_manager.request_shared(self.rate_reduced_waveform())}' \
                  f'{rate_adjustment});'
        else:
            wf_name = waveform_manager.request_concatenated(self.rate_reduced_waveform())
            wf_len = self.samples()
            play_cmd = f'{line_prefix}playWaveIndexed({wf_name}, {pos_var_name}, {wf_len}{rate_adjustment});'

            if advance_pos_var:
                advance_cmd = f' {pos_var_name} = {pos_var_name} + {wf_len};'
            else:
                advance_cmd = self.ADVANCE_DISABLED_COMMENT
            yield play_cmd + advance_cmd


_PROGRAM_SELECTION_BLOCK = """\
while (true) {{
  // read program selection value
  prog_sel = getUserReg(PROG_SEL_REGISTER);
  
  // calculate value to write back to PROG_SEL_REGISTER
  new_prog_sel = prog_sel | playback_finished;
  if (!(prog_sel & NO_RESET_MASK)) new_prog_sel &= INVERTED_PROG_SEL_MASK;
  setUserReg(PROG_SEL_REGISTER, new_prog_sel);
  
  // reset playback flag
  playback_finished = 0;
  
  // only use part of prog sel that does not mean other things to select the program.
  prog_sel &= PROG_SEL_MASK;
  
  switch (prog_sel) {{
{program_cases}
    default:
      wait(IDLE_WAIT_CYCLES);
  }}
}}"""

_PROGRAM_SELECTION_CASE = """\
    case {selection_index}:
      {program_function_name}();
      waitWave();
      playback_finished = PLAYBACK_FINISHED_MASK;"""


def _make_program_selection_block(programs: Iterable[Tuple[int, str]]):
    program_cases = []
    for selection_index, program_function_name in programs:
        program_cases.append(_PROGRAM_SELECTION_CASE.format(selection_index=selection_index,
                                                            program_function_name=program_function_name))
    return _PROGRAM_SELECTION_BLOCK.format(program_cases="\n".join(program_cases))
