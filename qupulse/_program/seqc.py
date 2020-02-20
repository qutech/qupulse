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

from typing import Optional, Union, Sequence, Dict, Iterator, Tuple, Callable, NamedTuple, MutableMapping, Mapping,\
    Iterable, Any
from types import MappingProxyType
import abc
import itertools
import inspect
import glob
import os.path
import hashlib
from collections import OrderedDict
import string
import warnings
import numbers

import numpy as np
from pathlib import Path

from qupulse.utils.types import ChannelID, TimeType
from qupulse.utils import replace_multiple
from qupulse._program.waveforms import Waveform
from qupulse._program._loop import Loop
from qupulse._program.volatile import VolatileRepetitionCount, VolatileProperty
from qupulse.hardware.awgs.base import ProgramEntry

try:
    import zhinst.utils
except ImportError:
    zhinst = None


__all__ = ["HDAWGProgramManager"]


class BinaryWaveform:
    """This class represents a sampled waveform in the native HDAWG format as returned
    by zhinst.utils.convert_awg_waveform.

    BinaryWaveform.data can be uploaded directly to {device]/awgs/{awg}/waveform/waves/{wf}

    `to_csv_compatible_table` can be used to create a compatible compact csv file (with marker data included)
    """
    __slots__ = ('data',)

    def __init__(self, data: np.ndarray):
        """ always use both channels?

        Args:
            data: data as returned from zhinst.utils.convert_awg_waveform
        """
        n_quantum, remainder = divmod(data.size, 3 * 16)
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
        all_input = (ch1, ch2, *markers)
        assert any(x is not None for x in all_input)
        size = {x.size for x in all_input if x is not None}
        assert len(size) == 1, "Inputs have incompatible dimension"
        size, = size
        if ch1 is None:
            ch1 = np.zeros(size)
        if ch2 is None:
            ch2 = np.zeros(size)
        marker_data = np.zeros(size, dtype=np.uint16)
        for idx, marker in enumerate(markers):
            if marker is not None:
                marker_data += np.uint16((marker > 0) * 2**idx)
        return cls(zhinst.utils.convert_awg_waveform(ch1, ch2, marker_data))

    def __len__(self):
        return self.data.size // 3

    def __eq__(self, other):
        return np.array_equal(self.data, other.data)

    def __hash__(self):
        return hash(bytes(self.data))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def to_csv_compatible_table(self):
        """The integer values in that file should be 18-bit unsigned integers with the two least significant bits
        being the markers. The values are mapped to 0 => -FS, 262143 => +FS, with FS equal to the full scale.

        >>> np.savetxt(waveform_dir, binary_waveform.to_csv_compatible_table(), fmt='%u')
        """
        table = np.zeros((len(self), 2), dtype=np.uint32)
        table[:, 0] = self.ch1
        table[:, 1] = self.ch2
        np.left_shift(table, 2, out=table)
        table[:, 0] += self.markers_ch1
        table[:, 1] += self.markers_ch2

        return table


class ConcatenatedWaveform:
    """Handle the concatenation of multiple binary waveforms to create a big indexable waveform."""
    def __init__(self):
        self._concatenated = []
        self._as_binary = None

    def __bool__(self):
        return bool(self._concatenated)

    def is_finalized(self):
        return self._as_binary is not None or self._concatenated is None

    def as_binary(self) -> Optional[BinaryWaveform]:
        assert self.is_finalized()
        return self._as_binary

    def append(self, binary_waveform):
        assert not self.is_finalized()
        self._concatenated.append(binary_waveform)

    def finalize(self):
        assert not self.is_finalized()
        if self._concatenated:
            concatenated_data = np.concatenate([wf.data for wf in self._concatenated])
            self._as_binary = BinaryWaveform(concatenated_data)
        else:
            self._concatenated = None

    def clear(self):
        if self._concatenated is None:
            self._concatenated = []
        else:
            self._concatenated.clear()
        self._as_binary = None


class WaveformMemory:
    """Global waveform "memory" representation (currently the file system)"""
    CONCATENATED_WAVEFORM_TEMPLATE = '{program_name}_concatenated_waveform'
    SHARED_WAVEFORM_TEMPLATE = '{program_name}_shared_waveform_{hash}'
    WF_PLACEHOLDER_TEMPLATE = '*{id}*'
    FILE_NAME_TEMPLATE = '{hash}.csv'

    _WaveInfo = NamedTuple('_WaveInfo', [('wave_name', str),
                                         ('wave_placeholder', str),
                                         ('file_name', str),
                                         ('binary_waveform', BinaryWaveform)])

    def __init__(self):
        self.shared_waveforms = OrderedDict()  # type: MutableMapping[BinaryWaveform, set]
        self.concatenated_waveforms = OrderedDict()  # type: MutableMapping[str, ConcatenatedWaveform]

    def clear(self):
        self.shared_waveforms.clear()
        self.concatenated_waveforms.clear()

    def _shared_waveforms_iter(self) -> Iterator[_WaveInfo]:
        for wf, program_set in self.shared_waveforms.items():
            if program_set:
                wave_hash = wf.fingerprint()
                wave_name = self.SHARED_WAVEFORM_TEMPLATE.format(program_name='_'.join(program_set),
                                                                 hash=wave_hash)
                wave_placeholder = self.WF_PLACEHOLDER_TEMPLATE.format(id=id(program_set))
                file_name = self.FILE_NAME_TEMPLATE.format(hash=wave_hash)
                yield self._WaveInfo(wave_name, wave_placeholder, file_name, wf)

    def _concatenated_waveforms_iter(self) -> Iterator[_WaveInfo]:
        for program_name, concatenated_waveform in self.concatenated_waveforms.items():
            if concatenated_waveform:
                wave_hash = concatenated_waveform.as_binary().fingerprint()
                wave_placeholder = self.WF_PLACEHOLDER_TEMPLATE.format(id=id(concatenated_waveform))
                wave_name = self.CONCATENATED_WAVEFORM_TEMPLATE.format(program_name=program_name)
                file_name = self.FILE_NAME_TEMPLATE.format(hash=wave_hash)
                yield self._WaveInfo(wave_name, wave_placeholder, file_name, concatenated_waveform.as_binary())

    def waveform_name_replacements(self) -> Dict[str, str]:
        """replace place holders of complete seqc program with

        >>> waveform_name_translation = waveform_memory.waveform_name_replacements()
        >>> seqc_program = qupulse.utils.replace_multiple(seqc_program, waveform_name_translation)
        """
        translation = {}
        for wave_info in self._shared_waveforms_iter():
            translation[wave_info.wave_placeholder] = wave_info.wave_name

        for wave_info in self._concatenated_waveforms_iter():
            translation[wave_info.wave_placeholder] = wave_info.wave_name
        return translation

    def waveform_declaration(self) -> str:
        """Produces a string that declares all needed waveforms.
        It is needed to know the waveform index in case we want to update a waveform during playback."""
        declarations = []
        for wave_info in self._concatenated_waveforms_iter():
            declarations.append(
                'wave {wave_name} = "{file_name}";'.format(wave_name=wave_info.wave_name,
                                                           file_name=wave_info.file_name.replace('.csv', ''))
            )

        for wave_info in self._shared_waveforms_iter():
            declarations.append(
                'wave {wave_name} = "{file_name}";'.format(wave_name=wave_info.wave_name,
                                                           file_name=wave_info.file_name.replace('.csv', ''))
            )
        return '\n'.join(declarations)

    def sync_to_file_system(self, path: Path, delete=True, write_all=False):
        to_save = {path.joinpath(wave_info.file_name): wave_info.binary_waveform
                   for wave_info in itertools.chain(self._concatenated_waveforms_iter(),
                                                    self._shared_waveforms_iter())}

        for file_name in glob.glob(os.path.join(path, '*.csv')):
            if file_name in to_save:
                if not write_all:
                    to_save.pop(file_name)
            elif delete:
                try:
                    os.remove(file_name)
                except OSError:
                    # TODO: log
                    pass

        for file_name, binary_waveform in to_save.items():
            table = binary_waveform.to_csv_compatible_table()
            np.savetxt(file_name, table, '%u')


class ProgramWaveformManager:
    """Manages waveforms of a program"""
    def __init__(self, name, memory: WaveformMemory):
        self._program_name = name
        self._memory = memory

        assert self._program_name not in self._memory.concatenated_waveforms
        assert all(self._program_name not in programs for programs in self._memory.shared_waveforms.values())
        self._memory.concatenated_waveforms[self._program_name] = ConcatenatedWaveform()

    @property
    def program_name(self) -> str:
        return self._program_name

    def clear_requested(self):
        for programs in self._memory.shared_waveforms.values():
            programs.discard(self._program_name)
        self._memory.concatenated_waveforms[self._program_name].clear()

    def request_shared(self, binary_waveform: BinaryWaveform) -> str:
        """Register waveform if not already registered and return a unique identifier placeholder.

        The unique identifier currently is computed from the id of the set which stores all programs using this
        waveform.
        """
        program_set = self._memory.shared_waveforms.setdefault(binary_waveform, set())
        program_set.add(self._program_name)
        return self._memory.WF_PLACEHOLDER_TEMPLATE.format(id=id(program_set))

    def request_concatenated(self, binary_waveform: BinaryWaveform) -> str:
        """Append the waveform to the concatenated waveform"""
        bin_wf_list = self._memory.concatenated_waveforms[self._program_name]
        bin_wf_list.append(binary_waveform)
        return self._memory.WF_PLACEHOLDER_TEMPLATE.format(id=id(bin_wf_list))

    def finalize(self):
        self._memory.concatenated_waveforms[self._program_name].finalize()

    def prepare_delete(self):
        """Delete all references in waveform memory to this program. Cannot be used afterwards."""
        self.clear_requested()
        del self._memory.concatenated_waveforms[self._program_name]


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
                 channels: Tuple[Optional[ChannelID], Optional[ChannelID]],
                 markers: Tuple[Optional[ChannelID], Optional[ChannelID], Optional[ChannelID], Optional[ChannelID]],
                 amplitudes: Tuple[float, float],
                 offsets: Tuple[float, float],
                 voltage_transformations: Tuple[Optional[Callable], Optional[Callable]],
                 sample_rate: TimeType):
        super().__init__(loop, channels=channels, markers=markers,
                         amplitudes=amplitudes,
                         offsets=offsets,
                         voltage_transformations=voltage_transformations,
                         sample_rate=sample_rate)
        for waveform, (sampled_channels, sampled_markers) in self._waveforms.items():
            self._waveforms[waveform] = BinaryWaveform.from_sampled(*sampled_channels, sampled_markers)

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

    def get_binary_waveform(self, waveform: Waveform) -> BinaryWaveform:
        return self._waveforms[waveform]

    def prepare_delete(self):
        """Delete all references to this program. Cannot be used afterwards"""
        self._waveform_manager.prepare_delete()
        self._seqc_node = None
        self._seqc_source = None


class HDAWGProgramManager:
    """This class contains everything that is needed to create the final seqc program and provides an interface to write
    the required waveforms to the file system. It does not talk to the device."""

    GLOBAL_CONSTS = dict(PROG_SEL_REGISTER=UserRegister(zero_based_value=0),
                         TRIGGER_REGISTER=UserRegister(zero_based_value=1),
                         TRIGGER_RESET_MASK=bin(1 << 15),
                         PROG_SEL_NONE=0,
                         NO_RESET_MASK=bin(1 << 15),
                         PROG_SEL_MASK=bin((1 << 15) - 1),
                         IDLE_WAIT_CYCLES=300)
    PROGRAM_FUNCTION_NAME_TEMPLATE = '{program_name}_function'
    INIT_PROGRAM_SWITCH = '// INIT program switch.\nvar prog_sel = 0;'
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

    def __init__(self):
        self._waveform_memory = WaveformMemory()
        self._programs = OrderedDict()  # type: MutableMapping[str, HDAWGProgramEntry]

    @property
    def waveform_memory(self):
        return self._waveform_memory

    def _get_low_unused_index(self):
        existing = {entry.selection_index for entry in self._programs.values()}
        for idx in itertools.count():
            if idx not in existing and idx != self.GLOBAL_CONSTS['PROG_SEL_NONE']:
                return idx

    def add_program(self, name: str, loop: Loop,
                    channels: Tuple[Optional[ChannelID], Optional[ChannelID]],
                    markers: Tuple[Optional[ChannelID], Optional[ChannelID], Optional[ChannelID], Optional[ChannelID]],
                    amplitudes: Tuple[float, float],
                    offsets: Tuple[float, float],
                    voltage_transformations: Tuple[Optional[Callable], Optional[Callable]],
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

        # TODO: de-hardcode these parameters and put compilation in seperate function
        program_entry.compile(20, 1000, '  ', self.WAIT_FOR_SOFTWARE_TRIGGER,
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

    def to_seqc_program(self) -> str:
        lines = []
        for const_name, const_val in self.GLOBAL_CONSTS.items():
            if isinstance(const_val, (int, str)):
                const_repr = str(const_val)
            else:
                const_repr = const_val.to_seqc()
            lines.append('const {const_name} = {const_repr};'.format(const_name=const_name, const_repr=const_repr))

        lines.append(self._waveform_memory.waveform_declaration())

        lines.append('\n//function used by manually triggered programs')
        lines.append(self.SOFTWARE_WAIT_FOR_TRIGGER_FUNCTION_DEFINITION)

        replacements = self._waveform_memory.waveform_name_replacements()

        lines.append('\n// program definitions')
        for program_name, program in self.programs.items():
            program_function_name = self.PROGRAM_FUNCTION_NAME_TEMPLATE.format(program_name=program_name)
            lines.append('void {program_function_name}() {{'.format(program_function_name=program_function_name))
            lines.append(replace_multiple(program.seqc_source, replacements))
            lines.append('}\n')

        lines.append(self.INIT_PROGRAM_SWITCH)

        lines.append('\n//runtime block')
        lines.append('while (true) {')
        lines.append('  // read program selection value')
        lines.append('  prog_sel = getUserReg(PROG_SEL_REGISTER);')
        lines.append('  if (!(prog_sel & NO_RESET_MASK))  setUserReg(PROG_SEL_REGISTER, 0);')
        lines.append('  prog_sel = prog_sel & PROG_SEL_MASK;')
        lines.append('  ')
        lines.append('  switch (prog_sel) {')

        for program_name, program_entry in self.programs.items():
            program_function_name = self.PROGRAM_FUNCTION_NAME_TEMPLATE.format(program_name=program_name)
            lines.append('    case {selection_index}:'.format(selection_index=program_entry.selection_index))
            lines.append('      {program_function_name}();'.format(program_function_name=program_function_name))
            lines.append('      waitWave();')

        lines.append('    default:')
        lines.append('      wait(IDLE_WAIT_CYCLES);')
        lines.append('  }')
        lines.append('}')

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


def to_node_clusters(loop: Union[Sequence[Loop], Loop], loop_to_seqc_kwargs: dict) -> Sequence[Sequence['SEQCNode']]:
    """transform to seqc recursively noes and cluster them if they have compatible stepping"""
    assert len(loop) > 1

    node_clusters = []
    last_node = loop_to_seqc(loop[0], **loop_to_seqc_kwargs)
    current_nodes = [last_node]

    # compress all nodes in clusters of the same stepping
    for child in itertools.islice(loop, 1, None):
        current_node = loop_to_seqc(child, **loop_to_seqc_kwargs)

        if last_node.same_stepping(current_node):
            current_nodes.append(current_node)
        else:
            node_clusters.append(current_nodes)
            current_nodes = [current_node]

        last_node = current_node
    node_clusters.append(current_nodes)
    return node_clusters


def loop_to_seqc(loop: Loop,
                 min_repetitions_for_for_loop: int,
                 min_repetitions_for_shared_wf: int,
                 waveform_to_bin: Callable[[Waveform], BinaryWaveform],
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

    __slots__ = ('waveform', 'shared')

    def __init__(self, waveform: BinaryWaveform, shared: bool = False):
        self.waveform = waveform
        self.shared = shared

    def samples(self):
        if self.shared:
            return 0
        else:
            return len(self.waveform)

    def same_stepping(self, other: 'WaveformPlayback'):
        same_type = type(self) is type(other) and self.shared == other.shared
        if self.shared:
            return same_type and self.waveform == other.waveform
        else:
            return same_type and len(self.waveform) == len(other.waveform)

    def iter_waveform_playbacks(self) -> Iterator['WaveformPlayback']:
        yield self

    def _visit_nodes(self, waveform_manager: ProgramWaveformManager):
        if not self.shared:
            waveform_manager.request_concatenated(self.waveform)

    def to_source_code(self, waveform_manager: ProgramWaveformManager,
                       node_name_generator: Iterator[str], line_prefix: str, pos_var_name: str,
                       advance_pos_var: bool = True):
        if self.shared:
            yield '{line_prefix}playWave({waveform});'.format(waveform=waveform_manager.request_shared(self.waveform),
                                                              line_prefix=line_prefix)
        else:
            wf_name = waveform_manager.request_concatenated(self.waveform)
            wf_len = len(self.waveform)
            play_cmd = '{line_prefix}playWaveIndexed({wf_name}, {pos_var_name}, {wf_len});'

            if advance_pos_var:
                advance_cmd = ' {pos_var_name} = {pos_var_name} + {wf_len};'
            else:
                advance_cmd = self.ADVANCE_DISABLED_COMMENT
            yield (play_cmd + advance_cmd).format(wf_name=wf_name,
                                                  wf_len=wf_len,
                                                  pos_var_name=pos_var_name,
                                                  line_prefix=line_prefix)
