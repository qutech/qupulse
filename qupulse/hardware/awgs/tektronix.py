from typing import Tuple, Callable, Optional, Sequence, Union, Dict, Mapping, Set
from types import MappingProxyType
import numpy as np
import contextlib
import itertools
import functools
import warnings
import logging

try:
    import TekAwg
except ImportError:
    warnings.warn("Could not import Tektronix driver backend. "
                  "If you wish to use it execute qctoolkit.hardware.awgs.install_requirements('tektronix')")
    raise

from qctoolkit.hardware.awgs.base import AWG
from qctoolkit import ChannelID
from qctoolkit._program._loop import Loop, make_compatible
from qctoolkit.utils.types import TimeType
from qctoolkit.hardware.util import voltage_to_uint16
from qctoolkit.utils import pairwise


__all__ = ['TektronixAWG']


class WaveformEntry:
    def __init__(self, name: str, length: int, waveform: TekAwg.Waveform, timestamp):
        self.name = name
        self.waveform = waveform
        self.length = length
        self.timestamp = timestamp


class WaveformStorage:
    def __init__(self, waveforms: Sequence[WaveformEntry]=()):
        self._waveforms_by_name = {wf.name: wf for wf in waveforms}
        self._by_name = MappingProxyType(self._waveforms_by_name)

        self._waveforms_by_data = {wf.waveform: wf for wf in waveforms}
        self._by_data = MappingProxyType(self._waveforms_by_data)

    @property
    def by_name(self) -> Mapping[str, WaveformEntry]:
        return self._by_name

    @property
    def by_data(self) -> Mapping[TekAwg.Waveform, WaveformEntry]:
        return self._by_data

    def __iter__(self):
        return iter(self._waveforms_by_data.values())

    def __len__(self):
        return len(self._waveforms_by_data)

    def add_waveform(self, name: str, waveform_entry: WaveformEntry, overwrite: bool=False):
        with contextlib.ExitStack() as exit_stack:
            if name in self._by_name:
                if overwrite:
                    # re-adding old entry in case of failure
                    exit_stack.callback(functools.partial(self.add_waveform, name, self.pop_waveform(name)))
                else:
                    raise RuntimeError('Waveform already existing', name)

            self._waveforms_by_data[waveform_entry.waveform] = waveform_entry
            self._waveforms_by_name[name] = waveform_entry

            # remove cleanup
            exit_stack.pop_all()

    def pop_waveform(self, name: str) -> WaveformEntry:
        wf = self._waveforms_by_name.pop(name)
        del self._waveforms_by_data[wf.waveform]
        return wf


def parse_program(program: Loop,
                  channels: Tuple[ChannelID, ...],
                  markers: Tuple[Tuple[ChannelID, ChannelID], ...],
                  sample_rate: TimeType,
                  amplitudes: Tuple[float, ...],
                  voltage_transformations: Tuple[Callable, ...]) -> Tuple[Sequence[TekAwg.SequenceEntry],
                                                                          Sequence[TekAwg.Waveform]]:
    """Convert the program into a sequence of sequence table entries and a sequence of waveforms that can be uploaded
    to the device."""
    assert program.depth() == 1, "Invalid program depth."

    sequencing_elements = []

    ch_waveforms = {}
    bin_waveforms = {}

    sample_rate_in_GHz = sample_rate / 10**9

    time_per_sample = float(1 / sample_rate_in_GHz)

    def make_binary_converter(amp, trafo):
        def to_uint16(voltage):
            return voltage_to_uint16(trafo(voltage),
                                     output_amplitude=amp,
                                     output_offset=0., resolution=14)

        return to_uint16
    longest_waveform_n_samples = int(max(loop.waveform.duration for loop in program) * sample_rate_in_GHz)
    time_array = np.arange(longest_waveform_n_samples) * time_per_sample

    binary_converters = [make_binary_converter(amplitude, voltage_transformation)
                         for amplitude, voltage_transformation in zip(amplitudes, voltage_transformations)]

    for loop in program:
        n_samples = int(loop.waveform.duration * sample_rate_in_GHz)

        entries = []

        for channel, (marker_1, marker_2), binary_converter in zip(channels, markers, binary_converters):
            chs = {channel, marker_1, marker_2} - {None}

            if not chs:
                entries.append(n_samples)
                bin_waveforms[n_samples] = None

            else:
                ch_waveform = loop.waveform.get_subset_for_channels(chs)

                if ch_waveform not in ch_waveforms:
                    if channel is None:
                        channel_volts = 0.
                    else:
                        channel_volts = loop.waveform.get_sampled(channel=channel,
                                                                  sample_times=time_array[:n_samples])
                    channel_data = binary_converter(channel_volts)

                    if marker_1 is None:
                        marker_1_data = 0
                    else:
                        marker_1_data = loop.waveform.get_sampled(channel=marker_1,
                                                                  sample_times=time_array[:n_samples])

                    if marker_2 is None:
                        marker_2_data = 0
                    else:
                        marker_2_data = loop.waveform.get_sampled(channel=marker_2,
                                                                  sample_times=time_array[:n_samples])

                    bin_waveform = TekAwg.Waveform(channel=channel_data,
                                                   marker_1=marker_1_data,
                                                   marker_2=marker_2_data)
                    if bin_waveform in bin_waveforms:
                        bin_waveform = ch_waveforms[bin_waveforms[bin_waveform]]
                    else:
                        bin_waveforms[bin_waveform] = ch_waveform

                    ch_waveforms[ch_waveform] = bin_waveform

                entries.append(ch_waveforms[ch_waveform])

        sequencing_elements.append(
            TekAwg.SequenceEntry(entries=entries,
                                 loop_count=loop.repetition_count)
        )
    return tuple(sequencing_elements), tuple(bin_waveforms.keys())


class TektronixProgram:
    """Bundles all information used to generate the sequence table entries and waveforms."""

    def __init__(self,
                 program: Loop,
                 channels: Sequence[ChannelID],
                 markers: Sequence[Tuple[ChannelID, ChannelID]],
                 sample_rate: TimeType,
                 amplitudes: Sequence[float],
                 voltage_transformations: Sequence[Callable]):
        assert len(channels) == len(markers) and all(len(marker) == 2 for marker in markers), "Driver can currently only handle awgs wth two markers per channel"

        assert len(channels) == len(amplitudes)

        self._program = program.copy_tree_structure()
        self._sample_rate = sample_rate
        self._amplitudes = tuple(amplitudes)
        self._channels = tuple(channels)
        self._markers = tuple(markers)
        self._voltage_transformations = tuple(voltage_transformations)

        self._sequencing_elements = None
        self._waveforms = None

        make_compatible(program, 250, 1, sample_rate / 10**9)
        self._program.flatten_and_balance(1)

        self._sequencing_elements, self._waveforms = parse_program(program=self._program,
                                                                   channels=self.channels,
                                                                   markers=self.markers,
                                                                   sample_rate=self._sample_rate,
                                                                   amplitudes=self._amplitudes,
                                                                   voltage_transformations=self._voltage_transformations)

    def get_sequencing_elements(self) -> Sequence[TekAwg.SequenceEntry]:
        """The entries are either of type TekAwh.Waveform or integers which signal an idle waveform of this length"""
        return self._sequencing_elements

    def get_waveforms(self) -> Sequence[Union[TekAwg.Waveform, int]]:
        """Integers denote idle waveforms of this length"""
        return self._waveforms

    @property
    def sample_rate(self) -> TimeType:
        return self._sample_rate

    @property
    def channels(self) -> Tuple[ChannelID]:
        return self._channels

    @property
    def markers(self) -> Tuple[Tuple[ChannelID, ChannelID]]:
        return self._markers

    @property
    def amplitudes(self) -> Tuple[float]:
        return self._amplitudes


class TektronixAWG(AWG):
    """TODO: Explain general idea here"""

    def __init__(self, tek_awg: TekAwg.TekAwg,
                 synchronize: str,
                 identifier='Tektronix',
                 manual_cleanup=False):
        """
        Args:
            tek_awg: Instance of the underlying driver from TekAwg package
            synchronize: Either 'read' or 'clear'.
            identifier:
            manual_cleanup:
        """
        super().__init__(identifier=identifier)

        if TekAwg is None:
            raise RuntimeError('Please install the TekAwg package or run "install_requirements" from this module')

        self._device = tek_awg

        self.idle_value = 8191
        self._idle_waveform = self.make_idle_waveform(4000)

        self._programs = dict()

        self._idle_program_index = None

        self._armed_program = None, None

        self._sequence_entries = []

        self._waveforms = WaveformStorage()

        self._cleanup_stack = contextlib.ExitStack() if manual_cleanup else None

        if synchronize.lower() == 'read':
            self.synchronize()
        elif synchronize.lower() == 'clear':
            self.clear()
        else:
            raise ValueError('synchronize must be one of ["clear", "read"]')

        self.initialize_idle_program()

    def _clear_waveforms(self):
        """Clear all waveforms on the device and synchronize the waveform list."""
        self.device.write('WLIS:WAV:DEL ALL')
        self.read_waveforms()

    def _clear_sequence(self):
        """Clear sequence on device and synchronize sequence entries."""
        self.device.write('SEQ:LENG 0')
        self.read_sequence()

    def clear(self):
        """Clear all waveforms and sequence and synchronize state"""
        self._clear_sequence()
        self._clear_waveforms()
        self._programs = dict()
        self._armed_program = None
        self._idle_program_index = None

    def synchronize(self):
        """Read waveforms and sequences from device"""
        self.read_waveforms()
        self.read_sequence()

    @property
    def armed_program(self) -> Optional[str]:
        return self._armed_program[0]

    def initialize_idle_program(self):
        """Make sure we can arm the idle program which plays the idle waveform(default 0V) on all channels.


        """
        if self._idle_waveform in self._waveforms.by_data:
            idle_waveform_name = self._waveforms.by_data[self._idle_waveform].name
        else:
            idle_waveform_name = self.idle_pulse_name(self._idle_waveform.size)
            self._upload_waveform(self._idle_waveform, idle_waveform_name)

        idle_sequence_element = TekAwg.SequenceEntry(entries=[idle_waveform_name] * self.device.n_channels,
                                                     wait=False,
                                                     loop_inf=True,
                                                     loop_count=None,
                                                     goto_ind=None,
                                                     goto_state=False,
                                                     jmp_type='OFF',
                                                     jmp_ind=None)
        try:
            self._idle_program_index = self._sequence_entries.index(idle_sequence_element) + 1
        except ValueError:
            idle_index, *_ = self._get_empty_sequence_positions(1)
            self._upload_sequencing_element(idle_index, idle_sequence_element)

            self._idle_program_index = self._sequence_entries.index(idle_sequence_element) + 1

    @property
    def device(self) -> TekAwg.TekAwg:
        return self._device

    def read_waveforms(self):
        """Read all waveform data from the device (including binary data)."""
        wf_names = self.device.get_waveform_names()
        wf_times = self.device.get_waveform_timestamps(wf_names)
        wf_lengths = self.device.get_waveform_lengths(wf_names)
        wf_datas = [self.device.get_waveform_data(wf_name)
                    for wf_name in wf_names]

        waveforms = [WaveformEntry(name=name, length=length, timestamp=time, waveform=data)
                     for name, length, time, data in zip(wf_names, wf_lengths, wf_times, wf_datas)]

        self._waveforms = WaveformStorage(waveforms)

    def read_sequence(self):
        """Read all sequence data from the device"""
        entries = [self.device.get_seq_element(i)
                   for i in range(1, 1 + self.device.get_seq_length())]
        entries = [None if all(wf == '' for wf in entry.entries) else entry
                   for entry in entries]
        self._sequence_entries = entries

    @property
    def num_channels(self) -> int:
        return self.device.n_channels

    @property
    def num_markers(self) -> int:
        return self.num_channels * 2

    @property
    def programs(self) -> Set[str]:
        return set(self._programs.keys())

    def cleanup(self):
        """Delete all waveforms not used anymore and rewrite sequence entries if they are fragmented"""

        used_waveforms = set()
        if self._idle_waveform in self._waveforms.by_data:
            used_waveforms.add(self._waveforms.by_data[self._idle_waveform].name)

        programs = self._programs.copy()
        self._clear_sequence()

        self.initialize_idle_program()

        for name, (_, tek_program, sequencing_elements) in programs.items():
            used_waveforms.update(itertools.chain.from_iterable(element.entries for element in sequencing_elements))
            self._upload_parsed(name, tek_program)

        for name in set(self._waveforms.by_name.keys()) - used_waveforms:
            self._delete_waveform(name)

    def remove(self, name: str):
        raise NotImplementedError()

    @property
    def sample_rate(self) -> float:
        return self.device.get_freq()

    def _set_state(self, sequence_entries, waveforms, waveform_references):
        raise NotImplementedError()

    def upload(self, *args, **kwargs):
        if self._cleanup_stack is None:
            with contextlib.ExitStack() as auto_cleanup:
                self._upload(*args, **kwargs, cleanup_stack=auto_cleanup)
                auto_cleanup.pop_all()
        else:
            cleanup_stack = contextlib.ExitStack()
            self._cleanup_stack.push(cleanup_stack)
            self._upload(*args, **kwargs, cleanup_stack=cleanup_stack)
            cleanup_stack.pop_all()

    def _process_program(self, name: str, tek_program: TektronixProgram) -> Tuple[Sequence[TekAwg.SequenceEntry],
                                                                                  Mapping[TekAwg.Waveform, str]]:
        """Detect which waveforms are missing and create sequencing entries.
        This function does not change the state of the device.

        Args:
            name:
            tek_program:

        Returns:
            sequencing_elements: List of SequenceEntries
            waveforms_to_upload: Missing waveforms
        """
        waveforms_to_upload = dict()
        required_idle_pulses = dict()

        sequencing_elements = []
        for entries, *sequencing_info in tek_program.get_sequencing_elements():
            new_entries = []

            for entry in entries:
                if isinstance(entry, int):
                    if entry in required_idle_pulses:
                        wf_name = required_idle_pulses[entry]

                    else:
                        wf_name = self.idle_pulse_name(entry)
                        wf_data = self.make_idle_waveform(entry)

                        if wf_data in self._waveforms.by_data:
                            wf_name = self._waveforms.by_data[wf_data].name

                        else:
                            # rename waveform to idle waveform for clarity
                            waveforms_to_upload[wf_data] = wf_name

                        required_idle_pulses[entry] = wf_name

                elif isinstance(entry, TekAwg.Waveform):
                    if entry in self._waveforms.by_data:
                        wf_name = self._waveforms.by_data[entry].name

                    elif entry in waveforms_to_upload:
                        wf_name = waveforms_to_upload[entry]

                    else:
                        wf_name = name + '_' + str(abs(hash(entry)))
                        waveforms_to_upload[entry] = wf_name

                else:
                    # check that we know the waveform
                    wf_name = self._waveforms.by_name[entry].name

                new_entries.append(wf_name)

            sequencing_elements.append(TekAwg.SequenceEntry(new_entries,
                                                            *sequencing_info))
        return sequencing_elements, waveforms_to_upload

    def _upload_parsed(self, name: str, tek_program: TektronixProgram, cleanup_stack: contextlib.ExitStack=None):
        sequencing_elements, waveforms_to_upload = self._process_program(name, tek_program)

        logger = logging.getLogger('tektronix')

        for idx, (waveform_data, waveform_name) in enumerate(waveforms_to_upload.items()):
            self._upload_waveform(waveform_data=waveform_data,
                                  waveform_name=waveform_name,
                                  cleanup_stack=cleanup_stack)
            if idx % 10 == 0:
                logging.debug('Waiting for sync after waveform %d' % idx)
                self.device.wait_until_commands_executed()
                logging.debug('Synced after waveform %d' % idx)

        logger.info('Waiting for all waveforms to be uploaded...')
        self.device.wait_until_commands_executed()
        logger.info('All waveforms uploaded')

        positions = self._get_empty_sequence_positions(len(sequencing_elements))

        for (element_index, next_element), sequencing_element in zip(
                pairwise(positions, fillvalue=self._idle_program_index),
                sequencing_elements):
            assert next_element is not None

            if element_index + 1 != next_element:
                sequencing_element.goto_ind = next_element
                sequencing_element.goto_state = True

            self._upload_sequencing_element(element_index, sequencing_element)

            if element_index % 100 == 0:
                logger.debug('Waiting for sync after element %d' % element_index)
                self.device.wait_until_commands_executed()
                logger.debug('Synced after element %d' % element_index)

        self.device.wait_until_commands_executed()
        logger.info('All sequence elements uploaded')

        self._programs[name] = (positions, tek_program, sequencing_elements)

    def _unload(self, name: str) -> TektronixProgram:
        positions, tek_program, seq_entries = self._programs.pop(name)

        for position in positions:
            if position <= len(self._sequence_entries):
                self._sequence_entries[position - 1] = None

        return tek_program

    def _upload(self, name: str,
                program: Loop,
                channels: Tuple[Optional[ChannelID], ...],
                markers: Tuple[Optional[ChannelID], ...],
                voltage_transformation: Tuple[Callable, ...],
                force: bool,
                cleanup_stack: contextlib.ExitStack):

        assert self._idle_program_index

        if name in self._programs:
            if not force:
                raise ValueError('{} is already known on {}'.format(name, self.identifier))

            else:
                old_program = self._unload(name)
                cleanup_stack.callback(lambda: self._upload_parsed(name, old_program))

        # group markers in by channels
        markers = tuple(zip(markers[0::2], markers[1::2]))

        tek_program = TektronixProgram(program, channels=channels, markers=markers,
                                       amplitudes=self.device.get_amplitude(),
                                       voltage_transformations=voltage_transformation,
                                       sample_rate=TimeType(self.sample_rate))

        self._upload_parsed(name, tek_program, cleanup_stack)

    def _get_empty_sequence_positions(self, length: int) -> Sequence[int]:
        """Return a list of ``length`` empty sequence positions"""
        free_positions = [idx + 1
                          for idx, sequencing_element in enumerate(self._sequence_entries)
                          if sequencing_element is None]
        missing = range(len(self._sequence_entries) + 1,
                        len(self._sequence_entries) + 1 + length - len(free_positions))
        if missing:
            self.device.set_seq_length(len(self._sequence_entries) + len(missing))
            self._sequence_entries.extend(itertools.repeat(None, len(missing)))
            free_positions.extend(missing)
        return free_positions

    def _delete_waveform(self, waveform_name: str):
        self.device.del_waveform(waveform_name)
        if waveform_name in self._waveforms.by_name:
            self._waveforms.pop_waveform(waveform_name)

    def _upload_waveform(self, waveform_data: TekAwg.Waveform, waveform_name, cleanup_stack: contextlib.ExitStack=None):
        self.device.new_waveform(waveform_name, waveform_data)
        if cleanup_stack:
            cleanup_stack.callback(functools.partial(self._delete_waveform, waveform_name))
        timestamp = self.device.get_waveform_timestamps(waveform_name)
        self._waveforms.add_waveform(waveform_name,
                                     waveform_entry=WaveformEntry(name=waveform_name,
                                                                  length=waveform_data.size,
                                                                  waveform=waveform_data,
                                                                  timestamp=timestamp))

    def _upload_sequencing_element(self, element_index, sequencing_element: TekAwg.SequenceEntry):
        self._sequence_entries[element_index - 1] = sequencing_element
        self.device.set_seq_element(element_index, sequencing_element)

    def make_idle_waveform(self, length) -> TekAwg.Waveform:
        return TekAwg.Waveform(channel=np.full(length,
                                               fill_value=self.idle_value,
                                               dtype=np.uint16),
                               marker_1=0, marker_2=0)

    @staticmethod
    def idle_pulse_name(idle_length: int) -> str:
        return 'idle_{}'.format(idle_length)

    def get_program_information(self) -> Dict[str, dict]:
        return {name: {'first entry': positions[0]}
                for name, (_, positions, _) in self._programs.items()}

    def arm(self, name: Optional[str]):
        _, positions, _ = self._programs[name]
        self._armed_program = (name, positions[0])

    def run_current_program(self):
        _, program_index = self._armed_program

        self.device.run()
        self.device.jump_to_sequence_element(program_index)
        self.device.wait_until_commands_executed()
