from typing import Tuple, Callable, Optional, Sequence, Union, Dict, Mapping, Set
from types import MappingProxyType
import numpy as np
import contextlib
import itertools
import functools
import warnings
import logging

try:
    import tek_awg
except ImportError:  # pragma: no cover
    warnings.warn("Could not import Tektronix driver backend. "
                  "If you wish to use it execute qupulse.hardware.awgs.install_requirements('tektronix')")
    raise

from qupulse.hardware.awgs.base import AWG, AWGAmplitudeOffsetHandling, ProgramOverwriteException
from qupulse import ChannelID
from qupulse._program._loop import Loop, make_compatible
from qupulse._program.waveforms import Waveform as QuPulseWaveform
from qupulse.utils.types import TimeType
from qupulse.hardware.util import voltage_to_uint16, get_sample_times, traced
from qupulse.utils import pairwise


__all__ = ['TektronixAWG']


class WaveformEntry:
    def __init__(self, name: str, length: int, waveform: tek_awg.Waveform, timestamp):
        self.name = name
        self.waveform = waveform
        self.length = length
        self.timestamp = timestamp


class WaveformStorage:
    """Consistent map name->WaveformEntry tek_awg.Waveform->WaveformEntry"""
    def __init__(self, waveforms: Sequence[WaveformEntry]=()):
        self._waveforms_by_name = {wf.name: wf for wf in waveforms}
        self._by_name = MappingProxyType(self._waveforms_by_name)

        self._waveforms_by_data = {wf.waveform: wf for wf in waveforms}
        self._by_data = MappingProxyType(self._waveforms_by_data)

    @property
    def by_name(self) -> Mapping[str, WaveformEntry]:
        return self._by_name

    @property
    def by_data(self) -> Mapping[tek_awg.Waveform, WaveformEntry]:
        return self._by_data

    def __iter__(self):
        return iter(self._waveforms_by_data.values())

    def __len__(self):
        return len(self._waveforms_by_data)

    def add_waveform(self, waveform_entry: WaveformEntry, overwrite: bool = False):
        """waveform must not be known by data"""
        assert waveform_entry.waveform not in self.by_data

        if waveform_entry.name in self._by_name:
            if overwrite:
                self.pop_waveform(waveform_entry.name)
            else:
                raise RuntimeError('Waveform already existing', waveform_entry.name)

        self._waveforms_by_data[waveform_entry.waveform] = waveform_entry
        self._waveforms_by_name[waveform_entry.name] = waveform_entry

    def pop_waveform(self, name: str) -> WaveformEntry:
        wf = self._waveforms_by_name.pop(name)
        del self._waveforms_by_data[wf.waveform]
        return wf


def _make_binary_waveform(waveform: QuPulseWaveform,
                          time_array: np.ndarray,
                          channel: Optional[ChannelID],
                          marker_1: Optional[ChannelID],
                          marker_2: Optional[ChannelID],
                          voltage_transformation: Callable,
                          voltage_to_uint16_kwargs: dict) -> tek_awg.Waveform:
    def sample_channel(ch_id: Optional[ChannelID]):
        if ch_id is None:
            return np.zeros_like(time_array)
        else:
            return waveform.get_sampled(channel=ch_id,
                                        sample_times=time_array)

    channel_volts, marker_1_data, marker_2_data = map(sample_channel, (channel, marker_1, marker_2))
    channel_volts = voltage_transformation(channel_volts)
    channel_data = voltage_to_uint16(channel_volts, **voltage_to_uint16_kwargs)

    return tek_awg.Waveform(channel=channel_data,
                            marker_1=marker_1_data,
                            marker_2=marker_2_data)


def parse_program(program: Loop,
                  channels: Tuple[Optional[ChannelID], ...],
                  markers: Tuple[Tuple[Optional[ChannelID], Optional[ChannelID]], ...],
                  sample_rate: TimeType,
                  amplitudes: Tuple[float, ...],
                  voltage_transformations: Tuple[Callable, ...],
                  offsets: Tuple[float, ...] = None) -> Tuple[Sequence[tek_awg.SequenceEntry],
                                                              Sequence[tek_awg.Waveform]]:
    """Convert the program into a sequence of sequence table entries and a sequence of waveforms that can be uploaded
    to the device."""
    assert program.depth() == 1, ("Invalid program depth: %d" % program.depth())
    assert program.repetition_count == 1, ("Cannot repeat program a finite number of times (only once not %d)" %
                                           program.repetition_count)

    # For backward compatibility
    # EDIT: I think this is not needed? (Simon)
    if offsets is None:
        offsets = (0.,) * len(amplitudes)

    assert len(channels) == len(markers) == len(amplitudes) == len(voltage_transformations) == len(offsets)

    sequencing_elements = []

    ch_waveforms = {}
    bin_waveforms = {}

    sample_rate_in_GHz = sample_rate / 10**9

    time_array, n_samples = get_sample_times([loop.waveform for loop in program],
                                             sample_rate_in_GHz=sample_rate_in_GHz)

    channel_wise_kwargs = [dict(voltage_to_uint16_kwargs=dict(output_amplitude=amplitude,
                                                              output_offset=offset,
                                                              resolution=14),
                                voltage_transformation=voltage_trafo)
                           for amplitude, offset, voltage_trafo in zip(amplitudes, offsets, voltage_transformations)]

    # List of Tuple[positional channel tuple, set chs to sample]
    channel_infos = [((channel, marker_1, marker_2), {channel, marker_1, marker_2} - {None})
                     for channel, (marker_1, marker_2) in zip(channels, markers)]

    for n_sample, loop in zip(n_samples, program):

        entries = []

        for (positional_chs, chs_to_sample), kwargs in zip(channel_infos, channel_wise_kwargs):

            if not chs_to_sample:
                entries.append(n_sample)
                bin_waveforms[n_sample] = None

            else:
                ch_waveform = loop.waveform.get_subset_for_channels(chs_to_sample)

                if ch_waveform not in ch_waveforms:
                    bin_waveform = _make_binary_waveform(ch_waveform,
                                                         time_array[:n_sample],
                                                         *positional_chs, **kwargs)

                    if bin_waveform in bin_waveforms:
                        # use identical binary waveform already created to save memory
                        bin_waveform = ch_waveforms[bin_waveforms[bin_waveform]]
                    else:
                        bin_waveforms[bin_waveform] = ch_waveform

                    ch_waveforms[ch_waveform] = bin_waveform

                entries.append(ch_waveforms[ch_waveform])

        sequencing_elements.append(
            tek_awg.SequenceEntry(entries=entries,
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
                 voltage_transformations: Sequence[Callable],
                 offsets: Sequence[float] = None):
        assert len(channels) == len(markers) and all(len(marker) == 2 for marker in markers),\
            "Driver can currently only handle awgs wth two markers per channel"

        assert len(channels) == len(amplitudes)

        self._program = program.copy_tree_structure()
        self._sample_rate = sample_rate
        self._amplitudes = tuple(amplitudes)
        self._offsets = tuple(offsets) if offsets is not None else None
        self._channels = tuple(channels)
        self._markers = tuple(markers)
        self._voltage_transformations = tuple(voltage_transformations)

        self._sequencing_elements = None
        self._waveforms = None

        make_compatible(self._program, 250, 1, sample_rate / 10**9)
        self._program.flatten_and_balance(1)

        self._sequencing_elements, self._waveforms = parse_program(program=self._program,
                                                                   channels=self.channels,
                                                                   markers=self.markers,
                                                                   sample_rate=self._sample_rate,
                                                                   amplitudes=self._amplitudes,
                                                                   voltage_transformations=self._voltage_transformations,
                                                                   offsets=self._offsets)

    def get_sequencing_elements(self) -> Sequence[tek_awg.SequenceEntry]:
        """The entries are either of type TekAwg.Waveform or integers which signal an idle waveform of this length"""
        return self._sequencing_elements

    def get_waveforms(self) -> Sequence[Union[tek_awg.Waveform, int]]:
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


@traced
class TektronixAWG(AWG):
    """Driver for Tektronix AWG object (5000/7000 series).

    Special characteristics:
     - Changing the run mode to 'running' takes a lot of time (depending on the sequence)
     - To keep the "arm" time low each program is uploaded to the sequence table! This reduces the number of programs
       drastically but allows very fast switching with arm IF the awg runs.
     - Arm starts the awg if it does not run. The first call to arm after uploading new programs is therefore slow. This
       guarantees that subsequent calls to arm and run_current_program are always fast.
     - Uploading while the awg runs is VERY slow. The private properties _allow_upload_while_running and
       _auto_stop_on_upload control the behaviour in this case (stopping the awg or raising an exception).
       These properties are currently not a stable interface to that functionality hence the privateness.

    This driver implements an interface for changing the program repetition mode consisting of:
     - the property default_program_repetition_mode
     - set_program_repetition_mode(program_name, mode)
     - get_program_repetition_mode(program_name)

     Synchronization:
     The driver relies on the fact that internal variables correspond to the device state:
     _sequence_entries and _waveforms
     We do not aim to detect user manipulation but want to make sure invalidate the internal state on upload errors.
     To do this the attribute _synchronized is set to False. Methods that rely on the synchronization state to correctly
     modify the awg check that attribute.

     All private functions assume that the error queue is empty when called.

    TODO: Move logger and repetition mode functionality to AWG interface"""

    def __init__(self, device: tek_awg.TekAwg,
                 synchronize: str,
                 identifier='Tektronix',
                 logger=None,
                 default_program_repetition_mode='once',
                 idle_waveform_length=250):
        """
        Args:
            device: Instance of the underlying driver from tek_awg package
            synchronize: Either 'read' or 'clear'.
            identifier: Some identifier
            logger: Logging will happen here (defaults to 'qupulse.tektronix' otherwise)
            default_program_repetition_mode: 'once' or 'infinite'
            idle_waveform_length: length of the idle_waveform in samples
        """
        super().__init__(identifier=identifier)
        self.logger = logger or logging.getLogger("qupulse.tektronix")

        if device is None:
            raise RuntimeError('Please install the tek_awg package or run "install_requirements" from this module')

        self._device = device
        self._synchronized = False # this gets set to True by synchronize or clear and to False on error during manupulation

        self.idle_value = 8191
        self._idle_waveform = self.make_idle_waveform(idle_waveform_length)

        self._programs = dict()

        self._idle_program_index = None

        self._armed_program = None, None

        self._sequence_entries = []

        self._waveforms = WaveformStorage()

        self._sequence_element_upload_sync_interval = 100
        self._waveform_upload_sync_interval = 10

        self._allow_upload_while_running = False
        self._auto_stop_on_upload = True

        self._default_program_repetition_mode = None
        self.default_program_repetition_mode = default_program_repetition_mode

        if synchronize.lower() == 'read':
            self.synchronize()
        elif synchronize.lower() == 'clear':
            self.clear()
        else:
            raise ValueError('synchronize must be one of ["clear", "read"]')

        self.initialize_idle_program()

    def set_volatile_parameters(self, program_name: str, parameters):
        raise NotImplementedError()

    @staticmethod
    def _validate_program_repetition_code(mode: str):
        if mode not in ('once', 'infinite'):
            raise ValueError("Invalid program repetition mode (not 'once' or 'infinite')", mode)

    @property
    def default_program_repetition_mode(self) -> str:
        """repetition mode for newly uploaded programs. Valid values are 'once' and 'infinite'. You can use
        set_program_repetition_mode to change the repetition mode of an existing program"""
        return self._default_program_repetition_mode

    @default_program_repetition_mode.setter
    def default_program_repetition_mode(self, mode: str):
        self._validate_program_repetition_code(mode)
        self._default_program_repetition_mode = mode

    def set_program_repetition_mode(self, program_name: str, mode: str):
        self.assert_synchronized()
        self._validate_program_repetition_code(mode)

        (positions, _, sequencing_elements) = self._programs[program_name]
        if mode == 'infinite':
            last_jump_to = positions[0]
        elif mode == 'once':
            last_jump_to = self._idle_program_index
        assert isinstance(last_jump_to, int)
        self.logger.debug("Setting repetition mode of '%s' to '%s'", program_name, mode)

        self.device.set_seq_element_jmp_ind(positions[-1], last_jump_to)
        sequencing_elements[-1].goto_ind = last_jump_to

    def get_program_repetition_mode(self, program_name: str) -> str:
        """This function uses cached data and does not talk to the awg"""
        (positions, _, sequencing_elements) = self._programs[program_name]
        if sequencing_elements[-1].goto_ind == self._idle_program_index:
            return 'once'
        elif sequencing_elements[-1].goto_ind == positions[0] :
            return 'infinite'
        else:
            self.logger.warning("Could not extract repetition mode of %s. Last element goto index is %r",
                                program_name, sequencing_elements[-1].goto_ind)
            return 'unknown'

    @property
    def synchronized(self) -> bool:
        return self._synchronized

    def assert_synchronized(self):
        if not self.synchronized:
            raise RuntimeError("The drive might be out of sync with the device (probably due to an error "
                               "during some operation). Call synchronize() to fix that. This is not done automatically "
                               "because you maybe know how to recover cleverly yourself.")

    def _clear_waveforms(self):
        """Clear all waveforms on the device and synchronize the waveform list."""
        self.device.write('WLIS:WAV:DEL ALL')
        self.read_waveforms()

    def _clear_sequence(self):
        """Clear sequence on device and synchronize sequence entries."""
        self.device.write('SEQ:LENG 0')
        self._sequence_entries = [None] * self.device.get_seq_length()

    def clear(self):
        """Clear all waveforms, the sequence table and program registry and initialize the idle program."""
        self._synchronized = False
        self._clear_sequence()
        self._clear_waveforms()
        self._programs = dict()
        self._armed_program = None
        self._idle_program_index = None
        self._synchronized = True
        self.initialize_idle_program()

    def synchronize(self):
        """Read waveforms from device and re-upload all programs"""
        self.read_waveforms()

        to_upload = {program_name: (tek_program, self.get_program_repetition_mode(program_name))
                     for program_name, (_, tek_program, _) in self._programs.items()}

        self.programs.clear()
        self._sequence_entries = [None] * self.device.get_seq_length()
        self._synchronized = True

        self.initialize_idle_program()

        for program_name, (tek_program, mode) in to_upload.items():
            self._upload_parsed(program_name, tek_program)
            if not mode == 'unknown':
                self.set_program_repetition_mode(program_name, mode)
            else:
                self.logger.error("Could not restore repetition mode of %s as it was invalid", program_name)

        self._synchronized = True

    @property
    def armed_program(self) -> Optional[str]:
        return self._armed_program[0]

    def initialize_idle_program(self):
        """Make sure we can arm the idle program which plays the idle waveform(default 0V) on all channels."""
        self.assert_synchronized()

        self.logger.info("Initializing idle progam")
        if self._idle_waveform in self._waveforms.by_data:
            self.logger.debug("Idle waveform found on device")
            idle_waveform_name = self._waveforms.by_data[self._idle_waveform].name
        else:
            self.logger.debug("Idle waveform not found on device")
            idle_waveform_name = self.idle_pulse_name(self._idle_waveform.size)
            self._upload_waveform(self._idle_waveform, idle_waveform_name)

        idle_sequence_element = tek_awg.SequenceEntry(entries=[idle_waveform_name] * self.device.n_channels,
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
            self.logger.debug("Idle sequence entry not found on device. Uploading it to %d", idle_index)
            self._upload_sequencing_element(idle_index, idle_sequence_element)

            self._idle_program_index = self._sequence_entries.index(idle_sequence_element) + 1
        else:
            self.logger.debug("Idle sequence entry found on device: %d", self._idle_program_index)

    @property
    def device(self) -> tek_awg.TekAwg:
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
        self._unload(name)

    @property
    def sample_rate(self) -> float:
        return self.device.get_freq()

    def warn_if_errors_are_present(self):
        """Emit a warning with present errors."""
        errors = self.device.get_error_queue()
        if errors:
            self.logger.warning("Previous error(s): %r", errors)

    def upload(self, *args, **kwargs):
        self.warn_if_errors_are_present()

        if self.device.get_run_state().lower() == 'running':
            if self._auto_stop_on_upload:
                self.device.stop()
                self.device.wait_until_commands_executed()
            elif not self._allow_upload_while_running:
                raise RuntimeError("Tektronix AWG %r is running and allow_upload_while_running "
                                   "is False (its very slow)" % self.identifier)

        try:
            self._upload(*args, **kwargs)
        except:
            self.logger.exception("Error during upload. Set synced to false")

    def _process_program(self, name: str, tek_program: TektronixProgram) -> Tuple[Sequence[tek_awg.SequenceEntry],
                                                                                  Mapping[tek_awg.Waveform, str]]:
        """Detect which waveforms are missing and create sequencing entries.
        This function does not communicate with the device.

        Args:
            name:
            tek_program:

        Returns:
            sequencing_elements: List of SequenceEntries
            waveforms_to_upload: Missing waveforms with names
        """
        waveforms_to_upload = dict()
        required_idle_pulses = dict()

        sequencing_elements = []
        for entries, *sequencing_info in tek_program.get_sequencing_elements():
            new_entries = []

            for entry in entries:
                if isinstance(entry, str):
                    # check that we know the waveform
                    wf_name = self._waveforms.by_name[entry].name

                elif isinstance(entry, tek_awg.Waveform):
                    if entry in self._waveforms.by_data:
                        wf_name = self._waveforms.by_data[entry].name

                    elif entry in waveforms_to_upload:
                        wf_name = waveforms_to_upload[entry]

                    else:
                        wf_name = name + '_' + str(abs(hash(entry)))
                        waveforms_to_upload[entry] = wf_name

                else:
                    assert entry - int(entry) == 0
                    entry = int(entry)

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

                new_entries.append(wf_name)

            sequencing_elements.append(tek_awg.SequenceEntry(new_entries,
                                                            *sequencing_info))
        return sequencing_elements, waveforms_to_upload

    def _upload_linked_sequencing_elements(self,
                                           positions: Sequence[int],
                                           sequencing_elements: Sequence[tek_awg.SequenceEntry],
                                           last_jump_to: int):
        """Helper function to upload a linked list of sequencing elements. The goto+ind of each element is set if the
        next element requires a jump. Will result in unsynchronized state on error."""
        assert len(positions) == len(sequencing_elements)
        self.assert_synchronized()

        previous_errors = self.device.get_error_queue()
        if previous_errors:
            self.logger.warning("Error queue not empty before sequence upload: %r", previous_errors)

        positions_with_next = pairwise(positions, zip_function=itertools.zip_longest, fillvalue=last_jump_to)

        self._synchronized = False
        for idx, ((element_index, next_element), sequencing_element) in enumerate(zip(positions_with_next, sequencing_elements)):
            if element_index + 1 != next_element:
                sequencing_element.goto_ind = next_element
                sequencing_element.goto_state = True
            else:
                sequencing_element.goto_ind = None
                sequencing_element.goto_state = False

            self._upload_sequencing_element(element_index, sequencing_element)

            if idx % self._sequence_element_upload_sync_interval == 0:
                self.logger.debug('Waiting for sync after element %d' % idx)
                self.device.wait_until_commands_executed()
                self.logger.debug('Synced after element %d' % idx)

        self.device.wait_until_commands_executed()
        errors = self.device.get_error_queue()
        if errors:
            raise RuntimeError("Error(s) during sequence upload", errors)
        self._synchronized = True

    def _upload_waveforms(self, waveforms: Dict[str, tek_awg.Waveform]):
        """Upload waveforms from given dictionary. Will result in unsynchronized state on error."""
        self._synchronized = False
        for idx, (waveform_name, waveform_data) in enumerate(waveforms.items()):
            self._upload_waveform(waveform_data=waveform_data,
                                  waveform_name=waveform_name)
            if idx % self._waveform_upload_sync_interval == 0:
                self.logger.debug('Waiting for sync after waveform %d' % idx)
                self.device.wait_until_commands_executed()
                self.logger.debug('Synced after waveform %d' % idx)

        self.logger.debug('Waiting for all waveforms to be uploaded...')
        self.device.wait_until_commands_executed()

        errors = self.device.get_error_queue()
        if errors:
            raise RuntimeError("Error(s) during waveform upload", errors)
        self._synchronized = True

    def _upload_parsed(self, program_name: str, tek_program: TektronixProgram):
        """Requires to be in a synchronized state

        Args:
            name:
            tek_program:
            infinite_repetition: If true the last jump will be to the first entry. If false it will be the idle entry.
        """
        self.assert_synchronized()

        sequencing_elements, waveforms_to_upload = self._process_program(program_name, tek_program)
        positions = self._get_empty_sequence_positions(len(sequencing_elements))

        waveforms = {waveform_name: waveform_data for waveform_data, waveform_name in waveforms_to_upload.items()}
        assert len(waveforms) == len(waveforms_to_upload), "multiple waveforms with the same name (BUG)"

        self._upload_waveforms(waveforms)
        self.logger.info("All waveforms of '%s' uploaded", program_name)

        self._upload_linked_sequencing_elements(positions, sequencing_elements, self._idle_program_index)
        self.logger.info('All sequence elements of %s uploaded', program_name)

        self._programs[program_name] = (positions, tek_program, sequencing_elements)

        self.set_program_repetition_mode(program_name, self.default_program_repetition_mode)

    def _unload(self, name: str) -> TektronixProgram:
        """Remove program from internal state (allows overwriting the sequence entries)"""
        self.logger.debug("Deleting %s from known programs", name)
        positions, tek_program, seq_entries = self._programs.pop(name)

        self.logger.debug("Marking sequence entries of %s as free", name)
        for position in positions:
            if position <= len(self._sequence_entries):
                self._sequence_entries[position - 1] = None

        return tek_program

    def _upload(self, name: str,
                program: Loop,
                channels: Tuple[Optional[ChannelID], ...],
                markers: Tuple[Optional[ChannelID], ...],
                voltage_transformation: Tuple[Callable, ...],
                force: bool):
        assert self._idle_program_index

        if name in self._programs and not force:
            raise ProgramOverwriteException(name)

        # group markers in by channels
        markers = tuple(zip(markers[0::2], markers[1::2]))

        if self._amplitude_offset_handling == AWGAmplitudeOffsetHandling.IGNORE_OFFSET:
            offsets = None
        elif self._amplitude_offset_handling == AWGAmplitudeOffsetHandling.CONSIDER_OFFSET:
            offsets = self.device.get_offset()
        else:
            raise ValueError('{} is invalid as AWGAmplitudeOffsetHandling'.format(self._amplitude_offset_handling))

        tek_program = TektronixProgram(program, channels=channels, markers=markers,
                                       amplitudes=self.device.get_amplitude(),
                                       offsets=offsets,
                                       voltage_transformations=voltage_transformation,
                                       sample_rate=TimeType(self.sample_rate))

        self.logger.debug("Successfully parsed %s", name)

        if name in self._programs:
            self._unload(name)

        self._upload_parsed(name, tek_program)

    def _get_empty_sequence_positions(self, length: int) -> Sequence[int]:
        """Return a list of ``length`` empty sequence positions. This function talks to the device and requires a
        synchronized state"""
        self.assert_synchronized()

        free_positions = [idx + 1
                          for idx, sequencing_element in enumerate(self._sequence_entries)
                          if sequencing_element is None]
        missing = range(len(self._sequence_entries) + 1,
                        len(self._sequence_entries) + 1 + length - len(free_positions))
        if missing:
            self.device.set_seq_length(len(self._sequence_entries) + len(missing))
            self._sequence_entries.extend(itertools.repeat(None, len(missing)))
            free_positions.extend(missing)
        return free_positions[:length]

    def _delete_waveform(self, waveform_name: str):
        self.device.del_waveform(waveform_name)
        self.device.wait_until_commands_executed()
        if waveform_name in self._waveforms.by_name:
            self._waveforms.pop_waveform(waveform_name)

    def _upload_waveform(self, waveform_data: tek_awg.Waveform, waveform_name):
        self.device.new_waveform(waveform_name, waveform_data)
        timestamp = self.device.get_waveform_timestamps(waveform_name)
        self._waveforms.add_waveform(waveform_entry=WaveformEntry(name=waveform_name,
                                                                  length=waveform_data.size,
                                                                  waveform=waveform_data,
                                                                  timestamp=timestamp))

    def _upload_sequencing_element(self, element_index, sequencing_element: tek_awg.SequenceEntry):
        self._sequence_entries[element_index - 1] = sequencing_element
        self.device.set_seq_element(element_index, sequencing_element)

    def make_idle_waveform(self, length) -> tek_awg.Waveform:
        return tek_awg.Waveform(channel=np.full(length,
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
        """Arming starts the awg"""
        self.logger.info("Arming program %r", name)

        self.device.jump_to_sequence_element(self._idle_program_index)
        self.device.wait_until_commands_executed()
        self.logger.debug("Jumped to idle program")

        self.device.run()
        self.device.wait_until_commands_executed()
        self.logger.debug("Switched to run")

        if name is None:
            self._armed_program = None, None

        else:
            positions, _, _ = self._programs[name]
            self._armed_program = (name, positions[0])

    def run_current_program(self, channel_states: Optional[Tuple[bool, bool, bool, bool]] = None):
        """Runs the currentlz armed program

        Args:
            channel_states: If given the channel states are set to these values

        Returns:

        """
        assert channel_states is None or len(channel_states) == 4

        program_name, program_index = self._armed_program
        if program_name is None:
            self.logger.warning("Ignoring run_current_program call as no program is armed")
            return
        else:
            self.logger.info("Running program '%s'", program_name)
            self.device.jump_to_sequence_element(program_index)
            self.device.wait_until_commands_executed()
