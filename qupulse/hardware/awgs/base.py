"""This module defines the common interface for arbitrary waveform generators.

Classes:
    - AWG: Common AWG interface.
    - DummyAWG: A software stub implementation of the AWG interface.
    - ProgramOverwriteException
    - OutOfWaveformMemoryException
"""

from abc import abstractmethod
from numbers import Real
from typing import Set, Tuple, Callable, Optional, Mapping, Sequence, List, Union, NamedTuple
from collections import OrderedDict
from enum import Enum
# from itertools import chain

from qupulse.hardware.util import get_sample_times, not_none_indices
from qupulse.utils.types import ChannelID
from qupulse.program.linspace import LinSpaceNode, LinSpaceArbitraryWaveform, to_increment_commands, Command, \
    Increment, Set as LSPSet, LoopLabel, LoopJmp, Wait, Play, DEFAULT_INCREMENT_RESOLUTION, DepDomain
from qupulse.program.loop import Loop
from qupulse.program.waveforms import Waveform, WaveformCollection
from qupulse.comparable import Comparable
from qupulse.utils.types import TimeType

import numpy

__all__ = ["AWG", "Program", "ProgramOverwriteException",
           "OutOfWaveformMemoryException", "AWGAmplitudeOffsetHandling"]

Program = Loop


class AWGAmplitudeOffsetHandling:
    IGNORE_OFFSET = 'ignore_offset'   # Offset is ignored.
    CONSIDER_OFFSET = 'consider_offset' # Offset is discounted from the waveforms.
    # TODO OPTIMIZED = 'optimized' # Offset and amplitude are set depending on the waveforms to maximize the waveforms resolution

    _valid = [IGNORE_OFFSET, CONSIDER_OFFSET]


class AWG(Comparable):
    """An arbitrary waveform generator abstraction class.

    It represents a set of channels that have to have(hardware enforced) the same:
        -control flow
        -sample rate

    It keeps track of the AWG state and manages waveforms and programs on the hardware.
    """

    def __init__(self, identifier: str):
        self._identifier = identifier
        self._amplitude_offset_handling = AWGAmplitudeOffsetHandling.IGNORE_OFFSET

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def amplitude_offset_handling(self) -> str:
        return self._amplitude_offset_handling

    @amplitude_offset_handling.setter
    def amplitude_offset_handling(self, value):
        """
        value (str): See possible values at `AWGAmplitudeOffsetHandling`
        """
        if value not in AWGAmplitudeOffsetHandling._valid:
            raise ValueError('"{}" is invalid as AWGAmplitudeOffsetHandling'.format(value))

        self._amplitude_offset_handling = value

    @property
    @abstractmethod
    def num_channels(self):
        """Number of channels"""

    @property
    @abstractmethod
    def num_markers(self):
        """Number of marker channels"""

    @abstractmethod
    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional[ChannelID], ...],
               markers: Tuple[Optional[ChannelID], ...],
               voltage_transformation: Tuple[Optional[Callable], ...],
               force: bool=False) -> None:
        """Upload a program to the AWG.

        Physically uploads all waveforms required by the program - excluding those already present -
        to the device and sets up playback sequences accordingly.
        This method should be cheap for program already on the device and can therefore be used
        for syncing. Programs that are uploaded should be fast(~1 sec) to arm.

        Args:
            name: A name for the program on the AWG.
            program: The program (a sequence of instructions) to upload.
            channels: Tuple of length num_channels that ChannelIDs of  in the program to use. Position in the list corresponds to the AWG channel
            markers: List of channels in the program to use. Position in the List in the list corresponds to the AWG channel
            voltage_transformation: transformations applied to the waveforms extracted rom the program. Position
            in the list corresponds to the AWG channel
            force: If a different sequence is already present with the same name, it is
                overwritten if force is set to True. (default = False)
        """

    @abstractmethod
    def remove(self, name: str) -> None:
        """Remove a program from the AWG.

        Also discards all waveforms referenced only by the program identified by name.

        Args:
            name: The name of the program to remove.
        """

    @abstractmethod
    def clear(self) -> None:
        """Removes all programs and waveforms from the AWG.

        Caution: This affects all programs and waveforms on the AWG, not only those uploaded using qupulse!
        """

    @abstractmethod
    def arm(self, name: Optional[str]) -> None:
        """Load the program 'name' and arm the device for running it. If name is None the awg will "dearm" its current
        program."""

    @property
    @abstractmethod
    def programs(self) -> Set[str]:
        """The set of program names that can currently be executed on the hardware AWG."""

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """The sample rate of the AWG."""

    @property
    def compare_key(self) -> int:
        """Comparison and hashing is based on the id of the AWG so different devices with the same properties
        are ot equal"""
        return id(self)

    @abstractmethod
    def set_volatile_parameters(self, program_name: str, parameters: Mapping[str, Real]):
        """Set the values of parameters which were marked as volatile on program creation."""

    def __copy__(self) -> None:
        raise NotImplementedError()

    def __deepcopy__(self, memodict={}) -> None:
        raise NotImplementedError()


class ProgramOverwriteException(Exception):

    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    def __str__(self) -> str:
        return "A program with the given name '{}' is already present on the device." \
               " Use force to overwrite.".format(self.name)


AllowedProgramTypes = Union[Loop,Sequence[LinSpaceNode],]

class _ProgramType(Enum):
    FSP = -1
    Loop = 0
    Linspace = 1


class ChannelTransformation(NamedTuple):
    amplitude: float
    offset: float
    voltage_transformation: Optional[callable]


class ProgramEntry:
    """This is a helper class for implementing awgs drivers. A driver can subclass it to help organizing sampled
    waveforms"""
    def __init__(self, program: AllowedProgramTypes,
                 channels: Tuple[Optional[ChannelID], ...],
                 markers: Tuple[Optional[ChannelID], ...],
                 amplitudes: Tuple[float, ...],
                 offsets: Tuple[float, ...],
                 voltage_transformations: Tuple[Optional[Callable], ...],
                 sample_rate: TimeType,
                 waveforms: Sequence[Waveform] = None,
                 # voltage_resolution: Optional[float] = None,
                 program_type: _ProgramType = _ProgramType.Loop):
        """

        Args:
            loop:
            channels:
            markers:
            amplitudes:
            offsets:
            voltage_transformations:
            sample_rate:
            waveforms: These waveforms are sampled and stored in _waveforms. If None the waveforms are extracted from
            loop
            # voltage_resolution: voltage resolution for LinSpaceProgram, i.e. 2**(-16) for 16 bit AWG
            program_type: type of program from _ProgramType, determined by the ProgramBuilder used.
        """
        assert len(channels) == len(amplitudes) == len(offsets) == len(voltage_transformations)

        self._channels = tuple(channels)
        self._markers = tuple(markers)
        self._amplitudes = tuple(amplitudes)
        self._offsets = tuple(offsets)
        self._voltage_transformations = tuple(voltage_transformations)

        self._sample_rate = sample_rate
        
        self._program_type = program_type
        self._program = program
        
        # self._voltage_resolution = voltage_resolution
        
        if program_type == _ProgramType.Linspace:
            #!!! the voltage resolution may not be adequately represented if voltage transformations are not None?
            self._transformed_commands = self._transform_linspace_commands(to_increment_commands(program,))
        
        if waveforms is None:
            if program_type is _ProgramType.Loop:
                    waveforms = OrderedDict((node.waveform, None)
                                        for node in program.get_depth_first_iterator() if node.is_leaf()).keys()
            elif program_type is _ProgramType.Linspace:
                    #not so clean
                    #TODO: also marker handling not optimal
                    waveforms_d = OrderedDict()
                    for command in self._transformed_commands:
                        if not isinstance(command,Play):
                            continue
                        if isinstance(command.waveform,Waveform):
                            waveforms_d[command.waveform] = None
                        elif isinstance(command.waveform,WaveformCollection):
                            for w in command.waveform.flatten():
                                waveforms_d[w] = None
                        else:
                            raise NotImplementedError()
                    waveforms = waveforms_d.keys()
            else:
                raise NotImplementedError()
                    
        if waveforms:
            self._waveforms = OrderedDict(zip(waveforms, self._sample_waveforms(waveforms)))
        else:
            self._waveforms = OrderedDict()
    
    @property
    def _loop(self,) -> Loop:
        if self._program_type not in (_ProgramType.Loop, _ProgramType.FSP):
            raise AttributeError("The _loop attribute can only be get on loop-like program entries.")
        return self._program
    
    @_loop.setter
    def _loop(self, program: Loop):
        if self._program_type not in (_ProgramType.Loop, _ProgramType.FSP):
            raise AttributeError("The _loop attribute can only be set on loop-like program entries.")
        self._program = program
    
    def _sample_empty_channel(self, time: numpy.ndarray) -> Optional[numpy.ndarray]:
        """Override this in derived class to change how empty channels are handled"""
        return None

    def _sample_empty_marker(self, time: numpy.ndarray) -> Optional[numpy.ndarray]:
        """Override this in derived class to change how empty channels are handled"""
        return None

    def _channel_transformations(self) -> Mapping[ChannelID, ChannelTransformation]:
        return {ch: ChannelTransformation(amplitude, offset, trafo)
                for ch, trafo, amplitude, offset in zip(self._channels,
                                                        self._voltage_transformations,
                                                        self._amplitudes,
                                                        self._offsets)}
    
    def _transform_linspace_commands(self, command_list: List[Command]) -> List[Command]:
        # all commands = Union[Increment, Set, LoopLabel, LoopJmp, Wait, Play]
        # TODO: voltage resolution
        
        # trafos_by_channel_idx = list(self._channel_transformations().values())
        # increment_domains_to_transform = {DepDomain.VOLTAGE, DepDomain.WF_SCALE, DepDomain.WF_OFFSET}
        
        for command in command_list:
            if isinstance(command, (LoopLabel, LoopJmp, Play, Wait)):
                # play is handled by transforming the sampled waveform
                continue
            elif isinstance(command, Increment):
                if command.key.domain is not DepDomain.VOLTAGE or \
                    command.channel not in self._channels:
                    #for sweeps of wf-scale and wf-offset, the channel amplitudes/offsets are already considered in the wf sampling.
                    continue
                
                ch_trafo = self._channel_transformations()[command.channel]
                if ch_trafo.voltage_transformation:
                    raise RuntimeError("Cannot apply a voltage transformation to a linspace increment command")
                command.value /= ch_trafo.amplitude
            elif isinstance(command, LSPSet):
                if command.key.domain is not DepDomain.VOLTAGE or \
                    command.channel not in self._channels:
                    #for sweeps of wf-scale and wf-offset, the channel amplitudes/offsets are already considered in the wf sampling.
                    continue
                ch_trafo = self._channel_transformations()[command.channel]
                if ch_trafo.voltage_transformation:
                    # for the case of swept parameters, this is defaulted to identity
                    command.value = float(ch_trafo.voltage_transformation(command.value))
                command.value -= ch_trafo.offset
                command.value /= ch_trafo.amplitude
            else:        
                raise NotImplementedError(command)
        
        return command_list
    
    def _sample_waveforms(self, waveforms: Sequence[Waveform]) -> List[Tuple[Tuple[numpy.ndarray, ...],
                                                                             Tuple[numpy.ndarray, ...]]]:
        sampled_waveforms = []

        time_array, segment_lengths = get_sample_times(waveforms, self._sample_rate)
        sample_memory = numpy.zeros_like(time_array, dtype=float)

        n_samples = numpy.sum(segment_lengths)
        ch_to_mem, n_ch = not_none_indices(self._channels)
        mk_to_mem, c_mk = not_none_indices(self._markers)

        ch_memory = numpy.zeros((n_ch, n_samples), dtype=float)
        marker_memory = numpy.zeros((c_mk, n_samples), dtype=bool)
        segment_begin = 0

        for waveform, segment_length in zip(waveforms, segment_lengths):
            segment_length = int(segment_length)
            segment_end = segment_begin + segment_length

            wf_time = time_array[:segment_length]
            wf_sample_memory = sample_memory[:segment_length]

            sampled_channels = []
            for channel, ch_mem_pos, trafo, amplitude, offset in zip(self._channels, ch_to_mem,
                                                                     self._voltage_transformations,
                                                                     self._amplitudes, self._offsets):
                final_memory = ch_memory[ch_mem_pos, segment_begin:segment_end]

                if channel is None:
                    sampled_channels.append(self._sample_empty_channel(wf_time))
                else:
                    if trafo is None:
                        # sample directly into the final memory
                        sampled = waveform.get_sampled(channel, wf_time, output_array=final_memory)
                    else:
                        # sample into temporary memory and write the trafo result in the final memory
                        # unfortunately trafo will always allocate :(
                        sampled = waveform.get_sampled(channel, wf_time, output_array=wf_sample_memory)
                        assert sampled is wf_sample_memory
                        final_memory[:] = trafo(sampled)
                        sampled = final_memory
                    assert sampled is final_memory
                    sampled -= offset
                    sampled /= amplitude
                    sampled_channels.append(sampled)

            sampled_markers = []
            for marker, mk_mem_pos in zip(self._markers, mk_to_mem):
                final_memory = marker_memory[mk_mem_pos, segment_begin:segment_end]

                if marker is None:
                    sampled_markers.append(self._sample_empty_marker(wf_time))
                else:
                    sampled = waveform.get_sampled(marker, wf_time, output_array=wf_sample_memory)
                    sampled = numpy.not_equal(sampled, 0., out=final_memory)
                    assert sampled is final_memory

                    sampled_markers.append(sampled)

            sampled_waveforms.append((tuple(sampled_channels), tuple(sampled_markers)))

            segment_begin = segment_end
        assert segment_begin == n_samples
        return sampled_waveforms


class OutOfWaveformMemoryException(Exception):

    def __str__(self) -> str:
        return "Out of memory error adding waveform to waveform memory."


class ChannelNotFoundException(Exception):
    def __init__(self, channel):
        self.channel = channel

    def __str__(self) -> str:
        return 'Marker or channel not found: {}'.format(self.channel)
