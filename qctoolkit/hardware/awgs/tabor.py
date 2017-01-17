from qctoolkit.pulses.pulse_template import ChannelID
from qctoolkit.hardware.program import Loop, MultiChannelProgram
from qctoolkit.hardware.util import voltage_to_uint16

import sys
import numpy as np
from typing import List, Tuple, Iterable

# Provided by Tabor electronics for python 2.7
# a python 3 version is in a private repository on https://git.rwth-aachen.de/qutech
# Beware of the string encoding change!
import pytabor
import teawg

assert(sys.byteorder == 'little')


class TaborProgram:
    WAVEFORM_MODES = ('single', 'advanced', 'sequence')

    def __init__(self, program: MultiChannelProgram, device_properties, channels: Tuple[ChannelID, ChannelID]):
        if len(channels) > 2:
            raise Exception('TaborProgram only supports 2 channels')
        channel_set = frozenset(channel for channel in channels if channel is not None)
        self.__root_loop = None
        for known_channels in program.programs.keys():
            if known_channels.issuperset(channel_set):
                self.__root_loop = program.programs[known_channels]
        if self.__root_loop is None:
            raise TaborException("{} not found in program.".format(channel_set))

        self.__waveform_mode = 'advanced'
        self.__channels = channels
        self.__device_properties = device_properties

        self.__waveforms = []
        self.__sequencer_tables = []
        self.__advanced_sequencer_table = []

        if self.program.depth() == 0:
            self.setup_single_waveform_mode()
        elif self.program.depth() == 1:
            self.setup_single_sequence_table_mode()
        else:
            self.setup_advanced_sequence_mode()

    def setup_single_waveform_mode(self):
        raise NotImplementedError()

    def setup_single_sequence_mode(self):
        self.__waveform_mode = 'sequence'
        if len(self.program) < self.__device_properties['min_seq_len']:
            raise TaborException('SEQuence:LENGth has to be >={min_seq_len}'.format(**self.__device_properties))
        raise NotImplementedError()

    def setup_advanced_sequence_mode(self):
        while self.program.depth() > 2 or not self.program.is_balanced():
            for i, sequence_table in enumerate(self.program):
                if sequence_table.depth() == 0:
                    sequence_table.encapsulate()
                elif sequence_table.depth() == 1:
                    assert (sequence_table.is_balanced())
                elif len(sequence_table) == 1 and len(sequence_table[0]) == 1:
                    sequence_table.join_loops()
                elif sequence_table.is_balanced():
                    if len(self.program) < self.__device_properties['min_aseq_len'] or (sequence_table.repetition_count / self.__device_properties['max_aseq_len'] <
                                                max(entry.repetition_count for entry in sequence_table) /
                                                self.__device_properties['max_seq_len']):
                        sequence_table.unroll()
                    else:
                        for entry in sequence_table.children:
                            entry.unroll()

                else:
                    depth_to_unroll = sequence_table.depth() - 1
                    for entry in sequence_table:
                        if entry.depth() == depth_to_unroll:
                            entry.unroll()

        i = 0
        while i < len(self.program):
            sequence_table = self.program[i]
            if len(sequence_table) > self.__device_properties['max_seq_len']:
                raise TaborException()
            elif len(sequence_table) < self.__device_properties['min_seq_len']:
                # try to merge with neighbours
                if sequence_table.repetition_count == 1:
                    if i > 0 and self.program[i-1].repetition_count == 1:
                        self.program[i-1][len(self.program[i-1]):] = sequence_table[:]
                        self.program[i:i+1] = []
                    elif i+1 < len(self.program) and self.program[i+1].repetition_count == 1:
                        self.program[i+1][:0] = sequence_table[:]
                        self.program[i:i+1] = []
                    else:
                        self.increase_sequence_table_length(sequence_table, self.__device_properties)
                        i += 1
                else:
                    self.increase_sequence_table_length(sequence_table, self.__device_properties)
                    i += 1
            else:
                i += 1

        assert (self.program.repetition_count == 1)
        if len(self.program) < self.__device_properties['min_aseq_len']:
            raise TaborException()
        if len(self.program) > self.__device_properties['max_aseq_len']:
            raise TaborException()
        for sequence_table in self.program:
            if len(sequence_table) < self.__device_properties['min_seq_len']:
                raise TaborException()
            if len(sequence_table) > self.__device_properties['max_seq_len']:
                raise TaborException()

        advanced_sequencer_table = []
        sequencer_tables = []
        waveforms = []
        for sequencer_table_loop in self.program:
            current_sequencer_table = []
            for waveform_loop in sequencer_table_loop:
                if waveform_loop.instruction.waveform in waveforms:
                    segment_no = waveforms.index(waveform_loop.instruction.waveform) + 1
                else:
                    segment_no = len(waveforms) + 1
                    waveforms.append(waveform_loop.instruction.waveform)
                current_sequencer_table.append((waveform_loop.repetition_count, segment_no, 0))

            if current_sequencer_table in sequencer_tables:
                sequence_no = sequencer_tables.index(current_sequencer_table) + 1
            else:
                sequence_no = len(sequencer_tables) + 1
                sequencer_tables.append(current_sequencer_table)

            advanced_sequencer_table.append((sequencer_table_loop.repetition_count, sequence_no, 0))

        self.__advanced_sequencer_table = advanced_sequencer_table
        self.__sequencer_tables = sequencer_tables
        self.__waveforms = waveforms

    @property
    def program(self):
        return self.__root_loop

    def get_sequencer_tables(self):
        return self.__sequencer_tables

    @staticmethod
    def increase_sequence_table_length(sequence_table: Loop, device_properties):
        assert(sequence_table.depth() == 1)
        if len(sequence_table) < device_properties['min_seq_len']:

            if sum(entry.repetition_count for entry in sequence_table)*sequence_table.repetition_count >= device_properties['min_seq_len']:
                if sum(entry.repetition_count for entry in sequence_table) < device_properties['min_seq_len']:
                    sequence_table.unroll_children()
                while len(sequence_table) < device_properties['min_seq_len']:
                    sequence_table.split_one_child()
            else:
                TaborException('Sequence table too short: ', sequence_table)

    def get_advanced_sequencer_table(self):
        """Advanced sequencer table that can be used  via the download_adv_seq_table pytabor command"""
        return self.__advanced_sequencer_table

    def get_waveform_data(self, device_properties, samplerate: float, voltage_amplitude: Tuple[float, float], voltage_offset: Tuple[float, float]):
        if any(not(waveform.duration*samplerate).is_integer() for waveform in self.__waveforms):
            raise TaborException('At least one waveform has a length that is no multiple of the time per sample')
        maximal_length = int(max(waveform.duration for waveform in self.__waveforms) * samplerate)
        time_array = np.arange(0, maximal_length, 1)
        maximal_size = int(2 * (samplerate*sum(waveform.duration for waveform in self.__waveforms) + 16*len(self.__waveforms)))
        data = np.empty(maximal_size, dtype=np.uint16)
        offset = 0
        segment_lengths = np.zeros(len(self.__waveforms), dtype=np.uint32)

        def voltage_to_data(waveform, time, channel):
            return voltage_to_uint16(waveform[self.__channels[channel]].sample(time),
                                     voltage_amplitude[channel],
                                     voltage_offset[channel],
                                     resolution=14) if self.__channels[channel] else None

        for i, waveform in enumerate(self.__waveforms):
            t = time_array[:int(waveform.duration*samplerate)]
            segment1 = voltage_to_data(waveform, t, 0)
            segment2 = voltage_to_data(waveform, t, 1)
            segment_lengths[i] = len(segment1) if segment1 is not None else len(segment2)
            assert(segment2 is None or segment_lengths[i] == len(segment2))
            offset = pytabor.make_combined_wave(
                segment1,
                segment2,
                data, offset, add_idle_pts=True)

        if np.any(segment_lengths < device_properties['min_seq_len']):
            raise Exception()
        if np.any(segment_lengths % device_properties['seg_quantum']>0):
            raise Exception()

        return data[:offset], segment_lengths

    def upload_to_device(self, device: 'TaborAWG', channel_pair):
        if channel_pair not in ((1, 2), (3, 4)):
            raise Exception('Invalid channel pair', channel_pair)

        if self.__waveform_mode == 'advanced':
            samplerate = device.samplerate(channel_pair[0])
            amplitude = (device.amplitude(channel_pair[0]), device.amplitude(channel_pair[1]))
            offset = (device.offset(channel_pair[0]), device.offset(channel_pair[1]))

            wf_data, segment_lengths = self.get_waveform_data(device_properties=device.dev_properties,
                                                              samplerate=samplerate,
                                                              voltage_amplitude=amplitude,
                                                              voltage_offset=offset)
            # download the waveform data as one big waveform
            device.select_channel(channel_pair[0])
            device.send_cmd(':FUNC:MODE ASEQ')
            device.send_cmd(':TRAC:DEF 1,{}'.format(len(wf_data)))
            device.send_cmd(':TRAC:SEL 1')
            device.send_binary_data(pref=':TRAC:DATA', bin_dat=wf_data)
            # partition the memory
            device.download_segment_lengths(segment_lengths)

            #download all sequence tables
            for i, sequencer_table in enumerate(self.get_sequencer_tables()):
                device.send_cmd('SEQ:SEL {}'.format(i+1))
                device.download_sequencer_table(sequencer_table)

            device.download_adv_seq_table(self.get_advanced_sequencer_table())
            device.send_cmd('SEQ:SEL 1')



class TaborAWG(teawg.TEWXAwg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__selected_channel = None

    @property
    def is_open(self):
        return self.visa_inst is not None

    def select_channel(self, channel):
        if channel not in (1, 2, 3, 4):
            raise TaborException('Invalid channel: {}'.format(channel))
        if self.__selected_channel != channel:
            self.send_cmd(':INST:SEL {channel}'.format(channel=channel))
            self.__selected_channel = channel

    def samplerate(self, channel):
        self.select_channel(channel)
        return int(float(self.send_query(':FREQ:RAST?'.format(channel=channel))))

    def amplitude(self, channel):
        self.select_channel(channel)
        couplig = self.send_query(':OUTP:COUP?'.format(channel=channel))
        if couplig == 'DC':
            return float(self.send_query(':VOLT?'))
        elif couplig == 'HV':
            return float(self.send_query(':VOLD:HV?'))

    def offset(self, channel):
        self.select_channel(channel)
        return float(self.send_query(':VOLT:OFFS?'.format(channel=channel)))


class TaborException(Exception):
    pass
