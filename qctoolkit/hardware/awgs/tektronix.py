import pip
import importlib
from typing import Tuple, Callable, Optional
import recordclass
import numpy as np


try:
    import TekAwg
except ImportError:
    TekAwg = None

from qctoolkit.hardware.awgs.base import AWG
from qctoolkit import ChannelID
from qctoolkit.hardware.program import Loop


def install_requirements():
    global TekAwg

    # backend from Dar Dahlend I forked it to have a "stable" version
    package_repo = 'https://github.com/terrorfisch/TekAwg/archive/master.zip'

    pip.main(['install', package_repo])

    TekAwg = importlib.import_module('TekAwg')


class SequenceEntry:
    def __init__(self, entries, wait=False,
                 loop_inf: bool=False,
                 loop_count: int=1,
                 goto_idx: int=1,
                 goto_state: bool=False,
                 event_type: str='OFF',
                 event_idx: int=1):
        if entries is None:
            entries = []

        self.entries = entries
        self.wait = wait
        self.loop_inf = loop_inf
        self.loop_count = loop_count
        self.goto_idx = goto_idx
        self.goto_state = goto_state
        self.event_type = event_type
        self.event_idx = event_idx


class Waveform:
    channel_mask = np.uint16(2**14 - 1)
    marker_1_mask = np.uint16(2**14)
    marker_2_mask = np.uint16(2**15)

    def __init__(self, channel: np.ndarray, marker_1: np.ndarray, marker_2: np.ndarray):
        self.channel = channel
        self.marker_1 = marker_1
        self.marker_2 = marker_2

    @classmethod
    def from_binary(cls, binary):
        return cls(channel = np.bitwise_and(binary, cls.channel_mask),
                   marker_1= np.bitwise_and(binary, cls.marker_1_mask),
                   marker_2= np.bitwise_and(binary, cls.marker_2_mask))

    def to_binary(self):
        result = np.bitwise_and(self.channel, self.channel_mask)
        result |= self.marker_1_mask * self.marker_1.astype(bool)
        result |= self.marker_2_mask * self.marker_2.astype(bool)
        return result


class WaveformEntry:
    def __init__(self, name: str, length: int, waveform: Waveform, timestamp):
        self.name = name
        self.waveform = waveform
        self.length = length
        self.timestamp = timestamp


class TektronixAWG(AWG):
    def __init__(self, tek_awg: TekAwg.TekAwg, identifier='Tektronix'):
        super().__init__(identifier=identifier)

        if TekAwg is None:
            raise RuntimeError('Please install the TekAwg package or run "install_requirements" from this module')

        self._device = tek_awg

        self._sequence_entries = []
        self._waveforms = []
        self._waveform_references = np.zeros((), dtype=bool)

    @property
    def device(self) -> TekAwg.TekAwg:
        return self._device

    def read_waveforms(self, read_data=True):
        wf_names = self.device.get_waveform_names()
        wf_times = self.device.get_waveform_timestamps(wf_names)
        wf_lengths = self.device.get_waveform_lengths(wf_names)

        if read_data:
            wf_datas = [Waveform.from_binary(self.device.get_waveform_data(wf_name))
                        for wf_name in wf_names]
        else:
            wf_datas = [None] * len(wf_names)

        waveforms = [WaveformEntry(name, length, time, data)
                     for name, length, time, data in zip(wf_names, wf_lengths, wf_times, wf_datas)]

        self._waveforms = waveforms

    def read_sequencer(self):
        entries = [SequenceEntry(entries=self.device.get_seq_element(i),
                                 wait=self.device.get_seq_element_wait(i),
                                 loop_inf=self.device.get_seq_element_loop_inf(i),
                       loop_count=self.device.get_seq_element_loop_count(i),
                       goto_idx=self.device.get_seq_element_goto_ind(i),
                       goto_state=self.device.get_seq_element_goto_state(i),
                       event_type=self.device.get_seq_element_jmp_type(i),
                       event_idx=self.device.get_seq_element_jmp_ind(i)
                       )
                   for i in range(1, 1 + self.device.get_seq_length())]

        self._sequence_entries = entries

    def read_all(self):
        self.read_waveforms(read_data=True)
        self.read_sequencer()

    @property
    def num_channels(self):
        return self.device.n_channels

    @property
    def num_markers(self):
        return self.num_channels * 2

    @property
    def programs(self):
        raise NotImplementedError()

    def remove(self, name: str):
        raise NotImplementedError()

    @property
    def sample_rate(self):
        raise NotImplementedError()

    def upload(self, name: str,
               program: Loop,
               channels: Tuple[Optional[ChannelID], ...],
               markers: Tuple[Optional[ChannelID], ...],
               voltage_transformation: Tuple[Optional[Callable], ...],
               force: bool=False):
        raise NotImplementedError()

    def arm(self, name: Optional[str]):
        raise NotImplementedError()








