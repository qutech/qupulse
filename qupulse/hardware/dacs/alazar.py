from typing import Dict, Any, Optional, Tuple, Sequence
from collections import defaultdict
import logging
import math
import functools
import abc

import numpy as np

from atsaverage.config import ScanlineConfiguration
from atsaverage.masks import CrossBufferMask, Mask

from qupulse.hardware.dacs.dac_base import DAC


class AlazarProgram:
    def __init__(self, masks=list(), operations=list(), total_length=None):
        self.masks = masks
        self.operations = operations
        self.total_length = total_length
    def __iter__(self):
        yield self.masks
        yield self.operations
        yield self.total_length


def gcd_set(data):
    return functools.reduce(math.gcd, data)


class BufferStrategy(metaclass=abc.ABCMeta):
    """This class defines the strategy how the buffer size is chosen. Buffers might impact the signal due to hardware
    imperfections. The aim of this class is to allow the user to work around that."""
    @abc.abstractmethod
    def calculate_acquisition_properties(self,
                                         masks: Sequence[CrossBufferMask],
                                         buffer_length_divisor: int) -> Tuple[int, int]:
        """

        Args:
            windows: Measurement windows in samples
            buffer_length_divisor: Necessary divisor of the buffer length

        Returns:
            A tuple (buffer_length, total_acquisition_length)
        """

    @staticmethod
    def minimum_total_length(masks: Sequence[CrossBufferMask]) -> int:
        mtl = 0
        for mask in masks:
            mtl = max(mtl, mask.begin[-1] + mask.length[-1])
        return mtl


class ForceBufferSize(BufferStrategy):
    def __init__(self, target_size: int):
        """
        Args:
            aimed_size: Try to use that length
        """
        super().__init__()
        self.target_buffer_size = target_size

    def calculate_acquisition_properties(self,
                                         masks: Sequence[CrossBufferMask],
                                         buffer_length_divisor: int) -> Tuple[int, int]:
        buffer_size = int(self.target_buffer_size or buffer_length_divisor)
        if buffer_size % buffer_length_divisor:
            raise ValueError('Target size not possible for required buffer length divisor',
                             buffer_size, buffer_length_divisor)

        mtl = self.minimum_total_length(masks)
        total_length = int(math.ceil(mtl / buffer_size) * buffer_size)
        return buffer_size, total_length

    def __repr__(self):
        return 'ForceBufferSize(target_size=%r)' % self.target_buffer_size


class AvoidSingleBufferAcquisition(BufferStrategy):
    def __init__(self, wrapped_strategy: BufferStrategy):
        self.wrapped_strategy = wrapped_strategy

    def calculate_acquisition_properties(self,
                                         masks: Sequence[CrossBufferMask],
                                         buffer_length_divisor: int) -> Tuple[int, int]:
        buffer_size, total_length = self.wrapped_strategy.calculate_acquisition_properties(masks,
                                                                                           buffer_length_divisor)
        if buffer_size == total_length and buffer_size != buffer_length_divisor:
            # resize the buffer and recalculate total length

            # n is at least 2
            n = total_length // buffer_length_divisor
            buffer_size = (n // 2) * buffer_length_divisor
            mtl = self.minimum_total_length(masks)
            total_length = int(math.ceil(mtl / buffer_size) * buffer_size)
        return buffer_size, total_length

    def __repr__(self):
        return 'AvoidSingleBufferAcquisition(wrapped_strategy=%r)' % self.wrapped_strategy


class OneBufferPerWindow(BufferStrategy):
    """Choose the greatest common divisor of all window periods (diff(begin)) as buffer size. Aim is to only have an
    integer number of buffers in a measurement window."""

    def calculate_acquisition_properties(self,
                                         masks: Sequence[CrossBufferMask],
                                         buffer_length_divisor: int) -> Tuple[int, int]:
        gcd = None
        for mask in masks:
            c_gcd = gcd_set(np.unique(np.diff(mask.begin.as_ndarray())))
            if gcd is None:
                gcd = c_gcd
            else:
                gcd = math.gcd(gcd, c_gcd)

        buffer_size = max((gcd // buffer_length_divisor) *  buffer_length_divisor, buffer_length_divisor)
        mtl = self.minimum_total_length(masks)
        total_length = int(math.ceil(mtl / buffer_size) * buffer_size)
        return buffer_size, total_length

    def __repr__(self):
        return 'OneBufferPerWindow()'


class AlazarCard(DAC):
    def __init__(self, card, config: Optional[ScanlineConfiguration]=None):
        self.__card = card

        self.__armed_program = None
        self.update_settings = True

        self.__definitions = dict()
        self.config = config

        # defaults to self.__card.minimum_record_size
        self._buffer_strategy = None

        self._mask_prototypes = dict()  # type: Dict

        self._registered_programs = defaultdict(AlazarProgram)  # type: Dict[str, AlazarProgram]

    @property
    def card(self) -> Any:
        return self.__card

    @property
    def buffer_strategy(self) -> BufferStrategy:
        if self._buffer_strategy is None:
            return AvoidSingleBufferAcquisition(ForceBufferSize(self.config.aimedBufferSize))
        else:
            return self._buffer_strategy

    @buffer_strategy.setter
    def buffer_strategy(self, strategy):
        if strategy is not None and not isinstance(strategy, BufferStrategy):
            raise TypeError('Buffer strategy must be of type BufferStrategy or None')
        self._buffer_strategy = strategy

    def _make_mask(self, mask_id: str, begins, lengths) -> Mask:
        if mask_id not in self._mask_prototypes:
            raise KeyError('Measurement window {} can not be converted as it is not registered.'.format(mask_id))

        hardware_channel, mask_type = self._mask_prototypes[mask_id]

        if np.any(begins[:-1]+lengths[:-1] > begins[1:]):
            raise ValueError('Found overlapping windows in begins')

        mask = CrossBufferMask()
        mask.identifier = mask_id
        mask.begin = begins
        mask.length = lengths
        mask.channel = hardware_channel
        return mask

    def register_measurement_windows(self,
                                     program_name: str,
                                     windows: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        if not windows:
            self._registered_programs[program_name].masks = []
        total_length = 0
        for mask_id, (begins, lengths) in windows.items():

            sample_factor = self.config.captureClockConfiguration.numeric_sample_rate(self.__card.model) / 10**9

            begins = np.rint(begins*sample_factor).astype(dtype=np.uint64)
            lengths = np.floor(lengths*sample_factor).astype(dtype=np.uint64)

            sorting_indices = np.argsort(begins)
            begins = begins[sorting_indices]
            lengths = lengths[sorting_indices]

            windows[mask_id] = (begins, lengths)

        self._registered_programs[program_name].masks = [
            self._make_mask(mask_id, *window_begin_length)
            for mask_id, window_begin_length in windows.items()]
        self._registered_programs[program_name].total_length = total_length

    def register_operations(self, program_name: str, operations) -> None:
        self._registered_programs[program_name].operations = operations

    def arm_program(self, program_name: str) -> None:
        to_arm = self._registered_programs[program_name]
        if self.update_settings or self.__armed_program is not to_arm:
            config = self.config
            config.masks, config.operations, total_record_size = self._registered_programs[program_name]

            if len(config.operations) == 0:
                raise RuntimeError('No operations configured for program {}'.format(program_name))

            if not config.masks:
                if config.operations:
                    raise RuntimeError('Invalid configuration. Operations have no masks to work with')
                else:
                    return

            if config.totalRecordSize == 0:
                config.totalRecordSize = total_record_size
            elif config.totalRecordSize < total_record_size:
                raise ValueError('specified total record size is smaller than needed {} < {}'.format(config.totalRecordSize,
                                                                                                     total_record_size))
            old_aimed_buffer_size = config.aimedBufferSize

            config.aimedBufferSize, config.totalRecordSize = self.buffer_strategy.calculate_acquisition_properties(config.masks, self.__card.minimum_record_size)
            config.apply(self.__card, True)

            # Keep user value
            config.aimedBufferSize = old_aimed_buffer_size

            self.update_settings = False
            self.__armed_program = to_arm
        self.__card.startAcquisition(1)

    def delete_program(self, program_name: str) -> None:
        self._registered_programs.pop(program_name)
        # todo [2018-06-14]: what if program to delete is currently armed?

    def clear(self) -> None:
        self._registered_programs = dict()
        self.__armed_program = None

    @property
    def mask_prototypes(self) -> Dict[str, Tuple[int, str]]:
        return self._mask_prototypes

    def register_mask_for_channel(self, mask_id: str, hw_channel: int, mask_type='auto') -> None:
        """

        Args:
            mask_id: Identifier of the measurement windows
            hw_channel: Associated hardware channel (0, 1, 2, 3)
            mask_type: Either 'auto' or 'periodical
        """
        if hw_channel not in range(4):
            raise ValueError('{} is not a valid hw channel'.format(hw_channel))
        if mask_type not in ('auto', 'cross_buffer', None):
            raise NotImplementedError('Currently only can do cross buffer mask')
        self._mask_prototypes[mask_id] = (hw_channel, mask_type)

