from typing import Union, Dict, NamedTuple, List, Any, Optional, Tuple
from collections import deque, defaultdict

import numpy as np

from atsaverage.config import ScanlineConfiguration
from atsaverage.masks import CrossBufferMask, Mask
from atsaverage.operations import OperationDefinition

from qctoolkit.hardware.dacs import DAC


class AlazarProgram:
    def __init__(self, masks=list(), operations=list(), total_length=None):
        self.masks = masks
        self.operations = operations
        self.total_length = total_length
    def __iter__(self):
        yield self.masks
        yield self.operations
        yield self.total_length


class AlazarCard(DAC):
    def __init__(self, card, config: Optional[ScanlineConfiguration]=None):
        self.__card = card

        self.__definitions = dict()
        self.config = config

        self._mask_prototypes = dict()  # type: Dict[str, Tuple[Any, str]]

        self._registered_programs = defaultdict(AlazarProgram)  # type: Dict[str, AlazarProgram]

    @property
    def card(self) -> Any:
        return self.__card

    def __make_mask(self, mask_id: str, begins, lengths) -> Mask:
        if mask_id not in self._mask_prototypes:
            raise KeyError('Measurement window {} can not be converted as it is not registered.'.format(mask_id))

        hardware_channel, mask_type = self._mask_prototypes[mask_id]
        if mask_type not in ('auto', 'cross_buffer', None):
            raise ValueError('Currently only can do cross buffer mask')

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
            return
        total_length = 0
        for mask_id, (begins, lengths) in windows.items():

            sample_factor = self.config.captureClockConfiguration.numeric_sample_rate(self.__card.model) / 10**9

            begins = np.rint(begins*sample_factor).astype(dtype=np.uint64)
            lengths = np.floor(lengths*sample_factor).astype(dtype=np.uint64)

            sorting_indices = np.argsort(begins)
            begins = begins[sorting_indices]
            lengths = lengths[sorting_indices]

            windows[mask_id] = (begins, lengths)
            total_length = max(total_length, begins[-1]+lengths[-1])

        total_length = np.ceil(total_length/self.__card.minimum_record_size) * self.__card.minimum_record_size

        self._registered_programs[program_name].masks = [
            self.__make_mask(mask_id, *window_begin_length)
            for mask_id, window_begin_length in windows.items()]
        self._registered_programs[program_name].total_length = total_length

    def register_operations(self, program_name: str, operations) -> None:
        self._registered_programs[program_name].operations = operations

    def arm_program(self, program_name: str) -> None:
        config = self.config
        config.masks, config.operations, total_record_size = self._registered_programs[program_name]

        if config.totalRecordSize == 0:
            config.totalRecordSize = total_record_size
        elif config.totalRecordSize < total_record_size:
            raise ValueError('specified total record size is smaller than needed {} < {}'.format(config.totalRecordSize,
                                                                                                 total_record_size))

        config.apply(self.__card, True)
        self.__card.startAcquisition(1)

    def delete_program(self, program_name: str) -> None:
        self.__registered_operations.pop(program_name, None)
        self.__registered_masks.pop(program_name, None)

    @property
    def mask_prototypes(self):
        return self._mask_prototypes

    def register_mask_for_channel(self, mask_id, hw_channel, mask_type='auto'):
        self._mask_prototypes[mask_id] = (hw_channel, mask_type)

