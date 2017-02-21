from typing import Union, Dict, NamedTuple, List, Any
from collections import deque

import numpy as np

from atsaverage.config import ScanlineConfiguration
from atsaverage.masks import CrossBufferMask, Mask
from atsaverage.operations import OperationDefinition

from qctoolkit.hardware.dacs import DAC


class AlazarProgram:
    def __init__(self, masks=None, operations=None, total_length=None):
        self.masks = masks
        self.operations = operations
        self.total_length = total_length
    def __iter__(self):
        yield self.masks
        yield self.operations
        yield self.total_length


class AlazarCard(DAC):
    def __init__(self, card, config: Union[ScanlineConfiguration, None] = None):
        self.__card = card

        self.__definitions = dict()
        self.config = config

        self.__mask_prototypes = dict()  # type: Dict[str, Tuple[Any, str]]

        self.__registered_programs = dict()  # type: Dict[str, AlazarProgram]

    @property
    def card(self) -> Any:
        return self.__card

    def __make_mask(self, mask_id: str, window_deque: deque) -> Mask:
        if mask_id not in self.__mask_prototypes:
            raise KeyError('Measurement window {} can not be converted as it is not registered.'.format(mask_id))

        hardware_channel, mask_type = self.__mask_prototypes[mask_id]
        if mask_type not in ('auto', 'cross_buffer', None):
            raise ValueError('Currently only can do cross buffer mask')
        begins_lengths = np.asarray(window_deque)

        begins = begins_lengths[:, 0]
        lengths = begins_lengths[:, 1]

        sorting_indices = np.argsort(begins)
        begins = begins[sorting_indices]
        lengths = lengths[sorting_indices]

        if np.any(begins[:-1]+lengths[:-1] >= begins[1:]):
            raise ValueError('Found overlapping windows in begins')

        mask = CrossBufferMask()
        mask.begin = begins
        mask.length = lengths
        mask.channel = hardware_channel
        return mask

    def register_measurement_windows(self, program_name: str, windows: Dict[str, deque]) -> None:
        for mask_id, window_deque in windows.items():
            begins_lengths = np.asarray(window_deque)
            begins = begins_lengths[:, 0]
            lengths = begins_lengths[:, 1]
            sorting_indices = np.argsort(begins)
            begins = begins[sorting_indices]
            lengths = lengths[sorting_indices]

            windows[mask_id] = (begins, lengths)
            total_length = max(total_length, begins[-1]+lengths[-1])

        total_length = np.ceil(total_length/self.__card.minimum_record_size) * self.__card.minimum_record_size

        self.__registered_programs.get(program_name,
                                       default=AlazarProgram()).masks = [
            self.__make_mask(mask_id, window_deque)
            for mask_id, window_deque in windows.items()]
        self.__registered_programs[program_name].total_length = total_length

    def register_operations(self, program_name: str, operations) -> None:
        self.__registered_programs.get(program_name,
                                       default=AlazarProgram()
                                       ).operations = self.__registered_programs.get(program_name, self)

    def arm_program(self, program_name: str) -> None:
        config = self.config
        config.masks, config.operations, total_record_size = self.__registered_programs[program_name]

        if config.totalRecordSize is None:
            config.totalRecordSize = total_record_size
        elif config.totalRecordSize < total_record_size:
            raise ValueError('specified total record size is smaller than needed {} < {}'.format(config.totalRecordSize,
                                                                                                 total_record_size))

        config.apply(self.__card)
        self.__card.startAcquisition(1)

    def delete_program(self, program_name: str) -> None:
        self.__registered_operations.pop(program_name, None)
        self.__registered_masks.pop(program_name, None)
