from typing import Dict, Any, Optional, Tuple, List, Iterable
from collections import defaultdict

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


class AlazarCard(DAC):
    def __init__(self, card, config: Optional[ScanlineConfiguration]=None):
        self.__card = card

        self.__armed_program = None
        self.update_settings = True

        self.__definitions = dict()
        self.config = config

        self._mask_prototypes = dict()  # type: Dict

        self._registered_programs = defaultdict(AlazarProgram)  # type: Dict[str, AlazarProgram]

    @property
    def card(self) -> Any:
        return self.__card

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
            total_length = max(total_length, begins[-1]+lengths[-1])

        total_length = np.ceil(total_length/self.__card.minimum_record_size) * self.__card.minimum_record_size

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
            
            # work around for measurments not working with one buffer
            if config.totalRecordSize < 5*config.aimedBufferSize:
                config.aimedBufferSize = config.totalRecordSize // 5

            config.apply(self.__card, True)

            # "Hide" work around from the user
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

    def measure_program(self, channels: Iterable[str]) -> Dict[str, np.ndarray]:
        """
        Get all measurements at once and write them in a dictionary.
        """

        scanline_data = self.__card.extractNextScanline()

        scanline_definition = scanline_data.definition
        operation_definitions = {operation.identifier: operation
                                 for operation in scanline_definition.operations}
        mask_definitions = {mask.identifier: mask
                            for mask in scanline_definition.masks}

        def get_input_range(operation_id: str):
            # currently does not work for ComputeMomentDefinition :(
            mask_id = operation_definitions[operation_id].maskID

            hw_channel = int(mask_definitions[mask_id].channel)

            # This fails if new changes have been applied to the card in the meantime
            # It is better than self.config.inputConfiguration but still
            return self.__card.scanConfiguration.inputConfiguration[hw_channel].inputRange

        data = {}
        for op_name in channels:
            input_range = get_input_range(op_name)
            data[op_name] = scanline_data.operationResults[op_name].getAsVoltage(input_range)

        return data
