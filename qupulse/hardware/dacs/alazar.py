from typing import Dict, Any, Optional, Tuple, List, Iterable, Callable
from collections import defaultdict

import numpy as np

from atsaverage.config import ScanlineConfiguration
from atsaverage.masks import CrossBufferMask, Mask

from qupulse.utils.types import TimeType
from qupulse.hardware.dacs.dac_base import DAC


class AlazarProgram:
    def __init__(self):
        self._sample_factor = None
        self._masks = {}
        self.operations = []
        self._total_length = None

    def masks(self, mask_maker: Callable[[str, np.ndarray, np.ndarray], Mask]) -> List[Mask]:
        return [mask_maker(mask_name, *data) for mask_name, data in self._masks.items()]

    @property
    def total_length(self) -> int:
        if not self._total_length:
            total_length = 0
            for begins, lengths in self._masks.values():
                total_length = max(begins[-1] + lengths[-1], total_length)

            return total_length
        else:
            return self._total_length

    @total_length.setter
    def total_length(self, val: int):
        self._total_length = val

    def clear_masks(self):
        self._masks.clear()

    @property
    def sample_factor(self) -> Optional[TimeType]:
        return self._sample_factor

    def set_measurement_mask(self, mask_name: str, sample_factor: TimeType,
                             begins: np.ndarray, lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Raise error if sample factor has changed"""
        if self._sample_factor is None:
            self._sample_factor = sample_factor

        elif sample_factor != self._sample_factor:
            raise RuntimeError('class AlazarProgram has already masks with differing sample factor')

        assert begins.dtype == np.float and lengths.dtype == np.float

        # optimization potential here (hash input?)
        begins = np.rint(begins * float(sample_factor)).astype(dtype=np.uint64)
        lengths = np.floor_divide(lengths * float(sample_factor.numerator), float(sample_factor.denominator)).astype(dtype=np.uint64)

        sorting_indices = np.argsort(begins)
        begins = begins[sorting_indices]
        lengths = lengths[sorting_indices]

        begins.flags.writeable = False
        lengths.flags.writeable = False

        self._masks[mask_name] = begins, lengths

        return begins, lengths

    def iter(self, mask_maker):
        yield self.masks(mask_maker)
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

    def set_measurement_mask(self, program_name, mask_name, begins, lengths) -> Tuple[np.ndarray, np.ndarray]:
        sample_factor = TimeType(int(self.config.captureClockConfiguration.numeric_sample_rate(self.card.model)), 10**9)
        return self._registered_programs[program_name].set_measurement_mask(mask_name, sample_factor, begins, lengths)

    def register_measurement_windows(self,
                                     program_name: str,
                                     windows: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        program = self._registered_programs[program_name]
        sample_factor = TimeType.from_fraction(int(self.config.captureClockConfiguration.numeric_sample_rate(self.card.model)),
                                 10 ** 9)
        program.clear_masks()

        for mask_name, (begins, lengths) in windows.items():
            program.set_measurement_mask(mask_name, sample_factor, begins, lengths)

    def register_operations(self, program_name: str, operations) -> None:
        self._registered_programs[program_name].operations = operations

    def arm_program(self, program_name: str) -> None:
        to_arm = self._registered_programs[program_name]
        if self.update_settings or self.__armed_program is not to_arm:
            config = self.config
            config.masks, config.operations, total_record_size = self._registered_programs[program_name].iter(
                self._make_mask)

            sample_factor = TimeType.from_fraction(self.config.captureClockConfiguration.numeric_sample_rate(self.card.model),
                                                   10 ** 9)

            if not config.operations:
                raise RuntimeError("No operations: Arming program without operations is an error as there will "
                                   "be no result: %r" % program_name)

            elif not config.masks:
                raise RuntimeError("No masks although there are operations in program: %r" % program_name)

            elif self._registered_programs[program_name].sample_factor != sample_factor:
                raise RuntimeError("Masks were registered with a different sample rate {}!={}".format(
                    self._registered_programs[program_name].sample_factor, sample_factor))

            assert total_record_size > 0

            minimum_record_size = self.__card.minimum_record_size
            total_record_size = (((total_record_size - 1) // minimum_record_size) + 1) * minimum_record_size

            if config.totalRecordSize == 0:
                config.totalRecordSize = total_record_size
            elif config.totalRecordSize < total_record_size:
                raise ValueError('specified total record size is smaller than needed {} < {}'.format(config.totalRecordSize,
                                                                                                     total_record_size))
            
            old_aimed_buffer_size = config.aimedBufferSize
            
            # work around for measurments not working with one buffer
            if config.totalRecordSize < 5*config.aimedBufferSize:
                config.aimedBufferSize = config.totalRecordSize // 5

            self.__card.applyConfiguration(config, True)

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
