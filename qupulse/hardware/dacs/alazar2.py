from typing import Union, Iterable, Dict, Tuple, Mapping, Optional
from types import MappingProxyType
import logging

import numpy

from qupulse.utils.types import TimeType
from qupulse.hardware.dacs.dac_base import DAC
from qupulse.hardware.dacs.alazar import AlazarProgram

import atsaverage
from atsaverage.masks import make_best_mask
from atsaverage.config2 import BoardConfiguration, create_scanline_definition, BufferStrategySettings


logger = logging.getLogger(__name__)


class AlazarCard(DAC):
    def __init__(self, atsaverage_card: 'atsaverage.core.AlazarCard'):
        super().__init__()
        self._atsaverage_card = atsaverage_card
        self._registered_programs = {}

        # for auto retrigger
        self._armed_program: Optional[str] = None
        self._remaining_auto_triggers = 0

        # for debugging purposes
        self._raw_data_mask = None
        self.default_buffer_strategy: Optional[BufferStrategySettings] = None

    @property
    def atsaverage_card(self):
        return self._atsaverage_card

    @property
    def registered_programs(self) -> Mapping[str, AlazarProgram]:
        return MappingProxyType(self._registered_programs)

    @property
    def current_sample_rate_in_giga_herz(self) -> TimeType:
        numeric_sample_rate = self._atsaverage_card.board_configuration_cache.get_numeric_sample_rate()
        if numeric_sample_rate is None:
            raise RuntimeError("The sample rate was not set yet. The instrument does not support retrieving the sample "
                               "rate via an API. We need to cache a set command.")
        return TimeType.from_fraction(numeric_sample_rate, 10 ** 9)

    def get_current_input_range(self, channel: Union[str, int]):
        input_range = self._atsaverage_card.board_configuration_cache.get_channel_input_range(channel)
        if input_range is None:
            raise RuntimeError("The input range was not set yet. The instrument does not support retrieving the input "
                               "range via an API. We need to cache a set command.")
        return input_range

    def register_measurement_windows(self, program_name: str, windows: Dict[str, Tuple[numpy.ndarray,
                                                                                       numpy.ndarray]]) -> None:
        program = self._registered_programs.setdefault(program_name, AlazarProgram())
        sample_rate = self.current_sample_rate_in_giga_herz
        program.clear_masks()
        for mask_name, (begins, lengths) in windows.items():
            program.set_measurement_mask(mask_name, sample_rate, begins, lengths)

    def set_measurement_mask(self, program_name: str, mask_name: str,
                             begins: numpy.ndarray, lengths: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        program = self._registered_programs.setdefault(program_name, AlazarProgram())
        return program.set_measurement_mask(mask_name, self.current_sample_rate_in_giga_herz, begins, lengths)

    def register_operations(self, program_name: str, operations) -> None:
        self._registered_programs.setdefault(program_name, AlazarProgram()).operations = operations

    def _make_scanline_definition(self, program: AlazarProgram):
        sample_rate_in_ghz = self.current_sample_rate_in_giga_herz
        sample_rate_in_hz = int(sample_rate_in_ghz * 10 ** 9)

        masks = program.masks(make_best_mask)
        if sample_rate_in_ghz != program.sample_rate:
            raise RuntimeError("Masks were registered with a different sample rate")
        return create_scanline_definition(masks, program.operations,
                                          raw_data_mask=self._raw_data_mask,
                                          board_spec=self._atsaverage_card.get_board_spec(),
                                          buffer_strategy=program.buffer_strategy,
                                          numeric_sample_rate=sample_rate_in_hz)

    def _prepare_program(self, program: AlazarProgram):
        scanline_definition = self._make_scanline_definition(program)
        self._atsaverage_card.configureMeasurement(scanline_definition)

    def arm_program(self, program_name: str) -> None:
        logger.debug("Arming program %s on %r", program_name, self._atsaverage_card)

        if program_name == self._armed_program and self._remaining_auto_triggers > 0:
            logger.info("Relying on atsaverage auto-arm with %d auto triggers remaining after this one: %d",
                        self._remaining_auto_triggers)

        else:
            program = self._registered_programs[program_name]
            scanline_definition = self._make_scanline_definition(program)

            self._atsaverage_card.configureMeasurement(scanline_definition)

            self._atsaverage_card.startAcquisition(program.auto_rearm_count)
            self._remaining_auto_triggers = program.auto_rearm_count - 1

    def delete_program(self, program_name: str) -> None:
        self._registered_programs.pop(program_name)

    def clear(self) -> None:
        self._registered_programs.clear()

    def measure_program(self, channels: Iterable[str] = None) -> Dict[str, numpy.ndarray]:
        scanline_data = self._atsaverage_card.extractNextScanline()

        if channels is None:
            channels = scanline_data.operationResults.keys()

        scanline_definition = scanline_data.definition
        operation_definitions = {operation.identifier: operation
                                 for operation in scanline_definition.operations}
        mask_definitions = {mask.identifier: mask
                            for mask in scanline_definition.masks}

        def get_input_range(operation_id: str):
            mask_id = operation_definitions[operation_id].maskID
            hw_channel = int(mask_definitions[mask_id].channel)
            return self.get_current_input_range(hw_channel)

        data = {}
        for op_name in channels:
            input_range = get_input_range(op_name)
            data[op_name] = scanline_data.operationResults[op_name].getAsVoltage(input_range)
        return data
