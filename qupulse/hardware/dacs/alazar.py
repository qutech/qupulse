import dataclasses
from typing import Dict, Any, Optional, Tuple, List, Iterable, Callable, Sequence
from collections import defaultdict
import copy
import warnings
import math
import functools
import abc
import logging

import numpy as np

from atsaverage.config import ScanlineConfiguration
from atsaverage.masks import CrossBufferMask, Mask

from qupulse.utils.types import TimeType
from qupulse.hardware.dacs.dac_base import DAC
from qupulse.hardware.util import traced
from qupulse.utils.performance import time_windows_to_samples

logger = logging.getLogger(__name__)


def _windows_to_samples(begins: np.ndarray, lengths: np.ndarray,
                        sample_rate: TimeType) -> Tuple[np.ndarray, np.ndarray]:
    return time_windows_to_samples(begins, lengths, float(sample_rate))


@dataclasses.dataclass
class AcquisitionProgram:
    _sample_rate: Optional[TimeType] = dataclasses.field(default=None)
    _masks: dict = dataclasses.field(default_factory=dict)

    @property
    def sample_rate(self) -> Optional[TimeType]:
        return self._sample_rate

    def set_measurement_mask(self, mask_name: str, sample_rate: TimeType,
                             begins: np.ndarray, lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Raise error if sample factor has changed"""
        if self._sample_rate is None:
            self._sample_rate = sample_rate
        elif sample_rate != self.sample_rate:
            raise RuntimeError('class AcquisitionProgram has already masks with differing sample rate.')

        assert begins.dtype == float and lengths.dtype == float

        begins, lengths = self._masks[mask_name] = _windows_to_samples(begins, lengths, sample_rate)

        return begins, lengths

    def clear_masks(self):
        self._masks.clear()
        self._sample_rate = None


@dataclasses.dataclass
class AlazarProgram(AcquisitionProgram):
    operations: Sequence = dataclasses.field(default_factory=list)
    _total_length: Optional[int] = dataclasses.field(default=None)
    _auto_rearm_count: int = dataclasses.field(default=1)
    buffer_strategy: Optional = dataclasses.field(default=None)

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

    @property
    def auto_rearm_count(self) -> int:
        """This is passed to AlazarCard.startAcquisition. The card will (re-)arm automatically for this many times."""
        return self._auto_rearm_count

    @auto_rearm_count.setter
    def auto_rearm_count(self, value: int):
        trigger_count = int(value)
        if trigger_count == 0:
            raise ValueError("Trigger count of 0 is not supported in qupulse (yet) because tracking the number of "
                             "remaining triggers is too hard in case of infinity :(")
        if not 0 < trigger_count < 2**64:
            raise ValueError("Trigger count has to be in the interval [0, 2**64-1]")
        self._auto_rearm_count = trigger_count

    def iter(self, mask_maker):
        yield self.masks(mask_maker)
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


@traced
class AlazarCard(DAC):
    def __init__(self, card, config: Optional[ScanlineConfiguration]=None):
        self.__card = card

        self.__armed_program = None
        self.update_settings = True

        self.__definitions = dict()

        # this ScanlineConfig is used by default for each program
        # masks and operations are overwritten
        self.default_config = config

        # the currently active ScanlineConfig
        self._current_config = None

        self._buffer_strategy = None

        self._remaining_auto_triggers = 0

        self._mask_prototypes = dict()  # type: Dict

        self._registered_programs = defaultdict(AlazarProgram)  # type: Dict[str, AlazarProgram]

        # defaults to self.__card.minimum_record_size if None
        # we use a page size here because this is allocated anyways for a buffer
        # This might lead to problems with small sample rates
        self._record_size_factor = 1024 * 4

    @property
    def card(self) -> Any:
        return self.__card

    @property
    def record_size_factor(self) -> int:
        """The total record size of each measurement gets extended to be a multiple of this. None means that the
        minimal value supported by the card is taken."""
        if self._record_size_factor is None:
            return self.__card.minimum_record_size
        else:
            return self._record_size_factor

    @record_size_factor.setter
    def record_size_factor(self, value: Optional[int]):
        self._record_size_factor = value

    @property
    def config(self):
        warnings.warn("AlazarCard.config is deprecated. Use AlazarCard.default_config or AlazarCard.current_config",
                      DeprecationWarning)
        if self._current_config is None:
            return self.default_config
        else:
            return self._current_config

    @property
    def current_config(self):
        return self._current_config

    @property
    def buffer_strategy(self) -> BufferStrategy:
        if self._buffer_strategy is None:
            return AvoidSingleBufferAcquisition(ForceBufferSize(self.default_config.aimedBufferSize))
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

        if mask_type not in ('auto', 'cross_buffer', None):
            warnings.warn("Currently only CrossBufferMask is implemented.")

        if np.any(begins[:-1]+lengths[:-1] > begins[1:]):
            raise ValueError('Found overlapping windows in begins')

        mask = CrossBufferMask()
        mask.identifier = mask_id
        mask.begin = begins
        mask.length = lengths
        mask.channel = hardware_channel
        return mask

    def set_measurement_mask(self, program_name, mask_name, begins, lengths) -> Tuple[np.ndarray, np.ndarray]:
        sample_rate = TimeType.from_fraction(int(self.default_config.captureClockConfiguration.numeric_sample_rate(self.card.model)), 10**9)
        return self._registered_programs[program_name].set_measurement_mask(mask_name, sample_rate, begins, lengths)

    def register_measurement_windows(self,
                                     program_name: str,
                                     windows: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        program = self._registered_programs[program_name]
        sample_rate = TimeType.from_fraction(int(self.default_config.captureClockConfiguration.numeric_sample_rate(self.card.model)),
                                               10 ** 9)
        program.clear_masks()

        for mask_name, (begins, lengths) in windows.items():
            program.set_measurement_mask(mask_name, sample_rate, begins, lengths)

    def register_operations(self, program_name: str, operations) -> None:
        self._registered_programs[program_name].operations = operations

    def arm_program(self, program_name: str) -> None:
        logger.debug("Arming program %s on %r", program_name, self.__card)

        to_arm = self._registered_programs[program_name]
        if self.update_settings or self.__armed_program is not to_arm:
            logger.info("Arming %r by calling applyConfiguration. Update settings flag: %r",
                        self.__card, self.update_settings)

            config = copy.deepcopy(self.default_config)
            config.masks, config.operations, total_record_size = self._registered_programs[program_name].iter(
                self._make_mask)

            sample_rate_in_hz = config.captureClockConfiguration.numeric_sample_rate(self.card.model)

            # sample rate in GHz
            sample_rate = TimeType.from_fraction(sample_rate_in_hz, 10 ** 9)

            if not config.operations:
                raise RuntimeError("No operations: Arming program without operations is an error as there will "
                                   "be no result: %r" % program_name)

            elif not config.masks:
                raise RuntimeError("No masks although there are operations in program: %r" % program_name)

            elif self._registered_programs[program_name].sample_rate != sample_rate:
                raise RuntimeError("Masks were registered with a different sample rate {}!={}".format(
                    self._registered_programs[program_name].sample_rate, sample_rate))

            assert total_record_size > 0

            # extend the total record size to be a multiple of record_size_factor
            record_size_factor = self.record_size_factor
            total_record_size = (((total_record_size - 1) // record_size_factor) + 1) * record_size_factor

            if config.totalRecordSize == 0:
                config.totalRecordSize = total_record_size
            elif config.totalRecordSize < total_record_size:
                raise ValueError('specified total record size is smaller than needed {} < {}'.format(config.totalRecordSize,
                                                                                                     total_record_size))
            self.__card.applyConfiguration(config, True)
            self._current_config = config

            self.update_settings = False
            self.__armed_program = to_arm

        elif self.__armed_program is to_arm and self._remaining_auto_triggers > 0:
            self._remaining_auto_triggers -= 1
            logger.info("Relying on atsaverage auto-arm with %d auto triggers remaining after this one",
                        self._remaining_auto_triggers)
            return

        self.__card.startAcquisition(to_arm.auto_rearm_count)
        self._remaining_auto_triggers = to_arm.auto_rearm_count - 1

    def delete_program(self, program_name: str) -> None:
        self._registered_programs.pop(program_name)
        # todo [2018-06-14]: what if program to delete is currently armed?

    def clear(self) -> None:
        self._registered_programs.clear()
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
