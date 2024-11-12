from typing import Iterable, Dict, Tuple, Union
import logging

import time

import numpy

from qupulse.hardware.dacs.dac_base import DAC

from atssimple import acquire_sample_rates_time_windows, ATSSimpleCard

logger = logging.getLogger(__name__)


class ATSSimpleCard(ATSSimpleCard):
    def __init__(
        self,
        board_ids: Tuple[int, int] = (1, 1),
        samples_per_second: float = 125_000_000,
        channel_mask: int = 0b1111,
        voltage_range: Union[float, Tuple[float, float, float, float]] = 1.0,
    ):
        """
        QuPulse DAC interface for ATSSimple.

        Args:
            board_ids, (int, int) (optional, default: (1, 1)):
                systemId, boardId to select the alazar card
            samples_per_second, float (optional, default: 125_000_000):
                Sample rate configured on board.
            channel_mask, int (optional, default: 0b1111):
                Bitmap representing the channels to be acquired.
                0b0001 = Channel A
                0b0010 = Channel B
                0b0100 = Channel C
                0b1000 = Channel D
            voltage_range, float or 4-tuple of floats (optional, default: 1.0):
                The voltage ranges of each channel.
        """

        super().__init__(
            acquisition_function=acquire_sample_rates_time_windows, board_ids=board_ids
        )

        self.samples_per_second = samples_per_second
        self.channel_mask = channel_mask
        self.voltage_range = voltage_range

        self.current_program = None
        self.registered_programs = {}

        self._armed_sample_windows = None
        self._armed_window_names = None

        self._results_raw = None
        self._samples_raw = None

        self._results = {}

    def _pad_and_validate_measurement_windows(
        self, windows: Dict[str, Tuple[numpy.ndarray, numpy.ndarray]]
    ) -> Dict[str, Tuple[numpy.ndarray, numpy.ndarray]]:
        """
        Only non-overlapping measurement windows are allowed. Gaps are padded with padding windows.
        """

        # Strip previous padding
        windows["_padding"] = [numpy.array([]), numpy.array([])]

        # Collect all windows and discard names
        windows_flat = [numpy.array([]), numpy.array([])]
        for k, v in windows.items():
            windows_flat[0] = numpy.append(windows_flat[0], v[0])
            windows_flat[1] = numpy.append(windows_flat[1], v[1])

        # Sort by window starts
        args = numpy.argsort(windows_flat[0])
        windows_flat[0] = windows_flat[0][args]
        windows_flat[1] = windows_flat[1][args]

        padding_windows = [numpy.array([]), numpy.array([])]
        # Prepend padding in case that the first measurement window starts with some delay
        if windows_flat[0][0] > 0:
            padding_windows[0] = numpy.append(
                padding_windows[0], 0
            )
            padding_windows[1] = numpy.append(
                padding_windows[1],
                windows_flat[0][0]
            )

        # Pad between windows
        for index in range(len(windows_flat[0]) - 1):
            # Raise error if windows overlap
            if (
                windows_flat[0][index] + windows_flat[1][index]
                > windows_flat[0][index + 1]
            ):
                raise ValueError("Overlapping measurement windows not allowed!")

            # Calculate necessary padding
            if (
                windows_flat[0][index] + windows_flat[1][index]
                < windows_flat[0][index + 1]
            ):
                padding_windows[0] = numpy.append(
                    padding_windows[0], windows_flat[0][index] + windows_flat[1][index]
                )
                padding_windows[1] = numpy.append(
                    padding_windows[1],
                    windows_flat[0][index + 1]
                    - (windows_flat[0][index] + windows_flat[1][index]),
                )

        windows["_padding"] = padding_windows

        return windows

    def _smallest_compatible_sample_rate(
        self, window_length: float, sample_rate: float
    ):
        if sample_rate < 1e-6:
            raise RuntimeError("Could not find sample rate for too short window!")

        if (sample_rate / 10) * window_length > 1:
            return self._smallest_compatible_sample_rate(
                window_length, sample_rate / 10
            )
        else:
            return sample_rate

    def register_measurement_windows(
        self, program_name: str, windows: Dict[str, Tuple[numpy.ndarray, numpy.ndarray]]
    ) -> None:

        self.registered_programs[program_name] = {
            "windows": (self._pad_and_validate_measurement_windows(windows))
        }

    def set_measurement_mask(
        self,
        program_name: str,
        mask_name: str,
        begins: numpy.ndarray,
        lengths: numpy.ndarray,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:

        windows = self.registered_programs[program_name]["windows"].copy()
        windows[mask_name] = (
            begins,
            lengths,
        )

        self.registered_programs[program_name]["windows"] = (
            self._pad_and_validate_measurement_windows(windows)
        )

    def register_operations(
        self, program_name: str, operations: Dict[str, float]
    ) -> None:
        """
        Operations: {"mask1": sample_rate1, "mask2": sample_rate2, ...}
        """

        if not "_padding" in operations.keys():
            operations["_padding"] = 125000000  # 125000000 Hz acquisition padding

        self.registered_programs[program_name]["operations"] = operations

    def arm_program(self, program_name: str) -> None:
        if not program_name in self.registered_programs.keys():
            raise ValueError(f'"{program_name}" not registered!')
        self.current_program = program_name

        # Collect all windows and discard names
        windows_flat = [numpy.array([]), numpy.array([]), numpy.array([])]
        for k, v in self.registered_programs[program_name]["windows"].items():
            windows_flat[0] = numpy.append(windows_flat[0], v[0])
            windows_flat[1] = numpy.append(windows_flat[1], v[1])
            windows_flat[2] = numpy.append(
                windows_flat[2], numpy.array(len(v[0]) * [k])
            )

        # Sort by window starts
        args = numpy.argsort(windows_flat[0])
        windows_flat[0] = windows_flat[0][args]
        windows_flat[1] = windows_flat[1][args]
        windows_flat[2] = windows_flat[2][args]

        # Compile acquisition parameters
        window_names = windows_flat[2]
        window_lengths = windows_flat[1] / 1e09  # In sec.
        sample_rates = []
        for i, window_name in enumerate(window_names):
            # If measurement would result in 0 samples due to too small sample rate,
            # modify sample rate such that at least on sample is acquired.
            if (
                1.0
                / (
                    sample_rate := self.registered_programs[program_name]["operations"][
                        window_name
                    ]
                )
                <= window_lengths[i]
            ):
                sample_rates.append(sample_rate)
            else:
                sample_rates.append(
                    self._smallest_compatible_sample_rate(
                        window_lengths[i], self.samples_per_second
                    )
                )

        self._armed_sample_windows = numpy.array([window_lengths, sample_rates]).T
        self._armed_window_names = window_names

        self._results = {}
        self._results_raw = {}
        self._samples_raw = {}

        # Start Acquisition
        self.start_acquisition(
            sample_rates=self._armed_sample_windows,
            channel_mask=self.channel_mask,
            samples_per_second=self.samples_per_second,
            voltage_range=self.voltage_range,
            return_samples_in_seconds=True,
        )

        # Additional wait to get acquisition ready before continuing
        time.sleep(0.1)

    def delete_program(self, program_name: str) -> None:
        self.registered_programs.pop(program_name)

    def clear(self) -> None:
        self.registered_programs.clear()

    def measure_program(
        self, channels: Iterable[str] = None
    ) -> Dict[str, numpy.ndarray]:
        if self.current_program == None:
            raise RuntimeError("No programm armed yet!")

        # Collect thread and data
        self._acquisition_process.join()
        self._results_raw, self._samples_raw = self._result_queue.get(timeout=1)

        # Sort results by window
        total_samples = 0
        window_boundries = numpy.insert(numpy.cumsum(self._armed_sample_windows[:, 0]), 0, 0)
        for i, window_name in enumerate(self._armed_window_names):
            samples = self._samples_raw[numpy.where((self._samples_raw >= window_boundries[i]) * (self._samples_raw < window_boundries[i+1]))[0]]
            results = self._results_raw[:, numpy.where((self._samples_raw >= window_boundries[i]) * (self._samples_raw < window_boundries[i+1]))[0]]

            if self._results.get(window_name) == None:
                self._results[window_name] = []

            self._results[window_name] += [[samples, results]]
            total_samples += len(samples)

        # Compile result dict
        result_dict = {}
        for channel in channels:
            result_dict[channel] = self._results[channel]

        return result_dict
