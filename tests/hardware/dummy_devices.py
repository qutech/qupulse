from typing import Tuple, Set, Dict
from collections import deque


from qupulse.hardware.awgs.base import AWG, ProgramOverwriteException
from qupulse.hardware.dacs import DAC

class DummyDAC(DAC):
    def __init__(self):
        self._measurement_windows = dict()
        self._operations = dict()
        self.measured_data = deque([])
        self._meas_masks = {}
        self._armed_program = None

    @property
    def armed_program(self):
        return self._armed_program

    def register_measurement_windows(self, program_name: str, windows: Dict[str, Tuple['numpy.ndarray',
                                                                                       'numpy.ndarray']]):
        self._measurement_windows[program_name] = windows

    def register_operations(self, program_name: str, operations):
        self._operations[program_name] = operations

    def arm_program(self, program_name: str):
        self._armed_program = program_name

    def delete_program(self, program_name):
        if program_name in self._operations:
            self._operations.pop(program_name)
        if program_name in self._measurement_windows:
            self._measurement_windows.pop(program_name)

    def clear(self) -> None:
        self._measurement_windows = dict()
        self._operations = dict()
        self._armed_program = None

    def measure_program(self, channels):
        return self.measured_data.pop()

    def set_measurement_mask(self, program_name, mask_name, begins, lengths) -> Tuple['numpy.ndarray', 'numpy.ndarray']:
        self._meas_masks.setdefault(program_name, {})[mask_name] = (begins, lengths)
        return begins, lengths


class DummyAWG(AWG):
    """Dummy AWG for debugging purposes."""

    def __init__(self,
                 memory: int=100,
                 sample_rate: float=10,
                 output_range: Tuple[float, float]=(-5, 5),
                 num_channels: int=1,
                 num_markers: int=1) -> None:
        """Create a new DummyAWG instance.

        Args:
            memory (int): Available memory slots for waveforms. (default = 100)
            sample_rate (float): The sample rate of the dummy. (default = 10)
            output_range (float, float): A (min,max)-tuple of possible output values.
                (default = (-5,5)).
        """
        super().__init__(identifier="DummyAWG{0}".format(id(self)))

        self._programs = {} # contains program names and programs
        self._sample_rate = sample_rate
        self._output_range = output_range
        self._num_channels = num_channels
        self._num_markers = num_markers
        self._channels = ('default',)
        self._armed = None

        # todo [2018-06-14]: The following attributes (and thus the memory argument) are never used. Remove?
        self._waveform_memory = [None for i in range(memory)]
        self._waveform_indices = {}  # dict that maps from waveform hash to memory index
        self._program_wfs = {}  # contains program names and necessary waveforms indices

    def set_volatile_parameters(self, program_name: str, parameters):
        raise NotImplementedError()

    def upload(self, name, program, channels, markers, voltage_transformation, force=False) -> None:
        if name in self.programs:
            if not force:
                raise ProgramOverwriteException(name)
            else:
                self.remove(name)
                self.upload(name, program)
        else:
            self._programs[name] = (program, channels, markers, voltage_transformation)

    def remove(self, name) -> None:
        if name in self.programs:
            self._programs.pop(name)

    def clear(self) -> None:
        self._programs = {}

    def arm(self, name: str) -> None:
        self._armed = name

    @property
    def programs(self) -> Set[str]:
        return set(self._programs.keys())

    @property
    def output_range(self) -> Tuple[float, float]:
        return self._output_range

    @property
    def identifier(self) -> str:
        return "DummyAWG{0}".format(id(self))

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def num_markers(self):
        return self._num_markers