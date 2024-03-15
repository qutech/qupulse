from .base import AWG

class DummyAWG(AWG):
    """Dummy AWG for debugging purposes."""

    def __init__(self,
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
