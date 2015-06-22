"""STANDARD LIBRARY IMPORTS"""
from abc import ABCMeta, abstractmethod

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from pulses.Pulse import Pulse

class Waveform(metaclass = ABCMeta):
    """!@brief Waveform representation of a pulse. Interface to hardware.
    """
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def get_current_tick(self) -> int:
        """!@brief Return the index of the next unused tick (i.e. the amount of currently used ticks)."""
        pass
        
    @abstractmethod
    def set(self, time: int, voltage: float) -> None:
        """!brief Set the voltage for the tick specified by time.
        
        If time is greater than the index of the next unused tick, the resulting
        gap is filled with the value of the previous tick."""
        pass
        
    @abstractmethod
    def add(self, voltage: float) -> None:
        """!@brief Set the voltage for the next tick."""
        pass
    
    @abstractmethod
    def interpolate_to(self, time: int, voltage: float) -> None:
        """!@brief Interpolate voltages for the range from the last tick to time to the given value."""
        pass
        
    @abstractmethod
    def get_handle(self) -> int:
        """!@brief Return the unique identifier for the generated waveform."""
        pass
        
    def get_tick_length(self) -> int:
        """!@brief Return the length of a single tick in microseconds"""
        pass
        
    
class PulseHardwareUploadInterface(metaclass = ABCMeta):
    """!@brief An interface to upload pulses to some pulse playback hardware.
    
    Pulses are compiled into a hardware-understood format represented by Waveform objects.
    PulseHardwareUploadInterface features an instruction queue which is initially empty.
    The methods enqueue, configureBranch and prepareHalt prepare the execution of pulses by
    adding relevant instructions to the end of this queue."""

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def obtain_waveform(pulse: Pulse) -> Waveform:
        """!@brief Return the Waveform object associated with the given Pulse object.
        
        If no Waveform is associated with the given pulse, a new one is created.
        """
        pass
    
    @abstractmethod
    def enqueue(waveform: Waveform) -> None:
        """!@brief Enqueues a waveform for sequential execution at the end of the instruction queue."""
        pass
        
    @abstractmethod
    def configureBranch(ifWaveform: Waveform, elseWaveform: Waveform, conditionHandle: int) -> None:
        """!@brief Configures the hardware to branch at the end of the current instruction queue."""
        pass
        
    @abstractmethod
    def prepareHalt() -> None:
        """!@brief Configures the hardware to (temporarily) stop playback after the current instruction queue was executed."""
        pass