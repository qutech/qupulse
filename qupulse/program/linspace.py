import abc
import contextlib
import dataclasses
import numpy as np
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, ContextManager, Iterable, Tuple, Union, Dict, List, Iterator, Generic,\
    Set as TypingSet, Callable
from enum import Enum
from itertools import dropwhile, count
from numbers import Real, Number
from collections import defaultdict


from qupulse import ChannelID, MeasurementWindow
from qupulse.parameter_scope import Scope, MappedScope, FrozenDict
# from qupulse.pulses.pulse_template import PulseTemplate
# from qupulse.pulses import ForLoopPT
from qupulse.program import ProgramBuilder, HardwareTime, HardwareVoltage, Waveform, RepetitionCount, TimeType
from qupulse.expressions.simple import SimpleExpression, NumVal, SimpleExpressionStepped
from qupulse.program.waveforms import MultiChannelWaveform, TransformingWaveform, WaveformCollection

from qupulse.program.transformation import ChainedTransformation, ScalingTransformation, OffsetTransformation,\
    IdentityTransformation, ParallelChannelTransformation, Transformation

# this resolution is used to unify increments
# the increments themselves remain floats
# !!! translated: this is NOT a hardware resolution,
# just a programmatic 'small epsilon' to avoid rounding errors.
DEFAULT_INCREMENT_RESOLUTION: float = 1e-9
DEFAULT_TIME_RESOLUTION: float = 1e-3

class DepDomain(Enum):
    VOLTAGE = 0
    TIME_LIN = -1
    TIME_LOG = -2
    FREQUENCY = -3
    WF_SCALE = -4
    WF_OFFSET = -5
    STEP_INDEX = -6
    NODEP = None


class InstanceCounterMeta(type):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls._instance_tracker = {}

    def __call__(cls, *args, **kwargs):
        normalized_args = cls._normalize_args(*args, **kwargs)
        # Create a key based on the arguments
        key = tuple(sorted(normalized_args.items()))
        cls._instance_tracker.setdefault(key,count(start=0))
        instance = super().__call__(*args, **kwargs)
        instance._channel_num = next(cls._instance_tracker[key])
        return instance
    
    def _normalize_args(cls, *args, **kwargs):
        # Get the parameter names from the __init__ method
        param_names = cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]
        # Create a dictionary with default values
        normalized_args = dict(zip(param_names, args))
        # Update with any kwargs
        normalized_args.update(kwargs)
        return normalized_args

@dataclass
class StepRegister(metaclass=InstanceCounterMeta):
    #set this as name of sweepval var
    register_name: str
    register_nesting: int
    #should be increased by metaclass every time the class is instantiated with the same arguments
    _channel_num: int = dataclasses.field(default_factory=lambda: None)
    
    @property
    def reg_var_name(self):
        return self.register_name+'_'+str(self.register_num)+'_'+str(self._channel_num)
    
    def __hash__(self):
        return hash((self.register_name,self.register_nesting,self._channel_num))
    

GeneralizedChannel = Union[DepDomain,ChannelID,StepRegister]

# is there any way to cast the numpy cumprod to int?
int_type = Union[np.int64,np.int32,int]

class ResolutionDependentValue(Generic[NumVal]):
    
    def __init__(self,
                 bases: Tuple[NumVal],
                 multiplicities: Tuple[int],
                 offset: NumVal):
    
        self.bases = tuple(bases)
        self.multiplicities = tuple(multiplicities)
        self.offset = offset
        self.__is_time_or_int = all(isinstance(b,(TimeType,int_type)) for b in bases) and isinstance(offset,(TimeType,int_type))
 
    #this is not to circumvent float errors in python, but rounding errors from awg-increment commands.
    #python float are thereby accurate enough if no awg with a 500 bit resolution is invented.
    def __call__(self, resolution: Optional[float]) -> Union[NumVal,TimeType]:
        #with resolution = None handle TimeType/int case?
        if resolution is None:
            assert self.__is_time_or_int
            return sum(b*m for b,m in zip(self.bases,self.multiplicities)) + self.offset
        #resolution as float value of granularity of base val.
        #to avoid conflicts between positive and negative vals from casting half to even,
        #use abs val
        return sum(np.sign(b) * round(abs(b) / resolution) * m * resolution for b,m in zip(self.bases,self.multiplicities))\
             + np.sign(self.offset) * round(abs(self.offset) / resolution) * resolution
             #cast the offset only once?
    
    def __bool__(self):
        return any(bool(b) for b in self.bases) or bool(self.offset)

    def __add__(self, other):
        # this should happen in the context of an offset being added to it, not the bases being modified.
        if isinstance(other, (float, int, TimeType)):
            return ResolutionDependentValue(self.bases,self.multiplicities,self.offset+other)
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self.__add__(-other)
    
    def __mul__(self, other):
        # this should happen when the amplitude is being scaled
        if isinstance(other, (float, int, TimeType)):
            return ResolutionDependentValue(tuple(b*other for b in self.bases),self.multiplicities,self.offset*other)
        return NotImplemented
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __truediv__(self,other):
        return self.__mul__(1/other)

    def __float__(self):
        return float(self(resolution=None))
    
    def __str__(self):
        return f"RDP of {sum(b*m for b,m in zip(self.bases,self.multiplicities)) + self.offset}"
    
    def __repr__(self):
        return "RDP("+",".join([f"{k}="+v.__str__() for k,v in vars(self).items()])+")"
    
    def __eq__(self,o):
        if not isinstance(o,ResolutionDependentValue):
            return False
        return self.__dict__ == o.__dict__
    
    def __hash__(self):
        return hash((self.bases,self.offset,self.multiplicities,self.__is_time_or_int))
    
    
@dataclass(frozen=True)
class DepKey:
    """The key that identifies how a certain set command depends on iteration indices. The factors are rounded with a
    given resolution to be independent on rounding errors.

    These objects allow backends which support it to track multiple amplitudes at once.
    """
    factors: Tuple[int, ...]
    domain: DepDomain
    # strategy: DepStrategy
    
    @classmethod
    def from_domain(cls, factors, resolution, domain):
        # # remove trailing zeros
        #why was this done in the first place? this seems to introduce more bugs than it solves
        # while factors and factors[-1] == 0:
        #     factors = factors[:-1]
        return cls(tuple(int(round(factor / resolution)) for factor in factors),
                   domain)
    
    @classmethod
    def from_voltages(cls, voltages: Sequence[float], resolution: float):
        return cls.from_domain(voltages, resolution, DepDomain.VOLTAGE)
    
    @classmethod
    def from_lin_times(cls, times: Sequence[float], resolution: float):
        return cls.from_domain(times, resolution, DepDomain.TIME_LIN)


@dataclass
class DummyMeasurementMemory:
    measurements: List[MeasurementWindow] = field(default_factory=lambda: [])
    
    def add_measurements(self, measurements: List[MeasurementWindow]):
        self.measurements.extend(measurements)


@dataclass
class LinSpaceNode:
    """AST node for a program that supports linear spacing of set points as well as nested sequencing and repetitions"""
    
    _cached_body_duration: TimeType|None = field(default=None, kw_only=True)
    _measurement_memory: DummyMeasurementMemory|None = field(default_factory=lambda:DummyMeasurementMemory(), kw_only=True)
    
    def dependencies(self) -> Mapping[GeneralizedChannel, set]:
        # doing this as a set _should_ get rid of non-active deps that are one level above?
        #!!! 
        raise NotImplementedError
        
    @property
    def body_duration(self) -> TimeType:
        raise NotImplementedError
        
    def _get_measurement_windows(self) -> Mapping[str, np.ndarray]:
        """Private implementation of get_measurement_windows with a slightly different data format for easier tiling.
        Returns:
             A dictionary (measurement_name -> array) with begin == array[:, 0] and length == array[:, 1]
        """
        raise NotImplementedError

    def get_measurement_windows(self, drop: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Iterates over all children and collect the begin and length arrays of each measurement window.
        Args:
            drop: NO EFFECT CURRENTLY
        Returns:
            A dictionary (measurement_name -> (begin, length)) with begin and length being :class:`numpy.ndarray`
        """
        return {mw_name: (begin_length_list[:, 0], begin_length_list[:, 1])
                for mw_name, begin_length_list in self._get_measurement_windows().items()}


@dataclass
class LinSpaceTopLevel(LinSpaceNode):
    
    body: Tuple[LinSpaceNode, ...]
    _play_marker_when_constant: bool
    _defined_channels: TypingSet[ChannelID]
    
    @property
    def play_marker_when_constant(self) -> bool:
        return self._play_marker_when_constant
    
    @property
    def body_duration(self) -> TimeType:
        if self._cached_body_duration is None:
            self._cached_body_duration = self.duration_base
        return self._cached_body_duration
    
    def _get_measurement_windows(self) -> Mapping[str, np.ndarray]:
        """Private implementation of get_measurement_windows with a slightly different data format for easier tiling.
        Returns:
              A dictionary (measurement_name -> array) with begin == array[:, 0] and length == array[:, 1]
        """
        return _get_measurement_windows_loop(self._measurement_memory.measurements,1,self.body)
    
    def get_defined_channels(self) -> TypingSet[ChannelID]:
        return  self._defined_channels
    

@dataclass
class LinSpaceNodeChannelSpecific(LinSpaceNode):
    
    channels: Tuple[GeneralizedChannel, ...]
    
    @property
    def play_channels(self) -> Tuple[ChannelID, ...]:
        return tuple(ch for ch in self.channels if isinstance(ch,ChannelID))
    
    def _get_measurement_windows(self,) -> Mapping[str, np.ndarray]:
        """Private implementation of get_measurement_windows with a slightly different data format for easier tiling.
        Returns:
              A dictionary (measurement_name -> array) with begin == array[:, 0] and length == array[:, 1]
        """
        return _get_measurement_windows_leaf(self._measurement_memory.measurements)
    
    
@dataclass
class LinSpaceHold(LinSpaceNodeChannelSpecific):
    """Hold voltages for a given time. The voltages and the time may depend on the iteration index."""

    bases: Dict[GeneralizedChannel, float]
    factors: Dict[GeneralizedChannel, Optional[Tuple[float, ...]]]

    duration_base: TimeType
    duration_factors: Optional[Tuple[TimeType, ...]]
    
    def dependencies(self) -> Mapping[DepDomain, Mapping[ChannelID, set]]:
        return {dom: {ch: {factors}}
                for dom, ch_to_factors in self._dep_by_domain().items()
                for ch, factors in ch_to_factors.items()
                if factors}
    
    def _dep_by_domain(self) -> Mapping[DepDomain, Mapping[GeneralizedChannel, set]]:
        return {DepDomain.VOLTAGE: self.factors,
                DepDomain.TIME_LIN: {DepDomain.TIME_LIN:self.duration_factors},
                }
    
    @property
    def body_duration(self) -> TimeType:
        if self.duration_factors:
            raise NotImplementedError
        if self._cached_body_duration is None:
            self._cached_body_duration = self.duration_base
        return self._cached_body_duration
    

@dataclass
class LinSpaceArbitraryWaveform(LinSpaceNodeChannelSpecific):
    """This is just a wrapper to pipe arbitrary waveforms through the system."""
    waveform: Waveform
    
    def dependencies(self):
        return {}
    
    @property
    def body_duration(self) -> TimeType:
        if self._cached_body_duration is None:
            self._cached_body_duration = self.waveform.duration
        return self._cached_body_duration
    

@dataclass
class LinSpaceArbitraryWaveformIndexed(LinSpaceNodeChannelSpecific):
    """This is just a wrapper to pipe arbitrary waveforms through the system."""
    waveform: Union[Waveform,WaveformCollection]
    
    scale_bases: Dict[ChannelID, float]
    scale_factors: Dict[ChannelID, Optional[Tuple[float, ...]]]
        
    offset_bases: Dict[ChannelID, float]
    offset_factors: Dict[ChannelID, Optional[Tuple[float, ...]]]
    
    index_factors: Optional[Dict[StepRegister,Tuple[int, ...]]] = dataclasses.field(default_factory=lambda: None)
    
    def __post_init__(self):
        #somewhat assert the integrity in this case.
        if isinstance(self.waveform,WaveformCollection):
            assert self.index_factors is not None
    
    def dependencies(self) -> Mapping[DepDomain, Mapping[GeneralizedChannel, set]]:
        return {dom: {ch: {factors}}
                for dom, ch_to_factors in self._dep_by_domain().items()
                for ch, factors in ch_to_factors.items()
                if factors}
    
    def _dep_by_domain(self) -> Mapping[DepDomain, Mapping[GeneralizedChannel, set]]:
        return {DepDomain.WF_SCALE: self.scale_factors,
                DepDomain.WF_OFFSET: self.offset_factors,
                DepDomain.STEP_INDEX: self.index_factors}
    
    @property
    def step_channels(self) -> Optional[Tuple[StepRegister]]:
        return tuple(self.index_factors.keys()) if self.index_factors else ()
    
    @property
    def body_duration(self) -> TimeType:
        if self._cached_body_duration is None:
            self._cached_body_duration = self.waveform.duration
        return self._cached_body_duration
    

@dataclass
class LinSpaceRepeat(LinSpaceNode):
    """Repeat the body count times."""
    body: Tuple[LinSpaceNode, ...]
    count: int

    def dependencies(self):
        dependencies = {}
        for node in self.body:
            for dom, ch_to_deps in node.dependencies().items():
                for ch, deps in ch_to_deps.items():
                    dependencies.setdefault(dom,{}).setdefault(ch, set()).update(deps)
        return dependencies
    
    @property
    def body_duration(self) -> TimeType:
        if self._cached_body_duration is None:
            self._cached_body_duration = self.count*sum(b.body_duration for b in self.body)
        return self._cached_body_duration
    
    def _get_measurement_windows(self,) -> Mapping[str, np.ndarray]:
        """Private implementation of get_measurement_windows with a slightly different data format for easier tiling.
        Returns:
              A dictionary (measurement_name -> array) with begin == array[:, 0] and length == array[:, 1]
        """
        return _get_measurement_windows_loop(self._measurement_memory.measurements,self.count,self.body)
    
    
@dataclass
class LinSpaceIter(LinSpaceNode):
    """Iteration in linear space are restricted to range 0 to length.

    Offsets and spacing are stored in the hold node."""
    body: Tuple[LinSpaceNode, ...]
    length: int
    
    to_be_stepped: bool
    
    def dependencies(self):
        dependencies = {}
        for node in self.body:
            for dom, ch_to_deps in node.dependencies().items():
                for ch, deps in ch_to_deps.items():
                    # remove the last element in index because this iteration sets it -> no external dependency
                    shortened = {dep[:-1] for dep in deps}
                    if shortened != {()}:
                        dependencies.setdefault(dom,{}).setdefault(ch, set()).update(shortened)
        return dependencies
    
    @property
    def body_duration(self) -> TimeType:
        if self._cached_body_duration is None:
            self._cached_body_duration = self.length*sum(b.body_duration for b in self.body)
        return self._cached_body_duration
    
    def _get_measurement_windows(self,) -> Mapping[str, np.ndarray]:
        """Private implementation of get_measurement_windows with a slightly different data format for easier tiling.
        Returns:
              A dictionary (measurement_name -> array) with begin == array[:, 0] and length == array[:, 1]
        """
        return _get_measurement_windows_loop(self._measurement_memory.measurements,self.length,self.body)

    

def _get_measurement_windows_leaf(measurements: List[MeasurementWindow]) -> Mapping[str, np.ndarray]:
    """Private implementation of get_measurement_windows with a slightly different data format for easier tiling.
    Returns:
          A dictionary (measurement_name -> array) with begin == array[:, 0] and length == array[:, 1]
    """
    temp_meas_windows = defaultdict(list)
    if measurements:
        for (mw_name, begin, length) in measurements:
            temp_meas_windows[mw_name].append((begin, length))

        for mw_name, begin_length_list in temp_meas_windows.items():
            temp_meas_windows[mw_name] = np.asarray(begin_length_list, dtype=float)

    return temp_meas_windows


def _get_measurement_windows_loop(measurements: List[MeasurementWindow], count: int,
                                  body: List[LinSpaceNode]) -> Mapping[str, np.ndarray]:
    """Private implementation of get_measurement_windows with a slightly different data format for easier tiling.
    Returns:
          A dictionary (measurement_name -> array) with begin == array[:, 0] and length == array[:, 1]
    """
    temp_meas_windows = defaultdict(list)
    if measurements:
        for (mw_name, begin, length) in measurements:
            temp_meas_windows[mw_name].append((begin, length))

        for mw_name, begin_length_list in temp_meas_windows.items():
            temp_meas_windows[mw_name] = [np.asarray(begin_length_list, dtype=float)]

    offset = TimeType(0)
    for child in body:
        for mw_name, begins_length_array in child._get_measurement_windows().items():
            begins_length_array[:, 0] += float(offset)
            temp_meas_windows[mw_name].append(begins_length_array)
        offset += child.body_duration

    body_duration = float(offset)

    # formatting like this for easier debugging
    result = {}

    # repeat and add repetition based offset
    for mw_name, begin_length_list in temp_meas_windows.items():
        result[mw_name] = _repeat_loop_measurements(begin_length_list, count, body_duration)

    return result


def _repeat_loop_measurements(begin_length_list: List[np.ndarray],
                              repetition_count: int,
                              body_duration: float
                              ) -> np.ndarray:
    temp_begin_length_array = np.concatenate(begin_length_list)

    begin_length_array = np.tile(temp_begin_length_array, (repetition_count, 1))

    shaped_begin_length_array = np.reshape(begin_length_array, (repetition_count, -1, 2))

    shaped_begin_length_array[:, :, 0] += (np.arange(repetition_count) * body_duration)[:, np.newaxis]

    return begin_length_array


class LinSpaceBuilder(ProgramBuilder):
    """This program builder supports efficient translation of pulse templates that use symbolic linearly
    spaced voltages and durations.

    The channel identifiers are reduced to their index in the given channel tuple.

    Arbitrary waveforms are not implemented yet
    """

    def __init__(self,
                 # channels: Tuple[ChannelID, ...]
                 to_stepping_repeat: TypingSet[Union[str,'ForLoopPT']] = set(),
                 # identifier, loop_index or ForLoopPT which is to be stepped.
                 play_marker_when_constant: bool = False,
                 ):
        super().__init__()
        # self._name_to_idx = {name: idx for idx, name in enumerate(channels)}
        # self._voltage_idx_to_name = channels

        self._stack = [[]]
        self._ranges = []
        self._to_stepping_repeat = to_stepping_repeat
        self._play_marker_when_constant = play_marker_when_constant
        self._pt_channels = None
        self._meas_queue = []
        
    def _root(self):
        return self._stack[0]

    def _get_rng(self, idx_name: str) -> range:
        return self._get_ranges()[idx_name]

    def inner_scope(self, scope: Scope, pt_obj: Optional['ForLoopPT']=None) -> Scope:
        """This function is necessary to inject program builder specific parameter implementations into the build
        process."""
        if self._ranges:
            name, rng = self._ranges[-1]
            if pt_obj and (pt_obj in self._to_stepping_repeat or pt_obj.identifier in self._to_stepping_repeat \
                or pt_obj.loop_index in self._to_stepping_repeat):
                    # the nesting level should be simply the amount of this type in the scope.
                    nest = len(tuple(v for v in scope.values() if isinstance(v,SimpleExpressionStepped)))
                    return scope.overwrite({name:SimpleExpressionStepped(
                        base=0,offsets={name: 1},step_nesting_level=nest+1,rng=rng)})
            else:
                if isinstance(scope.get(name,None),SimpleExpressionStepped):
                    return scope
                else:
                    return scope.overwrite({name: SimpleExpression(base=0, offsets={name: 1})})
        else:
            return scope

    def _get_ranges(self):
        return dict(self._ranges)

    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[ChannelID, HardwareVoltage]):
        # voltages = sorted((self._name_to_idx[ch_name], value) for ch_name, value in voltages.items())
        # voltages = [value for _, value in voltages]

        ranges = self._get_ranges()
        factors = {}
        bases = {}
        duration_base = duration
        duration_factors = None
        
        for ch_name,value in voltages.items():
            if isinstance(value, (float, int)):
                bases[ch_name] = float(value)
                factors[ch_name] = None
                continue
            offsets = value.offsets
            base = value.base
            incs = []
            for rng_name, rng in ranges.items():
                start = 0.
                step = 0.
                offset = offsets.get(rng_name, None)
                if offset:
                    start += rng.start * offset
                    step += rng.step * offset
                base += start
                incs.append(step)
            factors[ch_name] = tuple(incs)
            bases[ch_name] = base

        if isinstance(duration, SimpleExpression):
            # duration_factors = duration.offsets
            # duration_base = duration.base
            duration_offsets = duration.offsets
            duration_base = duration.base
            duration_factors = []
            for rng_name, rng in ranges.items():
                start = TimeType(0)
                step = TimeType(0)
                offset = duration_offsets.get(rng_name, None)
                if offset:
                    start += rng.start * offset
                    step += rng.step * offset
                duration_base += start
                duration_factors.append(step)
            

        set_cmd = LinSpaceHold(channels=tuple(voltages.keys()),
                               bases=bases,
                               factors=factors,
                               duration_base=duration_base,
                               duration_factors=tuple(duration_factors) if duration_factors else None,
                               )

        self._stack[-1].append(set_cmd)
        if self._meas_queue:
            self._stack[-1][-1]._measurement_memory.add_measurements(self._meas_queue.pop())
        

    def play_arbitrary_waveform(self, waveform: Union[Waveform,WaveformCollection],
                                stepped_var_list: Optional[List[Tuple[str,SimpleExpressionStepped]]] = None):
        
        # recognize voltage trafo sweep syntax from a transforming waveform.
        # other sweepable things may need different approaches.
        if not isinstance(waveform,(TransformingWaveform,WaveformCollection)):
            assert stepped_var_list is None
            ret = self._stack[-1].append(LinSpaceArbitraryWaveform(waveform=waveform,channels=waveform.defined_channels,))
            if self._meas_queue:
                self._stack[-1][-1]._measurement_memory.add_measurements(self._meas_queue.pop())
            return ret
        
            
        #should be sufficient to test the first wf, as all should have the same trafo
        waveform_propertyextractor = waveform
        while isinstance(waveform_propertyextractor,WaveformCollection):
            waveform_propertyextractor = waveform_propertyextractor.waveform_collection[0] 
        
        if isinstance(waveform_propertyextractor,TransformingWaveform):
            #test for transformations that contain SimpleExpression
            wf_transformation = waveform_propertyextractor.transformation
            
            # chainedTransformation should now have flat hierachy.        
            collected_trafos, dependent_trafo_vals_flag = collect_scaling_and_offset_per_channel(
                waveform_propertyextractor.defined_channels,wf_transformation)
        else:
            dependent_trafo_vals_flag = False
        
        #fast track
        if not dependent_trafo_vals_flag and not isinstance(waveform,WaveformCollection):
            ret = self._stack[-1].append(LinSpaceArbitraryWaveform(waveform=waveform,channels=waveform.defined_channels,))
            if self._meas_queue:
                self._stack[-1][-1]._measurement_memory.add_measurements(self._meas_queue.pop())
            return ret
    
        ranges = self._get_ranges()
        ranges_list = list(ranges)
        index_factors = {}
        
        if stepped_var_list:
            # the index ordering shall be with the last index changing fastest.
            # (assuming the WaveformColleciton will be flattened)
            # this means increments on last shall be 1, next lower 1*len(fastest),
            # next 1*len(fastest)*len(second_fastest),... -> product(higher_reg_range_lens)
            # total_reg_len = len(stepped_var_list)
            reg_lens = tuple(len(v.rng) for s,v in stepped_var_list)
            total_rng_len = np.cumprod(reg_lens)[-1]
            reg_incr_values = list(np.cumprod(reg_lens[::-1]))[::-1][1:] + [1,]
            
            assert isinstance(waveform,WaveformCollection)
            
            for reg_num,(var_name,value) in enumerate(stepped_var_list):
                # this should be given anyway:
                assert isinstance(value, SimpleExpressionStepped)
                
                """
                # by definition, every var_name should be relevant for the waveform/
                # has been included in the nested WaveformCollection.
                # so, each time this code is called, a new waveform node containing this is called,
                # and one can/must increase the offset by the 
                
                # assert value.base += total_rng_len
                """
                
                assert value.base == 0

                offsets = value.offsets
                #there can never be more than one key in this
                # (nowhere is an evaluation of arithmetics betwen steppings intended)
                assert len(offsets)==1
                assert all(v==1 for v in offsets.values())
                assert set(offsets.keys())=={var_name,}
                
                # this makes the search through ranges pointless; have tuple of zeros
                # except for one inc at the position of the stepvar in the ranges dict
                
                incs = [0 for v in ranges_list]
                incs[ranges_list.index(var_name)] = reg_incr_values[reg_num]
                
                #needs to be new "channel" each time? should be handled by metaclass
                reg_channel = StepRegister(var_name,reg_num)
                index_factors[reg_channel] = tuple(incs)
                # bases[reg_channel] = value.base

        scale_factors, offset_factors = {}, {}
        scale_bases, offset_bases = {}, {}
        
        if dependent_trafo_vals_flag:
            for ch_name,scale_offset_dict in collected_trafos.items():
                for bases,factors,key in zip((scale_bases, offset_bases),(scale_factors, offset_factors),('s','o')):
                    value = scale_offset_dict[key]
                    if isinstance(value, float):
                        bases[ch_name] = value
                        factors[ch_name] = None
                        continue
                    offsets = value.offsets
                    base = value.base
                    incs = []
                    for rng_name, rng in ranges.items():
                        start = 0.
                        step = 0.
                        offset = offsets.get(rng_name, None)
                        if offset:
                            start += rng.start * offset
                            step += rng.step * offset
                        base += start
                        incs.append(step)
                    factors[ch_name] = tuple(incs)
                    bases[ch_name] = base

        # assert ba        

        ret = self._stack[-1].append(LinSpaceArbitraryWaveformIndexed(
            waveform=waveform,
            channels=waveform_propertyextractor.defined_channels.union(set(index_factors.keys())),
            scale_bases=scale_bases,
            scale_factors=scale_factors,
            offset_bases=offset_bases,
            offset_factors=offset_factors,
            index_factors=index_factors,
            ))
        if self._meas_queue:
            self._stack[-1][-1]._measurement_memory.add_measurements(self._meas_queue.pop())
        return ret

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        # """Ignores measurements"""
        
        self._meas_queue.append(measurements)

    def with_repetition(self, repetition_count: RepetitionCount,
                        measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        if repetition_count == 0:
            return
        self._stack.append([])
        yield self
        blocks = self._stack.pop()
        if blocks:
            self._stack[-1].append(LinSpaceRepeat(body=tuple(blocks), count=repetition_count))

    @contextlib.contextmanager
    def with_sequence(self,
                      measurements: Optional[Sequence[MeasurementWindow]] = None) -> ContextManager['ProgramBuilder']:
        yield self

    def new_subprogram(self, global_transformation: 'Transformation' = None) -> ContextManager['ProgramBuilder']:
        
        inner_builder = LinSpaceBuilder(self._to_stepping_repeat,self._play_marker_when_constant)
        yield inner_builder
        inner_program = inner_builder.to_program()
        
        # if inner_program is not None:
    
        # # measurements = [(name, begin, length)
        #     #                 for name, (begins, lengths) in inner_program.get_measurement_windows().items()
        #     #                 for begin, length in zip(begins, lengths)]
        #     # self._top.add_measurements(measurements)
        # waveform = to_waveform(inner_program,self._idx_to_name)
        # if global_transformation is not None:
        #     waveform = TransformingWaveform.from_transformation(waveform, global_transformation)
        # self.play_arbitrary_waveform(waveform)
    
        raise NotImplementedError('Not implemented yet (postponed)')

    def with_iteration(self, index_name: str, rng: range,
                       pt_obj: 'ForLoopPT',
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        if len(rng) == 0:
            return
        self._stack.append([])
        self._ranges.append((index_name, rng))
        yield self
        cmds = self._stack.pop()
        self._ranges.pop()
        if cmds:
            stepped = False
            if pt_obj in self._to_stepping_repeat or pt_obj.identifier in self._to_stepping_repeat \
                or pt_obj.loop_index in self._to_stepping_repeat:
                stepped = True
            self._stack[-1].append(LinSpaceIter(body=tuple(cmds), length=len(rng), to_be_stepped=stepped))
            
            
    def evaluate_nested_stepping(self, scope: Scope, parameter_names: set[str]) -> bool:
        
        stepped_vals = {k:v for k,v in scope.items() if isinstance(v,SimpleExpressionStepped)}
        #when overlap, then the PT is part of the stepped progression
        if stepped_vals.keys() & parameter_names:
            return True   
        return False     
    
    def dispatch_to_stepped_wf_or_hold(self,
            build_func: Callable[[Mapping[str, Real],Dict[ChannelID, Optional[ChannelID]]],Optional[Waveform]],
            build_parameters: Scope,
            parameter_names: set[str],
            channel_mapping: Dict[ChannelID, Optional[ChannelID]],
            #measurements tbd
            global_transformation: Optional["Transformation"]) -> None:
        
        stepped_vals = {k:v for k,v in build_parameters.items()
                        if isinstance(v,SimpleExpressionStepped) and k in parameter_names}
        sorted_steps = list(sorted(stepped_vals.items(), key=lambda item: item[1].step_nesting_level))

        def build_nested_wf_colls(remaining_ranges: List[Tuple], fixed_elements: List[Tuple] = []):
            
            if len(remaining_ranges) == 0:
                inner_scope = build_parameters.overwrite(dict(fixed_elements))
                #by now, no SimpleExpressionStepped should remain here that is relevant for the current loop.
                assert not any(isinstance(v,SimpleExpressionStepped) for k,v in inner_scope.items() if k in parameter_names)
                waveform = build_func(inner_scope,channel_mapping=channel_mapping)
                if global_transformation:
                    waveform = TransformingWaveform.from_transformation(waveform, global_transformation)
                #this case should not happen, should have been caught beforehand:
                # or maybe not, if e.g. amp is zero for some reason
                # assert waveform.constant_value_dict() is None
                return waveform
            else:
                return WaveformCollection(
                    tuple(build_nested_wf_colls(remaining_ranges[1:],
                          fixed_elements+[(remaining_ranges[0][0],remaining_ranges[0][1].value({remaining_ranges[0][0]:it})),])
                          for it in remaining_ranges[0][1].rng))
        
        # not completely convinced this works as intended.
        # doesn't this - also in pulse_template program creation - lead to complications with ParallelConstantChannelTrafo?
        # dirty, quick workaround - if this doesnt work, assume it is also not constant:
        try:
            potential_waveform = build_func(build_parameters,channel_mapping=channel_mapping)
            if global_transformation:
                potential_waveform = TransformingWaveform.from_transformation(potential_waveform, global_transformation)
                constant_values = potential_waveform.constant_value_dict()
        except:
            constant_values = None
        
        if constant_values is None:
            wf_coll = build_nested_wf_colls(sorted_steps)
            self.play_arbitrary_waveform(wf_coll,sorted_steps)
        else:
            # in the other case, all dependencies _should_ be on amp and length, which is covered by hold appropriately
            # and doesn't need to be stepped?
            self.hold_voltage(potential_waveform.duration, constant_values)
    
    def to_program(self, defined_channels: TypingSet[ChannelID]) -> Optional[Sequence[LinSpaceNode]]:
        assert not self._meas_queue
        if self._root():
            return LinSpaceTopLevel(body=tuple(self._root()),
                                    _play_marker_when_constant=self._play_marker_when_constant,
                                    _defined_channels=defined_channels)


def collect_scaling_and_offset_per_channel(channels: Sequence[ChannelID],
                                           transformation: Transformation) \
    -> Tuple[Dict[ChannelID,Dict[str,Union[NumVal,SimpleExpression]]], bool]:
    
    ch_trafo_dict = {ch: {'s':1.,'o':0.} for ch in channels}
    
    # allowed_trafos = {IdentityTransformation,}
    if not isinstance(transformation,ChainedTransformation):
        transformations = (transformation,)
    else:
        transformations = transformation.transformations
    
    is_dependent_flag = []
    
    for trafo in transformations:
        #first elements of list are applied first in trafos.
        assert trafo.is_constant_invariant()
        if isinstance(trafo,ParallelChannelTransformation):
            for ch,val in trafo._channels.items():
                is_dependent_flag.append(trafo.contains_sweepval)
                # assert not ch in ch_trafo_dict.keys()
                # the waveform is sampled with these values taken into account, no change needed.
                # ch_trafo_dict[ch]['o'] = val
                # ch_trafo_dict.setdefault(ch,{'s':1.,'o':val})
        elif isinstance(trafo,ScalingTransformation):
            is_dependent_flag.append(trafo.contains_sweepval)
            for ch,val in trafo._factors.items():
                try:
                    ch_trafo_dict[ch]['s'] = reduce_non_swept(ch_trafo_dict[ch]['s']*val)
                    ch_trafo_dict[ch]['o'] = reduce_non_swept(ch_trafo_dict[ch]['o']*val)
                except TypeError as e:
                    print('Attempting scale sweep of other sweep val')
                    raise e
        elif isinstance(trafo,OffsetTransformation):
            is_dependent_flag.append(trafo.contains_sweepval)
            for ch,val in trafo._offsets.items():
                ch_trafo_dict[ch]['o'] += val
        elif isinstance(trafo,IdentityTransformation):
            continue
        elif isinstance(trafo,ChainedTransformation):
            raise RuntimeError()
        else:
            raise NotImplementedError()
    
    return ch_trafo_dict, any(is_dependent_flag)
    

def reduce_non_swept(val: Union[SimpleExpression,NumVal]) -> Union[SimpleExpression,NumVal]:
    if isinstance(val,SimpleExpression) and all(v==0 for v in val.offsets.values()):
        return val.base
    return val


@dataclass
class LoopLabel:
    idx: int
    count: int


@dataclass
class Increment:
    channel: Optional[GeneralizedChannel]
    value: Union[ResolutionDependentValue,Tuple[ResolutionDependentValue]]
    key: DepKey
    
    def __hash__(self):
        return hash((self.channel,self.value,self.key))
    
    def __str__(self):
        return "Increment("+",".join([f"{k}="+v.__str__() for k,v in vars(self).items()])+")"

@dataclass
class Set:
    channel: Optional[GeneralizedChannel]
    value: Union[ResolutionDependentValue,Tuple[ResolutionDependentValue]]
    key: DepKey = dataclasses.field(default_factory=lambda: DepKey((),DepDomain.NODEP))

    def __hash__(self):
        return hash((self.channel,self.value,self.key))
    
    def __str__(self):
        return "Set("+",".join([f"{k}="+v.__str__() for k,v in vars(self).items()])+")"
    
@dataclass
class Wait:
    duration: Optional[TimeType]
    key_by_domain: Dict[DepDomain,DepKey] = dataclasses.field(default_factory=lambda: {})

    def __hash__(self):
        return hash((self.duration,frozenset(self.key_by_domain.items())))

@dataclass
class LoopJmp:
    idx: int


@dataclass
class Play:
    waveform: Union[Waveform,WaveformCollection]
    play_channels: Tuple[ChannelID]
    step_channels: Tuple[StepRegister] = ()
    #actually did the name
    keys_by_domain_by_ch: Dict[ChannelID,Dict[DepDomain,DepKey]] = None
    
    def __post_init__(self):
        if self.keys_by_domain_by_ch is None:
            self.keys_by_domain_by_ch = {ch: {} for ch in self.play_channels+self.step_channels}
    
    def __hash__(self):
        return hash((self.waveform,self.play_channels,self.step_channels,
                     frozenset((k,frozenset(d.items())) for k,d in self.keys_by_domain_by_ch.items())))


Command = Union[Increment, Set, LoopLabel, LoopJmp, Wait, Play]


@dataclass(frozen=True)
class DepState:
    base: float
    iterations: Tuple[int, ...]

    def required_increment_from(self, previous: 'DepState',
                                factors: Sequence[float]) -> ResolutionDependentValue:
        assert len(self.iterations) == len(previous.iterations) #or (all(self.iterations)==0 and all(previous.iterations)==0)
        assert len(self.iterations) == len(factors)

        # increment = self.base - previous.base
        res_bases, res_mults, offset = [], [], self.base - previous.base
        for old, new, factor in zip(previous.iterations, self.iterations, factors):
            # By convention there are only two possible values for each integer here: 0 or the last index
            # The three possible increments are none, regular and jump to next line

            if old == new:
                # we are still in the same iteration of this sweep
                pass

            elif old < new:
                assert old == 0
                # regular iteration, although the new value will probably be > 1, the resulting increment will be
                # applied multiple times so only one factor is needed.
                # increment += factor
                res_bases.append(factor)
                res_mults.append(1)
                
            else:
                assert new == 0
                # we need to jump back. The old value gives us the number of increments to reverse
                # increment -= factor * old
                res_bases.append(-factor)
                res_mults.append(old)
                
        return ResolutionDependentValue(res_bases,res_mults,offset)


@dataclass
class _TranslationState:
    """This is the state of a translation of a LinSpace program to a command sequence."""

    label_num: int = dataclasses.field(default=0)
    commands: List[Command] = dataclasses.field(default_factory=list)
    iterations: List[int] = dataclasses.field(default_factory=list)
    active_dep: Dict[GeneralizedChannel, Dict[DepDomain, DepKey]] = dataclasses.field(default_factory=dict)
    dep_states: Dict[GeneralizedChannel, Dict[DepKey, DepState]] = dataclasses.field(default_factory=dict)
    plain_value: Dict[GeneralizedChannel, Dict[DepDomain,float]] = dataclasses.field(default_factory=dict)
    resolution: float = dataclasses.field(default_factory=lambda: DEFAULT_INCREMENT_RESOLUTION)
    resolution_time: float = dataclasses.field(default_factory=lambda: DEFAULT_TIME_RESOLUTION)

    def new_loop(self, count: int):
        label = LoopLabel(self.label_num, count)
        jmp = LoopJmp(self.label_num)
        self.label_num += 1
        return label, jmp

    def get_dependency_state(self, dependencies: Mapping[DepDomain, Mapping[GeneralizedChannel, set]]):
        dom_to_ch_to_depstates = {}
        
        for dom, ch_to_deps in dependencies.items():
            dom_to_ch_to_depstates.setdefault(dom,{})
            for ch, deps in ch_to_deps.items():
                dom_to_ch_to_depstates[dom].setdefault(ch,set())
                for dep in deps:
                    dom_to_ch_to_depstates[dom][ch].add(self.dep_states.get(ch, {}).get(
                        DepKey.from_domain(dep, self.resolution, dom),None))
        
        return dom_to_ch_to_depstates
        # return {
        #     dom: self.dep_states.get(ch, {}).get(DepKey.from_domain(dep, self.resolution, dom),
        #         None)
        #     for dom, ch_to_deps in dependencies.items()
        #     for ch, deps in ch_to_deps.items()
        #     for dep in deps
        # }
    
    def compare_ignoring_post_trailing_zeros(self,
                                             pre_state: Mapping[DepDomain, Mapping[GeneralizedChannel, set]],
                                             post_state: Mapping[DepDomain, Mapping[GeneralizedChannel, set]]) -> bool:
        
        def reduced_or_none(dep_state: DepState) -> Union[DepState,None]:
            new_iterations = tuple(dropwhile(lambda x: x == 0, reversed(dep_state.iterations)))[::-1]
            return DepState(dep_state.base, new_iterations) if len(new_iterations)>0 else None
        
        has_changed = False
        dom_keys = set(pre_state.keys()).union(post_state.keys())
        for dom_key in dom_keys:
            pre_state_dom, post_state_dom = pre_state.get(dom_key,{}), post_state.get(dom_key,{})
            ch_keys = set(pre_state_dom.keys()).union(post_state_dom.keys())
            for ch_key in ch_keys:
                pre_state_dom_ch, post_state_dom_ch = pre_state_dom.get(ch_key,set()), post_state_dom.get(ch_key,set())
                # reduce the depStates to the ones which do not just contain zeros
                reduced_pre_set = set(reduced_or_none(dep_state) for dep_state in pre_state_dom_ch
                                      if dep_state is not None) - {None}
                reduced_post_set = set(reduced_or_none(dep_state) for dep_state in post_state_dom_ch
                                       if dep_state is not None) - {None}
                
                if not reduced_post_set <= reduced_pre_set:
                    has_changed == True
                    
        return has_changed
    
    def set_voltage(self, channel: ChannelID, value: float):
        self.set_non_indexed_value(channel, value, domain=DepDomain.VOLTAGE, always_emit_set=True)
        
    def set_wf_scale(self, channel: ChannelID, value: float):
        self.set_non_indexed_value(channel, value, domain=DepDomain.WF_SCALE)
        
    def set_wf_offset(self, channel: ChannelID, value: float):
        self.set_non_indexed_value(channel, value, domain=DepDomain.WF_OFFSET)
            
    def set_non_indexed_value(self, channel: GeneralizedChannel, value: float,
                              domain: DepDomain, always_emit_set: bool=False):
        key = DepKey((),domain)
        # I do not completely get why it would have to be set again if not in active dep.
        # if not key != self.active_dep.get(channel, None)  or
        if self.plain_value.get(channel, {}).get(domain, None) != value or always_emit_set:
            self.commands.append(Set(channel, ResolutionDependentValue((),(),offset=value), key))
            # there has to be no active dep when the value is not indexed?
            # self.active_dep.setdefault(channel,{})[DepDomain.NODEP] = key
            self.plain_value.setdefault(channel,{})
            self.plain_value[channel][domain] = value
    
    # def _add_repetition_node(self, node: LinSpaceRepeat):
    #     pre_dep_state = self.get_dependency_state(node.dependencies())
    #     label, jmp = self.new_loop(node.count)
    #     initial_position = len(self.commands)
    #     self.commands.append(label)
    #     self.add_node(node.body)
    #     post_dep_state = self.get_dependency_state(node.dependencies())
    #     # the last index in the iterations may not be initialized in pre_dep_state if the outer loop only sets an index
    #     # after this loop is in the sequence of the current level,
    #     # meaning that an trailing 0 at the end of iterations of each depState in the post_dep_state
    #     # should be ignored when comparing.
    #     # zeros also should only mean a "Set" command, which is not harmful if executed multiple times.
    #     # if pre_dep_state != post_dep_state:
    #     if self.compare_ignoring_post_trailing_zeros(pre_dep_state,post_dep_state):
    #         # hackedy
    #         self.commands.pop(initial_position)
    #         self.commands.append(label)
    #         label.count -= 1
    #         self.add_node(node.body)
    #     self.commands.append(jmp)
    
    
    def _add_repetition_node(self, node: LinSpaceRepeat):
        pre_dep_state = self.get_dependency_state(node.dependencies())
        label, jmp = self.new_loop(node.count)
        initial_position = len(self.commands)
        self.commands.append(label)
        self.add_node(node.body)
        post_dep_state = self.get_dependency_state(node.dependencies())
        # the last index in the iterations may not be initialized in pre_dep_state if the outer loop only sets an index
        # after this loop is in the sequence of the current level,
        # meaning that an trailing 0 at the end of iterations of each depState in the post_dep_state
        # should be ignored when comparing.
        # zeros also should only mean a "Set" command, which is not harmful if executed multiple times.
        # if pre_dep_state != post_dep_state:
        #EDIT: even this is not enough it seems; if a dependency from an outer
        # loop is present that the repetition does not know about, this is still necessary.
        # why not always in the first place?
        # if self.compare_ignoring_post_trailing_zeros(pre_dep_state,post_dep_state):
        if True:
            # hackedy
            self.commands.pop(initial_position)
            self.commands.append(label)
            label.count -= 1
            self.add_node(node.body)
        self.commands.append(jmp)
    
    def _add_iteration_node(self, node: LinSpaceIter):
        
        self.iterations.append(0)
        self.add_node(node.body)

        if node.length > 1:
            self.iterations[-1] = node.length - 1
            label, jmp = self.new_loop(node.length - 1)
            self.commands.append(label)
            self.add_node(node.body)
            self.commands.append(jmp)
        self.iterations.pop()
        
    def _set_indexed_voltage(self, channel: ChannelID, base: float, factors: Sequence[float]):
        key = DepKey.from_voltages(voltages=factors, resolution=self.resolution)
        self.set_indexed_value(key, channel, base, factors, domain=DepDomain.VOLTAGE, always_emit_incr=True)
    
    def _set_indexed_lin_time(self, base: TimeType, factors: Sequence[TimeType]):
        key = DepKey.from_lin_times(times=factors, resolution=self.resolution)
        self.set_indexed_value(key, DepDomain.TIME_LIN, base, factors, domain=DepDomain.TIME_LIN)

    def set_indexed_value(self, dep_key: DepKey, channel: GeneralizedChannel,
                          base: Union[float,TimeType], factors: Sequence[Union[float,TimeType]],
                          domain: DepDomain, always_emit_incr: bool = False):
        new_dep_state = DepState(
            base,
            iterations=tuple(self.iterations)
        )

        current_dep_state = self.dep_states.setdefault(channel, {}).get(dep_key, None)
        if current_dep_state is None:
            assert all(it == 0 for it in self.iterations)
            self.commands.append(Set(channel, ResolutionDependentValue((),(),offset=base), dep_key))
            self.active_dep.setdefault(channel,{})[dep_key.domain] = dep_key

        else:
            inc = new_dep_state.required_increment_from(previous=current_dep_state, factors=factors)

            # we insert all inc here (also inc == 0) because it signals to activate this amplitude register
            # -> since this is not necessary for other domains, make it stricter and bypass if necessary for voltage.
            if ((inc or self.active_dep.get(channel, {}).get(dep_key.domain) != dep_key)
                and new_dep_state != current_dep_state)\
                or always_emit_incr:
                # if always_emit_incr and new_dep_state == current_dep_state, inc should be zero.
                #this is not always the case, e.g. if multiple sequenced pts with different
                #dependencies on param exist and some with same are chained? very complicated case
                #and probably not handled correctly
                # if always_emit_incr and new_dep_state == current_dep_state:
                #     assert inc==0.
                self.commands.append(Increment(channel, inc, dep_key))
            self.active_dep.setdefault(channel,{})[dep_key.domain] = dep_key
        self.dep_states[channel][dep_key] = new_dep_state
        
    def _add_hold_node(self, node: LinSpaceHold):

        for ch in node.play_channels:
            if node.factors[ch] is None:
                self.set_voltage(ch, node.bases[ch])
                continue
            else:
                self._set_indexed_voltage(ch, node.bases[ch], node.factors[ch])
                
        if node.duration_factors:
            self._set_indexed_lin_time(node.duration_base,node.duration_factors)
            # raise NotImplementedError("TODO")
            self.commands.append(Wait(None, {DepDomain.TIME_LIN: self.active_dep[DepDomain.TIME_LIN][DepDomain.TIME_LIN]}))
        else:
            self.commands.append(Wait(node.duration_base))
            
    def _add_indexed_play_node(self, node: LinSpaceArbitraryWaveformIndexed):
        
        #assume this as criterion:
        if len(node.scale_bases) and len(node.offset_bases):
            for ch in node.play_channels:
                for base,factors,domain in zip((node.scale_bases[ch], node.offset_bases[ch]),
                                               (node.scale_factors[ch], node.offset_factors[ch]),
                                               (DepDomain.WF_SCALE,DepDomain.WF_OFFSET)): 
                    if factors is None:
                        continue
                        # assume here that the waveform will have the correct settings the TransformingWaveform,
                        # where no SimpleExpression is replaced now.
                        # will yield the correct trafo already without having to make adjustments
                        # self.set_non_indexed_value(ch, base, domain)
                    else:
                        key = DepKey.from_domain(factors, resolution=self.resolution, domain=domain)
                        self.set_indexed_value(key, ch, base, factors, key.domain)
            
        for st_ch, st_factors in node.index_factors.items():
            #this should not happen:
            assert st_factors is not None
            key = DepKey.from_domain(st_factors, resolution=self.resolution, domain=DepDomain.STEP_INDEX)
            self.set_indexed_value(key, st_ch, 0, st_factors, key.domain)
            
            
        self.commands.append(Play(node.waveform, node.channels, step_channels=node.step_channels,
                                  keys_by_domain_by_ch={c: self.active_dep.get(c,{}) for c in node.channels}))
        
            
    def add_node(self, node: Union[LinSpaceNode, Sequence[LinSpaceNode]]):
        """Translate a (sequence of) linspace node(s) to commands and add it to the internal command list."""

        if isinstance(node, Sequence):
            for lin_node in node:
                self.add_node(lin_node)

        elif isinstance(node, LinSpaceRepeat):
            self._add_repetition_node(node)

        elif isinstance(node, LinSpaceIter):
            self._add_iteration_node(node)

        elif isinstance(node, LinSpaceHold):
            self._add_hold_node(node)
        
        elif isinstance(node, LinSpaceArbitraryWaveformIndexed):
            self._add_indexed_play_node(node)
        
        elif isinstance(node, LinSpaceArbitraryWaveform):
            self.commands.append(Play(node.waveform, node.play_channels))

        else:
            raise TypeError("The node type is not handled", type(node), node)


def to_increment_commands(linspace_nodes: LinSpaceTopLevel,
                          # resolution: float = DEFAULT_INCREMENT_RESOLUTION
                          ) -> List[Command]:
    """translate the given linspace node tree to a minimal sequence of set and increment commands as well as loops."""
    # if resolution: raise NotImplementedError('wrongly assumed resolution. need to fix')
    state = _TranslationState()
    state.add_node(linspace_nodes.body)
    return state.commands

