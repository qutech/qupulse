import abc
import contextlib
import dataclasses
import numpy as np
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, ContextManager, Iterable, Tuple, Union, Dict, List, Iterator, Generic
from enum import Enum

from qupulse import ChannelID, MeasurementWindow
from qupulse.parameter_scope import Scope, MappedScope, FrozenDict
from qupulse.program import ProgramBuilder, HardwareTime, HardwareVoltage, Waveform, RepetitionCount, TimeType
from qupulse.expressions.simple import SimpleExpression, NumVal
from qupulse.program.waveforms import MultiChannelWaveform, TransformingWaveform

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
    NODEP = None

GeneralizedChannel = Union[DepDomain,ChannelID]


class ResolutionDependentValue(Generic[NumVal]):
    
    def __init__(self,
                 bases: Sequence[NumVal],
                 multiplicities: Sequence[int],
                 offset: NumVal):
    
        self.bases = bases
        self.multiplicities = multiplicities
        self.offset = offset
        self.__is_time_or_int = all(isinstance(b,(TimeType,int)) for b in bases) and isinstance(offset,(TimeType,int))
 
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
        # remove trailing zeros
        while factors and factors[-1] == 0:
            factors = factors[:-1]
        return cls(tuple(int(round(factor / resolution)) for factor in factors),
                   domain)
    
    @classmethod
    def from_voltages(cls, voltages: Sequence[float], resolution: float):
        return cls.from_domain(voltages, resolution, DepDomain.VOLTAGE)
    
    @classmethod
    def from_lin_times(cls, times: Sequence[float], resolution: float):
        return cls.from_domain(times, resolution, DepDomain.TIME_LIN)


@dataclass
class LinSpaceNode:
    """AST node for a program that supports linear spacing of set points as well as nested sequencing and repetitions"""
        
    def dependencies(self) -> Mapping[GeneralizedChannel, set]:
        raise NotImplementedError

@dataclass
class LinSpaceNodeChannelSpecific(LinSpaceNode):
    
    channels: Tuple[GeneralizedChannel, ...]
    
    @property
    def play_channels(self) -> Tuple[ChannelID, ...]:
        return tuple(ch for ch in self.channels if isinstance(ch,ChannelID))
    

@dataclass
class LinSpaceHold(LinSpaceNodeChannelSpecific):
    """Hold voltages for a given time. The voltages and the time may depend on the iteration index."""

    bases: Dict[GeneralizedChannel, float]
    factors: Dict[GeneralizedChannel, Optional[Tuple[float, ...]]]

    duration_base: TimeType
    duration_factors: Optional[Tuple[TimeType, ...]]

    def dependencies(self) -> Mapping[GeneralizedChannel, set]:
        return {idx: {factors}
                for idx, factors in self.factors.items()
                if factors}


@dataclass
class LinSpaceArbitraryWaveform(LinSpaceNodeChannelSpecific):
    """This is just a wrapper to pipe arbitrary waveforms through the system."""
    waveform: Waveform


@dataclass
class LinSpaceArbitraryWaveformIndexed(LinSpaceNodeChannelSpecific):
    """This is just a wrapper to pipe arbitrary waveforms through the system."""
    waveform: Waveform
    
    scale_bases: Dict[ChannelID, float]
    scale_factors: Dict[ChannelID, Optional[Tuple[float, ...]]]
        
    offset_bases: Dict[ChannelID, float]
    offset_factors: Dict[ChannelID, Optional[Tuple[float, ...]]]


@dataclass
class LinSpaceRepeat(LinSpaceNode):
    """Repeat the body count times."""
    body: Tuple[LinSpaceNode, ...]
    count: int

    def dependencies(self):
        dependencies = {}
        for node in self.body:
            for idx, deps in node.dependencies().items():
                dependencies.setdefault(idx, set()).update(deps)
        return dependencies


@dataclass
class LinSpaceIter(LinSpaceNode):
    """Iteration in linear space are restricted to range 0 to length.

    Offsets and spacing are stored in the hold node."""
    body: Tuple[LinSpaceNode, ...]
    length: int

    def dependencies(self):
        dependencies = {}
        for node in self.body:
            for idx, deps in node.dependencies().items():
                # remove the last elemt in index because this iteration sets it -> no external dependency
                shortened = {dep[:-1] for dep in deps}
                if shortened != {()}:
                    dependencies.setdefault(idx, set()).update(shortened)
        return dependencies


class LinSpaceBuilder(ProgramBuilder):
    """This program builder supports efficient translation of pulse templates that use symbolic linearly
    spaced voltages and durations.

    The channel identifiers are reduced to their index in the given channel tuple.

    Arbitrary waveforms are not implemented yet
    """

    def __init__(self,
                 # channels: Tuple[ChannelID, ...]
                 ):
        super().__init__()
        # self._name_to_idx = {name: idx for idx, name in enumerate(channels)}
        # self._voltage_idx_to_name = channels

        self._stack = [[]]
        self._ranges = []

    def _root(self):
        return self._stack[0]

    def _get_rng(self, idx_name: str) -> range:
        return self._get_ranges()[idx_name]

    def inner_scope(self, scope: Scope) -> Scope:
        """This function is necessary to inject program builder specific parameter implementations into the build
        process."""
        if self._ranges:
            name, _ = self._ranges[-1]
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
                               duration_factors=duration_factors)

        self._stack[-1].append(set_cmd)

    def play_arbitrary_waveform(self, waveform: Waveform):
        
        #recognize voltage trafo sweep syntax from a transforming waveform. other sweepable things may need different approaches.
        if not isinstance(waveform,TransformingWaveform):
            return self._stack[-1].append(LinSpaceArbitraryWaveform(waveform=waveform,channels=waveform.defined_channels,))
        
        #test for transformations that contain SimpleExpression
        wf_transformation = waveform.transformation
        
        # chainedTransformation should now have flat hierachy.        
        collected_trafos, dependent_vals_flag = collect_scaling_and_offset_per_channel(waveform.defined_channels,wf_transformation)
        
        #fast track
        if not dependent_vals_flag:
            return self._stack[-1].append(LinSpaceArbitraryWaveform(waveform=waveform,channels=waveform.defined_channels,))
        
        ranges = self._get_ranges()
        scale_factors, offset_factors = {}, {}
        scale_bases, offset_bases = {}, {}
        
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

        
        return self._stack[-1].append(LinSpaceArbitraryWaveformIndexed(waveform=waveform,
                                                                       channels=waveform.defined_channels,
                                                                       scale_bases=scale_bases,
                                                                       scale_factors=scale_factors,
                                                                       offset_bases=offset_bases,
                                                                       offset_factors=offset_factors,
                                                                       ))

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        """Ignores measurements"""
        pass

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
        raise NotImplementedError('Not implemented yet (postponed)')

    def with_iteration(self, index_name: str, rng: range,
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        if len(rng) == 0:
            return
        self._stack.append([])
        self._ranges.append((index_name, rng))
        yield self
        cmds = self._stack.pop()
        self._ranges.pop()
        if cmds:
            self._stack[-1].append(LinSpaceIter(body=tuple(cmds), length=len(rng)))

    def to_program(self) -> Optional[Sequence[LinSpaceNode]]:
        if self._root():
            return self._root()


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
    dependency_key: DepKey


@dataclass
class Set:
    channel: Optional[GeneralizedChannel]
    value: Union[ResolutionDependentValue,Tuple[ResolutionDependentValue]]
    key: DepKey = dataclasses.field(default_factory=lambda: DepKey((),DepDomain.NODEP))


@dataclass
class Wait:
    duration: Optional[TimeType]
    key: DepKey = dataclasses.field(default_factory=lambda: DepKey((),DepDomain.NODEP))


@dataclass
class LoopJmp:
    idx: int


@dataclass
class Play:
    waveform: Waveform
    channels: Tuple[ChannelID]
    keys: Sequence[DepKey] = None
    def __post_init__(self):
        if self.keys is None:
            self.keys = tuple(DepKey((),DepDomain.NODEP) for i in range(len(self.channels)))
    

Command = Union[Increment, Set, LoopLabel, LoopJmp, Wait, Play]


@dataclass(frozen=True)
class DepState:
    base: float
    iterations: Tuple[int, ...]

    def required_increment_from(self, previous: 'DepState',
                                factors: Sequence[float]) -> ResolutionDependentValue:
        assert len(self.iterations) == len(previous.iterations)
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
    active_dep: Dict[GeneralizedChannel, DepKey] = dataclasses.field(default_factory=dict)
    dep_states: Dict[GeneralizedChannel, Dict[DepKey, DepState]] = dataclasses.field(default_factory=dict)
    plain_value: Dict[GeneralizedChannel, Dict[DepDomain,float]] = dataclasses.field(default_factory=dict)
    resolution: float = dataclasses.field(default_factory=lambda: DEFAULT_INCREMENT_RESOLUTION)
    resolution_time: float = dataclasses.field(default_factory=lambda: DEFAULT_TIME_RESOLUTION)

    def new_loop(self, count: int):
        label = LoopLabel(self.label_num, count)
        jmp = LoopJmp(self.label_num)
        self.label_num += 1
        return label, jmp

    def get_dependency_state(self, dependencies: Mapping[GeneralizedChannel, set]):
        return {
            self.dep_states.get(ch, {}).get(DepKey.from_domain(dep, self.resolution), None)
            for ch, deps in dependencies.items()
            for dep in deps
        }

    def set_voltage(self, channel: ChannelID, value: float):
        self.set_non_indexed_value(channel, value, domain=DepDomain.VOLTAGE)
        
    def set_wf_scale(self, channel: ChannelID, value: float):
        self.set_non_indexed_value(channel, value, domain=DepDomain.WF_SCALE)
        
    def set_wf_offset(self, channel: ChannelID, value: float):
        self.set_non_indexed_value(channel, value, domain=DepDomain.WF_OFFSET)
            
    def set_non_indexed_value(self, channel: GeneralizedChannel, value: float, domain: DepDomain):
        key = DepKey((),domain)
        # I do not completely get why it would have to be set again if not in active dep.
        # if not key != self.active_dep.get(channel, None)  or
        if self.plain_value.get(channel, {}).get(domain, None) != value:
            self.commands.append(Set(channel, ResolutionDependentValue((),(),offset=value), key))
            self.active_dep[channel] = key
            self.plain_value.setdefault(channel,{})
            self.plain_value[channel][domain] = value    
    
    def _add_repetition_node(self, node: LinSpaceRepeat):
        pre_dep_state = self.get_dependency_state(node.dependencies())
        label, jmp = self.new_loop(node.count)
        initial_position = len(self.commands)
        self.commands.append(label)
        self.add_node(node.body)
        post_dep_state = self.get_dependency_state(node.dependencies())
        if pre_dep_state != post_dep_state:
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
        self.set_indexed_value(key, channel, base, factors, domain=DepDomain.VOLTAGE)
    
    def _set_indexed_lin_time(self, base: TimeType, factors: Sequence[TimeType]):
        key = DepKey.from_lin_times(times=factors, resolution=self.resolution)
        self.set_indexed_value(key, DepDomain.TIME_LIN, base, factors, domain=DepDomain.TIME_LIN)

    def set_indexed_value(self, dep_key: DepKey, channel: GeneralizedChannel,
                          base: Union[float,TimeType], factors: Sequence[Union[float,TimeType]],
                          domain: DepDomain):
        new_dep_state = DepState(
            base,
            iterations=tuple(self.iterations)
        )

        current_dep_state = self.dep_states.setdefault(channel, {}).get(dep_key, None)
        if current_dep_state is None:
            assert all(it == 0 for it in self.iterations)
            self.commands.append(Set(channel, ResolutionDependentValue((),(),offset=base), dep_key))
            self.active_dep[channel] = dep_key

        else:
            inc = new_dep_state.required_increment_from(previous=current_dep_state, factors=factors)

            # we insert all inc here (also inc == 0) because it signals to activate this amplitude register
            if inc or self.active_dep.get(channel, None) != dep_key:
                self.commands.append(Increment(channel, inc, dep_key))
            self.active_dep[channel] = dep_key
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
            self.commands.append(Wait(None, self.active_dep[DepDomain.TIME_LIN]))
        else:
            self.commands.append(Wait(node.duration_base))
            
    def _add_indexed_play_node(self, node: LinSpaceArbitraryWaveformIndexed):
        
        for ch in node.channels:
            for base,factors,domain in zip((node.scale_bases[ch], node.offset_bases[ch]),
                                           (node.scale_factors[ch], node.offset_factors[ch]),
                                           (DepDomain.WF_SCALE,DepDomain.WF_OFFSET)): 
                if factors is None:
                    self.set_non_indexed_value(ch, base, domain)
                else:
                    key = DepKey.from_domain(factors, resolution=self.resolution, domain=domain)
                    self.set_indexed_value(key, ch, base, factors, key.domain)
                
        self.commands.append(Play(node.waveform, node.channels, keys=tuple(self.active_dep[ch] for ch in node.channels)))
        
            
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
            self.commands.append(Play(node.waveform, node.channels))

        else:
            raise TypeError("The node type is not handled", type(node), node)


def to_increment_commands(linspace_nodes: Sequence[LinSpaceNode],
                          # resolution: float = DEFAULT_INCREMENT_RESOLUTION
                          ) -> List[Command]:
    """translate the given linspace node tree to a minimal sequence of set and increment commands as well as loops."""
    # if resolution: raise NotImplementedError('wrongly assumed resolution. need to fix')
    state = _TranslationState()
    state.add_node(linspace_nodes)
    return state.commands

