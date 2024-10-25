import abc
import contextlib
import dataclasses
import numpy as np
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, ContextManager, Iterable, Tuple, Union, Dict, List, Iterator, Generic,\
    Set, Callable, Self, Any
from enum import Enum
from itertools import dropwhile, count, zip_longest
from numbers import Real, Number
from collections import defaultdict


from qupulse import ChannelID, MeasurementWindow
from qupulse.parameter_scope import Scope, MappedScope, FrozenDict
# from qupulse.pulses.pulse_template import PulseTemplate
# from qupulse.pulses import ForLoopPT
from qupulse.program import ProgramBuilder, HardwareTime, HardwareVoltage, Waveform, RepetitionCount, TimeType, Program
from qupulse.expressions.simple import SimpleExpression, NumVal, SimpleExpressionStepped
from qupulse.program.waveforms import MultiChannelWaveform, TransformingWaveform, WaveformCollection, SubsetWaveform
from qupulse.utils import cached_property

from qupulse.program.transformation import ChainedTransformation, ScalingTransformation, OffsetTransformation,\
    IdentityTransformation, ParallelChannelTransformation, Transformation

from copy import deepcopy

NestedPBMapping = Dict[str, Union[ProgramBuilder, 'NestedPBMapping']]

class MultiProgramBuilder(ProgramBuilder):
    
    def __init__(self,
                 sub_program_builders: Dict[str,ProgramBuilder|Self],
                 channel_subsets: Dict[str,Set[ChannelID]|Self],
                 _scheduler_options: Dict[str,Any] = {},
                 ):
        
        super().__init__()
        self._program_builder_map = sub_program_builders
        
        self._stack = [('top',sub_program_builders)]
        
        self._channel_subsets = channel_subsets
        
        self._scheduler_options = _scheduler_options
        
        self._donotcreatenext = []
        
    # def get_program_builder(self, key) -> NestedPBMapping:
    #     return self._program_builder_map.setdefault(key,deepcopy(self._program_builder_map[-1]))
    
    @classmethod
    def from_mapping(cls, default_program_builder: ProgramBuilder,
                     channel_subsets: Dict[str,Set[ChannelID]|Self],
                     _scheduler_options: Dict[str,Any] = {},
                     ):
        
        structure = deepcopy(channel_subsets)
        
        def recursive_mpb(d, default_value):
            for key, value in d.items():
                if isinstance(value, dict):
                    # d[key] = replace_final_values(value, default_program_builder)
                    d[key] = cls.from_mapping(default_program_builder,value,_scheduler_options)
                else:
                    d[key] = deepcopy(default_program_builder)
            return d
        
        return cls(recursive_mpb(structure,default_program_builder),channel_subsets)
    
    
    @cached_property
    def _flattened_channel_subsets(self) -> Dict[str,Set[ChannelID]]:
        def flatten_dict(d: Dict[str, Union[Set[Any], 'Dict']], parent_key: str = '', separator: str = '.') -> Dict[str, Set[Any]]:

            flattened = {}
        
            for k, v in d.items():
                # new_key = parent_key + separator + k if parent_key else k
                new_key = k
                assert new_key!=parent_key

                if isinstance(v, dict):
                    # Recursive case: flatten the nested dictionary
                    flattened.update(flatten_dict(v, new_key, separator))
                elif isinstance(v, set):
                    # Base case: add the set to the flattened dictionary
                    flattened[new_key] = v
                else:
                    raise ValueError(f"Unsupported value type: {type(v)} at key {new_key}")
        
            return flattened
    
        return flatten_dict(deepcopy(self._channel_subsets))
    
    
    def _get_subbuilder(self, target_key: str) -> ProgramBuilder:
        
        stack = [self._program_builder_map]  # Stack for DFS, starting with the root dictionary
    
        while stack:
            current_dict = stack.pop()  # Get the last element from the stack (LIFO)
            
            for key, value in current_dict.items():
                if key == target_key:
                    return value  # Return the value if the target key is found
                
                # If the value is a MultiProgramBuilder, add it to the stack to continue searching
                if isinstance(value, MultiProgramBuilder):
                    stack.append(value._program_builder_map)
        
        return None  # Return None if the key is not found
            
    
    @property
    def program_builder_map(self) -> Dict[str,ProgramBuilder|Self]:
        return self._program_builder_map
    
    def inner_scope(self, scope: Scope, pt_obj: Optional['ForLoopPT']=None) -> Mapping[str,Scope]:
        #???
        # if self._stack[-1][0] in {"top","sequence",}:
        #     return {k: pb.inner_scope(scope,pt_obj) for k,pb in self._stack[-1][1].items()}
        # elif self._stack[-1][0] in {"iteration","repetition"}:
        #     return {k: pb.inner_scope(scope,pt_obj) for k,pb in self._stack[-1][1].items()}
        # return None #DUMMY
        return scope,pt_obj #DUMMY

        # return {k: pb.inner_scope(scope,pt_obj) for k,pb in self.program_builder_map.items()}

        

    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[str, HardwareVoltage]):
        #delegate to subbuilders? defining hold should be enough as this is used for fillers in SchedulerPT
        
        grouped_channels = ()
        handled_chs = set()
        for key,channel_set in self._flattened_channel_subsets.items():
            relevant_chs = channel_set.intersection(voltages.keys())
            if not (len(relevant_chs)==0 or len(relevant_chs)==len(channel_set)):
                raise RuntimeError()
            if any(ch in handled_chs for ch in relevant_chs):
                raise RuntimeError()
                
            self._get_subbuilder(key).hold_voltage(duration, {ch: voltages[ch] for ch in relevant_chs})
            
            handled_chs = handled_chs.union(relevant_chs)
        
        assert len(handled_chs)==len(voltages), f'{len(handled_chs)},{len(voltages)}'
        # if len(handled_chs)!=len(voltages):
        #     print(voltages)
        #     print(handled_chs)
        #     raise RuntimeError()
        
    def play_arbitrary_waveform(self, waveform: Waveform):
        
        SubsetWaveform
        
        #delegate to subbuilders? defining hold should be enough as this is used for fillers in SchedulerPT
        
        grouped_channels = ()
        handled_chs = set()
        for key,channel_set in self._flattened_channel_subsets.items():
            relevant_chs = channel_set.intersection(waveform.defined_channels)
            if not (len(relevant_chs)==0 or len(relevant_chs)==len(channel_set)):
                raise RuntimeError()
            if any(ch in handled_chs for ch in relevant_chs):
                raise RuntimeError()
                
            self._get_subbuilder(key).play_arbitrary_waveform(SubsetWaveform(waveform, relevant_chs))
            
            handled_chs = handled_chs.union(relevant_chs)
        
        assert len(handled_chs)==len(waveform.defined_channels)
        
        # """"""
        # raise RuntimeError()

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        """Unconditionally add given measurements relative to the current position."""
        # if measurements:
        # let all subbuilders measure
        for pb in self.program_builder_map.values():
            pb.measure(measurements)
            
    # def _with_repetition_if_sub_mpb(self,
    def _with_repetition(self,

                            # subset_key: str,
                            repetition_count: RepetitionCount,
                            measurements: Optional[Sequence[MeasurementWindow]]) -> Iterable['ProgramBuilder']:
        
        # assert isinstance(self.program_builder_map[subset_key],MultiProgramBuilder)
        # flattened_builders = list(flatten_dict(self.program_builder_map[subset_key]).values())
        flattened_builders = list(flatten_dict(self.program_builder_map,
                                               type_to_flatten=MultiProgramBuilder,
                                               attribute_to_access="program_builder_map").values())
        assert not any(isinstance(flat_builder,MultiProgramBuilder) for flat_builder in flattened_builders)
        
        flattened_generators = [b.with_repetition(repetition_count,measurements) for b in flattened_builders]
        
        for i,builders_vertical in enumerate(zip_longest(*flattened_generators,fillvalue=None)):
            #the builders 
            
            for b in builders_vertical:
                if b is None:
                    b._donotcreatenext.append([])

            yield self

        
            
    def with_repetition(self, repetition_count: RepetitionCount,
                        measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        # self._stack.append(('repetition',{k:pb.with_repetition(repetition_count,measurements) for k,pb in self.program_builder_map.items()}))
        self._stack.append(('repetition',{k:(pb,repetition_count,measurements) for k,pb in self.program_builder_map.items()}))
        # for k,subbuilder in self.program_builder_map.items():
            
        yield self
        self._stack.pop()
    
    @contextlib.contextmanager
    def with_sequence(self,
                      measurements: Optional[Sequence[MeasurementWindow]] = None) -> ContextManager['ProgramBuilder']:
        context_managers = {k:pb.with_sequence(measurements) for k,pb in self.program_builder_map.items()}
        with contextlib.ExitStack() as stack:
            context_objects = {k:stack.enter_context(cm) for k,cm in context_managers.items()}
            # for cm in self._stack[-1][1].values():
                # stack.enter_context(cm)
            self._stack.append(('sequence',context_objects))
            yield self
        self._stack.pop()

    def new_subprogram(self, global_transformation: 'Transformation' = None) -> ContextManager['ProgramBuilder']:
        """Create a context managed program builder whose contents are translated into a single waveform upon exit if
        it is not empty."""
        raise NotImplementedError()
    
    # @contextlib.contextmanager
    
    # def _with_iteration_if_sub_mpb(self,
    def _with_iteration(self,

                            # subset_key: str,
                            index_name: str, rng: range,
                            pt_obj: 'ForLoopPT', #hack this in for now.
                            # can be placed more suitably, like in pulsemetadata later on, but need some working thing now.
                            measurements: Optional[Sequence[MeasurementWindow]]) -> Iterable['ProgramBuilder']:
        
        # assert isinstance(self.program_builder_map[subset_key],MultiProgramBuilder)
        # flattened_builders = list(flatten_dict(self.program_builder_map[subset_key]).values())
        flattened_builders = list(flatten_dict(self.program_builder_map,
                                               type_to_flatten=MultiProgramBuilder,
                                               attribute_to_access="program_builder_map").values())

        assert not any(isinstance(flat_builder,MultiProgramBuilder) for flat_builder in flattened_builders)
        
        flattened_generators = [b.with_iteration(index_name,rng,pt_obj,measurements) for b in flattened_builders]
        
        for i,builders_vertical in enumerate(zip_longest(*flattened_generators,fillvalue=None)):
            #the builders 
            
            for b in builders_vertical:
                if b is None:
                    b._donotcreatenext.append([])

            yield self
    
    def with_iteration(self, index_name: str, rng: range,
                       pt_obj: 'ForLoopPT', #hack this in for now.
                       # can be placed more suitably, like in pulsemetadata later on, but need some working thing now.
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        # self._stack.append(('iteration',{k:pb.with_iteration(index_name,rng,pt_obj,measurements) for k,pb in self.program_builder_map.items()}))
        self._stack.append(('iteration',{k:(pb,index_name,rng,pt_obj,measurements) for k,pb in self.program_builder_map.items()}))

        yield self
        self._stack.pop()
    
    def evaluate_nested_stepping(self, scope: Scope, parameter_names: set[str]) -> bool:
        return False
    
    def to_program(self,
                   # defined_channels: Set[ChannelID]
                   ) -> Optional[Dict[str,Program|Self]]:
        top = self._stack.pop()
        assert top[0]=='top'
        assert not self._stack
        assert not self._donotcreatenext
        return MultiProgram({k:sub.to_program(
            # self._channel_subsets[k]
            ) for k,sub in self.program_builder_map.items()})
            
        
        
class MultiProgram:
    
    def __init__(self, program_map: Dict[str,Union[Program,"MultiProgram"]]):
        
        self._program_map = program_map
        
    @property
    def program_map(self) -> Dict[str,Union[Program,"MultiProgram"]]:
        return self._program_map
    
    
    def get_measurement_windows(self, drop: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        
        if drop:
            print('ignorign drop measurements for now')
        
        meas = {}
        
        for program in self._flattened_program_map:
            prog_meas_win = program.get_measurement_windows()
            for key, begins_lengths_tuple in prog_meas_win.items():
                if key in meas.keys():
                    
                    new_begins = np.concatenate((meas[key][0], begins_lengths_tuple[0]))
                    new_lengths = np.concatenate((meas[key][1], begins_lengths_tuple[1]))
                    
                    #just delete non-unique(?)
                    #if the lengths would have been different for different begins,
                    #this still would have been an overlap and error
                    new_begins, idxs = np.unique(new_begins,return_index=True)
                    new_lengths = new_lengths[idxs]
                
                    meas[key] = (new_begins,new_lengths)
                    
                else:
                    meas[key] = begins_lengths_tuple
                
                
        return meas
    
    @cached_property
    def _flattened_program_map(self) -> Dict[str,Program]:
        
        return flatten_dict(self.program_map)
        
    
def flatten_dict(input_dict: Dict[str,Any|Self],
                 parent_key: str = '',
                 type_to_flatten: type = MultiProgram,
                 attribute_to_access: str = "program_map"
                         ) -> Dict[str,Any]:
    new_dict = {}

    for k, v in input_dict.items():
        # Construct the new key
        # new_key = parent_key + separator + k if parent_key else k
        new_key = k
        assert new_key != parent_key
        
        if isinstance(v, type_to_flatten):
            # If the value is a dictionary, recursively flatten it
            # nested_dict = flatten_mp_dict(v.program_map, new_key,)
            nested_dict = flatten_dict(getattr(v,attribute_to_access), new_key,)

            new_dict.update(nested_dict)
        else:
            # If the value is a set, add it to the new dictionary
            new_dict[new_key] = v

    return new_dict