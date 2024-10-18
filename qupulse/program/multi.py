import abc
import contextlib
import dataclasses
import numpy as np
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, ContextManager, Iterable, Tuple, Union, Dict, List, Iterator, Generic,\
    Set, Callable, Self, Any
from enum import Enum
from itertools import dropwhile, count
from numbers import Real, Number
from collections import defaultdict


from qupulse import ChannelID, MeasurementWindow
from qupulse.parameter_scope import Scope, MappedScope, FrozenDict
# from qupulse.pulses.pulse_template import PulseTemplate
# from qupulse.pulses import ForLoopPT
from qupulse.program import ProgramBuilder, HardwareTime, HardwareVoltage, Waveform, RepetitionCount, TimeType, Program
from qupulse.expressions.simple import SimpleExpression, NumVal, SimpleExpressionStepped
from qupulse.program.waveforms import MultiChannelWaveform, TransformingWaveform, WaveformCollection

from qupulse.program.transformation import ChainedTransformation, ScalingTransformation, OffsetTransformation,\
    IdentityTransformation, ParallelChannelTransformation, Transformation

from copy import deepcopy

NestedPBMapping = Dict[str, Union[ProgramBuilder, 'NestedPBMapping']]

class MultiProgramBuilder(ProgramBuilder):
    
    def __init__(self,
                 sub_program_builders: Dict[str,ProgramBuilder|Self],
                 ):
        
        super().__init__()
        self._program_builder_map = sub_program_builders

    # def get_program_builder(self, key) -> NestedPBMapping:
    #     return self._program_builder_map.setdefault(key,deepcopy(self._program_builder_map[-1]))
    
    @classmethod
    def from_mapping(cls, default_program_builder: ProgramBuilder,
                     structure: Dict[str,Any|Self],
                     ):
        
        def replace_final_values(d, default_value):
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = replace_final_values(value, default_program_builder)
                else:
                    d[key] = deepcopy(default_program_builder)
            return d
        
        return cls(replace_final_values(structure, default_program_builder))
    
    @property
    def program_builder_map(self) -> Dict[str,ProgramBuilder|Self]:
        return self._program_builder_map
    
    def inner_scope(self, scope: Scope, pt_obj: Optional['ForLoopPT']=None) -> Mapping[str,Scope]:
        #???
        return {k: pb.inner_scope(scope,pt_obj) for k,pb in self.program_builder_map.items()}
        
        

    def hold_voltage(self, duration: HardwareTime, voltages: Mapping[str, HardwareVoltage]):
        """Supports dynamic i.e. for loop generated offsets and duration"""
        raise RuntimeError()
        
    def play_arbitrary_waveform(self, waveform: Waveform):
        """"""
        raise RuntimeError()

    def measure(self, measurements: Optional[Sequence[MeasurementWindow]]):
        """Unconditionally add given measurements relative to the current position."""
        raise NotImplementedError()

    def with_repetition(self, repetition_count: RepetitionCount,
                        measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        self._stack.extend(('repetition',{k:pb.with_repetition(measurements) for k,pb in self.program_builder_map.items()}))
        yield self
        self._stack.pop()
        
    def with_sequence(self,
                      measurements: Optional[Sequence[MeasurementWindow]] = None) -> ContextManager['ProgramBuilder']:
        self._stack.extend(('sequence',{k:pb.with_sequence(measurements) for k,pb in self.program_builder_map.items()}))
        yield self
        self._stack.pop()

    def new_subprogram(self, global_transformation: 'Transformation' = None) -> ContextManager['ProgramBuilder']:
        """Create a context managed program builder whose contents are translated into a single waveform upon exit if
        it is not empty."""
        raise NotImplementedError()

    def with_iteration(self, index_name: str, rng: range,
                       pt_obj: 'ForLoopPT', #hack this in for now.
                       # can be placed more suitably, like in pulsemetadata later on, but need some working thing now.
                       measurements: Optional[Sequence[MeasurementWindow]] = None) -> Iterable['ProgramBuilder']:
        self._stack.extend(('iteration',{k:pb.with_iteration(index_name,rng,pt_obj,measurements) for k,pb in self.program_builder_map}))
        yield self
        self._stack.pop()
    
    def evaluate_nested_stepping(self, scope: Scope, parameter_names: set[str]) -> bool:
        return False
    
    def to_program(self, defined_channels: Set[ChannelID]) -> Optional[Dict[str,Program|Self]]:
        
        return MultiProgram({k:sub.to_program(defined_channels) for k,sub in self.program_builder_map.items()})
        
        
class MultiProgram:
    
    def __init__(self, program_map: Dict[str,Program|Self]):
        
        self._program_map = program_map
        
    @property
    def program_map(self) -> Dict[str,Program|Self]:
        return self._program_map