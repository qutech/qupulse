# -*- coding: utf-8 -*-

import textwrap
from abc import ABC, abstractmethod
from typing import Tuple, Iterator, List, Iterable, Dict
from zhinst.toolkit import CommandTable
from qupulse._program._loop import make_compatible
from qupulse.pulses.pulse_template import PulseTemplate
from qupulse.pulses import ForLoopPT, PointPT, SequencePT, AtomicMultiChannelPT
from qupulse.hardware.awgs.zihdawg import  HDAWGChannelGroup
from qupulse._program.seqc import HDAWGProgramEntry, WaveformMemory, BinaryWaveform, ConcatenatedWaveform
from qupulse.hardware.awgs.base import AWGAmplitudeOffsetHandling
from qupulse.hardware.setup import PlaybackChannel, MarkerChannel, HardwareSetup
from qupulse.hardware.dacs.alazar import AlazarCard
from qupulse.hardware.setup import MeasurementMask
from atsaverage.operations import Downsample, ChunkedAverage
from qupulse.pulses.pulse_template import PulseTemplate
from qupulse.program.linspace import LinSpaceBuilder, to_increment_commands, LoopLabel, Play, Wait, LoopJmp, Set, Increment
from qupulse.utils.types import TimeType
from zhinst.toolkit import CommandTable
from itertools import islice, groupby
from typing import Tuple, List, Union, Optional
from dataclasses import dataclass
from enum import Enum

import time
import json

import numpy as np
import matplotlib.pyplot as plt
# import gridspec

class FixedStructureProgram(ABC):
    
    CHANNELSTRING = 'ABCDEFGH'
    NEW_MIN_QUANT = 32
    FILE_NAME_TEMPLATE = '{hash}.csv'
    ALAZAR_CHANNELS = "ABCD"
    MEASUREMENT_NAME_TEMPLATE = 'M{i}'
    
    def __init__(self, 
                 name: str,
                 final_pt: PulseTemplate,
                 parameters: dict,
                 measurement_channels: List[str],
                 awg_object: HDAWGChannelGroup,
                 hardware_setup: HardwareSetup,
                 dac_object: AlazarCard, #or possibly other dac object?
                 auto_register: bool = True,
                 ):
        
        self._name = name
        self._awg = awg_object
        self._dac = dac_object
        self._hardware_setup = hardware_setup
        self._seqc_body = None
        
        self._loop_obj = final_pt.create_program(parameters=parameters)
        
        self.original = self._loop_obj
        
        self._measurement_channels = measurement_channels
        if len(self._measurement_channels) != 0 and self._dac is not None:
            self.register_measurements(measurement_channels, list(final_pt.measurement_names))
        self._measurement_result = None
        
        loop_method_list = [method_name for method_name in dir(self._loop_obj.__class__) if callable(getattr(self._loop_obj.__class__, method_name)) and not method_name.startswith("__")]
        loop_attribute_list = [method_name for method_name in dir(self._loop_obj.__class__) if not callable(getattr(self._loop_obj.__class__, method_name)) and not method_name.startswith("__")]

        for name in loop_method_list:
            self.add_loop_method(name)
        for name in loop_attribute_list:
            self.add_loop_attribute(name)
        

        #TODO: remove if already registered (should be just .remove_prgoram); but is it clean? - seems to work
        #TODO: run_callback - seems to work 
        #TODO: if not auto registered with this run_callback, plot func will not display correct result as not measure_program called automatically?
        #TODO: some weird error about "arming program without operations".... appears when adding with same name. perhaps some bug in qupulse/alazar.py?

        if auto_register:
            self._hardware_setup.remove_program(self.name)
            self._hardware_setup.register_program(self.name,self,run_callback=self.run_func)
        
        return
    
    def add_loop_method(self,method_name):
        setattr(self.__class__,method_name,eval('self._loop_obj.'+method_name))
        return
    
    def add_loop_attribute(self,attribute_name):
        setattr(self.__class__,attribute_name,eval('self._loop_obj.'+attribute_name))
        return
    
    def run_func(self):
        print("I'm executed")
        self._awg.run_current_program()
        if len(self._measurement_channels) != 0 and self._dac is not None:
            # self._measurement_result = self._dac.measure_program(self._measurement_channels)
            #!!! only works with measure_program copied from alazar2, where channels=None->self-inferred
            self._measurement_result = self._dac.measure_program()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def measurement_result(self) -> dict:
        return self._measurement_result
    
    @property
    @abstractmethod
    def corresponding_pt(self) -> Tuple[PulseTemplate,dict]:
        pass
    
    @property
    @abstractmethod
    def waveform_dict(self) -> dict:
        pass
    
    @abstractmethod
    def expand_ct(self,
            ct: CommandTable,
            starting_index: int
            ) -> CommandTable:
        pass
    
    @abstractmethod
    def wf_definitions_iter(self, indent: str) -> Iterator[Tuple[str,str]]:
        pass
    
    @abstractmethod
    def get_seqc_program_body(self, pos_var_name: str) -> str:
        pass
    
    @abstractmethod
    def plot_all_measurements(self):
        pass
    
    def measure_program(self) -> dict:
        self._measurement_result = self._dac.measure_program()
        return self.measurement_result
        
    #TODO: something with measurements
    #TODO: other op. than Downsample?
    def register_measurements(self,
                              measurement_channels: List[str], #from ["A","B","C","D"]
                              mask_list: Iterable[str],
                              ):
        
        operations = []
        self._dac.update_settings = True
        #!!! DONT USE i; CAUSE LIST SHUFFLED
        for i,mask in enumerate(mask_list):
            for channel in measurement_channels:
                id_string = f'{mask}{channel}'
                self._dac.register_mask_for_channel(id_string, self.ALAZAR_CHANNELS.find(channel))
                operations.append(Downsample(id_string,id_string))
                # operations.append(ChunkedAverage(id_string,id_string,chunkSize=100))

        for i,mask in enumerate(mask_list):
            self._hardware_setup.set_measurement(mask,
                                                 [MeasurementMask(self._dac, f'{mask}{channel}') for channel in measurement_channels],
                                                 allow_multiple_registration=True)
            
        self._dac.register_operations(self.name,operations)
        self._dac.update_settings = True #in MB's example set again at this stage; any difference?
            
    #TODO:
    def pt_to_binaries(self,pt,parameters,
                       ) -> tuple[BinaryWaveform]:

        def get_default_info(awg):
            return ([None] * awg.num_channels,
                    [None] * awg.num_channels,
                    [None] * awg.num_markers)
        
        playback_ids, voltage_trafos, marker_ids = get_default_info(self._awg)
        
        for channel_id in pt.defined_channels:
            for single_channel in self._hardware_setup._channel_map[channel_id]:
                if isinstance(single_channel, PlaybackChannel):
                    playback_ids[single_channel.channel_on_awg] = channel_id
                    voltage_trafos[single_channel.channel_on_awg] = single_channel.voltage_transformation
                elif isinstance(single_channel, MarkerChannel):
                    marker_ids[single_channel.channel_on_awg] = channel_id

        loop = pt.create_program(parameters=parameters)
        
        # from awg.upload
        # Go to qupulse nanoseconds time base.
        q_sample_rate = self._awg.sample_rate / 10**9

        # Adjust program to fit criteria.
        #TODO: is this required/favorable?
        make_compatible(loop,
                        minimal_waveform_length=self._awg.MIN_WAVEFORM_LEN,
                        waveform_quantum=self._awg.WAVEFORM_LEN_QUANTUM,
                        sample_rate=q_sample_rate)

        #TODO: FORCE LOOP TO BE LEAF IN ITSELF?

        if self._awg._amplitude_offset_handling == AWGAmplitudeOffsetHandling.IGNORE_OFFSET:
            voltage_offsets = (0.,) * self._awg.num_channels
        elif self._awg._amplitude_offset_handling == AWGAmplitudeOffsetHandling.CONSIDER_OFFSET:
            voltage_offsets = self._awg.offsets()
        else:
            raise ValueError('{} is invalid as AWGAmplitudeOffsetHandling'.format(self._amplitude_offset_handling))

        amplitudes = self._awg.amplitudes()
        
        helper_program_entry = HDAWGProgramEntry(loop,
                                                 selection_index=0,
                                                 waveform_memory=WaveformMemory(self._awg),
                                                 program_name="dummy_entry",
                                                 #TODO: handle channels, marks, voltage_trafos correctly
                                                 channels=tuple(playback_ids),
                                                 markers=tuple(marker_ids),
                                                 voltage_transformations=tuple(voltage_trafos),
                                                 amplitudes=amplitudes,
                                                 offsets=voltage_offsets,
                                                 sample_rate=q_sample_rate,
                                                 command_tables={0:'',1:'',2:'',3:''}
                                                 )
        
        wf = ConcatenatedWaveform()
        
        #should be ordered (?)
        for waveform, bin_tuple in helper_program_entry._waveforms.items():
            wf.append(bin_tuple)
            
        wf.finalize()
        
        return wf.as_binary()
 
#%% this is very ugly - just copied and adapted from hacky implementation of other things


class LinspaceProgram(FixedStructureProgram):
    
    MIN_SAMPLES = 32

    def __init__(self,
                 # name: str,
                 pt: PulseTemplate,

                 awg_object: HDAWGChannelGroup,
                 hardware_setup: HardwareSetup,
                 
                 ):
        
        
        self._awg = awg_object
        self._awgs = (awg_object,)
        self.pt = pt
        
        
        self.binary_wf_tuple, self.inner_wave_names = None, {}
        self.binary_wf_dict = None
        
        super().__init__('',pt,{},
                         measurement_channels=[],
                         awg_object=awg_object,hardware_setup=hardware_setup,dac_object=None,
                         auto_register=False,)#TODO: measurements)
        
        
        _, _, self.seqc, _ = pt_to_seqc(self.pt)
        
       
    @property
    def corresponding_pt(self) -> Tuple[PulseTemplate,dict]:
        return self.pt, {}
   
    
    def get_seqc_program_body(self, pos_var_name: str, indent="  ") -> Tuple[str,int]:
    
        seqc = self.seqc
        
        return textwrap.indent(textwrap.dedent(seqc),indent)
    
    def _compile_bwf(self):
        self.binary_wf_dict = {
           # 'TEST': self.pt_to_binaries(self._prepend_pt,self.parameters_scsp,self.prepend_div),
                             }
       # self.inner_wave_names = ["linspace_"+str(id(x)) for name_part,part_waves in self.binary_wf_dict.items() for awg_waves in part_waves.values() for x in awg_waves]
        self.inner_wave_names = []
       
        
    def wf_definitions_iter(self,awg) -> Iterator[Tuple[str,str]]:
                
        if self.binary_wf_dict is None:
            self._compile_bwf()
        
        f_counter = 0
        for binary,wf_name in zip([x for name_part,part_waves in self.binary_wf_dict.items() for awg_waves in part_waves.values() for x in awg_waves],self.inner_wave_names):
            f_name = self.FILE_NAME_TEMPLATE.format(hash=wf_name.replace('.csv','')+'_'+str(f_counter))
            f_counter += 1
            yield f_name, binary
    
    
    def wf_declarations_and_ct(self,awg: HDAWGChannelGroup,
                               ct_tuple: Dict[int,str],
                               ct_start_index: int,
                               wf_start_index: int,
                               waveforms_tuple: tuple):
        
        # print(awg)
        assert awg in self._awgs, "unknown awg, shouldn't have happened"
        
        awg_idx = self._awgs.index(awg) #index for the channel tuple. is not what we want here, i think.
        quant = self.MIN_SAMPLES
        
        
        for i in range(4):
            ct_tuple[i], next_ct_idx, _, ones_wf_id = pt_to_seqc(self.pt,initial_ct=ct_tuple[i],
                                                     ct_start_idx=ct_start_index,
                                                     wf_start_index=wf_start_index
                                                     )
        
        wf_decl_string = []
        ones_dcl_str = ','.join([f'placeholder({quant},true,true)']*8) #all hardcoded for now...
        wf_decl_string.append(f'assignWaveIndex({ones_dcl_str},{ones_wf_id});')
        
        for i in range(4):
            waveforms_tuple[i][ones_wf_id] = (np.ones(quant),np.ones(quant),(0b1111*np.ones(quant)).astype(int))
        
        return '\n'.join(wf_decl_string), next_ct_idx, ones_wf_id+1, waveforms_tuple

    

    def waveform_dict(self) -> dict:
        if self.binary_wf_dict is None:
            self._compile_bwf()
        
        #TODO: originally intended as "Waveform" key, but did not trace correct definitions yet
        #not needed anymore
        # return {0: self.binary_wf_dict['prepend'], 1: self.binary_wf_dict['reload'], 2: self.binary_wf_dict['postpend']}
        return {0: None}

    
    #here not necessary to employ? Seems to be.
    def expand_ct(self,
            ct: CommandTable,
            starting_index: int
            ) -> CommandTable:
        
        assert self._seqc_body is not None
        return ct
    
    def get_2d_data_dict(self) -> np.ndarray:
        pass
    
    def plot_all_measurements(self):
        pass


def get_mod2_amp(ct,i):
    if i%2==0:
        return ct.amplitude0
    else:
        return ct.amplitude1


#%% part from other file

recent_awg_pull = {'schema': [{'timestamp': 9069517248534394, 'flags': 0, 'vector': '{\n  "$schema": "https://json-schema.org/draft-07/schema#",\n  "title": "AWG Command Table Schema",\n  "description": "Schema for ZI HDAWG AWG Command Table",\n  "version": "1.2.1",\n  "definitions": {\n    "header": {\n      "type": "object",\n      "properties": {\n        "version": {\n          "type": "string",\n          "pattern": "^1\\\\.[0-2](\\\\.[0-9]+)?$",\n          "description": "File format version (Major.Minor / Major.Minor.Patch). This version must match with the relevant schema version."\n        },\n        "partial": {\n          "description": "Set to true for incremental table updates",\n          "type": "boolean",\n          "default": false\n        },\n        "userString": {\n          "description": "User-definable label",\n          "type": "string",\n          "maxLength": 30\n        }\n      },\n      "required": [\n        "version"\n      ]\n    },\n    "table": {\n      "type": "array",\n      "items": {\n        "$ref": "#/definitions/entry"\n      },\n      "minItems": 0,\n      "maxItems": 1024\n    },\n    "entry": {\n      "type": "object",\n      "properties": {\n        "index": {\n          "$ref": "#/definitions/tableindex"\n        },\n        "waveform": {\n          "$ref": "#/definitions/waveform"\n        },\n        "phase0": {\n          "$ref": "#/definitions/phase"\n        },\n        "phase1": {\n          "$ref": "#/definitions/phase"\n        },\n        "amplitude0": {\n          "$ref": "#/definitions/amplitude"\n        },\n        "amplitude1": {\n          "$ref": "#/definitions/amplitude"\n        }\n      },\n      "additionalProperties": false,\n      "anyOf": [\n        {\n          "required": [\n            "index",\n            "waveform"\n          ]\n        },\n        {\n          "required": [\n            "index",\n            "phase0"\n          ]\n        },\n        {\n          "required": [\n            "index",\n            "phase1"\n          ]\n        },\n        {\n          "required": [\n            "index",\n            "amplitude0"\n          ]\n        },\n        {\n          "required": [\n            "index",\n            "amplitude1"\n          ]\n        }\n      ]\n    },\n    "tableindex": {\n      "type": "integer",\n      "minimum": 0,\n      "maximum": 1023\n    },\n    "waveform": {\n      "type": "object",\n      "properties": {\n        "index": {\n          "$ref": "#/definitions/waveformindex"\n        },\n        "length": {\n          "$ref": "#/definitions/waveformlength"\n        },\n        "samplingRateDivider": {\n          "$ref": "#/definitions/samplingratedivider"\n        },\n        "awgChannel0": {\n          "$ref": "#/definitions/awgchannel"\n        },\n        "awgChannel1": {\n          "$ref": "#/definitions/awgchannel"\n        },\n        "precompClear": {\n          "$ref": "#/definitions/precompclear"\n        },\n        "playZero": {\n          "$ref": "#/definitions/playzero"\n        },\n        "playHold": {\n          "$ref": "#/definitions/playhold"\n        }\n      },\n      "additionalProperties": false,\n      "oneOf": [\n        {\n          "required": [\n            "index"\n          ]\n        },\n        {\n          "required": [\n            "playZero",\n            "length"\n          ]\n        },\n        {\n          "required": [\n            "playHold",\n            "length"\n          ]\n        }\n      ]\n    },\n    "waveformindex": {\n      "description": "Index of the waveform to play as defined with the assignWaveIndex sequencer instruction",\n      "type": "integer",\n      "minimum": 0,\n      "maximum": 15999\n    },\n    "waveformlength": {\n      "description": "The length of the waveform in samples",\n      "type": "integer",\n      "multipleOf": 16,\n      "minimum": 32\n    },\n    "samplingratedivider": {\n      "descpription": "Integer exponent n of the sample rate divider: SampleRate / 2^n, n in range 0 ... 13",\n      "type": "integer",\n      "minimum": 0,\n      "maximum": 13\n    },\n    "awgchannel": {\n      "description": "Assign the given AWG channel to signal output 0 & 1",\n      "type": "array",\n      "minItems": 1,\n      "maxItems": 2,\n      "uniqueItems": true,\n      "items": [\n        {\n          "type": "string",\n          "enum": [\n            "sigout0",\n            "sigout1"\n          ]\n        }\n      ]\n    },\n    "precompclear": {\n      "description": "Set to true to clear the precompensation filters",\n      "type": "boolean",\n      "default": false\n    },\n    "playzero": {\n      "description": "Play a zero-valued waveform for specified length of waveform, equivalent to the playZero sequencer instruction",\n      "type": "boolean",\n      "default": false\n    },\n    "playhold": {\n      "description": "Hold the last played value for the specified number of samples, equivalent to the playHold sequencer instruction",\n      "type": "boolean",\n      "default": false\n    },\n    "phase": {\n      "type": "object",\n      "properties": {\n        "value": {\n          "description": "Phase value of the given sine generator in degree",\n          "type": "number"\n        },\n        "increment": {\n          "description": "Set to true for incremental phase value, or to false for absolute",\n          "type": "boolean",\n          "default": false\n        }\n      },\n      "additionalProperties": false,\n      "required": [\n        "value"\n      ]\n    },\n    "amplitude": {\n      "type": "object",\n      "properties": {\n        "value": {\n          "description": "Amplitude scaling factor of the given AWG channel",\n          "type": "number",\n          "minimum": -1.0,\n          "maximum": 1.0\n        },\n        "increment": {\n          "description": "Set to true for incremental amplitude value, or to false for absolute",\n          "type": "boolean",\n          "default": false\n        },\n        "register": {\n          "description": "Index of amplitude register that is selected for scaling the pulse amplitude.",\n          "type": "integer",\n          "minimum": 0,\n          "maximum": 3\n        }\n      },\n      "additionalProperties": false,\n      "anyOf": [\n        {\n          "required": [\n            "value"\n          ]\n        },\n        {\n          "required": [\n            "register"\n          ]\n        }\n      ]\n    }\n  },\n  "type": "object",\n  "properties": {\n    "$schema": {\n      "type": "string"\n    },\n    "header": {\n      "$ref": "#/definitions/header"\n    },\n    "table": {\n      "$ref": "#/definitions/table"\n    }\n  },\n  "additionalProperties": false,\n  "required": [\n    "header",\n    "table"\n  ]\n}\n'}]}

#this is normally pulled from the device, placed here just to make it work without hardware connection
EXAMPLE_SCHEMA = recent_awg_pull['schema'][0]['vector']


@dataclass
class Commands:
    Play = Play
    Increment = Increment
    Set = Set
    LoopLabel = LoopLabel
    LoopJmp = LoopJmp
    Wait = Wait
    

def group_commands(commands: List) -> List[Tuple]:
    
    return [tuple(group) for _, group in groupby(commands, key=type)]


class _CompatibilityLevel(Enum):
    compatible = 0
    action_required = 1
    incompatible_too_short = 2
    incompatible_fraction = 3
    incompatible_quantum = 4

    def is_incompatible(self) -> bool:
        return self in (self.incompatible_fraction, self.incompatible_quantum, self.incompatible_too_short)

def check_compat_hdawg(commands_grouped: List) -> bool:
    
    
    #- check if can be done with available registers
    #- set followed by wait, and minimum wait compatible with 2*32+n*16 at sampling rate
    #- same for increment
    #- waveform compatibility
    #- ...
    
    return True
    

def get_ct_and_seqc_from_commands(commands: list,
                                  ct_start_idx: int = 0,
                                  wf_start_index: int = 0,
                                  initial_ct: Optional[CommandTable] = None,
                                  ) -> Tuple[CommandTable, int, str]:
    
    valid_commands = Commands()
    ct = CommandTable(EXAMPLE_SCHEMA,active_validation=False) if initial_ct is None else initial_ct
    # activate_validation=False is important, as otherwise ZI decided to do
    # costly checks after every change in CT, which amounts to ridiculous runtime
    
    #group by type such that consecutive instructions are combined (e.g. for Set combinable to one CT entry)
    commands_grouped = group_commands(commands)
    
    ct_idx = ct_start_idx
    ones_segment_id = [type(c[0]) in (valid_commands.Set,valid_commands.Increment) for c in commands_grouped].index(True)
    commands_grouped_iter = iter(commands_grouped)
    seqc_lines = []
    wf_decl_lines = []
    indent_level, indent_spaces = 1, 2
    
    DUMMY_RATE, MIN_SAMPLES = 1, 32
    
    borrowed_samples: int = 0
    
    #registers, one possibility:
    MAX_REGISTERS = 4
    START_REG = 1 #reserve 0 for normal wf playback
    ch_reg_iter = [iter(range(START_REG,MAX_REGISTERS)) for i in range(2)]
    depkey_to_reg_dict_list = [{} for i in range(2)]
    
    for command_tuple in commands_grouped_iter:
        if ct_idx > 1023:
            raise RuntimeError('too many CT entries in program')
        match type(command_tuple[0]):
            case valid_commands.Play:
                
                raise NotImplementedError('too lazy to handle arbitrary wf in first test')
                
                #hacky way, move to compatibility check
                if borrowed_samples != 0:
                    raise RuntimeError()
                
                for play in command_tuple:
                    #reserve register 0 for 'normal wf playback' for now.
                    #could also resort to playWave here to save CT entries, which might be more clever
                    ct.table[ct_idx].amplitude0.register = 0
                    ct.table[ct_idx].amplitude1.register = 0
                    ct.table[ct_idx].amplitude0.value = 1.0
                    ct.table[ct_idx].amplitude1.value = 1.0
                    ct.table[ct_idx].amplitude0.increment = False
                    ct.table[ct_idx].amplitude1.increment = False
                    
                    # could decouple this since it is not strictly dependent,
                    # but might be easier to debug if 1-1 correspondence,
                    # and have many more wf entries (16k) than CT (1k) anyway
                    ct.table[ct_idx].waveform.index = ct_idx
                    #TODO: handle sampling rate accordingly.
                    ct.table[ct_idx].waveform.length = play.waveform.duration * DUMMY_RATE
                    
                    seqc_lines.append(f'{indent_level*indent_spaces*" "}executeTableEntry({ct_idx});')
                    # wf_decl_lines.append(TODO)
                    #TODO
                    ct_idx += 1
                continue
            case valid_commands.LoopLabel:
                if borrowed_samples != 0:
                    raise RuntimeError()
                for repeat in command_tuple:
                    seqc_lines.append('{}repeat({}){{'.format(indent_level*indent_spaces*' ',repeat.count))
                    indent_level += 1 if indent_level < 4 else NotImplementedError('more celver way to handle nested loops for registers needed')
                continue
            case valid_commands.LoopJmp:
                if borrowed_samples != 0:
                    raise RuntimeError()
                for repeat_end in command_tuple:
                    seqc_lines.append(f'{indent_level*indent_spaces*" "}}}')
                    indent_level -= 1 if indent_level > 1 else RuntimeError('shouldnt have happened, makes no sense')
                continue
            case valid_commands.Set:
                if borrowed_samples != 0:
                    raise RuntimeError()
                    
                # #default: 0 - bad idea, probably. but ensure 'Set' on channel precedes 'Increment', otherwise undefined behavior
                # # ct.table[ct_idx].amplitude0.register = indent_level-1
                # ct.table[ct_idx].amplitude0.register = 1
                # ct.table[ct_idx].amplitude0.value = 0
                # ct.table[ct_idx].amplitude0.increment = False
                # # ct.table[ct_idx].amplitude0.register = indent_level-1
                # ct.table[ct_idx].amplitude0.register = 1
                # ct.table[ct_idx].amplitude1.value = 0
                # ct.table[ct_idx].amplitude1.increment = False
                    
                set_chans = []
                for setter in command_tuple:
                    #TODO: needs the correct handling of multiple CTs to make sense
                    if setter.channel in set_chans:
                        raise RuntimeError('Does not make sense to set same channel twice / handle accordingly')
                    ct.table[ct_idx].waveform.index = ones_segment_id
                    match setter.channel:
                        case 0:
                            #TODO: register handling. <- might be done with depkey now.
                            #TODO: amplitude handling (scaling by awg amp.)
                            # ct.table[ct_idx].amplitude0.register = indent_level-1
                            ct.table[ct_idx].amplitude0.register = depkey_to_reg_dict_list[0][setter.key] if setter.key in depkey_to_reg_dict_list[0].keys() else depkey_to_reg_dict_list[0].setdefault(setter.key,next(ch_reg_iter[0]))
                            ct.table[ct_idx].amplitude0.value = setter.value
                            ct.table[ct_idx].amplitude0.increment = False
                        case 1:
                            # ct.table[ct_idx].amplitude1.register = indent_level-1
                            ct.table[ct_idx].amplitude1.register = depkey_to_reg_dict_list[1][setter.key] if setter.key in depkey_to_reg_dict_list[1].keys() else depkey_to_reg_dict_list[1].setdefault(setter.key,next(ch_reg_iter[1]))
                            ct.table[ct_idx].amplitude1.value = setter.value
                            ct.table[ct_idx].amplitude1.increment = False
                        case _:
                            raise NotImplementedError('only rudimentary example, no handling of many channels')
                    set_chans.append(setter.channel)
                seqc_lines.append(f'{indent_level*indent_spaces*" "}executeTableEntry({ct_idx});')
                ct_idx += 1
                borrowed_samples = MIN_SAMPLES
                continue
            case valid_commands.Increment:
                if borrowed_samples != 0:
                    raise RuntimeError()
                    
                # #default: 0
                # # ct.table[ct_idx].amplitude0.register = indent_level-1
                # ct.table[ct_idx].amplitude0.register = 1
                # ct.table[ct_idx].amplitude0.value = 0
                # ct.table[ct_idx].amplitude0.increment = False
                # # ct.table[ct_idx].amplitude0.register = indent_level-1
                # ct.table[ct_idx].amplitude0.register = 1
                # ct.table[ct_idx].amplitude1.value = 0
                # ct.table[ct_idx].amplitude1.increment = False
                    
                # TODO: due to 'unfortunate' (=stupid) programming from zurich, a jump of e.g. -2.0 (from max to min amp)
                # is not allowed despite doable. need to circumvent this for now.
                
                # is almost the same as Set, could combine more cleverly codewise
                incr_chans = []
                for incr in command_tuple:
                    #TODO: needs the correct handling of multiple CTs to make sense
                    if incr.channel in incr_chans:
                        raise RuntimeError('Does not make sense to set same channel twice / handle accordingly')
                    ct.table[ct_idx].waveform.index = ones_segment_id
                    match incr.channel:
                        case 0:
                            #TODO: register handling.
                            #TODO: amplitude handling (scaling by awg amp.)
                            # ct.table[ct_idx].amplitude0.register = indent_level-1
                            ct.table[ct_idx].amplitude0.register = depkey_to_reg_dict_list[0][incr.dependency_key] if incr.dependency_key in depkey_to_reg_dict_list[0].keys() else depkey_to_reg_dict_list[0].setdefault(incr.dependency_key,next(ch_reg_iter[0]))
                            ct.table[ct_idx].amplitude0.value = incr.value
                            ct.table[ct_idx].amplitude0.increment = True
                        case 1:
                            # ct.table[ct_idx].amplitude1.register = indent_level-1
                            ct.table[ct_idx].amplitude1.register = depkey_to_reg_dict_list[1][incr.dependency_key] if incr.dependency_key in depkey_to_reg_dict_list[1].keys() else depkey_to_reg_dict_list[1].setdefault(incr.dependency_key,next(ch_reg_iter[1]))
                            ct.table[ct_idx].amplitude1.value = incr.value
                            ct.table[ct_idx].amplitude1.increment = True
                        case _:
                            raise NotImplementedError('only rudimentary example, no handling of many channels')
                    incr_chans.append(incr.channel)
                seqc_lines.append(f'{indent_level*indent_spaces*" "}executeTableEntry({ct_idx});')
                ct_idx += 1
                borrowed_samples = MIN_SAMPLES
                continue
            case valid_commands.Wait:
                total_wait_time = sum([w.duration for w in command_tuple]) - borrowed_samples/DUMMY_RATE
                assert total_wait_time >= MIN_SAMPLES*DUMMY_RATE
                borrowed_samples = 0
                ct.table[ct_idx].waveform.playHold = True
                #TODO: handle sampling rate accordingly.
                ct.table[ct_idx].waveform.length = int(total_wait_time * DUMMY_RATE)
                seqc_lines.append(f'{indent_level*indent_spaces*" "}executeTableEntry({ct_idx});')
                ct_idx += 1
                continue
            case _:
                raise NotImplementedError()
                
    next_ct_idx = ct_idx
    return ct, next_ct_idx, '\n'.join(seqc_lines), ones_segment_id
    

def pt_to_seqc(pt: PulseTemplate,
               ct_start_idx: int = 0,
               wf_start_index: int = 0,
               initial_ct: Optional[CommandTable] = None,
               **kwargs
               ) -> Tuple[CommandTable,int,str]:
    
    program = pt.create_program(program_builder=LinSpaceBuilder(channels=pt.defined_channels))
    commands = to_increment_commands([program])
    
    ct, next_ct_idx, seqc_string, ones_wf_id = get_ct_and_seqc_from_commands(commands,ct_start_idx=ct_start_idx,wf_start_index=wf_start_index,initial_ct=initial_ct)


    return ct, next_ct_idx, seqc_string, ones_wf_id
