from abc import ABC
from numbers import Real
from typing import Tuple, Callable, Optional, Sequence, Union, Dict, Mapping, Set
import serial

from qupulse import ChannelID
from qupulse._program._loop import Loop, to_waveform
from qupulse.hardware.awgs.base import AWG

'''upload general plan:
    get timeline, with each instance being an index in an array
    convert every voltage value to a decadac value
    look for ramps '''
def convert_voltage(v) -> int:
    return int((v + 10) / 20 * 65535)

#TODO delete this at one point
'''class Single:

    def __int__(self, voltage, duration):
        self.d = voltage
        self. duration = duration

    def get_ascii(self):
        return 'D' + str(self.d)

class Ramp:

    def __init__(self, start_v: int, end_v: int, timestep: int, steps: int):
        self.start = start_v
        self.end = end_v
        self.timestep = timestep
        self.steps = steps
        self.lim = 'U'

    def get_ascii(self):
        if self.start > self.end:
            self.lim = 'L'
        s = int((self.end - self.start) / self.steps * 65536)
        return self.lim + str(self.end) + ';T' + str(self.timestep) + ';S' + str(s) + ';'

    def reset(self):
        edge = 65535
        if self.start > self.end:
            edge = 0
        return self.lim + str(edge)

    def increase_step(self):
        self.steps += 1

    @property
    def endv(self):
        return self.end

    @endv.setter
    def endv(self, newv):
        self.end = newv

    @property
    def T(self):
        return self.timestep

    @property
    def duration(self):
        return self.steps * self.timestep




class Iteration:

    def __int__(self, location: int, iteration_number: int, code_block: int):
        self._loc = location
        self._num = iteration_number
        self._block = code_block

    @property
    def loc(self):
        return self._loc

    @property
    def num(self):
        return self._num

    @property
    def block(self):
        return self._block

    # TODO see if you need to also return ;
    def point(self) -> str:
        return 'A' + str(self.loc)

    def set_itr(self) -> str:
        return 'P' + str(self.num)

    @staticmethod
    def decrement():
        return '+-1'
        '''

'''class Initial:
    def __int__(self, voltage, duration):
        self.voltage = voltage
        self.duration = duration'''
class Stay:
    def __init__(self, voltage):
        self.voltage = voltage
class Ramp:
    def __init__(self, v_step):
        self.v_step = v_step
        
class Assign:

    def __init__(self, voltage, channel: tuple):
        self._voltage = voltage
        self._channel = channel # in the form of (B,C) where B is the board and C is the channel. Possible values: B 0-4, C 0-3


    def get_ascii(self):
        command = 'B' + str(self._channel[0]) + ';C' + str(self._channel[1]) + ';'
        command += 'D' + str(self._voltage) + ';'
        return command

    def is_ramp(self):
        return False


class Ramp:
    def __init__(self, duration: int, voltage_step, channel: tuple, memory_dict: dict):
        self._step = voltage_step
        self._channel = channel # in the form of (B,C) where B is the board and C is the channel. Possible values: B 0-4, C 0-3
        self._memory_dict = memory_dict # dictionary correlating the channel to the location in memory for the channel voltage
        self._duration = duration

    def get_ascii(self):
        command = 'A' + str(self._memory_dict[self._channel]) + ';'
        command += '+' + str(self._step) + ';'
        return command

    def is_ramp(self):
        return True

    @property
    def duration(self):
        return self._duration


class Single:

    def __init__(self, current_label: int, duration: int):
        self._instructions = []
        self._current_label = current_label
        self._duration = duration

    def add_instruction(self, instruction):
        self._instructions.append(instruction)

    def get_ascii(self):
        command = '*' + str(self._current_label) + ':'
        command += 'X' + str(768+self._current_label) + ';'
        for instruction in self._instructions:
            if instruction.is_ramp():
                command += instruction.get_ascii()
        command += '$' + str(self._duration)
        return command, self._current_label+1


class Iteration:

    def __init__(self, current_label: int, duration: int, loops: int, memory_location: int):
        self._instructions = []
        self._current_label = current_label
        self._duration = duration
        self._memory_location = memory_location
        self._loops = loops

    def get_ascii(self):
        command = 'A' + str(self._memory_location) + 'P' + str(self._loops)
        command += '*' + str(self._current_label) + ':'
        command += 'X' + str(768 + self._current_label) + ';'
        for instruction in self._instructions:
            command += instruction.get_ascii()
        command += '$' + str(self._duration)
        command += 'A' + str(self._memory_location)
        command += 'X' + str(1280 + self._current_label) + ';'
        return command, self._current_label+1

    def add_instruction(self, instruction):
        self._instructions.append(instruction)


class DecaDACAWG(AWG):

    def __init__(self, port: str, identifier='DecaDAC',):
        super().__init__(identifier=identifier)

        if port is not str:
            raise TypeError(
                'Declared port must be a string. Usually in the form of COM + n, with n being the port number')
        self._ser = serial.Serial(port, 9600, timeout=0)
        self._channel_voltage_location = {1: 1545,
                                          2: 1561,
                                          3: 1577,
                                          4: 1593,
                                          5: 1609,
                                          6: 1625,
                                          7: 1641,
                                          8: 1657,
                                          9: 1673,
                                          10: 1689,
                                          11: 1705,
                                          12: 1721,
                                          13: 1737,
                                          14: 1753,
                                          15: 1769,
                                          16: 1785,
                                          17: 1801,
                                          18: 1817,
                                          19: 1833,
                                          20: 1849}
        self._safe_memory_locations = {40693, 40694, 40695, 40696, 40697, 40698, 40699, 40700, 40701, 40702, 40703}
        self._block_number = 1
        self._iteration_count = 0

    @property
    def num_channels(self):
        return 20

    @property
    def num_markers(self):
        return 20

    def upload(self, name: str, program: Loop,
               channels: Tuple[Optional[ChannelID], ...],
               markers: Tuple[Optional[ChannelID], ...],
               voltage_transformation: Tuple[Optional[Callable], ...],
               force: bool = False) -> None:

        '''current_label = 0
        chans = list(to_waveform(program).defined_channels)
        timeline = self.get_timeline(program)

        # TODO change all of this. It's only temporary
        command = '{'
        n = 1
        for loop in program.get_depth_first_iterator():
            if loop.is_leaf():
                c_values = loop.waveform.constant_value_dict()
                if c_values is None:
                    raise NotImplementedError('Only constant waveforms implemented for now')
                channels = list(loop.waveform.defined_channels)
                command += '*' + str(n) + ':X' + str(768 + n) + ';'
                for ch in channels:
                    voltage = int((loop.waveform.constant_value_dict()[ch] + 10) / 20 * 65535)
                    command += ch + ';D' + str(voltage) + ';'
                command += '$' + str(int(loop.waveform.duration / 1000)) + ';'
        command += '}' '''
        self._block_position = 1
        self._iteration_count = 0
        channels = list(to_waveform(program).defined_channels)
        timeline = self.get_timeline(program)
        sequence = self.parse_timeline(timeline, channels)
        window_size = 1
        while window_size< len(sequence):
            check, sequence = self.window_compression(sequence)
            if not check:
                window_size += 1
        command = self.recursive_command(sequence, channels)
        
        return '{' + command + '}'

    def remove(self) -> None:
        self._ser.write(b'{}')

    def clear(self) -> None:
        self._ser.write(b'{}')

    def arm(self) -> None:
        command = ''
        for i in range(5):
            command += f'B{i};M2;'
        command += 'X1;'
        self._ser.write(command.encode())

    @property
    def programs(self) -> Set[str]:
        return set('Only one program can exist in memory')

    @property
    def sample_rate(self) -> float:
        return 100

    def set_volatile_parameters(self, program_name: str, parameters: Mapping[str, Real]):
        pass

    def __copy__(self) -> None:
        pass

    def __deepcopy__(self, memodict={}) -> None:
        pass

    #def arbitrary_waveform(self, program: Loop):


    def get_timeline(self, program: Loop) -> list:
        '''converts the loop into a list of dictionaries. Each dictionary contains the DAC values
        for each time block and the duration of each block'''
        timeline = []
        for loop in program.get_depth_first_iterator():
            if loop.is_leaf():
                c_values = loop.waveform.constant_value_dict()
                if c_values is None:
                    raise NotImplementedError('Only constant waveforms implemented for now')
                channels = list(loop.waveform.defined_channels)
                instance = {}  # dictionary for 'ch': voltage,...., duration: duration in micro s
                for ch in channels:
                    instance[ch] = convert_voltage(loop.waveform.constant_value_dict()[ch])
                instance['duration'] = int(loop.waveform.duration / 1000)
                timeline.append(instance)
        return timeline

    def parse_timeline(self, timeline: list, channels):
        '''parses the timeline, converting each channel value to the difference in DAC value from
        the previous time block. Except for the first time block which remains unchanged'''
        temp_dict = {}
        temp_dict['duration'] = timeline[0]['duration']
        for ch in channels:
            temp_dict[ch] = 'P' + str(timeline[0][ch])
        parsed = [temp_dict]
        for i in range(1,len(timeline)):
            temp_dict = {}
            temp_dict['duration'] = timeline[i]['duration']
            for ch in channels:
                temp_dict[ch] = '+' + str(timeline[i][ch] - timeline[i-1][ch])
            parsed.append(temp_dict)
        return parsed

    def find_recursion_nn(parsed_timeline: list):
        '''looks at the nearest neighbour and does a dictionary comparison, if they match,
        it adds the current instance to the current recursion object. If they don't, it appends
        the current recursion object and starts a new one with the specifications of the current
        instance'''
        rucursed_timeline = []
        recursion_object = parsed_timeline[0].copy()
        recursion_object['repetitions'] = 0
        i=1
        while i < len(parsed_timeline):
            if i == len(parsed_timeline)-1:
                if parsed_timeline[i] == parsed_timeline[i-1]:
                    recursion_object['repetitions'] += 1
                    i += 1
                else:
                    rucursed_timeline.append(recursion_object)
                    recursion_object = parsed_timeline[i].copy()
                    recursion_object['repetitions'] = 0
                    i += 1
                rucursed_timeline.append(recursion_object)
            elif parsed_timeline[i] == parsed_timeline[i-1]:
                recursion_object['repetitions'] += 1
                i += 1
            else:
                rucursed_timeline.append(recursion_object)
                recursion_object = parsed_timeline[i].copy()
                recursion_object['repetitions'] = 0
                i += 1
        return rucursed_timeline
        #TODO optimise this
        
    def meta_recursion(rucursed_timeline: list):
        '''looks through the recursed timeline and finds thurther recursions, returnung the
        further recursed timeline and a boolean indicating whether the timeline has undergone
        additional recursion'''
    def window_compression(self, n, sequence):
        '''does a window compression, inspired by LZ77. It takes a window and compares the sequence of the same length'''
        compressed = False
        compressed_sequence = []
        rep = 1
        i = 0
        length = len(sequence)
        while i<length-n-1:
            if sequence[i:i+n]==sequence[i+n:i+2*n]:
                print(sequence[i+n:i+2*n], i)
                rep+=1
                i+=n
                compressed = True
                if not i <length-n-1:
                    compressed_sequence.append((rep,sequence[i-2*n:i-n]))
            else:
                if rep==1:
                    print(sequence[i],i)
                    compressed_sequence.append(sequence[i])
                    rep=1
                    i+=1
                elif rep>1:
                    print(sequence[i-n:i],i)
                    compressed_sequence.append((rep,sequence[i-n:i]))
                    rep=1
                    i+=n
        if i == length-1:
            compressed_sequence.append(sequence[i])
        
        if compressed:
            return compressed, compressed_sequence
        elif not compressed:
            return compressed, sequence

    def recursive_command(self, sequence, channels):
        command = ''
        rep_flag = ''
        for obj in sequence:
            obj_type = type(obj)
            if obj_type is int:
                block_number = self._block_position
                location = str(self._safe_memory_locations[self._iteration_count])
                iteration_command = 'A' + location + ';P' + str(obj) + ';'
                command += f'*{block_number}:' + iteration_command
                X_command = 1280 + block_number + 1 #go to block next block (the one being iterated) if count is not yero
                rep_flag = f'A{location};+-1;X' + str(X_command) + ';' #point to iteration count location, reduce it by 1, goto block being iterated if count not 0
                self._block_number += 1
                self._iteration_count += 1
            elif obj_type is dict:
                block_number = self._block_position
                command += f'*{block_number}:' #declare current block
                wait_command = 768 + block_number
                command += 'X' + str(wait_command) + ';' #wait till count is finished
                for ch in channels:
                    if self._channel_voltage_location[ch] == '+0':
                        continue
                    command += 'A' + str(self._channel_voltage_location[ch]) + ';'#point to voltage location
                    command += obj[ch] + ';'#change voltage
                command += '$' + obj['duration'] + ';'
            else:
                command += self.recursive_command(obj, channels)
        command += rep_flag 
        return command
        
                
            
        
        