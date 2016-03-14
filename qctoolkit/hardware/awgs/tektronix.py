from bidict import bidict
import visa
import datetime
import numpy as np
from functools import reduce
import socket
import select
from itertools import chain, repeat
import ipdb 

from .awg import AWG, DummyAWG, ProgramOverwriteException, OutOfWaveformMemoryException
from qctoolkit.pulses.instructions import EXECInstruction


__all__ = ['TektronixAWG', 'AWGSocket', 'EchoTestServer']

class EchoTestServer():
    def __init__(self, port):
        self.port = port

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', self.port))
        s.listen(5)
        while True:
            client, address = s.accept()
            data = client.recv(65535)
            if data:
                client.send(data)
            client.close()

def grouper(n, iterable, padvalue=None):
    return zip(*[chain(iterable, repeat(padvalue, n-1))]*n)


class AWGSocket():
    def __init__(self, ip, port, buffersize=1024, timeout=5):
        self.__ip = ip
        self.__port = port
        self.__buffersize = buffersize
        self.__timeout = timeout

    def _cleanstring(self, bytestring):
        # accept strings as well, but encode them to bytes
        if not isinstance(bytestring, bytes):
                bytestring = bytestring.encode()
        # make sure the message is delimited by b'\n'
        if not bytestring.endswith(b'\n'):
            bytestring = bytestring + b'\n'
        return bytestring

    def send(self, bytestring):
        """Sends a command to the AWG with no answer."""
        bytestring = self._cleanstring(bytestring)
        s = socket.create_connection((self.__ip, self.__port))
        for chunk in grouper(self.__buffersize, bytestring):
            s.send(bytestring)
        s.close()

    def query(self, bytestring):
        """Queries the AWG and returns its answer."""
        bytestring = self._cleanstring(bytestring)
        if not bytestring.endswith(b'?\n'):
            raise ValueError("Invalid query, does not end with '?'.")
        # create socket and make query
        s = socket.create_connection((self.__ip, self.__port))
        s.send(bytestring)
        # receive answer, terminated by '\n'
        chunks = []
        bytes_recvd = 0
        while True:
            ready_to_read, _, _ = select.select([s],[],[], self.__timeout)
            if ready_to_read:
                s = ready_to_read[0]
                data = s.recv(self.__buffersize)
                if not data:
                    break # opposite side closed the connection (probably)
                if b'\n' in data: # opposite side has sent message delimiter
                    chunks.append(data)
                    break
                else:
                    chunks.append(data)
        s.close()
        answer = b''.join(chunks)
        return answer


class TektronixAWG(AWG):
    def __init__(self, ip: str, port: int, samplerate: float, first_index=None):
        self.__programs = {} # holds names and programs
        self.__program_indices = {} # holds programs and their first index (for jumping to it)
        self.__waveform_memory = set() #map sequence indices to waveforms and vice versa
        self.__program_waveforms = {} # maps program names to set of waveforms used by the program
        self.__ip = ip
        self.__port = port
        self.__rm = visa.ResourceManager()
        self.inst = self.__rm.open_resource('TCPIP::{0}::INSTR'.format(self.__ip, self.__port))
        self.__identifier = self.inst.query('*IDN?\n')
        self.inst.write('AWGCONTROL:RMODE SEQUENCE') # play sequence
        self.__samplerate = samplerate
        self.__scale = 2.
        self.__offset = 1.
        self.__channel_template = '{0:s}_{1:d}'
        self.__channels = [1] # use 2 channels. TODO: fix this
        if not first_index:
            # query how long the sequence already is
            sequence_length_before = self.inst.query('SEQUENCE:LENGTH?')
            # add 400 (?, arbitrary) more slots
            self.inst.query('SEQUENCE:LENGTH', sequence_length_before + 400) # magic number
            self.__first_index = sequence_length_before
        else:
            self.__first_index = first_index
        self.__current_index = self.__first_index + 1
        # TODO: load dummy pulse to first_index



    @property
    def outputRange(self):
        return (-1, 1)

    @property
    def identifier(self):
        return self.__identifier

    @property
    def programs(self):
        return list(self.__programs.keys())

    @property
    def samplerate(self):
        return self.__samplerate

    def rescale(self, voltages: np.ndarray) -> np.ndarray:
        """Converts an array of voltages to an array of unsigned integers for upload."""
        data = (voltages + self.__offset) / self.__scale + 1
        # scale data to uint14 range
        data = (data * (2**13 - 1))
        data[data > 2**14 - 1] = 2**14 - 1
        data = data.astype(np.uint16)
        # data = data + marker.astype(np.uint16) * 2**14
        return data

    def waveform2name(self, waveform):
        return str(hash(waveform))

    def add_waveform(self, waveform, offset):
        """Samples a Waveform object to actual data and sends it to the AWG."""
        # check if waveform is on the AWG already
        if waveform in self.__waveform_memory:
            pass
        else:
            # first sample the waveform to get an array of data
            ts = np.arange(waveform.duration * self.__samplerate) * 1/self.samplerate
            data = waveform.sample(ts, offset)
            wf_name = self.waveform2name(waveform)

            # now create a new waveform on the awg
            total = len(data)
            for ic, c in enumerate(self.__channels):
                name = self.__channel_template.format(wf_name, c)
                self.inst.write('WLIST:WAVEFORM:NEW "{0}", {1:d}, INT'.format(name, total))
                chunksize = 65536
                data = self.rescale(data[:,ic])
                n_chunks = np.ceil(total/chunksize)
                for i, chunk in enumerate(np.array_split(data, n_chunks)):
                    header ='#{0:d}{1:d}'.format(len(str(len(chunk))), len(chunk))
                    self.inst.write_binary_values('WLIST:WAVEFORM:DATA "{0:s}", {1:d}, {2:d}, '.format(name, i*chunksize, len(chunk)), chunk, datatype='H')

            self.__waveform_memory.add(waveform)

    def build_sequence(self, name):
        '''Puts a new sequence in the AWG sequence memory'''
        length = len(self.__programs[name]) - 1 # - 1 for the stop instruction in the end
        program = self.__programs[name]
        for i in range(length):
            waveform = program[i]
            wf_name = self.waveform2name(waveform)
            # set i'th sequence element
            # TODO: for multiple channels write the following line for each channel, respectively
            for c in self.__channels:
                name = self.__channel_template.format(wf_name, c)
                self.inst.write('SEQUENCE:ELEMENT{0:d}:WAVEFORM{1:d} "{2:s}"'.format(self.__current_index + i + 1, c, name))
        # have sequence go back to index 1 after playback of index N
        self.inst.write('SEQUENCE:ELEMENT{0:d}:GOTO:STATE ON'.format(length))


    def upload(self, name, program, force=False):
        '''Uploads all necessary waveforms for the program to the AWG and create a corresponding sequence.'''
        if name in self.programs:
            if not force:
                raise ProgramOverwriteException(name)
            else:
                self.remove(name)
                self.upload(name, program)
        else:
            self.__programs[name] = program
            exec_blocks = list(filter(lambda x: type(x) == EXECInstruction, program))
            offset = 0
            for block in exec_blocks:
                self.add_waveform(block.waveform, offset)
                offset += block.waveform.duration
            used_waveforms = frozenset([block.waveform for block in exec_blocks])
            self.__program_waveforms[name] = used_waveforms
            self.__program_indices[name] = self.__current_index
            self.__current_index = self.__current_index + len(program)

            self.build_sequence(name)

    def run(self, name, autorun=False):
        # there is only one sequence per devicee, so program the sequence first
        self.inst.write('SEQUENCE:ELEMENT1:GOTO:STATE ON')
        self.inst.write('SEQUENCE:ELEMENT1:GOTO:INDEX', self.__program_indices[name])
        self.inst.write('AWGCONTROL:RUN')


    def remove(self, name):
        if name in self.programs:
            self.__programs.pop(name)
            self.__program_waveforms.pop(name)
            self.clean()

    def clean(self):
        necessary_wfs = reduce(lambda acc, s: acc.union(s), self.__program_waveforms.values(), set())
        all_wfs = self.__waveform_memory
        delete = all_wfs - necessary_wfs
        for waveform in delete:
            wf_name = self.waveform2name(waveform)
            for c in self.__channels:
                name = self.__channel_template.format(wf_name, c)
                self.inst.write('WLIST:WAVEFORM:DELETE "{0:s}"'.format(name))
            self.__waveform_memory.remove(waveform)
        print('Deleted {0:d} waveforms'.format(len(delete)))

        # reset sequence
        self.inst.write('SEQUENCE:LENGTH', self.__first_index)
        self.inst.write('SEQUENCE:LENGTH', self.__first_index + 400)
        return len(delete)
