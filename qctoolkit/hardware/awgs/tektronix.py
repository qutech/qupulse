from bidict import bidict
import visa
import datetime

from .awg import AWG, DummyAWG, ProgramOverwriteException, OutOfWaveformMemoryException


__all__ = ['TektronixAWG']

class TektronixAWG(AWG):
    def __init__(self, ip: str, samplerate: float):
        self.__programs = {} # holds names and programs
        self.__waveform_memory = set() #map index to waveform
        self.__program_waveforms = {} # maps program names to set of waveforms used by the program
        self.__ip = ip
        self.rm = visa.ResourceManager()
        self.inst = rm.open_resource('TCPIP::{0}::INSTR'.format(self.__ip))
        self.__samplerate = samplerate
        self.__scale = 1.
        self.__offset = 0.
        self.__channel_template = '{0:s}_{1:d}'
        self.__channels = [1] # use 2 channels. TODO: fix this

    def rescale_data(self, voltages: np.ndarray) -> np.ndarray:
        """Converts an array of voltages to an array of unsigned integers for upload."""
        data = (voltages + self.__offset) / self.__scale + 1
        # scale data to uint14 range
        data = (data * (2**13 - 1))
        data[data > 2**14 - 1] = 2**14 - 1
        data = data.astype(np.uint16)
        data = marker.astype(np.uint16) * 2**14
        return data

    def waveform2name(self, waveform):
        return str(hash(waveform))

    def add_waveform(self, waveform, offset):
        """Samples a Waveform object to actual data and sends it to the AWG."""
        # check if waveform is on the AWG already
        if waveform in self.__waveform_memory:
            pass
        else:
            start = datetime.datetime.now()
            # first sample the waveform to get an array of data
            ts = np.arange(waveform.duration/self.__samplerate)
            data = waveform.sample(ts, offset)
            wf_name = waveform2name(waveform)

            # now create a new waveform on the awg
            total = len(data)
            # TODO: also do this for multiple_channels
            for c in self.__channels:
                name = channel_template.format(wf_name, c)
                self.inst.write_ascii_values('WLIST:WAVEFORM:NEW "{0}", {1:d}, INT'.format(name, total))
                chunksize = 65536
                for i, chunk in enumerate(np.split(data, chunksize)):
                    """This needs a lot of testing!!!""" #TODO: test
                    self.inst.write_binary_values('WLIST:WAVEFORM:DATA "{0:s}", {1:d}, {2:d}'.format(wf_name, i*chunksize, len(chunk), 2*len(chunk), chunk.astype('uint8')))

            end = datetime.datetime.now()
            duration = end - start
            print('Load time for pulse {0}: {1:g}, seconds for {2:d} points.'.format(
                wf_name, duration.total_seconds(), total))
            self.__waveform_memory += set(waveform)

    def build_sequence(self, name):
        '''Puts a new sequence in the AWG sequence memory'''
        length = len(self.__programs[name])
        self.inst.write('SEQUENCE:LENGTH {0:d}'.format(length)) # create new sequence
        for i in range(length):
            wf_name = waveform2name(waveform)
            # set i'th sequence element
            # TODO: for multiple channels write the following line for each channel, respectively
            for c in __channels:
                self.inst.write('SEQUENCE:ELEMENT{0:d}:WAVEFORM{1:d} "{2:s}"'.format(i+1, c+1, wf_name))
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
            exec_blocks = filter(lambda x: type(x) == EXECInstruction, program)
            offset = 0
            for block in exec_blocks:
                self.add_waveform(block.waveform, offset)
                offset += block.waveform.duration
            used_waveforms = frozenset([block.waveform for block in exec_blocks])
            self.__program_wfs[name] = used_waveforms
            self.build_sequence(name)

    def run(self, name, autorun=False):
        # there is only one sequence per devicee, so program the sequence first
        self.build_sequence(name)
        self.inst.write('AWGCONTROL:RMODE:SEQUENCE') # play sequence
        self.inst.write('AWGCONTROL:RUN')


    def remove(self, name):
        if name in self.programs:
            self.__programs.pop(name)
            self.program_wfs.pop(name)
            self.clean()

    def clean(self):
        necessary_wfs = reduce(lambda acc, s: acc.union(s), self.__program_wfs.values(), set())
        all_wfs = self.__waveform_memory
        delete = all_wfs - necessary_wfs
        for waveform in delete:
            wf_name = self.waveform2name(waveform)
            for c in channel:
                name = self.__channel_template.format(wf_name, c)
                self.inst.write('WLIST:WAVEFORM:DELETE {0:s}'.format(wf_name))
            self.__waveform_memory.remove(wf)
        print('Deleted {0:d} waveforms'.format(len(delete)))
        return len(delete)
