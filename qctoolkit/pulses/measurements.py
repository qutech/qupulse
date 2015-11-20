from typing import NamedTuple

from .instructions import Waveform
from .parameters import Parameter, ParameterDeclaration

SleepWindow = NamedTuple("SleepWindow", [('start', float), ('end', float)])

class Measurement:
    def __init__(self, waveform: Waveform, offset: float = 0):
        self.__offset = offset
        self.__windows = []
        self.__waveform = waveform

    def measure(self, end: float, start: float=0):
        # Standard usage
        if isinstance(end, Window):
            end.offset = self.__get_end_offset()
            end.parent = self
            self.__windows.append(end)
        else:
            self.__windows.append(
                Window(start = start, end = end, parent = self, offset = self.__get_end_offset()))
        return self

    def measure_list(self, list_, end, start = 0):
        offset = self.__get_end_offset()
        for i in list_:
            i.offset = self.__get_end_offset()
            self.__windows.append(i)
        return self

    def __get_end_offset(self):
        if len(self.__windows) == 0:
            return 0
        else:
            return self.__windows[-1].end

    def sleep(self, end: float, start: float = 0):
        pass

    @property
    def duration(self):
        self.__waveform.duration

    @property
    def offset(self):
        return self.__offset

    @offset.setter
    def offset(self, value):
        self.__offset = value

    def __len__(self):
        return len(self.__windows)

    def __iter__(self):
        for i in self.__windows:
            yield (i.start, i.end)

    def __repr__(self):
        res = "Measurement: "
        as_list = []
        for i in self:
            as_list.append("({},{})".format(i.start, i.end))
        res += ",".join(as_list)

    def build(self, parameters: Parameter):
        list_ = []
        for i in self.__windows:
            tuple = 0
            if i.is_measure:
                tuple = ()
                if isinstance(i.start, ParameterDeclaration):
                    tuple[0] = i.start.get_value(parameters)
                else:
                    tuple[0] = i.start
                if isinstance(i.end, ParameterDeclaration):
                    tuple[0] = i.end.get_value(parameters)
                else:
                    tuple[0] = i.end
                list_.append(tuple)
        return list_


class Window:
    def __init__(self, start: float, end: float, parent: Measurement, is_measure: bool = True, offset: float = 0):
        self.__start = start
        self.__end = end
        self.__offset = offset
        self.__parent = parent
        self.__is_measure = is_measure

    @property
    def start(self):
        return self.__start + self.__offset + self.__parent.offset

    @property
    def end(self):
        return self.__end + self.__offset + self.__parent.offset

    @property
    def offset(self):
        return self.__offset

    @offset.setter
    def offset(self, value):
        self.__offset = value

    @property
    def parent(self):
        return self.__parent

    @parent.setter
    def parent(self, value):
        self.__parent = value

    @property
    def is_measure(self):
        return self.is_measure
