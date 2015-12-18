from typing import NamedTuple, Dict, Union
from copy import deepcopy

from .instructions import Waveform
from .parameters import Parameter, ParameterDeclaration

SleepWindow = NamedTuple("SleepWindow", [('start', float), ('end', float)])

class Measurement:
    def __init__(self, waveform: Waveform, offset: Union[float, ParameterDeclaration] = 0):
        self.__offset = offset
        self.__windows = []
        self.__waveform = waveform

    def measure(self, start: Union[float, ParameterDeclaration], end: Union[float, ParameterDeclaration]):
        if not isinstance(end, ParameterDeclaration) and not isinstance(start, ParameterDeclaration):
            if end < start:
                raise ValueError("Measure: Start has to be smaller than end!")
        offset = self.__get_end_offset()
        if start != 0:
            self.sleep(start)
        self.__windows.append(
                Window(start=start, end=end, parent = self, offset = offset))
        return deepcopy(self)

    def measure_list(self, list_, end, start = 0):
        offset = self.__get_end_offset()
        for i in list_:
            i.offset = self.__get_end_offset()
            self.__windows.append(i)
        return deepcopy(self)

    def __get_end_offset(self):
        if len(self.__windows) == 0:
            return 0
        else:
            return self.__windows[-1].end

    def sleep(self, duration: float):
        self.__windows.append(Window(start=0,
                                     end = duration,
                                     parent = self,
                                     is_measure=False,
                                     offset = self.__get_end_offset()))

    @property
    def duration(self):
        return self.__waveform.duration

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
            if i.is_measure:
                yield (i.start, i.end)

    def __repr__(self):
        res = "Measurement: "
        res += ",".join(list(self))

    def build(self):
        list_ = []
        for i in self.__windows:
            if i.is_measure:
                list_.append((i.start,i.end))
        return list_

    def instantiate(self, parameters: Dict[str, Parameter]):
        if isinstance(self.__offset, ParameterDeclaration):
            offset = self.__offset.get_value(parameters)
        else:
            offset = self.__offset
        res = Measurement(self.__waveform, offset)
        for i in self.__windows:
            if i.is_measure:
                instance_ = i.instantiate(parameters, res)
                res.__windows.append(instance_)
        return res

class Window:
    def __init__(self, start: Union[float,ParameterDeclaration],
                 end: Union[float,ParameterDeclaration],
                 parent: Measurement,
                 is_measure: bool = True,
                 offset: Union[float,ParameterDeclaration] = 0):
        self.__start = start
        self.__end = end
        self.__offset = offset
        self.__parent = parent
        self.__is_measure = is_measure

    @property
    def start(self):
        if isinstance(self.__start, ParameterDeclaration):
            raise ValueError("Can't calculate with ParameterDeclaration: start")
        elif isinstance(self.__offset, ParameterDeclaration):
            raise ValueError("Can't calculate with ParameterDeclaration: offset")
        elif isinstance(self.__parent.offset, ParameterDeclaration):
            raise ValueError("Can't calculate with ParameterDeclaration: parent.offset")
        else:
            return self.__start + self.__offset + self.__parent.offset

    @property
    def end(self):
        if isinstance(self.__end,ParameterDeclaration):
            raise ValueError("Can't calculate with ParameterDeclaration: end")
        elif isinstance(self.__offset,ParameterDeclaration):
            raise ValueError("Can't calculate with ParameterDeclaration: offset")
        elif isinstance(self.__parent.offset, ParameterDeclaration):
            raise ValueError("Can't calculate with ParameterDeclaration: parent.offset")
        else:
            return self.__end + self.__offset + self.__parent.offset

    @property
    def offset(self):
        return self.__offset

    @property
    def parent(self):
        return self.__parent

    @property
    def is_measure(self):
        return self.__is_measure

    def instantiate(self, parameters: Dict[str, Parameter], parent: Measurement):
        args = {}
        if isinstance(self.__start, ParameterDeclaration):
            args["start"] = self.__start.get_value(parameters)
        else:
            args["start"] = self.__start

        if isinstance(self.__end, ParameterDeclaration):
            args["end"] = self.__end.get_value(parameters)
        else:
            args["end"] = self.__end

        if isinstance(self.__offset, ParameterDeclaration):
            args["offset"] = self.__offset.get_value(parameters)
        else:
            args["offset"] = self.__offset
        args["parent"] = parent
        args["is_measure"] = self.__is_measure
        return Window(**args)
