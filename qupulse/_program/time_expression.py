from typing import Union

import numpy


ProgramExpression = Union[float, 'TimeDependentValue']


class TimeDependentValue:
    __slots__ = ('_constant', '_code', '_str')

    def __init__(self, value):
        if isinstance(value, (float, int)):
            self._constant = value
        else:
            self._constant = None

        if isinstance(value, str):
            self._str = value
            self._code = compile(value, filename='<string>', mode='eval')
        else:
            self._str = None
            self._code = None

    def __float__(self):
        if self._constant is not None:
            return float(self._constant)
        else:
            return float(eval(self._code, numpy.__dict__))

    def at_sample_times(self, times):
        """Returns broadcastable to time.shape"""
        if self._constant is not None:
            return self._constant
        else:
            return eval(self._code, {'t': times}, numpy.__dict__)

    def __str__(self):
        return self._str or str(self._constant)

    def __repr__(self):
        return f'{type(self)}({str(self)})'

    def __getstate__(self):
        return self._str or self._constant,

    def __setstate__(self, state):
        state, = state
        self.__init__(state)


def at_sample_times(tdv: ProgramExpression, times):
    if hasattr(tdv, 'at_sample_times'):
        return tdv.at_sample_times(times)
    else:
        return tdv


def to_program_expression():
    pass