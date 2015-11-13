try:
    import numexpr
    USE_NUMEXPR = True
except ImportError:
    USE_NUMEXPR = False
from py_expression_eval import Parser
from typing import Any, Dict, Iterable, Optional

from qctoolkit.serialization import Serializable
from qctoolkit.pulses.parameters import Parameter

__all__ = ["Expression"]


class Expression(Serializable):

    def __init__(self, ex: str) -> None:
        self.__string = str(ex) # type: str
        self.__expression = Parser().parse(ex.replace('**', '^')) # type: py_expression_eval.Expression

    @property
    def string(self) -> str:
        return self.__string

    def variables(self) -> Iterable[str]:
        return self.__expression.variables()

    def evaluate(self, **kwargs):
        if USE_NUMEXPR:
            return numexpr.evaluate(self.__string, global_dict={}, local_dict=kwargs)
        else:
            return self.__expression.evaluate(kwargs)

    def get_serialization_data(self, serializer: 'Serializer') -> Dict[str, Any]:
        return dict(type='Expression', expression=self.__string)

    @staticmethod
    def deserialize(serializer: 'Serializer', **kwargs) -> Serializable:
        return Expression(kwargs['expression'])

    @property
    def identifier(self) -> Optional[str]:
        return None
