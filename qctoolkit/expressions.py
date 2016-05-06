from py_expression_eval import Parser
from typing import Any, Dict, Iterable, Optional

from qctoolkit.serialization import Serializable

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

    def evaluate(self, **kwargs) -> float:
        return self.__expression.evaluate(kwargs)

    def get_serialization_data(self, serializer: 'Serializer') -> Dict[str, Any]:
        return dict(type='Expression', expression=self.__string)

    @staticmethod
    def deserialize(serializer: 'Serializer', **kwargs) -> Serializable:
        return Expression(kwargs['expression'])

    @property
    def identifier(self) -> Optional[str]:
        return None
