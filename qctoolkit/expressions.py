try:
    import numexpr
    USE_NUMEXPR = True
except ImportError:
    USE_NUMEXPR = False
from py_expression_eval import Parser

from qctoolkit.serialization import Serializable


class Expression(Serializable):
    def __init__(self, ex: str):
        ex = str(ex)
        self._string = ex
        self._expression = Parser().parse(ex.replace('**', '^'))

    @property
    def string(self):
        return self._string

    def variables(self):
        return self._expression.variables()

    def evaluate(self, parameters):
        if USE_NUMEXPR:
            return numexpr.evaluate(self._string, global_dict={}, local_dict=parameters)
        else:
            return self._expression.evaluate(parameters)

    def get_serialization_data(self, serializer: 'Serializer'):
        return dict(type='Expression', expression=self._string)

    @staticmethod
    def deserialize(serializer: 'Serializer', **kwargs):
        return Expression(kwargs['expression'])

    @property
    def identifier(self):
        return None
