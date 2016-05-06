import py_expression_eval
from typing import Any, Dict, Iterable, Optional

from qctoolkit.comparable import Comparable
from qctoolkit.serialization import Serializable, Serializer

__all__ = ["Expression"]


class Expression(Serializable, Comparable):
    """A mathematical expression."""

    def __init__(self, ex: str) -> None:
        """Creates an Expression object.

        Receives the mathematical expression which shall be represented by the object as a string which will be parsed
        using py_expression_eval. For available operators, functions and constants see
        https://github.com/AxiaCore/py-expression-eval/#available-operators-constants-and-functions . In addition,
        the ** operator may be used for exponentiation instead of the ^ operator.

        Args:
            ex (string): The mathematical expression represented as a string
        """
        self.__string = str(ex) # type: str
        self.__expression = py_expression_eval.Parser().parse(ex.replace('**', '^')) # type: py_expression_eval.Expression

    def __str__(self) -> str:
        """Returns a string representation of this expression."""
        return self.__string

    @property
    def _compare_key(self) -> Any:
        """Returns the string representation of this expression as unique key used in comparison and hashing operations."""
        return str(self)

    def variables(self) -> Iterable[str]:
        """Returns all variables occurring in the expression."""
        return self.__expression.variables()

    def evaluate(self, **kwargs) -> float:
        """Evaluates the expression with the required variables passed in as kwargs.

        Keyword Args:
            <variable name> float: Values for the free variables of the expression.
        Raises:
            ExpressionVariableMissingException if a value for a variable is not provided.
        """
        try:
            return self.__expression.evaluate(kwargs)
        except Exception as e:
            raise ExpressionVariableMissingException(str(e).split(' ')[2], self) from e

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        return dict(type=serializer.get_type_identifier(self), expression=str(self))

    @staticmethod
    def deserialize(serializer: 'Serializer', **kwargs) -> Serializable:
        return Expression(kwargs['expression'])

    @property
    def identifier(self) -> Optional[str]:
        return None


class ExpressionVariableMissingException(Exception):

    def __init__(self, variable: str, expression: Expression) -> None:
        super().__init__()
        self.__variable = variable
        self.__expression = expression

    @property
    def expression(self) -> Expression:
        return self.__expression

    @property
    def variable(self) -> str:
        return self.__variable

    def __str__(self) -> str:
        return "Could not evaluate <{}>: A value for variable <{}> is missing!".format(str(self.expression), self.variable)
