"""
This module defines the class Expression to represent mathematical expression as well as
corresponding exception classes.
"""
from typing import Any, Dict, Iterable, Optional
import py_expression_eval

from qctoolkit.comparable import Comparable
from qctoolkit.serialization import Serializable, Serializer

__all__ = ["Expression", "ExpressionVariableMissingException"]


class Expression(Serializable, Comparable):
    """A mathematical expression instantiated from a string representation."""

    def __init__(self, ex: str) -> None:
        """Create an Expression object.

        Receives the mathematical expression which shall be represented by the object as a string
        which will be parsed using py_expression_eval. For available operators, functions and
        constants see
        https://github.com/AxiaCore/py-expression-eval/#available-operators-constants-and-functions.
        In addition, the ** operator may be used for exponentiation instead of the ^ operator.

        Args:
            ex (string): The mathematical expression represented as a string
        """
        super().__init__()
        self.__string = str(ex)
        self.__expression = py_expression_eval.Parser().parse(ex.replace('**', '^'))

    def __str__(self) -> str:
        return self.__string

    @property
    def compare_key(self) -> Any:
        return str(self)

    def variables(self) -> Iterable[str]:
        """ Get all free variables in the expression.

        Returns:
            A collection of all free variables occurring in the expression.
        """
        return self.__expression.variables()

    def evaluate(self, **kwargs) -> float:
        """Evaluate the expression with the required variables passed in as kwargs.

        Args:
            float <variable_name>: Values for the free variables of the expression as keyword
                arguments where <variable_name> stand for the name of the variable. For example,
                evaluation of the expression "2*x" could be implemented as
                Expresson("2*x").evaluate(x=2.5).
        Returns:
            The result of evaluating the expression with the given values for the free variables.
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
    """An exception indicating that a variable value was not provided during expression evaluation.

    See also:
         qctoolkit.expressions.Expression
    """

    def __init__(self, variable: str, expression: Expression) -> None:
        super().__init__()
        self.variable = variable
        self.expression = expression

    def __str__(self) -> str:
        return "Could not evaluate <{}>: A value for variable <{}> is missing!".format(
            str(self.expression), self.variable)
