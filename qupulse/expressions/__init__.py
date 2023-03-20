"""This subpackage contains qupulse's expression logic. The submodule :py:`protocol` defines the :py:`typing.Protocol`
that expression functionality providers must implement. This allows to substitute the powerful and expressive but slow
default implementation with a faster less expressive backend.
"""

from typing import Type, TypeVar
from numbers import Real

import numpy as np
import sympy

from . import legacy, protocol, wrapper


__all__ = ["Expression", "ExpressionVector", "ExpressionScalar",
           "NonNumericEvaluation", "ExpressionVariableMissingException"]


Expression: Type[protocol.Expression] = legacy.Expression
ExpressionScalar: Type[protocol.ExpressionScalar] = legacy.ExpressionScalar
ExpressionVector: Type[protocol.ExpressionVector] = legacy.ExpressionVector


Expression, ExpressionScalar, ExpressionVector = wrapper.make_wrappers(legacy.Expression,
                                                                       legacy.ExpressionScalar,
                                                                       legacy.ExpressionVector)


ExpressionLike = TypeVar('ExpressionLike', str, Real, sympy.Expr, ExpressionScalar)


class ExpressionVariableMissingException(Exception):
    """An exception indicating that a variable value was not provided during expression evaluation.

    See also:
         qupulse.expressions.Expression
    """

    def __init__(self, variable: str, expression: Expression) -> None:
        super().__init__()
        self.variable = variable
        self.expression = expression

    def __str__(self) -> str:
        return f"Could not evaluate <{self.expression}>: A value for variable <{self.variable}> is missing!"


class NonNumericEvaluation(Exception):
    """An exception that is raised if the result of evaluate_numeric is not a number.

    See also:
        qupulse.expressions.Expression.evaluate_numeric
    """

    def __init__(self, expression: Expression, non_numeric_result, call_arguments):
        self.expression = expression
        self.non_numeric_result = non_numeric_result
        self.call_arguments = call_arguments

    def __str__(self) -> str:
        if isinstance(self.non_numeric_result, np.ndarray):
            dtype = self.non_numeric_result.dtype

            if dtype == np.dtype('O'):
                dtypes = set(map(type, self.non_numeric_result.flat))
                return f"The result of evaluate_numeric is an array with the types {dtypes} which is not purely numeric"
        else:
            dtype = type(self.non_numeric_result)
        return f"The result of evaluate_numeric is of type {dtype} which is not a number"
