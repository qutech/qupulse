"""
This module defines the class Expression to represent mathematical expression as well as
corresponding exception classes.
"""
from typing import Any, Dict, Iterable, Optional, Union
from numbers import Number
import sympy
from sympy.core.numbers import Number as SympyNumber
import numpy

from qctoolkit.comparable import Comparable
from qctoolkit.serialization import AnonymousSerializable, ExtendedJSONEncoder

__all__ = ["Expression", "ExpressionVariableMissingException"]


class Expression(AnonymousSerializable, Comparable):
    """A mathematical expression instantiated from a string representation."""

    def __init__(self, ex: Union[str, Number, sympy.Expr, Dict], numpy_evaluation: bool=True) -> None:
        """Create an Expression object.

        Receives the mathematical expression which shall be represented by the object as a string
        which will be parsed using py_expression_eval. For available operators, functions and
        constants see SymPy documentation

        Args:
            ex (string): The mathematical expression represented as a string
            numpy_evaluation: If True the evaluation will be done via sympy.lambdify with the numpy argument
        """
        super().__init__()
        if isinstance(ex, dict):
            ex, numpy_evaluation = ex['expression'], ex['numpy_evaluation']

        self._original_expression = str(ex) if isinstance(ex, sympy.Expr) else ex
        self._sympified_expression = sympy.sympify(ex)
        self._variables = tuple(str(var) for var in self._sympified_expression.free_symbols)
        if numpy_evaluation:
            self._expression_lambda = sympy.lambdify(self._variables,
                                                     self._sympified_expression, 'numpy')
        else:
            self._expression_lambda = None

    def __str__(self) -> str:
        return str(self._sympified_expression)

    def __repr__(self) -> str:
        if self.numpy_evaluation:
            return 'Expression({})'.format(repr(self._original_expression))
        else:
            return 'Expression({}, numpy_evaluation=False)'.format(repr(self._original_expression))

    def get_most_simple_representation(self) -> Union[str, int, float, complex, 'Expression']:
        if self.numpy_evaluation is False:
            return self
        elif self._sympified_expression.free_symbols:
            return str(self._sympified_expression)
        elif self._sympified_expression.is_integer:
            return int(self._sympified_expression)
        elif self._sympified_expression.is_real:
            return float(self._sympified_expression)
        elif self._sympified_expression.is_complex:
            return complex(self._sympified_expression)
        else:
            return self._original_expression  # pragma: no cover

    @staticmethod
    def all_numpy_evaluated(*args) -> bool:
        return all(expr._expression_lambda if isinstance(expr, Expression) else True for expr in args)

    @staticmethod
    def _sympify(other: Union['Expression', Number, sympy.Expr]) -> sympy.Expr:
        return other._sympified_expression if isinstance(other, Expression) else sympy.sympify(other)

    def __lt__(self, other: Union['Expression', Number, sympy.Expr]) -> Union[bool, None]:
        result = self._sympified_expression < Expression._sympify(other)
        return None if isinstance(result, sympy.Rel) else bool(result)

    def __gt__(self, other: Union['Expression', Number, sympy.Expr]) -> Union[bool, None]:
        result = self._sympified_expression > Expression._sympify(other)
        return None if isinstance(result, sympy.Rel) else bool(result)

    def __ge__(self, other: Union['Expression', Number, sympy.Expr]) -> Union[bool, None]:
        result = self._sympified_expression >= Expression._sympify(other)
        return None if isinstance(result, sympy.Rel) else bool(result)

    def __le__(self, other: Union['Expression', Number, sympy.Expr]) -> Union[bool, None]:
        result = self._sympified_expression <= Expression._sympify(other)
        return None if isinstance(result, sympy.Rel) else bool(result)

    def __eq__(self, other: Union['Expression', Number, sympy.Expr]) -> bool:
        """Overwrite Comparable's test for equality to incorporate comparisons with Numbers"""
        return self._sympified_expression == Expression._sympify(other)

    def __add__(self, other: Union['Expression', Number, sympy.Expr]) -> 'Expression':
        return Expression(self._sympified_expression.__add__(Expression._sympify(other)),
                          numpy_evaluation=self.all_numpy_evaluated(self, other))

    def __radd__(self, other: Union['Expression', Number, sympy.Expr]) -> 'Expression':
        return Expression(Expression._sympify(other).__radd__(self._sympified_expression),
                          numpy_evaluation=self.all_numpy_evaluated(self, other))

    def __sub__(self, other: Union['Expression', Number, sympy.Expr]) -> 'Expression':
        return Expression(self._sympified_expression.__sub__(Expression._sympify(other)),
                          numpy_evaluation=self.all_numpy_evaluated(self, other))

    def __rsub__(self, other: Union['Expression', Number, sympy.Expr]) -> 'Expression':
        return Expression(self._sympified_expression.__rsub__(Expression._sympify(other)),
                          numpy_evaluation=self.all_numpy_evaluated(self, other))

    def __mul__(self, other: Union['Expression', Number, sympy.Expr]) -> 'Expression':
        return Expression(self._sympified_expression.__mul__(Expression._sympify(other)),
                          numpy_evaluation=self.all_numpy_evaluated(self, other))

    def __rmul__(self, other: Union['Expression', Number, sympy.Expr]) -> 'Expression':
        return Expression(self._sympified_expression.__rmul__(Expression._sympify(other)),
                          numpy_evaluation=self.all_numpy_evaluated(self, other))

    def __truediv__(self, other: Union['Expression', Number, sympy.Expr]) -> 'Expression':
        return Expression(self._sympified_expression.__truediv__(Expression._sympify(other)),
                          numpy_evaluation=self.all_numpy_evaluated(self, other))

    def __rtruediv__(self, other: Union['Expression', Number, sympy.Expr]) -> 'Expression':
        return Expression(self._sympified_expression.__rtruediv__(Expression._sympify(other)),
                          numpy_evaluation=self.all_numpy_evaluated(self, other))

    def __neg__(self) -> 'Expression':
        return Expression(self._sympified_expression.__neg__(),
                          numpy_evaluation=self.numpy_evaluation)

    @property
    def compare_key(self) -> sympy.Expr:
        return self._sympified_expression

    @property
    def original_expression(self) -> Union[str, Number]:
        return self._original_expression

    @property
    def numpy_evaluation(self) -> bool:
        return self._expression_lambda is not None

    @property
    def sympified_expression(self) -> sympy.Expr:
        return self._sympified_expression

    @property
    def variables(self) -> Iterable[str]:
        """ Get all free variables in the expression.

        Returns:
            A collection of all free variables occurring in the expression.
        """
        return self._variables

    def evaluate_numeric(self, **kwargs) -> Union[Number, numpy.ndarray]:
        """Evaluate the expression with the required variables passed in as kwargs.

        Args:
            <variable_name> (float): Values for the free variables of the expression as keyword
                arguments where <variable_name> stand for the name of the variable. For example,
                evaluation of the expression "2*x" could be implemented as
                Expression("2*x").evaluate(x=2.5).
        Returns:
            The numeric result of evaluating the expression with the given values for the free variables.
        Raises:
            ExpressionVariableMissingException if a value for a variable is not provided.
        """
        try:
            # drop irrelevant variables
            variables = {v: kwargs[v] for v in self.variables}
        except KeyError as key_error:
            raise ExpressionVariableMissingException(key_error.args[0], self) from key_error

        if self._expression_lambda:
            result = self._expression_lambda(**dict((v, kwargs[v]) for v in self.variables))

            allowed_types = (int, float, bool, numpy.generic)
            if isinstance(result, numpy.ndarray) and issubclass(result.dtype.type, allowed_types):
                return result
            if isinstance(result, allowed_types):
                return result
            raise NonNumericEvaluation(self, result, kwargs)

        else:
            result = self._sympified_expression.subs(variables.items()).doit()
            try:
                if not result.is_real:
                    return complex(result)
                elif result.is_integer:
                    return int(result)
                else:
                    return float(result.evalf())
            except TypeError as type_error:
                raise NonNumericEvaluation(self, result, kwargs) from type_error

    def evaluate_symbolic(self, substitutions: Dict[Any, Any]) -> 'Expression':
        """Evaluate the expression symbolically.

        Args:
            substitutions (dict): Substitutions to undertake
        Returns:

        """
        numpy_evaluation = Expression.all_numpy_evaluated(self,
                                                          *(ex for ex in substitutions if isinstance(ex, Expression)))
        substitutions = dict((k, v.sympified_expression if isinstance(v, Expression) else v)
                             for k, v in substitutions.items())
        return Expression(self._sympified_expression.subs(substitutions),
                          numpy_evaluation=numpy_evaluation)

    def get_serialization_data(self) -> Union[str, Dict]:
        if self.numpy_evaluation is True:
            return self.original_expression
        else:
            return dict(expression=self.original_expression, numpy_evaluation=self.numpy_evaluation)

    def is_nan(self) -> bool:
        return sympy.sympify('nan') == self._sympified_expression


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


class NonNumericEvaluation(Exception):
    """An exception that is raised if the result of evaluate_numeric is not a number.

    See also:
        qctoolkit.expressions.Expression.evaluate_numeric
    """

    def __init__(self, expression: Expression, non_numeric_result: Any, call_arguments: Dict):
        self.expression = expression
        self.non_numeric_result = non_numeric_result
        self.call_arguments = call_arguments

    def __str__(self) -> str:
        if isinstance(self.non_numeric_result, numpy.ndarray):
            dtype = self.non_numeric_result.dtype
        else:
            dtype = type(self.non_numeric_result)
        return "The result of evaluate_numeric is of type {} " \
               "which is not a number".format(dtype)
