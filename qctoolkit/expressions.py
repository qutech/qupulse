"""
This module defines the class Expression to represent mathematical expression as well as
corresponding exception classes.
"""
from typing import Any, Dict, Union, Sequence, Callable, TypeVar, Type
from numbers import Number
import warnings
import functools
import array

import sympy
import numpy

from qctoolkit.serialization import AnonymousSerializable
from qctoolkit.utils.sympy import sympify, substitute_with_eval, to_numpy, recursive_substitution, evaluate_lambdified

__all__ = ["Expression", "ExpressionVariableMissingException", "ExpressionScalar", "ExpressionVector"]


_ExpressionType = TypeVar('_ExpressionType', bound='Expression')


class _ExpressionMeta(type):
    """Metaclass that forwards calls to Expression(...) to Expression.make(...) to make subclass objects"""
    def __call__(cls: Type[_ExpressionType], *args, **kwargs) -> _ExpressionType:
        if cls is Expression:
            return cls.make(*args, **kwargs)
        else:
            return type.__call__(cls, *args, **kwargs)


class Expression(AnonymousSerializable, metaclass=_ExpressionMeta):
    """Base class for expressions."""
    def __init__(self, *args, **kwargs):
        self._expression_lambda = None

    def _parse_evaluate_numeric_arguments(self, eval_args: Dict[str, Number]) -> Dict[str, Number]:
        try:
            return {v: eval_args[v] for v in self.variables}
        except KeyError as key_error:
            raise ExpressionVariableMissingException(key_error.args[0], self) from key_error

    def _parse_evaluate_numeric_result(self,
                                       result: Union[Number, numpy.ndarray],
                                       call_arguments: Any) -> Union[Number, numpy.ndarray]:
        allowed_types = (float, numpy.number, int, complex, bool, numpy.bool_)
        if isinstance(result, tuple):
            result = numpy.array(result)
        if isinstance(result, numpy.ndarray) and issubclass(result.dtype.type, allowed_types):
            return result
        elif isinstance(result, allowed_types):
            return result
        elif isinstance(result, sympy.Float):
            return float(result)
        elif isinstance(result, sympy.Integer):
            return int(result)
        else:
            raise NonNumericEvaluation(self, result, call_arguments)

    def evaluate_numeric(self, **kwargs) -> Union[Number, numpy.ndarray]:
        parsed_kwargs = self._parse_evaluate_numeric_arguments(kwargs)

        result, self._expression_lambda = evaluate_lambdified(self.underlying_expression, self.variables,
                                                              parsed_kwargs, lambdified=self._expression_lambda)

        return self._parse_evaluate_numeric_result(result, kwargs)

    def evaluate_symbolic(self, substitutions: Dict[Any, Any]) -> 'Expression':
        return Expression.make(recursive_substitution(sympify(self.underlying_expression), substitutions))

    @property
    def variables(self) -> Sequence[str]:
        """ Get all free variables in the expression.

        Returns:
            A collection of all free variables occurring in the expression.
        """
        raise NotImplementedError()

    @classmethod
    def make(cls: Type[_ExpressionType],
             expression_or_dict,
             numpy_evaluation=None) -> Union['ExpressionScalar', 'ExpressionVector', _ExpressionType]:
        """Backward compatible expression generation"""
        if numpy_evaluation is not None:
            warnings.warn('numpy_evaluation keyword argument is deprecated and ignored.')

        if isinstance(expression_or_dict, dict):
            expression = expression_or_dict['expression']
        elif isinstance(expression_or_dict, cls):
            return expression_or_dict
        else:
            expression = expression_or_dict

        if cls is Expression:
            if isinstance(expression, (list, tuple, numpy.ndarray, sympy.NDimArray, array.array)):
                return ExpressionVector(expression)
            else:
                return ExpressionScalar(expression)
        else:
            return cls(expression)

    @property
    def underlying_expression(self) -> Union[sympy.Expr, numpy.ndarray]:
        raise NotImplementedError()


class ExpressionVector(Expression):
    """N-dimensional expression.
    TODO: write doc!
    TODO: write tests!
    """
    sympify_vector = numpy.vectorize(sympify)

    def __init__(self, expression_vector: Sequence):
        super().__init__()
        if isinstance(expression_vector, sympy.NDimArray):
            expression_vector = to_numpy(expression_vector)
        self._expression_vector = self.sympify_vector(expression_vector)
        variables = {str(x)
                     for expr in self._expression_vector.ravel()
                     for x in expr.free_symbols}
        self._variables = tuple(variables)

    @property
    def expression_lambda(self) -> Callable:
        if self._expression_lambda is None:
            expression_lambda = sympy.lambdify(self.variables, self.underlying_expression,
                                                     [{'ceiling': ceiling}, 'numpy'])

            @functools.wraps(expression_lambda)
            def expression_wrapper(*args, **kwargs):
                result = expression_lambda(*args, **kwargs)
                if isinstance(result, sympy.NDimArray):
                    return numpy.array(result.tolist())
                elif isinstance(result, list):
                    return numpy.array(result).reshape(self.underlying_expression.shape)
                else:
                    return result.reshape(self.underlying_expression.shape)
            self._expression_lambda = expression_wrapper
        return self._expression_lambda

    @property
    def variables(self) -> Sequence[str]:
        return self._variables

    def evaluate_numeric(self, **kwargs) -> Union[numpy.ndarray, Number]:
        parsed_kwargs = self._parse_evaluate_numeric_arguments(kwargs)

        result, self._expression_lambda = evaluate_lambdified(self.underlying_expression, self.variables,
                                                              parsed_kwargs, lambdified=self._expression_lambda)

        if isinstance(result, (list, tuple)):
            result = numpy.array(result)

        return self._parse_evaluate_numeric_result(numpy.array(result), kwargs)

    def get_serialization_data(self) -> Sequence[str]:
        return numpy.vectorize(str)(self._expression_vector).tolist()

    def __str__(self):
        return str(self.get_serialization_data())

    def __repr__(self):
        return 'ExpressionVector({})'.format(repr(self.get_serialization_data()))

    @property
    def underlying_expression(self) -> numpy.ndarray:
        return self._expression_vector


class ExpressionScalar(Expression):
    """A scalar mathematical expression instantiated from a string representation.
        TODO: update doc!
        TODO: write tests!
        """

    def __init__(self, ex: Union[str, Number, sympy.Expr]) -> None:
        """Create an Expression object.

        Receives the mathematical expression which shall be represented by the object as a string
        which will be parsed using py_expression_eval. For available operators, functions and
        constants see SymPy documentation

        Args:
            ex (string): The mathematical expression represented as a string
        """
        super().__init__()

        if isinstance(ex, sympy.Expr):
            self._original_expression = str(ex)
            self._sympified_expression = ex
        else:
            self._original_expression = ex
            self._sympified_expression = sympify(ex)

        self._variables = tuple(str(var) for var in self._sympified_expression.free_symbols)

    @property
    def underlying_expression(self) -> sympy.Expr:
        return self._sympified_expression

    def __str__(self) -> str:
        return str(self._sympified_expression)

    def __repr__(self) -> str:
        return 'Expression({})'.format(repr(self._original_expression))

    @property
    def variables(self) -> Sequence[str]:
        return self._variables

    def get_most_simple_representation(self) -> Union[str, int, float, complex]:
        if self._sympified_expression.free_symbols:
            return str(self._sympified_expression)
        elif self._sympified_expression.is_integer:
            return int(self._sympified_expression)
        elif self._sympified_expression.is_real:
            return float(self._sympified_expression)
        elif self._sympified_expression.is_complex:
            return complex(self._sympified_expression)
        else:
            return self._original_expression  # pragma: no cover

    @classmethod
    def _sympify(cls, other: Union['ExpressionScalar', Number, sympy.Expr]) -> sympy.Expr:
        return other._sympified_expression if isinstance(other, cls) else sympify(other)

    def __lt__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> Union[bool, None]:
        result = self._sympified_expression < self._sympify(other)
        return None if isinstance(result, sympy.Rel) else bool(result)

    def __gt__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> Union[bool, None]:
        result = self._sympified_expression > self._sympify(other)
        return None if isinstance(result, sympy.Rel) else bool(result)

    def __ge__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> Union[bool, None]:
        result = self._sympified_expression >= self._sympify(other)
        return None if isinstance(result, sympy.Rel) else bool(result)

    def __le__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> Union[bool, None]:
        result = self._sympified_expression <= self._sympify(other)
        return None if isinstance(result, sympy.Rel) else bool(result)

    def __eq__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> bool:
        """Enable comparisons with Numbers"""
        return self._sympified_expression == self._sympify(other)

    def __add__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__add__(self._sympify(other)))

    def __radd__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympify(other).__radd__(self._sympified_expression))

    def __sub__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__sub__(self._sympify(other)))

    def __rsub__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__rsub__(self._sympify(other)))

    def __mul__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__mul__(self._sympify(other)))

    def __rmul__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__rmul__(self._sympify(other)))

    def __truediv__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__truediv__(self._sympify(other)))

    def __rtruediv__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__rtruediv__(self._sympify(other)))

    def __neg__(self) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__neg__())

    @property
    def original_expression(self) -> Union[str, Number]:
        return self._original_expression

    @property
    def sympified_expression(self) -> sympy.Expr:
        return self._sympified_expression

    def get_serialization_data(self) -> Union[str, Dict]:
        return self.original_expression

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
