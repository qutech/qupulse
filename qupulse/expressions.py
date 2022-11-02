"""
This module defines the class Expression to represent mathematical expression as well as
corresponding exception classes.
"""
import operator
from typing import Any, Dict, Union, Sequence, Callable, TypeVar, Type, Mapping
from numbers import Number
import warnings
import functools
import array
import itertools

import sympy
import numpy

from qupulse.serialization import AnonymousSerializable
from qupulse.utils.sympy import sympify, to_numpy, recursive_substitution, evaluate_lambdified,\
    get_most_simple_representation, get_variables, evaluate_lamdified_exact_rational
from qupulse.utils.types import TimeType

__all__ = ["Expression", "ExpressionVariableMissingException", "ExpressionScalar", "ExpressionVector", "ExpressionLike"]


_ExpressionType = TypeVar('_ExpressionType', bound='Expression')


ALLOWED_NUMERIC_SCALAR_TYPES = (float, numpy.number, int, complex, bool, numpy.bool_, TimeType)


def _parse_evaluate_numeric(result) -> Union[Number, numpy.ndarray]:
    """Tries to parse the result as a scalar if possible. Falls back to an array otherwise.
    Raises:
        ValueError if scalar result is not parsable
    """
    allowed_scalar = ALLOWED_NUMERIC_SCALAR_TYPES

    if isinstance(result, allowed_scalar):
        # fast path for regular evaluations
        return result
    if isinstance(result, tuple):
        result, = result
    elif isinstance(result, numpy.ndarray):
        result = result[()]

    if isinstance(result, allowed_scalar):
        return result
    if isinstance(result, sympy.Float):
        return float(result)
    elif isinstance(result, sympy.Integer):
        return int(result)

    if isinstance(result, numpy.ndarray):
        # allow numeric vector values
        return _parse_evaluate_numeric_vector(result)
    raise ValueError("Non numeric result", result)


def _parse_evaluate_numeric_vector(vector_result: numpy.ndarray) -> numpy.ndarray:
    allowed_scalar = ALLOWED_NUMERIC_SCALAR_TYPES
    if not issubclass(vector_result.dtype.type, allowed_scalar):
        obj_types = set(map(type, vector_result.flat))
        if all(issubclass(obj_type, sympy.Integer) for obj_type in obj_types):
            result = vector_result.astype(numpy.int64)
        elif all(issubclass(obj_type, (sympy.Integer, sympy.Float)) for obj_type in obj_types):
            result = vector_result.astype(float)
        else:
            raise ValueError("Could not parse vector result", vector_result)
    return vector_result


def _flat_iter(arr):
    if len(arr.shape) > 1:
        for sub_arr in arr:
            yield from _flat_iter(sub_arr)
    else:
        yield from arr


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

    def _parse_evaluate_numeric_arguments(self, eval_args: Mapping[str, Number]) -> Dict[str, Number]:
        try:
            return {v: eval_args[v] for v in self.variables}
        except KeyError as key_error:
            if type(key_error).__module__.startswith('qupulse'):
                # we forward qupulse errors, I down like this
                raise
            else:
                raise ExpressionVariableMissingException(key_error.args[0], self) from key_error

    def evaluate_in_scope(self, scope: Mapping) -> Union[Number, numpy.ndarray]:
        """Evaluate the expression by taking the variables from the given scope (typically of type Scope but it can be
        any mapping.)
        Args:
            scope:

        Returns:

        """
        raise NotImplementedError("")

    def evaluate_numeric(self, **kwargs) -> Union[Number, numpy.ndarray]:
        return self.evaluate_in_scope(kwargs)

    def __float__(self):
        if self.variables:
            return NotImplemented
        else:
            e = self.evaluate_numeric()
            return float(e)

    def evaluate_symbolic(self, substitutions: Mapping[Any, Any]) -> 'Expression':
        if len(substitutions)==0:
            return self
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
            expression_shape = expression_vector.shape
            expression_items = tuple(_flat_iter(expression_vector))
        else:
            expression_ndarray = self.sympify_vector(expression_vector)
            expression_items = tuple(expression_ndarray.flat)
            expression_shape = expression_ndarray.shape

        self._expression_items = expression_items
        self._expression_shape = expression_shape

        self._lambdified_items = [None] * len(self._expression_items)

        variables = set(itertools.chain.from_iterable(map(get_variables, self._expression_items)))
        self._variables = tuple(sorted(variables))

    @property
    def variables(self) -> Sequence[str]:
        return self._variables

    def evaluate_in_scope(self, scope: Mapping) -> numpy.ndarray:
        parsed_kwargs = self._parse_evaluate_numeric_arguments(scope)
        flat_result = []
        for idx, expr in enumerate(self._expression_items):
            result, self._lambdified_items[idx] = evaluate_lambdified(expr, self.variables, parsed_kwargs,
                                                                      lambdified=self._lambdified_items[idx])
            flat_result.append(result)
        result = numpy.array(flat_result).reshape(self._expression_shape)
        try:
            return _parse_evaluate_numeric_vector(result)
        except ValueError as err:
            raise NonNumericEvaluation(self, result, scope) from err

    def get_serialization_data(self) -> Sequence[str]:
        serialized_items = list(map(get_most_simple_representation, self._expression_items))
        if len(self._expression_shape) == 0:
            return serialized_items[0]
        elif len(self._expression_shape) == 1:
            return serialized_items
        else:
            return numpy.array(serialized_items).reshape(self._expression_shape).tolist()

    def __str__(self):
        return str(self.get_serialization_data())

    def __repr__(self):
        return f'ExpressionVector({self.get_serialization_data()!r})'

    def _sympy_(self):
        return sympy.NDimArray(self.to_ndarray())

    def __eq__(self, other):
        if not isinstance(other, Expression):
            try:
                other = Expression.make(other)
            except (ValueError, TypeError):
                return NotImplemented
        if isinstance(other, ExpressionScalar):
            return self._expression_shape in ((), (1,)) and self._expression_items[0] == other.sympified_expression
        else:
            return self._expression_shape == other._expression_shape and \
                   self._expression_items == other._expression_items

    def __hash__(self):
        if self._expression_shape in ((), (1,)):
            return hash(self._expression_items[0])
        else:
            return hash((self._expression_items, self._expression_shape))

    def __getitem__(self, item) -> Expression:
        if len(self._expression_shape) == 0:
            assert item == ()
            expr, = self._expression_items
            return ExpressionScalar(expr)
        if len(self._expression_shape) == 1:
            return ExpressionScalar(self._expression_items[item])
        else:
            return ExpressionVector(self.to_ndarray()[item])

    def to_ndarray(self) -> numpy.ndarray:
        return numpy.array(self._expression_items).reshape(self._expression_shape)

    @property
    def underlying_expression(self) -> numpy.ndarray:
        return self.to_ndarray()


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
            self._original_expression = None
            self._sympified_expression = ex
            self._variables = get_variables(self._sympified_expression)
        elif isinstance(ex, ExpressionScalar):
            self._original_expression = ex._original_expression
            self._sympified_expression = ex._sympified_expression            
            self._variables = ex._variables
        elif isinstance(ex, (int, float)):
            if isinstance(ex, numpy.float64):
                ex = float(ex)
            self._original_expression = ex
            self._sympified_expression = sympify(ex)
            self._variables = ()
        else:
            self._original_expression = ex
            self._sympified_expression = sympify(ex)
            self._variables = get_variables(self._sympified_expression)

        self._exact_rational_lambdified = None

    def __float__(self):
        if isinstance(self._original_expression, float):
            return self._original_expression
        else:
            return super().__float__()

    @property
    def underlying_expression(self) -> sympy.Expr:
        return self._sympified_expression

    def __str__(self) -> str:
        return str(self._sympified_expression)

    def __repr__(self) -> str:
        if self._original_expression is None:
            return f"ExpressionScalar('{self._sympified_expression!r}')"
        else:
            return f"ExpressionScalar({self._original_expression!r})"

    def __format__(self, format_spec):
        if format_spec == '':
            return str(self)
        return format(float(self), format_spec)
    
    @property
    def variables(self) -> Sequence[str]:
        return self._variables

    @classmethod
    def _sympify(cls, other: Union['ExpressionScalar', Number, sympy.Expr]) -> sympy.Expr:
        return other._sympified_expression if isinstance(other, cls) else sympify(other)

    @classmethod
    def _extract_sympified(cls, other: Union['ExpressionScalar', Number, sympy.Expr]) \
                            -> Union['ExpressionScalar', Number, sympy.Expr]:
        return getattr(other, '_sympified_expression', other)

    def __lt__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> Union[bool, None]:
        result = self._sympified_expression < self._extract_sympified(other)
        return None if isinstance(result, sympy.Rel) else bool(result)

    def __gt__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> Union[bool, None]:
        result = self._sympified_expression > self._extract_sympified(other)
        return None if isinstance(result, sympy.Rel) else bool(result)

    def __ge__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> Union[bool, None]:
        result = self._sympified_expression >= self._extract_sympified(other)
        return None if isinstance(result, sympy.Rel) else bool(result)

    def __le__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> Union[bool, None]:
        result = self._sympified_expression <= self._extract_sympified(other)
        return None if isinstance(result, sympy.Rel) else bool(result)

    def __eq__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> bool:
        """Enable comparisons with Numbers"""
        # sympy's __eq__ checks for structural equality to be consistent regarding __hash__ so we do that too
        # see https://github.com/sympy/sympy/issues/18054#issuecomment-566198899
        return self._sympified_expression == self._sympify(other)

    def __hash__(self) -> int:
        return hash(self._sympified_expression)

    def __add__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__add__(self._extract_sympified(other)))

    def __radd__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympify(other).__radd__(self._sympified_expression))

    def __sub__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__sub__(self._extract_sympified(other)))

    def __rsub__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__rsub__(self._extract_sympified(other)))

    def __mul__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__mul__(self._extract_sympified(other)))

    def __rmul__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__rmul__(self._extract_sympified(other)))

    def __truediv__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__truediv__(self._extract_sympified(other)))

    def __rtruediv__(self, other: Union['ExpressionScalar', Number, sympy.Expr]) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__rtruediv__(self._extract_sympified(other)))

    def __neg__(self) -> 'ExpressionScalar':
        return self.make(self._sympified_expression.__neg__())

    def __pos__(self):
        return self.make(self._sympified_expression.__pos__())

    def _sympy_(self):
        return self._sympified_expression

    @property
    def original_expression(self) -> Union[str, Number]:
        if self._original_expression is None:
            return str(self._sympified_expression)
        else:
            return self._original_expression

    @property
    def sympified_expression(self) -> sympy.Expr:
        return self._sympified_expression

    def get_serialization_data(self) -> Union[str, float, int]:
        serialized = get_most_simple_representation(self._sympified_expression)
        if isinstance(serialized, str):
            return self.original_expression
        else:
            return serialized

    def is_nan(self) -> bool:
        return sympy.sympify('nan') == self._sympified_expression

    def evaluate_with_exact_rationals(self, scope: Mapping) -> Union[Number, numpy.ndarray]:
        parsed_kwargs = self._parse_evaluate_numeric_arguments(scope)
        result, self._exact_rational_lambdified = evaluate_lamdified_exact_rational(self.sympified_expression,
                                                                                    self.variables,
                                                                                    parsed_kwargs,
                                                                                    self._exact_rational_lambdified)
        try:
            return _parse_evaluate_numeric(result)
        except ValueError as err:
            raise NonNumericEvaluation(self, result, scope) from err

    def evaluate_in_scope(self, scope: Mapping) -> Union[Number, numpy.ndarray]:
        parsed_kwargs = self._parse_evaluate_numeric_arguments(scope)
        result, self._expression_lambda = evaluate_lambdified(self.underlying_expression, self.variables,
                                                              parsed_kwargs, lambdified=self._expression_lambda)
        try:
            return _parse_evaluate_numeric(result)
        except ValueError as err:
            raise NonNumericEvaluation(self, result, scope) from err


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
        return "Could not evaluate <{}>: A value for variable <{}> is missing!".format(
            str(self.expression), self.variable)


class NonNumericEvaluation(Exception):
    """An exception that is raised if the result of evaluate_numeric is not a number.

    See also:
        qupulse.expressions.Expression.evaluate_numeric
    """

    def __init__(self, expression: Expression, non_numeric_result: Any, call_arguments: Mapping):
        self.expression = expression
        self.non_numeric_result = non_numeric_result
        self.call_arguments = call_arguments

    def __str__(self) -> str:
        if isinstance(self.non_numeric_result, numpy.ndarray):
            dtype = self.non_numeric_result.dtype

            if dtype == numpy.dtype('O'):
                dtypes = set(map(type, self.non_numeric_result.flat))
                "The result of evaluate_numeric is an array with the types {} " \
                "which is not purely numeric".format(dtypes)
        else:
            dtype = type(self.non_numeric_result)
        return "The result of evaluate_numeric is of type {} " \
               "which is not a number".format(dtype)


ExpressionLike = TypeVar('ExpressionLike', str, Number, sympy.Expr, ExpressionScalar)
