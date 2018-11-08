from typing import Union, Dict, Tuple, Any, Sequence, Optional
from numbers import Number
from types import CodeType
import warnings

import builtins
import math

import sympy
import numpy

try:
    import scipy.special as _special_functions
except ImportError:
    _special_functions = {fname: numpy.vectorize(fobject)
                          for fname, fobject in math.__dict__.items()
                          if not fname.startswith('_') and fname not in numpy.__dict__}
    warnings.warn('scipy is not installed. This reduces the set of available functions to those present in numpy + '
                  'manually vectorized functions in math.')


__all__ = ["sympify", "substitute_with_eval", "to_numpy", "get_variables", "get_free_symbols", "recursive_substitution",
           "evaluate_lambdified", "get_most_simple_representation"]


Sympifyable = Union[str, Number, sympy.Expr, numpy.str_]


class IndexedBasedFinder:
    """Acts as a symbol lookup and determines which symbols in an expression a subscripted."""

    def __init__(self):
        self.symbols = set()
        self.indexed_base = set()
        self.indices = set()

        class SubscriptionChecker(sympy.Symbol):
            """A symbol stand-in which detects whether the symbol is subscripted."""

            def __getitem__(s, k):
                self.indexed_base.add(str(s))
                self.indices.add(k)
                if isinstance(k, SubscriptionChecker):
                    k = sympy.Symbol(str(k))
                return sympy.IndexedBase(str(s))[k]

        self.SubscriptionChecker = SubscriptionChecker

    def __getitem__(self, k) -> sympy.Expr:
        """Return an instance of the internal SubscriptionChecker class for each symbol to determine which symbols are
        indexed/subscripted.

        __getitem__ is (apparently) called by symbol for each token and gets either symbol names or type names such as
        'Integer', 'Float', etc. We have to take care of returning correct types for symbols (-> SubscriptionChecker)
        and the base types (-> Integer, Float, etc).
        """
        if hasattr(sympy, k): # if k is a sympy base type identifier, return the base type
            return getattr(sympy, k)

        # otherwise track the symbol name and return a SubscriptionChecker instance
        self.symbols.add(k)
        return self.SubscriptionChecker(k)

    def __contains__(self, k) -> bool:
        return True


class Len(sympy.Function):
    nargs = 1

    @classmethod
    def eval(cls, arg) -> Optional[sympy.Integer]:
        if hasattr(arg, '__len__'):
            return sympy.Integer(len(arg))

    is_Integer = True
Len.__name__ = 'len'


sympify_namespace = {'len': Len,
                     'Len': Len}


def numpy_compatible_mul(*args) -> Union[sympy.Mul, sympy.Array]:
    if any(isinstance(a, sympy.NDimArray) for a in args):
        result = 1
        for a in args:
            result = result * (numpy.array(a.tolist()) if isinstance(a, sympy.NDimArray) else a)
        return sympy.Array(result)
    else:
        return sympy.Mul(*args)


def numpy_compatible_ceiling(input_value: Any) -> Any:
    if isinstance(input_value, numpy.ndarray):
        return numpy.ceil(input_value).astype(numpy.int64)
    else:
        return sympy.ceiling(input_value)


def to_numpy(sympy_array: sympy.NDimArray) -> numpy.ndarray:
    if isinstance(sympy_array, sympy.DenseNDimArray):
        if len(sympy_array.shape) == 2:
            return numpy.asarray(sympy_array.tomatrix())
        elif len(sympy_array.shape) == 1:
            return numpy.asarray(sympy_array)
    return numpy.array(sympy_array.tolist())


def get_subscripted_symbols(expression: str) -> set:
    # track all symbols that are subscipted in here
    indexed_base_finder = IndexedBasedFinder()
    sympy.sympify(expression, locals=indexed_base_finder)

    return indexed_base_finder.indexed_base


def sympify(expr: Union[str, Number, sympy.Expr, numpy.str_], **kwargs) -> sympy.Expr:
    if isinstance(expr, numpy.str_):
        # putting numpy.str_ in sympy.sympify behaves unexpected in version 1.1.1
        # It seems to ignore the locals argument
        expr = str(expr)
    try:
        return sympy.sympify(expr, **kwargs, locals=sympify_namespace)
    except TypeError as err:
        if True:#err.args[0] == "'Symbol' object is not subscriptable":

            indexed_base = get_subscripted_symbols(expr)
            return sympy.sympify(expr, **kwargs, locals={**{k: sympy.IndexedBase(k)
                                                            for k in indexed_base},
                                                         **sympify_namespace})

        else:
            raise


def get_most_simple_representation(expression: sympy.Expr) -> Union[str, int, float]:
    if expression.free_symbols:
        return str(expression)
    elif expression.is_Integer:
        return int(expression)
    elif expression.is_Float:
        return float(expression)
    else:
        return str(expression)


def get_free_symbols(expression: sympy.Expr) -> Sequence[sympy.Symbol]:
    return tuple(symbol
                 for symbol in expression.free_symbols
                 if not isinstance(symbol, sympy.Indexed))


def get_variables(expression: sympy.Expr) -> Sequence[str]:
    return tuple(map(str, get_free_symbols(expression)))


def substitute_with_eval(expression: sympy.Expr,
                         substitutions: Dict[str, Union[sympy.Expr, numpy.ndarray, str]]) -> sympy.Expr:
    """Substitutes only sympy.Symbols. Workaround for numpy like array behaviour. ~Factor 3 slower compared to subs"""
    substitutions = {k: v if isinstance(v, sympy.Expr) else sympify(v)
                     for k, v in substitutions.items()}

    for symbol in get_free_symbols(expression):
        symbol_name = str(symbol)
        if symbol_name not in substitutions:
            substitutions[symbol_name] = symbol

    string_representation = sympy.srepr(expression)
    return eval(string_representation, sympy.__dict__, {'Symbol': substitutions.__getitem__,
                                                        'Mul': numpy_compatible_mul})


def _recursive_substitution(expression: sympy.Expr,
                           substitutions: Dict[sympy.Symbol, sympy.Expr]) -> sympy.Expr:
    if not expression.free_symbols:
        return expression
    elif expression.func is sympy.Symbol:
        return substitutions.get(expression, expression)

    elif expression.func is sympy.Mul:
        func = numpy_compatible_mul
    else:
        func = expression.func
    substitutions = {s: substitutions.get(s, s) for s in get_free_symbols(expression)}
    return func(*(_recursive_substitution(arg, substitutions) for arg in expression.args))


def recursive_substitution(expression: sympy.Expr,
                           substitutions: Dict[str, Union[sympy.Expr, numpy.ndarray, str]]) -> sympy.Expr:
    substitutions = {sympy.Symbol(k): sympify(v) for k, v in substitutions.items()}
    for s in get_free_symbols(expression):
        substitutions.setdefault(s, s)
    return _recursive_substitution(expression, substitutions)


_base_environment = {'builtins': builtins, '__builtins__':  builtins}
_math_environment = {**_base_environment, **math.__dict__}
_numpy_environment = {**_base_environment, **numpy.__dict__}
_sympy_environment = {**_base_environment, **sympy.__dict__}

_lambdify_modules = [{'ceiling': numpy_compatible_ceiling}, 'numpy', _special_functions]


def evaluate_compiled(expression: sympy.Expr,
             parameters: Dict[str, Union[numpy.ndarray, Number]],
             compiled: CodeType=None, mode=None) -> Tuple[any, CodeType]:
    if compiled is None:
        compiled = compile(sympy.printing.lambdarepr.lambdarepr(expression),
                           '<string>', 'eval')

    if mode == 'numeric' or mode is None:
        result = eval(compiled, parameters.copy(), _numpy_environment)
    elif mode == 'exact':
        result = eval(compiled, parameters.copy(), _sympy_environment)
    else:
        raise ValueError("Unknown mode: '{}'".format(mode))

    return result, compiled


def evaluate_lambdified(expression: Union[sympy.Expr, numpy.ndarray],
                        variables: Sequence[str],
                        parameters: Dict[str, Union[numpy.ndarray, Number]],
                        lambdified) -> Tuple[Any, Any]:
    lambdified = lambdified or sympy.lambdify(variables, expression, _lambdify_modules)

    return lambdified(**parameters), lambdified


def almost_equal(lhs: sympy.Expr, rhs: sympy.Expr, epsilon: float=1e-15) -> Optional[bool]:
    """Returns True (or False) if the two expressions are almost equal (or not). Returns None if this cannot be
    determined."""
    relation = sympy.simplify(sympy.Abs(lhs - rhs) <= epsilon)

    if relation is sympy.true:
        return True
    elif relation is sympy.false:
        return False
    else:
        return None
