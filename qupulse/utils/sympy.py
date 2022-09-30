from typing import Union, Dict, Tuple, Any, Sequence, Optional, Callable
from numbers import Number
from types import CodeType
import warnings
import functools

import builtins
import math

import sympy
import numpy

try:
    from sympy.printing.numpy import NumPyPrinter
except ImportError:
    # sympy moved NumPyPrinter in release 1.8
    from sympy.printing.pycode import NumPyPrinter
    warnings.warn("Please update sympy.", DeprecationWarning)

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


_lru_cache = functools.lru_cache(maxsize=2048, typed=True)


Sympifyable = Union[str, Number, sympy.Expr, numpy.str_]

SYMPY_DURATION_ERROR_MARGIN = 1e-15 # error margin when checking sympy expression durations

class IndexedBasedFinder(dict):
    """Acts as a symbol lookup and determines which symbols in an expression a subscripted."""

    def __init__(self):
        super().__init__()
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

        def unimplementded(*args, **kwargs):
            raise NotImplementedError("Not a full dict")

        for m in vars(dict).keys():
            if not m.startswith('_') and (m not in ('pop',)):
                setattr(self, m, unimplementded)

    def __getitem__(self, k) -> sympy.Expr:
        """Return an instance of the internal SubscriptionChecker class for each symbol to determine which symbols are
        indexed/subscripted.

        __getitem__ is (apparently) called by symbol for each token and gets either symbol names or type names such as
        'Integer', 'Float', etc. We have to take care of returning correct types for symbols (-> SubscriptionChecker)
        and the base types (-> Integer, Float, etc).
        """
        if not k:
            raise KeyError(k)
        if hasattr(sympy, k): # if k is a sympy base type identifier, return the base type
            return getattr(sympy, k)

        # otherwise track the symbol name and return a SubscriptionChecker instance
        self.symbols.add(k)
        return self.SubscriptionChecker(k)

    def pop(self, key, *args, **kwargs):
        # this is a workaround for some sympy 1.9 code
        if args:
            default, = args
        elif kwargs:
            default, = kwargs.values()
        else:
            raise KeyError(key)
        return default

    def __setitem__(self, key, value):
        raise NotImplementedError("Not a full dict")

    def __delitem__(self, key):
        raise NotImplementedError("Not a full dict")

    def __contains__(self, k) -> bool:
        return bool(k)


class Broadcast(sympy.Function):
    """Broadcast x to the specified shape using numpy.broadcast_to. The shape must not be symbolic.

    Examples:
        >>> bc = Broadcast('a', (3,))
        >>> assert bc.subs({'a': 2}) == sympy.Array([2, 2, 2])
        >>> assert bc.subs({'a': (1, 2, 3)}) == sympy.Array([1, 2, 3])
    """
    nargs = (2,)

    @classmethod
    def eval(cls, x, shape: Tuple[int]) -> Optional[sympy.Array]:
        shape = _parse_broadcast_shape(shape, user=cls)
        if shape is None:
            return None

        if hasattr(x, '__len__') or not x.free_symbols:
            return sympy.Array(numpy.broadcast_to(x, shape))

    def __getitem__(self, item: Union):
        return IndexedBroadcast(*self.args, item)

    # Not iterable. If not set to None __getitem__ would be used for iterating
    __iter__ = None

    def _eval_Integral(self, *symbols, **assumptions):
        x, shape = self.args
        return Broadcast(sympy.Integral(x, *symbols, **assumptions), shape)

    def _eval_derivative(self, sym):
        x, shape = self.args
        return Broadcast(sympy.diff(x, sym), shape)

    def _numpycode(self, printer, **kwargs):
        x, shape = map(functools.partial(printer._print, **kwargs), self.args)
        return f'broadcast_to({x}, {shape})'


class IndexedBroadcast(sympy.Function):
    """Broadcast x to the specified shape using numpy.broadcast_to and index in the result."""
    nargs = (3,)

    @classmethod
    def eval(cls, x, shape: Tuple[int], idx: int) -> Optional[sympy.Expr]:
        shape = _parse_broadcast_shape(shape, user=cls)
        idx = _parse_broadcast_index(idx, user=cls)
        if shape is None or idx is None:
            return None

        if hasattr(x, '__len__') or not x.free_symbols:
            return sympy.Array(numpy.broadcast_to(x, shape))[idx]

    def _eval_Integral(self, *symbols, **assumptions):
        x, shape, idx = self.args
        return IndexedBroadcast(sympy.Integral(x, *symbols, **assumptions), shape, idx)

    def _eval_derivative(self, sym):
        x, shape, idx = self.args
        return IndexedBroadcast(sympy.diff(x, sym), shape, idx)

    def _eval_is_commutative(self):
        x, shape, idx = self.args
        result = self.eval(*self.args)
        if result is None:
            return x.is_commutative
        else:
            return result.is_commutative

    def _numpycode(self, printer, **kwargs):
        x, shape, idx = map(functools.partial(printer._print, **kwargs), self.args)
        return f'broadcast_to({x}, {shape})[{idx}]'


class Len(sympy.Function):
    nargs = 1

    @classmethod
    def eval(cls, arg) -> Optional[sympy.Integer]:
        if hasattr(arg, '__len__'):
            return sympy.Integer(len(arg))

    is_Integer = True
Len.__name__ = 'len'


sympify_namespace = {'len': Len,
                     'Len': Len,
                     'Broadcast': Broadcast,
                     'IndexedBroadcast': IndexedBroadcast}


def numpy_compatible_mul(*args) -> Union[sympy.Mul, sympy.Array]:
    if any(isinstance(a, sympy.NDimArray) for a in args):
        result = 1
        for a in args:
            result = result * (numpy.array(a.tolist()) if isinstance(a, sympy.NDimArray) else a)
        return sympy.Array(result)
    else:
        return sympy.Mul(*args)


def numpy_compatible_add(*args) -> Union[sympy.Add, sympy.Array]:
    if any(isinstance(a, sympy.NDimArray) for a in args):
        result = 0
        for a in args:
            result = result + (numpy.array(a.tolist()) if isinstance(a, sympy.NDimArray) else a)
        return sympy.Array(result)
    else:
        return sympy.Add(*args)


_NUMPY_COMPATIBLE = {
    sympy.Add: numpy_compatible_add,
    sympy.Mul: numpy_compatible_mul
}


def _float_arr_to_int_arr(float_arr):
    """Try to cast array to int64. Return original array if data is not representable."""
    int_arr = float_arr.astype(numpy.int64)
    if numpy.any(int_arr != float_arr):
        # we either have a float that is too large or NaN
        return float_arr
    else:
        return int_arr


def numpy_compatible_ceiling(input_value: Any) -> Any:
    if isinstance(input_value, numpy.ndarray):
        return _float_arr_to_int_arr(numpy.ceil(input_value))
    else:
        return sympy.ceiling(input_value)


def _floor_to_int(input_value: Any) -> Any:
    if isinstance(input_value, numpy.ndarray):
        return _float_arr_to_int_arr(numpy.floor(input_value))
    else:
        return sympy.floor(input_value)


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
    if isinstance(expr, (tuple, list)):
        expr = numpy.array(expr)
    try:
        return sympy.sympify(expr, **kwargs, locals=sympify_namespace)
    except TypeError as err:
        if True:#err.args[0] == "'Symbol' object is not subscriptable":

            indexed_base = get_subscripted_symbols(expr)
            return sympy.sympify(expr, **kwargs, locals={**{k: k if isinstance(k, Broadcast) else sympy.IndexedBase(k)
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
    warnings.warn("substitute_with_eval does not handle dummy symbols correctly and is planned to be removed",
                  FutureWarning)

    substitutions = {k: v if isinstance(v, sympy.Expr) else sympify(v)
                     for k, v in substitutions.items()}

    for symbol in get_free_symbols(expression):
        symbol_name = str(symbol)
        if symbol_name not in substitutions:
            substitutions[symbol_name] = symbol

    string_representation = sympy.srepr(expression)
    return eval(string_representation, sympy.__dict__, {'Symbol': substitutions.__getitem__,
                                                        'Mul': numpy_compatible_mul,
                                                        'Add': numpy_compatible_add})


get_free_symbols_cache = _lru_cache(get_free_symbols)


def _recursive_substitution(expression: sympy.Expr,
                           substitutions: Dict[sympy.Symbol, sympy.Expr]) -> sympy.Expr:
    if not expression.free_symbols:
        return expression
    elif expression.func in (sympy.Symbol, sympy.Dummy):
        return substitutions.get(expression, expression)

    func = _NUMPY_COMPATIBLE.get(expression.func, expression.func)
    substitutions = {s: substitutions.get(s, s) for s in get_free_symbols_cache(expression)}
    operands = (_recursive_substitution(arg, substitutions) for arg in expression.args)
    return func(*operands)



_cached_sympify = _lru_cache(sympify)


def sympify_cache(value):
    """Cache sympify result for all hashable types"""
    if getattr(value, '__hash__', None) is not None:
        try:
            return _cached_sympify(value)
        except TypeError:
            pass
    # type is either not hashable or the sympification failed for another reason
    return sympify(value)


def recursive_substitution(expression: sympy.Expr,
                           substitutions: Dict[str, Union[sympy.Expr, numpy.ndarray, str]]) -> sympy.Expr:
    substitutions = {k if isinstance(k, (sympy.Symbol, sympy.Dummy)) else sympy.Symbol(k): sympify_cache(v)
                     for k, v in substitutions.items()}
    for s in get_free_symbols(expression):
        substitutions.setdefault(s, s)
    return _recursive_substitution(expression, substitutions)


_base_environment = {'builtins': builtins, '__builtins__':  builtins}
_math_environment = {**_base_environment, **math.__dict__}
_numpy_environment = {**_base_environment, **numpy.__dict__}
_sympy_environment = {**_base_environment, **sympy.__dict__}

_lambdify_modules = [{'ceiling': numpy_compatible_ceiling, 'floor': _floor_to_int,
                      'Broadcast': numpy.broadcast_to}, 'numpy', _special_functions]


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
                        lambdified: Optional[Callable]) -> Tuple[Any, Any]:
    lambdified = lambdified or sympy.lambdify(variables, expression, _lambdify_modules)
    return lambdified(**parameters), lambdified


class HighPrecPrinter(NumPyPrinter):
    """Custom printer that translates sympy.Rational into TimeType"""
    def _print_Rational(self, expr):
        return f'TimeType.from_fraction({expr.p}, {expr.q})'

    @classmethod
    def make(cls, expr, modules, use_imps=True):
        """This is basically the printer creation code from sympy 1.6 lambdify"""
        namespaces = []
        if use_imps:
            raise NotImplementedError('this is copied from lambdify printer creation but _imp_namespace is not puplic')
        # Check for dict before iterating
        namespaces += list(modules)

        user_functions = {}
        for m in _lambdify_modules[::-1]:
            if isinstance(m, dict):
                for k in m:
                    user_functions[k] = k
        return cls({'fully_qualified_modules': False, 'inline': True,
                    'allow_unknown_functions': True,
                    'user_functions': user_functions})


def evaluate_lamdified_exact_rational(expression: sympy.Expr,
                                      variables: Sequence[str],
                                      parameters: Dict[str, Union[numpy.ndarray, Number]],
                                      lambdified: Optional[Callable]) -> Tuple[Any, Any]:
    """Evaluates Rational as TimeType. Only supports scalar expressions"""
    from qupulse.utils.types import TimeType
    _lambdify_modules[0]['TimeType'] = TimeType
    printer = HighPrecPrinter.make(expression, _lambdify_modules, use_imps=False)
    lambdified = lambdified or sympy.lambdify(variables, expression, _lambdify_modules, printer=printer)
    return lambdified(**parameters), lambdified


def almost_equal(lhs: sympy.Expr, rhs: sympy.Expr, epsilon: Optional[float]=None) -> Optional[bool]:
    """Returns True (or False) if the two expressions are almost equal (or not). Returns None if this cannot be
    determined."""
    if epsilon is None:
        epsilon = SYMPY_DURATION_ERROR_MARGIN
    relation = sympy.simplify(sympy.Abs(lhs - rhs) <= epsilon)

    if relation is sympy.true:
        return True
    elif relation is sympy.false:
        return False
    else:
        return None


class UnsupportedBroadcastArgumentWarning(RuntimeWarning):
    pass


def _parse_broadcast_shape(shape: Tuple[int], user: type) -> Optional[Tuple[int]]:
    try:
        return tuple(map(int, shape))
    except TypeError as err:
        warnings.warn(f"The shape passed to {user.__module__}.{user.__name__} is not convertible to a tuple of integers: {err}\n"
                      "Be aware that using a symbolic shape can lead to unexpected behaviour.",
                      category=UnsupportedBroadcastArgumentWarning,
                      # probably sympy version dependent what is most useful here...
                      stacklevel=7)
    return None


def _parse_broadcast_index(idx: int, user: type) -> Optional[int]:
    try:
        return int(idx)
    except TypeError as err:
        warnings.warn(f"The index passed to {user.__module__}.{user.__name__} is not convertible to an integer: {err}\n"
                      "Be aware that using a symbolic index can lead to unexpected behaviour.",
                      category=UnsupportedBroadcastArgumentWarning,
                      # probably sympy version dependent what is most useful here...
                      stacklevel=7)
    return None
