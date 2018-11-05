from typing import Union, Dict, Tuple, Any, Sequence, Optional
from numbers import Number
from types import CodeType

import builtins
import math

import sympy
import numpy


__all__ = ["sympify", "substitute_with_eval", "to_numpy", "get_variables", "get_free_symbols", "recursive_substitution",
           "evaluate_lambdified", "get_most_simple_representation"]


Sympifyable = Union[str, Number, sympy.Expr, numpy.str_]

namespace_separator = '.'


class NamespaceSymbol(sympy.Symbol):
    """A sympy.Symbol that acts as a namespace.

    Grants attribute access to all known members of that namespace, as given during initialization to the
    namespace_member parameter.
    """

    def __init__(self, *args, namespace_members: Sequence[sympy.Symbol], **kwargs):
        sympy.Symbol.__init__(*args, **kwargs)
        self._namespace_members = namespace_members
        self._namespace, self._inner_name = self._split_namespaced_name(self.name)

    def _split_namespaced_name(self, name: str) -> Tuple[str, str]:
        split = name.rsplit(namespace_separator, maxsplit=1)
        if len(split) > 1:
            namespace, inner_name = split
        else:
            inner_name = split[0]
            namespace = ''
        return namespace, inner_name

    @property
    def inner_name(self) -> str:
        return self._inner_name

    @property
    def namespace(self) -> str:
        return self._namespace

    def __getattr__(self, k: str) -> 'Symbol':
        try:
            return self._namespace_members[k]
        except KeyError:
            raise AttributeError(k)

    def __repr__(self) -> str:
        return "NamespaceSymbol({})".format(self.name)

    def __str__(self) -> str:
        return str(self.name)


##############################################################################
### Utilities to automatically detect usage of indexed/subscripted symbols ###
##############################################################################

class IndexedBasedFinder:
    """Acts as a symbol lookup and determines which symbols in an expression a subscripted."""

    def __init__(self):
        self.symbols = dict()
        self.indexed_base = set()
        self.indices = set()

        class SubscriptionChecker(sympy.Symbol):
            """A symbol stand-in which detects whether the symbol is subscripted."""

            def __init__(s, *args, **kwargs) -> None:
                sympy.Symbol.__init__(*args, **kwargs)
                s._symbols = dict()
                s._is_index_base = False

            def __getitem__(s, k):
                self.indexed_base.add(str(s))
                self.indices.add(k)
                s._is_index_base = True
                if isinstance(k, SubscriptionChecker):
                    k = sympy.Symbol(str(k))
                return sympy.IndexedBase(str(s))[k]

            def __getattr__(s, k: str):
                if k.startswith('_'):
                    raise AttributeError(k)
                if k not in s._symbols:
                    s._symbols[k] = self.SubscriptionChecker(s.name + namespace_separator + k)
                return s._symbols[k]

            def get_symbol(s) -> sympy.Symbol:
                if s._is_index_base:
                    return sympy.IndexedBase(s.name)
                elif not s._symbols:
                    return sympy.Symbol(s.name)
                return NamespaceSymbol(s.name, namespace_members={n: s.get_symbol() for n, s in s._symbols.items()})

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
        if k not in self.symbols:
            self.symbols[k] = self.SubscriptionChecker(k)
        return self.symbols[k]

    def __contains__(self, k) -> bool:
        return True

    def get_symbols(self) -> Dict[str, sympy.Symbol]:
        namespace_symbols = dict()
        for symbol in self.symbols.values():
            namespace_symbol = symbol.get_symbol()
            namespace_symbols[namespace_symbol.name] = namespace_symbol
        return namespace_symbols


def get_subscripted_symbols(expression: str) -> set:
    # track all symbols that are subscipted in here
    indexed_base_finder = IndexedBasedFinder()
    sympy.sympify(expression, locals=indexed_base_finder)

    indexed_bases = indexed_base_finder.indexed_base
    namespace_symbols = indexed_base_finder.get_symbols()

    return indexed_bases, namespace_symbols

#############################################################
### "Built-in" length function for expressions in qupulse ###
#############################################################

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

#########################################
### Functions for numpy compatability ###
#########################################

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

#######################################################################################################
### Custom sympify method (which introduces all utility methods defined above into the sympy world) ###
#######################################################################################################

def sympify(expr: Union[str, Number, sympy.Expr, numpy.str_], **kwargs) -> sympy.Expr:
    if isinstance(expr, numpy.str_):
        # putting numpy.str_ in sympy.sympify behaves unexpected in version 1.1.1
        # It seems to ignore the locals argument
        expr = str(expr)
    try:
        return sympy.sympify(expr, **kwargs, locals=sympify_namespace)
    except (TypeError, AttributeError) as err:
        if True:#err.args[0] == "'Symbol' object is not subscriptable":

            indexed_base, namespace_symbols = get_subscripted_symbols(expr)
            locals = {**{k: sympy.IndexedBase(k) for k in indexed_base},
                      **namespace_symbols,
                      **sympify_namespace}
            return sympy.sympify(expr, **kwargs, locals=locals)

        else:
            raise


###############################################################################
### Utility functions for expression manipulation/simplification/evaluation ###
###############################################################################

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
    lambdified = lambdified or sympy.lambdify(variables, expression,
                                              [{'ceiling': numpy_compatible_ceiling}, 'numpy'])

    return lambdified(**parameters), lambdified
