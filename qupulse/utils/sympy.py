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

namespace_separator = '.'


class SubscriptableSymbol(sympy.Symbol):

    def __new__(cls, name: str, namespace: "SymbolNamespace"=None, **kwargs):
        if "." in name:
            raise ValueError("namespace name may not contain a namespace separator \".\"!")
        prefix = ""
        if namespace:
            prefix = namespace.name + "."
        name = prefix + name
        return super().__new__(cls, name, **kwargs)

    def __init__(self, name: str, namespace: "SymbolNamespace"=None) -> None:
        self._namespace = namespace
        self._inner_name = name

    def _subs(self, old, new, **hints):
        # print(self._inner_name)
        # print(self.name)
        # print(str(old) + " " + str(type(old)))
        # print(str(new) + " " + str(type(new)))
        # print(hints)
        if old.name == self.name:
            return new
        return self

    # @property
    # def _iterable(self) -> bool:
    #     return False

    def _sympyrepr(self, settings=None):
        prefix = ""
        if self._namespace:
            prefix = self._namespace._sympyrepr() + "."
        return prefix + self._inner_name

    def __repr__(self) -> str:
        return str(self)

    # def __getitem__(self, item):
    #     print(str(self) + "[" + str(item) + "]")
    #     return sympy.IndexedBase(self._inner_name)[item]


class SymbolNamespace:
    """A sympy.Symbol that acts as a namespace.

    Grants attribute access to all known members of that namespace, as given during initialization to the
    namespace_member parameter.
    """

    def __init__(self, name: Union[str, sympy.Symbol], members: Optional[Dict[str, sympy.Symbol]]=None, parent: "SymbolNamespace"=None):
        self._name = name
        if isinstance(self._name, sympy.Symbol):
            self._name = self._name.name
        if "." in self._name:
            raise ValueError("namespace name may not contain a namespace separator \".\"!")
        self._members = members
        if not self._members:
            self._members = dict()
        self._parent = parent
        #self._namespace, self._inner_name = self._split_namespaced_name(self.name)

    @staticmethod
    def split_namespaced_name(name: str) -> Tuple[str, str]:
        split = name.rsplit(namespace_separator, maxsplit=1)
        if len(split) > 1:
            namespace, inner_name = split
        else:
            inner_name = split[0]
            namespace = ''
        return namespace, inner_name

    # @property
    # def inner_name(self) -> str:
    #     return self._inner_name
    #
    # @property
    # def namespace(self) -> str:
    #     return self._namespace

    @property
    def name(self) -> str:
        prefix = ""
        if self._parent:
            prefix = self._parent.name + "."
        return prefix + self._name

    def __getattr__(self, k: str) -> sympy.Expr:
        if k == "NS" or k == "SymbolNamespace":
            return SymbolNamespace(k, parent=self)
        if k not in self._members:
            #self._members[k] = SubscriptableSymbol(self.name + "." + k)
            self._members[k] = SubscriptableSymbol(k, namespace=self)
        return self._members[k]

    def renamed(self, new_name: str) -> 'SymbolNamespace':
        return SymbolNamespace(new_name, self._members)

    def concat(self, next: 'SymbolNamespace') -> 'SymbolNamespace':
        assert(not self._members)
        return next.renamed(self.name + "." + next.name)

    def _sympyrepr(self) -> str:
        prefix = ""
        if self._parent:
            prefix = self._parent._sympystr() + "."
        return prefix + "NS('" + self._name + "')"

    def __repr__(self) -> str:
        return "SymbolNamespace({})".format(self.name)

    def __str__(self) -> str:
        return str(self.name)


##############################################################################
### Utilities to automatically detect usage of indexed/subscripted symbols ###
##############################################################################

class CustomSyntaxInterpreter:
    """Acts as a symbol lookup for sympify and determines which symbols in an expression are subscripted/indexed or
    form a namespace."""

    def __init__(self):
        self.symbols = dict()
        self.bases = dict()

        class CheckingSymbol(sympy.Symbol):
            """A symbol stand-in which detects whether the symbol is subscripted or a namespace and simulates proper
            behavior.

            Overloads __getattr__ to simulate namespace behavior: All attribute accesses to non-existent attributes of
            sympy.Symbol are treated as Symbols nested in the namespace represented by this CheckingSymbol; returned
            is another instance of CheckingSymbol.

            Overloads __getitem__ to detect item accesses to determine whether this CheckingSymbol represents an indexed
            symbol. Returns sympy.Indexed.

            The method get_symbol() is used to return the proper class representation for use after detection.
            It returns:
                - sympy.IndexedBase if __getitem__ was called
                - sympy.NamespaceSymbol if __getattr_ was called (i.e., CheckedSymbol is a namespace)
                - sympy.Symbol if __getitem__, __getattr__ were never called (i.e., CheckedSymbol is neither namespace
                    nor indexed symbol)

            The case of detected calls to __getitem__ and __getattr__ (indexed symbol AND namespace) is unspecified.
            """

            def __init__(s, *args, **kwargs) -> None:
                sympy.Symbol.__init__(*args, **kwargs)
                s._symbols = dict()
                s._is_index_base = False

            def __getitem__(s, k):
                s._is_index_base = True
                if isinstance(k, CheckingSymbol):
                    k = sympy.Symbol(str(k))
                self.bases[str(s)] = sympy.IndexedBase(str(s))
                return sympy.IndexedBase(str(s))[k]

            # def __getattr__(s, k: str):
            #     # note: getattr only gets called once standard attribute lookup procedures fail, i.e., python cannot
            #     # find a class attribute/function named k
            #
            #     # do not allow namespace members starting with '_' to ensure that
            #     #  object attributes starting with '_' are not shadowed
            #     if k.startswith('_'):
            #         raise AttributeError(k)
            #
            #     if k not in s._symbols:
            #         s._symbols[k] = self.SubscriptionChecker(s.name + namespace_separator + k)
            #     return s._symbols[k]
            #
            # def get_symbol(s) -> sympy.Symbol:
            #     if s._is_index_base:
            #         return sympy.IndexedBase(s.name)
            #     elif not s._symbols:
            #         return sympy.Symbol(s.name)
            #     return NamespaceSymbol(s.name, namespace_members={n: s.get_symbol() for n, s in s._symbols.items()})

        self.SubscriptionChecker = CheckingSymbol

    def __getitem__(self, k) -> sympy.Expr:
        """Return an instance of the internal SubscriptionChecker class for each symbol to determine which symbols are
        indexed/subscripted.

        __getitem__ is (apparently) called by symbol for each token and gets either symbol names or type names such as
        'Integer', 'Float', etc. We have to take care of returning correct types for symbols (-> SubscriptionChecker)
        and the base types (-> Integer, Float, etc).
        """
        if k in sympify_namespace:
            # if k in namespace_locals:
            #     return SymbolNamespacePrototype(self.SubscriptionChecker)
            return sympify_namespace[k]
        if hasattr(sympy, k): # if k is a sympy base type identifier, return the base type
            return getattr(sympy, k)

        # otherwise track the symbol name and return a SubscriptionChecker instance
        if k not in self.symbols:
            self.symbols[k] = SubscriptableSymbol(k)
        return self.symbols[k]

    def __contains__(self, k) -> bool:
        return True

    def get_symbols(self) -> Dict[str, sympy.Symbol]:
        # namespace_symbols = dict()
        # for symbol in self.symbols.values():
        #     namespace_symbol = symbol.get_symbol()
        #     namespace_symbols[namespace_symbol.name] = namespace_symbol
        # return namespace_symbols
        #return self.bases
        return self.symbols


def get_symbols_for_custom_syntax(expression: str) -> Dict[str, sympy.Symbol]:
    # track all symbols that are subscipted in here
    indexed_base_finder = CustomSyntaxInterpreter()
    sympy.sympify(expression, locals=indexed_base_finder)

    return indexed_base_finder.get_symbols()

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


namespace_locals = {'SymbolNamespace': SymbolNamespace,
                    'NS': SymbolNamespace,
                    'SubscriptableSymbol': SubscriptableSymbol}

sympify_namespace = {'len': Len,
                     'Len': Len,
                     **namespace_locals}

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
        # first try to use vanilla sympify for parsing -> faster for standard sympy syntax
        return sympy.sympify(expr, **kwargs, locals=sympify_namespace)
    except (TypeError, AttributeError):
        # expression contains custom syntax: subscripted or namcespace symbols, do custom symbol parsing
        namespace_symbols = get_symbols_for_custom_syntax(expr)
        locals = {**namespace_symbols, **sympify_namespace}
        return sympy.sympify(expr, **kwargs, locals=locals) # todo [2018-11]: until here we invoke 3 times sympy, can we do this better?


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
    print(string_representation)
    return eval(string_representation, sympy.__dict__, {'Symbol': substitutions.__getitem__,
                                                        'Mul': numpy_compatible_mul,
                                                        **sympify_namespace})


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
