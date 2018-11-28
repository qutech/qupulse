from typing import Union, Dict, Tuple, Any, Sequence, Optional
from numbers import Number
from types import CodeType
import warnings

import builtins
import math

import sympy
import numpy
import re

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


class NamespacedSymbol(sympy.Symbol):

    def __new__(cls, name: Union[str, sympy.Symbol], namespace: "SymbolNamespace"=None):
        if isinstance(name, sympy.Symbol):
            name = name.name
        if "." in name:
            raise ValueError("namespace name may not contain a namespace separator \".\"!")
        # prefix = ""
        # if namespace:
        #     prefix = namespace.name + "."
        # name = prefix + name
        prefix = ""
        if namespace:
            #prefix = self._namespace._sympyrepr() + "."
            prefix = namespace._sympystr() + "."
        name = prefix + name
        return super().__new__(cls, name)

    def __init__(self, name: str, namespace: "SymbolNamespace"=None) -> None:
        self._namespace = namespace
        self._inner_name = name

    @classmethod
    def instantiate(cls, name: str, namespace: "SymbolNamespace"=None):
        return cls(name, namespace=namespace)

    def change_namespace(self, namespace: "SymbolNamespace"=None) -> "NamespacedSymbol":
        return self.instantiate(name=self._inner_name, namespace=namespace)

    def _subs(self, old, new, **hints):
        if old.name == self.name:
            return new
        return self

    def _sympystr(self, options=None):
        prefix = ""
        if self._namespace:
            prefix = self._namespace._sympystr(options) + "."
        return prefix + self._inner_name

    def __repr__(self) -> str:
        return str(self)

    def to_sympy_symbol(self) -> sympy.Symbol:
        return sympy.Symbol(self.name)

    def __hash__(self) -> int:
        return hash(sympy.Symbol(self.name))

    def __eq__(self, other) -> bool:
        return sympy.Symbol(self.name) == other

    def _pythoncode(self, settings=None) -> str:
        prefix = ""
        if self._namespace:
            prefix = self._namespace._pythoncode(settings) + "____"
        return prefix + self._inner_name

    def _numpycode(self, settings=None) -> str:
        return self._pythoncode(settings)

    def _lambdacode(self, settings=None) -> str:
        return self._pythoncode(settings)


class NamespaceIndexedBase(NamespacedSymbol):

    def __getitem__(self, item):
        return sympy.IndexedBase(self.name)[item]

    def to_sympy_symbol(self) -> sympy.IndexedBase:
        return sympy.IndexedBase(self.name)

    @property
    def free_symbols(self):
        return {NamespacedSymbol(self._inner_name, namespace=self._namespace)}


class SymbolNamespace:
    """A sympy.Symbol that acts as a namespace.

    Grants attribute access to all known members of that namespace, as given during initialization to the
    namespace_member parameter.
    """

    def __init__(self, name: Union[str, sympy.Symbol, "SymbolNamespace"], members: Optional[Dict[str, NamespacedSymbol]]=None, parent: "SymbolNamespace"=None):
        self._name = name
        if isinstance(self._name, (sympy.Symbol, SymbolNamespace)):
            self._name = self._name.name
        if "." in self._name:
            raise ValueError("namespace name may not contain a namespace separator \".\"!")
        self._members = members
        self._parent = parent

        if not self._members:
            self._members = dict()
        else:
            self._members = {k:s.change_namespace(self) for k,s in self._members.items()}
            # for s in self._members:
            #     if isinstance(s, SymbolNamespace):
            #         s._parent = self
            #     elif isinstance(s, NamespacedSymbol):
            #         s._namespace = self
        #self._namespace, self._inner_name = self._split_namespaced_name(self.name)

    def change_namespace(self, namespace: "SymbolNamespace") -> "SymbolNamespace":
        return SymbolNamespace(self._name, members=self._members, parent=namespace)

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
            # return SymbolNamespaceFactory(parent=self)
            return SymbolNamespaceFactory(instances=self._members)
        if k not in self._members:
            #self._members[k] = SubscriptableSymbol(self.name + "." + k)
            self._members[k] = NamespacedSymbol(k, namespace=self)
        return self._members[k]

    def renamed(self, new_name: str) -> 'SymbolNamespace':
        return SymbolNamespace(new_name, self._members)

    def concat(self, next: 'SymbolNamespace') -> 'SymbolNamespace':
        assert(not self._members)
        return next.renamed(self.name + "." + next.name)

    def _sympystr(self, options=None) -> str:
        prefix = ""
        if self._parent:
            prefix = self._parent._sympystr(options) + "."
        return prefix + "NS(" + self._name + ")"

    def members(self) -> Dict[Union[str, sympy.Symbol], Union[NamespacedSymbol, "SymbolNamespace"]]:
        return self._members

    def _pythoncode(self, settings=None) -> str:
        prefix = ""
        if self._parent:
            prefix = self._parent._pythoncode(settings) + "____"
        return prefix + self._name

    def _numpycode(self, settings=None) -> str:
        return self._pythoncode(settings)

    def _lambdacode(self, settings=None) -> str:
        return self._pythoncode(settings)


class SymbolNamespaceFactory:

    def __init__(self, instances: Dict[Union[str, sympy.Symbol], SymbolNamespace]):
        self._instances = instances

    def __call__(self, name: Union[str, sympy.Symbol]):
        if str(name) not in self._instances:
            self._instances[str(name)] = SymbolNamespace(name=name, parent=self._parent)
        return self._instances[str(name)]

    def get_instances(self):
        return self._instances



##############################################################################
### Utilities to automatically detect usage of indexed/subscripted symbols ###
##############################################################################

class CustomSyntaxInterpreter:
    """Acts as a symbol lookup for sympify and determines which symbols in an expression are subscripted/indexed or
    form a namespace."""

    def __init__(self):
        self.symbols = dict()
        # self.bases = dict()

        class CheckingNamespaceFactory:

            def __init__(self, parent: "CheckingNamespace"=None):
                self._parent = parent
                self._instances = dict()

            def __call__(self, name: Union[str, sympy.Symbol]):
                if str(name) not in self._instances:
                    self._instances[str(name)] = CheckingNamespace(name=name, parent=self._parent)
                return self._instances[str(name)]

            def get_instances(self):
                return self._instances

        class CheckingNamespace(SymbolNamespace):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._fac = CheckingNamespaceFactory(parent=self)

            def __getattr__(self, k: str):
                if k == "NS" or k == "SymbolNamespace":
                    return self._fac
                if k not in self._members:
                    self._members[k] = CheckingSymbol(k)
                return self._members[k]

            def get_symbol(self) -> SymbolNamespace:
                members = dict()
                # convert all symbols
                for k, s in self._members.items():
                    assert(isinstance(s, CheckingSymbol))
                    members[k] = s.get_symbol()

                # convert all nested namespaces
                members.update(**{
                    k:s.get_symbol() for k,s in self._fac.get_instances().items()
                })
                return SymbolNamespace(self._name, members=members)


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
                s._is_index_base = False

            def __getitem__(s, k):
                s._is_index_base = True
                if isinstance(k, CheckingSymbol):
                    k = sympy.Symbol(str(k))
                return sympy.IndexedBase(str(s))[k]

            def get_symbol(s):
                if s._is_index_base:
                    return NamespaceIndexedBase(str(s))
                else:
                    return NamespacedSymbol(str(s))

        self.CheckingSymbol = CheckingSymbol
        self.namespace_fac = CheckingNamespaceFactory()

    def __getitem__(self, k) -> sympy.Expr:
        """Return an instance of the internal SubscriptionChecker class for each symbol to determine which symbols are
        indexed/subscripted.

        __getitem__ is (apparently) called by symbol for each token and gets either symbol names or type names such as
        'Integer', 'Float', etc. We have to take care of returning correct types for symbols (-> SubscriptionChecker)
        and the base types (-> Integer, Float, etc).
        """
        if k in sympify_namespace:
            return sympify_namespace[k]
        if k in namespace_locals:
            return self.namespace_fac
        if hasattr(sympy, k): # if k is a sympy base type identifier, return the base type
            return getattr(sympy, k)

        # otherwise track the symbol name and return a SubscriptionChecker instance
        if k not in self.symbols:
            self.symbols[k] = self.CheckingSymbol(k)
        return self.symbols[k]

    def __contains__(self, k) -> bool:
        return True

    def get_symbols(self) -> Dict[str, Union[sympy.Symbol, SymbolNamespace]]:
        # namespace_symbols = dict()
        # for symbol in self.symbols.values():
        #     namespace_symbol = symbol.get_symbol()
        #     namespace_symbols[namespace_symbol.name] = namespace_symbol
        # return namespace_symbols
        #return self.bases
        symbols = dict()
        # convert all symbols
        for k, s in self.symbols.items():
            assert (isinstance(s, self.CheckingSymbol))
            symbols[k] = s.get_symbol().to_sympy_symbol()

        # convert all nested namespaces
        nested = {k: s.get_symbol() for k, s in self.namespace_fac.get_instances().items()}
        rootFac = SymbolNamespaceFactory(instances=nested)
        symbols.update(**{
            k: rootFac for k in namespace_locals
        })

        # symbols.update(**{
        #     k: s.get_symbol() for k, s in self.namespace_fac.get_instances().items()
        # })

        return symbols


def get_symbols_for_custom_syntax(expression: str) -> Dict[str, sympy.Symbol]:
    # track all symbols that are subscipted in here
    syntax_interpreter = CustomSyntaxInterpreter()
    sympy.sympify(expression, locals=syntax_interpreter)

    return syntax_interpreter.get_symbols()

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


# namespace_locals = {'SymbolNamespace': SymbolNamespace,
#                     'NS': SymbolNamespace,
#                     'NamespacedSymbol': NamespacedSymbol}

namespace_locals = {'SymbolNamespace', 'NS'}

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
        # first try to use vanilla sympify for parsing -> faster for standard sympy syntax
        return sympy.sympify(expr, **kwargs, locals={**sympify_namespace, **{k: SymbolNamespace for k in namespace_locals}})
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
    raise NotImplementedError()
    # """Substitutes only sympy.Symbols. Workaround for numpy like array behaviour. ~Factor 3 slower compared to subs"""
    # substitutions = {k: v if isinstance(v, sympy.Expr) else sympify(v)
    #                  for k, v in substitutions.items()}
    #
    # for symbol in get_free_symbols(expression):
    #     symbol_name = str(symbol)
    #     if symbol_name not in substitutions:
    #         substitutions[symbol_name] = symbol
    #
    # string_representation = sympy.srepr(expression)
    # return eval(string_representation, sympy.__dict__, {'Symbol': substitutions.__getitem__,
    #                                                     'Mul': numpy_compatible_mul,
    #                                                     **sympify_namespace})


def _recursive_substitution(expression: sympy.Expr,
                           substitutions: Dict[sympy.Symbol, sympy.Expr]) -> sympy.Expr:
    if not expression.free_symbols:
        return expression
    elif isinstance(expression, sympy.Symbol):
        return substitutions.get(expression, expression)

    elif expression.func is sympy.Mul:
        func = numpy_compatible_mul
    else:
        func = expression.func
    substitutions = {s: substitutions.get(s, s) for s in get_free_symbols(expression)}
    return func(*(_recursive_substitution(arg, substitutions) for arg in expression.args))


def recursive_substitution(expression: sympy.Expr,
                           substitutions: Dict[str, Union[sympy.Expr, numpy.ndarray, str]]) -> sympy.Expr:
    substitutions = {sympify(k): sympify(v) for k, v in substitutions.items()}
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


parameter_namespace_filter_regex = re.compile(r'(NS|SymbolNamespace)\(["\']?(\w+)["\']?\)\.')


def evaluate_lambdified(expression: Union[sympy.Expr, numpy.ndarray],
                        variables: Sequence[str],
                        parameters: Dict[str, Union[numpy.ndarray, Number]],
                        lambdified) -> Tuple[Any, Any]:
    # mapping namespaced parameter names to something that doesn't provoke syntax errors when used as argument for lamda
    # NS(foo).bar -> foo____bar
    name_map = {v:parameter_namespace_filter_regex.sub(r'\2____', v) for v in variables}
    variables = name_map.values()
    parameters = {name_map[k]: v for k, v in parameters.items()}
    substitutions = {sympy.Symbol(k):sympy.Symbol(v) for k, v in name_map.items()}

    # while I've overloaded the corresponding printing functions (_pythoncode, _lamdadcode, _numpycode) of
    # NamespacedSymbol (and thus NamespacedIndexedBase) and SymbolNamespace to produce the above renamed identifier,
    # these are not reliable called by the code printers when nested in numpy arrays, sums, etc..
    # -> perform symbol substitution up front
    if isinstance(expression, numpy.ndarray):
        for i in range(expression.size):
            #expression[i] = recursive_substitution(expression[i], substitutions)
            expression[i] = expression[i].subs(substitutions)
    else:
        #expression = recursive_substitution(expression, substitutions)
        expression = expression.subs(substitutions)

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
