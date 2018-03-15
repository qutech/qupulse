from typing import Union, Dict, Any
from numbers import Number

import sympy
import numpy


__all__ = ["sympify", "substitute_with_eval", "to_numpy"]


Sympifyable = Union[str, Number, sympy.Expr, numpy.str_]


class IndexedBasedFinder:
    def __init__(self):
        self.symbols = set()
        self.indexed_base = set()
        self.indices = set()

        class SubscriptionChecker(sympy.Symbol):
            def __getitem__(s, k):
                self.indexed_base.add(str(s))
                self.indices.add(k)
                return sympy.IndexedBase(str(s))[sympy.Symbol(str(k))]

        self.SubscriptionChecker = SubscriptionChecker

    def __getitem__(self, k) -> sympy.Expr:
        self.symbols.add(k)
        return self.SubscriptionChecker(k)

    def __contains__(self, k) -> bool:
        return True


def numpy_compatible_mul(*args) -> Union[sympy.Mul, sympy.Array]:
    if any(isinstance(a, sympy.NDimArray) for a in args):
        result = 1
        for a in args:
            result = result * (numpy.array(a.tolist()) if isinstance(a, sympy.NDimArray) else a)
        return sympy.Array(result)
    else:
        return sympy.Mul(*args)


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
        return sympy.sympify(expr, **kwargs)
    except TypeError as err:
        if err.args[0] == "'Symbol' object is not subscriptable":

            indexed_base = get_subscripted_symbols(expr)
            return sympy.sympify(expr, **kwargs, locals={k: sympy.IndexedBase(k) for k in indexed_base})

        else:
            raise


def substitute_with_eval(expression: sympy.Expr,
                         substitutions: Dict[str, Union[sympy.Expr, numpy.ndarray, str]]) -> sympy.Expr:
    """Substitutes only sympy.Symbols. Workaround for numpy like array behaviour. ~Factor 3 slower compared to subs"""
    substitutions = {k: v if isinstance(v, sympy.Expr) else sympify(v)
                     for k, v in substitutions.items()}

    for symbol in expression.free_symbols:
        symbol_name = str(symbol)
        if symbol_name not in substitutions:
            substitutions[symbol_name] = symbol

    string_representation = sympy.srepr(expression)
    return eval(string_representation, sympy.__dict__, {'Symbol': substitutions.__getitem__,
                                                        'Mul': numpy_compatible_mul})
