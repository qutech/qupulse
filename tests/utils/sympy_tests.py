import unittest
import contextlib
import math
import sys

from typing import Union

import sympy
import numpy as np

from sympy.abc import a, b, c, d, e, f, k, l, m, n, i, j, x, y, z
from sympy import sin, Sum, IndexedBase

from qupulse.utils.sympy import sympify as qc_sympify, substitute_with_eval, recursive_substitution, Len,\
    evaluate_lambdified, evaluate_compiled, get_most_simple_representation, get_variables, get_free_symbols,\
    almost_equal, SymbolNamespace, NamespacedSymbol, NamespaceIndexedBase, subs_namespaces, get_root_level_namespace_ids

a_ = IndexedBase(a)
b_ = IndexedBase(b)
foo_bar = NamespacedSymbol('bar', namespace=SymbolNamespace('foo'))
foo_bar_ = NamespaceIndexedBase('bar', namespace=SymbolNamespace('foo'))
scope_n = NamespacedSymbol('n', namespace=SymbolNamespace('scope'))


################################################### SUBSTITUTION #######################################################
simple_substitution_cases = [
    (a*b, {'a': c}, b*c),
    (a*b, {'a': b, 'b': a}, a*b),
    (a*b, {'a': 1, 'b': 2}, 2),
    (foo_bar*b, {'NS(foo).bar': c}, b*c),
    (foo_bar*b, {'b': scope_n}, scope_n*foo_bar),
    (foo_bar*b, {'b': foo_bar, 'NS(foo).bar': b}, foo_bar*b),
    (foo_bar*scope_n, {'NS(foo).bar': scope_n, 'NS(scope).n': foo_bar}, foo_bar*scope_n),
    (foo_bar*b, {'NS(foo).bar': 5, 'b': 0.2}, 0.2*5)
]

elem_func_substitution_cases = [
    (a*b + sin(c), {'a': b, 'c': sympy.pi/2}, b**2 + 1),
    (a*scope_n + sin(foo_bar), {'a': scope_n, 'NS(foo).bar': sympy.pi/2}, scope_n**2 + 1)
]

sum_substitution_cases = [
    (a*b + Sum(c * k, (k, 0, n)), {'a': b, 'b': 2, 'k': 1, 'n': 2}, b*2 + c*(1 + 2)),
    (a*foo_bar + Sum(c * k, (k, 0, scope_n)), {'a': foo_bar, 'NS(foo).bar': 2, 'k': 1, 'NS(scope).n': 2}, foo_bar*2 + c*(1 + 2)),
]

indexed_substitution_cases = [
    (a_[i] * b, {'b': 3}, a_[i] * 3),
    (a_[i] * b, {'a': sympy.Array([1, 2, 3])}, sympy.Array([1, 2, 3])[i] * b),
    (sympy.Array([1, 2, 3])[i] * b, {'i': 1}, 2 * b),
    (foo_bar_[i]*scope_n, {'NS(scope).n': 3}, foo_bar_[i]*3),
    (foo_bar_[i]*scope_n, {'NS(foo).bar': sympy.Array([1, 2, 3])}, sympy.Array([1, 2, 3])[i]*scope_n),
    (sympy.Array([1, 2, 3])[foo_bar]*b, {'NS(foo).bar': 1}, 2*b),
]

vector_valued_cases = [
    (a*b, {'a': sympy.Array([1, 2, 3])}, sympy.Array([1, 2, 3])*b),
    (a*b, {'a': sympy.Array([1, 2, 3]), 'b': sympy.Array([4, 5, 6])}, sympy.Array([4, 10, 18])),
    (foo_bar*b, {'NS(foo).bar': sympy.Array([1, 2, 3])}, sympy.Array([1, 2, 3])*b),
    (foo_bar*b, {'NS(foo).bar': sympy.Array([1, 2, 3]), 'b': sympy.Array([4, 5, 6])}, sympy.Array([4, 10, 18])),
    (foo_bar*scope_n, {'NS(foo).bar': sympy.Array([1, 2, 3]), 'NS(scope).n': sympy.Array([4, 5, 6])}, sympy.Array([4, 10, 18])),
]

full_featured_cases = [
    (Sum(a_[i], (i, 0, Len(a) - 1)), {'a': sympy.Array([1, 2, 3])}, 6),
    (Sum(foo_bar_[i], (i, 0, Len(foo_bar) - 1)), {'NS(foo).bar': sympy.Array([1, 2, 3])}, 6),
]


##################################################### SYMPIFY ##########################################################
simple_sympify = [
    ('a*b', a*b),
    ('a*6', a*6),
    ('sin(a)', sin(a)),
    ('SymbolNamespace(\'foo\').bar*6', foo_bar*6),
    ('sin(SymbolNamespace(\'foo\').bar)', sin(foo_bar))
]

complex_sympify = [
    ('Sum(a, (i, 0, n))', Sum(a, (i, 0, n))),
    ('Sum(NS(foo).bar, (i, 0, m))', Sum(foo_bar, (i, 0, m)))
]

len_sympify = [
    ('len(a)', Len(a)),
    ('Len(a)', Len(a))
]

index_sympify = [
    ('a[i]', a_[i]),
    ('SymbolNamespace(foo).bar[i]', foo_bar_[i])
]


#################################################### EVALUATION ########################################################
eval_simple = [
    (a*b, {'a': 2, 'b': 3}, 6),
    (a*b, {'a': 2, 'b': np.float32(3.5)}, 2*np.float32(3.5)),
    (a+b, {'a': 3.4, 'b': 76.7}, 3.4+76.7),
    (foo_bar+scope_n, {'foo': {'bar': 1.2}, 'scope': {'n': 3.3}}, 1.2+3.3),
    (foo_bar*scope_n, {'foo': {'bar': 1.2}, 'scope': {'n': np.float32(3.3)}}, 1.2*np.float32(3.3)),
    (foo_bar*scope_n, {'foo': {'bar': 1.2}, 'scope': {'n': 3.3}}, 1.2*3.3)
]

eval_many_arguments = [
    (sum(sympy.symbols(list('a_' + str(i) for i in range(300)))), {'a_' + str(i): 1 for i in range(300)}, 300),
    (sum(list(NamespacedSymbol('a_' + str(i), namespace=SymbolNamespace('scope')) for i in range(300))), {'scope' : {'a_' + str(i): 1 for i in range(300)}}, 300)
]

eval_simple_functions = [
    (a*sin(b), {'a': 3.5, 'b': 1.2}, 3.5*math.sin(1.2)),
    (a*sin(foo_bar), {'a': 3.5, 'foo': {'bar': 1.2}}, 3.5*math.sin(1.2)),
]

eval_array_values = [
    (a * b, {'a': 2, 'b': np.array([3])}, np.array([6])),
    (a * b, {'a': 2, 'b': np.array([3, 4, 5])}, np.array([6, 8, 10])),
    (a * b, {'a': np.array([2, 3]), 'b': np.array([100, 200])}, np.array([200, 600])),
    (a * foo_bar, {'a': 2, 'foo': {'bar': np.array([3])}}, np.array([6])),
    (a * foo_bar, {'a': 2, 'foo': {'bar': np.array([3, 4, 5])}}, np.array([6, 8, 10])),
    (a * foo_bar, {'a': np.array([2, 3]), 'foo': {'bar': np.array([100, 200])}}, np.array([200, 600])),
]

eval_sum = [
    (Sum(a_[i], (i, 0, Len(a) - 1)), {'a': np.array([1, 2, 3])}, 6),
    (Sum(foo_bar_[i], (i, 0, Len(foo_bar) - 1)), {'foo': {'bar': np.array([1, 2, 3])}}, 6),
]

eval_array_expression = [
    (np.array([a*c, b*c]), {'a': 2, 'b': 3, 'c': 4}, np.array([8, 12])),
    (np.array([a*foo_bar, scope_n*foo_bar]), {'a': 2, 'scope': {'n': 3}, 'foo': {'bar': 4}}, np.array([8, 12]))
]


class TestCase(unittest.TestCase):
    def assertRaises(self, expected_exception, *args, **kwargs):
        if expected_exception is None:
            return contextlib.suppress()
        else:
            return super().assertRaises(expected_exception, *args, **kwargs)


class SympifyTests(TestCase):

    def sympify(self, expression) -> sympy.Expr:
        waa = qc_sympify(expression)
        return waa

    def assertEqual1(self, first, second, msg=None):
        if sympy.Eq(first, second):
            return
        raise self.failureException(msg=msg)

    def test_simple_sympify(self) -> None:
        for s, expected in simple_sympify:
            result = self.sympify(s)
            self.assertEqual(expected, result)

    def test_complex_sympify(self) -> None:
        for s, expected in complex_sympify:
            result = self.sympify(s)
            self.assertEqual(expected, result)

    def test_len_sympify(self) -> None:
        for s, expected in len_sympify:
            result = self.sympify(s)
            self.assertEqual(expected, result)

    def test_index_sympify(self) -> None:
        for s, expected in index_sympify:
            result = self.sympify(s)
            self.assertEqual(expected, result)


class SubstitutionTests(TestCase):
    def substitute(self, expression: sympy.Expr, substitutions: dict):
        for key, value in substitutions.items():
            if not isinstance(value, sympy.Expr):
                substitutions[key] = qc_sympify(value)
                #substitutions[key] = sympy.sympify(value)
        return expression.subs(substitutions, simultaneous=True).doit()

    def test_simple_substitution_cases(self):
        for expr, subs, expected in simple_substitution_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(expected, result, msg=str((expr, subs, expected)))

    def test_elem_func_substitution_cases(self):
        for expr, subs, expected in elem_func_substitution_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(expected, result)

    def test_sum_substitution_cases(self):
        for expr, subs, expected in sum_substitution_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(expected, result)

    def test_indexed_substitution_cases(self):
        if type(self) is SubstitutionTests:
            raise unittest.SkipTest('sympy.Expr.subs does not handle simultaneous substitutions of indexed entities.')

        for expr, subs, expected in indexed_substitution_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(expected, result)

    def test_vector_valued_cases(self):
        if type(self) is SubstitutionTests:
            raise unittest.SkipTest('sympy.Expr.subs does not handle simultaneous substitutions of indexed entities.')

        for expr, subs, expected in vector_valued_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(expected, result, msg="test: {}".format((expr, subs, expected)))

    def test_full_featured_cases(self):
        if type(self) is SubstitutionTests:
            raise unittest.SkipTest('sympy.Expr.subs does not handle simultaneous substitutions of indexed entities.')

        for expr, subs, expected in full_featured_cases:
            result = self.substitute(expr, subs)
            self.assertEqual(expected, result)

@unittest.SkipTest # substitute_with_eval currently not implemented"
class SubstituteWithEvalTests(SubstitutionTests):
    def substitute(self, expression: sympy.Expr, substitutions: dict):
        return substitute_with_eval(expression, substitutions)

    @unittest.expectedFailure
    def test_sum_substitution_cases(self):
        super().test_sum_substitution_cases()

    @unittest.expectedFailure
    def test_full_featured_cases(self):
        super().test_full_featured_cases()


class RecursiveSubstitutionTests(SubstitutionTests):
    def substitute(self, expression: sympy.Expr, substitutions: dict):
        return recursive_substitution(expression, substitutions)#.doit()

    def test_array_argument(self) -> None:
        expr = np.array([a * c, b * c])
        substitutions = {'a': x, 'b': a, 'c': y}
        expected = np.array([x*y, a*y])
        result = self.substitute(expr, substitutions)
        np.testing.assert_equal(expected, result)

        expr = np.array([a * foo_bar, scope_n * foo_bar])
        substitutions = {'a': x, 'NS(scope).n': a, 'NS(foo).bar': scope_n}
        expected = np.array([x * scope_n, a * scope_n])
        result = self.substitute(expr, substitutions)
        np.testing.assert_equal(expected, result)

    def test_nested_array_argument(self) -> None:
        expr = np.array([[a * c, b * c], [a * foo_bar, scope_n * foo_bar]])
        substitutions = {'a': x, 'b': a, 'c': y, 'NS(scope).n': a, 'NS(foo).bar': scope_n}
        expected = np.array([[x * y, a * y], [x * scope_n, a * scope_n]])
        result = self.substitute(expr, substitutions)
        np.testing.assert_equal(expected, result)


class GetFreeSymbolsTests(TestCase):
    def assert_symbol_sets_equal(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(set(expected), set(actual))

    def test_get_free_symbols(self):
        expr = a * b / 5
        self.assert_symbol_sets_equal([a, b], get_free_symbols(expr))

    def test_get_free_symbols_indexed(self):
        expr = a_[i] * IndexedBase(a*b)[j]
        self.assert_symbol_sets_equal({a, b, i, j}, set(get_free_symbols(expr)))

    def test_get_variables(self):
        expr = a * b / 5
        self.assertEqual({'a', 'b'}, set(get_variables(expr)))

    def test_get_variables_indexed(self):
        expr = a_[i] * IndexedBase(a*b)[j]
        self.assertEqual({'a', 'b', 'i', 'j'}, set(get_variables(expr)))


class EvaluationTestsBase:

    def test_eval_simple(self):
        for expr, parameters, expected in eval_simple:
            result = self.evaluate(expr, parameters)
            self.assertEqual(expected, result)

    def test_eval_many_arguments(self):
        for expr, parameters, expected in eval_many_arguments:
            result = self.evaluate(expr, parameters)
            self.assertEqual(expected, result)

    def test_eval_simple_functions(self):
        for expr, parameters, expected in eval_simple_functions:
            result = self.evaluate(expr, parameters)
            self.assertEqual(expected, result)

    def test_eval_array_values(self):
        for expr, parameters, expected in eval_array_values:
            result = self.evaluate(expr, parameters)
            np.testing.assert_equal(expected, result)

    def test_eval_sum(self):
        for expr, parameters, expected in eval_sum:
            result = self.evaluate(expr, parameters)
            self.assertEqual(expected, result)

    def test_eval_array_expression(self):
        for expr, parameters, expected in eval_array_expression:
            result = self.evaluate(expr, parameters)
            np.testing.assert_equal(expected, result)


class EvaluationTests(EvaluationTestsBase, unittest.TestCase):

    def evaluate(self, expression: Union[sympy.Expr, np.ndarray], parameters):
        if isinstance(expression, np.ndarray):
            return np.array(expr.evalf(subs=parameters) for expr in expression.flat)
        return expression.evalf(subs=parameters)

    @unittest.expectedFailure
    def test_eval_array_expression(self):
        super().test_eval_array_expression()

    @unittest.expectedFailure
    def test_eval_array_values(self):
        super().test_eval_array_values()

    @unittest.expectedFailure
    def test_eval_sum(self):
        super().test_eval_sum()


class LamdifiedEvaluationTest(EvaluationTestsBase, unittest.TestCase):

    def evaluate(self, expression: Union[sympy.Expr, np.ndarray], parameters):
        if isinstance(expression, np.ndarray):
            variables = set.union(*[get_variables(e).union(get_root_level_namespace_ids(e)) for e in expression.flat])
        else:
            variables = get_variables(expression).union(get_root_level_namespace_ids(expression))
        print(variables)
        return evaluate_lambdified(expression, variables=list(variables), parameters=parameters, lambdified=None)[0]

    @unittest.skipIf(sys.version_info[0] == 3 and sys.version_info[1] < 7, "causes syntax error for python < 3.7")
    def test_eval_many_arguments(self):
        super().test_eval_many_arguments()

@unittest.SkipTest
class CompiledEvaluationTest(EvaluationTestsBase, unittest.TestCase):

    def evaluate(self, expression: Union[sympy.Expr, np.ndarray], parameters):
        if isinstance(expression, np.ndarray):
            return self.evaluate(sympy.Array(expression), parameters)

        result, _ = evaluate_compiled(expression, parameters, compiled=None)

        if isinstance(result, (list, tuple)):
            return np.array(result)
        else:
            return result

    def test_eval_many_arguments(self):
        super().test_eval_many_arguments()


class RepresentationTest(unittest.TestCase):
    def test_get_most_simple_representation(self):
        cpl = get_most_simple_representation(qc_sympify('1 + 1j'))
        self.assertIsInstance(cpl, str)
        self.assertTrue(bool(sympy.Eq(sympy.sympify(cpl), 1 + 1j)))

        integer = get_most_simple_representation(qc_sympify('3'))
        self.assertIsInstance(integer, int)
        self.assertEqual(integer, 3)

        flt = get_most_simple_representation(qc_sympify('3.1'))
        self.assertIsInstance(flt, float)
        self.assertEqual(flt, 3.1)

        st = get_most_simple_representation(qc_sympify('a + b'))
        self.assertIsInstance(st, str)
        self.assertEqual(st, 'a + b')

        sym = get_most_simple_representation(qc_sympify('b + NS(foo).NS(bar).test'))
        self.assertIsInstance(sym, str)
        self.assertEqual('NS(foo).NS(bar).test + b', sym)


class AlmostEqualTests(unittest.TestCase):
    def test_almost_equal(self):
        self.assertTrue(almost_equal(sympy.sin(a) * 0.5, sympy.sin(a) / 2))
        self.assertIsNone(almost_equal(sympy.sin(a) * 0.5, sympy.sin(b) / 2))
        self.assertFalse(almost_equal(sympy.sin(a), sympy.sin(a) + 1e-14))

        self.assertTrue(almost_equal(sympy.sin(a), sympy.sin(a) + 1e-14, epsilon=1e-13))


class TestSubsNamespaces(unittest.TestCase):

    def test_subs_namespaces(self) -> None:
        expr = SymbolNamespace('foo').bar * a
        result = subs_namespaces(expr, {SymbolNamespace('foo'): SymbolNamespace('oof')})
        self.assertEqual(SymbolNamespace('oof').bar * a, result)
        self.assertEqual(SymbolNamespace('foo').bar * a, expr) # ensure original wasnt changed

    def test_subs_namespaces_nested(self) -> None:
        expr = SymbolNamespace('foo').SymbolNamespace('bar').x * a + SymbolNamespace('bar').x

        # change only NS(foo)
        result = subs_namespaces(expr, {SymbolNamespace('foo'): SymbolNamespace('oof')})
        self.assertEqual(SymbolNamespace('oof').SymbolNamespace('bar').x * a + SymbolNamespace('bar').x, result)
        self.assertEqual(SymbolNamespace('foo').SymbolNamespace('bar').x * a + SymbolNamespace('bar').x, expr)

        # change unnested NS(bar)
        result = subs_namespaces(expr, {SymbolNamespace('bar'): SymbolNamespace('rab')})
        self.assertEqual(SymbolNamespace('foo').SymbolNamespace('bar').x * a + SymbolNamespace('rab').x, result)
        self.assertEqual(SymbolNamespace('foo').SymbolNamespace('bar').x * a + SymbolNamespace('bar').x, expr)  # ensure original wasnt changed

        # change nested NS(foo).NS(bar)
        result = subs_namespaces(expr, {SymbolNamespace('foo').SymbolNamespace('bar'): SymbolNamespace('bar')})
        self.assertEqual(SymbolNamespace('bar').x * a + SymbolNamespace('bar').x, result)
        self.assertEqual(SymbolNamespace('foo').SymbolNamespace('bar').x * a + SymbolNamespace('bar').x, expr)  # ensure original wasnt changed


class NamespaceSymbolTests(unittest.TestCase):

    def test_stringify_and_back(self) -> None:
        sym = NamespacedSymbol('bar', namespace=SymbolNamespace('foo', parent=SymbolNamespace('oof')))
        sym_str = str(sym)
        self.assertEqual("NS(oof).NS(foo).bar", sym_str)
        sym_sympified = qc_sympify(sym_str)
        self.assertEqual(sym, sym_sympified)

    def test_atoms(self) -> None:
        expr = SymbolNamespace('foo').bar * 3 + SymbolNamespace('foo').c
        self.assertEqual({SymbolNamespace('foo').bar, SymbolNamespace('foo').c}, expr.atoms(NamespacedSymbol))

    def test_subs(self) -> None:
        expr = SymbolNamespace('foo').bar * a
        result = expr.subs({'NS(foo).bar': a})
        self.assertEqual(a*a, result)

        result = expr.subs({SymbolNamespace('foo').bar: a})
        self.assertEqual(a * a, result)

        result = expr.subs({NamespacedSymbol('bar', namespace=SymbolNamespace('foo')): a})
        self.assertEqual(a * a, result)

    def test_subs_nested(self) -> None:
        expr = SymbolNamespace('foo').SymbolNamespace('off').bar * a
        result = expr.subs({'NS(foo).NS(off).bar': a})
        self.assertEqual(a * a, result)

    @unittest.expectedFailure
    def test_subs_do_not_change_namespaces(self) -> None:
        expr = SymbolNamespace('foo').bar * a
        result = expr.subs({'NS(foo)': 'NS(oof)'})
        self.assertEqual(SymbolNamespace('oof').bar * a, result)


class NamespaceTests(unittest.TestCase):

    def test_illegal_name(self) -> None:
        with self.assertRaises(AttributeError):
            qc_sympify("qubit._forbidden")

    def test_sympify_dot_namespace_notations(self) -> None:
        expr = qc_sympify("NS('qubit').a + NS('qubit').NS('spec2').a * 1.3")
        expected = sympy.Add(NamespacedSymbol('a', namespace=SymbolNamespace(sympy.Symbol('qubit'))),
                             sympy.Mul(NamespacedSymbol('a',
                                                        namespace=SymbolNamespace(sympy.Symbol('spec2'),
                                                                                     parent=SymbolNamespace(sympy.Symbol('qubit'))))
                                       , sympy.RealNumber(1.3)))
        self.assertEqual(expected, expr)

    def test_sympify_indexed_dot_namespace_notation(self) -> None:
        expr = qc_sympify("NS(qubit).a[i]*1.3")
        expected = sympy.Mul(NamespaceIndexedBase('a', namespace=SymbolNamespace('qubit'))[sympy.Symbol('i')], sympy.RealNumber(1.3))
        self.assertEqual(expected, expr)

    def test_sympify_dot_namespace_notation_as_index(self) -> None:
        expr = qc_sympify("NS(qubit).a[NS(index_ns).i]*1.3")
        expected = sympy.Mul(NamespaceIndexedBase('a', namespace=SymbolNamespace('qubit'))[NamespacedSymbol('i', namespace=SymbolNamespace('index_ns'))], sympy.RealNumber(1.3))
        self.assertEqual(expected, expr)

    def test_vanilla_sympify_compatability(self) -> None:
        # taken from sympy test suite: sympy/parsing/tests/test_sympy_parser.py::test_sympy_parser()
        x = sympy.Symbol("x")
        inputs = {
            '2*x': 2 * x,
            '3.00': sympy.Float(3),
            '22/7': sympy.Rational(22, 7),
            '2+3j': 2 + 3 * sympy.I,
            'exp(x)': sympy.exp(x),
            'x!': sympy.factorial(x),
            'x!!': sympy.factorial2(x),
            '(x + 1)! - 1': sympy.factorial(x + 1) - 1,
            '3.[3]': sympy.Rational(10, 3),
            '.0[3]': sympy.Rational(1, 30),
            '3.2[3]': sympy.Rational(97, 30),
            '1.3[12]': sympy.Rational(433, 330),
            '1 + 3.[3]': sympy.Rational(13, 3),
            '1 + .0[3]': sympy.Rational(31, 30),
            '1 + 3.2[3]': sympy.Rational(127, 30),
            '.[0011]': sympy.Rational(1, 909),
            '0.1[00102] + 1': sympy.Rational(366697, 333330),
            '1.[0191]': sympy.Rational(10190, 9999),
            '10!': 3628800,
            '-(2)': -sympy.Integer(2),
            '[-1, -2, 3]': [sympy.Integer(-1), sympy.Integer(-2), sympy.Integer(3)],
            'Symbol("x").free_symbols': x.free_symbols,
            "S('S(3).n(n=3)')": 3.00,
            'factorint(12, visual=True)': sympy.Mul(sympy.Pow(2, 2, evaluate=False), sympy.Pow(3, 1, evaluate=False), evaluate=False),
            'Limit(sin(x), x, 0, dir="-")': sympy.Limit(sin(x), x, 0, dir='-')
        }
        for text, result in inputs.items():
            self.assertEqual(qc_sympify(text), result, msg="failed for {}".format(text))

    def test_evaluate_lambdified_dot_namespace_notation(self) -> None:
        res, _ = evaluate_lambdified(
            qc_sympify("NS(qubit).a + NS(qubit).NS(spec2).a * 1.3"),
            ["NS(qubit).a", "NS(qubit).NS(spec2).a"],
            {"NS(qubit).a": 2.1, "NS(qubit).NS(spec2).a": .1},
            lambdified=None
        )
        self.assertEqual(2.23, res)

    @unittest.expectedFailure
    def test_evaluate_compiled_dot_namespace_notation(self) -> None:
        res = evaluate_compiled("NS(qubit).a + NS('qubit.spec2').a * 1.3", {"qubit.a": 2.1, "qubit.spec2.a": .1})
        self.assertEqual(2.23, res)
