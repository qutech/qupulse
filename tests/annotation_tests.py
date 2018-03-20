import ast
import os
import pathlib
import functools
import unittest


src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qctoolkit')

ignored_functions = ["__init__", "__new__", "__str__", "__repr__", "__hash__", "__len__", "__eq__", "__iter__",
                     "__float__", "__int__", "__bool__"]


def assert_function_annotations(test_case: unittest.TestCase, func: ast.FunctionDef, name: str):
    if func.name in ignored_functions:
        return

    if any('abstract' in decorator.id for decorator in func.decorator_list if isinstance(decorator, ast.Name)):
        return

    name = name + '.' + func.name

    def find_return_statement(body):
        for statement in body:
            if isinstance(statement, ast.Return):
                yield statement
            elif isinstance(statement, ast.If):
                yield from find_return_statement(statement.body)
                yield from find_return_statement(statement.orelse)
            elif isinstance(statement, (ast.For, ast.ExceptHandler, ast.With, ast.While)):
                yield from find_return_statement(statement.body)
            elif isinstance(statement, ast.Try):
                yield from find_return_statement(ast.iter_child_nodes(statement))
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, (ast.Yield, ast.YieldFrom)):
                yield statement.value

    def find_uncached_raise(body, catched=list()):
        for statement in body:
            if isinstance(statement, ast.If):
                yield from find_uncached_raise(statement.body, catched)
                yield from find_uncached_raise(statement.orelse, catched)
            elif isinstance(statement, (ast.For, ast.ExceptHandler, ast.With, ast.While)):
                yield from find_uncached_raise(statement.body, catched)
            elif isinstance(statement, ast.Try):
                yield find_uncached_raise(statement.body, catched + [handler.type.id for handler in statement.handlers])
                yield from find_uncached_raise(statement.handlers, catched)
                yield from find_uncached_raise(statement.orelse, catched)
                yield from find_uncached_raise(statement.finalbody, catched)
            elif isinstance(statement, ast.Raise):
                if isinstance(statement.exc, ast.Call) and statement.exc.func.id in catched:
                    continue
                else:
                    yield statement

    function_returns = any(find_return_statement(func.body))

    if function_returns:
        test_case.assertIsNotNone(func.returns, "Missing return annotation in function {}".format(name))
    else:
        if func.returns is None:
            return
        elif isinstance(func.returns, ast.NameConstant):
            return
        else:
            if any(find_uncached_raise(func.body)):
                return
            raise AssertionError("Return annotation set but no return statement in function {}".format(name))


def assert_class_annotations(test_case: unittest.TestCase, cls: ast.ClassDef, name: str):
    name = name + '.' + cls.name

    for declaration in cls.body:
        if isinstance(declaration, ast.FunctionDef) and declaration.name not in ignored_functions:
            assert_function_annotations(test_case, declaration, name)


def assert_module_annotations(test_case: unittest.TestCase, module_file, name: str):
    with open(str(module_file), 'r') as file_handle:
        src = file_handle.read()

    module_tree = ast.parse(src)

    for declaration in module_tree.body:
        if isinstance(declaration, ast.ClassDef):
            assert_class_annotations(test_case, declaration, name)
        elif isinstance(declaration, ast.FunctionDef):
            assert_function_annotations(test_case, declaration, name)


class AnnotationMetaClass(type):
    def __new__(mcs, name, bases, dct, modules):
        def make_test_method(m_file, m_name, f_name):
            def test_annotation(self):
                assert_module_annotations(self, module_file=m_file, name=m_name)
            test_annotation.__name__ = f_name
            return test_annotation

        name_templ = 'test_{}_annotations'
        for module_file in modules:
            rel_module_file = str(module_file.relative_to(src_path))
            module_name = rel_module_file.rstrip('.py').replace(os.path.sep, '.')
            method_name = name_templ.format(module_name.replace('.', '_'))

            dct[method_name] = make_test_method(module_file, module_name, method_name)
        return type.__new__(mcs, name, bases, dct)

    def __init__(self, *args, modules, **kwargs):
        type.__init__(self, *args, **kwargs)


class AnnotationsTests(unittest.TestCase,
                       metaclass=AnnotationMetaClass,
                       modules=pathlib.Path(src_path).glob('**/*.py')):
    def setUp(self):
        pass
