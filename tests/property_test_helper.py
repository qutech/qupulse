import inspect
import unittest

class assert_all_properties_tested:
    def __init__(self, to_test: type):
        self.expected_tests = []
        base_class_members = [member
                              for baseclass in inspect.getmro(to_test)[1:]
                              for member in inspect.getmembers(baseclass)]
        for name, member in inspect.getmembers(to_test):
            if name.startswith('_'):
                continue
            if (name, member) in base_class_members:
                continue
            if inspect.isdatadescriptor(member):
                self.expected_tests.append((name, 'test_' + name))

    def __call__(self, tester_type: type):
        def test_all_properties_tested(tester: unittest.TestCase):
            not_tested = [member_name
                          for member_name, test_name in self.expected_tests if not hasattr(tester, test_name)]
            tester.assertFalse(not_tested, 'Missing property test for {}'.format(not_tested))

        tester_type.test_all_properties_tested = test_all_properties_tested
        return tester_type


def assert_public_functions_tested_tester(testing_module, tested_module):
    def get_public_functions(cls):
        for name, member in inspect.getmembers(cls):
            if inspect.getmodule(member) is tested_module:
                if not name.startswith('_'):
                    if inspect.isfunction(member):
                        yield name
                    elif inspect.isclass(member):
                        yield from get_public_functions(member)
    to_test = list(get_public_functions(tested_module))

    def get_tests(cls):
        for name, member in inspect.getmembers(cls):
            if not name.startswith('_') and inspect.getmodule(member) is testing_module:
                if inspect.isfunction(member) and name.startswith('test_'):
                    yield name
                elif inspect.isclass(member) and issubclass(member, unittest.TestCase):
                    yield from get_tests(member)
    testing_functions = list(get_tests(testing_module))

    class PublicFunctionTest(unittest.TestCase):
        def test_all_public_functions_tested(self):
            non_tested_functions = [name for name in to_test if not any(testing_function.startswith('test_' + name)
                                                                        for testing_function in testing_functions)]
            self.assertFalse(non_tested_functions, "{} has no tests for {}".format(testing_module.__name__,
                                                                                   non_tested_functions))

    return PublicFunctionTest
