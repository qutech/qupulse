import unittest

from pulses.Parameter import ConstantParameter, ParameterDeclaration, NoDefaultValueException

class ParameterDeclarationTest(unittest.TestCase):

    def _test_helper(self, **kwargs) -> None:
        parameterDecl = ParameterDeclaration(**kwargs)
        if not 'min' in kwargs:
            kwargs['min'] = None
        if not 'max' in kwargs:
            kwargs['max'] = None
        if not 'default' in kwargs:
            kwargs['default'] = None
        self.assertEqual(kwargs['min'], parameterDecl.minValue)
        self.assertEqual(parameterDecl.minValue, parameterDecl.get_min_value())
        self.assertEqual(kwargs['max'], parameterDecl.maxValue)
        self.assertEqual(parameterDecl.maxValue, parameterDecl.get_max_value())
        self.assertEqual(kwargs['default'], parameterDecl.defaultValue)
        self.assertEqual(parameterDecl.defaultValue, parameterDecl.get_default_value())
        if kwargs['default'] is None:
            self.assertRaises(NoDefaultValueException, parameterDecl.get_default_parameter)
        else:
            self.assertIsInstance(parameterDecl.get_default_parameter(), ConstantParameter)
            self.assertEqual(kwargs['default'], parameterDecl.get_default_parameter().get_value())

    def test_no_values(self) -> None:
        self._test_helper()
        
    def test_valid_min_value(self) -> None:
        self._test_helper(min=4)
        
    def test_invalid_min_value(self) -> None:
        self.assertRaises(ValueError, ParameterDeclaration, min='foo')
        
    def test_valid_max_value(self) -> None:
        self._test_helper(max=4)
        
    def test_invalid_max_value(self) -> None:
        self.assertRaises(ValueError, ParameterDeclaration, max='foo')
        
    def test_valid_default_value(self) -> None:
        self._test_helper(default=4)
        
    def test_invalid_default_value(self) -> None:
        self.assertRaises(ValueError, ParameterDeclaration, default='foo')
    
