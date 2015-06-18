import unittest

from pulses.Parameter import ConstantParameter, ParameterDeclaration, NoDefaultValueException, TimeParameterDeclaration

class ConstantParameterTest(unittest.TestCase):

    def _test_valid_params(self, value: float) -> None:
        constantParameter = ConstantParameter(value)
        self.assertEqual(value, constantParameter.get_value())
        
    def test_float_values(self) -> None:
        self._test_valid_params(-0.3)
        self._test_valid_params(0)
        self._test_valid_params(0.3)
        
    # invalid arguments (non floats) should be excluded by type checks
        
    def test_requires_stop(self) -> None:
        constantParameter = ConstantParameter(0.3)
        self.assertFalse(constantParameter.requires_stop())

class ParameterDeclarationTest(unittest.TestCase):

    def _test_valid_params(self, **kwargs) -> None:
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
        self._test_valid_params()
        
    def test_valid_min_value(self) -> None:
        self._test_valid_params(min=0.1)
        self._test_valid_params(min=0)
        self._test_valid_params(min=-0.1)
        
    def test_invalid_min_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, min='foo')
        
    def test_valid_max_value(self) -> None:
        self._test_valid_params(max=0.1)
        self._test_valid_params(max=0)
        self._test_valid_params(max=-0.1)
        
    def test_invalid_max_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, max='foo')
        
    def test_valid_default_value(self) -> None:
        self._test_valid_params(default=0.1)
        self._test_valid_params(default=0)
        self._test_valid_params(default=-0.1)
        
    def test_invalid_default_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, default='foo')
        
    def test_valid_min_value_invalid_max_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, min=0.1, max='foo')
        self.assertRaises(ValueError, ParameterDeclaration, min=0.1, max=0.01)
        
    def test_valid_min_value_invalid_default_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, min=0.1, default='foo')
        self.assertRaises(ValueError, ParameterDeclaration, min=0.1, default=0.01)
    
    def test_valid_min_value_invalid_max_default_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, min=0.1, max='foo', default='foo')
        self.assertRaises(TypeError, ParameterDeclaration, min=0.1, max=0.01, default='foo')
        self.assertRaises(TypeError, ParameterDeclaration, min=0.1, max='foo', default=0.01)
        self.assertRaises(ValueError, ParameterDeclaration, min=0.1, max=0.02, default=0.01)
        
    def test_valid_max_value_invalid_min_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, max=0.1, min='foo')
        self.assertRaises(ValueError, ParameterDeclaration, max=0.1, min=0.11)
        
    def test_valid_max_value_invalid_default_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, max=0.1, default='foo')
        self.assertRaises(ValueError, ParameterDeclaration, max=0.1, default=0.11)
        
    def test_valid_max_value_invalid_min_default_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, max=0.1, min='foo', default='foo')
        self.assertRaises(TypeError, ParameterDeclaration, max=0.1, min=0.11, default='foo')
        self.assertRaises(TypeError, ParameterDeclaration, max=0.1, min='foo', default=0.11)
        self.assertRaises(ValueError, ParameterDeclaration, max=0.1, min=0.11, default=0.2)
        
    def test_valid_default_value_invalid_min_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, default=0.1, min='foo')
        self.assertRaises(ValueError, ParameterDeclaration, default=0.1, min=0.12)
        
    def test_valid_default_value_invalid_max_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, default=0.1, max='foo')
        self.assertRaises(ValueError, ParameterDeclaration, default=0.1, max=0.01)
        
    def test_valid_default_value_invalid_min_max_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, default=0.1, min='foo', max='foo')
        self.assertRaises(TypeError, ParameterDeclaration, default=0.1, min=0.11, max='foo')
        self.assertRaises(TypeError, ParameterDeclaration, default=0.1, min='foo', max=0.01)
        self.assertRaises(ValueError, ParameterDeclaration, default=0.1, min=0.11, max=0.01)
        
    def test_valid_min_max_value(self) -> None:
        self._test_valid_params(min=0.1, max=0.1)
        self._test_valid_params(min=0.1, max=0.15)
        self._test_valid_params(min=0, max=0.15)
        self._test_valid_params(min=-0.1, max=0.15)
        self._test_valid_params(min=-0.1, max=0)
        self._test_valid_params(min=-0.1, max=-0.09)
        
    def test_valid_min_max_value_invalid_default_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, min=0.1, max=0.15, default='foo')
        self.assertRaises(ValueError, ParameterDeclaration, min=0.1, max=0.15, default=0)
        self.assertRaises(ValueError, ParameterDeclaration, min=0.1, max=0.15, default=6)
        
    def test_valid_min_default_value(self) -> None:
        self._test_valid_params(min=0.1, default=0.1)
        self._test_valid_params(min=0.1, default=0.15)
        self._test_valid_params(min=0, default=0.15)
        self._test_valid_params(min=-0.1, default=0.15)
        self._test_valid_params(min=-0.1, default=0)
        self._test_valid_params(min=-0.1, default=-0.05)
        
    def test_valid_min_default_value_invalid_max_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, min=0.1, default=0.15, max='foo')
        self.assertRaises(ValueError, ParameterDeclaration, min=0.1, default=0.15, max=0.01)
        self.assertRaises(ValueError, ParameterDeclaration, min=0.1, default=0.15, max=0.12)
        
    def test_valid_default_max_value(self) -> None:
        self._test_valid_params(default=0.12, max=0.12)
        self._test_valid_params(default=0.12, max=0.15)
        self._test_valid_params(default=0, max=0.15)
        self._test_valid_params(default=-0.12, max=0.15)
        self._test_valid_params(default=-0.12, max=0)
        self._test_valid_params(default=-0.12, max=0.1)
        
    def test_valid_default_max_invalid_min_value(self) -> None:
        self.assertRaises(TypeError, ParameterDeclaration, default=0.11, max=0.15, min='foo');
        self.assertRaises(ValueError, ParameterDeclaration, default=0.11, max=0.15, min=0.16);
        self.assertRaises(ValueError, ParameterDeclaration, default=0.11, max=0.15, min=0.12);
        pass
        
    def test_valid_min_max_default_value(self) -> None:
        self._test_valid_params(min=0.11, default=0.12, max=0.13)
        self._test_valid_params(min=0, default=0.12, max=0.13)
        self._test_valid_params(min=-0.11, default=0.12, max=0.13)
        self._test_valid_params(min=-0.11, default=0, max=0.13)
        self._test_valid_params(min=-0.11, default=-0.10, max=0.13)
        self._test_valid_params(min=-0.11, default=-0.10, max=0)
        self._test_valid_params(min=-0.11, default=-0.10, max=-0.09)
        self._test_valid_params(min=0.3, default=0.3, max=0.3)
        
    def _test_is_parameter_valid(self, parameterValue: float, expectedResult: bool, **kwargs) -> None:
        parameter = ConstantParameter(parameterValue)
        parameterDecl = ParameterDeclaration(**kwargs)
        self.assertEqual(expectedResult, parameterDecl.is_parameter_valid(parameter))
        
    def test_no_values_is_parameter_valid(self) -> None:
        self._test_is_parameter_valid(0.1, True)
        self._test_is_parameter_valid(0, True)
        self._test_is_parameter_valid(-0.1, True)
        
    def test_min_value_is_parameter_valid(self) -> None:
        self._test_is_parameter_valid(0.2, True, min=0.1)
        self._test_is_parameter_valid(0.1, True, min=0.1)
        self._test_is_parameter_valid(-0.3, False, min=0.4)
        
    def test_max_value_is_parameter_valid(self) -> None:
        self._test_is_parameter_valid(-0.3, True, max=0.1)
        self._test_is_parameter_valid(0.1, True, max=0.1)
        self._test_is_parameter_valid(0.4, False, max=-1.9)
        
    def test_min_max_value_is_parameter_valid(self) -> None:
        self._test_is_parameter_valid(23.1, True, min=0.7, max=40.2)
        self._test_is_parameter_valid(0.7, True, min=0.7, max=40.2)
        self._test_is_parameter_valid(40.2, True, min=0.7, max=40.2)
        self._test_is_parameter_valid(-5.3, False, min=0.7, max=40.2)
        self._test_is_parameter_valid(0, False, min=-3.2, max=-2.6)

        
class TimeParameterDeclarationTest(unittest.TestCase):

    def _test_valid_params(self, **kwargs) -> None:
        parameterDecl = TimeParameterDeclaration(**kwargs)
        if not 'min' in kwargs:
            kwargs['min'] = 0
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

    def test_no_values(self):
        self._test_valid_params()
        
    def test_valid_min_value(self):
        self._test_valid_params(min=0)
        self._test_valid_params(min=3)
        
    def test_invalid_min_value(self):
        self.assertRaises(TypeError, TimeParameterDeclaration, min='foo')
        self.assertRaises(TypeError, TimeParameterDeclaration, min=0.6)
        self.assertRaises(ValueError, TimeParameterDeclaration, min=-1)
        
    def test_valid_max_value(self):
        self._test_valid_params(max=0)
        self._test_valid_params(max=1)
        
    def test_invalid_max_value(self):
        self.assertRaises(TypeError, TimeParameterDeclaration, max='foo')
        self.assertRaises(TypeError, TimeParameterDeclaration, max=2.7)
        self.assertRaises(ValueError, TimeParameterDeclaration, max=-3)
        
    def test_valid_default_value(self):
        self._test_valid_params(default=0)
        self._test_valid_params(default=5)
        
    def test_invalid_default_value(self):
        self.assertRaises(TypeError, TimeParameterDeclaration, default='foo')
        self.assertRaises(TypeError, TimeParameterDeclaration, default=3.8)
        self.assertRaises(ValueError, TimeParameterDeclaration, default=-5)
        
        
    def test_valid_min_invalid_max_value(self):
        self.assertRaises(TypeError, TimeParameterDeclaration, min=3, max='foo')
        self.assertRaises(TypeError, TimeParameterDeclaration, min=4, max=6.7)
        self.assertRaises(ValueError, TimeParameterDeclaration, min=1, max=0.6)
        self.assertRaises(ValueError, TimeParameterDeclaration, min=3, max=2)

    def test_valid_min_invalid_default_value(self):
        self.assertRaises(TypeError, TimeParameterDeclaration, min=3, default='foo')
        self.assertRaises(TypeError, TimeParameterDeclaration, min=4, default=6.7)
        self.assertRaises(ValueError, TimeParameterDeclaration, min=1, default=0.6)
        self.assertRaises(ValueError, TimeParameterDeclaration, min=3, default=2)
        
    def test_valid_max_invalid_min_value(self):
        self.assertRaises(TypeError, TimeParameterDeclaration, max=4, min='foo')
        self.assertRaises(TypeError, TimeParameterDeclaration, max=3, min=2.3)
        self.assertRaises(ValueError, TimeParameterDeclaration, max=1, min=3.7)
        self.assertRaises(ValueError, TimeParameterDeclaration, max=4, min=5)
        
    def test_valid_max_invalid_default_value(self):
        self.assertRaises(TypeError, TimeParameterDeclaration, max=4, default='foo')
        self.assertRaises(TypeError, TimeParameterDeclaration, max=3, default=2.3)
        self.assertRaises(ValueError, TimeParameterDeclaration, max=1, default=3.7)
        self.assertRaises(ValueError, TimeParameterDeclaration, max=4, default=5)
        
    def test_valid_min_max_value(self):
        self._test_valid_params(min=0, max=0)
        self._test_valid_params(min=0, max=5)
        self._test_valid_params(min=5, max=5)
        self._test_valid_params(min=6, max=11)
        
    def test_valid_min_max_invalid_default_value(self):
        self.assertRaises(TypeError, TimeParameterDeclaration, min=0, max=3, default='foo')
        self.assertRaises(TypeError, TimeParameterDeclaration, min=1, max=3, default=2.2)
        self.assertRaises(ValueError, TimeParameterDeclaration, min=1, max=3, default=0.1)
        self.assertRaises(ValueError, TimeParameterDeclaration, min=1, max=3, default=0)
        self.assertRaises(ValueError, TimeParameterDeclaration, min=1, max=3, default=4.7)
        self.assertRaises(ValueError, TimeParameterDeclaration, min=1, max=3, default=4)
        
    def test_valid_min_default_value(self):
        self._test_valid_params(min=0, default=0)
        self._test_valid_params(min=0, default=5)
        self._test_valid_params(min=5, default=5)
        self._test_valid_params(min=6, default=11)
        
    def test_valid_min_default_invalid_max_value(self):
        self.assertRaises(TypeError, TimeParameterDeclaration, min=0, default=3, max='foo')
        self.assertRaises(TypeError, TimeParameterDeclaration, min=1, default=3, max=4.2)
        self.assertRaises(ValueError, TimeParameterDeclaration, min=1, default=3, max=0.1)
        self.assertRaises(ValueError, TimeParameterDeclaration, min=1, default=3, max=0)
        self.assertRaises(ValueError, TimeParameterDeclaration, min=1, default=3, max=2.3)
        self.assertRaises(ValueError, TimeParameterDeclaration, min=1, default=3, max=2)
        
    def test_valid_default_max_value(self):
        self._test_valid_params(default=0, max=0)
        self._test_valid_params(default=0, max=5)
        self._test_valid_params(default=5, max=5)
        self._test_valid_params(default=6, max=11)
        
    def test_valid_default_max_invalid_min_value(self):
        self.assertRaises(TypeError, TimeParameterDeclaration, default=0, max=3, min='foo')
        self.assertRaises(TypeError, TimeParameterDeclaration, default=1, max=3, min=0.6)
        self.assertRaises(ValueError, TimeParameterDeclaration, default=1, max=3, min=9.3)
        self.assertRaises(ValueError, TimeParameterDeclaration, default=1, max=3, min=6)
        self.assertRaises(ValueError, TimeParameterDeclaration, default=1, max=3, min=2.3)
        self.assertRaises(ValueError, TimeParameterDeclaration, default=1, max=3, min=2)
        
    def test_valid_min_default_max_value(self):
        self._test_valid_params(min=0, default=0, max=0)
        self._test_valid_params(min=0, default=0, max=7)
        self._test_valid_params(min=0, default=3, max=7)
        self._test_valid_params(min=3, default=3, max=7)
        self._test_valid_params(min=3, default=7, max=7)
        self._test_valid_params(min=7, default=7, max=7)
        
    def _test_is_parameter_valid(self, parameterValue: float, expectedResult: bool, **kwargs) -> None:
        parameter = ConstantParameter(parameterValue)
        parameterDecl = TimeParameterDeclaration(**kwargs)
        self.assertEqual(expectedResult, parameterDecl.is_parameter_valid(parameter))
        
    def test_no_values_is_parameter_valid(self) -> None:
        self._test_is_parameter_valid(0.1, False)
        self._test_is_parameter_valid(-0.1, False)
        self._test_is_parameter_valid(-2, False)
        self._test_is_parameter_valid(0, True)
        self._test_is_parameter_valid(1, True)
        
    def test_min_value_is_parameter_valid(self) -> None:
        self._test_is_parameter_valid(0.2, False, min=1)
        self._test_is_parameter_valid(1.1, False, min=1)
        self._test_is_parameter_valid(1, True, min=1)
        self._test_is_parameter_valid(3, True, min=2)
        
    def test_max_value_is_parameter_valid(self) -> None:
        self._test_is_parameter_valid(0.3, False, max=4)
        self._test_is_parameter_valid(9.8, False, max=3)
        self._test_is_parameter_valid(3, True, max=3)
        self._test_is_parameter_valid(5, True, max=8)
        
    def test_min_max_value_is_parameter_valid(self) -> None:
        self._test_is_parameter_valid(0.1, False, min=4, max=23)
        self._test_is_parameter_valid(8.3, False, min=4, max=23)
        self._test_is_parameter_valid(28.9, False, min=4, max=23)
        self._test_is_parameter_valid(3, False, min=4, max=23)
        self._test_is_parameter_valid(40, False, min=4, max=23)
        self._test_is_parameter_valid(4, True, min=4, max=23)
        self._test_is_parameter_valid(17, True, min=4, max=23)
        self._test_is_parameter_valid(23, True, min=4, max=23)
        