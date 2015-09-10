import unittest
import os
import sys
import numbers
from typing import Union, Optional

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from pulses.Parameter import Parameter, ConstantParameter, ParameterDeclaration, ParameterNotProvidedException

class DummyParameter(Parameter):
        
        def __init__(self) -> None:
            self.value = 252.13
            
        def get_value(self) -> float:
            return self.value
        
        @property
        def requires_stop(self) -> bool:
            return False
    

class ParameterTest(unittest.TestCase):
    
    def test_float_conversion_method(self) -> None:
        parameter = DummyParameter()
        self.assertEqual(parameter.value, float(parameter))
    
class ConstantParameterTest(unittest.TestCase):
    
    def test_float_is_constant_parameter(self) -> None:
        self.assertTrue(issubclass(numbers.Real, ConstantParameter))
        self.assertTrue(isinstance(0.1, ConstantParameter))

    def __test_valid_params(self, value: float) -> None:
        constant_parameter = ConstantParameter(value)
        self.assertEqual(value, constant_parameter.get_value())
        self.assertEqual(value, float(constant_parameter))
        
    def test_float_values(self) -> None:
        self.__test_valid_params(-0.3)
        self.__test_valid_params(0)
        self.__test_valid_params(0.3)
        
    # invalid arguments (non floats) should be excluded by type checks
        
    def test_requires_stop(self) -> None:
        constant_parameter = ConstantParameter(0.3)
        self.assertFalse(constant_parameter.requires_stop)
    
class ParameterDeclarationTest(unittest.TestCase):
    
    def __test_valid_values(self, **kwargs):
        expected_values = {'min': float('-inf'), 'max': float('+inf'), 'default': None}
        for k in kwargs:
            expected_values[k] = kwargs[k]
            
        expected_values['absolute_min'] = expected_values['min']
        expected_values['absolute_max'] = expected_values['max']
            
        if isinstance(expected_values['absolute_min'], ParameterDeclaration):
            expected_values['absolute_min'] = expected_values['absolute_min'].absolute_min_value
        if isinstance(expected_values['absolute_max'], ParameterDeclaration):
            expected_values['absolute_max'] = expected_values['absolute_max'].absolute_max_value
        
        decl = ParameterDeclaration('test', **kwargs)
        
        self.assertEqual('test', decl.name)
        self.assertEqual(expected_values['min'], decl.min_value)
        self.assertEqual(expected_values['max'], decl.max_value)
        self.assertEqual(expected_values['default'], decl.default_value)
        self.assertEqual(expected_values['absolute_min'], decl.absolute_min_value)
        self.assertEqual(expected_values['absolute_max'], decl.absolute_max_value)
        
        decl = ParameterDeclaration('test', default=expected_values['default'])
        if 'min' in kwargs:
            decl.min_value = kwargs['min']
        if 'max' in kwargs:
            decl.max_value = kwargs['max']
            
        self.assertEqual(expected_values['min'], decl.min_value)
        self.assertEqual(expected_values['max'], decl.max_value)
        self.assertEqual(expected_values['default'], decl.default_value)
        self.assertEqual(expected_values['absolute_min'], decl.absolute_min_value)
        self.assertEqual(expected_values['absolute_max'], decl.absolute_max_value)
            
            
    def __max_assignment(self, decl: ParameterDeclaration, value: Union[ParameterDeclaration, float]) -> None:
        decl.max_value = value
        
    def __min_assignment(self, decl: ParameterDeclaration, value: Union[ParameterDeclaration, float]) -> None:
        decl.min_value = value
    
    def test_init_constant_values(self) -> None:
        self.__test_valid_values()
        self.__test_valid_values(min=-0.1)
        self.__test_valid_values(max=7.5)
        self.__test_valid_values(default=2.1)
        self.__test_valid_values(min=-0.1, max=3.5)
        self.__test_valid_values(min=-0.1, max=-0.1)
        self.__test_valid_values(min=-0.1, default=3.5)
        self.__test_valid_values(min=-0.1, default=-0.1)
        self.__test_valid_values(max=3.5, default=3.5)
        self.__test_valid_values(max=3.5, default=-0.1)
        self.__test_valid_values(min=-0.1, default=2.4, max=3.5)
        self.__test_valid_values(min=-0.1, default=-0.1, max=3.5)
        self.__test_valid_values(min=-0.1, default=3.5, max=3.5)
        self.__test_valid_values(min=1.7, default=1.7, max=1.7)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=-0.1, max=-0.2)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=-0.1, default=-0.2)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', max=-0.2, default=-0.1)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=-0.2, default=-0.3, max=0.1)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=-0.2, default=-0.3, max=-0.4)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=-0.2, default=-0.1, max=-0.4)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=-0.2, default=0.5, max=0.1)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=1.7, default=0.5, max=0.1)
        decl = ParameterDeclaration('test', min=-0.1)
        self.assertRaises(ValueError, self.__max_assignment, decl, -0.2)
        decl = ParameterDeclaration('test', min=-0.2, default=-0.1)
        self.assertRaises(ValueError, self.__max_assignment, decl, -0.4)
        decl = ParameterDeclaration('test', min=-0.2, default=0.5)
        self.assertRaises(ValueError, self.__max_assignment, decl, 0.1)
        decl = ParameterDeclaration('test', max=-0.1)
        self.assertRaises(ValueError, self.__min_assignment, decl, 0.2)
        decl = ParameterDeclaration('test', max=0.2, default=-0.1)
        self.assertRaises(ValueError, self.__min_assignment, decl, 0.4)
        decl = ParameterDeclaration('test', max=1.1, default=0.5)
        self.assertRaises(ValueError, self.__min_assignment, decl, 0.7)
    
    def test_init_min_reference(self) -> None:
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1))
        
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1), max=3.5)
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1), max=-0.1)
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=3.5)
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-0.1)
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-4.2)
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-17.3)
        
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=3.5, max=3.5)
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-0.1, max=3.5)
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-0.1, max=-0.1)
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-4.2, max=3.5)
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-4.2, max=-0.1)
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-17.3, max=3.5)
        self.__test_valid_values(min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-17.3, max=-0.1)
        
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=ParameterDeclaration('foo', min=-17.3, max=-0.1), max=-0.2)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-20.3)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-20.3, max=0.1)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-4.2, max=-0.4)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-20.3, max=-0.4)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=0.3, max=0.2)
        
    def test_init_max_reference(self) -> None:
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1))
        
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1), min=-22.2)
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1), min=-17.3)
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-22.2)
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-17.3)
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-4.2)
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-0.1)
        
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-22.2, min=-22.2)
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-17.3, min=-22.2)
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-17.3, min=-17.3)
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-4.2, min=-22.2)
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-4.2, min=-17.3)
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-0.1, min=-22.2)
        self.__test_valid_values(max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-0.1, min=-17.3)

        self.assertRaises(ValueError, ParameterDeclaration, 'test', max=ParameterDeclaration('foo', min=-17.3, max=-0.1), min=-0.2)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=0.01)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-0.01, min=-20.9)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=-4.2, min=-5.2)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=1.2, min=-0.4)
        self.assertRaises(ValueError, ParameterDeclaration, 'test', max=ParameterDeclaration('foo', min=-17.3, max=-0.1), default=1.2, min=3.9)
        
    def test_init_min_max_reference(self) -> None:
        self.__test_valid_values(min=ParameterDeclaration('fooi', max=3.5), max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', max=5.7), max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', max=13.2), max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', max=3.5), max=ParameterDeclaration('fooa', max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', max=3.5), max=ParameterDeclaration('fooa', max=3.5))
        
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=3.5), max=ParameterDeclaration('fooa', min=4.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=5.7), max=ParameterDeclaration('fooa', min=4.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3), max=ParameterDeclaration('fooa', min=4.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=3.5), max=ParameterDeclaration('fooa', min=0.3))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3), max=ParameterDeclaration('fooa', min=0.3))
        
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=3.5), max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=5.7), max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=13.2), max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=3.5), max=ParameterDeclaration('fooa', min=0.3, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=3.5), max=ParameterDeclaration('fooa', min=0.3, max=3.5))
        
        
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=3.5), default=0.3, max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=3.5), default=2.1, max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=3.5), default=3.5, max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=3.5), default=4.1, max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=3.5), default=4.2, max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=3.5), default=5.7, max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=3.5), default=13.2, max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=ParameterDeclaration('fooi', min=0.3, max=3.5), max=ParameterDeclaration('fooa', min=0.2, max=13.2))
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=ParameterDeclaration('fooi', min=0.3, max=13.5), max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3), max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3, max=3.5), max=ParameterDeclaration('fooa', max=13.2))
        self.__test_valid_values(min=ParameterDeclaration('fooi', min=0.3), max=ParameterDeclaration('fooa', max=13.2))
        
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=ParameterDeclaration('fooi', min=0.3, max=3.5), default=-0.2, max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        self.assertRaises(ValueError, ParameterDeclaration, 'test', min=ParameterDeclaration('fooi', min=0.3, max=3.5), default=22.1, max=ParameterDeclaration('fooa', min=4.2, max=13.2))
        
    def test_get_value(self) -> None:
        decl = ParameterDeclaration('foo')
        foo_param = ConstantParameter(2.1)
        self.assertEqual(foo_param.get_value(), decl.get_value({'foo': foo_param}))
        self.assertRaises(ParameterNotProvidedException, decl.get_value, {})
        
        decl = ParameterDeclaration('foo', default=2.7)
        self.assertEqual(decl.default_value, decl.get_value({}))
        
    def test_check_parameter_set_valid(self) -> None:
        min_decl = ParameterDeclaration('min', min=1.2, max=2.3)
        max_decl = ParameterDeclaration('max', min=1.2, max=5.1)
        
        min_param = ConstantParameter(1.3)
        max_param = ConstantParameter(2.3)
        parameters = {'min': min_param, 'max': max_param}
        
        decl = ParameterDeclaration('foo', min=1.3)
        self.assertTrue(decl.check_parameter_set_valid({'foo': ConstantParameter(1.4)}))
        self.assertTrue(decl.check_parameter_set_valid({'foo': ConstantParameter(1.3)}))
        self.assertFalse(decl.check_parameter_set_valid({'foo': ConstantParameter(1.1)}))
        
        decl = ParameterDeclaration('foo', max=2.3)
        self.assertTrue(decl.check_parameter_set_valid({'foo': ConstantParameter(1.4)}))
        self.assertTrue(decl.check_parameter_set_valid({'foo': ConstantParameter(2.3)}))
        self.assertFalse(decl.check_parameter_set_valid({'foo': ConstantParameter(3.1)}))
        
        decl = ParameterDeclaration('foo', min=1.3, max=2.3)
        self.assertFalse(decl.check_parameter_set_valid({'foo': ConstantParameter(0.9)}))
        self.assertTrue(decl.check_parameter_set_valid({'foo': ConstantParameter(1.3)}))
        self.assertTrue(decl.check_parameter_set_valid({'foo': ConstantParameter(1.4)}))
        self.assertTrue(decl.check_parameter_set_valid({'foo': ConstantParameter(2.3)}))
        self.assertFalse(decl.check_parameter_set_valid({'foo': ConstantParameter(3.1)}))
        
        decl = ParameterDeclaration('foo', min=min_decl, max=max_decl)
        self.assertFalse(decl.check_parameter_set_valid({'min': min_param, 'max': max_param, 'foo': ConstantParameter(0.9)}))
        self.assertFalse(decl.check_parameter_set_valid({'min': min_param, 'max': max_param, 'foo': ConstantParameter(1.2)}))
        self.assertFalse(decl.check_parameter_set_valid({'min': min_param, 'max': max_param, 'foo': ConstantParameter(1.25)}))
        self.assertTrue(decl.check_parameter_set_valid({'min': min_param, 'max': max_param, 'foo': ConstantParameter(1.3)}))
        self.assertTrue(decl.check_parameter_set_valid({'min': min_param, 'max': max_param, 'foo': ConstantParameter(1.7)}))
        self.assertTrue(decl.check_parameter_set_valid({'min': min_param, 'max': max_param, 'foo': ConstantParameter(2.3)}))
        self.assertFalse(decl.check_parameter_set_valid({'min': min_param, 'max': max_param, 'foo': ConstantParameter(3.5)}))
        self.assertFalse(decl.check_parameter_set_valid({'min': min_param, 'max': max_param, 'foo': ConstantParameter(5.1)}))
        self.assertFalse(decl.check_parameter_set_valid({'min': min_param, 'max': max_param, 'foo': ConstantParameter(17.2)}))
          
        
# class ImmutableParameterDeclarationTest(unittest.TestCase):
#
#     def test_init(self) -> None:
#         param_decl = ParameterDeclaration('hugo', min=13.2, max=15.3, default=None)
#         immutable = ImmutableParameterDeclaration(param_decl)
#         self.assertEqual(param_decl, immutable)
#         self.assertEqual(param_decl.name, immutable.name)
#         self.assertEqual(param_decl.min_value, immutable.min_value)
#         self.assertEqual(param_decl.absolute_min_value, immutable.absolute_min_value)
#         self.assertEqual(param_decl.max_value, immutable.max_value)
#         self.assertEqual(param_decl.absolute_max_value, immutable.absolute_max_value)
#         self.assertEqual(param_decl.default_value, immutable.default_value)
#
#     def test_reference_values(self) -> None:
#         min_decl = ParameterDeclaration('min', min=-0.1, max=34.7)
#         max_decl = ParameterDeclaration('max', min=23.1)
#         param_decl = ParameterDeclaration('foo', min=min_decl, max=max_decl, default=2.4)
#         immutable = ImmutableParameterDeclaration(param_decl)
#
#         self.assertEqual(param_decl, immutable)
#         self.assertEqual(param_decl.name, immutable.name)
#         self.assertEqual(param_decl.min_value, immutable.min_value)
#         self.assertEqual(param_decl.absolute_min_value, immutable.absolute_min_value)
#         self.assertEqual(param_decl.max_value, immutable.max_value)
#         self.assertEqual(param_decl.absolute_max_value, immutable.absolute_max_value)
#         self.assertEqual(param_decl.default_value, immutable.default_value)
#         self.assertIsInstance(immutable.min_value, ImmutableParameterDeclaration)
#         self.assertIsInstance(immutable.max_value, ImmutableParameterDeclaration)
#
#     def test_immutability(self) -> None:
#         param_decl = ParameterDeclaration('hugo', min=13.2, max=15.3, default=None)
#         immutable = ImmutableParameterDeclaration(param_decl)
#
#         self.assertRaises(Exception, immutable.min_value, 2.1)
#         self.assertEqual(13.2, immutable.min_value)
#         self.assertEqual(13.2, param_decl.min_value)
#
#         self.assertRaises(Exception, immutable.max_value, 14.1)
#         self.assertEqual(15.3, immutable.max_value)
#         self.assertEqual(15.3, param_decl.max_value)
        
if __name__ == "__main__":
    unittest.main(verbosity=2)
    