import unittest
from typing import Union

from qctoolkit.pulses.parameters import ConstantParameter, ParameterDeclaration, ParameterNotProvidedException

from tests.serialization_dummies import DummySerializer
from tests.pulses.sequencing_dummies import DummyParameter


class ParameterTest(unittest.TestCase):
    
    def test_float_conversion_method(self) -> None:
        parameter = DummyParameter()
        self.assertEqual(parameter.value, float(parameter))


class ConstantParameterTest(unittest.TestCase):
    def __test_valid_params(self, value: float) -> None:
        constant_parameter = ConstantParameter(value)
        self.assertEqual(value, constant_parameter.get_value())
        self.assertEqual(value, float(constant_parameter))
        
    def test_float_values(self) -> None:
        self.__test_valid_params(-0.3)
        self.__test_valid_params(0)
        self.__test_valid_params(0.3)

    def test_requires_stop(self) -> None:
        constant_parameter = ConstantParameter(0.3)
        self.assertFalse(constant_parameter.requires_stop)

    def test_repr(self) -> None:
        constant_parameter = ConstantParameter(0.2)
        self.assertEqual("<ConstantParameter 0.2>", repr(constant_parameter))

    def test_get_serialization_data(self) -> None:
        constant_parameter = ConstantParameter(-0.2)
        serializer = DummySerializer()
        data = constant_parameter.get_serialization_data(serializer)
        self.assertEqual(dict(type=serializer.get_type_identifier(constant_parameter), constant=-0.2), data)

    def test_deserialize(self) -> None:
        serializer = DummySerializer()
        constant_parameter = ConstantParameter.deserialize(serializer, constant=3.1)
        self.assertEqual(3.1, constant_parameter.get_value())
        self.assertIsNone(constant_parameter.identifier)

    
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

    def __assign_min_value(self, left_value: ParameterDeclaration, right_value: ParameterDeclaration) -> None:
        left_value.min_value = right_value

    def test_internal_set_value_exception_branches(self) -> None:
        foo = ParameterDeclaration('foo', min=2.1, max=2.6)
        bar = ParameterDeclaration('bar', min=foo, max=foo)
        self.assertRaises(ValueError, ParameterDeclaration, 'foobar', min=3.1, max=bar)

        bar = ParameterDeclaration('bar', min=foo, max=foo)
        foobar = ParameterDeclaration('foobar', max=1.1)
        self.assertRaises(ValueError, self.__assign_min_value, foobar, bar)

    def test_is_parameter_valid_no_bounds(self) -> None:
        decl = ParameterDeclaration('foo')
        param = ConstantParameter(2.4)
        self.assertTrue(decl.is_parameter_valid(param))

    def test_is_parameter_valid_min_bound(self) -> None:
        decl = ParameterDeclaration('foobar', min=-0.1)
        params = [(False, -0.5), (True, -0.1), (True, 0), (True, 17.3)]
        for expected, param in params:
            self.assertEqual(expected, decl.is_parameter_valid(param))

    def test_is_parameter_valid_max_bound(self) -> None:
        decl = ParameterDeclaration('foobar', max=-0.1)
        params = [(True, -0.5), (True, -0.1), (False, 0), (False, 17.3)]
        for expected, param in params:
            self.assertEqual(expected, decl.is_parameter_valid(param))

    def test_is_parameter_valid_min_max_bound(self) -> None:
        decl = ParameterDeclaration('foobar', min=-0.1, max=13.2)
        params = [(False, -0.5), (True, -0.1), (True, 0), (True, 7.9), (True, 13.2), (False, 17.3)]
        for expected, param in params:
            self.assertEqual(expected, decl.is_parameter_valid(param))

    def test_str_and_repr(self) -> None:
        decl = ParameterDeclaration('foo', min=0.1)
        self.assertEqual("{} 'foo', range (0.1, inf), default None".format(type(decl)), str(decl))
        self.assertEqual("<{} 'foo', range (0.1, inf), default None>".format(type(decl)), repr(decl))
        min_decl = ParameterDeclaration('minifoo', min=0.2)
        max_decl = ParameterDeclaration('maxifoo', max=1.1)
        decl = ParameterDeclaration('foo', min=min_decl)
        self.assertEqual("{} 'foo', range (Parameter 'minifoo' (min 0.2), inf), default None".format(type(decl)), str(decl))
        self.assertEqual("<{} 'foo', range (Parameter 'minifoo' (min 0.2), inf), default None>".format(type(decl)), repr(decl))
        decl = ParameterDeclaration('foo', max=max_decl)
        self.assertEqual("{} 'foo', range (-inf, Parameter 'maxifoo' (max 1.1)), default None".format(type(decl)), str(decl))
        self.assertEqual("<{} 'foo', range (-inf, Parameter 'maxifoo' (max 1.1)), default None>".format(type(decl)), repr(decl))
        decl = ParameterDeclaration('foo', min=min_decl, max=max_decl)
        self.assertEqual("{} 'foo', range (Parameter 'minifoo' (min 0.2), Parameter 'maxifoo' (max 1.1)), default None".format(type(decl)), str(decl))
        self.assertEqual("<{} 'foo', range (Parameter 'minifoo' (min 0.2), Parameter 'maxifoo' (max 1.1)), default None>".format(type(decl)), repr(decl))


class ParameterDeclarationSerializationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.serializer = DummySerializer()
        self.declaration = ParameterDeclaration('foo')
        self.expected_data = dict(name='foo', type=self.serializer.get_type_identifier(self.declaration))

    def test_get_serialization_data_all_default(self) -> None:
        self.expected_data['min_value'] = float('-inf')
        self.expected_data['max_value'] = float('+inf')
        self.expected_data['default_value'] = None
        self.assertEqual(self.expected_data, self.declaration.get_serialization_data(self.serializer))

    def test_get_serialization_data_all_floats(self) -> None:
        self.declaration = ParameterDeclaration('foo', min=-3.1, max=4.3, default=0.2)
        self.expected_data['min_value'] = -3.1
        self.expected_data['max_value'] = 4.3
        self.expected_data['default_value'] = 0.2
        self.assertEqual(self.expected_data, self.declaration.get_serialization_data(self.serializer))

    def test_get_serialization_data_min_max_references(self) -> None:
        bar_min = ParameterDeclaration('bar_min')
        bar_max = ParameterDeclaration('bar_max')
        self.declaration.min_value = bar_min
        self.declaration.max_value = bar_max
        self.expected_data['min_value'] = 'bar_min'
        self.expected_data['max_value'] = 'bar_max'
        self.expected_data['default_value'] = None
        self.assertEqual(self.expected_data, self.declaration.get_serialization_data(self.serializer))

    def test_deserialize_all_default(self) -> None:
        data = dict(min_value=float('-inf'), max_value=float('+inf'), default_value=None, name='foo')
        declaration = ParameterDeclaration.deserialize(self.serializer, **data)
        self.assertEqual(data['min_value'], declaration.min_value)
        self.assertEqual(data['max_value'], declaration.max_value)
        self.assertEqual(data['default_value'], declaration.default_value)
        self.assertEqual(data['name'], declaration.name)
        self.assertIsNone(declaration.identifier)

    def test_deserialize_all_floats(self) -> None:
        data = dict(min_value=33.3, max_value=44, default_value=41.1, name='foo')
        declaration = ParameterDeclaration.deserialize(self.serializer, **data)
        self.assertEqual(data['min_value'], declaration.min_value)
        self.assertEqual(data['max_value'], declaration.max_value)
        self.assertEqual(data['default_value'], declaration.default_value)
        self.assertEqual(data['name'], declaration.name)
        self.assertIsNone(declaration.identifier)

    def test_deserialize_min_max_references(self) -> None:
        data = dict(min_value='bar_min', max_value='bar_max', default_value=-23.5, name='foo')
        declaration = ParameterDeclaration.deserialize(self.serializer, **data)
        self.assertEqual(float('-inf'), declaration.min_value)
        self.assertEqual(float('+inf'), declaration.max_value)
        self.assertEqual(data['default_value'], declaration.default_value)
        self.assertEqual(data['name'], declaration.name)
        self.assertIsNone(declaration.identifier)


class ParameterNotProvidedExceptionTests(unittest.TestCase):

    def test(self) -> None:
        exc = ParameterNotProvidedException('foo')
        self.assertEqual("No value was provided for parameter 'foo' and no default value was specified.", str(exc))
        
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
    