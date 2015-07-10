import unittest
import os
import sys

"""Change the path as we were in the similar path in the src directory"""
srcPath = "src".join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))).rsplit('tests',1))
sys.path.insert(0,srcPath)

from .type_check import typecheck,MismatchingTypesException

INTEGERS = [0,1,-1]
BOOLEANS = [True,False]
COMPLEX = [1j,2+3j,-1-1j,3-1j,-4+4j]
FLOATS = [0.0,1.0,-1.0]
STRINGS = ["foo","bar"]
TYPES = [INTEGERS,BOOLEANS,COMPLEX,FLOATS,STRINGS]
TYPECHECKARGS = {"raise_on_error":True,"log":False}

class BaseTypeCheckTest(unittest.TestCase):
    def assertNotRaises(self,exception,function,*args,**kwargs) -> None:
        try: 
            function(*args,**kwargs)
        except exception:
            self.assertFalse(True,"{} raised by {}".format(exception.__name__,function.__name__))
           
    def _int_invoke_test(self,value) -> None:
        
        @typecheck(**TYPECHECKARGS)
        def f(x:int):
            pass 
        
        f(value)

    def _float_invoke_test(self,value) -> None:
        
        @typecheck(**TYPECHECKARGS)
        def f(x:float):
            pass 
        
        f(value)
        
    def _bool_invoke_test(self,value) -> None:
        
        @typecheck(**TYPECHECKARGS)
        def f(x:bool):
            pass 
        
        f(value)
    
    def _complex_invoke_test(self,value) -> None:
        
        @typecheck(**TYPECHECKARGS)
        def f(x:complex):
            pass 
        
        f(value)
    
    def _str_invoke_test(self,value) -> None:
        
        @typecheck(**TYPECHECKARGS)
        def f(x:str):
            pass 
        
        f(value)
    
    def _int_return_test(self,value):
        
        @typecheck(**TYPECHECKARGS)
        def f(x) -> int:
            return x
        
        f(value)
    
    def _float_return_test(self,value):
        
        @typecheck(**TYPECHECKARGS)
        def f(x) -> float:
            return x
        
        f(value)
    
    def _bool_return_test(self,value):
        
        @typecheck(**TYPECHECKARGS)
        def f(x) -> bool:
            return x
        
        f(value)

    def _complex_return_test(self,value):
        
        @typecheck(**TYPECHECKARGS)
        def f(x) -> complex:
            return x
        
        f(value)
        
    def _str_return_test(self,value):
        
        @typecheck(**TYPECHECKARGS)
        def f(x) -> str:
            return x
        
        f(value)
    
    def test_type_values(self):
        for type in TYPES:
            for value in type:
                
                # Integers
                if type is INTEGERS:
                    self.assertNotRaises(MismatchingTypesException, self._int_invoke_test,value)
                    self.assertNotRaises(MismatchingTypesException, self._int_return_test,value)
                else:
                    self.assertRaises(MismatchingTypesException, self._int_invoke_test,value)
                    self.assertRaises(MismatchingTypesException, self._int_return_test,value)
                
                # Booleans
                if type is BOOLEANS:
                    self.assertNotRaises(MismatchingTypesException, self._bool_invoke_test,value)
                    self.assertNotRaises(MismatchingTypesException, self._bool_return_test,value)
                else:
                    self.assertRaises(MismatchingTypesException, self._bool_invoke_test,value)
                    self.assertRaises(MismatchingTypesException, self._bool_return_test,value)
                    
                # Complex
                if type is COMPLEX:
                    self.assertNotRaises(MismatchingTypesException, self._complex_invoke_test,value)
                    self.assertNotRaises(MismatchingTypesException, self._complex_return_test,value)
                else:
                    self.assertRaises(MismatchingTypesException, self._complex_invoke_test,value)
                    self.assertRaises(MismatchingTypesException, self._complex_return_test,value)
                
                # Floats
                if type is FLOATS:
                    self.assertNotRaises(MismatchingTypesException, self._float_invoke_test,value)
                    self.assertNotRaises(MismatchingTypesException, self._float_return_test,value)
                else:
                    self.assertRaises(MismatchingTypesException, self._float_invoke_test,value)
                    self.assertRaises(MismatchingTypesException, self._float_return_test,value)
                
                # Strings
                if type is STRINGS:
                    self.assertNotRaises(MismatchingTypesException, self._str_invoke_test,value)
                    self.assertNotRaises(MismatchingTypesException, self._str_return_test,value)
                else:
                    self.assertRaises(MismatchingTypesException, self._str_invoke_test,value)
                    self.assertRaises(MismatchingTypesException, self._str_return_test,value)

class ListCheckTest(unittest.TestCase):
    def assertNotRaises(self,exception,function,*args,**kwargs) -> None:
        try: 
            function(*args,**kwargs)
        except exception:
            self.assertFalse(True,"{} raised by {}".format(exception.__name__,function.__name__))
            
    def test_list_as_argument(self):
        
        @typecheck(**TYPECHECKARGS)
        def g(x:[int]):
            pass
        
        self.assertRaises(MismatchingTypesException, g,[STRINGS[0]])
        self.assertNotRaises(MismatchingTypesException, g,[INTEGERS[0]])
    
    def test_list_as_return(self):
        @typecheck(**TYPECHECKARGS)
        def g(x) -> [int]:
            return [x]
        
        self.assertRaises(MismatchingTypesException, g,STRINGS[0])
        self.assertNotRaises(MismatchingTypesException, g,INTEGERS[0])
        
    def test_nested_list(self):
        @typecheck(**TYPECHECKARGS)
        def g(x,y) -> [[int],[str]]:
            return [[x],[y]]
        
        self.assertRaises(MismatchingTypesException, g,STRINGS[0],INTEGERS[0])
        self.assertNotRaises(MismatchingTypesException, g,INTEGERS[0],STRINGS[0])

class DictionaryCheckTest(unittest.TestCase):
    def assertNotRaises(self,exception,function,*args,**kwargs) -> None:
        try: 
            function(*args,**kwargs)
        except exception:
            self.assertFalse(True,"{} raised by {}".format(exception.__name__,function.__name__))
            
    def test_dict_as_argument(self):
        @typecheck(**TYPECHECKARGS)
        def g(x:{str:int}):
            pass
        
        self.assertNotRaises(MismatchingTypesException,g,{STRINGS[0]:INTEGERS[0],STRINGS[1]:INTEGERS[1]})
        self.assertRaises(MismatchingTypesException,g,{STRINGS[0],STRINGS[1]})
        
    def test_dict_as_return(self):
        @typecheck(**TYPECHECKARGS)
        def g(x,y) -> {str:int}:
            return {x:y}
        
        self.assertNotRaises(MismatchingTypesException, g,STRINGS[0],INTEGERS[0])
        self.assertRaises(MismatchingTypesException, g,INTEGERS[0],STRINGS[0])
        
    def test_nested_dict(self):
        @typecheck(**TYPECHECKARGS)
        def g(x,y) -> {int:{str:int}}:
            return {int:{y:x}}
        
        self.assertRaises(MismatchingTypesException, g,STRINGS[0],INTEGERS[0])
        self.assertNotRaises(MismatchingTypesException, g,INTEGERS[0],STRINGS[0])
        
class CustomClassTest(unittest.TestCase):
    def assertNotRaises(self,exception,function,*args,**kwargs) -> None:
        try: 
            function(*args,**kwargs)
        except exception:
            self.assertFalse(True,"{} raised by {}".format(exception.__name__,function.__name__))
    

    def test_custom_as_argument(self):
        class A(object):
            pass

        @typecheck(**TYPECHECKARGS)
        def g(x:A):
            pass
         
        self.assertNotRaises(MismatchingTypesException, g, A())
        self.assertRaises(MismatchingTypesException, g, 1)
    
    def test_custom_as_return(self):
        class A(object):
            pass

        @typecheck(**TYPECHECKARGS)
        def g(x)-> A:
            return x
         
        self.assertNotRaises(MismatchingTypesException, g, A())
        self.assertRaises(MismatchingTypesException, g, 1)
        
class CustomSubClassTest(unittest.TestCase):
    pass
        
if __name__ == "__main__":
    unittest.main(verbosity=2)
    