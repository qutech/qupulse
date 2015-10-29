"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
import inspect
import logging 

logger = logging.getLogger(__name__)
def __equalTypes(x, y, same_length: bool = True) -> bool:
    """Test whether x and y are of (or reference to) the same type or not.
    
    This strictly internal function compares the type of two variables and the returned
    value (bool) is True if they have the same type or if one is the type of the other.
    
    The argument "same_length" allows to describe schemes. So for instance if you want
    to declare an array of a type of variable length, you just have to set the 
    "same_length" flag to false.
    
    """
    # If the variable has no annotation, every variable is a right variable
    if inspect._empty in [x,y]:
        return True
    x_type = type(x)
    if type(x) is type:
        x_type = x
    y_type = type(y)
    if type(y) is type:
        y_type = y
    if y_type is not x_type:
        return False

    #From here on, x_type can be seen as equal to y_type
    sequence_types = (list,dict,tuple)
    if x_type in sequence_types:
        if same_length and len(x) != len(y): 
            return False
        else:
            if type(x) is list:
                equal_subtypes = True
                index = 0
                while equal_subtypes and index < max(len(x),len(y)):
                    # This can also handle iterables of unequal lengths
                    equal_subtypes = equal_subtypes and __equalTypes(x[index%len(x)],y[index%len(y)],same_length)
                    index += 1
                return equal_subtypes
            elif type(x) is dict:
                # TODO: Fix this quadratic check
                for key_x in x:
                    for key_y in y:
                        if __equalTypes(key_x, key_y):
                            return __equalTypes(x[key_x], y[key_y],same_length)
                        return False
            return False    
    return True   

def typecheck(same_length: bool = False, raise_on_error: bool = False, log: bool = True):
    """Decorator for functions, invokes typechecking on their annotation
    
    This function invokes the typechecking on functions which uses the annotations of
    python3. This type check will happen in runtime.
    
    For the "same_length" argument, please reference to "__equal_types".
    
    There are two ways, this function can react on type errors:
    1. "raise_on_error" will raise an "MismatchingTypeException" defined below.
    2. "log" will create a logging message on the warning level.
    
    Usage:
    "@typecheck:
    'def f(x):"
    
    """
    def check(f):
        #Params needed because it is an ORDERED DICTIONARY, so we can derive the variable assignement
        params = inspect.signature(f).parameters 
        def new_f(*args, **kwargs):
            #Checking keyword arguments.
            for key in kwargs:
                if not __equalTypes(params[key].annotation, (kwargs[key]),same_length):
                    if log:
                        logger.warning("Type Checking Error in function '{3}': Mismatching types for keyword argument {0}: {1} passed but {2} expected!".format(key,type(kwargs[key]),params[key].annotation,f.__name__))
                    if raise_on_error:
                        raise MismatchingTypesException("Keyword argument {}".format(key))
                    
            #Checking non keyword arguments
            for index, key in enumerate([key for key in params if key not in kwargs and key != "return"]):
                if not __equalTypes(params[key].annotation,args[index],same_length):
                    if log:
                        logger.warning("Type Checking Error in function '{3}': Mismatching types for keywordless argument {0}: {1} passed but {2} expected!".format(key,type(args[index]),params[key].annotation,f.__name__))
                    if raise_on_error:
                        raise MismatchingTypesException("Non-Keyword argument {}".format(key))
            return_value = f(*args, **kwargs)
            
            #Checking the return value
            if "return" in f.__annotations__:
                if not __equalTypes(return_value,f.__annotations__["return"],same_length):
                    if log:
                        logger.warning("Type Checking Error in function '{2}': Mismatching return value: {0} returned but {1} expected!".format(type(return_value),f.__annotations__["return"],f.__name__))
                    if raise_on_error:
                        raise MismatchingTypesException("Return value")
            return return_value
        return new_f
    return check

class MismatchingTypesException(Exception):
    """Exception raised when a type error occurs in "type_check"
    
    Has the behaviour of an default Exception due inheritance of the self. 
    
    """
    def __init__(self,message) -> None:
        super().__init__()
        self.message = message
        
    def __str__(self, *args, **kwargs) -> str:
        return "Mismatching types Exception:{}".format(self.message)
    

