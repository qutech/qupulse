"""This module defines parameters and parameter declaration for the usage in pulse modelling.

Classes:
    - Parameter: A base class representing a single pulse parameter.
    - ConstantParameter: A single parameter with a constant value.
    - MappedParameter: A parameter whose value is mathematically computed from another parameter.
    - ParameterNotProvidedException.
    - ParameterValueIllegalException.
"""

from abc import abstractmethod
from typing import Optional, Union, Dict, Any, Iterable, Set, List, Sequence, Mapping
from numbers import Real
from collections import ChainMap

import sympy
import numpy

from qctoolkit.serialization import AnonymousSerializable
from qctoolkit.expressions import Expression
from qctoolkit.utils.types import HashableNumpyArray, DocStringABCMeta, ReadOnlyChainMap

__all__ = ["Parameter", "ConstantParameter",
           "ParameterNotProvidedException", "ParameterConstraintViolation",
           "ParameterMap", "ParameterEncyclopedia", "ParameterLibrary", "ParameterProviderContext"]


class Parameter(metaclass=DocStringABCMeta):
    """A parameter for pulses.
    
    Parameter specifies a concrete value which is inserted instead
    of the parameter declaration reference in a PulseTemplate if it satisfies
    the minimum and maximum boundary of the corresponding ParameterDeclaration.
    Implementations of Parameter may provide a single constant value or
    obtain values by computation (e.g. from measurement results).
    """
    @abstractmethod
    def get_value(self) -> Real:
        """Compute and return the parameter value."""

    @property
    @abstractmethod
    def requires_stop(self) -> bool:
        """Query whether the evaluation of this Parameter instance requires an interruption in
        execution/sequencing, e.g., because it depends on data that is only measured in during the
        next execution.

        Returns:
            True, if evaluating this Parameter instance requires an interruption.
        """

    @abstractmethod
    def __hash__(self) -> int:
        pass

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and hash(self) == hash(other)

        
class ConstantParameter(Parameter):
    """A pulse parameter with a constant value."""
    
    def __init__(self, value: Union[Real, numpy.ndarray]) -> None:
        """Create a ConstantParameter instance.

        Args:
            value (Real): The value of the parameter
        """
        super().__init__()
        if isinstance(value, Real):
            self._value = value
        else:
            self._value = numpy.array(value).view(HashableNumpyArray)
        
    def get_value(self) -> Union[Real, numpy.ndarray]:
        return self._value

    def __hash__(self) -> int:
        return hash(self._value)

    @property
    def requires_stop(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<ConstantParameter {0}>".format(self._value)


class MappedParameter(Parameter):
    """A pulse parameter whose value is derived from other parameters via some mathematical
    expression.

    The dependencies of a MappedParameter instance are defined by the free variables appearing
    in the expression that defines how its value is derived.

    MappedParameter holds a dictionary which assign Parameter objects to these dependencies.
    Evaluation of the MappedParameter will raise a ParameterNotProvidedException if a Parameter
    object is missing for some dependency.
    """

    def __init__(self,
                 expression: Expression,
                 dependencies: Optional[Dict[str, Parameter]]=None) -> None:
        """Create a MappedParameter instance.

        Args:
            expression (Expression): The expression defining how the the value of this
                MappedParameter instance is derived from its dependencies.
             dependencies (Dict(str -> Parameter)): Parameter objects of the dependencies. May also
                be defined via the dependencies public property. (Optional)
        """
        super().__init__()
        self._expression = expression
        self.dependencies = dict() if dependencies is None else dependencies
        self._cached_value = (None, None)

    def _collect_dependencies(self) -> Dict[str, float]:
        # filter only real dependencies from the dependencies dictionary
        try:
            return {dependency_name: self.dependencies[dependency_name].get_value()
                    for dependency_name in self._expression.variables}
        except KeyError as key_error:
            raise ParameterNotProvidedException(str(key_error)) from key_error

    def get_value(self) -> Union[Real, numpy.ndarray]:
        """Does not check explicitly if a parameter requires to stop."""
        current_hash = hash(self)
        if current_hash != self._cached_value[0]:
            self._cached_value = (current_hash, self._expression.evaluate_numeric(**self._collect_dependencies()))
        return self._cached_value[1]

    def __hash__(self):
        return hash(tuple(self.dependencies.items()))

    @property
    def requires_stop(self) -> bool:
        """Does not explicitly check that all parameters are provided if one requires stopping"""
        try:
            return any(self.dependencies[v].requires_stop
                       for v in self._expression.variables)
        except KeyError as err:
            raise ParameterNotProvidedException(err.args[0]) from err

    def __repr__(self) -> str:
        try:
            value = self.get_value()
        except:
            value = 'nothing'

        return "<MappedParameter {0} evaluating to {1}>".format(
            self._expression, value
        )


class ParameterConstraint(AnonymousSerializable):
    """A parameter constraint like 't_2 < 2.7' that can be used to set bounds to parameters."""
    def __init__(self, relation: Union[str, sympy.Expr]):
        super().__init__()
        if isinstance(relation, str) and '==' in relation:
            # The '==' operator is interpreted by sympy as exactly, however we need a symbolical evaluation
            self._expression = sympy.Eq(*sympy.sympify(relation.split('==')))
        else:
            self._expression = sympy.sympify(relation)
        if not isinstance(self._expression, sympy.boolalg.Boolean):
            raise ValueError('Constraint is not boolean')
        self._expression = Expression(self._expression)

    @property
    def affected_parameters(self) -> Set[str]:
        return set(self._expression.variables)

    def is_fulfilled(self, parameter: Dict[str, Any]) -> bool:
        if not self.affected_parameters <= set(parameter.keys()):
            raise ParameterNotProvidedException((self.affected_parameters-set(parameter.keys())).pop())

        return numpy.all(self._expression.evaluate_numeric(**parameter))

    @property
    def sympified_expression(self) -> sympy.Expr:
        return self._expression.sympified_expression

    def __eq__(self, other: 'ParameterConstraint') -> bool:
        return self._expression.underlying_expression == other._expression.underlying_expression

    def __str__(self) -> str:
        if isinstance(self._expression.sympified_expression, sympy.Eq):
            return '{}=={}'.format(self._expression.sympified_expression.lhs,
                                   self._expression.sympified_expression.rhs)
        else:
            return str(self._expression.sympified_expression)

    def get_serialization_data(self) -> str:
        return str(self)


class ParameterConstrainer:
    """A class that implements the testing of parameter constraints. It is used by the subclassing pulse templates."""
    def __init__(self, *,
                 parameter_constraints: Optional[Iterable[Union[str, ParameterConstraint]]]) -> None:
        if parameter_constraints is None:
            self._parameter_constraints = []
        else:
            self._parameter_constraints = [constraint if isinstance(constraint, ParameterConstraint)
                                           else ParameterConstraint(constraint)
                                           for constraint in parameter_constraints]

    @property
    def parameter_constraints(self) -> List[ParameterConstraint]:
        return self._parameter_constraints

    def validate_parameter_constraints(self, parameters: [str, Union[Parameter, Real]]) -> None:
        """Raises a ParameterConstraintViolation exception if one of the constraints is violated.
        :param parameters: These parameters are checked.
        :return:
        """
        for constraint in self._parameter_constraints:
            constraint_parameters = {k: v.get_value() if isinstance(v, Parameter) else v for k, v in parameters.items()}
            if not constraint.is_fulfilled(constraint_parameters):
                raise ParameterConstraintViolation(constraint, constraint_parameters)

    @property
    def constrained_parameters(self) -> Set[str]:
        if self._parameter_constraints:
            return set.union(*(c.affected_parameters for c in self._parameter_constraints))
        else:
            return set()


class ParameterConstraintViolation(Exception):
    def __init__(self, constraint: ParameterConstraint, parameters: Dict[str, Real]):
        super().__init__("The constraint '{}' is not fulfilled.\nParameters: {}".format(constraint, parameters))
        self.constraint = constraint
        self.parameters = parameters


class ParameterNotProvidedException(Exception):
    """Indicates that a required parameter value was not provided."""
    
    def __init__(self, parameter_name: str) -> None:
        super().__init__()
        self.parameter_name = parameter_name
        
    def __str__(self) -> str:
        return "No value was provided for parameter '{0}'.".format(self.parameter_name)


class InvalidParameterNameException(Exception):
    def __init__(self, parameter_name: str):
        self.parameter_name = parameter_name

    def __str__(self) -> str:
        return '{} is an invalid parameter name'.format(self.parameter_name)


ParameterMap = Mapping[str, Parameter]
ParameterEncyclopedia = Dict[str, ParameterMap]


class ParameterProviderContext():
    """A view at a parameter library within the context of a certain pulse.

    Implements the python context manager idiom."""

    Delimiter = '.'

    def __init__(self, *, parameter_library: 'ParameterLibrary', pulse_context: str='') -> None:
        self.__context = pulse_context
        self.__library = parameter_library

    @property
    def context(self) -> str:
        return self.__context

    @property
    def library(self) -> 'ParameterLibrary':
        return self.__library

    def enter_context(self, pulse_context_name: Optional[str]) -> 'ParameterProviderContext':
        """Steps into a subcontext.

        Returns a ParameterProviderContext for the new context. If the previous context was e.g. "foo" and enter_context
        was called with argument "bar", the context of the returned object will be "foo.bar".
        If the given pulse_context_name is empty or None, the context of the returned object is the same as the one on
        which enter_context was called, i.e., if the previous context was "foo" and enter_context was called with empty
        argument, the context of the returned object will also be "foo"."""
        if pulse_context_name is None or len(pulse_context_name) == 0:
            return ParameterProviderContext(self.__context)
        return ParameterProviderContext(self.__context + self.Delimiter + pulse_context_name)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_parameters(self, subst_params: Optional[ParameterMap]=None) -> ParameterMap:
        return self.__library.get_parameters(self.__context, subst_params=subst_params)

    def get_volatile_parameter_names(self) -> Set[str]:
        return set()

    def have_parameters_changed(self) -> bool:
        return True # no way to distinguish yet, always return True so that pulses are rebuilt in any case


class ParameterLibrary(ParameterProviderContext):
    """Composes pulse-specific parameter dictionaries from hierarchical source dictionaries.

    Nomenclature: In the following,
    - a ParameterMap refers to a simple (dictionary) mapping of parameter names to values
    - a ParameterEncyclopedia refers to a dictionary where pulse identifiers are keys and ParameterDicts are values.
        Additionally, there might be a 'global' key also referring to a ParameterDict. Parameter values under the 'global'
        key are applied to all pulses.

    Example for a ParameterMap:
    book = dict(foo_param=17.24, bar_param=-2363.4)

    Example for a ParameterEncyclopedia:
    encl = dict(
        global=dict(foo_param=17.24, bar_param=-2363.4),
        test_pulse_1=dict(foo_param=5.125, another_param=13.37)
    )

    ParameterLibrary gets a sequence of ParameterEncyclopedias on initialization. This sequence is
    understood to be in hierarchical order where parameter values in ParameterDicts later in the sequence supersede
    values for the same parameter in earlier dictionaries. In that sense, the order can be understood of being in
    increasing specialization, i.e., most general "default" values come in early ParameterDicts while more
    specialized parameter values (for a specific experiment, hardware setup) are placed in ParameterDicts later in the
    sequence and replace the default values.

    Within a single ParameterDict, parameter values under the 'global' key are applied to every pulse first. If a pulse
    has an identifier for which a key is present in the ParameterDict, the contained parameter values are applied after
    the globals and replace global values (if colliding).
    """

    def __init__(self, parameter_source_dicts: Sequence[ParameterEncyclopedia]) -> None:
        """Creates a ParameterLibrary instance.

        Args:
            parameter_source_dicts (Sequence(Dict(str -> Dict(str -> Parameter)))): A sequence of parameter source dictionaries.
        """
        super().__init__(parameter_library=self)
        self._parameter_sources = parameter_source_dicts
        self._dict_chains = []
        self._create_pulse_dict_chains()

    def _create_pulse_dict_chains(self) -> None:
        all_known_pulses = set.union(*(set(parameter_encl.keys()) for parameter_encl in self._parameter_sources))
        pulse_dict_chains = dict()
        for pulse in all_known_pulses:
            dict_chain = []
            for parameter_encl in reversed(self._parameter_sources):
                if pulse != 'global':
                    if pulse in parameter_encl:
                        dict_chain.append(parameter_encl[pulse])
                if 'global' in parameter_encl:
                    dict_chain.append(parameter_encl['global'])
            pulse_dict_chains[pulse] = dict_chain
        self._dict_chains = pulse_dict_chains

    @property
    def parameter_sources(self) -> Sequence[ParameterEncyclopedia]:
        return self._parameter_sources

    def update_internals(self) -> None:
        self._create_pulse_dict_chains()

    def get_parameters(self, pulse_context: str, subst_params: Optional[ParameterMap]=None) -> ParameterMap:
        """Returns a dictionary with parameters from the library for a given pulse template.

        The parameter source dictionaries (i.e. the library) given to the ParameterLibrary instance on construction
        are processed to extract the most specialized parameter values for the given pulse as described in the
        class docstring (of ParameterLibrary).

        Additionally, the optional argument subst_params is applied after processing the library to allow for a final
        specialization by custom replacements. subst_params must be a simple parameter dictionary of the form
        parameter_name -> parameter_value .

        Args:
            pulse_context (str): The pulse context to fetch parameters for.
            subst_params (Dict(str -> Parameter)): An optional additional parameter specialization dictionary to be applied
                after processing the parameter library.
        Returns:
            a mapping giving the most specialized parameter values for the given pulse template. also contains all
            globally specified parameters, even if they are not required by the pulse template.
        """
        maps = []
        if subst_params:
            maps.append(subst_params)
        if pulse_context in self._dict_chains:
            maps.extend(self._dict_chains[pulse_context])
        elif 'global' in self._dict_chains: # if no pulse specific chain is known, supply chain of global parameters if existent
            maps.extend(self._dict_chains['global'])

        return ReadOnlyChainMap(ChainMap(*maps))

