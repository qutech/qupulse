"""STANDARD LIBRARY IMPORTS"""
import logging
from typing import Dict, List, Tuple, Set, Callable
from functools import partial
"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""

from .PulseTemplate import PulseTemplate, MeasurementWindow, ParameterNotInPulseTemplateException
from .Parameter import ParameterDeclaration, Parameter, ConstantParameter, ParameterNotProvidedException
from .Sequencer import InstructionBlock, Sequencer
from .Instructions import WaveformTable, Waveform
from .TablePulseTemplate import TableEntry, TablePulseTemplate
from .Interpolation import HoldInterpolationStrategy


logger = logging.getLogger(__name__)

# type signatures used in this module
# a MappingFunction takes a dictionary with parameter declarations, keyed with strings and returns a float
MappingFunction = Callable[[Dict[str, ParameterDeclaration]], float]
# a subtemplate consists of a pulse template and mapping functions for its "internal" parameters
Subtemplate = Tuple[PulseTemplate, Dict[str, MappingFunction]]

class SequencePulseTemplate(PulseTemplate):
    """A sequence of different PulseTemplates.
    
    SequencePulseTemplate allows to group smaller
    PulseTemplates (subtemplates) into on larger sequence,
    i.e., when instantiating a pulse from a SequencePulseTemplate
    all pulses instantiated from the subtemplates are queued for
    execution right after one another.
    SequencePulseTemplate allows to specify a mapping of
    parameter declarations from its subtemplates, enabling
    renaming and mathematical transformation of parameters.
    The default behavior is to exhibit the union of parameter
    declarations of all subtemplates. If two subpulses declare
    a parameter with the same name, it is mapped to both. If the
    declarations define different minimal and maximal values, the
    more restricitive is chosen, if possible. Otherwise, an error
    is thrown and an explicit mapping must be specified.
    """

    def __init__(self, subtemplates: List[Subtemplate], external_parameters: List[str]) -> None:
        super().__init__()
        self._parameter_names = frozenset(external_parameters)
        for template, mapfuns in subtemplates:
            # Consistency checks
            open_parameters = template.parameter_names
            mapped_parameters = set(mapfuns.keys())
            missing_parameters = open_parameters - mapped_parameters
            for m in missing_parameters:
                raise MissingMappingException(template, m)
            unnecessary_parameters = mapped_parameters - open_parameters
            for m in unnecessary_parameters:
                raise UnnecessaryMappingException(template, m)

        self.subtemplates = subtemplates  # type: List[Subtemplate]
        self.__is_interruptable = True

    @property
    def parameter_names(self) -> Set[str]:
        return self._parameter_names

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        parameter_declarations = set()
        for subtemplate in self.subtemplates:
            for declaration in subtemplate.parameter_declarations:
                if declaration.name in self.mapping.get_targets():
                    #Rename the parameter according to the mapping. The target name becomes the source name.
                    declaration = ParameterDeclaration(
                                                       self.mapping.get_targets()[declaration.name], 
                                                       min=declaration.min_value, 
                                                       max=declaration.max_value, 
                                                       default=declaration.default_value
                                                       ) 
                    # TODO: min, max and default values might have to be mapped to? especially in the case that min, max are ParameterDeclarations as well
                parameter_declarations.add(declaration)
        return parameter_declarations

    def get_entries_instantiated(self, outer_parameters: Dict[str, ParameterDeclaration]) -> List[TableEntry]:
        """Applies mappings to the subtemplates, returns a list of "rendered" subtemplates without parameters."""
        # Consistency checks:
        missing = self.parameter_names - set(outer_parameters)
        for m in missing:
            raise ParameterNotProvidedException(m)
        unnecessary = set(outer_parameters) - self.parameter_names
        for un in unnecessary:
            raise ParameterNotInPulseTemplateException(un, self)

        # do work
        new_list = []
        # check supported types:
        typecheck = lambda x: isinstance(x, TablePulseTemplate) or isinstance(x, SequencePulseTemplate)
        if all(typecheck(a[0]) for a in self.subtemplates):
            for template, mappings in self.subtemplates:
                inner_parameters = {name: mappings[name](outer_parameters) for name in template.parameter_names}
                new_list.append(template.get_entries_instantiated(inner_parameters))
            total_length = 0
            if new_list:
                entries = [TableEntry(0,0,HoldInterpolationStrategy())] # List[TableEntry]
                for pulse in new_list:
                    for point in pulse:
                        new_time = point.t + total_length
                        if new_time != total_length: # skip the automatic (0,0) point
                            entries.append(TableEntry(new_time, point.v, point.interp))
                    total_length += pulse[-1].t
                return entries
            else:
                return []
        else:
            raise NotImplementedError('Instantiating SequencePulseTemplates is only supported for those consisting only of TablePulseTemplates and SequencePulseTemplates')

    @property
    def measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        measurement_windows = []
        for subtemplate, _ in self.subtemplates:
            measurement_windows = measurement_windows + subtemplate.measurement_windows(parameters)
        return measurement_windows

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None):
        return self.measurement_windows

    @property
    def is_interruptable(self) -> bool:
        return self.__is_interruptable
    
    @is_interruptable.setter
    def is_interruptable(self, new_value: bool):
        self.__is_interruptable

    def requires_stop(self, parameters: Dict[str, Parameter]) -> bool:
        pass

    def build_sequence(self, sequencer: "Sequencer", parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        missing = self.parameter_names - set(parameters)
        for m in missing:
            raise ParameterNotProvidedException(m)
        unnecessary = set(parameters) - self.parameter_names
        for un in unnecessary:
            raise ParameterNotInPulseTemplateException(un, self)

        for template, mappings in self.subtemplates:
            inner_parameters = {name: mappings[name](parameters) for name in template.parameter_names}
            sequencer.push(template, inner_parameters, instruction_block)

class MissingMappingException(Exception):
    def __init__(self, template, key) -> None:
        super().__init__()
        self.key = key
        self.template = template

    def __str__(self) -> str:
        return "The template {} needs a mapping function for parameter {}". format(self.template, self.key)

class UnnecessaryMappingException(Exception):
    def __init__(self, template, key):
        super().__init__()
        self.template = template
        self.key = key

    def __str__(self) -> str:
        return "Mapping function for parameter '{}', which template {} does not need".format(self.key, self.template)
