from typing import Dict, Set, Optional, Union, Any

from qctoolkit.serialization import Serializer
from .conditions import Condition
from .instructions import InstructionBlock
from .parameters import ParameterDeclaration, Parameter
from .pulse_template import PulseTemplate
from .sequencing import Sequencer


__all__ = ["RepetitionPulseTemplate", "ParameterNotIntegerException"]


class RepetitionPulseTemplate(PulseTemplate):

    def __init__(self, body: PulseTemplate, repetition_count: Union[int, ParameterDeclaration], identifier: Optional[str]=None) -> None:
        super().__init__(identifier=identifier)
        self.__body = body
        self.__repetition_count = repetition_count

    @property
    def body(self) -> PulseTemplate:
        return self.__body

    @property
    def repetition_count(self) -> Union[int, ParameterDeclaration]:
        return self.__repetition_count

    def __str__(self) -> str:
        return "RepetitionPulseTemplate: <{}> times <{}>".format(self.__repetition_count, self.__body)

    @property
    def parameter_names(self) -> Set[str]:
        return self.__body.parameter_names

    @property
    def parameter_declarations(self) -> Set[str]:
        return self.__body.parameter_declarations

    @property
    def is_interruptable(self) -> bool:
        return self.__body.is_interruptable

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       instruction_block: InstructionBlock) -> None:
        repetition_count = self.__repetition_count
        if isinstance(repetition_count, ParameterDeclaration):
            repetition_count = repetition_count.get_value(parameters)
            if not repetition_count.is_integer():
                raise ParameterNotIntegerException(self.__repetition_count.name, repetition_count)

        for i in range(0, int(repetition_count), 1):
            sequencer.push(self.__body, parameters, conditions, instruction_block)

    def requires_stop(self, parameters: Dict[str, Parameter], conditions: Dict[str, 'Condition']) -> bool:
        if isinstance(self.__repetition_count, ParameterDeclaration):
            if parameters[self.__repetition_count.name].requires_stop:
                return True
        return False

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        repetition_count = self.__repetition_count
        if isinstance(repetition_count, ParameterDeclaration):
            repetition_count = serializer._serialize_subpulse(repetition_count)
        return dict(
            type=serializer.get_type_identifier(self),
            body=serializer._serialize_subpulse(self.__body),
            repetition_count=repetition_count
        )

    @staticmethod
    def deserialize(serializer: Serializer,
                    repetition_count: Dict[str, Any],
                    body: Dict[str, Any],
                    identifier: Optional[str]=None) -> 'Serializable':
        body = serializer.deserialize(body)
        if isinstance(repetition_count, dict):
            repetition_count = serializer.deserialize(repetition_count)
        return RepetitionPulseTemplate(body, repetition_count, identifier=identifier)


class ParameterNotIntegerException(Exception):

    def __init__(self, parameter_name: str, parameter_value: float) -> None:
        super().__init__()
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value

    def __str__(self) -> str:
        return "The parameter <{}> must have an integral value (was <{}>) for use as repetition count.".format(
            self.parameter_name, self.parameter_value)