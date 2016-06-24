"""This module defines RepetitionPulseTemplate, a higher-order hierarchical pulse template that
represents the n-times repetition of another PulseTemplate."""

from typing import Dict, List, Set, Optional, Union, Any

from qctoolkit.serialization import Serializer

from qctoolkit.pulses.pulse_template import PulseTemplate, MeasurementWindow
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.instructions import InstructionBlock, InstructionPointer
from qctoolkit.pulses.parameters import ParameterDeclaration, Parameter
from qctoolkit.pulses.conditions import Condition


__all__ = ["RepetitionPulseTemplate", "ParameterNotIntegerException"]


class RepetitionPulseTemplate(PulseTemplate):
    """Repeat a PulseTemplate a constant number of times.

    The equivalent to a simple for-loop in common programming languages in qctoolkit's pulse
    modelling.
    """

    def __init__(self,
                 body: PulseTemplate,
                 repetition_count: Union[int, ParameterDeclaration],
                 identifier: Optional[str]=None) -> None:
        """Create a new RepetitionPulseTemplate instance.

        Args:
            body (PulseTemplate): The PulseTemplate which will be repeated.
            repetition_count (int or ParameterDeclaration): The number of repetitions either as a
                constant integer value or as a parameter declaration.
            identifier (str): A unique identifier for use in serialization. (optional)
        """
        super().__init__(identifier=identifier)
        self.__body = body
        self.__repetition_count = repetition_count

    @property
    def body(self) -> PulseTemplate:
        """The PulseTemplate which is repeated by this RepetitionPulseTemplate."""
        return self.__body

    @property
    def repetition_count(self) -> Union[int, ParameterDeclaration]:
        """The amount of repetitions. Either a constant integer or a ParameterDeclaration object."""
        return self.__repetition_count

    def __str__(self) -> str:
        return "RepetitionPulseTemplate: <{}> times <{}>"\
            .format(self.__repetition_count, self.__body)

    @property
    def parameter_names(self) -> Set[str]:
        return self.__body.parameter_names

    @property
    def parameter_declarations(self) -> Set[str]:
        return self.__body.parameter_declarations

    def get_measurement_windows(self,
                                parameters: Dict[str, Parameter]=None
                                ) -> List[MeasurementWindow]:
        """Return all measurement windows defined in this PulseTemplate."""
        raise NotImplementedError()

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

        body_block = instruction_block.create_embedded_block()
        body_block.return_ip = InstructionPointer(instruction_block, len(instruction_block))

        instruction_block.add_instruction_repj(int(repetition_count), body_block)
        sequencer.push(self.body, parameters, conditions, body_block)

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, Condition]) -> bool:
        if isinstance(self.__repetition_count, ParameterDeclaration):
            if parameters[self.__repetition_count.name].requires_stop:
                return True
        return False

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        repetition_count = self.__repetition_count
        if isinstance(repetition_count, ParameterDeclaration):
            repetition_count = serializer.dictify(repetition_count)
        return dict(
            type=serializer.get_type_identifier(self),
            body=serializer.dictify(self.__body),
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
    """Indicates that the value of the parameter given as repetition count was not an integer."""

    def __init__(self, parameter_name: str, parameter_value: float) -> None:
        super().__init__()
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value

    def __str__(self) -> str:
        return "The parameter <{}> must have an integral value (was <{}>) " \
            "for use as repetition count.".format(
                self.parameter_name,
                self.parameter_value
            )
