"""This module defines LoopPulseTemplate, a higher-order hierarchical pulse template that loops
another PulseTemplate based on a condition."""


from typing import Dict, Set, Optional, Any

from qctoolkit.serialization import Serializer

from qctoolkit.pulses.parameters import Parameter
from qctoolkit.pulses.pulse_template import PulseTemplate
from qctoolkit.pulses.conditions import Condition, ConditionMissingException
from qctoolkit.pulses.instructions import InstructionBlock
from qctoolkit.pulses.sequencing import Sequencer

__all__ = ['LoopPulseTemplate', 'ConditionMissingException']


class LoopPulseTemplate(PulseTemplate):
    """Conditional looping in a pulse.
    
    A LoopPulseTemplate is a PulseTemplate which body (subtemplate) is repeated
    during execution as long as a certain condition holds.
    """
    
    def __init__(self, condition: str, body: PulseTemplate, identifier: Optional[str]=None) -> None:
        """Create a new LoopPulseTemplate instance.

        Args:
            condition (str): A unique identifier for the looping condition. Will be used to obtain
                the Condition object from the mapping passed in during the sequencing process.
            body (PulseTemplate): The PulseTemplate which will be repeated as long as the condition
                holds.
            identifier (str): A unique identifier for use in serialization. (optional)
        """
        super().__init__(identifier=identifier)
        self.__condition = condition
        self.__body = body

    def __str__(self) -> str:
        return "LoopPulseTemplate: Condition <{}>, Body <{}>".format(self.__condition, self.__body)

    @property
    def body(self) -> PulseTemplate:
        """This LoopPulseTemplate's body/subtemplate."""
        return self.__body

    @property
    def condition(self) -> str:
        """This LoopPulseTemplate's condition."""
        return self.__condition

    @property
    def parameter_names(self) -> Set[str]:
        return self.__body.parameter_names

    @property
    def parameter_declarations(self) -> Set[str]:
        return self.__body.parameter_declarations

    @property
    def is_interruptable(self) -> bool:
        return self.__body.is_interruptable

    @property
    def defined_channels(self) -> Set['ChannelID']:
        return self.__body.defined_channels

    @property
    def measurement_names(self) -> Set[str]:
        return self.__body.measurement_names

    def __obtain_condition_object(self, conditions: Dict[str, Condition]) -> Condition:
        try:
            return conditions[self.__condition]
        except:
            raise ConditionMissingException(self.__condition)

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: InstructionBlock) -> None:
        self.__obtain_condition_object(conditions).build_sequence_loop(self,
                                                                       self.__body,
                                                                       sequencer,
                                                                       parameters,
                                                                       conditions,
                                                                       measurement_mapping,
                                                                       channel_mapping,
                                                                       instruction_block)

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, Condition]) -> bool:
        return self.__obtain_condition_object(conditions).requires_stop()

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict(
            type=serializer.get_type_identifier(self),
            condition=self.__condition,
            body=serializer.dictify(self.__body)
        )
        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    condition: str,
                    body: Dict[str, Any],
                    identifier: Optional[str]=None) -> 'LoopPulseTemplate':
        body = serializer.deserialize(body)
        return LoopPulseTemplate(condition, body, identifier=identifier)
