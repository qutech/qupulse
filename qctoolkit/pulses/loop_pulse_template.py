import logging
from typing import Dict, Set, List, Optional, Any, Union

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from .parameters import Parameter
from .pulse_template import PulseTemplate, MeasurementWindow
from .conditions import Condition, SoftwareCondition, ProxyCondition
from .instructions import InstructionBlock
from .sequencing import Sequencer
from qctoolkit.serialization import Serializer

logger = logging.getLogger(__name__)


class LoopPulseTemplate(PulseTemplate):
    """Conditional looping in a pulse.
    
    A LoopPulseTemplate is a PulseTemplate which is repeated
    during execution as long as a certain condition holds.
    """
    
    def __init__(self, condition: Union[Condition, str], body: PulseTemplate, identifier: Optional[str]=None) -> None:
        super().__init__(identifier=identifier)
        if isinstance(condition, str):
            condition = ProxyCondition(condition)
        self.__condition = condition
        self.__body = body

    @staticmethod
    def create_constant_for_loop(count: int, body: PulseTemplate, identifier: Optional[str]=None) -> 'LoopPulseTemplate':
        for_condition = SoftwareCondition(lambda i, count=count: i < count)
        return LoopPulseTemplate(for_condition, body, identifier=identifier)

    def __str__(self) -> str:
        return "LoopPulseTemplate: Condition <{}>, Body <{}>".format(self.__condition, self.__body)

    @property
    def parameter_names(self) -> Set[str]:
        return self.__body.parameter_names

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None):
        raise NotImplemented()

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
        self.__condition.build_sequence_loop(self, self.__body, sequencer, parameters, conditions, instruction_block)

    def requires_stop(self, parameters: Dict[str, Parameter], conditions: Dict[str, 'Condition']) -> bool:
        return self.__condition.requires_stop()

    def get_serialization_data(self, serializer: Serializer):
        data = dict(
            type=serializer.get_type_identifier(self),
            body=serializer._serialize_subpulse(self.__body)
        )
        # TODO: serialize condition
        raise NotImplementedError()
        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    condition: Dict[str, Any],
                    body: Dict[str, Any],
                    identifier: Optional[str]=None):
        body = serializer.deserialize(body)
        #TODO: deserialize condition
        raise NotImplementedError()
        