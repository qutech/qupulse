import logging
from typing import Dict, Set, List, Optional, Any

"""RELATED THIRD PARTY IMPORTS"""

"""LOCAL IMPORTS"""
from .Parameter import Parameter
from .PulseTemplate import PulseTemplate, MeasurementWindow
from .Condition import Condition
from .Instructions import InstructionBlock
from .Sequencer import Sequencer
from .Serializer import Serializer

logger = logging.getLogger(__name__)


class LoopPulseTemplate(PulseTemplate):
    """Conditional looping in a pulse.
    
    A LoopPulseTemplate is a PulseTemplate which is repeated
    during execution as long as a certain condition holds.
    """
    
    def __init__(self, condition: Condition, body: PulseTemplate, identifier: Optional[str]=None) -> None:
        super().__init__(identifier=identifier)
        self.__condition = condition
        self.__body = body

    def __str__(self) -> str:
        raise NotImplementedError()

    @property
    def parameter_names(self) -> Set[str]:
        return self.__body.parameter_names

    @property
    def parameter_declarations(self) -> Set[str]:
        return self.__body.parameter_declarations

    def get_measurement_windows(self, parameters: Dict[str, Parameter] = None) -> List[MeasurementWindow]:
        """Return all measurement windows defined in this PulseTemplate."""
        raise NotImplementedError()

    @property
    def is_interruptable(self) -> bool:
        return self.__body.is_interruptable

    def build_sequence(self, sequencer: Sequencer, parameters: Dict[str, Parameter], instruction_block: InstructionBlock):
        self.__condition.build_sequence_loop(self, self.__body, sequencer, parameters, instruction_block)

    def requires_stop(self, parameters: Dict[str, Parameter]):
        return self.__condition.requires_stop

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
        