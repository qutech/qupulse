"""This module defines LoopPulseTemplate, a higher-order hierarchical pulse template that loops
another PulseTemplate based on a condition."""


from typing import Dict, Set, Optional, Any, Union, Tuple, Generator

from qctoolkit.serialization import Serializer

from qctoolkit.expressions import Expression
from qctoolkit.utils import checked_int_cast
from qctoolkit.pulses.parameters import Parameter, ConstantParameter, InvalidParameterNameException
from qctoolkit.pulses.pulse_template import PulseTemplate, ChannelID
from qctoolkit.pulses.conditions import Condition, ConditionMissingException
from qctoolkit.pulses.instructions import InstructionBlock
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.sequence_pulse_template import SequenceWaveform as ForLoopWaveform

__all__ = ['WhileLoopPulseTemplate', 'ConditionMissingException']


class LoopPulseTemplate(PulseTemplate):
    """Base class for loop based pulse templates"""
    def __init__(self, body: PulseTemplate, identifier: Optional[str]=None):
        super().__init__(identifier=identifier)
        self.__body = body

    @property
    def body(self) -> PulseTemplate:
        return self.__body

    @property
    def defined_channels(self) -> Set['ChannelID']:
        return self.__body.defined_channels

    @property
    def measurement_names(self) -> Set[str]:
        return self.__body.measurement_names

    @property
    def is_interruptable(self):
        raise NotImplementedError()  # pragma: no cover


class ParametrizedRange:
    """Parametrized range """
    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise TypeError('ParametrizedRange only takes either positional or keyword arguments')
        elif kwargs:
            start = kwargs['start']
            stop = kwargs['stop']
            step = kwargs['step']
        elif len(args) in (1, 2, 3):
            if len(args) == 3:
                start, stop, step = args
            elif len(args) == 2:
                (start, stop), step = args, 1
            elif len(args) == 1:
                start, (stop,), step = 0, args, 1
        else:
            raise TypeError('ParametrizedRange expected 1 to 3 arguments, got {}'.format(len(args)))

        self.start = Expression(start)
        self.stop = Expression(stop)
        self.step = Expression(step)

    def to_tuple(self) -> Tuple[Any, Any, Any]:
        """Return a simple representation of the range which is useful for comparison and serialization"""
        return (self.start.get_most_simple_representation(),
                self.stop.get_most_simple_representation(),
                self.step.get_most_simple_representation())

    def to_range(self, parameters: Dict[str, Any]) -> range:
        return range(checked_int_cast(self.start.evaluate_numeric(**parameters)),
                     checked_int_cast(self.stop.evaluate_numeric(**parameters)),
                     checked_int_cast(self.step.evaluate_numeric(**parameters)))

    @property
    def parameter_names(self) -> Set[str]:
        return set(self.start.variables) | set(self.stop.variables) | set(self.step.variables)


class ForLoopPulseTemplate(LoopPulseTemplate):
    def __init__(self,
                 body: PulseTemplate,
                 loop_index: str,
                 loop_range: Union[int,
                                   range,
                                   str,
                                   Tuple[Any, Any],
                                   Tuple[Any, Any, Any],
                                   ParametrizedRange],
                 identifier: Optional[str]=None):
        super().__init__(body=body, identifier=identifier)

        if isinstance(loop_range, ParametrizedRange):
            self._loop_range = loop_range
        elif isinstance(loop_range, (int, str)):
            self._loop_range = ParametrizedRange(loop_range)
        elif isinstance(loop_range, (tuple, list)):
            self._loop_range = ParametrizedRange(*loop_range)
        elif isinstance(loop_range, range):
            self._loop_range = ParametrizedRange(start=loop_range.start,
                                                 stop=loop_range.stop,
                                                 step=loop_range.step)
        else:
            raise ValueError('loop_range is not valid')

        if not loop_index.isidentifier():
            raise InvalidParameterNameException(loop_index)
        body_parameters = self.body.parameter_names
        if loop_index not in body_parameters:
            raise LoopIndexNotUsedException(loop_index, body_parameters)
        self._loop_index = loop_index

    @property
    def loop_index(self) -> str:
        return self._loop_index

    @property
    def loop_range(self) -> ParametrizedRange:
        return self._loop_range

    @property
    def duration(self) -> Expression:
        count = (self._loop_range.stop.sympified_expression - self._loop_range.start.sympified_expression)
        step = self._loop_range.step.sympified_expression
        return Expression((count - count % step)/step * self.body.duration.sympified_expression)

    @property
    def parameter_names(self) -> Set[str]:
        parameter_names = self.body.parameter_names.copy()
        parameter_names.remove(self._loop_index)
        return parameter_names | self._loop_range.parameter_names

    def _body_parameter_generator(self, parameters: Dict[str, Parameter], forward=True) -> Generator:
        loop_range_parameters = dict((parameter_name, parameters[parameter_name].get_value())
                                     for parameter_name in self._loop_range.parameter_names)
        loop_range = self._loop_range.to_range(loop_range_parameters)

        parameters = dict((parameter_name, parameters[parameter_name])
                          for parameter_name in self.body.parameter_names if parameter_name != self._loop_index)
        loop_range = loop_range if forward else reversed(loop_range)
        for loop_index_value in loop_range:
            local_parameters = parameters.copy()
            local_parameters[self._loop_index] = ConstantParameter(loop_index_value)
            yield local_parameters

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: InstructionBlock) -> None:
        for local_parameters in self._body_parameter_generator(parameters, forward=False):
            sequencer.push(self.body,
                           parameters=local_parameters,
                           conditions=conditions,
                           window_mapping=measurement_mapping,
                           channel_mapping=channel_mapping,
                           target_block=instruction_block)

    def build_waveform(self, parameters: Dict[str, Parameter]) -> ForLoopWaveform:
        return ForLoopWaveform([self.body.build_waveform(local_parameters)
                                for local_parameters in self._body_parameter_generator(parameters, forward=True)])

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return any(parameters[parameter_name].requires_stop for parameter_name in self._loop_range.parameter_names)

    def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
        data = dict(
            body=serializer.dictify(self.body),
            loop_range=self._loop_range.to_tuple(),
            loop_index=self._loop_index
        )
        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    body: Dict[str, Any],
                    loop_range: Tuple,
                    loop_index: str,
                    identifier: Optional[str]=None) -> 'ForLoopPulseTemplate':
        body = serializer.deserialize(body)
        return ForLoopPulseTemplate(body=body,
                                    identifier=identifier,
                                    loop_range=loop_range,
                                    loop_index=loop_index)


class WhileLoopPulseTemplate(LoopPulseTemplate):
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
        super().__init__(body=body, identifier=identifier)
        self._condition = condition

    def __str__(self) -> str:
        return "LoopPulseTemplate: Condition <{}>, Body <{}>".format(self._condition, self.body)

    @property
    def condition(self) -> str:
        """This LoopPulseTemplate's condition."""
        return self._condition

    @property
    def parameter_names(self) -> Set[str]:
        return self.body.parameter_names

    @property
    def duration(self) -> Expression:
        return Expression('nan')

    @property
    def parameter_declarations(self) -> Set[str]:
        return self.body.parameter_declarations

    def __obtain_condition_object(self, conditions: Dict[str, Condition]) -> Condition:
        try:
            return conditions[self._condition]
        except:
            raise ConditionMissingException(self._condition)

    def build_sequence(self,
                       sequencer: Sequencer,
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, Condition],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: InstructionBlock) -> None:
        self.__obtain_condition_object(conditions).build_sequence_loop(self,
                                                                       self.body,
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
            condition=self._condition,
            body=serializer.dictify(self.body)
        )
        return data

    @staticmethod
    def deserialize(serializer: Serializer,
                    condition: str,
                    body: Dict[str, Any],
                    identifier: Optional[str]=None) -> 'WhileLoopPulseTemplate':
        body = serializer.deserialize(body)
        result = WhileLoopPulseTemplate(condition=condition,
                                        body=body,
                                        identifier=identifier)
        return result


class LoopIndexNotUsedException(Exception):
    def __init__(self, loop_index: str, body_parameter_names: Set[str]):
        self.loop_index = loop_index
        self.body_parameter_names = body_parameter_names

    def __str__(self) -> str:
        return "The parameter {} is missing in the body's parameter names: {}".format(self.loop_index,
                                                                                      self.body_parameter_names)
