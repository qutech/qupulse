import unittest
import os
import sys
import copy

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from pulses.TablePulseTemplate import TablePulseTemplate, TableEntry
from pulses.SequencePulseTemplate import SequencePulseTemplate, MissingMappingException, UnnecessaryMappingException, RuntimeMappingError
from pulses.PulseTemplate import ParameterNotInPulseTemplateException,\
    PulseTemplate
from pulses.Parameter import ParameterDeclaration, Parameter, ParameterNotProvidedException
from pulses.Instructions import EXECInstruction
from tests.pulses.SequencingDummies import DummySequencer, DummyInstructionBlock, DummySequencingElement


class DummyParameter(Parameter):

    def __init__(self, value: float = 0, requires_stop: bool = False) -> None:
        super().__init__()
        self.__value = value
        self.__requires_stop = requires_stop

    def get_value(self) -> float:
        return self.__value

    @property
    def requires_stop(self) -> bool:
        return self.__requires_stop


class SequencePulseTemplateTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Setup test data
        self.square = TablePulseTemplate()
        self.square.add_entry('up', 'v', 'hold')
        self.square.add_entry('down', 0, 'hold')
        self.square.add_entry('length', 0)

        self.mapping1 = {}
        self.mapping1['up'] = lambda ps: ps['uptime']
        self.mapping1['down'] = lambda ps: ps['uptime'] + ps['length']
        self.mapping1['v'] = lambda ps: ps['voltage']
        self.mapping1['length'] = lambda ps: ps['pulse-length'] * 0.5

        self.outer_parameters = ['uptime', 'length', 'pulse-length', 'voltage']

        self.parameters = {}
        self.parameters['uptime'] = 5
        self.parameters['length'] = 10
        self.parameters['pulse-length'] = 100
        self.parameters['voltage'] = 10

        self.sequence = SequencePulseTemplate([(self.square, self.mapping1)], self.outer_parameters)

    def test_missing_mapping(self):
        mapping = self.mapping1
        mapping.pop('v')

        subtemplates = [(self.square, mapping)]
        with self.assertRaises(MissingMappingException):
            SequencePulseTemplate(subtemplates, self.outer_parameters)

    def test_unnecessary_mapping(self):
        mapping = self.mapping1
        mapping['unnecessary'] = lambda ps: ps['voltage']

        subtemplates = [(self.square, mapping)]
        with self.assertRaises(UnnecessaryMappingException):
            SequencePulseTemplate(subtemplates, self.outer_parameters)

    def test_identifier(self):
        identifier = 'some name'
        pulse = SequencePulseTemplate([], [], identifier=identifier)
        self.assertEqual(identifier, pulse.identifier)



class SequencePulseTemplateSequencingTests(SequencePulseTemplateTest):
    def test_missing_parameter(self):
        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        parameters = copy.deepcopy(self.parameters)
        parameters.pop('uptime')
        with self.assertRaises(ParameterNotProvidedException):
            self.sequence.build_sequence(sequencer, parameters, block)

    def test_build_sequence(self) -> None:
        sequencer = DummySequencer()
        instruction_block = DummyInstructionBlock()
        block = DummyInstructionBlock()
        element1 = DummySequencingElement(push_elements=(block, [self.square]))
        element2 = DummySequencingElement()
        mapping = {}
        subtemplates = [(self.square, self.mapping1),
                        (element2, mapping)]
        sequence = SequencePulseTemplate(subtemplates, [])
        sequence.build_sequence(sequencer, self.parameters, instruction_block)
        # TODO: use real sequencer and check output

    def test_requires_stop(self) -> None:
        pass #TODO: implement

    def test_runtime_mapping_exception(self):
        mapping = self.mapping1
        mapping['up'] = lambda ps: ps['parameter that does not exist']

        subtemplates = [(self.square, mapping)]
        sequence = SequencePulseTemplate(subtemplates, self.outer_parameters)
        sequencer = DummySequencer()
        block = DummyInstructionBlock()
        with self.assertRaises(RuntimeMappingError):
            sequence.build_sequence(sequencer, self.parameters, block)

class SequencePulseTemplateStringTest(unittest.TestCase):
    def test_str(self):
        T = TablePulseTemplate()
        a = [RuntimeMappingError(T,T,"c","d"),
             UnnecessaryMappingException(T,"b"),
             MissingMappingException(T,"b")]
        
        b = [x.__str__() for x in a]
        for s in b:
            self.assertIsInstance(s, str)
            
class SequencePulseTemplateTestProperties(SequencePulseTemplateTest):
    def test_is_interruptable(self):
        self.assertTrue(self.sequence.is_interruptable)
        self.sequence.is_interruptable = False
        self.assertFalse(self.sequence.is_interruptable)
        
    def test_parameter_declarations(self):
        #FIXME
        decl = self.sequence.parameter_declarations
        pass
if __name__ == "__main__":
    unittest.main(verbosity=2)
