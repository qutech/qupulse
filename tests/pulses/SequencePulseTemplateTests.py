import unittest
import os
import sys
import copy

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from pulses.TablePulseTemplate import TablePulseTemplate, TableEntry
from pulses.SequencePulseTemplate import SequencePulseTemplate, MissingMappingException, UnnecessaryMappingException, MissingParameterDeclarationException, RuntimeMappingError
from pulses.PulseTemplate import ParameterNotInPulseTemplateException,\
    PulseTemplate
from pulses.Parameter import ParameterDeclaration, Parameter, ParameterNotProvidedException, ConstantParameter
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

        self.mapping1 = {
            'up': 'lambda uptime: uptime',
            'down': 'lambda uptime, length: uptime + length',
            'v': 'lambda voltage: voltage',
            'length': 'lambda pulse_length: 0.5 * pulse_length'
        }

        self.outer_parameters = ['uptime', 'length', 'pulse_length', 'voltage']

        self.parameters = {}
        self.parameters['uptime'] = ConstantParameter(5)
        self.parameters['length'] = ConstantParameter(10)
        self.parameters['pulse_length'] = ConstantParameter(100)
        self.parameters['voltage'] = ConstantParameter(10)

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
        sequence = SequencePulseTemplate(subtemplates, self.parameters.keys())
        sequence.build_sequence(sequencer, self.parameters, instruction_block)
        # TODO: use real sequencer and check output

    def test_requires_stop(self) -> None:
        pass #TODO: implement

    def test_runtime_mapping_exception(self):
        mapping = self.mapping1
        mapping['up'] = "lambda foo: foo"

        subtemplates = [(self.square, mapping)]
        with self.assertRaises(MissingParameterDeclarationException):
            sequence = SequencePulseTemplate(subtemplates, self.outer_parameters)

class SequencePulseTemplateStringTest(unittest.TestCase):
    def test_str(self):
        T = TablePulseTemplate()
        a = [RuntimeMappingError(T,T,"c","d"),
             UnnecessaryMappingException(T,"b"),
             MissingMappingException(T,"b"),
             MissingParameterDeclarationException(T, "c")]
        
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

if __name__ == "__main__":
    unittest.main(verbosity=2)
