import unittest
from typing import Dict

from pulses.Parameter import Parameter, ConstantParameter
from pulses.Instructions import InstructionBlock, Waveform, WaveformTable
from pulses.Sequencer import SequencingElement, SequencingHardwareInterface, Sequencer

class DummySequencingElement(SequencingElement):

    def __init__(self, requires_stop: bool = False):
        super().__init__()
        self.call_counter = 0
        self.target_block = None
        self.time_parameters = None
        self.voltage_parameters = None
        self.requires_stop_ = requires_stop
    
    def build_sequence(self, sequencer: Sequencer, time_parameters: Dict[str, Parameter], voltage_parameters: Dict[str, Parameter], instruction_block: InstructionBlock) -> None:
        self.call_counter += 1
        self.target_block = instruction_block
        self.time_parameters = time_parameters
        self.voltage_parameters = voltage_parameters
        
    def requires_stop(self, time_parameters: Dict[str, Parameter], voltage_parameters: Dict[str, Parameter]) -> bool:
        return self.requires_stop_

        
class DummySequencingHardware(SequencingHardwareInterface):

    def __init__(self):
        super().__init__()
        self.waveforms = [] # type: List[WaveformTable]
        

    def register_waveform(self, waveform_table: WaveformTable) -> Waveform:
        self.waveforms.append(waveform_table)
        return Waveform(len(waveform_table))
        
        
class SequencerTest(unittest.TestCase):
    
    def test_initialization(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
        self.assertTrue(sequencer.has_finished())
        self.assertFalse(dummy_hardware.waveforms)
        
    def test_register_waveform(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
        
        wt1 = ((0, 0), (2, 3.5), (5, 3.8))
        wf1 = sequencer.register_waveform(wt1)
        
        wt2 = ((0, 0), (1, 2.7), (2, 3.5))
        wf2 = sequencer.register_waveform(wt2)
        
        wt1b = wt1
        wf1b = sequencer.register_waveform(wt1b)
        
        self.assertIs(wf1, wf1b)
        self.assertNotEqual(wf1, wf2)
        self.assertEqual([wt1, wt2], dummy_hardware.waveforms)
        self.assertTrue(sequencer.has_finished())
        
    def test_push(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
        
        tps = {'foo': ConstantParameter(1)}
        vps = {'bar': ConstantParameter(7.3)}
        elem = DummySequencingElement()
        
        sequencer.push(elem, tps, vps)
        self.assertFalse(sequencer.has_finished())
        
    ## code of Sequencer.build() with branch enumeration
    ##def build(self) -> InstructionBlock:
    ##1    main_block = InstructionBlock()
    ##1    if not self.has_finished():
    ##2        (element, time_parameters, voltage_parameters, target_block) = self.__sequencing_stack.pop()
    ##3        while True: # there seems to be no do-while loop in python and "while True" with a break condition is a suggested solution
    ##3            if target_block is None:
    ##4                target_block = main_block
    ##5            element.build_sequence(self, time_parameters, voltage_parameters, target_block)
    ##5            if (self.has_finished()):
    ##                 break
    ##6            if (self.__sequencing_stack[-1].requires_stop()):
    ##                 break
    ##7            (element, time_parameters, voltage_parameters, target_block) = self.__sequencing_stack.pop()
    ##    
    ##8    main_block.finalize()
    ##8    return main_block
    
    # zero and single iteration path covering tests; one two iteration test which completes at least branch covering
    def test_build_path_1_8(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
        
        block = sequencer.build()
        self.assertTrue(sequencer.has_finished())
        self.assertFalse(dummy_hardware.waveforms)
        self.assertEqual(0, len(block))
        
    
    def test_build_path_1_2_3_5_8(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
    
        elem = DummySequencingElement(True)
        tps = {'foo': ConstantParameter(1)}
        vps = {'bar': ConstantParameter(7.3)}
        target_block = InstructionBlock()
        sequencer.push(elem, tps, vps, target_block)
        sequencer.build()
        
        self.assertTrue(sequencer.has_finished())
        self.assertIs(target_block, elem.target_block)
        self.assertEqual(tps, elem.time_parameters)
        self.assertEqual(vps, elem.voltage_parameters)
        self.assertEqual(1, elem.call_counter)
        
        
    def test_build_path_1_2_3_4_5_8(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
    
        elem = DummySequencingElement(True)
        tps = {'foo': ConstantParameter(1)}
        vps = {'bar': ConstantParameter(7.3)}
        sequencer.push(elem, tps, vps)
        block = sequencer.build()
        
        self.assertTrue(sequencer.has_finished())
        self.assertIs(block, elem.target_block)
        self.assertEqual(tps, elem.time_parameters)
        self.assertEqual(vps, elem.voltage_parameters)
        self.assertEqual(1, elem.call_counter)
    
    def test_build_path_1_2_3_5_6_8(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
    
        stop_elem = DummySequencingElement(True)
        sequencer.push(stop_elem, [], [])
        
        elem = DummySequencingElement(True)
        tps = {'foo': ConstantParameter(1)}
        vps = {'bar': ConstantParameter(7.3)}
        sequencer.push(elem, tps, vps)
        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertIs(block, elem.target_block)
        self.assertEqual(tps, elem.time_parameters)
        self.assertEqual(vps, elem.voltage_parameters)
        self.assertEqual(1, elem.call_counter)
    
    def test_build_path_1_2_3_4_5_6_8(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
    
        stop_elem = DummySequencingElement(True)
        sequencer.push(stop_elem, [], [])
        
        elem = DummySequencingElement(True)
        tps = {'foo': ConstantParameter(1)}
        vps = {'bar': ConstantParameter(7.3)}
        target_block = InstructionBlock()
        sequencer.push(elem, tps, vps, target_block)
        sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertIs(target_block, elem.target_block)
        self.assertEqual(tps, elem.time_parameters)
        self.assertEqual(vps, elem.voltage_parameters)
        self.assertEqual(1, elem.call_counter)
    
    
    #path 1-2-3-4-5-6-7-3-5-8 (branch covering with the above, two loop iterations)
    def test_build_path_1_2_3_4_5_6_7_3_5_8(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
    
        second_elem = DummySequencingElement(False)
        target_block = InstructionBlock()
        sequencer.push(second_elem, [], [], target_block)
        
        elem = DummySequencingElement(False)
        tps = {'foo': ConstantParameter(1)}
        vps = {'bar': ConstantParameter(7.3)}
        sequencer.push(elem, tps, vps)
        block = sequencer.build()
        
        self.assertTrue(sequencer.has_finished())
        
        self.assertIs(block, elem.target_block)
        self.assertEqual(tps, elem.time_parameters)
        self.assertEqual(vps, elem.voltage_parameters)
        self.assertEqual(1, elem.call_counter)
        
        self.assertIs(target_block, second_elem.target_block)
        self.assertEqual([], second_elem.time_parameters)
        self.assertEqual([], second_elem.voltage_parameters)
        self.assertEqual(1, second_elem.call_counter)
    
    
    
    
    
    
    
    
    