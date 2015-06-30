import unittest
from typing import Dict

from pulses.Parameter import Parameter, ConstantParameter
from pulses.Instructions import InstructionBlock, Waveform, WaveformTable
from pulses.Sequencer import SequencingElement, SequencingHardwareInterface, Sequencer

class DummySequencingElement(SequencingElement):

    def __init__(self, requiresStop: bool = False):
        super().__init__()
        self.callCounter = 0
        self.targetBlock = None
        self.timeParameters = None
        self.voltageParameters = None
        self.requiresStop = requiresStop
    
    def build_sequence(self, sequencer: Sequencer, timeParameters: Dict[str, Parameter], voltageParameters: Dict[str, Parameter], instructionBlock: InstructionBlock) -> None:
        self.callCounter += 1
        self.targetBlock = instructionBlock
        self.timeParameters = timeParameters
        self.voltageParameters = voltageParameters
        
    def requires_stop(self) -> bool:
        return self.requiresStop

        
class DummySequencingHardware(SequencingHardwareInterface):

    def __init__(self):
        super().__init__()
        self.waveforms = [] # type: List[WaveformTable]
        

    def register_waveform(self, waveformTable: WaveformTable) -> Waveform:
        self.waveforms.append(waveformTable)
        return Waveform(len(waveformTable))
        
        
class SequencerTest(unittest.TestCase):
    
    def test_initialization(self) -> None:
        dummyHardware = DummySequencingHardware()
        sequencer = Sequencer(dummyHardware)
        self.assertTrue(sequencer.has_finished())
        self.assertFalse(dummyHardware.waveforms)
        
    def test_register_waveform(self) -> None:
        dummyHardware = DummySequencingHardware()
        sequencer = Sequencer(dummyHardware)
        
        wt1 = ((0, 0), (2, 3.5), (5, 3.8))
        wf1 = sequencer.register_waveform(wt1)
        
        wt2 = ((0, 0), (1, 2.7), (2, 3.5))
        wf2 = sequencer.register_waveform(wt2)
        
        wt1b = wt1
        wf1b = sequencer.register_waveform(wt1b)
        
        self.assertIs(wf1, wf1b)
        self.assertEquals([wt1, wt2], dummyHardware.waveforms)
        self.assertTrue(sequencer.has_finished())
        
    def test_push(self) -> None:
        dummyHardware = DummySequencingHardware()
        sequencer = Sequencer(dummyHardware)
        
        tps = {'foo': ConstantParameter(1)}
        vps = {'bar': ConstantParameter(7.3)}
        elem = DummySequencingElement()
        
        sequencer.push(elem, tps, vps)
        self.assertFalse(sequencer.has_finished())
        
    ## code of Sequencer.build() with branch enumeration
    ##def build(self) -> InstructionBlock:
    ##1    mainBlock = InstructionBlock()
    ##1    if not self.has_finished():
    ##2        (element, timeParameters, voltageParameters, targetBlock) = self._sequencingStack.pop()
    ##3        while True: # there seems to be no do-while loop in python and "while True" with a break condition is a suggested solution
    ##3            if targetBlock is None:
    ##4                targetBlock = mainBlock
    ##5            element.build_sequence(self, timeParameters, voltageParameters, targetBlock)
    ##5            if (self.has_finished()):
    ##                 break
    ##6            if (self._sequencingStack[-1].requires_stop()):
    ##                 break
    ##7            (element, timeParameters, voltageParameters, targetBlock) = self._sequencingStack.pop()
    ##    
    ##8    mainBlock.finalize()
    ##8    return mainBlock
    
    # zero and single iteration path covering tests; one two iteration test which completes at least branch covering
    def test_build_path_1_8(self) -> None:
        dummyHardware = DummySequencingHardware()
        sequencer = Sequencer(dummyHardware)
        
        block = sequencer.build()
        self.assertTrue(sequencer.has_finished())
        self.assertFalse(dummyHardware.waveforms)
        self.assertEqual(1, len(block)) # STOP instruction is always generated, so len(block) is 1
        self.assertTrue(block.finalized)
        
    
    def test_build_path_1_2_3_5_8(self) -> None:
        dummyHardware = DummySequencingHardware()
        sequencer = Sequencer(dummyHardware)
    
        elem = DummySequencingElement(True)
        tps = {'foo': ConstantParameter(1)}
        vps = {'bar': ConstantParameter(7.3)}
        targetBlock = InstructionBlock()
        sequencer.push(elem, tps, vps, targetBlock)
        block = sequencer.build()
        
        self.assertTrue(sequencer.has_finished())
        self.assertTrue(block.finalized)
        self.assertIs(targetBlock, elem.targetBlock)
        self.assertEqual(tps, elem.timeParameters)
        self.assertEqual(vps, elem.voltageParameters)
        self.assertEqual(1, elem.callCounter)
        
        
    def test_build_path_1_2_3_4_5_8(self) -> None:
        dummyHardware = DummySequencingHardware()
        sequencer = Sequencer(dummyHardware)
    
        elem = DummySequencingElement(True)
        tps = {'foo': ConstantParameter(1)}
        vps = {'bar': ConstantParameter(7.3)}
        targetBlock = InstructionBlock()
        sequencer.push(elem, tps, vps)
        block = sequencer.build()
        
        self.assertTrue(sequencer.has_finished())
        self.assertTrue(block.finalized)
        self.assertIs(block, elem.targetBlock)
        self.assertEqual(tps, elem.timeParameters)
        self.assertEqual(vps, elem.voltageParameters)
        self.assertEqual(1, elem.callCounter)
    
    def test_build_path_1_2_3_5_6_8(self) -> None:
        dummyHardware = DummySequencingHardware()
        sequencer = Sequencer(dummyHardware)
    
        stopElem = DummySequencingElement(True)
        sequencer.push(stopElem, [], [])
        
        elem = DummySequencingElement(True)
        tps = {'foo': ConstantParameter(1)}
        vps = {'bar': ConstantParameter(7.3)}
        targetBlock = InstructionBlock()
        sequencer.push(elem, tps, vps)
        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertTrue(block.finalized)
        self.assertIs(block, elem.targetBlock)
        self.assertEqual(tps, elem.timeParameters)
        self.assertEqual(vps, elem.voltageParameters)
        self.assertEqual(1, elem.callCounter)
    
    def test_build_path_1_2_3_4_5_6_8(self) -> None:
        dummyHardware = DummySequencingHardware()
        sequencer = Sequencer(dummyHardware)
    
        stopElem = DummySequencingElement(True)
        sequencer.push(stopElem, [], [])
        
        elem = DummySequencingElement(True)
        tps = {'foo': ConstantParameter(1)}
        vps = {'bar': ConstantParameter(7.3)}
        targetBlock = InstructionBlock()
        sequencer.push(elem, tps, vps, targetBlock)
        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertTrue(block.finalized)
        self.assertIs(targetBlock, elem.targetBlock)
        self.assertEqual(tps, elem.timeParameters)
        self.assertEqual(vps, elem.voltageParameters)
        self.assertEqual(1, elem.callCounter)
    
    
    #path 1-2-3-4-5-6-7-3-5-8 (branch covering with the above, two loop iterations)
    def test_build_path_1_2_3_4_5_6_7_3_5_8(self) -> None:
        dummyHardware = DummySequencingHardware()
        sequencer = Sequencer(dummyHardware)
    
        secondElem = DummySequencingElement(False)
        targetBlock = InstructionBlock()
        sequencer.push(secondElem, [], [], targetBlock)
        
        elem = DummySequencingElement(False)
        tps = {'foo': ConstantParameter(1)}
        vps = {'bar': ConstantParameter(7.3)}
        sequencer.push(elem, tps, vps)
        block = sequencer.build()
        
        self.assertTrue(sequencer.has_finished())
        self.assertTrue(block.finalized)
        
        self.assertIs(block, elem.targetBlock)
        self.assertEqual(tps, elem.timeParameters)
        self.assertEqual(vps, elem.voltageParameters)
        self.assertEqual(1, elem.callCounter)
        
        self.assertIs(targetBlock, secondElem.targetBlock)
        self.assertEqual([], secondElem.timeParameters)
        self.assertEqual([], secondElem.voltageParameters)
        self.assertEqual(1, secondElem.callCounter)
    
    
    
    
    
    
    
    
    