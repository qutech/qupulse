import unittest
import os
import sys
from typing import Any

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from tests.pulses.SequencingDummies import DummyWaveform, DummyInstructionBlock

from pulses.Instructions import InstructionBlockAlreadyFinalizedException,InstructionBlock, InstructionPointer,\
    InstructionBlockNotYetPlacedException, Trigger, CJMPInstruction, GOTOInstruction,EXECInstruction, STOPInstruction,\
    MissingReturnAddressException, InstructionSequence, Comparable


class DummyComparable(Comparable):

    def __init__(self, compare_key: Any) -> None:
        super().__init__()
        self.compare_key_ = compare_key

    @property
    def _compare_key(self) -> Any:
        return self.compare_key_


class ComparableTests(unittest.TestCase):

    def test_hash(self) -> None:
        comp_a = DummyComparable(17)
        self.assertEqual(hash(17), hash(comp_a))

    def test_eq(self) -> None:
        comp_a = DummyComparable(17)
        comp_b = DummyComparable(18)
        comp_c = DummyComparable(18)
        self.assertNotEqual(comp_a, comp_b)
        self.assertNotEqual(comp_b, comp_a)
        self.assertEqual(comp_b, comp_c)
        self.assertNotEqual(comp_a, "foo")
        self.assertNotEqual("foo", comp_a)

 
class InstructionPointerTest(unittest.TestCase):

    def test_invalid_offset(self):
        block = InstructionBlock()
        self.assertRaises(ValueError, InstructionPointer, block, -1)
        self.assertRaises(ValueError, InstructionPointer, block, -12)
        
    def test_initialization_main_block(self):
        block = InstructionBlock()
        for offset in [0, 1, 924]:
            ip = InstructionPointer(block, offset)
            self.assertIs(block, ip.block)
            self.assertEqual(offset, ip.offset)
            self.assertEqual(offset, ip.get_absolute_address())
            
    def test_initialization_relative_block(self):
        block = InstructionBlock().create_embedded_block()
        for offset in [0, 1, 924]:
            ip = InstructionPointer(block, offset)
            self.assertIs(block, ip.block)
            self.assertEqual(offset, ip.offset)
            self.assertRaises(InstructionBlockNotYetPlacedException, ip.get_absolute_address)
            
    def test_equality(self):
        blocks = [InstructionBlock(), InstructionBlock()]
        blocks.append(blocks[0].create_embedded_block())
        ips = []
        for block in blocks:
            for offset in [0, 1, 2352]:
                ip = InstructionPointer(block, offset)
                self.assertEqual(ip, ip)
                for other in ips:
                    self.assertNotEqual(ip, other)
                    self.assertNotEqual(other, ip)
                    self.assertNotEqual(hash(ip), hash(other))
                ips.append(ip)

        
class TriggerTest(unittest.TestCase):
    
    def test_equality(self):
        t1 = Trigger()
        t2 = Trigger()
        self.assertEqual(t1, t1)
        self.assertNotEqual(t1, t2)
        self.assertNotEqual(t2, t1)
        self.assertNotEqual(hash(t1), hash(t2))


class CJMPInstructionTest(unittest.TestCase):
    
    def test_initialization(self):
        block = InstructionBlock()
        trigger = Trigger()
        for offset in [0, 1, 23]:
            instr = CJMPInstruction(trigger, block, offset)
            self.assertEqual(trigger, instr.trigger)
            self.assertEqual(block, instr.target.block)
            self.assertEqual(offset, instr.target.offset)
        
    def test_equality(self):
        blocks = [InstructionBlock(), InstructionBlock()]
        for offset in [0, 1, 23]:
            instrA = CJMPInstruction(0, blocks[0], offset)
            instrB = CJMPInstruction(0, blocks[0], offset)
            self.assertEqual(instrA, instrB)
            self.assertEqual(instrB, instrA)
        instrs = []
        for trigger in [Trigger(), Trigger()]:
            for block in blocks:
                for offset in [0, 17]:
                    instruction = CJMPInstruction(trigger, block, offset)
                    self.assertEqual(instruction, instruction)
                    for other in instrs:
                        self.assertNotEqual(instruction, other)
                        self.assertNotEqual(other, instruction)
                        self.assertNotEqual(hash(instruction), hash(other))
                    instrs.append(instruction)

    def test_str(self) -> None:
        block = DummyInstructionBlock()
        trigger = Trigger()
        instr = CJMPInstruction(trigger, block, 3)
        self.assertEqual("cjmp to {} on {}".format(InstructionPointer(block, 3), trigger), str(instr))


class GOTOInstructionTest(unittest.TestCase):
    
    def test_initialization(self) -> None:
        block = InstructionBlock()
        for offset in [0, 1, 23]:
            instr = GOTOInstruction(block, offset)
            self.assertIs(block, instr.target.block)
            self.assertEqual(offset, instr.target.offset)
        
    def test_equality(self) -> None:
        blocks = [InstructionBlock(), InstructionBlock()]
        for offset in [0, 1, 23]:
            instrA = GOTOInstruction(blocks[0], offset)
            instrB = GOTOInstruction(blocks[0], offset)
            self.assertEqual(instrA, instrB)
            self.assertEqual(instrB, instrA)
        instrs = []
        for block in blocks:
            for offset in [0, 17]:
                instruction = GOTOInstruction(block, offset)
                self.assertEqual(instruction, instruction)
                for other in instrs:
                    self.assertNotEqual(instruction, other)
                    self.assertNotEqual(other, instruction)
                    self.assertNotEqual(hash(instruction), hash(other))
                instrs.append(instruction)

    def test_str(self) -> None:
        block = DummyInstructionBlock()
        instr = GOTOInstruction(block, 3)
        self.assertEqual("goto to {}".format(str(InstructionPointer(block, 3))), str(instr))


class EXECInstructionTest(unittest.TestCase):
    
    def test_initialization(self):
        waveform = DummyWaveform()
        instr = EXECInstruction(waveform)
        self.assertIs(waveform, instr.waveform)
        
    def test_equality(self):
        wf1 = DummyWaveform()
        wf2 = DummyWaveform()
        instr11 = EXECInstruction(wf1)
        instr12 = EXECInstruction(wf1)
        instr20 = EXECInstruction(wf2)
        self.assertEqual(instr11, instr11)
        self.assertEqual(instr11, instr12)
        self.assertEqual(instr12, instr11)
        self.assertNotEqual(instr11, instr20)
        self.assertNotEqual(instr20, instr11)
        self.assertEqual(hash(instr11), hash(instr12))
        self.assertNotEqual(hash(instr11), hash(instr20))

    def test_str(self) -> None:
        wf = DummyWaveform()
        instr = EXECInstruction(wf)
        self.assertEqual("exec {}".format(str(wf)), str(instr))


class STOPInstructionTest(unittest.TestCase):
    
    def test_str(self):
        instr = STOPInstruction()
        self.assertEqual('stop', str(instr))
        
    def test_equality(self):
        instr1 = STOPInstruction()
        instr2 = STOPInstruction()
        self.assertEqual(instr1, instr1)
        self.assertEqual(instr1, instr2)
        self.assertEqual(instr2, instr1)
        self.assertEqual(hash(instr1), hash(instr2))


class InstructionBlockTest(unittest.TestCase):

    def __verify_block(self, block: InstructionBlock, expected_instructions: InstructionSequence, expected_compiled_instructions: InstructionSequence) -> None:
        self.assertEqual(len(expected_instructions), len(block))
        self.assertEqual(expected_instructions, block.instructions)
        self.assertEqual(expected_compiled_instructions, block.compile_sequence())

    def test_empty_unreturning_main_block(self):
        block = InstructionBlock()
        self.__verify_block(block, [], [STOPInstruction()])
        self.assertEqual(0, block.get_start_address())
        self.assertIsNone(block.return_ip)
        self.__verify_block(block, [], [STOPInstruction()])
        
    def test_empty_returning_main_block(self):
        block = InstructionBlock()
        self.assertEqual(0, block.get_start_address())
        self.assertIsNone(block.return_ip)
        ip = InstructionPointer(InstructionBlock(), 7)
        block.return_ip = ip # must have no effect
        self.__verify_block(block, [], [STOPInstruction()])
        
    def test_empty_relative_block(self):
        return_block = InstructionBlock()
        block = InstructionBlock(return_block)
        self.assertRaises(InstructionBlockNotYetPlacedException, block.get_start_address)
        self.assertRaises(MissingReturnAddressException, block.compile_sequence)
        ip = InstructionPointer(return_block, 7)
        block.return_ip = ip
        self.__verify_block(block, [], [GOTOInstruction(ip.block, ip.offset)])
        
    def test_create_embedded_block(self):
        parent_block = InstructionBlock()
        block = parent_block.create_embedded_block()
        self.assertRaises(InstructionBlockNotYetPlacedException, block.get_start_address)
        self.assertRaises(MissingReturnAddressException, block.compile_sequence)
        block.return_ip = InstructionPointer(parent_block, 0)
        self.__verify_block(block, [], [GOTOInstruction(parent_block, 0)])
        self.__verify_block(parent_block, [], [STOPInstruction(), GOTOInstruction(parent_block, 0)])
        self.assertEqual(1, block.get_start_address())
        
    def test_add_instruction_exec(self):
        block = InstructionBlock()
        expected_instructions = []
        
        waveforms = [DummyWaveform(), DummyWaveform(), DummyWaveform()]
        LOOKUP = [0, 1, 1, 0, 2, 1, 0, 0, 0, 1, 2, 2]
        for id in LOOKUP:
            waveform = waveforms[id]
            instruction = EXECInstruction(waveform)
            expected_instructions.append(instruction)
            block.add_instruction_exec(waveform)
            
        expected_compiled_instructions = expected_instructions.copy()
        expected_compiled_instructions.append(STOPInstruction())
        self.__verify_block(block, expected_instructions, expected_compiled_instructions)
        
        
    def test_add_instruction_goto(self):
        block = InstructionBlock()
        expected_instructions = []
        
        targets = [(InstructionBlock(), 0), (InstructionBlock(), 1), (InstructionBlock(), 50)]
        LOOKUP = [0, 1, 1, 0, 2, 1, 0, 0, 0, 1, 2, 2]
        for id in LOOKUP:
            target = targets[id]
            instruction = GOTOInstruction(target[0], target[1])
            expected_instructions.append(instruction)
            block.add_instruction_goto(target[0], target[1])
            
        expected_compiled_instructions = expected_instructions.copy()
        expected_compiled_instructions.append(STOPInstruction())
        self.__verify_block(block, expected_instructions, expected_compiled_instructions)
        
    def test_add_instruction_cjmp(self):
        block = InstructionBlock()
        expected_instructions = []
        expected_compiled_instructions = []
        
        targets = [(InstructionBlock(), 0), (InstructionBlock(), 1), (InstructionBlock(), 50)]
        triggers = [Trigger(), Trigger()]
        LOOKUP = [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (1, 0), (0, 1), (0, 1), (0, 0), (1, 0), (2, 1), (2, 1)]
        for i in LOOKUP:
            block.add_instruction_cjmp(triggers[i[1]], targets[i[0]])
            expected_instructions.append(CJMPInstruction(triggers[i[1]], targets[i[0]], 0))

        expected_compiled_instructions = expected_instructions.copy()
        expected_compiled_instructions.append(STOPInstruction())
        self.__verify_block(block, expected_instructions, expected_compiled_instructions)
        
    def test_add_instruction_stop(self):
        block = InstructionBlock()
        expected_instructions = [STOPInstruction(), STOPInstruction()]
        block.add_instruction_stop()
        block.add_instruction_stop()
        expected_compiled_instructions = expected_instructions.copy()
        expected_compiled_instructions.append(STOPInstruction())
        self.__verify_block(block, expected_instructions, expected_compiled_instructions)
        
    def test_nested_block_construction(self):
        main_block = InstructionBlock()
        expected_instructions = [[], [], [], []]
        expected_compiled_instructions = [[], [], [], []]
        
        blocks = []
            
        waveforms = [DummyWaveform(), DummyWaveform(), DummyWaveform()]
        
        main_block.add_instruction_exec(waveforms[0])
        expected_instructions[0].append(EXECInstruction(waveforms[0]))
        
        block = main_block.create_embedded_block() 
        trigger = Trigger()
        main_block.add_instruction_cjmp(trigger, block)
        expected_instructions[0].append(CJMPInstruction(trigger, block, 0))
        block.return_ip = InstructionPointer(main_block, len(main_block))
        blocks.append(block)
        
        block = main_block.create_embedded_block()
        trigger = Trigger()
        main_block.add_instruction_cjmp(trigger, block)
        expected_instructions[0].append(CJMPInstruction(trigger, block, 0))
        block.return_ip = InstructionPointer(main_block, len(main_block))
        blocks.append(block)
        
        WAVEFORM_LOOKUP = [[2, 2, 1, 1],[0, 1, 1, 0, 2, 1]]
        for i in [0, 1]:
            block = blocks[i]
            lookup = WAVEFORM_LOOKUP[i]
            for id in lookup:
                waveform = waveforms[id]
                expected_instructions[i + 1].append(EXECInstruction(waveform))
                block.add_instruction_exec(waveform)
            
        block = blocks[0].create_embedded_block()
        blocks[0].add_instruction_cjmp(trigger, block)
        expected_instructions[1].append(CJMPInstruction(trigger, block, 0))
        block.return_ip = InstructionPointer(blocks[0], len(blocks[0]))
        blocks.append(block)
        
        for id in [1, 2, 0, 2]:
            waveform = waveforms[id]
            expected_instructions[3].append(EXECInstruction(waveform))
            block.add_instruction_exec(waveform)
        
        for i in [0, 1, 2, 3]:
            expected_compiled_instructions[i] = expected_instructions[i].copy()
        
        expected_compiled_instructions[0].append(STOPInstruction())
        for i in [0, 1, 2]:
            expected_compiled_instructions[i + 1].append(GOTOInstruction(blocks[i].return_ip.block, blocks[i].return_ip.offset))
        
        positions = [0, None, None, None]                
        positions[3] = len(expected_compiled_instructions[1])
        
        expected_compiled_instructions[1].extend(expected_compiled_instructions[3])
        for i in [1, 2]:
            positions[i] = len(expected_compiled_instructions[0])
            expected_compiled_instructions[0].extend(expected_compiled_instructions[i])
            
        positions[3] += positions[1]
        
        self.__verify_block(blocks[2], expected_instructions[3], expected_compiled_instructions[3])
        self.__verify_block(blocks[1], expected_instructions[2], expected_compiled_instructions[2])
        self.__verify_block(blocks[0], expected_instructions[1], expected_compiled_instructions[1])
        self.__verify_block(main_block, expected_instructions[0], expected_compiled_instructions[0])
        
        self.assertEqual(positions[3], blocks[2].get_start_address())
        self.assertEqual(positions[2], blocks[1].get_start_address())
        self.assertEqual(positions[1], blocks[0].get_start_address())
        self.assertEqual(positions[0], main_block.get_start_address())
        
        for instruction in main_block.instructions:
            if isinstance(instruction, GOTOInstruction) or isinstance(instruction, CJMPInstruction):
                self.assertIsInstance(instruction.target.get_absolute_address(), int)
       
    def test_equality(self):
        block1 = InstructionBlock()
        block2 = InstructionBlock()
        self.assertEqual(block1, block1)
        self.assertNotEqual(block1, block2)
        self.assertNotEqual(hash(block1), hash(block2))


class InstructionStringRepresentation(unittest.TestCase):
    def test_str(self):
        IB = InstructionBlock()
        T = Trigger()
        W = DummyWaveform()

        a = [W,
             T,
             InstructionPointer(IB,1),
             CJMPInstruction(T,IB),
             GOTOInstruction(IB),
             EXECInstruction(W),
             InstructionBlockAlreadyFinalizedException(),
             InstructionBlockNotYetPlacedException(),
             MissingReturnAddressException(),
             IB
             ]
        
        b = [x.__str__() for x in a]
        
        for s in b:
            self.assertIsInstance(s, str)
        
if __name__ == "__main__":
    unittest.main(verbosity=2)