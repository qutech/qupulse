import unittest
from typing import List

import pulses.Instructions as instructions
 
class InstructionPointerTest(unittest.TestCase):

    def test_invalid_offset(self):
        block = instructions.InstructionBlock()
        self.assertRaises(ValueError, instructions.InstructionPointer, block, -1)
        self.assertRaises(ValueError, instructions.InstructionPointer, block, -12)
        
    def test_initialization_main_block(self):
        block = instructions.InstructionBlock()
        for offset in [0, 1, 924]:
            ip = instructions.InstructionPointer(block, offset)
            self.assertIs(block, ip.block)
            self.assertEqual(offset, ip.offset)
            self.assertEqual(offset, ip.get_absolute_address())
            
    def test_initialization_relative_block(self):
        block = instructions.InstructionBlock()
        block = block.add_instruction_cjmp(0)
        for offset in [0, 1, 924]:
            ip = instructions.InstructionPointer(block, offset)
            self.assertIs(block, ip.block)
            self.assertEqual(offset, ip.offset)
            self.assertRaises(instructions.InstructionBlockNotYetPlacedException, ip.get_absolute_address)
            
    def test_equality(self):
        blocks = [instructions.InstructionBlock(), instructions.InstructionBlock()]
        blocks.append(blocks[0].add_instruction_cjmp(0))
        ips = []
        for block in blocks:
            for offset in [0, 1, 2352]:
                ip = instructions.InstructionPointer(block, offset)
                self.assertEqual(ip, ip)
                for other in ips:
                    self.assertNotEqual(ip, other)
                    self.assertNotEqual(other, ip)
                    self.assertNotEqual(hash(ip), hash(other))
                ips.append(ip)
            

class WaveformTest(unittest.TestCase):
    
    def test_initialization(self):
        waveform = instructions.Waveform()
        self.assertEqual(0, len(waveform))
        for value in [0, 1, 22]:
            waveform = instructions.Waveform(value)
            self.assertEqual(value, len(waveform))
        self.assertRaises(ValueError, instructions.Waveform, -1)
        self.assertRaises(ValueError, instructions.Waveform, -22)
        
    def test_equality(self):
        wf1 = instructions.Waveform()
        wf2 = instructions.Waveform()
        self.assertEqual(wf1, wf1)
        self.assertNotEqual(wf1, wf2)
        self.assertNotEqual(wf2, wf1)
        self.assertNotEqual(hash(wf1), hash(wf2))
        
class CJMPInstructionTest(unittest.TestCase):
    
    def test_initialization(self):
        self.fail("!!FAILURE TEMPORARILY INTENDED!! How to pass conditions to pulses and thus conditional jump instructions is not yet specified. !!FAILURE TEMPORARILY INTENDED!!")
        
    def test_equality(self):
        blocks = [instructions.InstructionBlock(), instructions.InstructionBlock()]
        for offset in [0, 1, 23]:
            instrA = instructions.CJMPInstruction(0, blocks[0], offset)
            instrB = instructions.CJMPInstruction(0, blocks[0], offset)
            self.assertEqual(instrA, instrB)
            self.assertEqual(instrB, instrA)
        instrs = []
        for condition in [0, 1]:
            for block in blocks:
                for offset in [0, 17]:
                    instruction = instructions.CJMPInstruction(condition, block, offset)
                    self.assertEqual(instruction, instruction)
                    for other in instrs:
                        self.assertNotEqual(instruction, other)
                        self.assertNotEqual(other, instruction)
                        self.assertNotEqual(hash(instruction), hash(other))
                    instrs.append(instruction)
        
class GOTOInstructionTest(unittest.TestCase):
    
    def test_initialization(self):
        block = instructions.InstructionBlock()
        for offset in [0, 1, 23]:
            instr = instructions.GOTOInstruction(block, offset)
            self.assertEqual('goto', instr.get_instruction_code())
            self.assertIs(block, instr.target.block)
            self.assertEqual(offset, instr.target.offset)
        
    def test_equality(self):
        blocks = [instructions.InstructionBlock(), instructions.InstructionBlock()]
        for offset in [0, 1, 23]:
            instrA = instructions.GOTOInstruction(blocks[0], offset)
            instrB = instructions.GOTOInstruction(blocks[0], offset)
            self.assertEqual(instrA, instrB)
            self.assertEqual(instrB, instrA)
        instrs = []
        for block in blocks:
            for offset in [0, 17]:
                instruction = instructions.GOTOInstruction(block, offset)
                self.assertEqual(instruction, instruction)
                for other in instrs:
                    self.assertNotEqual(instruction, other)
                    self.assertNotEqual(other, instruction)
                    self.assertNotEqual(hash(instruction), hash(other))
                instrs.append(instruction)
            
class EXECInstructionTest(unittest.TestCase):
    
    def test_initialization(self):
        waveform = instructions.Waveform()
        instr = instructions.EXECInstruction(waveform)
        self.assertEqual('exec', instr.get_instruction_code())
        self.assertIs(waveform, instr.waveform)
        
    def test_equality(self):
        wf1 = instructions.Waveform()
        wf2 = instructions.Waveform()
        instr11 = instructions.EXECInstruction(wf1)
        instr12 = instructions.EXECInstruction(wf1)
        instr20 = instructions.EXECInstruction(wf2)
        self.assertEqual(instr11, instr11)
        self.assertEqual(instr11, instr12)
        self.assertEqual(instr12, instr11)
        self.assertNotEqual(instr11, instr20)
        self.assertNotEqual(instr20, instr11)
        self.assertEqual(hash(instr11), hash(instr12))
        self.assertNotEqual(hash(instr11), hash(instr20))
            
class STOPInstructionTest(unittest.TestCase):
    
    def test_initialization(self):
        instr = instructions.STOPInstruction()
        self.assertEqual('stop', instr.get_instruction_code())
        
    def test_equality(self):
        instr1 = instructions.STOPInstruction()
        instr2 = instructions.STOPInstruction()
        self.assertEqual(instr1, instr1)
        self.assertEqual(instr1, instr2)
        self.assertEqual(instr2, instr1)
        self.assertEqual(hash(instr1), hash(instr2))
            
class InstructionBlockTest(unittest.TestCase):

    def _verify_block(self, block: instructions.InstructionBlock, expectedInstructions: List[instructions.Instruction], finalized: bool) -> None:
        self.assertEqual(len(expectedInstructions), len(block))
        self.assertEqual(finalized, block.finalized)
        self.assertEqual(finalized, block.is_finalized())
        self.assertEqual(expectedInstructions, block.instructions)
        self.assertEqual(expectedInstructions, block.get_instructions())

    def test_empty_unreturning_main_block(self):
        block = instructions.InstructionBlock()
        self._verify_block(block, [], False)
        self.assertEqual(0, block._offset)
        self.assertEqual(0, block.get_start_address())
        self.assertIsNone(block._outerBlock)
        self.assertIsNone(block.returnIP)
        block.finalize()
        self._verify_block(block, [instructions.STOPInstruction()], True)
        
    def test_empty_returning_main_block(self):
        block = instructions.InstructionBlock()
        self._verify_block(block, [], False)
        self.assertEqual(0, block._offset)
        self.assertEqual(0, block.get_start_address())
        self.assertIsNone(block._outerBlock)
        self.assertIsNone(block.returnIP)
        ip = instructions.InstructionPointer(instructions.InstructionBlock(), 7)
        block.returnIP = ip # must have no effect
        block.finalize()
        self._verify_block(block, [instructions.STOPInstruction()], True)
        
    def test_empty_relative_block(self):
        returnBlock = instructions.InstructionBlock()
        block = instructions.InstructionBlock(returnBlock)
        self._verify_block(block, [], False)
        self.assertIs(returnBlock, block._outerBlock)
        self.assertIsNone(block._offset)
        self.assertRaises(instructions.InstructionBlockNotYetPlacedException, block.get_start_address)
        self.assertRaises(instructions.MissingReturnAddressException, block.finalize)
        ip = instructions.InstructionPointer(returnBlock, 7)
        block.returnIP = ip
        block.finalize()
        self._verify_block(block, [instructions.GOTOInstruction(ip.block, ip.offset)], True)
        
    def test_add_instruction_exec(self):
        block = instructions.InstructionBlock()
        self._verify_block(block, [], False)
        expectedInstructions = []
        
        waveforms = [instructions.Waveform(), instructions.Waveform(), instructions.Waveform()]
        LOOKUP = [0, 1, 1, 0, 2, 1, 0, 0, 0, 1, 2, 2]
        for id in LOOKUP:
            waveform = waveforms[id]
            instruction = instructions.EXECInstruction(waveform)
            expectedInstructions.append(instruction)
            block.add_instruction_exec(waveform)
            
        self._verify_block(block, expectedInstructions, False)
        
        
    def test_add_instruction_goto(self):
        block = instructions.InstructionBlock()
        self._verify_block(block, [], False)
        expectedInstructions = []
        
        targets = [(instructions.InstructionBlock(), 0), (instructions.InstructionBlock(), 1), (instructions.InstructionBlock(), 50)]
        LOOKUP = [0, 1, 1, 0, 2, 1, 0, 0, 0, 1, 2, 2]
        for id in LOOKUP:
            target = targets[id]
            instruction = instructions.GOTOInstruction(target[0], target[1])
            expectedInstructions.append(instruction)
            block.add_instruction_goto(target[0], target[1])
            
        self._verify_block(block, expectedInstructions, False)
        
    def test_add_instruction_cjmp(self):
        block = instructions.InstructionBlock()
        self._verify_block(block, [], False)
        expectedInstructions = []
        
        for i in [0, 1, 1, 35]:
            condition = i
            newBlock = block.add_instruction_cjmp(condition)
            instruction = instructions.CJMPInstruction(condition, newBlock, 0)
            expectedInstructions.append(instruction)
            
        self._verify_block(block, expectedInstructions, False)
        
    def test_add_instruction_stop(self):
        block = instructions.InstructionBlock()
        self._verify_block(block, [], False)
        expectedInstructions = [instructions.STOPInstruction(), instructions.STOPInstruction()]
        block.add_instruction_stop()
        block.add_instruction_stop()
        self._verify_block(block, expectedInstructions, False)        
        
    def test_nested_block_construction(self):
        mainBlock = instructions.InstructionBlock()
        self._verify_block(mainBlock, [], False)
        expectedInstructions = [[], [], [], []]
        returnIPs = []
        
        blocks = []
            
        waveforms = [instructions.Waveform(), instructions.Waveform(), instructions.Waveform()]
        
        mainBlock.add_instruction_exec(waveforms[0])
        expectedInstructions[0].append(instructions.EXECInstruction(waveforms[0]))
        
        block = mainBlock.add_instruction_cjmp(0)
        expectedInstructions[0].append(instructions.CJMPInstruction(0, block, 0))
        blocks.append(block)
        returnIPs.append(instructions.InstructionPointer(mainBlock, len(mainBlock)))
        
        block = mainBlock.add_instruction_cjmp(15)
        expectedInstructions[0].append(instructions.CJMPInstruction(15, block, 0))
        blocks.append(block)
        returnIPs.append(instructions.InstructionPointer(mainBlock, len(mainBlock)))
        
        WAVEFORM_LOOKUP = [[2, 2, 1, 1],[0, 1, 1, 0, 2, 1]]
        for i in [0, 1]:
            block = blocks[i]
            lookup = WAVEFORM_LOOKUP[i]
            for id in lookup:
                waveform = waveforms[id]
                expectedInstructions[i + 1].append(instructions.EXECInstruction(waveform))
                block.add_instruction_exec(waveform)
            
        block = blocks[0].add_instruction_cjmp(1)
        expectedInstructions[1].append(instructions.CJMPInstruction(1, block, 0))
        blocks.append(block)
        returnIPs.append(instructions.InstructionPointer(blocks[0], len(blocks[0])))
        
        for id in [1, 2, 0, 2]:
            waveform = waveforms[id]
            expectedInstructions[3].append(instructions.EXECInstruction(waveform))
            block.add_instruction_exec(waveform)
        
        # not expressed as a loop to better understand assertion failures
        self._verify_block(blocks[2], expectedInstructions[3], False)
        self._verify_block(blocks[1], expectedInstructions[2], False)
        self._verify_block(blocks[0], expectedInstructions[1], False)
        self._verify_block(mainBlock, expectedInstructions[0], False)
        
        self.assertIsNone(mainBlock.returnIP)
        self.assertEqual(blocks[2].returnIP, returnIPs[2])
        self.assertEqual(blocks[1].returnIP, returnIPs[1])
        self.assertEqual(blocks[0].returnIP, returnIPs[0])
        
        mainBlock.finalize()
        
        expectedInstructions[0].append(instructions.STOPInstruction())
        for i in [0, 1, 2]:
            expectedInstructions[i + 1].append(instructions.GOTOInstruction(returnIPs[i].block, returnIPs[i].offset))
        
        positions = [0, None, None, None]                
        positions[3] = len(expectedInstructions[1])
        
        expectedInstructions[1].extend(expectedInstructions[3])
        for i in [1, 2]:
            positions[i] = len(expectedInstructions[0])
            expectedInstructions[0].extend(expectedInstructions[i])
            
        positions[3] += positions[1]
        
        self._verify_block(blocks[2], expectedInstructions[3], True)
        self._verify_block(blocks[1], expectedInstructions[2], True)
        self._verify_block(blocks[0], expectedInstructions[1], True)
        self._verify_block(mainBlock, expectedInstructions[0], True)
        
        self.assertEqual(positions[3], blocks[2].get_start_address())
        self.assertEqual(positions[2], blocks[1].get_start_address())
        self.assertEqual(positions[1], blocks[0].get_start_address())
        self.assertEqual(positions[0], mainBlock.get_start_address())
        
        for instruction in mainBlock.instructions:
            if isinstance(instruction, instructions.GOTOInstruction) or isinstance(instruction, instructions.CJMPInstruction):
                self.assertIsInstance(instruction.target.get_absolute_address(), int)
       
       
    def test_equality(self):
        block1 = instructions.InstructionBlock()
        block2 = instructions.InstructionBlock()
        self.assertEqual(block1, block1)
        self.assertNotEqual(block1, block2)
        self.assertNotEqual(hash(block1), hash(block2))
        