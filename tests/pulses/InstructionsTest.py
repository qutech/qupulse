import unittest

from pulses.Instructions import *
 
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
        block = InstructionBlock()
        block = block.add_instruction_cjmp(0)
        for offset in [0, 1, 924]:
            ip = InstructionPointer(block, offset)
            self.assertIs(block, ip.block)
            self.assertEqual(offset, ip.offset)
            self.assertRaises(InstructionBlockNotYetPlacedException, ip.get_absolute_address)
            
    def test_equality(self):
        blocks = [InstructionBlock(), InstructionBlock()]
        blocks.append(blocks[0].add_instruction_cjmp(0))
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
            

class WaveformTest(unittest.TestCase):
    
    def test_initialization(self):
        waveform = Waveform()
        self.assertEqual(0, len(waveform))
        for value in [0, 1, 22]:
            waveform = Waveform(value)
            self.assertEqual(value, len(waveform))
        self.assertRaises(ValueError, Waveform, -1)
        self.assertRaises(ValueError, Waveform, -22)
        
    def test_equality(self):
        wf1 = Waveform()
        wf2 = Waveform()
        self.assertEqual(wf1, wf1)
        self.assertNotEqual(wf1, wf2)
        self.assertNotEqual(wf2, wf1)
        self.assertNotEqual(hash(wf1), hash(wf2))
        
class CJMPInstructionTest(unittest.TestCase):
    
    def test_initialization(self):
        self.fail("!!FAILURE TEMPORARILY INTENDED!! How to pass conditions to pulses and thus conditional jump instructions is not yet specified. !!FAILURE TEMPORARILY INTENDED!!")
        
    def test_equality(self):
        blocks = [InstructionBlock(), InstructionBlock()]
        for offset in [0, 1, 23]:
            instrA = CJMPInstruction(0, blocks[0], offset)
            instrB = CJMPInstruction(0, blocks[0], offset)
            self.assertEqual(instrA, instrB)
            self.assertEqual(instrB, instrA)
        instrs = []
        for condition in [0, 1]:
            for block in blocks:
                for offset in [0, 17]:
                    instruction = CJMPInstruction(condition, block, offset)
                    self.assertEqual(instruction, instruction)
                    for other in instrs:
                        self.assertNotEqual(instruction, other)
                        self.assertNotEqual(other, instruction)
                        self.assertNotEqual(hash(instruction), hash(other))
                    instrs.append(instruction)
        
class GOTOInstructionTest(unittest.TestCase):
    
    def test_initialization(self):
        block = InstructionBlock()
        for offset in [0, 1, 23]:
            instr = GOTOInstruction(block, offset)
            self.assertEqual('goto', instr.get_instruction_code())
            self.assertIs(block, instr.target.block)
            self.assertEqual(offset, instr.target.offset)
        
    def test_equality(self):
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
            
class EXECInstructionTest(unittest.TestCase):
    
    def test_initialization(self):
        waveform = Waveform()
        instr = EXECInstruction(waveform)
        self.assertEqual('exec', instr.get_instruction_code())
        self.assertIs(waveform, instr.waveform)
        
    def test_equality(self):
        wf1 = Waveform()
        wf2 = Waveform()
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
            
class STOPInstructionTest(unittest.TestCase):
    
    def test_initialization(self):
        instr = STOPInstruction()
        self.assertEqual('stop', instr.get_instruction_code())
        
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
        
    def test_add_instruction_exec(self):
        block = InstructionBlock()
        expected_instructions = []
        
        waveforms = [Waveform(), Waveform(), Waveform()]
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
        jump_back_instructions = []
        
        for i in [0, 1, 1, 35]:
            condition = i
            new_block = block.add_instruction_cjmp(condition)
            instruction = CJMPInstruction(condition, new_block, 0)
            expected_instructions.append(instruction)
            jump_back_instructions.append(GOTOInstruction(block, len(expected_instructions)))

        expected_compiled_instructions = expected_instructions.copy()
        expected_compiled_instructions.append(STOPInstruction())
        expected_compiled_instructions.extend(jump_back_instructions)
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
        return_ips = []
        
        blocks = []
            
        waveforms = [Waveform(), Waveform(), Waveform()]
        
        main_block.add_instruction_exec(waveforms[0])
        expected_instructions[0].append(EXECInstruction(waveforms[0]))
        
        block = main_block.add_instruction_cjmp(0)
        expected_instructions[0].append(CJMPInstruction(0, block, 0))
        blocks.append(block)
        return_ips.append(InstructionPointer(main_block, len(main_block)))
        
        block = main_block.add_instruction_cjmp(15)
        expected_instructions[0].append(CJMPInstruction(15, block, 0))
        blocks.append(block)
        return_ips.append(InstructionPointer(main_block, len(main_block)))
        
        WAVEFORM_LOOKUP = [[2, 2, 1, 1],[0, 1, 1, 0, 2, 1]]
        for i in [0, 1]:
            block = blocks[i]
            lookup = WAVEFORM_LOOKUP[i]
            for id in lookup:
                waveform = waveforms[id]
                expected_instructions[i + 1].append(EXECInstruction(waveform))
                block.add_instruction_exec(waveform)
            
        block = blocks[0].add_instruction_cjmp(1)
        expected_instructions[1].append(CJMPInstruction(1, block, 0))
        blocks.append(block)
        return_ips.append(InstructionPointer(blocks[0], len(blocks[0])))
        
        for id in [1, 2, 0, 2]:
            waveform = waveforms[id]
            expected_instructions[3].append(EXECInstruction(waveform))
            block.add_instruction_exec(waveform)
        
        self.assertIsNone(main_block.return_ip)
        self.assertEqual(blocks[2].return_ip, return_ips[2])
        self.assertEqual(blocks[1].return_ip, return_ips[1])
        self.assertEqual(blocks[0].return_ip, return_ips[0])
        
        for i in [0, 1, 2, 3]:
            expected_compiled_instructions[i] = expected_instructions[i].copy()
        
        expected_compiled_instructions[0].append(STOPInstruction())
        for i in [0, 1, 2]:
            expected_compiled_instructions[i + 1].append(GOTOInstruction(return_ips[i].block, return_ips[i].offset))
        
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
        