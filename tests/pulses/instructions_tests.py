import unittest

from typing import Dict

from qctoolkit.pulses.instructions import InstructionBlock, InstructionPointer,\
    Trigger, CJMPInstruction, GOTOInstruction,EXECInstruction, STOPInstruction,\
    InstructionSequence, AbstractInstructionBlock, ImmutableInstructionBlock

from tests.pulses.sequencing_dummies import DummyWaveform, DummyInstructionBlock

 
class InstructionPointerTest(unittest.TestCase):

    def test_invalid_offset(self) -> None:
        block = InstructionBlock()
        self.assertRaises(ValueError, InstructionPointer, block, -1)
        self.assertRaises(ValueError, InstructionPointer, block, -12)
        
    def test_initialization_main_block(self) -> None:
        block = InstructionBlock()
        for offset in [0, 1, 924]:
            ip = InstructionPointer(block, offset)
            self.assertIs(block, ip.block)
            self.assertEqual(offset, ip.offset)

    def test_initialization_relative_block(self) -> None:
        block = InstructionBlock().create_embedded_block()
        for offset in [0, 1, 924]:
            ip = InstructionPointer(block, offset)
            self.assertIs(block, ip.block)
            self.assertEqual(offset, ip.offset)

    def test_equality(self) -> None:
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
    
    def test_equality(self) -> None:
        t1 = Trigger()
        t2 = Trigger()
        self.assertEqual(t1, t1)
        self.assertNotEqual(t1, t2)
        self.assertNotEqual(t2, t1)
        self.assertNotEqual(hash(t1), hash(t2))


class CJMPInstructionTest(unittest.TestCase):
    
    def test_initialization(self) -> None:
        block = InstructionBlock()
        trigger = Trigger()
        for offset in [0, 1, 23]:
            instr = CJMPInstruction(trigger, InstructionPointer(block, offset))
            self.assertEqual(trigger, instr.trigger)
            self.assertEqual(block, instr.target.block)
            self.assertEqual(offset, instr.target.offset)
        
    def test_equality(self) -> None:
        blocks = [InstructionBlock(), InstructionBlock()]
        for offset in [0, 1, 23]:
            instrA = CJMPInstruction(0, InstructionPointer(blocks[0], offset))
            instrB = CJMPInstruction(0, InstructionPointer(blocks[0], offset))
            self.assertEqual(instrA, instrB)
            self.assertEqual(instrB, instrA)
        instrs = []
        for trigger in [Trigger(), Trigger()]:
            for block in blocks:
                for offset in [0, 17]:
                    instruction = CJMPInstruction(trigger, InstructionPointer(block, offset))
                    self.assertEqual(instruction, instruction)
                    for other in instrs:
                        self.assertNotEqual(instruction, other)
                        self.assertNotEqual(other, instruction)
                        self.assertNotEqual(hash(instruction), hash(other))
                    instrs.append(instruction)

    def test_str(self) -> None:
        block = DummyInstructionBlock()
        trigger = Trigger()
        instr = CJMPInstruction(trigger, InstructionPointer(block, 3))
        self.assertEqual("cjmp to {} on {}".format(InstructionPointer(block, 3), trigger), str(instr))


class GOTOInstructionTest(unittest.TestCase):
    
    def test_initialization(self) -> None:
        block = InstructionBlock()
        for offset in [0, 1, 23]:
            instr = GOTOInstruction(InstructionPointer(block, offset))
            self.assertIs(block, instr.target.block)
            self.assertEqual(offset, instr.target.offset)
        
    def test_equality(self) -> None:
        blocks = [InstructionBlock(), InstructionBlock()]
        for offset in [0, 1, 23]:
            instrA = GOTOInstruction(InstructionPointer(blocks[0], offset))
            instrB = GOTOInstruction(InstructionPointer(blocks[0], offset))
            self.assertEqual(instrA, instrB)
            self.assertEqual(instrB, instrA)
        instrs = []
        for block in blocks:
            for offset in [0, 17]:
                instruction = GOTOInstruction(InstructionPointer(block, offset))
                self.assertEqual(instruction, instruction)
                for other in instrs:
                    self.assertNotEqual(instruction, other)
                    self.assertNotEqual(other, instruction)
                    self.assertNotEqual(hash(instruction), hash(other))
                instrs.append(instruction)

    def test_str(self) -> None:
        block = DummyInstructionBlock()
        instr = GOTOInstruction(InstructionPointer(block, 3))
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

    def __init__(self, method_name: str) -> None:
        super().__init__(method_name)
        self.maxDiff = None

    def __verify_block(self, block: InstructionBlock,
                       expected_instructions: InstructionSequence,
                       expected_compiled_instructions: InstructionSequence,
                       expected_return_ip: InstructionPointer) -> None:
        self.assertEqual(len(expected_instructions), len(block.instructions))
        self.assertEqual(expected_instructions, block.instructions)
        self.assertEqual(expected_compiled_instructions, [x for x in block])
        self.assertEqual(expected_return_ip, block.return_ip)

    def test_empty_unreturning_block(self) -> None:
        block = InstructionBlock()
        self.__verify_block(block, [], [STOPInstruction()], None)
        
    def test_empty_returning_block(self) -> None:
        return_block = InstructionBlock()
        block = InstructionBlock()
        ip = InstructionPointer(return_block, 7)
        block.return_ip = ip
        self.__verify_block(block, [], [GOTOInstruction(ip)], ip)
        
    def test_create_embedded_block(self) -> None:
        parent_block = InstructionBlock()
        block = parent_block.create_embedded_block()
        block.return_ip = InstructionPointer(parent_block, 18)
        self.__verify_block(block, [], [GOTOInstruction(InstructionPointer(parent_block, 18))], InstructionPointer(parent_block, 18))
        self.__verify_block(parent_block, [], [STOPInstruction()], None)
        
    def test_add_instruction_exec(self) -> None:
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
        self.__verify_block(block, expected_instructions, expected_compiled_instructions, None)
        
    def test_add_instruction_goto(self) -> None:
        block = InstructionBlock()
        expected_instructions = []
        
        targets = [InstructionBlock(), InstructionBlock(), InstructionBlock()]
        LOOKUP = [0, 1, 1, 0, 2, 1, 0, 0, 0, 1, 2, 2]
        for id in LOOKUP:
            target = targets[id]
            instruction = GOTOInstruction(InstructionPointer(target))
            expected_instructions.append(instruction)
            block.add_instruction_goto(target)
            
        expected_compiled_instructions = expected_instructions.copy()
        expected_compiled_instructions.append(STOPInstruction())
        self.__verify_block(block, expected_instructions, expected_compiled_instructions, None)
        
    def test_add_instruction_cjmp(self) -> None:
        block = InstructionBlock()
        expected_instructions = []
        expected_compiled_instructions = []
        
        targets = [(InstructionBlock(), 0), (InstructionBlock(), 1), (InstructionBlock(), 50)]
        triggers = [Trigger(), Trigger()]
        LOOKUP = [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (1, 0), (0, 1), (0, 1), (0, 0), (1, 0), (2, 1), (2, 1)]
        for i in LOOKUP:
            block.add_instruction_cjmp(triggers[i[1]], targets[i[0]][0])
            expected_instructions.append(CJMPInstruction(triggers[i[1]], InstructionPointer(targets[i[0]][0], 0)))

        expected_compiled_instructions = expected_instructions.copy()
        expected_compiled_instructions.append(STOPInstruction())
        self.__verify_block(block, expected_instructions, expected_compiled_instructions, None)
        
    def test_add_instruction_stop(self) -> None:
        block = InstructionBlock()
        expected_instructions = [STOPInstruction(), STOPInstruction()]
        block.add_instruction_stop()
        block.add_instruction_stop()
        expected_compiled_instructions = expected_instructions.copy()
        expected_compiled_instructions.append(STOPInstruction())
        self.__verify_block(block, expected_instructions, expected_compiled_instructions, None)
        
    def test_nested_block_construction(self) -> None:
        main_block = InstructionBlock()
        expected_instructions = [[], [], [], []]
        expected_compiled_instructions = [[], [], [], []]
        expected_return_ips = [None]
        
        blocks = []
            
        waveforms = [DummyWaveform(), DummyWaveform(), DummyWaveform()]
        
        main_block.add_instruction_exec(waveforms[0])
        expected_instructions[0].append(EXECInstruction(waveforms[0]))
        
        block = main_block.create_embedded_block() 
        trigger = Trigger()
        ip = InstructionPointer(block)
        main_block.add_instruction_cjmp(trigger, block)
        expected_instructions[0].append(CJMPInstruction(trigger, ip))
        block.return_ip = InstructionPointer(main_block, len(main_block))
        expected_return_ips.append(InstructionPointer(main_block, len(main_block)))
        blocks.append(block)
        
        block = main_block.create_embedded_block()
        trigger = Trigger()
        ip = InstructionPointer(block)
        main_block.add_instruction_cjmp(trigger, block)
        expected_instructions[0].append(CJMPInstruction(trigger, ip))
        block.return_ip = InstructionPointer(main_block, len(main_block))
        expected_return_ips.append(InstructionPointer(main_block, len(main_block)))
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
        ip = InstructionPointer(block)
        blocks[0].add_instruction_cjmp(trigger, block)
        expected_instructions[1].append(CJMPInstruction(trigger, ip))
        block.return_ip = InstructionPointer(blocks[0], len(blocks[0]))
        expected_return_ips.append(InstructionPointer(blocks[0], len(blocks[0])))
        blocks.append(block)
        
        for id in [1, 2, 0, 2]:
            waveform = waveforms[id]
            expected_instructions[3].append(EXECInstruction(waveform))
            block.add_instruction_exec(waveform)
        
        for i in [0, 1, 2, 3]:
            expected_compiled_instructions[i] = expected_instructions[i].copy()
        
        expected_compiled_instructions[0].append(STOPInstruction())
        for i in [0, 1, 2]:
            expected_compiled_instructions[i + 1].append(GOTOInstruction(blocks[i].return_ip))
        
        positions = [0, None, None, None]                
        positions[3] = len(expected_compiled_instructions[1])

        self.__verify_block(blocks[2], expected_instructions[3], expected_compiled_instructions[3], expected_return_ips[3])
        self.__verify_block(blocks[1], expected_instructions[2], expected_compiled_instructions[2], expected_return_ips[2])
        self.__verify_block(blocks[0], expected_instructions[1], expected_compiled_instructions[1], expected_return_ips[1])
        self.__verify_block(main_block, expected_instructions[0], expected_compiled_instructions[0], expected_return_ips[0])

    def test_equality(self) -> None:
        block1 = InstructionBlock()
        block2 = InstructionBlock()
        self.assertEqual(block1, block1)
        self.assertNotEqual(block1, block2)
        self.assertNotEqual(hash(block1), hash(block2))


class ImmutableInstructionBlockTests(unittest.TestCase):

    def __init__(self, method_name: str) -> None:
        super().__init__(method_name)
        self.maxDiff = None

    def __verify_block(self,
                       block: AbstractInstructionBlock,
                       immutable_block: ImmutableInstructionBlock,
                       context: Dict[AbstractInstructionBlock, ImmutableInstructionBlock]) -> None:
        self.assertIsInstance(immutable_block, ImmutableInstructionBlock)
        self.assertEqual(len(block.instructions), len(immutable_block.instructions))
        self.assertEqual(len(block), len(immutable_block))
        if block.return_ip is None:
            self.assertIsNone(immutable_block.return_ip)
        else:
            self.assertEqual(InstructionPointer(context[block.return_ip.block], block.return_ip.offset), immutable_block.return_ip)

        for instruction, immutable_instruction in zip(block.instructions, immutable_block.instructions):
            self.assertEqual(type(instruction), type(immutable_instruction))
            if isinstance(instruction, (GOTOInstruction, CJMPInstruction)):
                target_block = instruction.target.block
                immutable_target_block = immutable_instruction.target.block
                self.assertEqual(instruction.target.offset, immutable_instruction.target.offset)
                self.assertIsInstance(immutable_target_block, ImmutableInstructionBlock)
                self.assertEqual(context[target_block], immutable_target_block)
                self.assertEqual(immutable_block, immutable_target_block.return_ip.block)
                self.__verify_block(target_block, immutable_target_block, context)

    def test_empty_unreturning_block(self) -> None:
        block = InstructionBlock()
        context = dict()
        immutable_block = ImmutableInstructionBlock(block, context)
        self.__verify_block(block, immutable_block, context.copy())

    def test_empty_returning_block(self) -> None:
        return_block = InstructionBlock()
        block = InstructionBlock()
        ip = InstructionPointer(return_block, 7)
        block.return_ip = ip
        context = {return_block: ImmutableInstructionBlock(return_block, dict())}
        immutable_block = ImmutableInstructionBlock(block, context)
        self.__verify_block(block, immutable_block, context)

    def test_nested_block_construction(self) -> None:
        main_block = InstructionBlock()
        blocks = []
        waveforms = [DummyWaveform(), DummyWaveform(), DummyWaveform()]

        main_block.add_instruction_exec(waveforms[0])

        block = main_block.create_embedded_block()
        trigger = Trigger()
        ip = InstructionPointer(block)
        main_block.add_instruction_cjmp(trigger, block)
        block.return_ip = InstructionPointer(main_block, len(main_block))
        blocks.append(block)

        block = main_block.create_embedded_block()
        trigger = Trigger()
        ip = InstructionPointer(block)
        main_block.add_instruction_cjmp(trigger, block)
        block.return_ip = InstructionPointer(main_block, len(main_block))
        blocks.append(block)

        WAVEFORM_LOOKUP = [[2, 2, 1, 1], [0, 1, 1, 0, 2, 1]]
        for i in [0, 1]:
            block = blocks[i]
            lookup = WAVEFORM_LOOKUP[i]
            for id in lookup:
                waveform = waveforms[id]
                block.add_instruction_exec(waveform)

        block = blocks[0].create_embedded_block()
        ip = InstructionPointer(block)
        blocks[0].add_instruction_cjmp(trigger, block)
        block.return_ip = InstructionPointer(blocks[0], len(blocks[0]))
        blocks.append(block)

        for id in [1, 2, 0, 2]:
            waveform = waveforms[id]
            block.add_instruction_exec(waveform)

        context = dict()
        immutable_block = ImmutableInstructionBlock(main_block, context)
        self.__verify_block(main_block, immutable_block, context.copy())


class InstructionStringRepresentation(unittest.TestCase):
    def test_str(self) -> None:
        IB = InstructionBlock()
        T = Trigger()
        W = DummyWaveform()

        a = [W,
             T,
             InstructionPointer(IB,1),
             CJMPInstruction(T,IB),
             GOTOInstruction(IB),
             EXECInstruction(W),
             IB
             ]
        
        b = [x.__str__() for x in a]
        
        for s in b:
            self.assertIsInstance(s, str)
        
if __name__ == "__main__":
    unittest.main(verbosity=2)