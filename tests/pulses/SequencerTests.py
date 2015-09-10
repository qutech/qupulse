import unittest
import os
import sys

srcPath = os.path.dirname(os.path.abspath(__file__)).rsplit('tests',1)[0] + 'src'
sys.path.insert(0,srcPath)

from tests.pulses.SequencingDummies import DummySequencingElement, DummySequencingHardware

from pulses.Parameter import  ConstantParameter
from pulses.Instructions import InstructionBlock
from pulses.Sequencer import Sequencer

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
        
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        elem = DummySequencingElement()
        
        sequencer.push(elem, ps)
        self.assertFalse(sequencer.has_finished())
    
#   The following are methods to test different execution path through the build() method.
#   Most relevant paths with up to 2 iterations for each loop are covered, excluding "mirror" configurations.
#   Note that there are paths which can never occur and thus are not tested.
#   Example for naming: o2_m1_i2_tf_m2_i1_f_i1_f
#   The outermost loop is iterated twice (o2)
#       - In the first iteration, the middle loop is iterated once (m1)
#           - Therein, the inner loop is iterated twice (i2) with branching decisions true (first iteration) and false (second iteration) (tf)
#       - In the second iteration, the middle loop is iterated twice (m2)
#           - In its first iteration, the inner loop is iterated once (i1) with branching decision false (f)
#           - In its second iteration, the inner loop is iterated once (i1) with branching decision false (f)

    
    def test_build_path_no_loop_nothing_to_do(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
        
        block = sequencer.build()
        self.assertTrue(sequencer.has_finished())
        
    def test_build_path_o1_m1_i1_f_single_element_requires_stop_main_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
    
        elem = DummySequencingElement(True)
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        sequencer.push(elem, ps)
        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertEqual(ps, elem.parameters)
        self.assertEqual(1, elem.requires_stop_call_counter)
        self.assertEqual(0, elem.build_call_counter)
        
    def test_build_path_o1_m2_i1_f_i0_one_element_custom_block_requires_stop(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
    
        elem = DummySequencingElement(True)
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        target_block = InstructionBlock()
        sequencer.push(elem, ps, target_block)
        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertEqual(ps, elem.parameters)
        self.assertEqual(1, elem.requires_stop_call_counter)
        self.assertEqual(0, elem.build_call_counter)
        
    def test_build_path_o1_m2_i1_f_i1_f_one_element_custom_and_main_block_requires_stop(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
    
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
    
        elem_main = DummySequencingElement(True)
        sequencer.push(elem_main, ps)
        
        elem_cstm = DummySequencingElement(True)
        target_block = InstructionBlock()
        sequencer.push(elem_cstm, ps, target_block)
        
        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertEqual(ps, elem_main.parameters)
        self.assertEqual(1, elem_main.requires_stop_call_counter)
        self.assertEqual(0, elem_main.build_call_counter)
        self.assertEqual(ps, elem_cstm.parameters)
        self.assertEqual(1, elem_cstm.requires_stop_call_counter)
        self.assertEqual(0, elem_cstm.build_call_counter)
        
    def test_build_path_o2_m1_i1_t_m1_i0_one_element_main_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
    
        elem = DummySequencingElement(False)
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        sequencer.push(elem, ps)
        block = sequencer.build()
        
        self.assertTrue(sequencer.has_finished())
        self.assertIs(block, elem.target_block)
        self.assertEqual(ps, elem.parameters)
        self.assertEqual(1, elem.requires_stop_call_counter)
        self.assertEqual(1, elem.build_call_counter)
        
    def test_build_path_o2_m1_i2_tf_m1_i1_f_two_elements_main_block_last_requires_stop(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
    
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        target_block = InstructionBlock()
        elem1 = DummySequencingElement(False)
        elem2 = DummySequencingElement(True)
        sequencer.push(elem2, ps)
        sequencer.push(elem1, ps)
        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertIs(block, elem1.target_block)
        self.assertEqual(ps, elem1.parameters)
        self.assertEqual(1, elem1.requires_stop_call_counter)
        self.assertEqual(1, elem1.build_call_counter)
        self.assertEqual(ps, elem2.parameters)
        self.assertEqual(2, elem2.requires_stop_call_counter)
        self.assertEqual(0, elem2.build_call_counter)
        
    def test_build_path_o2_m1_i2_tt_m1_i0_two_elements_main_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
    
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        elem1 = DummySequencingElement(False)
        elem2 = DummySequencingElement(False)
        sequencer.push(elem2, ps)
        sequencer.push(elem1, ps)
        block = sequencer.build()
        
        self.assertTrue(sequencer.has_finished())
        self.assertIs(block, elem1.target_block)
        self.assertEqual(ps, elem1.parameters)
        self.assertEqual(1, elem1.requires_stop_call_counter)
        self.assertEqual(1, elem1.build_call_counter)
        self.assertIs(block, elem2.target_block)
        self.assertEqual(ps, elem2.parameters)
        self.assertEqual(1, elem2.requires_stop_call_counter)
        self.assertEqual(1, elem2.build_call_counter)
        
    def test_build_path_o2_m1_i1_t_m2_i0_i1_f_one_element_main_block_adds_one_element_requires_stop_new_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
    
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        
        new_block = InstructionBlock()
        new_elem = DummySequencingElement(True)
        
        elem = DummySequencingElement(False, (new_block, [new_elem]))
        sequencer.push(elem, ps)

        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertIs(block, elem.target_block)
        self.assertEqual(ps, elem.parameters)
        self.assertEqual(1, elem.requires_stop_call_counter)
        self.assertEqual(1, elem.build_call_counter)
        self.assertEqual(ps, new_elem.parameters)
        self.assertEqual(1, new_elem.requires_stop_call_counter)
        self.assertEqual(0, new_elem.build_call_counter)
        
    def test_build_path_o2_m1_i2_tf_m2_i1_f_i1_f_two_elements_main_block_last_requires_stop_add_one_element_requires_stop_new_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
    
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        
        new_block = InstructionBlock()
        new_elem = DummySequencingElement(True)
        
        elem1 = DummySequencingElement(False, (new_block, [new_elem]))
        elem2 = DummySequencingElement(True)
        sequencer.push(elem2, ps)
        sequencer.push(elem1, ps)

        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertIs(block, elem1.target_block)
        self.assertEqual(ps, elem1.parameters)
        self.assertEqual(1, elem1.requires_stop_call_counter)
        self.assertEqual(1, elem1.build_call_counter)
        self.assertEqual(ps, elem2.parameters)
        self.assertEqual(2, elem2.requires_stop_call_counter)
        self.assertEqual(0, elem2.build_call_counter)
        self.assertEqual(ps, new_elem.parameters)
        self.assertEqual(1, new_elem.requires_stop_call_counter)
        self.assertEqual(0, new_elem.build_call_counter)
        
    def test_build_path_o2_m2_i0_i1_t_m2_i0_i0_one_element_custom_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
                
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        
        target_block = InstructionBlock()
        elem = DummySequencingElement(False)
        sequencer.push(elem, ps, target_block)
        
        block = sequencer.build()
        
        self.assertTrue(sequencer.has_finished())
        self.assertIs(target_block, elem.target_block)
        self.assertEqual(ps, elem.parameters)
        self.assertEqual(1, elem.requires_stop_call_counter)
        self.assertEqual(1, elem.build_call_counter)
        
    # which element requires stop is considered a mirror configuration and only tested for this example
    def test_build_path_o2_m2_i1_f_i1_t_m2_i1_f_i0_one_element_custom_block_one_element_requires_stop_main_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
                
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        
        elem_main = DummySequencingElement(True)
        sequencer.push(elem_main, ps)
        
        target_block = InstructionBlock()
        elem_cstm = DummySequencingElement(False)
        sequencer.push(elem_cstm, ps, target_block)
        
        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertEqual(ps, elem_main.parameters)
        self.assertEqual(2, elem_main.requires_stop_call_counter)
        self.assertEqual(0, elem_main.build_call_counter)
        self.assertIs(target_block, elem_cstm.target_block)
        self.assertEqual(ps, elem_cstm.parameters)
        self.assertEqual(1, elem_cstm.requires_stop_call_counter)
        self.assertEqual(1, elem_cstm.build_call_counter)
        
    def test_build_path_o2_m2_i1_t_i1_t_m2_i0_i0_one_element_custom_block_one_element_main_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
                
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        
        elem_main = DummySequencingElement(False)
        sequencer.push(elem_main, ps)
        
        target_block = InstructionBlock()
        elem_cstm = DummySequencingElement(False)
        sequencer.push(elem_cstm, ps, target_block)
        
        block = sequencer.build()
        
        self.assertTrue(sequencer.has_finished())
        self.assertIs(block, elem_main.target_block)
        self.assertEqual(ps, elem_main.parameters)
        self.assertEqual(1, elem_main.requires_stop_call_counter)
        self.assertEqual(1, elem_main.build_call_counter)
        self.assertIs(target_block, elem_cstm.target_block)
        self.assertEqual(ps, elem_cstm.parameters)
        self.assertEqual(1, elem_cstm.requires_stop_call_counter)
        self.assertEqual(1, elem_cstm.build_call_counter)
        
    def test_build_path_o2_m2_i0_i2_tf_m2_i0_i1_f_two_elements_custom_block_last_requires_stop(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
                
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        
        target_block = InstructionBlock()
        elem2 = DummySequencingElement(True)
        sequencer.push(elem2, ps, target_block)
        
        elem1 = DummySequencingElement(False)
        sequencer.push(elem1, ps, target_block)
        
        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertIs(target_block, elem1.target_block)
        self.assertEqual(ps, elem1.parameters)
        self.assertEqual(1, elem1.requires_stop_call_counter)
        self.assertEqual(1, elem1.build_call_counter)
        self.assertEqual(ps, elem2.parameters)
        self.assertEqual(2, elem2.requires_stop_call_counter)
        self.assertEqual(0, elem2.build_call_counter)
        
    def test_build_path_o2_m2_i0_i2_tt_m2_i0_i0_two_elements_custom_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
                
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        
        target_block = InstructionBlock()
        elem2 = DummySequencingElement(False)
        sequencer.push(elem2, ps, target_block)
        
        elem1 = DummySequencingElement(False)
        sequencer.push(elem1, ps, target_block)
        
        block = sequencer.build()
        
        self.assertTrue(sequencer.has_finished())
        self.assertIs(target_block, elem1.target_block)
        self.assertEqual(ps, elem1.parameters)
        self.assertEqual(1, elem1.requires_stop_call_counter)
        self.assertEqual(1, elem1.build_call_counter)
        self.assertIs(target_block, elem2.target_block)
        self.assertEqual(ps, elem2.parameters)
        self.assertEqual(1, elem2.requires_stop_call_counter)
        self.assertEqual(1, elem2.build_call_counter)
        
    def test_build_path_o2_m2_i1_f_i2_tf_m2_i1_f_i1_f_two_elements_custom_block_last_requires_stop_one_element_requires_stop_main_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
                
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        
        target_block = InstructionBlock()
        elem2 = DummySequencingElement(True)
        sequencer.push(elem2, ps, target_block)
        
        elem1 = DummySequencingElement(False)
        sequencer.push(elem1, ps, target_block)
        
        elem_main = DummySequencingElement(True)
        sequencer.push(elem_main, ps)
        
        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertIs(target_block, elem1.target_block)
        self.assertEqual(ps, elem1.parameters)
        self.assertEqual(1, elem1.requires_stop_call_counter)
        self.assertEqual(1, elem1.build_call_counter)
        self.assertEqual(ps, elem2.parameters)
        self.assertEqual(2, elem2.requires_stop_call_counter)
        self.assertEqual(0, elem2.build_call_counter)
        self.assertEqual(ps, elem_main.parameters)
        self.assertEqual(2, elem_main.requires_stop_call_counter)
        self.assertEqual(0, elem_main.build_call_counter)
        
    def test_build_path_o2_m2_i1_t_i2_tf_m2_i0_i1_f_two_elements_custom_block_last_requires_stop_one_element_main_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
                
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        
        target_block = InstructionBlock()
        elem2 = DummySequencingElement(True)
        sequencer.push(elem2, ps, target_block)
        
        elem1 = DummySequencingElement(False)
        sequencer.push(elem1, ps, target_block)
        
        elem_main = DummySequencingElement(False)
        sequencer.push(elem_main, ps)
        
        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertIs(target_block, elem1.target_block)
        self.assertEqual(ps, elem1.parameters)
        self.assertEqual(1, elem1.requires_stop_call_counter)
        self.assertEqual(1, elem1.build_call_counter)
        self.assertEqual(ps, elem2.parameters)
        self.assertEqual(2, elem2.requires_stop_call_counter)
        self.assertEqual(0, elem2.build_call_counter)
        self.assertIs(block, elem_main.target_block)
        self.assertEqual(ps, elem_main.parameters)
        self.assertEqual(1, elem_main.requires_stop_call_counter)
        self.assertEqual(1, elem_main.build_call_counter)
        
    def test_build_path_o2_m2_i1_t_i2_tt_m2_i0_i0_two_elements_custom_block_one_element_main_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
                
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        
        target_block = InstructionBlock()
        elem2 = DummySequencingElement(False)
        sequencer.push(elem2, ps, target_block)
        
        elem1 = DummySequencingElement(False)
        sequencer.push(elem1, ps, target_block)
        
        elem_main = DummySequencingElement(False)
        sequencer.push(elem_main, ps)
        
        block = sequencer.build()
        
        self.assertTrue(sequencer.has_finished())
        self.assertIs(target_block, elem1.target_block)
        self.assertEqual(ps, elem1.parameters)
        self.assertEqual(1, elem1.requires_stop_call_counter)
        self.assertEqual(1, elem1.build_call_counter)
        self.assertIs(target_block, elem2.target_block)
        self.assertEqual(ps, elem2.parameters)
        self.assertEqual(1, elem2.requires_stop_call_counter)
        self.assertEqual(1, elem2.build_call_counter)
        self.assertIs(block, elem_main.target_block)
        self.assertEqual(ps, elem_main.parameters)
        self.assertEqual(1, elem_main.requires_stop_call_counter)
        self.assertEqual(1, elem_main.build_call_counter)
        
    def test_build_path_o2_m2_i2_tf_t_i2_tf_m2_i1_f_i1_f_two_elements_custom_block_last_requires_stop_two_element_main_block_last_requires_stop(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
                
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        
        target_block = InstructionBlock()
        elem2 = DummySequencingElement(True)
        sequencer.push(elem2, ps, target_block)
        
        elem1 = DummySequencingElement(False)
        sequencer.push(elem1, ps, target_block)
        
        elem_main2 = DummySequencingElement(True)
        sequencer.push(elem_main2, ps)
        
        elem_main1 = DummySequencingElement(False)
        sequencer.push(elem_main1, ps)
        
        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertIs(target_block, elem1.target_block)
        self.assertEqual(ps, elem1.parameters)
        self.assertEqual(1, elem1.requires_stop_call_counter)
        self.assertEqual(1, elem1.build_call_counter)
        self.assertEqual(ps, elem2.parameters)
        self.assertEqual(2, elem2.requires_stop_call_counter)
        self.assertEqual(0, elem2.build_call_counter)
        self.assertIs(block, elem_main1.target_block)
        self.assertEqual(ps, elem_main1.parameters)
        self.assertEqual(1, elem_main1.requires_stop_call_counter)
        self.assertEqual(1, elem_main1.build_call_counter)
        self.assertEqual(ps, elem_main2.parameters)
        self.assertEqual(2, elem_main2.requires_stop_call_counter)
        self.assertEqual(0, elem_main2.build_call_counter)
        
    # which block contains the element that requires a stop is considered a mirror configuration and only tested for this example
    def test_build_path_o2_m2_i2_tt_t_i2_tf_m2_i0_i1_f_two_elements_custom_block_last_requires_stop_two_element_main_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
                
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        
        target_block = InstructionBlock()
        elem2 = DummySequencingElement(True)
        sequencer.push(elem2, ps, target_block)
        
        elem1 = DummySequencingElement(False)
        sequencer.push(elem1, ps, target_block)
        
        elem_main2 = DummySequencingElement(False)
        sequencer.push(elem_main2, ps)
        
        elem_main1 = DummySequencingElement(False)
        sequencer.push(elem_main1, ps)
        
        block = sequencer.build()
        
        self.assertFalse(sequencer.has_finished())
        self.assertIs(target_block, elem1.target_block)
        self.assertEqual(ps, elem1.parameters)
        self.assertEqual(1, elem1.requires_stop_call_counter)
        self.assertEqual(1, elem1.build_call_counter)
        self.assertEqual(ps, elem2.parameters)
        self.assertEqual(2, elem2.requires_stop_call_counter)
        self.assertEqual(0, elem2.build_call_counter)
        self.assertIs(block, elem_main1.target_block)
        self.assertEqual(ps, elem_main1.parameters)
        self.assertEqual(1, elem_main1.requires_stop_call_counter)
        self.assertEqual(1, elem_main1.build_call_counter)
        self.assertIs(block, elem_main2.target_block)
        self.assertEqual(ps, elem_main2.parameters)
        self.assertEqual(1, elem_main2.requires_stop_call_counter)
        self.assertEqual(1, elem_main2.build_call_counter)
        
    def test_build_path_o2_m2_i2_tt_t_i2_tt_m2_i0_i0_two_elements_custom_block_two_element_main_block(self) -> None:
        dummy_hardware = DummySequencingHardware()
        sequencer = Sequencer(dummy_hardware)
                
        ps = {'foo': ConstantParameter(1), 'bar': ConstantParameter(7.3)}
        
        target_block = InstructionBlock()
        elem2 = DummySequencingElement(False)
        sequencer.push(elem2, ps, target_block)
        
        elem1 = DummySequencingElement(False)
        sequencer.push(elem1, ps, target_block)
        
        elem_main2 = DummySequencingElement(False)
        sequencer.push(elem_main2, ps)
        
        elem_main1 = DummySequencingElement(False)
        sequencer.push(elem_main1, ps)
        
        block = sequencer.build()
        
        self.assertTrue(sequencer.has_finished())
        self.assertIs(target_block, elem1.target_block)
        self.assertEqual(ps, elem1.parameters)
        self.assertEqual(1, elem1.requires_stop_call_counter)
        self.assertEqual(1, elem1.build_call_counter)
        self.assertIs(target_block, elem2.target_block)
        self.assertEqual(ps, elem2.parameters)
        self.assertEqual(1, elem2.requires_stop_call_counter)
        self.assertEqual(1, elem2.build_call_counter)
        self.assertIs(block, elem_main1.target_block)
        self.assertEqual(ps, elem_main1.parameters)
        self.assertEqual(1, elem_main1.requires_stop_call_counter)
        self.assertEqual(1, elem_main1.build_call_counter)
        self.assertIs(block, elem_main2.target_block)
        self.assertEqual(ps, elem_main2.parameters)
        self.assertEqual(1, elem_main2.requires_stop_call_counter)
        self.assertEqual(1, elem_main2.build_call_counter)
                
    # path 1_2_3_4_5_6_7_8_5_3_9 can never occur: 8 sets shall_continue = True, so 3 cannot evaluate to False
        
if __name__ == "__main__":
    unittest.main(verbosity=2)