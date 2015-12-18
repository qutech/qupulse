# coding: utf-8
import unittest
import qctoolkit as qct
import qctoolkit.pulses as pls

class PersistenceBug(unittest.TestCase):
    def test_measurement_persistence_bug(self):
        p = pls.TablePulseTemplate(measurement=True, identifier='measure')
        p.add_entry(1,3, 'linear')
        p.add_entry(2,5, 'hold')
        p.add_entry(3,0,'jump')

        s = pls.Sequencer()
        s.push(p)
        print('Sequencer waveforms:\n', s._Sequencer__waveforms)
        print('Sequencer stacks:\n', s._Sequencer__sequencing_stacks)
        print('Sequencer main block:\n', s._Sequencer__main_block.__repr__())
        program = s.build()
        print('Program:\n', program)
        ex = program[0]
        windows = ex.waveform.measurement_windows

        s2 = pls.Sequencer()
        s2.push(p)
        print('Sequencer waveforms:\n', s._Sequencer__waveforms)
        print('Sequencer stacks:\n', s._Sequencer__sequencing_stacks)
        print('Sequencer main block:\n', s._Sequencer__main_block.__repr__())
        program2 = s2.build()
        print('Program:\n', program)
        ex2 = program[0]
        windows2 = ex2.waveform.measurement_windows

        s3 = pls.Sequencer()
        s3.push(p)
        print('Sequencer waveforms:\n', s._Sequencer__waveforms)
        print('Sequencer stacks:\n', s._Sequencer__sequencing_stacks)
        print('Sequencer main block:\n', s._Sequencer__main_block.__repr__())
        program3 = s3.build()
        print('Program:\n', program)
        ex3 = program[0]
        windows3 = ex3.waveform.measurement_windows

        print(windows, windows2, windows3)
        self.assertEqual(windows, windows2)
        self.assertEqual(windows, windows3)





