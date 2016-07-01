import unittest
import numpy
import numpy.random
from qctoolkit.qcmatlab.pulse_control import PulseControlInterface
from tests.pulses.sequencing_dummies import DummyWaveform, DummyInstructionBlock


class PulseControlInterfaceTests(unittest.TestCase):

    def test_create_waveform_struct(self) -> None:
        name = 'foo'
        sample_rate = 10
        expected_samples = numpy.random.rand(11)

        waveform = DummyWaveform(duration=1, sample_output=expected_samples)
        pci = PulseControlInterface(sample_rate, time_scaling=1)
        result = pci.create_waveform_struct(waveform, name=name)

        expected_sample_times = numpy.linspace(0, 1, 11).tolist()
        self.assertEqual((expected_sample_times, 0), waveform.sample_calls[0])
        expected_result = dict(name=name,
                               data=dict(wf=expected_samples.tolist(),
                                         marker=numpy.zeros_like(expected_samples).tolist(),
                                         clk=sample_rate
                                         )
                               )
        self.assertEqual(expected_result, result)

    def test_create_pulse_group_empty(self) -> None:
        name = 'foo_group'
        sample_rate = 10
        block = DummyInstructionBlock()

        pci = PulseControlInterface(sample_rate)
        (result, _) = pci.create_pulse_group(block, name=name)
        expected_result = dict(
            name=name,
            nrep=[],
            pulses=[],
            chan=1,
            ctrl='notrig'
        )
        self.assertEqual(expected_result, result)

    def test_create_pulse_group(self) -> None:
        name = 'foo_group'
        sample_rate = 10
        expected_samples_wf1 = numpy.random.rand(11)
        expected_samples_wf2 = numpy.random.rand(11)
        block = DummyInstructionBlock()
        wf1a = DummyWaveform(duration=1, sample_output=expected_samples_wf1)
        wf1b = DummyWaveform(duration=1, sample_output=expected_samples_wf1)
        wf2 = DummyWaveform(duration=1, sample_output=expected_samples_wf2)
        block.add_instruction_exec(wf1a)
        block.add_instruction_exec(wf1b)
        block.add_instruction_exec(wf2)
        block.add_instruction_exec(wf1a)

        registering_function = lambda x: x['data']
        pci = PulseControlInterface(sample_rate, time_scaling=1)
        (result, _) = pci.create_pulse_group(block, name=name)
        expected_result = dict(
            name=name,
            nrep=[2, 1, 1],
            pulses=[0, 1, 0],
            #pulses=[registering_function(pci.create_waveform_struct(wf1a, name='')),
            #        registering_function(pci.create_waveform_struct(wf2, name='')),
            #        registering_function(pci.create_waveform_struct(wf1a, name=''))],
            chan=1,
            ctrl='notrig'
        )
        self.assertEqual(expected_result, result)

    def test_create_pulse_group_invalid_instruction(self) -> None:
        name = 'foo_group'
        sample_rate = 10
        block = DummyInstructionBlock()
        block.add_instruction_goto(DummyInstructionBlock())

        pci = PulseControlInterface(sample_rate)
        with self.assertRaises(Exception):
            pci.create_pulse_group(block.compile_sequence(), name=name)

