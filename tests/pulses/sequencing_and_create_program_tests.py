import unittest
from unittest import mock
from typing import Dict, Union

from qctoolkit.pulses.pulse_template import PulseTemplate
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses import TablePT, FunctionPT, AtomicMultiChannelPT, ForLoopPT, RepetitionPT, SequencePT, MappingPT

from qctoolkit._program._loop import Loop, MultiChannelProgram


class SequencingCompatibilityTest:
    def get_pulse_template(self) -> PulseTemplate:
        raise NotImplementedError()

    def get_parameters(self) -> dict:
        raise NotImplementedError()

    def get_channel_mapping(self) -> Dict[str, str]:
        raise NotImplementedError()

    def get_measurement_mapping(self) -> Dict[str, str]:
        raise NotImplementedError()

    def build_program_with_sequencer(self: unittest.TestCase, pulse_template, measurement_mapping=None, **kwargs):
        sequencer = Sequencer()
        sequencer.push(sequencing_element=pulse_template, conditions=dict(), **kwargs, window_mapping=measurement_mapping)
        instruction_block = sequencer.build()
        mcp = MultiChannelProgram(instruction_block=instruction_block)
        self.assertEqual(len(mcp.programs), 1)
        return next(iter(mcp.programs.values()))

    def build_program_with_create_program(self, pulse_template: PulseTemplate, **kwargs):
        return pulse_template.create_program(**kwargs)

    def assert_results_in_exact_same_program(self: Union[unittest.TestCase, 'SequencingCompatibilityTest'], **kwargs):
        pt = self.get_pulse_template()

        seq_program = self.build_program_with_sequencer(pt, **kwargs)
        cre_program = self.build_program_with_create_program(pt, **kwargs)

        self.assertEqual(seq_program, cre_program)

    def test_exact_same_program(self):
        self.assert_results_in_exact_same_program(parameters=self.get_parameters(),
                                                  channel_mapping=self.get_channel_mapping(),
                                                  measurement_mapping=self.get_measurement_mapping())


class ComplexProgramSequencingCompatibilityTest(SequencingCompatibilityTest, unittest.TestCase):
    def get_pulse_template(self) -> PulseTemplate:
        fpt = FunctionPT('sin(omega*t)', 't_duration', 'X', measurements=[('M', 0, 't_duration')])
        tpt = TablePT({'Y': [(0, 'a'), ('t_duration', 2)],
                       'Z': [('t_duration', 1)]}, measurements=[('N', 0, 't_duration/2')])
        mpt = MappingPT(fpt, parameter_mapping={'omega': '2*pi/t_duration'}, allow_partial_parameter_mapping=True)

        ampt = AtomicMultiChannelPT(mpt, tpt)
        body = ampt @ ampt
        rpt = RepetitionPT(body, 'N_rep', measurements=[('O', 0, 1)])

        forpt = ForLoopPT(rpt, 'a', '6')

        final = SequencePT(rpt, forpt)
        return final

    def get_parameters(self) -> dict:
        return dict(t_duration=4,
                    N_rep=17,
                    a=-1)

    def get_channel_mapping(self) -> Dict[str, str]:
        return {'X': 'A', 'Z': None, 'Y': 'B'}

    def get_measurement_mapping(self) -> Dict[str, str]:
        return {'M': 'S', 'N': None, 'O': 'T'}
