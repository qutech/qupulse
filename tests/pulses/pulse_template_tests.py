import unittest

import copy
from typing import Optional, Dict, Set, Any, List

from qctoolkit import MeasurementWindow, ChannelID
from qctoolkit.expressions import Expression
from qctoolkit.pulses.pulse_template import AtomicPulseTemplate, MeasurementDeclaration
from qctoolkit.pulses.instructions import Waveform, EXECInstruction
from qctoolkit.pulses.parameters import Parameter, ParameterDeclaration
from qctoolkit.pulses.multi_channel_pulse_template import MultiChannelWaveform

from tests.pulses.sequencing_dummies import DummyWaveform, DummySequencer, DummyInstructionBlock


class AtomicPulseTemplateStub(AtomicPulseTemplate):

    def is_interruptable(self) -> bool:
        return super().is_interruptable()

    def __init__(self, *, waveform: Waveform=None, duration: Expression=None,
                 measurements: List[MeasurementDeclaration] = [],
                 identifier: Optional[str]=None) -> None:
        super().__init__(identifier=identifier, measurements=measurements)
        self.waveform = waveform
        self._duration = duration

    def build_waveform(self, parameters: Dict[str, Parameter], measurement_mapping, channel_mapping):
        return self.waveform

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return False

    @property
    def defined_channels(self) -> Set['ChannelID']:
        raise NotImplementedError()

    @property
    def parameter_declarations(self) -> Set[ParameterDeclaration]:
        raise NotImplementedError()

    @property
    def parameter_names(self) -> Set[str]:
        raise NotImplementedError()

    def get_serialization_data(self, serializer: 'Serializer') -> Dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def deserialize(serializer: 'Serializer', **kwargs) -> 'AtomicPulseTemplateStub':
        raise NotImplementedError()

    @property
    def duration(self) -> Expression:
        return self._duration


class AtomicPulseTemplateTests(unittest.TestCase):

    def test_is_interruptable(self) -> None:
        wf = DummyWaveform()
        template = AtomicPulseTemplateStub(waveform=wf)
        self.assertFalse(template.is_interruptable())
        template = AtomicPulseTemplateStub(waveform=wf, identifier="arbg4")
        self.assertFalse(template.is_interruptable())

    def test_build_sequence_no_waveform(self) -> None:
        sequencer = DummySequencer()
        block = DummyInstructionBlock()

        template = AtomicPulseTemplateStub()
        template.build_sequence(sequencer, {}, {}, {}, {}, block)
        self.assertFalse(block.instructions)

    def test_build_sequence(self) -> None:
        measurement_windows = [('M', 0, 5)]
        single_wf = DummyWaveform(duration=6, defined_channels={'A'}, measurement_windows=measurement_windows)
        wf = MultiChannelWaveform([single_wf])

        sequencer = DummySequencer()
        block = DummyInstructionBlock()

        template = AtomicPulseTemplateStub(waveform=wf, measurements=measurement_windows)
        template.build_sequence(sequencer, {}, {}, measurement_mapping={}, channel_mapping={}, instruction_block=block)
        self.assertEqual(len(block.instructions), 1)
        self.assertIsInstance(block.instructions[0], EXECInstruction)
        self.assertEqual(block.instructions[0].waveform.defined_channels, {'A'})
        self.assertEqual(list(block.instructions[0].waveform.get_measurement_windows()), [('M', 0, 5)])

    def test_measurement_windows(self) -> None:
        pulse = AtomicPulseTemplateStub(duration=Expression(5),
                                        measurements=[('mw', 0, 5)])
        with self.assertRaises(KeyError):
            pulse.get_measurement_windows(parameters=dict(), measurement_mapping=dict())
        windows = pulse.get_measurement_windows(parameters=dict(), measurement_mapping={'mw': 'asd'})
        self.assertEqual([('asd', 0, 5)], windows)
        self.assertEqual(pulse.measurement_declarations, [('mw', 0, 5)])

    def test_no_measurement_windows(self) -> None:
        pulse = AtomicPulseTemplateStub(duration=Expression(4))
        windows = pulse.get_measurement_windows(dict(), {'mw': 'asd'})
        self.assertEqual([], windows)
        self.assertEqual([], pulse.measurement_declarations)

    def test_measurement_windows_with_parameters(self) -> None:
        pulse = AtomicPulseTemplateStub(duration=Expression('length'),
                                        measurements=[('mw', 1, '(1+length)/2')])
        parameters = dict(length=100)
        windows = pulse.get_measurement_windows(parameters, measurement_mapping={'mw': 'asd'})
        self.assertEqual(windows, [('asd', 1, 101 / 2)])
        self.assertEqual(pulse.measurement_declarations, [('mw', 1, '(1+length)/2')])

    @unittest.skip('Move to AtomicPulseTemplate test')
    def test_multiple_measurement_windows(self) -> None:
        pulse = AtomicPulseTemplateStub(duration=Expression('length'),
                                        measurements=[('A', 0, '(1+length)/2'),
                                                      ('A', 1, 3),
                                                      ('B', 'begin', 2)])

        parameters = dict(length=5, begin=1)
        measurement_mapping = dict(A='A', B='C')
        windows = pulse.get_measurement_windows(parameters=parameters,
                                                measurement_mapping=measurement_mapping)
        expected = [('A', 0, 3), ('A', 1, 3), ('C', 1, 2)]
        self.assertEqual(sorted(windows), sorted(expected))

        expected = [('A', 0, '(1+length)/2'),
                    ('A', 1, 3),
                    ('B', 'begin', 2)]
        self.assertEqual(pulse.measurement_declarations,
                         expected)

    def test_measurement_windows_multi_out_of_pulse(self) -> None:
        pulse = AtomicPulseTemplateStub(duration=Expression('length'),
                                        measurements=[('mw', 'a', 'd')])
        measurement_mapping = {'mw': 'mw'}

        with self.assertRaises(ValueError):
            pulse.get_measurement_windows(measurement_mapping=measurement_mapping,
                                          parameters=dict(length=10, a=-1, d=3))
        with self.assertRaises(ValueError):
            pulse.get_measurement_windows(measurement_mapping=measurement_mapping,
                                          parameters=dict(length=10, a=5, d=30))
        with self.assertRaises(ValueError):
            pulse.get_measurement_windows(measurement_mapping=measurement_mapping,
                                          parameters=dict(length=10, a=11, d=3))
        with self.assertRaises(ValueError):
            pulse.get_measurement_windows(measurement_mapping=measurement_mapping,
                                          parameters=dict(length=10, a=3, d=-1))
