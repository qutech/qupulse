import unittest
from unittest import mock

from typing import Optional, Dict, Set, Any, Union

from qupulse.utils.types import ChannelID
from qupulse.expressions import Expression, ExpressionScalar
from qupulse.pulses.pulse_template import AtomicPulseTemplate, PulseTemplate
from qupulse._program.instructions import Waveform, EXECInstruction, MEASInstruction
from qupulse.pulses.parameters import Parameter, ConstantParameter, ParameterNotProvidedException
from qupulse.pulses.multi_channel_pulse_template import MultiChannelWaveform
from qupulse._program._loop import Loop, MultiChannelProgram

from qupulse._program.transformation import Transformation
from qupulse._program.waveforms import TransformingWaveform
from qupulse.pulses.sequencing import Sequencer

from tests.pulses.sequencing_dummies import DummyWaveform, DummySequencer, DummyInstructionBlock
from tests._program.transformation_tests import TransformationStub


class PulseTemplateStub(PulseTemplate):
    """All abstract methods are stubs that raise NotImplementedError to catch unexpected calls. If a method is needed in
    a test one should use mock.patch or mock.patch.object"""
    def __init__(self, identifier=None,
                 defined_channels=None,
                 duration=None,
                 parameter_names=None,
                 measurement_names=None,
                 registry=None):
        super().__init__(identifier=identifier)

        self._defined_channels = defined_channels
        self._duration = duration
        self._parameter_names = parameter_names
        self._measurement_names = set() if measurement_names is None else measurement_names
        self.internal_create_program_args = []
        self._register(registry=registry)

    @property
    def defined_channels(self) -> Set[ChannelID]:
        if self._defined_channels:
            return self._defined_channels
        else:
            raise NotImplementedError()

    @property
    def parameter_names(self) -> Set[str]:
        if self._parameter_names is None:
            raise NotImplementedError()
        return self._parameter_names

    def get_serialization_data(self, serializer: Optional['Serializer']=None) -> Dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def deserialize(cls, serializer: Optional['Serializer']=None, **kwargs) -> 'AtomicPulseTemplateStub':
        raise NotImplementedError()

    @property
    def duration(self) -> Expression:
        if self._duration is None:
            raise NotImplementedError()
        return self._duration

    def build_sequence(self,
                       sequencer: "Sequencer",
                       parameters: Dict[str, Parameter],
                       conditions: Dict[str, 'Condition'],
                       measurement_mapping: Dict[str, str],
                       channel_mapping: Dict['ChannelID', 'ChannelID'],
                       instruction_block: 'InstructionBlock'):
        raise NotImplementedError()

    def _internal_create_program(self, *,
                                 parameters: Dict[str, Parameter],
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional[Transformation],
                                 to_single_waveform: Set[Union[str, 'PulseTemplate']],
                                 parent_loop: Loop):
        raise NotImplementedError()

    def is_interruptable(self):
        raise NotImplementedError()

    @property
    def measurement_names(self):
        return self._measurement_names

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']):
        raise NotImplementedError()

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        raise NotImplementedError()


def get_appending_internal_create_program(waveform=DummyWaveform(),
                                          always_append=False,
                                          measurements: list=None):
    def internal_create_program(*, parameters, parent_loop: Loop, **_):
        if always_append or 'append_a_child' in parameters:
            if measurements is not None:
                parent_loop.add_measurements(measurements=measurements)
            parent_loop.append_child(waveform=waveform)

    return internal_create_program


class AtomicPulseTemplateStub(AtomicPulseTemplate):
    def is_interruptable(self) -> bool:
        return super().is_interruptable()

    def __init__(self, *, duration: Expression=None, measurements=None,
                 parameter_names: Optional[Set] = None, identifier: Optional[str]=None,
                 registry=None) -> None:
        super().__init__(identifier=identifier, measurements=measurements)
        self._duration = duration
        self._parameter_names = parameter_names
        self._register(registry=registry)

    def build_waveform(self, parameters: Dict[str, Parameter], channel_mapping):
        raise NotImplementedError()

    def requires_stop(self,
                      parameters: Dict[str, Parameter],
                      conditions: Dict[str, 'Condition']) -> bool:
        return False

    @property
    def defined_channels(self) -> Set['ChannelID']:
        raise NotImplementedError()

    @property
    def parameter_names(self) -> Set[str]:
        if self._parameter_names is None:
            raise NotImplementedError()
        return self._parameter_names

    def get_serialization_data(self, serializer: Optional['Serializer']=None) -> Dict[str, Any]:
        raise NotImplementedError()

    @property
    def measurement_names(self):
        raise NotImplementedError()

    @classmethod
    def deserialize(cls, serializer: Optional['Serializer']=None, **kwargs) -> 'AtomicPulseTemplateStub':
        raise NotImplementedError()

    @property
    def duration(self) -> Expression:
        return self._duration

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        raise NotImplementedError()


class PulseTemplateTest(unittest.TestCase):

    def test_create_program(self) -> None:
        template = PulseTemplateStub(defined_channels={'A'}, parameter_names={'foo'})
        parameters = {'foo': ConstantParameter(2.126), 'bar': -26.2, 'hugo': 'exp(sin(pi/2))', 'append_a_child': '1'}
        measurement_mapping = {'M': 'N'}
        channel_mapping = {'A': 'B'}

        expected_parameters = {'foo': ConstantParameter(2.126), 'bar': ConstantParameter(-26.2),
                               'hugo': ConstantParameter('exp(sin(pi/2))'), 'append_a_child': ConstantParameter('1')}
        to_single_waveform = {'voll', 'toggo'}
        global_transformation = TransformationStub()

        expected_internal_kwargs = dict(parameters=expected_parameters,
                                        measurement_mapping=measurement_mapping,
                                        channel_mapping=channel_mapping,
                                        global_transformation=global_transformation,
                                        to_single_waveform=to_single_waveform)

        dummy_waveform = DummyWaveform()
        expected_program = Loop(children=[Loop(waveform=dummy_waveform)])

        with mock.patch.object(template,
                               '_create_program',
                               wraps=get_appending_internal_create_program(dummy_waveform)) as _create_program:
            program = template.create_program(parameters=parameters,
                                              measurement_mapping=measurement_mapping,
                                              channel_mapping=channel_mapping,
                                              to_single_waveform=to_single_waveform,
                                              global_transformation=global_transformation)
            _create_program.assert_called_once_with(**expected_internal_kwargs, parent_loop=program)
        self.assertEqual(expected_program, program)

    def test__create_program(self):
        parameters = {'a': ConstantParameter(.1), 'b': ConstantParameter(.2)}
        measurement_mapping = {'M': 'N'}
        channel_mapping = {'B': 'A'}
        global_transformation = TransformationStub()
        to_single_waveform = {'voll', 'toggo'}
        parent_loop = Loop()

        template = PulseTemplateStub()
        with mock.patch.object(template, '_internal_create_program') as _internal_create_program:
            template._create_program(parameters=parameters,
                                     measurement_mapping=measurement_mapping,
                                     channel_mapping=channel_mapping,
                                     global_transformation=global_transformation,
                                     to_single_waveform=to_single_waveform,
                                     parent_loop=parent_loop)

            _internal_create_program.assert_called_once_with(parameters=parameters,
                                     measurement_mapping=measurement_mapping,
                                     channel_mapping=channel_mapping,
                                     global_transformation=global_transformation,
                                     to_single_waveform=to_single_waveform,
                                     parent_loop=parent_loop)

            self.assertEqual(parent_loop, Loop())

    def test__create_program_single_waveform(self):
        template = PulseTemplateStub(identifier='pt_identifier')

        for to_single_waveform in ({template}, {template.identifier}):
            for global_transformation in (None, TransformationStub()):
                parameters = {'a': ConstantParameter(.1), 'b': ConstantParameter(.2)}
                measurement_mapping = {'M': 'N'}
                channel_mapping = {'B': 'A'}
                parent_loop = Loop()

                wf = DummyWaveform()
                single_waveform = DummyWaveform()
                measurements = [('m', 0, 1), ('n', 0.1, .9)]

                expected_inner_program = Loop(children=[Loop(waveform=wf)], measurements=measurements)

                appending_create_program = get_appending_internal_create_program(wf,
                                                                                 measurements=measurements,
                                                                                 always_append=True)

                if global_transformation:
                    final_waveform = TransformingWaveform(single_waveform, global_transformation)
                else:
                    final_waveform = single_waveform

                expected_program = Loop(children=[Loop(waveform=final_waveform)],
                                        measurements=measurements)

                with mock.patch.object(template, '_internal_create_program',
                                       wraps=appending_create_program) as _internal_create_program:
                    with mock.patch('qupulse.pulses.pulse_template.to_waveform',
                                    return_value=single_waveform) as to_waveform:
                        template._create_program(parameters=parameters,
                                                 measurement_mapping=measurement_mapping,
                                                 channel_mapping=channel_mapping,
                                                 global_transformation=global_transformation,
                                                 to_single_waveform=to_single_waveform,
                                                 parent_loop=parent_loop)

                        _internal_create_program.assert_called_once_with(parameters=parameters,
                                                                         measurement_mapping=measurement_mapping,
                                                                         channel_mapping=channel_mapping,
                                                                         global_transformation=None,
                                                                         to_single_waveform=to_single_waveform,
                                                                         parent_loop=expected_inner_program)

                        to_waveform.assert_called_once_with(expected_inner_program)

                        expected_program._measurements = set(expected_program._measurements)
                        parent_loop._measurements = set(parent_loop._measurements)

                        self.assertEqual(expected_program, parent_loop)

    def test_create_program_defaults(self) -> None:
        template = PulseTemplateStub(defined_channels={'A', 'B'}, parameter_names={'foo'}, measurement_names={'hugo', 'foo'})

        expected_internal_kwargs = dict(parameters=dict(),
                                        measurement_mapping={'hugo': 'hugo', 'foo': 'foo'},
                                        channel_mapping={'A': 'A', 'B': 'B'},
                                        global_transformation=None,
                                        to_single_waveform=set())

        dummy_waveform = DummyWaveform()
        expected_program = Loop(children=[Loop(waveform=dummy_waveform)])

        with mock.patch.object(template,
                               '_internal_create_program',
                               wraps=get_appending_internal_create_program(dummy_waveform, True)) as _internal_create_program:
            program = template.create_program()
            _internal_create_program.assert_called_once_with(**expected_internal_kwargs, parent_loop=program)
        self.assertEqual(expected_program, program)

    def test_create_program_channel_mapping(self):
        template = PulseTemplateStub(defined_channels={'A', 'B'})

        expected_internal_kwargs = dict(parameters=dict(),
                                        measurement_mapping=dict(),
                                        channel_mapping={'A': 'C', 'B': 'B'},
                                        global_transformation=None,
                                        to_single_waveform=set())

        with mock.patch.object(template, '_internal_create_program') as _internal_create_program:
            template.create_program(channel_mapping={'A': 'C'})

            _internal_create_program.assert_called_once_with(**expected_internal_kwargs, parent_loop=Loop())


    def test_create_program_none(self) -> None:
        template = PulseTemplateStub(defined_channels={'A'}, parameter_names={'foo'})
        parameters = {'foo': ConstantParameter(2.126), 'bar': -26.2, 'hugo': 'exp(sin(pi/2))'}
        measurement_mapping = {'M': 'N'}
        channel_mapping = {'A': 'B'}
        expected_parameters = {'foo': ConstantParameter(2.126), 'bar': ConstantParameter(-26.2),
                               'hugo': ConstantParameter('exp(sin(pi/2))')}
        expected_internal_kwargs = dict(parameters=expected_parameters,
                                        measurement_mapping=measurement_mapping,
                                        channel_mapping=channel_mapping,
                                        global_transformation=None,
                                        to_single_waveform=set())

        with mock.patch.object(template,
                               '_internal_create_program') as _internal_create_program:
            program = template.create_program(parameters=parameters,
                                              measurement_mapping=measurement_mapping,
                                              channel_mapping=channel_mapping)
            _internal_create_program.assert_called_once_with(**expected_internal_kwargs, parent_loop=Loop())
        self.assertIsNone(program)

    def test_matmul(self):
        a = PulseTemplateStub()
        b = PulseTemplateStub()

        from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate
        with mock.patch.object(SequencePulseTemplate, 'concatenate', return_value='concat') as mock_concatenate:
            self.assertEqual(a @ b, 'concat')
            mock_concatenate.assert_called_once_with(a, b)

    def test_rmatmul(self):
        a = PulseTemplateStub()
        b = (1, 2, 3)

        from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate
        with mock.patch.object(SequencePulseTemplate, 'concatenate', return_value='concat') as mock_concatenate:
            self.assertEqual(b @ a, 'concat')
            mock_concatenate.assert_called_once_with(b, a)


class AtomicPulseTemplateTests(unittest.TestCase):

    def test_is_interruptable(self) -> None:
        template = AtomicPulseTemplateStub()
        self.assertFalse(template.is_interruptable())
        template = AtomicPulseTemplateStub(identifier="arbg4")
        self.assertFalse(template.is_interruptable())

    def test_build_sequence_no_waveform(self) -> None:
        sequencer = DummySequencer()
        block = DummyInstructionBlock()

        template = AtomicPulseTemplateStub()
        with mock.patch.object(template, 'build_waveform', return_value=None):
            template.build_sequence(sequencer, {}, {}, {}, {}, block)
        self.assertFalse(block.instructions)

    def test_build_sequence(self) -> None:
        measurement_windows = [('M', 0, 5)]
        single_wf = DummyWaveform(duration=6, defined_channels={'A'})
        wf = MultiChannelWaveform([single_wf])

        sequencer = DummySequencer()
        block = DummyInstructionBlock()

        parameters = {'a': ConstantParameter(1), 'b': ConstantParameter(2), 'c': ConstantParameter(3)}
        expected_parameters = {'a': 1, 'b': 2}
        channel_mapping = {'B': 'A'}

        template = AtomicPulseTemplateStub(measurements=measurement_windows, parameter_names={'a', 'b'})
        with mock.patch.object(template, 'build_waveform', return_value=wf) as build_waveform:
            template.build_sequence(sequencer, parameters=parameters, conditions={},
                                    measurement_mapping={'M': 'N'}, channel_mapping=channel_mapping,
                                    instruction_block=block)
            build_waveform.assert_called_once_with(parameters=expected_parameters, channel_mapping=channel_mapping)
        self.assertEqual(len(block.instructions), 2)

        meas, exec = block.instructions
        self.assertIsInstance(meas, MEASInstruction)
        self.assertEqual(meas.measurements, [('N', 0, 5)])

        self.assertIsInstance(exec, EXECInstruction)
        self.assertEqual(exec.waveform.defined_channels, {'A'})

    def test_internal_create_program(self) -> None:
        measurement_windows = [('M', 0, 5)]
        single_wf = DummyWaveform(duration=6, defined_channels={'A'})
        wf = MultiChannelWaveform([single_wf])

        template = AtomicPulseTemplateStub(measurements=measurement_windows, parameter_names={'foo'})
        parameters = {'foo': ConstantParameter(7.2)}
        measurement_mapping = {'M': 'N'}
        channel_mapping = {'B': 'A'}
        program = Loop()

        expected_parameters = {k: p.get_value() for k, p in parameters.items()}
        expected_program = Loop(children=[Loop(waveform=wf)],
                                measurements=[('N', 0, 5)])

        with mock.patch.object(template, 'build_waveform', return_value=wf) as build_waveform:
            template._internal_create_program(parameters=parameters,
                                              measurement_mapping=measurement_mapping,
                                              channel_mapping=channel_mapping,
                                              parent_loop=program,
                                              to_single_waveform=set(),
                                              global_transformation=None)
            build_waveform.assert_called_once_with(parameters=expected_parameters, channel_mapping=channel_mapping)

        self.assertEqual(expected_program, program)

        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(template, parameters=parameters, conditions={}, window_mapping=measurement_mapping,
                       channel_mapping=channel_mapping)
        with mock.patch.object(template, 'build_waveform', return_value=wf):
            block = sequencer.build()
        old_program = MultiChannelProgram(block, channels={'A'})
        self.assertEqual(old_program.programs[frozenset({'A'})], program)

    def test_internal_create_program_transformation(self):
        inner_wf = DummyWaveform()
        template = AtomicPulseTemplateStub(parameter_names=set())
        program = Loop()
        global_transformation = TransformationStub()

        expected_program = Loop(children=[Loop(waveform=TransformingWaveform(inner_wf, global_transformation))])

        with mock.patch.object(template, 'build_waveform', return_value=inner_wf):
            template._internal_create_program(parameters={},
                                              measurement_mapping={},
                                              channel_mapping={},
                                              parent_loop=program,
                                              to_single_waveform=set(),
                                              global_transformation=global_transformation)

        self.assertEqual(expected_program, program)

    def test_internal_create_program_no_waveform(self) -> None:
        measurement_windows = [('M', 0, 5)]

        template = AtomicPulseTemplateStub(measurements=measurement_windows, parameter_names={'foo'})
        parameters = {'foo': ConstantParameter(7.2)}
        measurement_mapping = {'M': 'N'}
        channel_mapping = {'B': 'A'}
        program = Loop()

        expected_parameters = {k: p.get_value() for k, p in parameters.items()}
        expected_program = Loop()

        with mock.patch.object(template, 'build_waveform', return_value=None) as build_waveform:
            with mock.patch.object(template,
                                   'get_measurement_windows',
                                   wraps=template.get_measurement_windows) as get_meas_windows:
                template._internal_create_program(parameters=parameters,
                                                  measurement_mapping=measurement_mapping,
                                                  channel_mapping=channel_mapping,
                                                  parent_loop=program,
                                                  to_single_waveform=set(),
                                                  global_transformation=None)
                build_waveform.assert_called_once_with(parameters=expected_parameters, channel_mapping=channel_mapping)
                get_meas_windows.assert_not_called()

        self.assertEqual(expected_program, program)

        # ensure same result as from Sequencer
        sequencer = Sequencer()
        sequencer.push(template, parameters=parameters, conditions={}, window_mapping=measurement_mapping, channel_mapping=channel_mapping)
        with mock.patch.object(template, 'build_waveform', return_value=None):
            block = sequencer.build()
        old_program = MultiChannelProgram(block, channels={'A'})
        self.assertEqual(old_program.programs[frozenset({'A'})], program)

    @unittest.skip('not a job of internal_create_program: remove?')
    def test_internal_create_program_invalid_measurement_mapping(self) -> None:
        measurement_windows = [('M', 0, 5)]
        wf = DummyWaveform(duration=6, defined_channels={'A'})

        template = AtomicPulseTemplateStub(measurements=measurement_windows, parameter_names={'foo'})
        parameters = {'foo': ConstantParameter(7.2)}
        program = Loop()
        with self.assertRaises(KeyError):
            template._internal_create_program(parameters=parameters,
                                              measurement_mapping=dict(),
                                              channel_mapping=dict(),
                                              parent_loop=program)

    def test_internal_create_program_missing_parameters(self) -> None:
        measurement_windows = [('M', 'z', 5)]
        wf = DummyWaveform(duration=6, defined_channels={'A'})

        template = AtomicPulseTemplateStub(measurements=measurement_windows, parameter_names={'foo'})

        program = Loop()
        # test parameter from declarations
        parameters = {}
        with self.assertRaises(ParameterNotProvidedException):
            template._internal_create_program(parameters=parameters,
                                              measurement_mapping=dict(),
                                              channel_mapping=dict(),
                                              parent_loop=program,
                                              to_single_waveform=set(),
                                              global_transformation=None)
