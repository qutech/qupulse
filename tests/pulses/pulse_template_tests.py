import unittest
import math
from unittest import mock

from typing import Optional, Dict, Set, Any, Union, Sequence

import frozendict
import sympy

from qupulse.parameter_scope import Scope, DictScope
from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate
from qupulse.utils.types import ChannelID
from qupulse.expressions import Expression, ExpressionScalar
from qupulse.pulses import ConstantPT, FunctionPT, RepetitionPT, ForLoopPT, ParallelChannelPT, MappingPT,\
    TimeReversalPT, AtomicMultiChannelPT, SequencePT
from qupulse.pulses.pulse_template import AtomicPulseTemplate, PulseTemplate, UnknownVolatileParameter
from qupulse.pulses.multi_channel_pulse_template import MultiChannelWaveform
from qupulse.program.loop import Loop
from qupulse.program import ProgramBuilder, default_program_builder

from qupulse._program.transformation import Transformation
from qupulse._program.waveforms import TransformingWaveform

from tests.pulses.sequencing_dummies import DummyWaveform
from tests._program.transformation_tests import TransformationStub

from qupulse.program.loop import LoopBuilder


class PulseTemplateStub(PulseTemplate):
    """All abstract methods are stubs that raise NotImplementedError to catch unexpected calls. If a method is needed in
    a test one should use mock.patch or mock.patch.object.
    Properties can be passed as init argument because mocking them is a pita."""
    def __init__(self, identifier=None,
                 defined_channels=None,
                 duration=None,
                 parameter_names=None,
                 measurement_names=None,
                 final_values=None,
                 registry=None):
        super().__init__(identifier=identifier)

        self._defined_channels = defined_channels
        self._duration = duration
        self._parameter_names = parameter_names
        self._measurement_names = set() if measurement_names is None else measurement_names
        self._final_values = final_values
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
        # required for hashability
        return {'id_self': id(self)}

    @classmethod
    def deserialize(cls, serializer: Optional['Serializer']=None, **kwargs) -> 'AtomicPulseTemplateStub':
        raise NotImplementedError()

    @property
    def duration(self) -> Expression:
        if self._duration is None:
            raise NotImplementedError()
        return self._duration

    def _internal_create_program(self, *,
                                 scope: Scope,
                                 measurement_mapping: Dict[str, Optional[str]],
                                 channel_mapping: Dict[ChannelID, Optional[ChannelID]],
                                 global_transformation: Optional[Transformation],
                                 to_single_waveform: Set[Union[str, 'PulseTemplate']],
                                 program_builder):
        raise NotImplementedError()

    @property
    def measurement_names(self):
        return self._measurement_names

    @property
    def integral(self) -> Dict[ChannelID, ExpressionScalar]:
        raise NotImplementedError()

    @property
    def initial_values(self) -> Dict[ChannelID, ExpressionScalar]:
        raise NotImplementedError()

    @property
    def final_values(self) -> Dict[ChannelID, ExpressionScalar]:
        if self._final_values is None:
            raise NotImplementedError()
        else:
            return self._final_values


def get_appending_internal_create_program(waveform=DummyWaveform(),
                                          always_append=False,
                                          measurements: list=None):
    def internal_create_program(*, scope, program_builder: ProgramBuilder, **_):
        if always_append or 'append_a_child' in scope:
            if measurements is not None:
                program_builder.measure(measurements=measurements)
            program_builder.play_arbitrary_waveform(waveform=waveform)

    return internal_create_program


class AtomicPulseTemplateStub(AtomicPulseTemplate):
    def __init__(self, *, duration: Expression=None, measurements=None,
                 parameter_names: Optional[Set] = None, identifier: Optional[str]=None,
                 registry=None) -> None:
        super().__init__(identifier=identifier, measurements=measurements)
        self._duration = duration
        self._parameter_names = parameter_names
        self._register(registry=registry)

    def build_waveform(self, parameters, channel_mapping):
        raise NotImplementedError()

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

    def _as_expression(self) -> Dict[ChannelID, ExpressionScalar]:
        raise NotImplementedError()


class PulseTemplateTest(unittest.TestCase):

    def test_create_program(self) -> None:
        template = PulseTemplateStub(defined_channels={'A'}, parameter_names={'foo'})
        parameters = {'foo': 2.126, 'bar': -26.2, 'hugo': 'exp(sin(pi/2))', 'append_a_child': '1'}
        previous_parameters = parameters.copy()
        measurement_mapping = {'M': 'N'}
        previos_measurement_mapping = measurement_mapping.copy()
        channel_mapping = {'A': 'B'}
        previous_channel_mapping = channel_mapping.copy()
        volatile = {'foo'}

        expected_scope = DictScope.from_kwargs(foo=2.126, bar=-26.2, hugo=math.exp(math.sin(math.pi/2)),
                                               volatile=volatile, append_a_child=1)
        to_single_waveform = {'voll', 'toggo'}
        global_transformation = TransformationStub()

        expected_internal_kwargs = dict(scope=expected_scope,
                                        measurement_mapping=measurement_mapping,
                                        channel_mapping=channel_mapping,
                                        global_transformation=global_transformation,
                                        to_single_waveform=to_single_waveform)

        dummy_waveform = DummyWaveform()
        expected_program = Loop(children=[Loop(waveform=dummy_waveform)])

        program_builder = LoopBuilder()

        with mock.patch.object(template,
                               '_create_program',
                               wraps=get_appending_internal_create_program(dummy_waveform)) as _create_program:
            with mock.patch('qupulse.pulses.pulse_template.default_program_builder', return_value=program_builder):
                program = template.create_program(parameters=parameters,
                                                  measurement_mapping=measurement_mapping,
                                                  channel_mapping=channel_mapping,
                                                  to_single_waveform=to_single_waveform,
                                                  global_transformation=global_transformation,
                                                  volatile=volatile)
            _create_program.assert_called_once_with(**expected_internal_kwargs, program_builder=program_builder)
        self.assertEqual(expected_program, program)
        self.assertEqual(previos_measurement_mapping, measurement_mapping)
        self.assertEqual(previous_channel_mapping, channel_mapping)
        self.assertEqual(previous_parameters, parameters)

    def test__create_program(self):
        scope = DictScope.from_kwargs(a=1., b=2., volatile={'c'})
        measurement_mapping = {'M': 'N'}
        channel_mapping = {'B': 'A'}
        global_transformation = TransformationStub()
        to_single_waveform = {'voll', 'toggo'}
        program_builder = LoopBuilder()

        template = PulseTemplateStub()
        with mock.patch.object(template, '_internal_create_program') as _internal_create_program:
            with self.assertWarns(DeprecationWarning):
                template._create_program(scope=scope,
                                         measurement_mapping=measurement_mapping,
                                         channel_mapping=channel_mapping,
                                         global_transformation=global_transformation,
                                         to_single_waveform=to_single_waveform,
                                         program_builder=program_builder)

            _internal_create_program.assert_called_once_with(
                scope=scope,
                measurement_mapping=measurement_mapping,
                channel_mapping=channel_mapping,
                global_transformation=global_transformation,
                to_single_waveform=to_single_waveform,
                program_builder=program_builder)

            self.assertIsNone(program_builder.to_program())

            with self.assertRaisesRegex(NotImplementedError, "volatile"):
                template._parameter_names = {'c'}
                with self.assertWarns(DeprecationWarning):
                    template._create_program(scope=scope,
                                             measurement_mapping=measurement_mapping,
                                             channel_mapping=channel_mapping,
                                             global_transformation=global_transformation,
                                             to_single_waveform={template},
                                             program_builder=program_builder)

    def test__create_program_single_waveform(self):
        template = PulseTemplateStub(identifier='pt_identifier', parameter_names={'alpha'})

        for to_single_waveform in ({template}, {template.identifier}):
            for global_transformation in (None, TransformationStub()):
                scope = DictScope.from_kwargs(a=1., b=2., volatile={'a'})
                measurement_mapping = {'M': 'N'}
                channel_mapping = {'B': 'A'}

                program_builder = LoopBuilder()
                inner_program_builder = LoopBuilder()

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
                    with mock.patch('qupulse.program.loop.to_waveform',
                                    return_value=single_waveform) as to_waveform:
                        with mock.patch('qupulse.program.loop.LoopBuilder', return_value=inner_program_builder):
                            with self.assertWarns(DeprecationWarning):
                                template._create_program(scope=scope,
                                                         measurement_mapping=measurement_mapping,
                                                         channel_mapping=channel_mapping,
                                                         global_transformation=global_transformation,
                                                         to_single_waveform=to_single_waveform,
                                                         program_builder=program_builder)

                        _internal_create_program.assert_called_once_with(scope=scope,
                                                                         measurement_mapping=measurement_mapping,
                                                                         channel_mapping=channel_mapping,
                                                                         global_transformation=None,
                                                                         to_single_waveform=to_single_waveform,
                                                                         program_builder=inner_program_builder)

                        to_waveform.assert_called_once_with(expected_inner_program)

                        program = program_builder.to_program()

                        expected_program._measurements = set(expected_program._measurements)
                        program._measurements = set(program._measurements)
                        self.assertEqual(expected_program, program)

    def test_create_program_defaults(self) -> None:
        template = PulseTemplateStub(defined_channels={'A', 'B'}, parameter_names={'foo'}, measurement_names={'hugo', 'foo'})

        expected_internal_kwargs = dict(scope=DictScope.from_kwargs(),
                                        measurement_mapping={'hugo': 'hugo', 'foo': 'foo'},
                                        channel_mapping={'A': 'A', 'B': 'B'},
                                        global_transformation=None,
                                        to_single_waveform=set())

        dummy_waveform = DummyWaveform()
        expected_program = Loop(children=[Loop(waveform=dummy_waveform)])
        program_builder = LoopBuilder()

        with mock.patch.object(template,
                               '_internal_create_program',
                               wraps=get_appending_internal_create_program(dummy_waveform, True)) as _internal_create_program:
            with mock.patch('qupulse.pulses.pulse_template.default_program_builder', return_value=program_builder) as pb:
                program = template.create_program()
            pb.assert_called_once_with()
            _internal_create_program.assert_called_once_with(**expected_internal_kwargs, program_builder=program_builder)
        self.assertEqual(expected_program, program)

    def test_create_program_channel_mapping(self):
        template = PulseTemplateStub(defined_channels={'A', 'B'})

        expected_internal_kwargs = dict(scope=DictScope.from_kwargs(),
                                        measurement_mapping=dict(),
                                        channel_mapping={'A': 'C', 'B': 'B'},
                                        global_transformation=None,
                                        to_single_waveform=set())

        with mock.patch('qupulse.pulses.pulse_template.default_program_builder') as pb:
            with mock.patch.object(template, '_internal_create_program') as _internal_create_program:
                template.create_program(channel_mapping={'A': 'C'})
            pb.assert_called_once_with()
            _internal_create_program.assert_called_once_with(**expected_internal_kwargs, program_builder=pb.return_value)

    def test_create_program_volatile(self):
        template = PulseTemplateStub(defined_channels={'A', 'B'})

        parameters = {'abc': 1.}

        expected_internal_kwargs = dict(scope=DictScope.from_kwargs(volatile={'abc'}, **parameters),
                                        measurement_mapping=dict(),
                                        channel_mapping={'A': 'A', 'B': 'B'},
                                        global_transformation=None,
                                        to_single_waveform=set())

        with mock.patch.object(template, '_internal_create_program') as _internal_create_program:
            program_builder = default_program_builder()
            with mock.patch('qupulse.pulses.pulse_template.default_program_builder', return_value=program_builder):
                template.create_program(parameters=parameters, volatile='abc')

            _internal_create_program.assert_called_once_with(**expected_internal_kwargs, program_builder=program_builder)
        with mock.patch.object(template, '_internal_create_program') as _internal_create_program:
            program_builder = default_program_builder()
            with mock.patch('qupulse.pulses.pulse_template.default_program_builder', return_value=program_builder):
                template.create_program(parameters=parameters, volatile={'abc'})

            _internal_create_program.assert_called_once_with(**expected_internal_kwargs, program_builder=program_builder)

        expected_internal_kwargs = dict(scope=DictScope.from_kwargs(volatile={'abc', 'dfg'}, **parameters),
                                        measurement_mapping=dict(),
                                        channel_mapping={'A': 'A', 'B': 'B'},
                                        global_transformation=None,
                                        to_single_waveform=set())

        program_builder = default_program_builder()
        with mock.patch('qupulse.pulses.pulse_template.default_program_builder', return_value=program_builder):
            with mock.patch.object(template, '_internal_create_program') as _internal_create_program:
                with self.assertWarns(UnknownVolatileParameter):
                    template.create_program(parameters=parameters, volatile={'abc', 'dfg'})
        _internal_create_program.assert_called_once_with(**expected_internal_kwargs, program_builder=program_builder)

    def test_pad_to(self):
        def to_multiple_of_192(x: Expression) -> Expression:
            return (x + 191) // 192 * 192

        final_values = frozendict.frozendict({'A': ExpressionScalar(0.1), 'B': ExpressionScalar('a')})
        measurements = [('M', 0, 'y')]

        pt = PulseTemplateStub(duration=ExpressionScalar(10))
        padded = pt.pad_to(10)
        self.assertIs(pt, padded)

        pt = PulseTemplateStub(duration=ExpressionScalar('duration'))
        padded = pt.pad_to('duration')
        self.assertIs(pt, padded)

        # padding with numeric durations

        pt = PulseTemplateStub(duration=ExpressionScalar(10),
                               final_values=final_values,
                               defined_channels=final_values.keys())
        padded = pt.pad_to(20)
        self.assertEqual(padded.duration, 20)
        self.assertEqual(padded.final_values, final_values)
        self.assertIsInstance(padded, SequencePT)
        self.assertIs(padded.subtemplates[0], pt)

        with self.assertWarns(DeprecationWarning):
            padded = pt.pad_to(20, pt_kwargs=dict(measurements=measurements))
        self.assertEqual(padded.duration, 20)
        self.assertEqual(padded.final_values, final_values)
        self.assertIsInstance(padded, SequencePT)
        self.assertIs(padded.subtemplates[0], pt)
        self.assertEqual(measurements, padded.measurement_declarations)

        with self.assertWarns(DeprecationWarning):
            padded = pt.pad_to(10, pt_kwargs=dict(measurements=measurements))
        self.assertEqual(padded.duration, 10)
        self.assertEqual(padded.final_values, final_values)
        self.assertIsInstance(padded, SequencePT)
        self.assertIs(padded.subtemplates[0], pt)
        self.assertEqual(measurements, padded.measurement_declarations)

        # padding with numeric duation and callable
        padded = pt.pad_to(to_multiple_of_192)
        self.assertEqual(padded.duration, 192)
        self.assertEqual(padded.final_values, final_values)
        self.assertIsInstance(padded, SequencePT)
        self.assertIs(padded.subtemplates[0], pt)

        # padding with metadata
        padded = pt.pad_to(to_multiple_of_192, spt_kwargs=dict(metadata={'to_single_waveform': 'always'}))
        self.assertEqual(padded.duration, 192)
        self.assertEqual(padded.final_values, final_values)
        self.assertIsInstance(padded, SequencePT)
        self.assertIs(padded.subtemplates[0], pt)
        self.assertEqual(padded.metadata.get_serialization_data(), {'to_single_waveform': 'always'})

        # padding with symbolic durations

        pt = PulseTemplateStub(duration=ExpressionScalar('duration'),
                               final_values=final_values,
                               defined_channels=final_values.keys())
        padded = pt.pad_to('new_duration')
        self.assertEqual(padded.duration, 'new_duration')
        self.assertEqual(padded.final_values, final_values)
        self.assertIsInstance(padded, SequencePT)
        self.assertIs(padded.subtemplates[0], pt)

        # padding symbolic durations with callable

        padded = pt.pad_to(to_multiple_of_192)
        self.assertEqual(padded.duration, '(duration + 191) // 192 * 192')
        self.assertEqual(padded.final_values, final_values)
        self.assertIsInstance(padded, SequencePT)
        self.assertIs(padded.subtemplates[0], pt)

        # padding with metadata
        padded = pt.pad_to(to_multiple_of_192, spt_kwargs=dict(metadata={'to_single_waveform': 'always'}))
        self.assertEqual(padded.duration, '(duration + 191) // 192 * 192')
        self.assertEqual(padded.final_values, final_values)
        self.assertIsInstance(padded, SequencePT)
        self.assertIs(padded.subtemplates[0], pt)
        self.assertEqual(padded.metadata.get_serialization_data(), {'to_single_waveform': 'always'})


    def test_pad_selected_subtemplates(self):
        def to_multiple_of_192(x: Expression) -> Expression:
            return (x + 191) // 192 * 192

        final_values = frozendict.frozendict({'A': ExpressionScalar(0.1), 'B': ExpressionScalar('a')})

        class DummyAPT(AtomicPulseTemplateStub):
            @property
            def final_values(self):
                return final_values

            @property
            def defined_channels(self):
                return final_values.keys()

            def get_serialization_data(self, serializer=None) -> Dict[str, Any]:
                assert not serializer
                return {'duration': self.duration}


        pt_10 = DummyAPT(duration=ExpressionScalar(10))
        padded_10 = pt_10.pad_to(to_multiple_of_192)
        padded_10_atomic = pt_10.pad_to(to_multiple_of_192, spt_kwargs=dict(metadata={'to_single_waveform': 'always'}))
        pt_192 = DummyAPT(duration=ExpressionScalar(192))
        pt_192_padded = pt_192.pad_to(to_multiple_of_192)
        pt_192_padded_atomic = pt_192.pad_to(to_multiple_of_192, spt_kwargs=dict(metadata={'to_single_waveform': 'always'}))
        pt_dyn = DummyAPT(duration=ExpressionScalar('duration'))
        pt_dyn_padded = pt_dyn.pad_to(to_multiple_of_192)
        pt_dyn_padded_atomic = pt_dyn.pad_to(to_multiple_of_192, spt_kwargs=dict(metadata={'to_single_waveform': 'always'}))
        self.assertEqual(pt_dyn_padded, pt_dyn_padded_atomic)
        self.assertEqual(padded_10, padded_10_atomic)
        self.assertFalse(padded_10._is_atomic())
        self.assertFalse(pt_dyn_padded._is_atomic())
        self.assertTrue(padded_10_atomic._is_atomic())
        self.assertTrue(pt_dyn_padded_atomic._is_atomic())
        self.assertIs(pt_192_padded, pt_192)
        self.assertIs(pt_192_padded_atomic, pt_192)

        flat_spt = SequencePT(pt_10, pt_192, pt_dyn)

        padded_flat = flat_spt.pad_selected_subtemplates_to(to_multiple_of_192)
        expected = SequencePT(padded_10_atomic, pt_192, pt_dyn_padded_atomic)
        self.assertEqual(expected, padded_flat)
        for subpt in padded_flat.subtemplates:
            self.assertTrue(subpt._is_atomic())

        padded_flat_non_atomic = flat_spt.pad_selected_subtemplates_to(to_multiple_of_192, spt_kwargs=dict(metadata={}))
        expected = SequencePT(padded_10, pt_192, pt_dyn_padded)
        self.assertEqual(expected, padded_flat_non_atomic)
        for expected_subpt, actual_subpt in zip(expected.subtemplates, padded_flat_non_atomic.subtemplates):
            self.assertEqual(expected_subpt._is_atomic(), actual_subpt._is_atomic())

        nested_pt = SequencePT(
            pt_dyn,
            RepetitionPT(pt_10, 2, 'rpt_10'),
            RepetitionPT(pt_192, 2, 'rpt_192'),
            pt_192,
        )
        nested_pt_padded = nested_pt.pad_selected_subtemplates_to(to_multiple_of_192)
        expected = SequencePT(
            pt_dyn_padded_atomic,
            RepetitionPT(padded_10_atomic, 2, 'rpt_10_padded'),
            RepetitionPT(pt_192, 2, 'rpt_192'),
            pt_192,
        )
        self.assertEqual(expected, nested_pt_padded)
        for subpt in nested_pt_padded.subtemplates:
            expected_atomic = getattr(subpt, 'body', subpt)
            self.assertTrue(expected_atomic._is_atomic())

    @mock.patch('qupulse.pulses.pulse_template.default_program_builder')
    def test_create_program_none(self, pb_mock) -> None:
        template = PulseTemplateStub(defined_channels={'A'}, parameter_names={'foo'})
        parameters = {'foo': 2.126, 'bar': -26.2, 'hugo': 'exp(sin(pi/2))'}
        measurement_mapping = {'M': 'N'}
        channel_mapping = {'A': 'B'}
        volatile = {'hugo'}

        scope = DictScope.from_kwargs(foo=2.126, bar=-26.2, hugo=math.exp(math.sin(math.pi/2)), volatile=volatile)
        expected_internal_kwargs = dict(scope=scope,
                                        measurement_mapping=measurement_mapping,
                                        channel_mapping=channel_mapping,
                                        global_transformation=None,
                                        to_single_waveform=set())
        pb_mock.return_value = LoopBuilder()

        with mock.patch.object(template,
                               '_internal_create_program') as _internal_create_program:
            program = template.create_program(parameters=parameters,
                                              measurement_mapping=measurement_mapping,
                                              channel_mapping=channel_mapping,
                                              volatile=volatile)
            pb_mock.assert_called_once_with()
            _internal_create_program.assert_called_once_with(**expected_internal_kwargs,
                                                             program_builder=pb_mock.return_value)
        self.assertIsNone(program)

    def test_matmul(self):
        a = PulseTemplateStub()
        b = PulseTemplateStub()

        from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate
        with mock.patch.object(SequencePulseTemplate, 'concatenate', return_value='concat') as mock_concatenate:
            self.assertEqual(a @ b, 'concat')
            mock_concatenate.assert_called_once_with(a, b)

    def test_pow(self):
        pt = PulseTemplateStub()
        pow_pt = pt ** 5
        self.assertEqual(pow_pt, pt.with_repetition(5))

    def test_rmatmul(self):
        a = PulseTemplateStub()
        b = (1, 2, 3)

        from qupulse.pulses.sequence_pulse_template import SequencePulseTemplate
        with mock.patch.object(SequencePulseTemplate, 'concatenate', return_value='concat') as mock_concatenate:
            self.assertEqual(b @ a, 'concat')
            mock_concatenate.assert_called_once_with(b, a)

    def test_format(self):
        a = PulseTemplateStub(identifier='asd', duration=Expression(5))
        self.assertEqual("PulseTemplateStub(identifier='asd')", str(a))
        self.assertEqual("PulseTemplateStub(identifier='asd')", format(a))
        self.assertEqual("PulseTemplateStub(identifier='asd', duration='5')",
                         "{:identifier;duration}".format(a))


class WithMethodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fpt = FunctionPT(1.4, 'sin(f*t)', 'X', identifier='fpt')
        self.cpt = ConstantPT(1.4, {'Y': 'start + idx * step'})
        self.spt = SequencePT(self.cpt, RepetitionPT(self.cpt, 2, identifier='rpt'), identifier='spt')

    def test_parallel_channels(self):
        expected = ParallelChannelPT(self.fpt, {'K': 'k'})
        actual = self.fpt.with_parallel_channels({'K': 'k'})
        self.assertEqual(expected, actual)

    def test_parallel_channels_optimization(self):
        expected = ParallelChannelPT(self.fpt, {'K': 'k', 'C': 'c'})
        actual = self.fpt.with_parallel_channels({'K': 'k'}).with_parallel_channels({'C': 'c'})
        self.assertEqual(expected, actual)

    def test_iteration(self):
        expected = ForLoopPT(self.cpt, 'idx', 'n_steps')
        actual = self.cpt.with_iteration('idx', 'n_steps')
        self.assertEqual(expected, actual)

    def test_appended(self):
        expected = self.fpt @ self.fpt.with_time_reversal()
        actual = self.fpt.with_appended(self.fpt.with_time_reversal())
        self.assertEqual(expected, actual)

    def test_repetition(self):
        expected = RepetitionPT(self.fpt, 6)
        actual = self.fpt.with_repetition(6)
        self.assertEqual(expected, actual)

    def test_repetition_optimization(self):
        # unstable test due to flimsy expression equality :(
        expected = RepetitionPT(self.fpt, ExpressionScalar(6) * 2)
        actual = self.fpt.with_repetition(6).with_repetition(2)
        self.assertEqual(expected, actual)

    def test_time_reversal(self):
        expected = TimeReversalPT(self.fpt)
        actual = self.fpt.with_time_reversal()
        self.assertEqual(expected, actual)

    def test_parallel_atomic(self):
        expected = AtomicMultiChannelPT(self.fpt, self.cpt)
        actual = self.fpt.with_parallel_atomic(self.cpt)
        self.assertEqual(expected, actual)

    def test_mapped_subtemplates(self):
        expected = self.fpt
        actual = self.fpt.with_mapped_subtemplates(map_fn=lambda x: 0/0)
        self.assertEqual(expected, actual)

        calls = []
        def swap_c_and_f(pt):
            calls.append(pt)
            if pt == self.cpt:
                return self.fpt
            elif pt == self.fpt:
                return self.cpt
            else:
                return pt

        def identifier_map(identifier):
            if identifier is None:
                return None
            else:
                return identifier + '_mapped'

        # PRE
        expected_pre = SequencePT(self.fpt,
                                  RepetitionPT(self.fpt, 2, identifier='rpt_mapped'),
                                  identifier='spt_mapped')
        expected_calls_pre = [
            self.cpt,
            self.cpt,
            RepetitionPT(self.fpt, 2, 'rpt_mapped')
        ]
        actual_pre = self.spt.with_mapped_subtemplates(map_fn=swap_c_and_f,
                                                       recursion_strategy='pre',
                                                       identifier_map=identifier_map)
        self.assertEqual(
            expected_calls_pre,
            calls,
        )
        self.assertEqual(expected_pre, actual_pre)

        # POST
        expected_post = SequencePT(self.fpt.renamed('fpt'),
                                  RepetitionPT(self.fpt.renamed('fpt'), 2, identifier='rpt_mapped'),
                                  identifier='spt_mapped')
        expected_calls_post = [
            self.cpt,
            RepetitionPT(self.cpt, 2, identifier='rpt'),
            self.cpt
        ]
        calls.clear()
        actual_post = self.spt.with_mapped_subtemplates(map_fn=swap_c_and_f,
                                                        identifier_map=identifier_map,
                                                        recursion_strategy='post')
        self.assertEqual(expected_calls_post, calls)
        self.assertEqual(expected_post, actual_post)
        calls.clear()

        inner = RepetitionPT(self.fpt, 2, identifier='new_rpt')
        expected_self = SequencePT(inner, inner, identifier='spt_mapped')
        actual_self = self.spt.with_mapped_subtemplates(map_fn=lambda x: inner, recursion_strategy='self', identifier_map=identifier_map)
        self.assertEqual(expected_self, actual_self)


class AtomicPulseTemplateTests(unittest.TestCase):

    def test_internal_create_program(self) -> None:
        measurement_windows = [('M', 0, 5)]
        single_wf = DummyWaveform(duration=6, defined_channels={'A'})
        wf = MultiChannelWaveform([single_wf])

        template = AtomicPulseTemplateStub(measurements=measurement_windows, parameter_names={'foo'})
        scope = DictScope.from_kwargs(foo=7.2, volatile={'gutes_zeuch'})
        measurement_mapping = {'M': 'N'}
        channel_mapping = {'B': 'A'}
        program_builder = LoopBuilder()

        expected_program = Loop(children=[Loop(waveform=wf)],
                                measurements=[('N', 0, 5)])

        with mock.patch.object(template, 'build_waveform', return_value=wf) as build_waveform:
            template._internal_create_program(scope=scope,
                                              measurement_mapping=measurement_mapping,
                                              channel_mapping=channel_mapping,
                                              program_builder=program_builder,
                                              to_single_waveform=set(),
                                              global_transformation=None)
            build_waveform.assert_called_once_with(parameters=scope, channel_mapping=channel_mapping)
        program = program_builder.to_program()
        self.assertEqual(expected_program, program)

    def test_internal_create_program_transformation(self):
        inner_wf = DummyWaveform()
        template = AtomicPulseTemplateStub(parameter_names=set())
        program_builder = LoopBuilder()
        global_transformation = TransformationStub()
        scope = DictScope.from_kwargs()
        expected_program = Loop(children=[Loop(waveform=TransformingWaveform(inner_wf, global_transformation))])

        with mock.patch.object(template, 'build_waveform', return_value=inner_wf):
            template._internal_create_program(scope=scope,
                                              measurement_mapping={},
                                              channel_mapping={},
                                              program_builder=program_builder,
                                              to_single_waveform=set(),
                                              global_transformation=global_transformation)
        program = program_builder.to_program()
        self.assertEqual(expected_program, program)

    def test_internal_create_program_no_waveform(self) -> None:
        measurement_windows = [('M', 0, 5)]

        template = AtomicPulseTemplateStub(measurements=measurement_windows, parameter_names={'foo'})
        scope = DictScope.from_kwargs(foo=3.5, bar=3, volatile={'bar'})
        measurement_mapping = {'M': 'N'}
        channel_mapping = {'B': 'A'}
        program_builder = LoopBuilder()

        expected_program = Loop()

        with mock.patch.object(template, 'build_waveform', return_value=None) as build_waveform:
            with mock.patch.object(template,
                                   'get_measurement_windows',
                                   wraps=template.get_measurement_windows) as get_meas_windows:
                template._internal_create_program(scope=scope,
                                                  measurement_mapping=measurement_mapping,
                                                  channel_mapping=channel_mapping,
                                                  program_builder=program_builder,
                                                  to_single_waveform=set(),
                                                  global_transformation=None)
                build_waveform.assert_called_once_with(parameters=scope, channel_mapping=channel_mapping)
                get_meas_windows.assert_not_called()

        self.assertIsNone(program_builder.to_program())

    def test_internal_create_program_volatile(self):
        template = AtomicPulseTemplateStub(parameter_names={'foo'})
        scope = DictScope.from_kwargs(foo=3.5, bar=3, volatile={'foo'})
        measurement_mapping = {'M': 'N'}
        channel_mapping = {'B': 'A'}

        program_builder = LoopBuilder()

        with self.assertRaisesRegex(AssertionError, "volatile"):
            template._internal_create_program(scope=scope,
                                              measurement_mapping=measurement_mapping,
                                              channel_mapping=channel_mapping,
                                              program_builder=program_builder,
                                              to_single_waveform=set(),
                                              global_transformation=None)
        self.assertIsNone(program_builder.to_program())
