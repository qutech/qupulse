import unittest
import warnings

import numpy

from qupulse.utils.types import time_from_float
from qupulse.pulses.multi_channel_pulse_template import MultiChannelWaveform, MappingPulseTemplate, ChannelMappingException, AtomicMultiChannelPulseTemplate
from qupulse.pulses.parameters import ParameterConstraint, ParameterConstraintViolation, ConstantParameter
from qupulse.expressions import ExpressionScalar, Expression
from qupulse._program.instructions import InstructionBlock

from tests.pulses.sequencing_dummies import DummyPulseTemplate, DummyWaveform
from tests.serialization_dummies import DummySerializer
from tests.pulses.pulse_template_tests import PulseTemplateStub
from tests.serialization_tests import SerializableTests


class AtomicMultiChannelPulseTemplateTest(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.subtemplates = [DummyPulseTemplate(parameter_names={'p1'},
                                                measurement_names={'m1'},
                                                defined_channels={'c1'}),
                             DummyPulseTemplate(parameter_names={'p2'},
                                                measurement_names={'m2'},
                                                defined_channels={'c2'}),
                             DummyPulseTemplate(parameter_names={'p3'},
                                                measurement_names={'m3'},
                                                defined_channels={'c3'})]
        self.no_param_maps = [{'p1': '1'}, {'p2': '2'}, {'p3': '3'}]
        self.param_maps = [{'p1': 'pp1'}, {'p2': 'pp2'}, {'p3': 'pp3'}]
        self.chan_maps = [{'c1': 'cc1'}, {'c2': 'cc2'}, {'c3': 'cc3'}]

    def test_init_empty(self) -> None:
        with self.assertRaises(ValueError):
            AtomicMultiChannelPulseTemplate()

        with self.assertRaises(ValueError):
            AtomicMultiChannelPulseTemplate(identifier='foo')

        with self.assertRaises(ValueError):
            AtomicMultiChannelPulseTemplate()

        with self.assertRaises(ValueError):
            AtomicMultiChannelPulseTemplate(identifier='foo')

        with self.assertRaises(ValueError):
            AtomicMultiChannelPulseTemplate(identifier='foo', parameter_constraints=[])

    def test_non_atomic_subtemplates(self):
        non_atomic_pt = PulseTemplateStub(duration='t1', defined_channels={'A'}, parameter_names=set())
        atomic_pt = DummyPulseTemplate(defined_channels={'B'}, duration='t1')
        non_atomic_mapping = MappingPulseTemplate(non_atomic_pt)

        with self.assertRaises(TypeError):
            AtomicMultiChannelPulseTemplate(non_atomic_pt)

        with self.assertRaises(TypeError):
            AtomicMultiChannelPulseTemplate(non_atomic_pt, atomic_pt)

        with self.assertRaises(TypeError):
            AtomicMultiChannelPulseTemplate(non_atomic_mapping, atomic_pt)

        with self.assertRaises(TypeError):
            AtomicMultiChannelPulseTemplate((non_atomic_pt, {'A': 'C'}), atomic_pt)

    def test_instantiation_duration_check(self):
        subtemplates = [DummyPulseTemplate(parameter_names={'p1'},
                                           measurement_names={'m1'},
                                           defined_channels={'c1'},
                                           duration='t_1',
                                           waveform=DummyWaveform(duration=3, defined_channels={'c1'})),
                        DummyPulseTemplate(parameter_names={'p2'},
                                           measurement_names={'m2'},
                                           defined_channels={'c2'},
                                           duration='t_2',
                                           waveform=DummyWaveform(duration=3, defined_channels={'c2'})),
                        DummyPulseTemplate(parameter_names={'p3'},
                                           measurement_names={'m3'},
                                           defined_channels={'c3'},
                                           duration='t_3',
                                           waveform=DummyWaveform(duration=4, defined_channels={'c3'}))]

        with self.assertRaisesRegex(ValueError, 'duration equality'):
            AtomicMultiChannelPulseTemplate(*subtemplates)

        amcpt = AtomicMultiChannelPulseTemplate(*subtemplates, duration=True)
        self.assertIs(amcpt.duration, subtemplates[0].duration)

        with self.assertRaisesRegex(ValueError, 'duration'):
            amcpt.build_waveform(parameters=dict(t_1=3, t_2=3, t_3=3),
                                 channel_mapping={ch: ch for ch in 'c1 c2 c3'.split()})

        subtemplates[2].waveform = None
        amcpt.build_waveform(parameters=dict(t_1=3, t_2=3, t_3=3),
                             channel_mapping={ch: ch for ch in 'c1 c2 c3'.split()})

        amcpt = AtomicMultiChannelPulseTemplate(*subtemplates, duration='t_0')
        with self.assertRaisesRegex(ValueError, 'duration'):
            amcpt.build_waveform(parameters=dict(t_1=3, t_2=3, t_3=3, t_0=4),
                                 channel_mapping={ch: ch for ch in 'c1 c2 c3'.split()})
        with self.assertRaisesRegex(ValueError, 'duration'):
            amcpt.build_waveform(parameters=dict(t_1=3+1e-9, t_2=3, t_3=3, t_0=4),
                                 channel_mapping={ch: ch for ch in 'c1 c2 c3'.split()})
        amcpt.build_waveform(parameters=dict(t_1=3, t_2=3, t_3=3, t_0=3),
                             channel_mapping={ch: ch for ch in 'c1 c2 c3'.split()})
        amcpt.build_waveform(parameters=dict(t_1=3+1e-11, t_2=3, t_3=3, t_0=3),
                             channel_mapping={ch: ch for ch in 'c1 c2 c3'.split()})

    def test_external_parameters_warning(self):
        with self.assertWarnsRegex(DeprecationWarning, "external_parameters",
                                   msg="AtomicMultiChannelPulseTemplate did not issue a warning for argument external_parameters"):
            AtomicMultiChannelPulseTemplate(DummyPulseTemplate(), external_parameters={'a'})

    def test_duration(self):
        sts = [DummyPulseTemplate(duration='t1', defined_channels={'A'}),
               DummyPulseTemplate(duration='t1', defined_channels={'B'}),
               DummyPulseTemplate(duration='t2', defined_channels={'C'})]
        with self.assertRaises(ValueError):
            AtomicMultiChannelPulseTemplate(*sts)

        with self.assertRaises(ValueError):
            AtomicMultiChannelPulseTemplate(sts[0], sts[2])
        template = AtomicMultiChannelPulseTemplate(*sts[:1])

        self.assertEqual(template.duration, 't1')

    def test_mapping_template_pure_conversion(self):
        template = AtomicMultiChannelPulseTemplate(*zip(self.subtemplates, self.param_maps, self.chan_maps))

        for st, pm, cm in zip(template.subtemplates, self.param_maps, self.chan_maps):
            self.assertEqual(st.parameter_names, set(pm.values()))
            self.assertEqual(st.defined_channels, set(cm.values()))

    def test_mapping_template_mixed_conversion(self):
        subtemp_args = [
            (self.subtemplates[0], self.param_maps[0], self.chan_maps[0]),
            MappingPulseTemplate(self.subtemplates[1], parameter_mapping=self.param_maps[1], channel_mapping=self.chan_maps[1]),
            (self.subtemplates[2], self.param_maps[2], self.chan_maps[2])
        ]
        template = AtomicMultiChannelPulseTemplate(*subtemp_args)

        for st, pm, cm in zip(template.subtemplates, self.param_maps, self.chan_maps):
            self.assertEqual(st.parameter_names, set(pm.values()))
            self.assertEqual(st.defined_channels, set(cm.values()))

    def test_channel_intersection(self):
        chan_maps = self.chan_maps.copy()
        chan_maps[-1]['c3'] = 'cc1'
        with self.assertRaises(ChannelMappingException):
            AtomicMultiChannelPulseTemplate(*zip(self.subtemplates, self.param_maps, chan_maps))

    def test_defined_channels(self):
        subtemp_args = [*zip(self.subtemplates, self.param_maps, self.chan_maps)]
        template = AtomicMultiChannelPulseTemplate(*subtemp_args)
        self.assertEqual(template.defined_channels, {'cc1', 'cc2', 'cc3'})

    def test_measurement_names(self):
        sts = [DummyPulseTemplate(duration='t1', defined_channels={'A'}, parameter_names={'a', 'b'}, measurement_names={'A', 'C'}),
               DummyPulseTemplate(duration='t1', defined_channels={'B'}, parameter_names={'a', 'c'}, measurement_names={'A', 'B'})]

        self.assertEqual(AtomicMultiChannelPulseTemplate(*sts, measurements=[('D', 1, 2)]).measurement_names,
                         {'A', 'B', 'C', 'D'})

    def test_parameter_names(self):
        sts = [DummyPulseTemplate(duration='t1', defined_channels={'A'}, parameter_names={'a', 'b'},
                                  measurement_names={'A', 'C'}),
               DummyPulseTemplate(duration='t1', defined_channels={'B'}, parameter_names={'a', 'c'},
                                  measurement_names={'A', 'B'})]
        pt = AtomicMultiChannelPulseTemplate(*sts, measurements=[('D', 'd', 2)], parameter_constraints=['d < e'])

        self.assertEqual(pt.parameter_names,
                         {'a', 'b', 'c', 'd', 'e'})

    def test_parameter_names_2(self):
        template = AtomicMultiChannelPulseTemplate(*zip(self.subtemplates, self.param_maps, self.chan_maps),
                                                   parameter_constraints={'pp1 > hugo'},
                                                   measurements={('meas', 'd', 1)},
                                                   duration='my_duration')
        self.assertEqual({'pp1', 'pp2', 'pp3', 'hugo', 'd', 'my_duration'}, template.parameter_names)

    def test_integral(self) -> None:
        sts = [DummyPulseTemplate(duration='t1', defined_channels={'A'},
                                  integrals={'A': ExpressionScalar('2+k')}),
               DummyPulseTemplate(duration='t1', defined_channels={'B', 'C'},
                                  integrals={'B': ExpressionScalar('t1-t0*3.1'), 'C': ExpressionScalar('l')})]
        pulse = AtomicMultiChannelPulseTemplate(*sts)
        self.assertEqual({'A': ExpressionScalar('2+k'),
                          'B': ExpressionScalar('t1-t0*3.1'),
                          'C': ExpressionScalar('l')},
                         pulse.integral)


class MultiChannelPulseTemplateSequencingTests(unittest.TestCase):

    def test_requires_stop(self):
        sts = [DummyPulseTemplate(duration='t1', defined_channels={'A'}, parameter_names={'a', 'b'}, requires_stop=False),
               DummyPulseTemplate(duration='t1', defined_channels={'B'}, parameter_names={'a', 'c'}, requires_stop=False)]

        self.assertFalse(AtomicMultiChannelPulseTemplate(*sts).requires_stop(dict(), dict()))
        sts = [
            DummyPulseTemplate(duration='t1', defined_channels={'A'}, parameter_names={'a', 'b'}, requires_stop=False),
            DummyPulseTemplate(duration='t1', defined_channels={'B'}, parameter_names={'a', 'c'}, requires_stop=True)]

        self.assertTrue(AtomicMultiChannelPulseTemplate(*sts).requires_stop(dict(), dict()))

    def test_build_waveform(self):
        wfs = [DummyWaveform(duration=1.1, defined_channels={'A'}), DummyWaveform(duration=1.1, defined_channels={'B'})]

        sts = [DummyPulseTemplate(duration='t1', defined_channels={'A'}, parameter_names={'a', 'b'},
                                  measurement_names={'A', 'C'}, waveform=wfs[0]),
               DummyPulseTemplate(duration='t1', defined_channels={'B'}, parameter_names={'a', 'c'},
                                  measurement_names={'A', 'B'}, waveform=wfs[1])]

        pt = AtomicMultiChannelPulseTemplate(*sts, parameter_constraints=['a < b'])

        parameters = dict(a=2.2, b = 1.1, c=3.3)
        channel_mapping = dict()
        with self.assertRaises(ParameterConstraintViolation):
            pt.build_waveform(parameters, channel_mapping=dict())

        parameters['a'] = 0.5
        wf = pt.build_waveform(parameters, channel_mapping=channel_mapping)
        self.assertEqual(wf['A'], wfs[0])
        self.assertEqual(wf['B'], wfs[1])

        for st in sts:
            self.assertEqual(st.build_waveform_calls, [(parameters, channel_mapping)])
            self.assertIs(parameters, st.build_waveform_calls[0][0])
            self.assertIs(channel_mapping, st.build_waveform_calls[0][1])

    def test_build_waveform_none(self):
        wfs = [DummyWaveform(duration=1.1, defined_channels={'A'}), DummyWaveform(duration=1.1, defined_channels={'B'})]

        sts = [DummyPulseTemplate(duration='t1', defined_channels={'A'}, waveform=wfs[0]),
               DummyPulseTemplate(duration='t1', defined_channels={'B'}, waveform=wfs[1]),
               DummyPulseTemplate(duration='t1', defined_channels={'C'}, waveform=None)]

        pt = AtomicMultiChannelPulseTemplate(*sts, parameter_constraints=['a < b'])

        parameters = dict(a=2.2, b=1.1, c=3.3)
        channel_mapping = dict(A=6)
        with self.assertRaises(ParameterConstraintViolation):
            # parameter constraints are checked before channel mapping is applied
            pt.build_waveform(parameters, channel_mapping=dict())

        parameters['a'] = 0.5
        wf = pt.build_waveform(parameters, channel_mapping=channel_mapping)
        self.assertIs(wf['A'], wfs[0])
        self.assertIs(wf['B'], wfs[1])

        sts[1].waveform = None
        wf = pt.build_waveform(parameters, channel_mapping=channel_mapping)
        self.assertIs(wf, wfs[0])

        sts[0].waveform = None
        wf = pt.build_waveform(parameters, channel_mapping=channel_mapping)
        self.assertIsNone(wf)

    def test_build_sequence(self):
        wfs = [DummyWaveform(duration=1.1, defined_channels={'A'}), DummyWaveform(duration=1.1, defined_channels={'B'})]
        sts = [DummyPulseTemplate(duration='t1', defined_channels={'A'}, waveform=wfs[0], measurements=[('m', 0, 1)]),
               DummyPulseTemplate(duration='t1', defined_channels={'B'}, waveform=wfs[1]),
               DummyPulseTemplate(duration='t1', defined_channels={'C'}, waveform=None)]

        pt = AtomicMultiChannelPulseTemplate(*sts, parameter_constraints=['a < b'], measurements=[('n', .1, .2)])

        params = dict(a=ConstantParameter(1.0), b=ConstantParameter(1.1))
        measurement_mapping = dict(m='foo', n='bar')
        channel_mapping = {'A': 'A', 'B': 'B', 'C': None}

        block = InstructionBlock()
        pt.build_sequence(None, parameters=params, conditions={}, measurement_mapping=measurement_mapping,
                          channel_mapping=channel_mapping, instruction_block=block)

        expected_waveform = MultiChannelWaveform(wfs)

        expected_block = InstructionBlock()
        measurements = [('bar', .1, .2), ('foo', 0, 1)]
        expected_block.add_instruction_meas(measurements)
        expected_block.add_instruction_exec(waveform=expected_waveform)

        self.assertEqual(len(block.instructions), len(expected_block.instructions))
        self.assertEqual(block.instructions[0].compare_key, expected_block.instructions[0].compare_key)
        self.assertEqual(block.instructions[1].compare_key, expected_block.instructions[1].compare_key)

    def test_get_measurement_windows(self):
        wfs = [DummyWaveform(duration=1.1, defined_channels={'A'}), DummyWaveform(duration=1.1, defined_channels={'B'})]
        sts = [DummyPulseTemplate(duration='t1', defined_channels={'A'}, waveform=wfs[0], measurements=[('m', 0, 1),
                                                                                                        ('n', 0.3, 0.4)]),
               DummyPulseTemplate(duration='t1', defined_channels={'B'}, waveform=wfs[1], measurements=[('m', 0.1, .2)])]

        pt = AtomicMultiChannelPulseTemplate(*sts, parameter_constraints=['a < b'], measurements=[('n', .1, .2)])

        measurement_mapping = dict(m='foo', n='bar')
        expected = [('bar', .1, .2), ('foo', 0, 1), ('bar', .3, .4), ('foo', .1, .2)]
        meas_windows = pt.get_measurement_windows({}, measurement_mapping)
        self.assertEqual(expected, meas_windows)


class AtomicMultiChannelPulseTemplateSerializationTests(SerializableTests, unittest.TestCase):

    @property
    def class_to_test(self):
        return AtomicMultiChannelPulseTemplate

    def make_kwargs(self):
        return {
            'subtemplates': [DummyPulseTemplate(duration='t1', defined_channels={'A'}, parameter_names={'a', 'b'}),
                             DummyPulseTemplate(duration='t1', defined_channels={'B'}, parameter_names={'a', 'c'})],
            'parameter_constraints': [str(ParameterConstraint('ilse>2')), str(ParameterConstraint('k>foo'))]
        }

    def make_instance(self, identifier=None, registry=None):
        kwargs = self.make_kwargs()
        subtemplates = kwargs['subtemplates']
        del kwargs['subtemplates']
        return self.class_to_test(identifier=identifier, *subtemplates, **kwargs, registry=registry)

    def assert_equal_instance_except_id(self, lhs: AtomicMultiChannelPulseTemplate, rhs: AtomicMultiChannelPulseTemplate):
        self.assertIsInstance(lhs, AtomicMultiChannelPulseTemplate)
        self.assertIsInstance(rhs, AtomicMultiChannelPulseTemplate)
        self.assertEqual(lhs.subtemplates, rhs.subtemplates)
        self.assertEqual(lhs.parameter_constraints, rhs.parameter_constraints)


class AtomicMultiChannelPulseTemplateOldSerializationTests(unittest.TestCase):

    def test_deserialize_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="AtomicMultiChannelPT does not issue warning for old serialization routines."):
            sts = [DummyPulseTemplate(duration='t1', defined_channels={'A'}, parameter_names={'a', 'b'}),
                   DummyPulseTemplate(duration='t1', defined_channels={'B'}, parameter_names={'a', 'c'})]

            def deserialization_callback(ident: str):
                self.assertIn(ident, ('0', '1'))

                if ident == '0':
                    return 0
                else:
                    return 1

            serializer = DummySerializer(deserialize_callback=deserialization_callback)
            serializer.subelements = sts

            data = dict(subtemplates=['0', '1'], parameter_constraints=['a < d'])

            template = AtomicMultiChannelPulseTemplate.deserialize(serializer, **data)

            self.assertIs(template.subtemplates[0], sts[0])
            self.assertIs(template.subtemplates[1], sts[1])
            self.assertEqual(template.parameter_constraints, [ParameterConstraint('a < d')])

    def test_serialize_old(self) -> None:
        # test for deprecated version during transition period, remove after final switch
        with self.assertWarnsRegex(DeprecationWarning, "deprecated",
                                   msg="AtomicMultiChannelPT does not issue warning for old serialization routines."):
            sts = [DummyPulseTemplate(duration='t1', defined_channels={'A'}, parameter_names={'a', 'b'}),
                   DummyPulseTemplate(duration='t1', defined_channels={'B'}, parameter_names={'a', 'c'})]
            constraints = ['a < d']
            template = AtomicMultiChannelPulseTemplate(*sts,
                                                       parameter_constraints=constraints)

            expected_data = dict(subtemplates=['0', '1'], parameter_constraints=['a < d'])

            def serialize_callback(obj) -> str:
                self.assertIn(obj, sts)
                return str(sts.index(obj))

            serializer = DummySerializer(serialize_callback=serialize_callback,
                                         identifier_callback=serialize_callback)

            data = template.get_serialization_data(serializer=serializer)

            self.assertEqual(expected_data, data)
