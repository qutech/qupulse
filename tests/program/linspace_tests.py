import copy
import unittest
from unittest import TestCase

from qupulse.pulses import *
from qupulse.program.linspace import *
from qupulse.program.transformation import *
from qupulse.pulses.function_pulse_template import FunctionPulseTemplate


def assert_vm_output_almost_equal(test: TestCase, expected, actual):
    """Compare two vm outputs with default TestCase.assertAlmostEqual accuracy"""
    test.assertEqual(len(expected), len(actual))
    for idx, ((t_e, vals_e), (t_a, vals_a)) in enumerate(zip(expected, actual)):
        test.assertEqual(t_e, t_a, f"Differing times in {idx}th element")
        test.assertEqual(len(vals_e), len(vals_a), f"Differing channel count in {idx} element")
        for ch, (val_e, val_a) in enumerate(zip(vals_e, vals_a)):
            test.assertAlmostEqual(val_e, val_a, msg=f"Differing values in {idx} of {len(expected)} element channel {ch}")


class SingleRampTest(TestCase):
    def setUp(self):
        hold = ConstantPT(10 ** 6, {'a': '-1. + idx * 0.01'})
        self.pulse_template = hold.with_iteration('idx', 200)

        self.program = LinSpaceIter(
            length=200,
            body=(LinSpaceHold(
                bases=(-1.,),
                factors=((0.01,),),
                duration_base=TimeType(10**6),
                duration_factors=None
            ),)
        )

        key = DepKey.from_voltages((0.01,), DEFAULT_INCREMENT_RESOLUTION)

        self.commands = [
            Set(0, -1.0, key),
            Wait(TimeType(10 ** 6)),
            LoopLabel(0, 199),
            Increment(0, 0.01, key),
            Wait(TimeType(10 ** 6)),
            LoopJmp(0)
        ]

        self.output = [
            (TimeType(10**6 * idx), [sum([-1.0] + [0.01] * idx)]) for idx in range(200)
        ]

    def test_program(self):
        program_builder = LinSpaceBuilder(('a',))
        program = self.pulse_template.create_program(program_builder=program_builder)
        self.assertEqual([self.program], program)

    def test_commands(self):
        commands = to_increment_commands([self.program])
        self.assertEqual(self.commands, commands)

    def test_output(self):
        vm = LinSpaceVM(1)
        vm.set_commands(commands=self.commands)
        vm.run()
        assert_vm_output_almost_equal(self, self.output, vm.history)


class SequencedRepetitionTest(TestCase):
    def setUp(self):
        
        base_time = 1e2
        rep_factor = 2
          
        wait = AtomicMultiChannelPT(
            ConstantPT(f'{base_time}', {'a': '-1. + idx_a * 0.01', }),
            ConstantPT(f'{base_time}', {'b': '-0.5 + idx_b * 0.05'})
            )
        
        dependent_constant = AtomicMultiChannelPT(
            ConstantPT(base_time, {'a': '-1.0 '}),
            ConstantPT(base_time, {'b': '-0.5 + idx_b*0.05',}),            
            )
        
        dependent_constant2 = AtomicMultiChannelPT(
            ConstantPT(base_time, {'a': '-0.5 '}),
            ConstantPT(base_time, {'b': '-0.3 + idx_b*0.05',}),            
            )
        
        #not working
        self.pulse_template = (
                dependent_constant @
                dependent_constant2.with_repetition(rep_factor) @
                wait.with_iteration('idx_a', rep_factor)
        ).with_iteration('idx_b', rep_factor)

        wait_hold = LinSpaceHold(
            bases=(-1.0, -0.5),
            factors=((0.0, 0.01), (0.05, 0.0),),
            duration_base=TimeType.from_float(base_time),
            duration_factors=None
        )
        dependent_hold_1 = LinSpaceHold(
            bases=(-1.0, -0.5),
            factors=(None, (0.05,),),
            duration_base=TimeType.from_float(base_time),
            duration_factors=None
        )
        dependent_hold_2 = LinSpaceHold(
            bases=(-0.5, -0.3),
            factors=(None, (0.05,),),
            duration_base=TimeType.from_float(base_time),
            duration_factors=None
        )

        self.program = LinSpaceIter(
             length=rep_factor,
             body=(
                 dependent_hold_1,
                 LinSpaceRepeat(body=(dependent_hold_2,), count=rep_factor),
                 LinSpaceIter(body=(wait_hold,), length=rep_factor),
             )
        )

        self.commands = [
            Set(channel=0, value=-1.0, key=DepKey(factors=())),
            Set(channel=1, value=-0.5, key=DepKey(factors=(50000000,))),
            Wait(duration=TimeType(100, 1)),

            Set(channel=0, value=-0.5, key=DepKey(factors=())),
            Increment(channel=1, value=0.2, dependency_key=DepKey(factors=(50000000,))),
            Wait(duration=TimeType(100, 1)),

            # This is the repetition
            LoopLabel(idx=0, count=1),
                Wait(duration=TimeType(100, 1)),
            LoopJmp(idx=0),

            Set(channel=0, value=-1.0, key=DepKey(factors=(0, 10000000))),
            Increment(channel=1, value=-0.2, dependency_key=DepKey(factors=(50000000,))),
            Wait(duration=TimeType(100, 1)),

            LoopLabel(idx=1, count=1),
                Increment(channel=0, value=0.01, dependency_key=DepKey(factors=(0, 10000000))),
                Wait(duration=TimeType(100, 1)),
            LoopJmp(idx=1),

            LoopLabel(idx=2, count=1),
                Set(channel=0, value=-1.0, key=DepKey(factors=())),
                Increment(channel=1, value=0.05, dependency_key=DepKey(factors=(50000000,))),
                Wait(duration=TimeType(100, 1)),
                Set(channel=0, value=-0.5, key=DepKey(factors=())),
                Increment(channel=1, value=0.2, dependency_key=DepKey(factors=(50000000,))),
                Wait(duration=TimeType(100, 1)),

                # next repetition
                LoopLabel(idx=3, count=1),
                    Wait(duration=TimeType(100, 1)),
                LoopJmp(idx=3),

                Increment(channel=0,
                   value=-0.01,
                   dependency_key=DepKey(factors=(0, 10000000))),
                Increment(channel=1, value=-0.2, dependency_key=DepKey(factors=(50000000,))),
                Wait(duration=TimeType(100, 1)),

                LoopLabel(idx=4, count=1),
                    Increment(channel=0, value=0.01, dependency_key=DepKey(factors=(0, 10000000))),
                    Wait(duration=TimeType(100, 1)),
                LoopJmp(idx=4),
            LoopJmp(idx=2)]

        time = TimeType(0)
        self.output = []
        for idx_b in range(rep_factor):
            # does not account yet for floating poit errors. We would need to sum here
            self.output.append((time, (-1.0, -0.5 + idx_b * 0.05)))
            time += TimeType.from_float(base_time)

            for _ in range(rep_factor):
                self.output.append((time, (-0.5, -0.3 + idx_b * 0.05)))
                time += TimeType.from_float(base_time)

            for idx_a in range(rep_factor):
                self.output.append((time, (-1.0 + 0.01 * idx_a, -0.5 + idx_b * 0.05)))
                time += TimeType.from_float(base_time)

    def test_program_1(self):
        program_builder = LinSpaceBuilder(('a','b'))
        program_1 = self.pulse_template.create_program(program_builder=program_builder)
        self.assertEqual([self.program], program_1)

    def test_commands_1(self):
        commands = to_increment_commands([self.program])
        self.assertEqual(self.commands, commands)

    def test_output_1(self):
        vm = LinSpaceVM(2)
        vm.set_commands(commands=self.commands)
        vm.run()
        assert_vm_output_almost_equal(self, self.output, vm.history)


class PlainCSDTest(TestCase):
    def setUp(self):
        hold = ConstantPT(10**6, {'a': '-1. + idx_a * 0.01', 'b': '-.5 + idx_b * 0.02'})
        scan_a = hold.with_iteration('idx_a', 200)
        self.pulse_template = scan_a.with_iteration('idx_b', 100)

        self.program = LinSpaceIter(length=100, body=(LinSpaceIter(
            length=200,
            body=(LinSpaceHold(
                bases=(-1., -0.5),
                factors=((0.0, 0.01),
                         (0.02, 0.0)),
                duration_base=TimeType(10**6),
                duration_factors=None
            ),)
        ),))

        key_0 = DepKey.from_voltages((0, 0.01,), DEFAULT_INCREMENT_RESOLUTION)
        key_1 = DepKey.from_voltages((0.02,), DEFAULT_INCREMENT_RESOLUTION)

        self.commands = [
            Set(0, -1.0, key_0),
            Set(1, -0.5, key_1),
            Wait(TimeType(10 ** 6)),

            LoopLabel(0, 199),
            Increment(0, 0.01, key_0),
            Wait(TimeType(10 ** 6)),
            LoopJmp(0),

            LoopLabel(1, 99),

            Increment(0, -1.99, key_0),
            Increment(1, 0.02, key_1),
            Wait(TimeType(10 ** 6)),

            LoopLabel(2, 199),
            Increment(0, 0.01, key_0),
            Wait(TimeType(10 ** 6)),
            LoopJmp(2),

            LoopJmp(1),
        ]

        a_values = [sum([-1.] + [0.01] * i) for i in range(200)]
        b_values = [sum([-.5] + [0.02] * j) for j in range(100)]

        self.output = [
            (
                TimeType(10 ** 6 * (i + 200 * j)),
                [a_values[i], b_values[j]]
            ) for j in range(100) for i in range(200)
        ]

    def test_program(self):
        program_builder = LinSpaceBuilder(('a', 'b'))
        program = self.pulse_template.create_program(program_builder=program_builder)
        self.assertEqual([self.program], program)

    def test_increment_commands(self):
        commands = to_increment_commands([self.program])
        self.assertEqual(self.commands, commands)

    def test_output(self):
        vm = LinSpaceVM(2)
        vm.set_commands(self.commands)
        vm.run()
        assert_vm_output_almost_equal(self, self.output, vm.history)


class TiltedCSDTest(TestCase):
    def setUp(self):
        repetition_count = 3
        hold = ConstantPT(10**6, {'a': '-1. + idx_a * 0.01 + idx_b * 1e-3', 'b': '-.5 + idx_b * 0.02 - 3e-3 * idx_a'})
        scan_a = hold.with_iteration('idx_a', 200)
        self.pulse_template = scan_a.with_iteration('idx_b', 100)
        self.repeated_pt = self.pulse_template.with_repetition(repetition_count)

        self.program = LinSpaceIter(length=100, body=(LinSpaceIter(
            length=200,
            body=(LinSpaceHold(
                bases=(-1., -0.5),
                factors=((1e-3, 0.01),
                         (0.02, -3e-3)),
                duration_base=TimeType(10**6),
                duration_factors=None
            ),)
        ),))
        self.repeated_program = LinSpaceRepeat(body=(self.program,), count=repetition_count)

        key_0 = DepKey.from_voltages((1e-3, 0.01,), DEFAULT_INCREMENT_RESOLUTION)
        key_1 = DepKey.from_voltages((0.02, -3e-3), DEFAULT_INCREMENT_RESOLUTION)

        self.commands = [
            Set(0, -1.0, key_0),
            Set(1, -0.5, key_1),
            Wait(TimeType(10 ** 6)),

            LoopLabel(0, 199),
            Increment(0, 0.01, key_0),
            Increment(1, -3e-3, key_1),
            Wait(TimeType(10 ** 6)),
            LoopJmp(0),

            LoopLabel(1, 99),

            Increment(0, 1e-3 + -199 * 1e-2, key_0),
            Increment(1, 0.02 + -199 * -3e-3, key_1),
            Wait(TimeType(10 ** 6)),

            LoopLabel(2, 199),
            Increment(0, 0.01, key_0),
            Increment(1, -3e-3, key_1),
            Wait(TimeType(10 ** 6)),
            LoopJmp(2),

            LoopJmp(1),
        ]
        inner_commands = copy.deepcopy(self.commands)
        for cmd in inner_commands:
            if hasattr(cmd, 'idx'):
                cmd.idx += 1
        self.repeated_commands = [LoopLabel(0, repetition_count)] + inner_commands + [LoopJmp(0)]

        self.output = [
            (
                TimeType(10 ** 6 * (i + 200 * j)),
                [-1. + i * 0.01 + j * 1e-3, -.5 + j * 0.02 - 3e-3 * i]
            ) for j in range(100) for i in range(200)
        ]
        self.repeated_output = [
            (t + TimeType(10**6) * (n * 100 * 200), vals)
            for n in range(repetition_count)
            for t, vals in self.output
        ]

    def test_program(self):
        program_builder = LinSpaceBuilder(('a', 'b'))
        program = self.pulse_template.create_program(program_builder=program_builder)
        self.assertEqual([self.program], program)

    def test_repeated_program(self):
        program_builder = LinSpaceBuilder(('a', 'b'))
        program = self.repeated_pt.create_program(program_builder=program_builder)
        self.assertEqual([self.repeated_program], program)

    def test_increment_commands(self):
        commands = to_increment_commands([self.program])
        self.assertEqual(self.commands, commands)

    def test_repeated_increment_commands(self):
        commands = to_increment_commands([self.repeated_program])
        self.assertEqual(self.repeated_commands, commands)

    def test_output(self):
        vm = LinSpaceVM(2)
        vm.set_commands(self.commands)
        vm.run()
        assert_vm_output_almost_equal(self, self.output, vm.history)

    def test_repeated_output(self):
        vm = LinSpaceVM(2)
        vm.set_commands(self.repeated_commands)
        vm.run()
        assert_vm_output_almost_equal(self, self.repeated_output, vm.history)


class SingletLoadProcessing(TestCase):
    def setUp(self):
        wait = ConstantPT(10 ** 6, {'a': '-1. + idx_a * 0.01', 'b': '-.5 + idx_b * 0.02'})
        load_random = ConstantPT(10 ** 5, {'a': -.4, 'b': -.3})
        meas = ConstantPT(10 ** 5, {'a': 0.05, 'b': 0.06})

        singlet_scan = (load_random @ wait @ meas).with_iteration('idx_a', 200).with_iteration('idx_b', 100)
        self.pulse_template = singlet_scan

        self.program = LinSpaceIter(length=100, body=(LinSpaceIter(
            length=200,
            body=(
                LinSpaceHold(bases=(-0.4, -0.3), factors=(None, None), duration_base=TimeType(10 ** 5),
                             duration_factors=None),
                LinSpaceHold(bases=(-1., -0.5),
                             factors=((0.0, 0.01),
                                      (0.02, 0.0)),
                             duration_base=TimeType(10 ** 6),
                             duration_factors=None),
                LinSpaceHold(bases=(0.05, 0.06), factors=(None, None), duration_base=TimeType(10 ** 5),
                             duration_factors=None),
            )
        ),))

        key_0 = DepKey.from_voltages((0, 0.01,), DEFAULT_INCREMENT_RESOLUTION)
        key_1 = DepKey.from_voltages((0.02,), DEFAULT_INCREMENT_RESOLUTION)

        self.commands = [
            Set(0, -0.4),
            Set(1, -0.3),
            Wait(TimeType(10 ** 5)),
            Set(0, -1.0, key_0),
            Set(1, -0.5, key_1),
            Wait(TimeType(10 ** 6)),
            Set(0, 0.05),
            Set(1, 0.06),
            Wait(TimeType(10 ** 5)),

            LoopLabel(0, 199),
            Set(0, -0.4),
            Set(1, -0.3),
            Wait(TimeType(10 ** 5)),
            Increment(0, 0.01, key_0),
            Increment(1, 0.00, key_1),
            Wait(TimeType(10 ** 6)),
            Set(0, 0.05),
            Set(1, 0.06),
            Wait(TimeType(10 ** 5)),
            LoopJmp(0),

            LoopLabel(1, 99),

            Set(0, -0.4),
            Set(1, -0.3),
            Wait(TimeType(10 ** 5)),
            Increment(0, -1.99, key_0),
            Increment(1, 0.02, key_1),
            Wait(TimeType(10 ** 6)),
            Set(0, 0.05),
            Set(1, 0.06),
            Wait(TimeType(10 ** 5)),

            LoopLabel(2, 199),

            Set(0, -0.4),
            Set(1, -0.3),
            Wait(TimeType(10 ** 5)),
            Increment(0, 0.01, key_0),
            Increment(1, 0.00, key_1),
            Wait(TimeType(10 ** 6)),
            Set(0, 0.05),
            Set(1, 0.06),
            Wait(TimeType(10 ** 5)),

            LoopJmp(2),

            LoopJmp(1),
        ]

        self.output = []
        time = TimeType(0)
        for idx_b in range(100):
            for idx_a in range(200):
                self.output.append(
                    (time, [-.4, -.3])
                )
                time += 10 ** 5
                self.output.append(
                    (time, [-1. + idx_a * 0.01, -.5 + idx_b * 0.02])
                )
                time += 10 ** 6
                self.output.append(
                    (time, [0.05, 0.06])
                )
                time += 10 ** 5

    def test_singlet_scan_program(self):
        program_builder = LinSpaceBuilder(('a', 'b'))
        program = self.pulse_template.create_program(program_builder=program_builder)
        self.assertEqual([self.program], program)

    def test_singlet_scan_commands(self):
        commands = to_increment_commands([self.program])
        self.assertEqual(self.commands, commands)

    def test_singlet_scan_output(self):
        vm = LinSpaceVM(2)
        vm.set_commands(self.commands)
        vm.run()
        assert_vm_output_almost_equal(self, self.output, vm.history)


class TransformedRampTest(TestCase):
    def setUp(self):
        hold = ConstantPT(10 ** 6, {'a': '-1. + idx * 0.01'})
        self.pulse_template = hold.with_iteration('idx', 200)
        self.transformation = ScalingTransformation({'a': 2.0})

        self.program = LinSpaceIter(
            length=200,
            body=(LinSpaceHold(
                bases=(-2.,),
                factors=((0.02,),),
                duration_base=TimeType(10 ** 6),
                duration_factors=None
            ),)
        )

    def test_global_trafo_program(self):
        program_builder = LinSpaceBuilder(('a',))
        program = self.pulse_template.create_program(program_builder=program_builder,
                                                     global_transformation=self.transformation)
        self.assertEqual([self.program], program)

    def test_local_trafo_program(self):
        program_builder = LinSpaceBuilder(('a',))
        with self.assertRaises(NotImplementedError):
            # not implemented yet. This test should work as soon as its implemented
            program = self.pulse_template.create_program(program_builder=program_builder,
                                                         global_transformation=self.transformation,
                                                         to_single_waveform={self.pulse_template})
            self.assertEqual([self.program], program)


class HarmonicPulseTest(TestCase):
    def setUp(self):
        hold_duration = TimeType(10 ** 6)
        hold = ConstantPT(hold_duration, {'a': '-1. + idx * 0.01'})
        sine = FunctionPulseTemplate('sin(2 * pi * t)', duration_expression='10 / pi', channel='a')
        self.pulse_template = (hold @ sine).with_iteration('idx', 100)

        self.sine_waveform = sine.build_waveform(parameters={}, channel_mapping={'a': 'a'})

        self.program = LinSpaceIter(
            length=100,
            body=(LinSpaceHold(
                bases=(-1.,),
                factors=((0.01,),),
                duration_base=hold_duration,
                duration_factors=None
            ),
            LinSpaceArbitraryWaveform(
                waveform=self.sine_waveform,
                channels=('a',)
            )
            )
        )

        key = DepKey.from_voltages((0.01,), DEFAULT_INCREMENT_RESOLUTION)
        self.commands = [
            Set(0, -1.0, key),
            Wait(hold_duration),
            Play(self.sine_waveform, channels=('a',)),
            LoopLabel(0, 99),
            Increment(0, 0.01, key),
            Wait(hold_duration),
            Play(self.sine_waveform, channels=('a',)),
            LoopJmp(0)
        ]

        self.sample_resolution = TimeType(1)
        n_samples = int(self.sine_waveform.duration // self.sample_resolution)

        step_duration = hold_duration + self.sine_waveform.duration

        self.output = []
        for idx in range(100):
            hold_ampl = sum([-1.0] + [0.01] * idx)
            self.output.append((idx * step_duration, [hold_ampl]))
            for n in range(n_samples):
                inner_time = n * self.sample_resolution
                time = idx * step_duration + hold_duration + inner_time
                value = np.sin(float(2 * np.pi * inner_time))
                self.output.append((time, [value]))

    def test_program(self):
        program_builder = LinSpaceBuilder(('a',))
        program = self.pulse_template.create_program(program_builder=program_builder)

        self.assertEqual([self.program], program)

    def test_commands(self):
        commands = to_increment_commands([self.program])
        self.assertEqual(self.commands, commands)

    def test_output(self):
        vm = LinSpaceVM(1, sample_resolution=self.sample_resolution)
        vm.set_commands(self.commands)
        vm.run()
        assert_vm_output_almost_equal(self, self.output, vm.history)
