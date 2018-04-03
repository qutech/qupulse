import unittest

from qctoolkit.pulses.parameters import ParameterConstraint, ParameterConstraintViolation,\
    ParameterNotProvidedException, ParameterConstrainer, ConstantParameter
from qctoolkit.pulses.measurement import MeasurementDefiner

from qctoolkit.pulses.instructions import InstructionBlock, MEASInstruction


class MeasurementDefinerTest(unittest.TestCase):
    def __init__(self, *args, to_test_constructor=None, **kwargs):
        super().__init__(*args, **kwargs)

        if to_test_constructor is None:
            self.to_test_constructor = lambda measurements=None: MeasurementDefiner(measurements=measurements)
        else:
            self.to_test_constructor = to_test_constructor

    def test_measurement_windows(self) -> None:
        pulse = self.to_test_constructor(measurements=[('mw', 0, 5)])
        with self.assertRaises(KeyError):
            pulse.get_measurement_windows(parameters=dict(), measurement_mapping=dict())
        windows = pulse.get_measurement_windows(parameters=dict(), measurement_mapping={'mw': 'asd'})
        self.assertEqual([('asd', 0, 5)], windows)
        self.assertEqual(pulse.measurement_declarations, [('mw', 0, 5)])

    def test_multiple_windows(self):
        pulse = self.to_test_constructor(measurements=[('mw', 0, 5), ('H', 'a', 'b')])
        with self.assertRaises(KeyError):
            pulse.get_measurement_windows(parameters=dict(), measurement_mapping=dict())
        windows = pulse.get_measurement_windows(parameters=dict(a=0.5, b=1), measurement_mapping={'mw': 'asd', 'H': 'H'})
        self.assertEqual([('asd', 0, 5), ('H', 0.5, 1)], windows)
        self.assertEqual(pulse.measurement_declarations, [('mw', 0, 5), ('H', 'a', 'b')])

    def test_no_measurement_windows(self) -> None:
        pulse = self.to_test_constructor()
        windows = pulse.get_measurement_windows(dict(), {'mw': 'asd'})
        self.assertEqual([], windows)
        self.assertEqual([], pulse.measurement_declarations)

    def test_measurement_windows_with_parameters(self) -> None:
        pulse = self.to_test_constructor(measurements=[('mw', 1, '(1+length)/2')])
        parameters = dict(length=100)
        windows = pulse.get_measurement_windows(parameters, measurement_mapping={'mw': 'asd'})
        self.assertEqual(windows, [('asd', 1, 101 / 2)])
        self.assertEqual(pulse.measurement_declarations, [('mw', 1, '(1+length)/2')])

    def test_measurement_windows_invalid(self) -> None:
        pulse = self.to_test_constructor(measurements=[('mw', 'a', 'd')])
        measurement_mapping = {'mw': 'mw'}

        with self.assertRaises(ValueError):
            pulse.get_measurement_windows(measurement_mapping=measurement_mapping,
                                          parameters=dict(length=10, a=-1, d=3))
        with self.assertRaises(ValueError):
            pulse.get_measurement_windows(measurement_mapping=measurement_mapping,
                                          parameters=dict(length=10, a=3, d=-1))

    def test_insert_measurement_instruction(self):
        pulse = self.to_test_constructor(measurements=[('mw', 'a', 'd')])
        parameters = {'a': ConstantParameter(0), 'd': ConstantParameter(0.9)}
        measurement_mapping = {'mw': 'as'}

        block = InstructionBlock()
        pulse.insert_measurement_instruction(instruction_block=block,
                                             parameters=parameters,
                                             measurement_mapping=measurement_mapping)

        expected_block = [MEASInstruction([('as', 0, 0.9)])]
        self.assertEqual(block.instructions, expected_block)


    def test_none_mappings(self):
        pulse = self.to_test_constructor(measurements=[('mw', 'a', 'd'), ('asd', 0, 1.)])

        parameters = dict(length=100, a=4, d=5)

        windows = pulse.get_measurement_windows(parameters, measurement_mapping={'mw': 'mw', 'asd': None})
        self.assertEqual(windows, [('mw', 4, 5)])

        windows = pulse.get_measurement_windows(dict(length=100), measurement_mapping={'mw': None, 'asd': None})
        self.assertEqual(windows, [])
