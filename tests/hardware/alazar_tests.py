import unittest
from unittest import mock

import numpy as np

from ..hardware import *
from qupulse.hardware.dacs.alazar import AlazarCard, AlazarProgram
from qupulse.utils.types import TimeType


class AlazarProgramTest(unittest.TestCase):
    def setUp(self) -> None:
        # we currently allow overlapping masks in AlazarProgram (It will throw an error on upload)
        # This probably will change in the future
        self.masks = {
            'unsorted': (np.array([1., 100, 13]), np.array([10., 999, 81])),
            'sorted': (np.array([30., 100, 1300]), np.array([10., 990, 811])),
            'overlapping': (np.array([30., 100, 300]), np.array([20., 900, 100]))
        }
        self.sample_factor = TimeType.from_fraction(10**8, 10**9)
        self.expected = {
            'unsorted': (np.array([0, 1, 10]).astype(np.uint64), np.array([1, 8, 99]).astype(np.uint64)),
            'sorted': (np.array([3, 10, 130]).astype(np.uint64), np.array([1, 99, 81]).astype(np.uint64)),
            'overlapping': (np.array([3, 10, 30]).astype(np.uint64), np.array([2, 90, 10]).astype(np.uint64))
        }

    def test_length_computation(self):
        program = AlazarProgram()
        for name, data in self.masks.items():
            program.set_measurement_mask(name, self.sample_factor, *data)

        self.assertEqual(program.total_length, 130 + 81)
        self.assertIsNone(program._total_length)
        program.total_length = 17
        self.assertEqual(program.total_length, 17)

    def test_masks(self):
        program = AlazarProgram()
        for name, data in self.masks.items():
            program.set_measurement_mask(name, self.sample_factor, *data)

        names = []
        def make_mask(name, *data):
            np.testing.assert_equal(data, self.expected[name])
            assert name not in names
            names.append(name)
            return name

        result = program.masks(make_mask)

        self.assertEqual(names, result)

    def test_set_measurement_mask(self):
        program = AlazarProgram()

        begins, lengths = self.masks['sorted']
        with self.assertRaises(AssertionError):
            program.set_measurement_mask('foo', self.sample_factor, begins.astype(int), lengths)

        expected_unsorted = np.array([0, 1, 10]).astype(np.uint64), np.array([1, 8, 99]).astype(np.uint64)
        result = program.set_measurement_mask('unsorted', self.sample_factor, *self.masks['unsorted'])

        np.testing.assert_equal(program._masks, {'unsorted': expected_unsorted})
        np.testing.assert_equal(result, expected_unsorted)
        self.assertFalse(result[0].flags.writeable)
        self.assertFalse(result[1].flags.writeable)

        with self.assertRaisesRegex(RuntimeError, 'differing sample factor'):
            program.set_measurement_mask('sorted', self.sample_factor*5/4, *self.masks['sorted'])

        result = program.set_measurement_mask('sorted', self.sample_factor, *self.masks['sorted'])
        np.testing.assert_equal(result, self.expected['sorted'])

    def test_iter(self):
        program = AlazarProgram()
        program._masks = [4, 5, 6]
        program.operations = [1, 2, 3]
        program.total_length = 13
        program.masks = mock.MagicMock(return_value=342)

        mask_maker = mock.MagicMock()

        a, b, c = program.iter(mask_maker)

        self.assertEqual(a, 342)
        self.assertEqual(b, [1, 2, 3])
        self.assertEqual(c, 13)
        program.masks.assert_called_once_with(mask_maker)


class AlazarTest(unittest.TestCase):

    def test_add_mask_prototype(self):
        card = AlazarCard(None)

        card.register_mask_for_channel('M', 3, 'auto')
        self.assertEqual(card._mask_prototypes, dict(M=(3, 'auto')))

        with self.assertRaises(ValueError):
            card.register_mask_for_channel('M', 'A', 'auto')

        with self.assertRaises(NotImplementedError):
            card.register_mask_for_channel('M', 1, 'periodic')

    def test_make_mask(self):
        card = AlazarCard(None)
        card.register_mask_for_channel('M', 3, 'auto')

        begins = np.arange(15, dtype=np.uint64)*16
        lengths = 1+np.arange(15, dtype=np.uint64)

        with self.assertRaises(KeyError):
            card._make_mask('N', begins, lengths)

        with self.assertRaises(ValueError):
            card._make_mask('M', begins, lengths*3)

        mask = card._make_mask('M', begins, lengths)
        self.assertEqual(mask.identifier, 'M')
        np.testing.assert_equal(mask.begin, begins)
        np.testing.assert_equal(mask.length, lengths)
        self.assertEqual(mask.channel, 3)

    def test_register_measurement_windows(self):
        raw_card = dummy_modules.dummy_atsaverage.core.AlazarCard()
        card = AlazarCard(raw_card)

        self.assertIs(card.card, raw_card)

        card.register_mask_for_channel('A', 3, 'auto')
        card.register_mask_for_channel('B', 1, 'auto')

        card.config = dummy_modules.dummy_atsaverage.config.ScanlineConfiguration()

        card.register_measurement_windows('empty', dict())

        begins = np.arange(100)*176.5
        lengths = np.ones(100)*10*np.pi
        card.register_measurement_windows('otto', dict(A=(begins, lengths)))

        self.assertEqual(set(card._registered_programs.keys()), {'empty', 'otto'})
        self.assertEqual(card._registered_programs['empty'].masks(lambda x: x), [])

        [(result_begins, result_lengths)] = card._registered_programs['otto'].masks(lambda _, b, l: (b, l))
        expected_begins = np.rint(begins / 10).astype(dtype=np.uint64)
        np.testing.assert_equal(result_begins, expected_begins)

        # pi ist genau 3
        np.testing.assert_equal(result_lengths if isinstance(result_lengths, np.ndarray) else result_lengths.as_ndarray(), 3)

    def test_register_operations(self):
        card = AlazarCard(None)

        operations = 'this is no operatoin but a string'
        card.register_operations('test', operations)
        self.assertEqual(len(card._registered_programs), 1)
        self.assertIs(card._registered_programs['test'].operations, operations)

    def test_mask_prototypes(self):
        card = AlazarCard(None)
        self.assertIs(card.mask_prototypes, card._mask_prototypes)

    def test_arm_operation(self):
        raw_card = dummy_modules.dummy_atsaverage.core.AlazarCard()
        card = AlazarCard(raw_card)

        card.register_mask_for_channel('A', 3, 'auto')
        card.register_mask_for_channel('B', 1, 'auto')

        card.register_operations('otto', [])

        card.config = dummy_modules.dummy_atsaverage.config.ScanlineConfiguration()

        with self.assertRaisesRegex(RuntimeError, 'No operations'):
            card.arm_program('otto')

        card.register_operations('otto', ['asd'])

        with self.assertRaisesRegex(RuntimeError, "No masks"):
            card.arm_program('otto')

        begins = np.arange(100) * 176.5
        lengths = np.ones(100) * 10 * np.pi
        card.register_measurement_windows('otto', dict(A=(begins, lengths)))

        card.config.totalRecordSize = 17

        with self.assertRaisesRegex(ValueError, "total record size is smaller than needed"):
            card.arm_program('otto')

        card.config.totalRecordSize = 0

        with mock.patch.object(card.card, 'applyConfiguration') as mock_applyConfiguration:
            with mock.patch.object(card.card, 'startAcquisition') as mock_startAcquisition:
                card.arm_program('otto')

                mock_applyConfiguration.assert_called_once_with(card.config, True)
                mock_startAcquisition.assert_called_once_with(1)

                mock_applyConfiguration.reset_mock()
                mock_startAcquisition.reset_mock()
                card.arm_program('otto')

                mock_applyConfiguration.assert_not_called()
                mock_startAcquisition.assert_called_once_with(1)
