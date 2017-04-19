import unittest

from . import dummy_modules
dummy_modules.import_package('atsaverage')

import atsaverage
import atsaverage.config
from atsaverage.config import ScanlineConfiguration

import numpy as np

from qctoolkit.hardware.dacs.alazar import AlazarCard


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
        self.assertIs(mask.begin, begins)
        self.assertIs(mask.length, lengths)
        self.assertEqual(mask.channel, 3)

    def test_register_measurement_windows(self):

        card = AlazarCard(dummy_modules.dummy_atsaverage.core.AlazarCard())
        card.register_mask_for_channel('A', 3, 'auto')
        card.register_mask_for_channel('B', 1, 'auto')

        card.config = dummy_modules.dummy_atsaverage.config.ScanlineConfiguration()

        card.register_measurement_windows('empty', dict())

        begins = np.arange(100)*176.5
        lengths = np.ones(100)*10*np.pi
        card.register_measurement_windows('otto', dict(A=(begins, lengths)))

        self.assertEqual(set(card._registered_programs.keys()), {'empty', 'otto'})
        self.assertEqual(card._registered_programs['empty'].masks, [])

        expected_begins = np.rint(begins / 10).astype(dtype=np.uint64)
        self.assertEqual(expected_begins.dtype, card._registered_programs['otto'].masks[0].begin.dtype)
        np.testing.assert_equal(card._registered_programs['otto'].masks[0].begin, expected_begins)

        # pi ist genau 3
        self.assertTrue(np.all(card._registered_programs['otto'].masks[0].length == 3))

        self.assertEqual(card._registered_programs['otto'].masks[0].channel, 3)
        self.assertEqual(card._registered_programs['otto'].masks[0].identifier, 'A')

