import unittest
from typing import Any

from qupulse.comparable import Comparable

class DummyComparable(Comparable):

    def __init__(self, compare_key: Any) -> None:
        super().__init__()
        self.compare_key_ = compare_key

    @property
    def compare_key(self) -> Any:
        return self.compare_key_


class ComparableTests(unittest.TestCase):

    def test_hash(self) -> None:
        comp_a = DummyComparable(17)
        self.assertEqual(hash(17), hash(comp_a))

    def test_eq(self) -> None:
        comp_a = DummyComparable(17)
        comp_b = DummyComparable(18)
        comp_c = DummyComparable(18)
        self.assertNotEqual(comp_a, comp_b)
        self.assertNotEqual(comp_b, comp_a)
        self.assertEqual(comp_b, comp_c)
        self.assertNotEqual(comp_a, "foo")
        self.assertNotEqual("foo", comp_a)
