"""
Test of special list class in colors module.
"""

import unittest

from genx.core.colors import CyclicList


class TestCyclicList(unittest.TestCase):

    def test_cycling(self):
        t = CyclicList([1, 2, 3])
        for i in range(-20, 21):
            self.assertEqual(t[i], [1, 2, 3][i % 3])

    def test_eq_hash(self):
        t = CyclicList([1, 2, 3])

        self.assertEqual(t, t)
        self.assertEqual(t, [1, 2, 3])
        _ = {t: 1}
