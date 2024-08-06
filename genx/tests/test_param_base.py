'''
Test of the models.lib.base classes used in model builds (new style).
'''

import unittest

from dataclasses import dataclass, field

from genx.models.lib.base import ModelParamBase, AltStrEnum

class TestModelParamBase(unittest.TestCase):

    def test_conversion(self):
        @dataclass
        class TestParam(ModelParamBase):
            p_str: str = field(default_factory=str)
            p_int: int = 0
            p_float: float = 0.0

        t = TestParam(p_int=1.1, p_float="1", p_str=13)
        self.assertEqual(t.p_int, 1)
        self.assertEqual(t.p_float, 1.0)
        self.assertEqual(t.p_str, "13")
        self.assertEqual(type(t.p_int), int)
        self.assertEqual(type(t.p_float), float)
        self.assertEqual(type(t.p_str), str)
