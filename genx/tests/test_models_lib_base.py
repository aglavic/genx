"""
Test of the models.lib.base classes used in model builds (new style).
"""

import unittest

from dataclasses import dataclass, field

from genx.models.lib.base import AltStrEnum, ModelParamBase


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

    def test_extraction(self):
        @dataclass
        class TestParam(ModelParamBase):
            p_str: str = field(default_factory=str)
            p_int: int = 0
            p_float: float = 0.0

        txt = 'TestParam(p_int = 1+1, p_float = 0.01+0.02, p_str="abc"+"def")'
        t: TestParam = eval(txt)
        self.assertEqual(t.p_int, 2)
        self.assertEqual(t.p_float, 0.03)
        self.assertEqual(t.p_str, "abcdef")
        t._extract_callpars(txt)
        self.assertEqual(t._ca["p_int"], "1+1")
        self.assertEqual(t._ca["p_float"], "0.01+0.02")
        self.assertEqual(eval(t._ca["p_str"]), "abcdef")

        txt = "TestParam()"
        t: TestParam = eval(txt)
        self.assertEqual(t.p_int, 0)
        self.assertEqual(t.p_float, 0.0)
        self.assertEqual(t.p_str, "")
        t._extract_callpars(txt)
        self.assertEqual(t._ca["p_int"], "0")
        self.assertEqual(t._ca["p_float"], "0.0")
        self.assertEqual(eval(t._ca["p_str"]), "")

    def test_getter_setter(self):
        @dataclass
        class TestParam(ModelParamBase):
            p_str: str = field(default_factory=str)
            p_int: int = 0
            p_float: float = 0.0
            p_complex: complex = 0 + 0j

        t = TestParam()
        self.assertTrue(hasattr(t, "setP_str"))
        self.assertTrue(hasattr(t, "getP_str"))
        self.assertTrue(hasattr(t, "setP_int"))
        self.assertTrue(hasattr(t, "getP_int"))
        self.assertTrue(hasattr(t, "setP_float"))
        self.assertTrue(hasattr(t, "getP_float"))
        self.assertTrue(hasattr(t, "setP_complexreal"))
        self.assertTrue(hasattr(t, "getP_complexreal"))
        self.assertTrue(hasattr(t, "setP_compleximag"))
        self.assertTrue(hasattr(t, "getP_compleximag"))

        self.assertEqual(t.p_int, t.getP_int())
        t.setP_int(24)
        self.assertEqual(t.p_int, 24)
        self.assertEqual(t.p_complex, t.getP_complexreal() + 1j * t.getP_compleximag())
        t.setP_complexreal(0.1)
        t.setP_compleximag(0.4)
        self.assertEqual(t.p_complex, 0.1 + 0.4j)

    def test_recreate(self):
        @dataclass
        class TestParam(ModelParamBase):
            p_str: str = field(default_factory=str)
            p_int: int = 0
            p_float: float = 0.0

        txt = 'TestParam(p_int = 1+1, p_float = 0.01+0.02, p_str="abc"+"def")'
        t: TestParam = eval(txt)
        # repr without extracted params, build from values
        self.assertEqual(t, eval(t._repr_call()))
        t._extract_callpars(txt)
        # repr with extracted params, should be re-evaluated
        self.assertEqual(t, eval(t._repr_call()))


class TestAltStrEnum(unittest.TestCase):

    def test_equivalency(self):
        class TestAltEnum(AltStrEnum):
            peter = "peter"
            kurt = "kurt"
            alternate1_peter = "gert"
            alternate2_peter = "hans"

        self.assertEqual(TestAltEnum("peter"), TestAltEnum("peter"))
        self.assertEqual(TestAltEnum("peter"), TestAltEnum("gert"))
        self.assertEqual(TestAltEnum("gert"), TestAltEnum("peter"))
        self.assertTrue(TestAltEnum("peter") != TestAltEnum("kurt"))
        self.assertTrue(TestAltEnum("gert") != TestAltEnum("kurt"))

    def test_basic_methods(self):
        class TestAltEnum(AltStrEnum):
            peter = "peter"
            kurt = "kurt"
            alternate1_peter = "gert"
            alternate2_peter = "hans"

        t = TestAltEnum("peter")
        self.assertEqual(str(t), "peter")
        self.assertEqual(eval(repr(t)), "peter")
        d = {t: "peter"}

    def test_indexing(self):
        class TestAltEnum(AltStrEnum):
            peter = "peter"
            kurt = "kurt"
            alternate1_peter = "gert"
            alternate2_peter = "hans"

        self.assertEqual(TestAltEnum("peter"), TestAltEnum(0))
        self.assertEqual(TestAltEnum("kurt"), TestAltEnum(1))
        self.assertEqual(TestAltEnum("gert"), TestAltEnum(2))
        self.assertEqual(TestAltEnum("hans"), TestAltEnum(3))

        with self.assertRaises(ValueError):
            TestAltEnum(4)
        with self.assertRaises(ValueError):
            TestAltEnum("nose")
        with self.assertRaises(ValueError):
            TestAltEnum(-1)
