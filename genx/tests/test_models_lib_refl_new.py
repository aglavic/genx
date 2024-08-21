"""
Test of the models.lib.base classes used in model builds (new style).
"""

import unittest

from dataclasses import dataclass, field
from typing import List

from genx.models.lib.refl_base import ReflBase, SampleBase, StackBase


class TestReflBase(unittest.TestCase):

    def test_refl_base(self):
        @dataclass
        class TestReflBase(ReflBase):
            p_float: float = 0.0
            p_int: int = 0

        t = TestReflBase()
        self.assertEqual(t._parameters, {"p_float": 0.0, "p_int": 0})
        for name, pinfo in t._parameter_info().items():
            self.assertEqual(pinfo.type, type(getattr(t, name)))

    def test_stack(self):
        @dataclass
        class TestStack(StackBase):
            p_float: float = 0.0
            p_int: int = 0

        t = TestStack(Layers=[])

        self.assertEqual(t._parameters, {"p_float": 0.0, "p_int": 0, "Layers": [], "Repetitions": 1})

        self.assertEqual(t.resolveLayerParameter("d"), [])

    def test_layer_stack(self):
        @dataclass
        class TestStack(StackBase):
            p_float: float = 0.0
            p_int: int = 0

        @dataclass
        class TestLayer(ReflBase):
            d: float = 0.0

        t = TestStack(Layers=[TestLayer()], Repetitions=1)
        self.assertEqual(t.resolveLayerParameter("d"), [0.0])
        t = TestStack(Layers=[TestLayer()], Repetitions=5)
        self.assertEqual(t.resolveLayerParameter("d"), [0.0] * 5)

    def test_sample(self):
        @dataclass
        class TestStack(StackBase):
            p_float: float = 0.0
            p_int: int = 0

        @dataclass
        class TestLayer(ReflBase):
            d: float = 0.0

        @dataclass
        class LayerParameters:
            d: List[float] = field(default_factory=list)

        @dataclass
        class TestSample(SampleBase):
            _layer_parameter_class = LayerParameters

        @dataclass
        class TestSample2(SampleBase): ...

        s = TestSample(
            Stacks=[TestStack(Layers=[TestLayer()], Repetitions=1)], Ambient=TestLayer(), Substrate=TestLayer()
        )
        self.assertEqual(s.resolveLayerParameters(), LayerParameters(d=[0.0, 0.0, 0.0]))
        s2 = TestSample2(
            Stacks=[TestStack(Layers=[TestLayer()], Repetitions=1)], Ambient=TestLayer(), Substrate=TestLayer()
        )
        self.assertEqual(s2.resolveLayerParameters(), dict(d=[0.0, 0.0, 0.0]))

        def sim_test(first, smpl, last):
            if first != last:
                raise ValueError()
            return smpl

        TestSample.setSimulationFunctions({"Bier": sim_test})
        self.assertTrue(hasattr(s, "SimBier"))
        self.assertEqual(s.SimBier(1, 1), s)
        with self.assertRaises(ValueError):
            s.SimBier(1, 2)
