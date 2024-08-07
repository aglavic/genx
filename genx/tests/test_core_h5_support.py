"""
Generate a simple test class that is used to test the hdf interface of all important
data types.
"""

import tempfile
import unittest

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

import h5py
import numpy as np

from genx.core.h5_support import H5HintedExport


def default_dict():
    return {"str": "a", "int": 13, "float": 134.2, "complex": 41.5 + 3j, "subdict": {"a": "a"}}


@dataclass
class H5TestSub(H5HintedExport):
    h5group_name = "subtest"
    _group_attr = {"NX_class": "NXentry", "default": "datasets"}

    testvalue: str


@dataclass
class H5Tester(H5HintedExport):
    h5group_name = "test_subgroup"
    # some attributes (without type hint) will not be saved
    non_saved = None
    non_saved_int = 13
    non_saved_float = 134.5

    saved_str: str
    saved_int: int
    saved_float: float
    saved_complex: complex
    saved_dict: dict
    saved_Dict: Dict[str, str]
    saved_Dict2: Dict[str, int]
    saved_array: np.ndarray
    saved_subclass: H5TestSub

    saved_default_str: str = "default"
    saved_default_int: int = 13
    saved_default_float: float = 31.45
    saved_default_complex: complex = 13.5 + 3j
    saved_default_dict: dict = field(default_factory=default_dict)
    saved_default_array: np.ndarray = field(default_factory=lambda: np.arange(100))


class H5Tester2(H5HintedExport):
    h5group_name = "test_group"

    _ignored: list = [1, 2, 3]

    checked_default: str = "abc"
    checked_list: list = [1, 2, 3]


class TestH5HintedExport(unittest.TestCase):

    def test_write_read(self):
        ds1 = H5Tester(
            "abc1", 10, 45.3, 85.4 + 5.3j, default_dict(), {}, {}, np.linspace(-1.0, 1.0, 50), H5TestSub("test")
        )
        ds2 = H5Tester(
            "abc2",
            20,
            25.3,
            45.4 + 2.3j,
            {"b": "c", "d": 13, "e": datetime(2024, 8, 1), "f": {1, 2, 3}},
            {"a": "äö"},
            {"a": 13},
            np.linspace(-2, 1, 50),
            H5TestSub("test"),
            "def",
            42,
            42.5,
            0.4 - 2j,
            {"b": "c", "d": 13},
            np.linspace(-2, 1, 50000),
        )  # overwritten defaults
        ds1re = H5Tester("", 0, 0.0, 0j, {}, {}, {}, np.array([]), H5TestSub(""))
        ds2re = H5Tester("", 0, 0.0, 0j, {}, {}, {}, np.array([]), H5TestSub(""), "", 0, 0.0, 0j, {}, np.array([]))

        with tempfile.TemporaryFile("w+b") as tmp:
            hdf = h5py.File(tmp, "w")
            g1 = hdf.create_group("ds1")
            ds1.write_h5group(g1)
            g2 = hdf.create_group("ds2")
            ds2.write_h5group(g2)
            hdf.close()

            hdf = h5py.File(tmp, "r")
            ds1re.read_h5group(hdf["ds1"])
            ds2re.read_h5group(hdf["ds2"])

        # remove array objects for comparison
        ds1a = ds1.saved_array
        ds1da = ds1.saved_default_array
        ds1.saved_array = None
        ds1.saved_default_array = None

        ds2a = ds2.saved_array
        ds2da = ds2.saved_default_array
        ds2.saved_array = None
        ds2.saved_default_array = None

        ds1ra = ds1re.saved_array
        ds1rda = ds1re.saved_default_array
        ds1re.saved_array = None
        ds1re.saved_default_array = None

        ds2ra = ds2re.saved_array
        ds2rda = ds2re.saved_default_array
        ds2re.saved_array = None
        ds2re.saved_default_array = None

        self.assertEqual(ds1, ds1re)
        self.assertEqual(ds2, ds2re)

        np.testing.assert_array_equal(ds1a, ds1ra)
        np.testing.assert_array_equal(ds1da, ds1rda)
        np.testing.assert_array_equal(ds2a, ds2ra)
        np.testing.assert_array_equal(ds2da, ds2rda)

    def test_hinted(self):
        t = H5Tester2()
        t.init_defaults()
        self.assertFalse(t.checked_list is H5Tester2.checked_list)
        self.assertEqual(t.checked_list, H5Tester2.checked_list)
        self.assertTrue(t._ignored is H5Tester2._ignored)
