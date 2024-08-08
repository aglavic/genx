"""
Load all example files, compile the script and make a simple simulation.
"""

import os
import unittest

from glob import glob

from genx import api

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "genx", "examples")


class TestExamples(unittest.TestCase):
    def test_all(self):
        fls = glob(os.path.join(BASE_DIR, "*.hgx"))
        for i, fl in enumerate(fls):
            with self.subTest(f"Datafile {fl}", i=i):
                model, optimizer = api.load(fl)
                model.simulate()
