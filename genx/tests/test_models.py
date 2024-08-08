"""
Test all reflectivity models generically and discover all models implementing test cases.
"""

import os
import unittest

from genx import api
from genx.models.lib.testing import ModelTestCases
from glob import glob
from importlib import import_module

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "genx", "models")

class TestReflModels(unittest.TestCase):
    def test_empty_models(self):
        for i, name in enumerate(["spec_nx", "spec_adaptive", "spec_inhom", "soft_nx", "mag_refl", "interdiff"]):
            with self.subTest(f"Model {name}", i=i):
                model, optimizer, refl = api.Reflectivity.create_new(name)
                refl.ReadModel()
                model.compile_script()
                model.simulate()


def load_tests(loader, standard_tests, pattern):
    """
    Build a test suite for all model test cases. First import all models.* modules to define these tests.
    """
    for mi in glob(os.path.join(MODEL_DIR, "*.py")):
        module_name = os.path.basename(mi)[:-3]
        if mi in ['__init__', 'symmetries', 'utils']:
            continue
        import_module(f'genx.models.{module_name}', )

    suite = unittest.TestSuite()
    for test_class in [TestReflModels]+ModelTestCases:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite
