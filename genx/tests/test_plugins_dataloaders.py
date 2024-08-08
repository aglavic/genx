"""
Test of data loader plugins.
"""

import os
import unittest

from genx import api

# TODO: Get test data for all loaders and put in test folder
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "genx", "examples")

class TestDataLoaders(unittest.TestCase):
    def test_data_loaders(self):
        model, optimizer, refl = api.Reflectivity.create_new("spec_nx")

        api.data_loader.d17_legacy.LoadData(model.data[0], os.path.join(EXAMPLE_DIR, "D17_SiO.out"))
        api.data_loader.default.LoadData(model.data[0], os.path.join(EXAMPLE_DIR, "xray-tutorial.dat"))
