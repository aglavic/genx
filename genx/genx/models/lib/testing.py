"""
Testing framework for GenX models. Allows the test case to discover all implemented model
tests and run them in a coherent way.
"""

import unittest
import numpy as np

ModelTestCases = []

class ModelTestCase(unittest.TestCase):
    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)
        ModelTestCases.append(cls)

    @property
    def tth(self):
        # x-points for tth in reflectivity
        return np.linspace(0.01, 6.0, 15)

    @property
    def qz(self):
        # x-points for q in reflectivity
        return np.linspace(0.005, 0.3, 15)
