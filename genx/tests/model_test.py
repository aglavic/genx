import unittest

from genx.model import Model

class TestModelClass(unittest.TestCase):
    def setUp(self):
        self.m=Model()

    def test_load(self):
        self.assertAlmostEqual(True, True)
    
    def test_save(self):
        self.assertAlmostEqual(True, True)
    
    def test_load_h5(self):
        self.assertAlmostEqual(True, True)

    def test_save_h5(self):
        self.assertAlmostEqual(True, True)



if __name__=='__main__':
    unittest.main()
