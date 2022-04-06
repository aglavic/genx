"""
Tests of low lever functionality of GenX Model class.
"""
import os
import unittest
import h5py
import tempfile
from pickle import loads, dumps

from genx.model import Model


class TestModelClass(unittest.TestCase):
    m: Model

    def setUp(self):
        self.m = Model()
        self.example_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         'genx', 'examples')

    def test_save_load_gx(self):
        fd, fname = tempfile.mkstemp(suffix='.gx')
        self.m.save(fname)
        self.m.load(fname)
        os.close(fd)

        self.assertEqual(self.m.filename, fname)
        self.assertTrue(self.m.saved)

        self.m = Model()

    def test_load_h5(self):
        fname = os.path.join(self.example_path, 'SuperAdam_SiO.hgx')
        f = h5py.File(fname.encode('utf-8'), 'r')
        g = f[self.m.h5group_name]
        self.m.read_h5group(g)

        self.assertTrue(self.m.saved)
        self.assertLess(0, len(self.m.data[0].x_raw))
        self.assertLess(3, len(self.m.parameters))
        self.assertNotEqual(self.m.script.strip(), '')

        self.m = Model()

    def test_save_h5(self):
        fd, fname = tempfile.mkstemp(suffix='.gx')
        f = h5py.File(fname.encode('utf-8'), 'w')
        g = f.create_group(self.m.h5group_name)
        self.m.write_h5group(g)
        f.close()
        os.close(fd)
        self.m = Model()

    def test_pickle(self):
        pstr = dumps(self.m)
        remodel = loads(pstr)
        for attr in ['opt', 'script', 'data', 'parameters']:
            with self.subTest(f'attr={attr}'):
                old = getattr(self.m, attr)
                new = getattr(remodel, attr)
                self.assertEqual(old, new)


if __name__=='__main__':
    unittest.main()
