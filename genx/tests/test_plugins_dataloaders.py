"""
Test of data loader plugins.
"""

import unittest

from pathlib import Path
from os import sep

from genx.data import DataSet
from genx.plugins.utils import PluginHandler

DATA_DIR = Path(__file__).absolute().parent/"data"

class Dummy:
    def SetStatusText(self, message):
        pass

class TestDataLoaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        head = DATA_DIR.parent.parent
        cls._handler = PluginHandler(None, str(head / "genx" / "plugins")+sep, "data_loaders")
        cls.data_loaders = {}
        parent = Dummy()
        for dl in cls._handler.get_possible_plugins():
            try:
                cls._handler.load_plugin(dl)
                cls._handler.loaded_plugins[dl].parent = parent
                cls.data_loaders[dl] = cls._handler.loaded_plugins[dl]
            except Exception as error:
                print(error)

    def test_d17_legacy(self):
        data = DataSet()
        self.data_loaders['d17_legacy'].LoadData(data, DATA_DIR / "D17_SiO.out")

    def test_default(self):
        data = DataSet()
        self.data_loaders['default'].LoadData(data,DATA_DIR / "xray-tutorial.dat")

    def test_auto(self):
        # use all wildcards provided by the auto loaded to try to open files in the data folder
        files = []
        auto =self.data_loaders['auto']
        print(auto.wildcard)
        for wc in auto.wildcard.split(';'):
            files += list(DATA_DIR.glob(wc))
        for i,fi in enumerate(set(files)):
            data = DataSet()
            auto.LoadData(data, str(fi))
            with self.subTest(msg=f'{fi.name} by {auto._last_loader_used}'):
                self.assertTrue(len(data.x_raw)>0)
                self.assertEqual(len(data.x_raw), len(data.y_raw))
                self.assertEqual(len(data.x_raw), len(data.error_raw))
                self.assertEqual(len(data.extra_data_raw.keys()), len(data.extra_commands.keys()))
                self.assertEqual(len(data.extra_data_raw.keys()), len(data.extra_data.keys()))
