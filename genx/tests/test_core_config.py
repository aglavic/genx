"""
Test of special list class in colors module.
"""

import os
import unittest

from dataclasses import dataclass, field, fields

from genx.core import config as c
from genx.exceptions import GenxIOError, GenxOptionError

CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test.conf")
EXP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "export.conf")


class TestConfig(unittest.TestCase):

    def test_load_default(self):
        config = c.Config()
        config.load_default(CFG_PATH, reset=True)

        # check that the default_config data was actually loaded and the model_config reset
        self.assertEqual(config.default_config["section_1"], {"p int": "1", "p float": "1.1", "p str": "hallo world"})
        self.assertEqual(config.model_config.sections(), [])

        with self.assertRaises(GenxIOError):
            config.load_default("none existant path")

    def test_write_default(self):
        config = c.Config()
        config.write_default(EXP_PATH)
        self.assertTrue(os.path.isfile(EXP_PATH))
        os.unlink(EXP_PATH)

    def test_fget(self):
        config = c.Config()
        config.default_config["section_1"] = {
            "p int": "1",
            "p float": "1.1",
            "p str": "hallo world",
            "p bool": True,
            "p2 bool": 0,
        }
        config.model_config["section_1"] = {"p int": "2", "p float": "2.2", "p str": "hallo local"}

        # items from local config
        self.assertEqual(config._getf("getint", "section_1", "p int"), 2)
        self.assertTrue(isinstance(config._getf("getint", "section_1", "p int"), int))
        self.assertEqual(config._getf("getfloat", "section_1", "p float"), 2.2)
        self.assertTrue(isinstance(config._getf("getfloat", "section_1", "p float"), float))
        self.assertEqual(config._getf("get", "section_1", "p str"), "hallo local")
        # items from default config
        self.assertEqual(config._getf("getboolean", "section_1", "p bool"), True)
        self.assertEqual(config._getf("getboolean", "section_1", "p2 bool"), False)
        self.assertTrue(isinstance(config._getf("getboolean", "section_1", "p bool"), bool))

        # conversion failing
        self.assertEqual(config._getf("getboolean", "section_1", "p str", fallback=True), True)

        with self.assertRaises(AttributeError):
            config._getf("getbeer", "section_1", "p bool")
        with self.assertRaises(GenxOptionError):
            config._getf("getint", "section_2", "p bool")
        with self.assertRaises(GenxOptionError):
            config._getf("getint", "section_1", "p bool")
        with self.assertRaises(GenxOptionError):
            config._getf("getint", "section_1", "not existing")

    def test_get_other(self):
        config = c.Config()
        config.default_config["section_1"] = {
            "p int": "1",
            "p float": "1.1",
            "p str": "hallo world",
            "p bool": True,
            "p2 bool": 0,
        }
        config.model_config["section_1"] = {"p int": "2", "p float": "2.2", "p str": "hallo local", "p list": "1;2;3"}

        self.assertEqual(config.getlist("section_1", "p list"), ["1", "2", "3"])
        with self.assertRaises(AttributeError):
            config.not_existant

    def test_read_string(self):
        config = c.Config()
        config.load_string("[section_1]\nbier = 13")
        self.assertEqual(config.model_config["section_1"]["bier"], "13")

        with self.assertRaises(GenxIOError):
            config.load_string("not the right format")

    def test_set(self):
        config = c.Config()
        config.set("section_1", "p int", 1)
        config.default_set("section_1", "p int", 2)
        self.assertEqual(config.model_config["section_1"]["p int"], "1")
        self.assertEqual(config.default_config["section_1"]["p int"], "2")

    def test_dump(self):
        config = c.Config()
        config.set("section_1", "p int", 1)
        config.set("section_1", "p float", 1.5)
        dump = config.model_dump()
        config.set("section_1", "p int", 2)
        self.assertEqual(config.getint("section_1", "p int"), 2)
        config.load_string(dump)
        self.assertEqual(config.getint("section_1", "p int"), 1)
        config.load_string(dump)


class TestBaseConfig(unittest.TestCase):

    def setUp(self):
        @dataclass
        class TestConfig(c.BaseConfig):
            section = "section_1"
            p_int: int
            p_float: float = 1.1
            p_str: str = "hallo world"
            p_bool: bool = True
            p_list: list = field(default_factory=list)

        self.test_class = TestConfig

    def test_abc(self):
        class Broken(c.BaseConfig): ...

        with self.assertRaises(TypeError):
            Broken()

    def test_load_default(self):
        cfg = self.test_class(p_int=1)
        self.assertEqual(
            cfg.asdict(), {"p_bool": True, "p_float": 1.1, "p_int": 1, "p_list": [], "p_str": "hallo world"}
        )

        c.config = c.Config()
        c.config.model_set("section_1", "p int", "13")
        c.config.model_set("section_1", "p float", "13.13")
        c.config.model_set("section_1", "p bool", "false")
        c.config.model_set("section_1", "p str", "hallo local")
        c.config.model_set("section_1", "p list", "1;2;3")
        cfg.load_config()
        self.assertEqual(
            cfg.asdict(),
            {"p_bool": False, "p_float": 13.13, "p_int": 13, "p_list": ["1", "2", "3"], "p_str": "hallo local"},
        )
        self.assertEqual(cfg.p_int, 13)
        self.assertEqual(cfg.p_bool, False)
        self.assertEqual(cfg.p_float, 13.13)
        self.assertEqual(cfg.p_list, ["1", "2", "3"])
        self.assertEqual(cfg.p_str, "hallo local")
        # ignore wrong values with a debug message
        c.config.model_set("section_1", "p int", "bier")
        cfg.load_config()
        self.assertEqual(cfg.p_int, 13)

    def test_save(self):
        cfg = self.test_class(p_int=1)
        c.config = c.Config()
        cfg.save_config(default=False)
        self.assertEqual(
            dict(c.config.model_config["section_1"]),
            {"p int": "1", "p float": "1.1", "p str": "hallo world", "p bool": "True", "p list": ""},
        )
        self.assertEqual(c.config.default_config.sections(), [])
        cfg.save_config(default=True)
        self.assertEqual(
            dict(c.config.default_config["section_1"]),
            {"p int": "1", "p float": "1.1", "p str": "hallo world", "p bool": "True", "p list": ""},
        )

    def test_copy(self):
        cfg = self.test_class(p_int=1)
        cpy = cfg.copy()
        self.assertEqual(cpy, cfg)
        self.assertFalse(cpy is cfg)

    def test_groups(self):
        cfg = self.test_class(p_int=1)
        # check that the items match, order could be arbitrary
        self.assertEqual(set(cfg.groups[""]), set(["p_bool", "p_float", "p_int", "p_list", "p_str"]))

    def test_fields(self):
        cfg = self.test_class(p_int=1)
        flds = list(fields(cfg))
        self.assertEqual(cfg.get_fields(), flds)
        self.assertEqual(len(cfg.get_fields()), 5)
        self.assertEqual(len(cfg.get_fields(["p_bool", "p_int"])), 2)
        self.assertEqual(len(cfg.get_fields(["p_bool", "none"])), 1)

    def test_gparm(self):
        res = self.test_class.GParam(13, pmin=53, pmax=12, label="hello world", descriptoin="this is a parameter")
        self.assertEqual(res.default, 13)
        self.assertEqual(
            res.metadata["genx"], {"pmin": 53, "pmax": 12, "label": "hello world", "descriptoin": "this is a parameter"}
        )

        res = self.test_class.GChoice(13, [13, 14, 15], label="hello world", descriptoin="this is a parameter")
        self.assertEqual(res.default, 13)
        self.assertEqual(
            res.metadata["genx"],
            {"selection": [13, 14, 15], "label": "hello world", "descriptoin": "this is a parameter"},
        )

    def test_merged(self):
        @dataclass
        class TestSecond(c.BaseConfig):
            section = "section_2"
            second: int = 14
            groups = {"second": ["second"]}

        cfg = self.test_class(p_int=1)
        cfg2 = TestSecond()
        mrg = cfg | cfg2
        mrg.p_float = 13.13
        self.assertEqual(cfg.p_float, 13.13)
        self.assertEqual(mrg.p_float, 13.13)
        with self.assertRaises(AttributeError):
            mrg.does_not_exist

        self.assertEqual(set(mrg.groups[""]), set(["p_bool", "p_float", "p_int", "p_list", "p_str"]))
        self.assertEqual(set(mrg.groups["second"]), set(["second"]))

        self.assertEqual(mrg.get_fields(), list(fields(cfg)) + list(fields(cfg2)))

        cpy = mrg.copy()
        self.assertNotEqual(cfg, mrg)
        self.assertEqual(cpy, mrg)
        self.assertFalse(cpy is mrg)

        self.assertEqual(len((mrg | mrg)._children), 4)
        self.assertEqual(len((mrg | cfg)._children), 3)


class TestConfigurable(unittest.TestCase):

    def test_subclassing(self):
        @dataclass
        class Config(c.BaseConfig):
            section = "section_1"
            test: str = "start value"

        class MyConfigurable(c.Configurable):
            opt: Config

        cls = MyConfigurable()
        cls.WriteConfig()
        cls.opt.test = "changed value"
        self.assertEqual(cls.opt.test, "changed value")
        cls.ReadConfig()
        self.assertEqual(cls.opt.test, "start value")

    def test_missing_opt(self):
        class MyConfigurable(c.Configurable): ...

        with self.assertRaises(TypeError):
            MyConfigurable()
        with self.assertRaises(ValueError):
            MyConfigurable(config_class=str)

        @dataclass
        class Config(c.BaseConfig):
            section = "section_1"
            test: str = "start value"

        MyConfigurable(config_class=Config)
