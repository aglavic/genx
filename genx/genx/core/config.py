'''
Configuration and config file handling.
'''
import dataclasses
import io
from configparser import ConfigParser
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Type, get_type_hints, Dict, List
from logging import debug

from .custom_logging import iprint
from ..exceptions import GenxIOError, GenxOptionError

# ==============================================================================
class Config:
    """
    GenX configuration handler.
    """
    def __init__(self):
        self.default_config=ConfigParser()
        self.model_config=ConfigParser()

    def load_default(self, filename, reset=False):
        '''
        Loads the default config from file filename. Raises a IOError if the
        can not be found.
        '''
        try:
            self.default_config.read(filename)
        except Exception as e:
            iprint(e)
            raise GenxIOError(f'Could not load default config file:\n {e}', filename)
        if reset:
            self.model_config=ConfigParser()

    def write_default(self, filename):
        '''
        Writes the current default configuration to filename
        '''
        try:
            with open(filename, 'w') as cfile:
                self.default_config.write(cfile)
        except Exception as e:
            iprint(e)
            raise GenxIOError(f'Could not write default config file:\n {e}', filename)

    def load_string(self, string: str):
        '''
        Loads a config from a string input.  Raises an IOError if the string can not be
        read.
        '''
        self.model_config=ConfigParser()
        try:
            self.model_config.read_string(string)
        except Exception as e:
            raise GenxIOError(f'Could not load model config file:\n {e}')

    def _getf(self, method: str, section: str, option: str, fallback=None):
        '''
        For the function function try to locate the section and option first in
        model_config if that fails try to locate it in default_config. If both
        fails raise a GenxOptionError.
        '''
        if self.model_config.has_option(section, option):
            func=getattr(self.model_config, method)
        elif self.default_config.has_option(section, option):
            func=getattr(self.default_config, method)
        else:
            raise GenxOptionError(section, option)
        try:
            return func(section, option, fallback=fallback)
        except ValueError:
            if fallback is None:
                raise GenxOptionError(section, option)
            else:
                debug(f'Return fallback due to ValueError in _getf, section={section}/option={option}')
                return fallback

    def getlist(self, section: str, option: str, fallback=None):
        str_value=self.get(section, option, fallback=fallback)
        return [s.strip() for s in str_value.split(';')]

    @lru_cache(maxsize=10)
    def __getattr__(self, item: str):
        """
        Handle all get functions from ConfigParser to be redirected through _getf,
        everything else is treated normally.
        The lambda functions are cached so they only get generated once.
        """
        if item.startswith('get') and hasattr(self.model_config, item):
            return lambda section, option, fallback=None: self._getf(item, section, option, fallback=fallback)
        else:
            raise AttributeError(f"'Config' object has no attribute '{item}'")

    def model_set(self, section, option, value):
        '''
        Set a value in section, option for the model configuration.
        '''
        if not self.model_config.has_section(section):
            self.model_config.add_section(section)
        self.model_config.set(section, option, str(value))

    def default_set(self, section, option, value):
        '''
        Set a value in section, option for the model configuration.
        '''
        if not self.default_config.has_section(section):
            self.default_config.add_section(section)
        self.default_config.set(section, option, str(value))

    def set(self, section, option, value):
        '''
        Set a value in section, option for the model configuration. 
        Just a duplication of model_set.
        '''
        self.model_set(section, option, value)

    def model_dump(self):
        '''
        dumps the model configuration to a string.
        '''
        buffer=io.StringIO()
        self.model_config.write(buffer)
        string=buffer.getvalue()
        buffer.close()
        return string

# create the global GenX config object used by the program, it does not depend on anything else
config=Config()

class BaseConfig(ABC):
    """
    Base class for handling configuration that shall be stored by a GenX Config object.

    Handles general parameter read/write so that derived classes only have to
    inherit from it and use dataclass attributes.
    """
    section:str

    @property
    @abstractmethod
    def section(self):
        raise NotImplementedError("Subclass needs to define the section attribute.")

    def asdict(self):
        # noinspection PyDataclass
        return dataclasses.asdict(self)

    def load_config(self):
        field: dataclasses.Field
        # noinspection PyDataclass
        for field in dataclasses.fields(self):
            if field.type is float:
                getf=config.getfloat
            elif field.type is int:
                getf=config.getint
            elif field.type is bool:
                getf=config.getboolean
            elif field.type is list:
                getf=config.getlist
            else:
                getf=config.get

            if field.default is dataclasses.MISSING:
                fallback=None
            else:
                fallback=field.default
            option=field.name.replace('_', ' ')
            try:
                value=getf(self.section, option, fallback=fallback)
            except GenxOptionError as e:
                debug(f'Could not read option {self.section}/{option} of type {field.type.__name__}:\n    {e}')
            else:
                setattr(self, field.name, value)

    def safe_config(self, default=False):
        data=self.asdict()
        if default:
            setter=config.default_set
        else:
            setter=config.model_set
        for key, value in data.items():
            option=key.replace('_', ' ')
            if type(value) is list:
                value=';'.join(value)
            setter(self.section, option, value)

    def copy(self):
        # noinspection PyDataclass
        return dataclasses.replace(self)

    @property
    def groups(self)->Dict[str,List[str]]:
        """
        Can be replaced by dictionary in sub-classes to define how to group parameters in display and
        configuration dialog entries.
        If a list entry contains a list of strings it is used to group the values horizontally.
        A list of 4 items with the second beeing a boolean is interpreted as a choice between two parameters:
            [{Choice Group Label}, bool-first is active, first value, second value]
        """
        res=list(self.asdict().keys())
        return {'': res}

    def get_fields(self, fltr=None)->List[dataclasses.Field]:
        """
        Return the fields for the dataclass.
        If fltr is supplied the results will be filtered by field names given in this list.
        """
        # noinspection PyDataclass
        fields=dataclasses.fields(self) # child classes will be dataclass objects
        if fltr is None:
            return list(fields)
        else:
            out=[]
            for fi in fields:
                if fi.name in fltr:
                    out.append(fi)
            return out

    @staticmethod
    def GParam(default, pmin=None, pmax=None, label=None, descriptoin=None):
        # allow to add some metadata to the parameter to use in dialog generation
        gmeta={}
        if pmin is not None:
            gmeta['pmin']=pmin
        if pmax is not None:
            gmeta['pmax']=pmax
        if label is not None:
            gmeta['label']=label
        if descriptoin is not None:
            gmeta['descriptoin']=descriptoin
        return dataclasses.field(default=default, metadata={'genx': gmeta})

    @staticmethod
    def GChoice(default, selection, label=None, descriptoin=None):
        """
        Allow to add some metadata to the parameter to use in dialog generation,
        selection is a list of alloweed string values for this parameter.
        """
        gmeta={'selection': selection}
        if label is not None:
            gmeta['label']=label
        if descriptoin is not None:
            gmeta['descriptoin']=descriptoin
        return dataclasses.field(default=default, metadata={'genx': gmeta})

    def __or__(self, other):
        return MergedConfig(self, other)

class MergedConfig(BaseConfig):
    """
    A config-like object that combines two or more config objects.

    Attribut access is passed to the child objects.
    """
    section = None
    _children: List[BaseConfig]

    def __init__(self, *children):
        self._children=list(children)

    def __or__(self, other):
        all_children=self._children+[other]
        return MergedConfig(*all_children)

    def get_fields(self, fltr=None) ->List[dataclasses.Field]:
        fout=[]
        for c in self._children:
            fout+=c.get_fields(fltr)
        return fout

    def copy(self):
        clist=[c.copy() for c in self._children]
        return MergedConfig(*clist)

    @property
    def groups(self) ->Dict[str,List[str]]:
        dout={}
        for c in self._children:
            dout.update(c.groups)
        return dout

    def __getattr__(self, name):
        # search through the child configuration fields if name is in there and return value
        for c in self._children:
            fnames=[fi.name for fi in c.get_fields()]
            if name in fnames:
                return getattr(c, name)
        raise AttributeError("{item} not found in any child configuration")

    def __setattr__(self, name, value):
        if name.startswith('_'):
            BaseConfig.__setattr__(self, name, value)
        # search through the child configuration fields if name is in there, otherwise set own attribute
        for c in self._children:
            fnames=[fi.name for fi in c.get_fields()]
            if name in fnames:
                return setattr(c, name, value)
        BaseConfig.__setattr__(self, name, value)


class Configurable:
    """
    A mixin class for classes that store parameters in a savable configuration.

    Defines methods for reading and writing configurations. Subclasses can
    define the config_updated method to be called after reading a configuration.

    If no config_class is passed to the constructor explicitly, any existing type hint
    to a BaseConfig derived object will be used to configure the opt attribute.
    """
    def __init__(self, config_class: Type[BaseConfig]=None):
        if config_class is None:
            hints=get_type_hints(self)
            config_class=hints.get('opt', None)

        if not issubclass(config_class, BaseConfig):
            raise ValueError("Configurable needs either an explicit config_class "
                             "or 'opt' type hint derived from BaseConfig")

        self.opt=config_class()

    def ReadConfig(self):
        """ Reads the variables stored in the config file."""
        self.opt.load_config()
        self.UpdateConfigValues()

    def WriteConfig(self, default=False):
        """Writes the varaibles to be stored to the config"""
        self.opt.safe_config(default=default)

    def UpdateConfigValues(self):
        """
        Sub-classes can overwrite this method to preform some action after
        a configuration has been loaded.
        """
