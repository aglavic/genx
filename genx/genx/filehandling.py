'''I/O functions for GenX. 
These include loading of initilazation files.
Also included is the config object.
'''
import dataclasses
import io
import os, sys
from configparser import ConfigParser
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Type
from logging import debug

import h5py
from .gui_logging import iprint
from .exceptions import GenxIOError, GenxOptionError

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

    def load_string(self, input: str):
        '''
        Loads a config from a string input.  Raises an IOError if the string can not be
        read.
        '''
        self.model_config=ConfigParser()
        try:
            self.model_config.read_string(input)
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
                return fallback

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
        str=buffer.getvalue()
        buffer.close()
        return str

# create the global GenX config object used by the program, it does not depend on anything else
config=Config()

class BaseConfig(ABC):
    """
    Base class for handling configuration that shall be stored by a GenX Config object.

    Handles general parameter read/write so that derived classes only have to
    inherit from it and use dataclass attributes.
    """
    @property
    @abstractmethod
    def section(self):
        raise NotImplementedError("Subclass needs to define the section attribute.")

    def asdict(self):
        return dataclasses.asdict(self)

    def load_config(self):
        field: dataclasses.Field
        for field in dataclasses.fields(self):
            if field.type is float:
                getf=config.getfloat
            elif field.type is int:
                getf=config.getint
            elif field.type is bool:
                getf=config.getboolean
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
                iprint(f'Could not read option {self.section}/{option} of type {field.type}:\n    {e}')
            else:
                setattr(self, field.name, value)
                debug(f'Read option {field.name}={value} from config {self.section}/{option}.')

    def safe_config(self, default=False):
        data=self.asdict()
        if default:
            setter=config.default_set
        else:
            setter=config.model_set
        for key, value in data.items():
            option=key.replace('_', ' ')
            debug(f'Write option {key}={value} to config {self.section}/{option}, default={default}.')
            setter(self.section, option, value)

    def copy(self):
        return dataclasses.replace(self)

class Configurable:
    """
    A mixin class for classes that store parameters in a savable configuration.

    Defines methods for reading and writing configurations. Subclasses can
    define the config_updated method to be called after changing a configuration.
    """
    def __init__(self, config_class: Type[BaseConfig]):
        self.opt=config_class()

    def ReadConfig(self):
        """ Reads the variables stored in the config file."""
        self.opt.load_config()
        self.UpdateConfigValues()

    def WriteConfig(self):
        """Writes the varaibles to be stored to the config"""
        self.opt.safe_config()

    def UpdateConfigValues(self):
        """
        Sub-classes can overwrite this method to preform some action after
        a configuration has been loaded.
        """



def save_file(fname: str, model, optimizer, config: Config):
    """
    Saves objects model, optimiser and config into file fnmame
    """
    if fname.endswith('.gx'):
        save_gx(fname, model, optimizer, config)
    elif fname.endswith('.hgx'):
        save_hgx(fname, model, optimizer, config)
    else:
        raise GenxIOError('Wrong file ending, should be .gx or .hgx')

    model.filename=os.path.abspath(fname)
    model.saved=True

def load_file(fname: str, model, optimizer, config: Config):
    """Loads parameters from fname into model, optimizer and config"""
    if fname.endswith('.gx'):
        load_gx(fname, model, optimizer, config)
    elif fname.endswith('.hgx'):
        load_hgx(fname, model, optimizer, config)
    else:
        raise GenxIOError('Wrong file ending, should be .gx or .hgx')

    model.filename=os.path.abspath(fname)

def save_gx(fname: str, model, optimizer, config: Config):
    model.save(fname)
    model.save_addition('config', config.model_dump())
    model.save_addition('optimizer',
                        optimizer.pickle_string(clear_evals=
                                                not config.getboolean('solver',
                                                                       'save all evals')))

def save_hgx(fname: str, model, optimizer, config: Config, group='current'):
    """ Saves the current objects to a hdf gx file (hgx).

    :param fname: filename
    :param model: model object
    :param optimizer: optimizer object
    :param config: config object
    :param group: name of the group, default current
    :return:
    """
    f=h5py.File(fname, 'w')
    g=f.create_group(group)
    model.write_h5group(g)
    try:
        clear_evals=not config.getboolean('solver', 'save all evals')
    except GenxOptionError as e:
        clear_evals=True
    optimizer.write_h5group(g.create_group('optimizer'), clear_evals=True)
    g['config']=config.model_dump().encode('utf-8')
    f.close()

def load_hgx(fname: str, model, optimizer, config: Config, group='current'):
    """ Loads the current objects to a hdf gx file (hgx).

    :param fname: filename
    :param model: model object
    :param optimizer: optimizer object
    :param config: config object
    :param group: name of the group, default current
    :return:
    """
    f=h5py.File(fname, 'r')
    g=f[group]
    model.read_h5group(g)
    optimizer.read_h5group(g['optimizer'])
    config.load_string(g['config'][()].decode('utf-8'))
    f.close()

def load_gx(fname: str, model, optimizer, config: Config):
    if not 'diffev' in sys.modules:
        # for compatibility define genx standard modules as base modules
        import genx.diffev
        import genx.data
        import genx.model
        sys.modules['model']=genx.model
        sys.modules['diffev']=genx.diffev
        sys.modules['data']=genx.data
    model.load(fname)
    config.load_string(model.load_addition('config').decode('utf-8'))
    optimizer.pickle_load(model.load_addition('optimizer'))

# Functions to handle optimiser configs
# ==============================================================================


def load_opt_config(optimizer, config: Config):
    """Load the config (Config class) values to the optimiser class (DiffEv class)."""

    class Container:
        error_bars_level=1.05
        save_all_evals=False

        def set_error_bars_level(self, val):
            self.error_bars_level=val

        def set_save_all_evals(self, val):
            self.save_all_evals=val

    c=Container()

    # Define all the options we want to set
    options_float=['km', 'kr', 'pop mult', 'pop size',
                   'max generations', 'max generation mult',
                   'sleep time', 'max log elements',
                   'errorbar level',
                   'autosave interval', 'parallel processes',
                   'parallel chunksize', 'allowed fom discrepancy']
    setfunctions_float=[optimizer.set_km, optimizer.set_kr,
                        optimizer.set_pop_mult,
                        optimizer.set_pop_size,
                        optimizer.set_max_generations,
                        optimizer.set_max_generation_mult,
                        optimizer.set_sleep_time,
                        optimizer.set_max_log,
                        c.set_error_bars_level,
                        optimizer.set_autosave_interval,
                        optimizer.set_processes,
                        optimizer.set_chunksize,
                        optimizer.set_fom_allowed_dis,
                        ]

    options_bool=['use pop mult', 'use max generations',
                  'use start guess', 'use boundaries',
                  'use parallel processing', 'use autosave',
                  'save all evals'
                  ]
    setfunctions_bool=[optimizer.set_use_pop_mult,
                       optimizer.set_use_max_generations,
                       optimizer.set_use_start_guess,
                       optimizer.set_use_boundaries,
                       optimizer.set_use_parallel_processing,
                       optimizer.set_use_autosave,
                       c.set_save_all_evals,
                       ]

    # Make sure that the config is set
    if config:
        # Start witht the float values
        for index in range(len(options_float)):
            try:
                val=config.getfloat('solver', options_float[index])
            except GenxOptionError as e:
                iprint('Could not locate option solver.'+options_float[index])
            else:
                setfunctions_float[index](val)

        # Then the bool flags
        for index in range(len(options_bool)):
            try:
                val=config.getboolean('solver', options_bool[index])
            except GenxOptionError as e:
                iprint('Could not read option solver.'+options_bool[index])
            else:
                setfunctions_bool[index](val)
        try:
            val=config.get('solver', 'create trial')
        except GenxOptionError as e:
            iprint('Could not read option solver.create trial')
        else:
            try:
                optimizer.set_create_trial(val)
            except LookupError:
                iprint('The mutation scheme %s does not exist'%val)

    return c.error_bars_level, c.save_all_evals

def save_opt_config(optimizer, config: Config, fom_error_bars_level=1.05, save_all_evals=False):
    """ Write the config values from optimizer (DiffEv class) to config (Config class) """

    # Define all the options we want to set
    options_float=['km', 'kr', 'pop mult', 'pop size',
                   'max generations', 'max generation mult',
                   'sleep time', 'max log elements', 'errorbar level',
                   'autosave interval',
                   'parallel processes', 'parallel chunksize',
                   'allowed fom discrepancy']
    set_float=[optimizer.km, optimizer.kr,
               optimizer.pop_mult,
               optimizer.pop_size,
               optimizer.max_generations,
               optimizer.max_generation_mult,
               optimizer.sleep_time,
               optimizer.max_log,
               fom_error_bars_level,
               optimizer.autosave_interval,
               optimizer.processes,
               optimizer.chunksize,
               optimizer.fom_allowed_dis
               ]

    options_bool=['use pop mult', 'use max generations',
                  'use start guess', 'use boundaries',
                  'use parallel processing', 'use autosave',
                  'save all evals',
                  ]
    set_bool=[optimizer.use_pop_mult,
              optimizer.use_max_generations,
              optimizer.use_start_guess,
              optimizer.use_boundaries,
              optimizer.use_parallel_processing,
              optimizer.use_autosave,
              save_all_evals,
              ]

    # Make sure that the config is set
    if config:
        # Start witht the float values
        for index in range(len(options_float)):
            try:
                config.set('solver', options_float[index], set_float[index])
            except GenxOptionError as e:
                iprint('Could not locate save solver.'+options_float[index])

        # Then the bool flags
        for index in range(len(options_bool)):
            try:
                config.set('solver', options_bool[index], set_bool[index])
            except GenxOptionError as e:
                iprint('Could not write option solver.'+options_bool[index])

        try:
            config.set('solver', 'create trial', optimizer.get_create_trial())
        except GenxOptionError as e:
            iprint('Could not write option solver.create trial')

