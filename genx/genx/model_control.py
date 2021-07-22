"""
GenX model and optimizer control classes. All functional aspects should be covered here (no GUI).
"""
import os
import sys
import h5py
from abc import ABC, abstractmethod

from .exceptions import GenxIOError, GenxOptionError
from .model import Model
from .filehandling import config

class GenxOptimizer(ABC):
    """
    Defines an abstract base class for a optimizer for GenX models.
    DiffEv is implementing this abstraction.
    """
    @abstractmethod
    def pickle_string(self, clear_evals: bool=False):
        """ Return a pickle string for the object """

    @abstractmethod
    def pickle_load(self, pickled_string: str):
        """ Configure object from pickled copy """

    @abstractmethod
    def write_h5group(self, group: h5py.Group, clear_evals: bool=False):
        """ Save configuration to hdf5 group """

    @abstractmethod
    def read_h5group(self, group: h5py.Group):
        """ Configure object from hdf5 group """


class ModelController:
    def __init__(self, optimizer: GenxOptimizer):
        self.model=Model()
        self.optimizer=optimizer

    def save_file(self, fname: str):
        """
        Saves objects model, optimiser and config into file fnmame
        """
        if fname.endswith('.gx'):
            self.save_gx(fname)
        elif fname.endswith('.hgx'):
            self.save_hgx(fname)
        else:
            raise GenxIOError('Wrong file ending, should be .gx or .hgx')
        self.model.filename=os.path.abspath(fname)
        self.model.saved=True

    def load_file(self, fname: str):
        """
        Loads parameters from fname into model, optimizer and config
        """
        if fname.endswith('.gx'):
            self.load_gx(fname)
        elif fname.endswith('.hgx'):
            self.load_hgx(fname)
        else:
            raise GenxIOError('Wrong file ending, should be .gx or .hgx')
        self.model.filename=os.path.abspath(fname)

    def save_hgx(self, fname: str, group='current'):
        f=h5py.File(fname, 'w')
        g=f.create_group(group)
        self.model.write_h5group(g)
        try:
            clear_evals=not config.getboolean('solver', 'save all evals')
        except GenxOptionError as e:
            clear_evals=True
        self.optimizer.write_h5group(g.create_group('optimizer'), clear_evals=True)
        g['config']=config.model_dump().encode('utf-8')
        f.close()

    def load_hgx(self, fname: str, group='current'):
        f=h5py.File(fname, 'r')
        g=f[group]
        self.model.read_h5group(g)
        self.optimizer.read_h5group(g['optimizer'])
        config.load_string(g['config'][()].decode('utf-8'))
        f.close()

    def save_gx(self, fname: str):
        self.model.save(fname)
        self.model.save_addition('config', config.model_dump())
        self.model.save_addition('optimizer', self.optimizer.pickle_string(clear_evals=
                                        not config.getboolean('solver', 'save all evals')))

    def load_gx(self, fname: str):
        if not 'diffev' in sys.modules:
            # for compatibility define genx standard modules as base modules
            import genx.diffev
            import genx.data
            import genx.model
            sys.modules['model']=genx.model
            sys.modules['diffev']=genx.diffev
            sys.modules['data']=genx.data
        self.model.load(fname)
        config.load_string(self.model.load_addition('config').decode('utf-8'))
        self.optimizer.pickle_load(self.model.load_addition('optimizer'))
