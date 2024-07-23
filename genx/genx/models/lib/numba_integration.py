import os
import sys
import appdirs
import numba.core.caching as nc

from logging import debug
from ...version import __version__ as program_version

config_path = os.path.abspath(appdirs.user_data_dir('GenX3', 'ArturGlavic'))

class _GenxCacheLocator(nc._SourceFileBackedLocatorMixin, nc._CacheLocator):
    """
    A locator that points to the GenX specific numba cache directory.
    It dempents on the genx version but not the location of the source files.
    """

    def __init__(self, py_func, py_file):
        self._py_file = py_file
        self._lineno = py_func.__code__.co_firstlineno
        cache_subpath = self.get_suitable_cache_subpath(py_file)
        base_dir = os.environ.get('NUMBA_CACHE_DIR', os.path.join(config_path, 'numba_cache'))
        self._cache_path = os.path.join(base_dir, cache_subpath)
        debug(f'numba cache path for {py_func} from {py_file} is {self._cache_path}')
        debug(f'    source stamp is {self.get_source_stamp()}')

    def get_source_stamp(self):
        # overwrite numba behavior to make sure the actual file times are used and not the executable
        st = os.stat(self._py_file)
        return st.st_mtime, st.st_size

    def get_cache_path(self):
        return self._cache_path

    @classmethod
    def from_function(cls, py_func, py_file):
        parent = super(_GenxCacheLocator, cls)
        if py_file.startswith(os.path.join('genx', 'models', 'lib')):
            from genx.models import lib
            base_dir = os.path.dirname(os.path.abspath(lib.__file__))
            py_file = os.path.join(base_dir, os.path.basename(py_file))
        output = parent.from_function(py_func, py_file)
        if output is None:
            debug(f'no locator created for file {py_file}, exists={os.path.exists(py_file)}')
            self = _GenxCacheLocator(py_func, py_file)
            try:
                self.ensure_cache_path()
            except OSError:
                debug('OSError when checkint cache path', exc_info=True)
            # Perhaps a placeholder (e.g. "<ipython-XXX>")
            return
        return output

    @classmethod
    def get_suitable_cache_subpath(cls, py_file):
        path = os.path.abspath(py_file)
        subpath = os.path.dirname(path)
        parentdir = os.path.split(subpath)[-1]
        # separate caches for source and binary distribution
        # mostly if source is used to test on the same machine
        if getattr(sys, 'frozen', False):
            prefix = 'genxsource'
        else:
            prefix = 'genxfrozen'
        return f'{prefix}_{program_version.replace(".", "_")}_{parentdir}'


def configure_numba():
    if hasattr(nc, '_CacheImpl'):
        nc._CacheImpl._locator_classes.insert(0, _GenxCacheLocator)
    else:
        # Newer version of numba
        nc.CacheImpl._locator_classes.insert(0, _GenxCacheLocator)
