"""
GenX main package.
"""
# make sure the models can import subpackages correctly
import os
import sys
from importlib import import_module
from importlib.abc import MetaPathFinder
from importlib.util import spec_from_file_location

_buildin_metafinder = list(sys.meta_path)
class GenxModuleFinder(MetaPathFinder):

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('genx'):
            return None
        itms=fullname.split('.')
        fname=os.path.join(os.path.dirname(os.path.abspath(__file__)), *itms)+'.py'
        pname=os.path.join(os.path.dirname(os.path.abspath(__file__)), *itms, '__init__.py')
        if os.path.exists(fname):
            m=import_module('genx.'+fullname)
            sys.modules[fullname]=m
            return spec_from_file_location(fullname, fname)
        if os.path.exists(pname):
            m=import_module('genx.'+fullname)
            sys.modules[fullname]=m
            return spec_from_file_location(fullname, pname)

sys.meta_path.insert(0, GenxModuleFinder())
