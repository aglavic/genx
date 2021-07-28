r"""
.. module:: parameters
   :synopsis: Classes to work with differnt materials in scattering experiments

"""

from os import path
import re
import pickle

import numpy as np
from scipy import interpolate

from .parameters import HasParameters, ComplexArray, Float, Calc
from .physical_constants import r_e
from . import scatteringlengths as _sl

# Transformations between AA and eV
_AA_to_eV=12398.42

_module_dir, _filename=path.split(__file__)
# Look only after the file name and not the ending since
# the file ending can be pyc if compiled...
_filename=_filename.split('.')[0]
_module_dir='.' if _module_dir=='' else _module_dir

_atomic_weights=_sl.load_atomic_weights_dabax(path.join(_module_dir, "../databases/AtomicWeights.dat"))

def create_fpdisp_lookup(path):
    """ Creates a lookup function to lookup element names and returns a function
    that yields a function of dispersive scattering factors f(E) at Q = 0. NOTE energy is in eV
    """

    def create_dispersion_func(name):
        e, f1, f2=np.loadtxt(path+'%s.nff'%name.lower(), skiprows=1, unpack=True)
        f1interp=interpolate.interp1d(e, f1, kind='linear')
        f2interp=interpolate.interp1d(e, f2, kind='linear')
        def_wl=_AA_to_eV/np.mean(e)

        @MemoizeF
        def f(wl=def_wl, **kwargs):
            return f1interp(_AA_to_eV/wl)-1.0J*f2interp(_AA_to_eV/wl)

        return Calc(f)

    return create_dispersion_func

f0=create_fpdisp_lookup(path.join(_module_dir, "../databases/f1f2_nist/"))

class MemoizeF:
    """Remember previous evaluations of function fn and stores it in a dictionary.

    Based on the Cookbook recipe: Memoizing (Caching) the Return Values of Functions in
    Python Cookbook by David Ascher, Alex Martelli.
    """

    def __init__(self, fn):
        self.fn=fn
        self.memo={}

    def __call__(self, wl=1.54, **kwds):
        pickled=pickle.dumps(wl, 1)
        if pickled not in self.memo:
            self.memo[pickled]=self.fn(wl=wl, **kwds)
        return self.memo[pickled]

class Material(HasParameters):
    """ Class to keep transform a chemical formula and mass density to scattering length

    The formula is given in ASCII format, for example 'Ag2(NO2)3' or 'HNO3', nested parentheses are allowed.
    Also non-integer indices are allowed for example Fe0.01Pd

    """
    validation_kwargs={'wl': 1.54}

    # The scattering length should be in AA-2
    # Need to transform g/cm3 to formula units per AA3
    # 1u = 1.66054×10−27 kg 1 AA=10−10m
    # density[kg/m3] = 1.66054*10^3*density[u/A3]
    # density [g/cm3] = density [kg/m3] * 1e-3/1e-2**3 = density [kg/m3] * 1e3
    # So formula units per AA^ = density [kg/m3]/1.66054*10^3/weight per formula unit
    # or formula units per AA^ = density [g/cm3]/1.66054/weight per formula unit

    def __init__(self, formula, **kwargs):
        self.sld_x=ComplexArray(0, unit="1/AA-2", help="Scattering length density")
        self.density=Float(0.0, unit="g/cm3", help="Weight density")
        self._formula_density=Float(0.0, unit="u", help="weight of one formula unit")
        HasParameters.__init__(self, **kwargs)
        self.formula_dict={}
        self.parse_formula(formula)

        # Create the composition variables needed to describe formula
        w_list=[]
        f0_list=[]
        for name in self.formula_dict:
            self.__setattr__(name, Float(self.formula_dict[name], help="Number of %s atoms"%name))
            element_count=self.__getattribute__(name)
            w_list.append(element_count*self.get_atomic_weight(name))
            f0_list.append(element_count*self.get_f0(name))
        self._formula_density=self.density/1.66054/sum(w_list)
        self.sld_x=self._formula_density*sum(f0_list)*r_e

    def get_atomic_weight(self, element):
        """Returns the atomic weight in unified atomic mass units (u)

        Parameters:
            element(string): name of the element

        """
        return _atomic_weights[element.lower()]

    def get_f0(self, element):
        """Returns the complex atomic forward scattering length normalised to the scattering length of one electron for
         element.

        Parameters:
            element(string): name of the element

        Returns:
            f0(function): function for f0 that takes the wavelength as a keyword argument
        """
        return f0(element)

    def parse_formula(self, formula, multiplier=1):
        """Parses the string formula and returns a list of isotopes

        Parameters:
            formula(string): A string containing the formula.
            multiplier(int): A number representing that will be used to multiply the element count
        """
        # regexp = re.compile(r"([A-Z][a-z]?)(\d*)|\((.*)\)(\d*)")
        regexp=re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)|\((.*)\)(\d*\.?\d*)")

        element, element_count, group, group_count=(1, 2, 3, 4)
        pos=0
        while pos<len(formula):
            match=regexp.match(formula, pos)
            if not match:
                raise ValueError("Error in formula: %s at position %d"%(formula, pos))
            pos=match.end()
            if match.group(element):
                # print "Match %s is an element" % match.group(0)
                count=self.formula_count(match.group(element_count), multiplier)
                try:
                    self.formula_dict[match.group(element)]+=count
                except KeyError:
                    self.formula_dict[match.group(element)]=count
            elif match.group(group):
                # print "Group found: %s, count: %s" % (match.group(group), match.group(group_count))
                count=self.formula_count(match.group(group_count), multiplier)
                self.parse_formula(match.group(group), multiplier=count)

    def formula_count(self, count_str, multiplier):
        """Handles the parsing result for the count of an formula part. Returns the count*multiplier"

        Parameters:
            count_str(string): The count of the element/group as a string
            multiplier(float): The  multiplier from previous groups.

        """
        if not count_str:
            # Set count to 1 if count is not exist, ie a single group is parsed
            count=1
        else:
            count=float(count_str)
        return count*multiplier
