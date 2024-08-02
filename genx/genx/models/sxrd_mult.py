"""
Library for surface x-ray diffraction simulations of superlattices
==================================================================

The model is based on Fullertons algorithm for superlattices as
described in Phys. Rev. B vol. 45 p. 9292 (1992). 
"""

# Programmed by Matts Bjorck 20091215

import time

import numpy as np

from . import sxrd
from .lib.physical_constants import r_e
from .sxrd import AtomGroup, Instrument, Slab, SymTrans, UnitCell
from .utils import f, rho

__pars__ = ["SLSample", "SLStandard", "UnitCell", "Slab", "AtomGroup", "Instrument"]

__xlabel__ = "q-scan [r.l.u.]"
__ylabel__ = "Instnsity [a.u.]"


class SLSample:
    """Class that models a multilayer sample on top of a
    substrate according to Fullertons model as given in
    PRB ....
    """

    def __init__(self, inst, bulk_slab, superlattice, unitcell, bulk_sym=None):
        if bulk_sym is None:
            bulk_sym = []
        self.set_bulk_slab(bulk_slab)
        self.set_bulk_sym(bulk_sym)
        self.superlattice = superlattice
        self.inst = inst
        self.set_unit_cell(unitcell)

    def set_bulk_slab(self, bulk_slab):
        """Set the bulk unit cell to bulk_slab"""
        if type(bulk_slab) != type(sxrd.Slab()):
            raise TypeError("The bulk slab has to be a member of" " class Slab")
        self.bulk = bulk_slab

    def set_unit_cell(self, unit_cell):
        """Sets the unitcell of the sample"""
        if type(unit_cell) != type(sxrd.UnitCell(1.0, 1.0, 1.0)):
            raise TypeError("The bulk slab has to be a member" " of class UnitCell")
        if unit_cell is None:
            unit_cell = sxrd.UnitCell(1.0, 1, 0.0, 1.0)
        self.unit_cell = unit_cell

    def set_bulk_sym(self, sym_list):
        """Sets the list of allowed symmetry operations for the bulk

        sym_list has to be a list ([]) of symmetry elements from the
        class SymTrans
        """
        # Type checking
        if type(sym_list) != type([]):
            raise TypeError("The surface symmetries has to contained" " in a list")

        if not sym_list:
            sym_list = [sxrd.SymTrans()]

        if min([type(sym) == type(sxrd.SymTrans()) for sym in sym_list]) == 0:
            raise TypeError("All members in the symmetry list has to" " be a memeber of class SymTrans")

        self.bulk_sym = sym_list

    def calc_i(self, h, k, l):
        """Calculate the diffracted intensity from a superlattice.
        The diffracted intensity from the superlattice and the substrate
        are added. I.e. it is assumed that the films is not coherent with
        the substrate.
        """
        bulk_i = np.abs(self.calc_fb(h, k, l)) ** 2
        sl_i = np.abs(self.superlattice.calc_i(h, k, l))
        return (bulk_i + sl_i) * self.inst.inten

    def calc_fb(self, h, k, l):
        """Calculate the structure factors from the bulk"""
        dinv = self.unit_cell.abs_hkl(h, k, l)
        x, y, z, el, u, oc, c = self.bulk._extract_values()
        oc = oc / float(len(self.bulk_sym))
        f = sxrd._get_f(self.inst, el, dinv)
        # Calculate the "shape factor" for the CTRs
        eff_thick = self.unit_cell.c / np.sin(self.inst.alpha * np.pi / 180.0)
        alpha = r_e * self.inst.wavel * eff_thick / self.unit_cell.vol() * np.sum(f.imag, 1)
        denom = np.exp(2.0 * np.pi * 1.0j * l) * np.exp(-alpha) - 1.0
        # Delta functions to remove finite size effect in hk plane
        delta_funcs = (abs(h - np.round(h)) < 1e-12) * (abs(k - np.round(k)) < 1e-12)
        # Sum up the uc struct factors
        f_u = np.sum(
            oc
            * f
            * np.exp(-2 * np.pi**2 * u * dinv[:, np.newaxis] ** 2)
            * np.sum(
                [
                    np.exp(
                        2.0
                        * np.pi
                        * 1.0j
                        * (
                            h[:, np.newaxis] * sym_op.trans_x(x, y)
                            + k[:, np.newaxis] * sym_op.trans_y(x, y)
                            + l[:, np.newaxis] * z[np.newaxis, :]
                        )
                    )
                    for sym_op in self.bulk_sym
                ],
                0,
            ),
            1,
        )
        # Putting it all togheter
        fb = f_u / denom * delta_funcs

        return fb


class Superlattice:
    """Class that describe a superlattice, can be subclassed
    to implement different strain profiles, interdiffusion etc..
    """

    def __init__(self, inst, unit_cell, a_slab, b_slab, a_sym=None, b_sym=None):
        if a_sym is None:
            a_sym = []
        if b_sym is None:
            b_sym = []
        self.a_slab = a_slab
        self.b_slab = b_slab
        if not a_sym:
            self.a_sym = [sxrd.SymTrans()]
        else:
            self.a_sym = a_sym
        if not b_sym:
            self.b_sym = [sxrd.SymTrans()]
        else:
            self.b_sym = b_sym
        self.unit_cell = unit_cell
        self.inst = inst

    def _extract_slab_values(self, slabs, sym):
        """Extracts the necessary parameters for simulating
        a list of stacked slabs
        """
        # Extract the parameters we need
        # the star in zip(*... transform the list elements to arguments
        xt, yt, zt, elt, ut, oct, ct = list(zip(*[slab._extract_values() for slab in slabs]))

        x = np.r_[xt]
        y = np.r_[yt]
        # scale and shift the slabs with respect to each other
        cn = np.cumsum(np.r_[0, ct])[:-1]
        z = np.concatenate([zs * c_s + c_cum for zs, c_cum, c_s in zip(zt, cn, ct)])
        # el = reduce(lambda x,y:x+y, elt)
        el = np.r_[elt]
        u = np.r_[ut]
        oc = np.r_[oct]
        # print x,y,z, u
        t_lay = sum(ct)
        return x, y, z, u, oc, el, t_lay

    def calc_fslab(self, slablist, sym, h, k, l):
        """Calculate the structure factors from the bulk"""
        dinv = self.unit_cell.abs_hkl(h, k, l)
        x, y, z, u, oc, el, t_lay = self._extract_slab_values(slablist, sym)
        oc = oc / float(len(sym))
        f = sxrd._get_f(self.inst, el, dinv)
        # Sum up the uc struct factors
        f_u = np.sum(
            oc
            * f
            * np.exp(-2 * np.pi**2 * u * dinv[:, np.newaxis] ** 2)
            * np.sum(
                [
                    np.exp(
                        2.0
                        * np.pi
                        * 1.0j
                        * (
                            h[:, np.newaxis] * sym_op.trans_x(x, y)
                            + k[:, np.newaxis] * sym_op.trans_y(x, y)
                            + l[:, np.newaxis] * z[np.newaxis, :]
                        )
                    )
                    for sym_op in sym
                ],
                0,
            ),
            1,
        )
        # return f_u, (z.max() - z.min())*np.ones(l.shape)
        return f_u, t_lay * np.ones(l.shape)

    def calc_fa(self, n, h, k, l):
        """Calculate the strucutre factor for a a layer
        n is the thickness of the bilayer in units of slabs"""
        pass

    def calc_fb(self, n, h, k, l):
        """Calcualte the structure factor for a b layer
        n is the thickness of the bilayer in units of slabs"""
        pass

    def calc_fsl(self, unit_cell, h, k, l):
        """Calculate the strucutre factor for the entire
        superlattice.
        """
        raise NotImplementedError("calc_fsl has to be implemented in " "a Superlattices subclass")


class SLStandard(Superlattice):
    """Class that implements a "standard" superlattice, no strain
    included.
    """

    _pars = {"sigmaa": 1e-12, "sigmab": 1e-12, "repetitions": 2, "na": 2, "nb": 2, "a": 0.0, "c": 1e-12}

    def __init__(self, inst, unit_cell, a_slab, b_slab, a_sym=None, b_sym=None):
        Superlattice.__init__(self, inst, unit_cell, a_slab, b_slab, a_sym=a_sym, b_sym=b_sym)
        if a_sym is None:
            a_sym = []
        if b_sym is None:
            b_sym = []
        [self._make_set_func(name, self._pars[name]) for name in list(self._pars.keys())]
        [self._make_get_func(name) for name in list(self._pars.keys())]

    def calc_fa(self, n, h, k, l):
        f_slab, t_z = self.calc_fslab([self.a_slab] * n, self.a_sym, h, k, l)
        return f_slab, t_z

    def calc_fb(self, n, h, k, l):
        f_slab, t_z = self.calc_fslab([self.b_slab] * n, self.b_sym, h, k, l)
        return f_slab, t_z

    def thick_prob(self, n_mean, stand_dev):
        # According to fullerton it's enough to include three
        # standard deviations in the averaging
        lower = np.floor(n_mean - 3.0 * stand_dev)
        lower = np.int(min(lower, n_mean - 1))
        # We can't have a negative thickness, altough we remove
        # the gaussian distribution How does this affect the theoretical
        # assumptions?
        if lower < 1:
            lower = 1
        upper = np.ceil(n_mean + 3.0 * stand_dev)
        n = np.arange(lower, np.int(max(upper, n_mean + 1) + 1))
        # print 'n: ', n
        prob = np.exp(-((n - n_mean) ** 2) / 2.0 / stand_dev**2)
        prob = prob / sum(prob)
        return n, prob

    def calc_i(self, h, k, l):
        """Function to calculate the form factor from a superlattice"""
        # Create the different thicknesses to avarage over
        na, pa = self.thick_prob(self.na, self.sigmaa)
        nb, pb = self.thick_prob(self.nb, self.sigmab)
        tmp = np.array([self.calc_fa(n, h, k, l) for n in na])
        fa = tmp[:, 0, :]
        ta = tmp[:, 1, :]
        tmp = np.array([self.calc_fb(n, h, k, l) for n in nb])
        fb = tmp[:, 0, :]
        tb = tmp[:, 1, :]
        # print pa.shape, fa.shape
        pa = pa[:, np.newaxis]
        pb = pb[:, np.newaxis]
        # Do the different averagning
        fafa = (pa * fa * fa.conj()).sum(0)
        fbfb = (pb * fb * fb.conj()).sum(0)
        fam = (pa * fa).sum(0)
        fbm = (pb * fb).sum(0)
        phia = (pa * np.exp(2 * np.pi * 1.0j * ta * l) * fa.conj()).sum(0)
        phib = (pb * np.exp(2 * np.pi * 1.0j * tb * l) * fb.conj()).sum(0)
        ta = (pa * np.exp(2 * np.pi * 1.0j * ta * l)).sum(0)
        tb = (pb * np.exp(2 * np.pi * 1.0j * tb * l)).sum(0)
        m = self.repetitions
        ksi = 2 * np.pi * 1.0j * l * self.a - (2 * np.pi * l * self.c) ** 2 / 2.0
        # Calculate the intensity
        int = m * (fafa + 2.0 * np.real(np.exp(ksi) * phia * fbm) + fbfb) + 2.0 * np.real(
            (np.exp(-ksi) * phib * fam / ta / tb + phia * fam / ta + phib * fbm / tb + np.exp(ksi) * phia * fbm)
            * (
                (m - (m + 1) * np.exp(2.0 * ksi) * ta * tb + (np.exp(2.0 * ksi) * ta * tb) ** (m + 1))
                / (1 - np.exp(2.0 * ksi) * ta * tb) ** 2
                - m
            )
        )
        return int

    def _make_set_func(self, name, val):
        """Creates a function to set value for attribute with name"""

        def set_func(value):
            setattr(self, name, value)

        # Init the variable
        set_func(val)
        setattr(self, "set" + name, set_func)

    def _make_get_func(self, name):
        """Creates a get function"""

        def get_func():
            return getattr(self, name)

        setattr(self, "get" + name, get_func)


if __name__ == "__main__":
    from pylab import *

    inst = sxrd.Instrument(wavel=0.77, alpha=0.2)
    inst.set_inten(100.0)

    lay_a = sxrd.Slab()
    lay_a.add_atom("Sr", "sr", 0.0, 0.0, 0.0, 0.001, 1.0)
    lay_a.add_atom("Ti", "ti", 0.5, 0.5, 0.5, 0.001, 1.0)
    lay_a.add_atom("O1", "o", 0.5, 0.0, 0.5, 0.001, 1.0)
    lay_a.add_atom("O2", "o", 0.0, 0.5, 0.5, 0.001, 1.0)
    lay_a.add_atom("O3", "o", 0.5, 0.5, 0.0, 0.001, 1.0)

    lay_b = sxrd.Slab(c=1.0)
    lay_b.add_atom("La", "la", 0.0, 0.0, 0.0, 0.001, 1.0, 1.0)
    lay_b.add_atom("Al", "al", 0.5, 0.5, 0.5, 0.001, 1.0, 1.0)
    lay_b.add_atom("O1", "o", 0.5, 0.5, 0.0, 0.001, 1.0, 1.0)
    lay_b.add_atom("O2", "o", 0.0, 0.5, 0.5, 0.001, 1.0, 1.0)
    lay_b.add_atom("O3", "o", 0.5, 0.0, 0.5, 0.001, 1.0, 1.0)

    uc = sxrd.UnitCell(3.945, 3.945, 3.945, 90, 90, 90)

    sl = SLStandard(inst, uc, lay_b, lay_a)
    sl_sample = SLSample(inst, lay_a, sl, uc)
    sl.seta(0.0)
    sl.setc(0.00001)
    sl.setna(4.0)
    # Seems to have a lower limit of about 0.3 UC to work fine
    # with the calculation of thicknesses..
    sl.setsigmaa(0.3)
    sl.setnb(2.0)
    sl.setsigmab(0.3)
    sl.setrepetitions(10)
    l = np.arange(0.1, 3, 0.0011)
    h = 0.0 * np.ones(l.shape)
    k = 0.0 * np.ones(l.shape)
    int = sl_sample.calc_i(h, k, l)

    sample = sxrd.Sample(inst, lay_a, ([lay_b] * 4 + [lay_a] * 2) * 10, sxrd.UnitCell(3.945, 3.945, 3.945, 90, 90, 90))
    f_ref = sample.calc_f(h, k, l)
    int_ref = abs(f_ref) ** 2
    # Comparison between the normal sxrd model and the superlattice model.
    semilogy(l, int_ref)
    semilogy(l, int)
    legend(("sxrd model", "sxrd_mult closed form"))
    xlabel("l [r.l.u.]")
    ylabel("Intensity")
    show()
