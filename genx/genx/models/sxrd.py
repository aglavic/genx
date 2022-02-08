'''
Library for surface x-ray diffraction simulations
=================================================
The problem of modelling the sample is divided to four different
classes: Sample, Slab, UnitCell and Instrument. A Slab is the basic unit
that builds up a sample and can be seen as a quasi-unitcell for the sxrd
problem. Stricitly it is a 2D unitcell with a finite extension
out-of-plane. The Sample is then built from these Slabs one slab for the
bulk and a list of slabs for the surface structure.

The unitcell consists of parameters for the unitcell and the instrument
contains instrument variables. See below for a full list.

Classes
-------

Slab
~~~~
``Slab(c = 1.0, slab_oc = 1.0)``

``c``
   A scale factor for ou-of-plane extension of the Slab. All z-positions
   will be scaled with this factor.
``slab_oc``
   A global scaling of the occupancy of all atoms in the slab.

``[Slab].add_atom(id, el, x, y, z, u = 0, oc = 1.0, m = 1.0)``

``id``
   A unique string identifier
``el``
   The element described in a string. Note that ions is denoted as
   "Sr2p" and "O2m" where 2 is the oxidation number and p and m denoted
   plus and minus charge.
``x``
   The x-position in Slab unit cell coords (same as given by the
   UnitCell)
``y``
   The y-position in Slab unit cell coords (same as given by the
   UnitCell)
``z``
   The z-position in Slab unit cell coords (The Unitcell c scaled by a
   factor of the c-value for the slab)
``u``
   The mean-square displacement for the atom
``oc``
   The occupancy of the atom
``m``
   The multiplicity of the site, defined as in the international tables
   of crystallogrphy. Note that it is plane goups and NOT space groups
   that will produce valid results.

``[Slab].copy()``
    Creates a copy of object [Slab]. This decouples the new object
    returned by copy from the original [Slab].
``[Slab].find_atoms(expression)``
    Function to locate atoms in a slab in order to connect parameters
    between them. Returns an AtomGroup.
``expression``
    Either a list of the same length as the number of atoms or a string
    that will evaluate to true or false for each atom. Allowed variables
    are: ``x, y, z, id, el, u, ov, m,``
``[Slab].all_atoms()``
    Yields all atoms inside a slab as an AtomGroup. Returns an AtomGroup.
``[Slab][id]``
    Locates atom that has id ``id``. Returns an AtomGroup
``id``
   Uniqe string identifer for one atom

Sample
~~~~~~
``Sample(inst, bulk_slab, slabs, unit_cell, surface_sym = [], bulk_sym = [])``

``inst``
   Instrument object for the sample
``bulk_slab``
   The Slab that describes the bulk strucutre
``slabs``
   A list ([]) of slabs for the surface structure
``unit_cell``
   A UnitCell object
``surface_sym``
   A list ([]) of SymTrans objects describing the surface symmetry.
   Default value - an empty list will implement a p1 symmetry, that is
   no symmetry operations at all.
``bulk_sym``
   A list ([]) of SymTrans objects describing the bulk symmetry. Default
   value - an empty list will implement a p1 symmetry, that is no
   symetry operations at all.

``[Sample].calc_f(h, k, l)``
    Calculates the total structure factor (complex number) from the the
    surface and bulk strucutre. Returns an array of the same size as h, k,
    l. (h, k, l should be of the same legth and is given in coordinates of
    the reciprocal lattice as defnined by the uit_cell coords)
``[Sample].turbo_calc_f(h, k, l)``
    A faster version of ``calc_f`` which uses inline c code to increase
    the speed. Can be more unstable than ``calc_f`` use on your own risk.
``[Sample].calc_rhos(x, y, z, sb)``
    Calculate the the surface electron density of a model. The parameter
    sb is a Gaussian convolution factor given the width of the Gaussian in
    reciprocal space. Used mainly for comparison with direct methods, i.e.
    DCAF. NOTE that the transformation from the width of the window
    function given in ``dimes.py`` is ``sqrt(2)*pi*[]``
'''

import numpy as np
from . import utils
from .lib.physical_constants import r_e
from .symmetries import SymTrans
from genx.core.custom_logging import iprint

from .lib import USE_NUMBA


if USE_NUMBA:
    try:
        from .lib.surface_scattering import surface_lattice_sum
    except ImportError:
        numba_ss = False
    else:
        numba_ss = True
else:
    numba_ss = False

__pars__ = ['Sample', 'UnitCell', 'Slab', 'AtomGroup', 'Instrument']

__xlabel__ = "q-scan [r.l.u.]"
__ylabel__ = "Instnsity [a.u.]"

class Sample:

    def __init__(self, inst, bulk_slab, slabs, unit_cell,
                 surface_sym=None, bulk_sym=None):
        if surface_sym is None:
            surface_sym = []
        if bulk_sym is None:
            bulk_sym = []
        self.set_bulk_slab(bulk_slab)
        self.set_slabs(slabs)
        self.set_surface_sym(surface_sym)
        self.set_bulk_sym(bulk_sym)
        self.inst = inst
        self.set_unit_cell(unit_cell)

    def set_bulk_slab(self, bulk_slab):
        '''Set the bulk unit cell to bulk_slab
        '''
        if type(bulk_slab)!=type(Slab()):
            raise TypeError("The bulk slab has to be a member of class Slab")
        self.bulk_slab = bulk_slab

    def set_slabs(self, slabs):
        '''Set the slabs of the sample.

        slabs should be a list of objects from the class Slab
        '''
        if type(slabs)!=type([]):
            raise TypeError("The surface slabs has to contained in a list")
        if min([type(slab)==type(Slab()) for slab in slabs])==0:
            raise TypeError("All members in the slabs list has to be a memeber of class Slab")
        self.slabs = slabs

    def set_surface_sym(self, sym_list):
        '''Sets the list of symmetry operations for the surface.

        sym_list has to be a list ([]) of symmetry elements from the
        class SymTrans
        '''
        # Type checking
        if type(sym_list)!=type([]):
            raise TypeError("The surface symmetries has to contained in a list")

        if not sym_list:
            sym_list = [SymTrans()]

        if min([type(sym)==type(SymTrans()) for sym in sym_list])==0:
            raise TypeError("All members in the symmetry list has to be a memeber of class SymTrans")

        self.surface_sym = sym_list

    def set_bulk_sym(self, sym_list):
        '''Sets the list of allowed symmetry operations for the bulk

        sym_list has to be a list ([]) of symmetry elements from the
        class SymTrans
        '''
        # Type checking
        if type(sym_list)!=type([]):
            raise TypeError("The surface symmetries has to contained in a list")

        if not sym_list:
            sym_list = [SymTrans()]

        if min([type(sym)==type(SymTrans()) for sym in sym_list])==0:
            raise TypeError("All members in the symmetry list has to be a memeber of class SymTrans")

        self.bulk_sym = sym_list

    def set_unit_cell(self, unit_cell):
        '''Sets the unitcell of the sample
        '''
        if type(unit_cell)!=type(UnitCell(1.0, 1.0, 1.0)):
            raise TypeError("The bulk slab has to be a member of class UnitCell")
        if unit_cell is None:
            unit_cell = UnitCell(1.0, 1.0, 1.0)
        self.unit_cell = unit_cell

    def calc_f(self, h, k, l):
        '''Calculate the structure factors for the sample
        '''
        fs = self.calc_fs(h, k, l)
        fb = self.calc_fb(h, k, l)
        ftot = fs+fb
        return ftot*self.inst.inten

    def turbo_calc_f(self, h, k, l):
        '''Calculate the structure factors for the sample with
        inline c code for the surface.
        '''
        fs = self.turbo_calc_fs(h, k, l)
        fb = self.calc_fb(h, k, l)
        ftot = fs+fb
        return ftot*self.inst.inten

    if numba_ss:
        # replace by faster version
        calc_f = turbo_calc_f

    def calc_fs(self, h, k, l):
        '''Calculate the structure factors from the surface
        '''
        dinv = self.unit_cell.abs_hkl(h, k, l)
        x, y, z, u, oc, el = self._surf_pars()
        # print x, y,z
        # Create all the atomic structure factors
        f = self._get_f(el, dinv)
        # print f.shape, h.shape, oc.shape, x.shape, y.shape, z.shape
        fs = np.sum(oc*f*np.exp(-2*np.pi**2*u*dinv[:, np.newaxis]**2) \
                    *np.sum([np.exp(2.0*np.pi*1.0J*(
                h[:, np.newaxis]*sym_op.trans_x(x, y)+
                k[:, np.newaxis]*sym_op.trans_y(x, y)+
                l[:, np.newaxis]*z[np.newaxis, :]))
                             for sym_op in self.surface_sym], 0)
                    , 1)
        return fs

    def turbo_calc_fs(self, h, k, l):
        '''Calculate the structure factors with weave (inline c code)
        Produces faster simulations of large structures.
        '''
        h = h.astype(np.float64)
        k = k.astype(np.float64)
        l = l.astype(np.float64)
        dinv = self.unit_cell.abs_hkl(h, k, l)
        x, y, z, u, oc, el = self._surf_pars()
        f = self._get_f(el, dinv)
        Pt = np.array([np.c_[so.P, so.t] for so in self.surface_sym])
        fs = surface_lattice_sum(x, y, z, h, k, l, u, oc, f, Pt, dinv)
        return fs

    def calc_fb(self, h, k, l):
        '''Calculate the structure factors from the bulk
        '''
        dinv = self.unit_cell.abs_hkl(h, k, l)
        x, y, z, el, u, oc, c = self.bulk_slab._extract_values()
        oc = oc/float(len(self.bulk_sym))
        f = self._get_f(el, dinv)
        # Calculate the "shape factor" for the CTRs
        eff_thick = self.unit_cell.c/np.sin(self.inst.alpha*np.pi/180.0)
        alpha = (r_e*self.inst.wavel*eff_thick/self.unit_cell.vol()*
                 np.sum(f.imag, 1))
        denom = np.exp(2.0*np.pi*1.0J*l)*np.exp(-alpha)-1.0
        # Delta functions to remove finite size effect in hk plane
        delta_funcs = (abs(h-np.round(h))<1e-12)*(
                abs(k-np.round(k))<1e-12)
        # Sum up the uc struct factors
        f_u = np.sum(oc*f*np.exp(-2*np.pi**2*u*dinv[:, np.newaxis]**2)*
                     np.sum([np.exp(2.0*np.pi*1.0J*(
                             h[:, np.newaxis]*sym_op.trans_x(x, y)+
                             k[:, np.newaxis]*sym_op.trans_y(x, y)+
                             l[:, np.newaxis]*z[np.newaxis, :]))
                             for sym_op in self.bulk_sym], 0)
                     , 1)
        # Putting it all togheter
        fb = f_u/denom*delta_funcs

        return fb

    def calc_rhos(self, x, y, z, sb=0.8):
        '''Calcualte the electron density of the unitcell
        '''
        px, py, pz, u, oc, el = self._surf_pars()
        rhos = self._get_rho(el)

        rho = np.sum([np.sum([rho(self.unit_cell.dist(x, y, z,
                                                      sym_op.trans_x(xat, yat)%1.0,
                                                      sym_op.trans_y(xat, yat)%1.0,
                                                      zat),
                                  0.5*uat+0.5/sb**2, ocat)
                              for rho, xat, yat, zat, uat, ocat in
                              zip(rhos, px, py, pz, u, oc)], 0)
                      for sym_op in self.surface_sym], 0)
        return rho

    def _surf_pars(self):
        '''Extracts the necessary parameters for simulating the surface part
        '''
        # Extract the parameters we need
        # the star in zip(*... transform the list elements to arguments
        xt, yt, zt, elt, ut, oct, ct = list(zip(*[slab._extract_values() for slab in self.slabs]))

        x = np.r_[xt]
        y = np.r_[yt]
        # scale and shift the slabs with respect to each other
        cn = np.cumsum(np.r_[0, ct])[:-1]
        z = np.concatenate([zs*c_s+c_cum
                            for zs, c_cum, c_s in zip(zt, cn, ct)])
        # el = reduce(lambda x,y:x+y, elt)
        el = np.r_[elt]
        u = np.r_[ut]
        # Account for overlapping atoms
        oc = np.r_[oct]/float(len(self.surface_sym))
        # print x,y,z, u

        return x, y, z, u, oc, el

    def create_uc_output(self):
        ''' Create atomic positions and such for output '''
        x, y, z, u, oc, el = self._surf_pars()
        ids = []
        [ids.extend(slab._extract_ids()) for slab in self.slabs]
        xout = np.array([])
        yout = np.array([])
        zout = np.array([])
        uout = np.array([])
        ocout = np.array([])
        elout = el[0:0].copy()
        idsout = []
        for sym_op in self.surface_sym:
            xout = np.r_[xout, sym_op.trans_x(x, y)]
            yout = np.r_[yout, sym_op.trans_y(x, y)]
            zout = np.r_[zout, z]
            uout = np.r_[uout, u]
            ocout = np.r_[ocout, oc]
            elout = np.r_[elout, el]
        idsout.extend(ids)

        return xout, yout, zout, uout, ocout, elout, idsout

    def _get_f(self, el, dinv):
        '''from the elements extract an array with atomic structure factors
        '''
        return _get_f(self.inst, el, dinv)

    def _get_rho(self, el):
        '''Returns the rho functions for all atoms in el
        '''
        return _get_rho(self.inst, el)

    def _fatom_eval(self, f, element, s):
        '''Smart (fast) evaluation of f_atom. Only evaluates f if not
        evaluated before.

        element - element string
        f - dictonary for lookup
        s - sintheta_over_lambda array
        '''
        return _fatom_eval(inst, f, element, s)

    def export_xyz(self, fname):
        from genx.version import __version__ as version
        if isinstance(fname, str) and not fname.endswith('.xyz'):
            fname += '.xyz'
        x, y, z, u, oc, el, ids = self.create_uc_output()
        uc_a = self.unit_cell.a
        uc_b = self.unit_cell.b
        uc_c = self.unit_cell.c
        c_total = uc_c*sum([sl.c for sl in self.slabs])
        if isinstance(fname, str):
            with open(fname, 'w') as fh:
                fh.write(f'{len(x)}\n')
                fh.write(f'# structure exported by GenX {version}, UC: a={uc_a} b={uc_b} c={c_total}\n')
                for xi, yi, zi, eli in zip(x, y, z, el):
                    fh.write(f'{eli:3s} {uc_a*xi:-12.7f} {uc_b*yi:-12.7f} {uc_c*zi:-12.7f}\n')
        else:
            # user has supplied a file handler instead of name
            fh = fname
            fh.write(f'{len(x)}\n')
            fh.write(f'# structure exported by GenX {version}, UC: a={uc_a} b={uc_b} c={c_total}\n')
            for xi, yi, zi, eli in zip(x, y, z, el):
                fh.write(f'{eli:3s} {uc_a*xi:-12.7f} {uc_b*yi:-12.7f} {uc_c*zi:-12.7f}\n')


class UnitCell:
    '''Class containing the  unitcell.
    This also allows for simple crystalloraphic computing of different
    properties.
    '''

    def __init__(self, a, b, c, alpha=90,
                 beta=90, gamma=90):
        self.set_a(a)
        self.set_b(b)
        self.set_c(c)
        self.set_alpha(alpha)
        self.set_beta(beta)
        self.set_gamma(gamma)

    def set_a(self, a):
        self.a = a

    def set_b(self, b):
        self.b = b

    def set_c(self, c):
        self.c = c

    def set_alpha(self, alpha):
        self.alpha = alpha*np.pi/180.

    def set_beta(self, beta):
        self.beta = beta*np.pi/180.

    def set_gamma(self, gamma):
        self.gamma = gamma*np.pi/180.

    def vol(self):
        '''Calculate the volume of the unit cell in AA**3
        '''
        vol = self.a*self.b*self.c*np.sqrt(1-np.cos(self.alpha)**2-
                                           np.cos(self.beta)**2-np.cos(self.gamma)**2+
                                           2*np.cos(self.alpha)*np.cos(self.beta)*np.cos(self.gamma))
        return vol

    def cart_coords(self, uc_x, uc_y, uc_z):
        '''Transform the uc coors uc_x, uc_y, uc_z to cartesian
        coordinates expressed in AA
        '''
        return (self.cart_coord_x(uc_x, uc_y, uc_z), self.cart_coord_y(uc_x, uc_y, uc_z),
                self.cart_coord_z(uc_x, uc_y, uc_z))

    def cart_coord_x(self, uc_x, uc_y, uc_z):
        '''Get the x-coord in the cart system
        '''
        return uc_x*self.a

    def cart_coord_y(self, uc_x, uc_y, uc_z):
        '''Get the y-coord in the cart system
        '''
        return uc_y*self.b

    def cart_coord_z(self, uc_x, uc_y, uc_z):
        '''Get the y-coord in the cart system
        '''
        return uc_z*self.c

    def dist(self, x1, y1, z1, x2, y2, z2):
        '''Calculate the distance in AA between the points
        (x1, y1, z1) and (x2, y2, z2). The coords has to be unit cell
        coordinates.
        '''
        # print 'Warning works only with orth cryst systems!'
        return np.sqrt(((x1-x2)*self.a)**2+((y1-y2)*self.b)**2+
                       ((z1-z2)*self.c)**2)

    def abs_hkl(self, h, k, l):
        '''Returns the absolute value of (h,k,l) vector in units of
        AA.

        This is equal to the inverse lattice spacing 1/d_hkl.
        '''
        dinv = np.sqrt(((h/self.a*np.sin(self.alpha))**2+
                        (k/self.b*np.sin(self.beta))**2+
                        (l/self.c*np.sin(self.gamma))**2+
                        2*k*l/self.b/self.c*(np.cos(self.beta)*
                                             np.cos(self.gamma)-
                                             np.cos(self.alpha))+
                        2*l*h/self.c/self.a*(np.cos(self.gamma)*
                                             np.cos(self.alpha)-
                                             np.cos(self.beta))+
                        2*h*k/self.a/self.b*(np.cos(self.alpha)*
                                             np.cos(self.beta)-
                                             np.cos(self.gamma)))
                       /(1-np.cos(self.alpha)**2-np.cos(self.beta)**2
                         -np.cos(self.gamma)**2+2*np.cos(self.alpha)
                         *np.cos(self.beta)*np.cos(self.gamma)))
        return dinv


class Slab:
    par_names = ['dx', 'dy', 'dz',
                 'u', 'oc', 'm']

    def __init__(self, name='', c=1.0, slab_oc=1.0):
        try:
            self.c = float(c)
        except:
            raise ValueError("Parameter c has to be a valid floating point number")
        try:
            self.slab_oc = float(slab_oc)
        except:
            raise ValueError("Parameter slab_oc has to be a valid floating point number")
        # Set the arrays to their default values
        self.x = np.array([], dtype=np.float64)
        self.y = np.array([], dtype=np.float64)
        self.z = np.array([], dtype=np.float64)
        self.dx = np.array([], dtype=np.float64)
        self.dy = np.array([], dtype=np.float64)
        self.dz = np.array([], dtype=np.float64)
        self.u = np.array([], dtype=np.float64)
        self.oc = np.array([], dtype=np.float64)
        self.m = np.array([], dtype=np.float64)
        self.id = np.array([], dtype=str)
        self.el = np.array([], dtype=str)

        # TODO: Type checking and defaults!
        # self.inst = inst
        self.name = str(name)

    def copy(self):
        '''Returns a copy of the object.
        '''
        cpy = Slab(c=self.c, slab_oc=self.slab_oc)
        for i in range(len(self.id)):
            cpy.add_atom(str(self.id[i]), str(self.el[i]),
                         self.x[i], self.y[i],
                         self.z[i], self.u[i], self.oc[i], self.m[i])
            cpy.dz[-1] = self.dz[i]
            cpy.dx[-1] = self.dx[i]
            cpy.dy[-1] = self.dy[i]
        return cpy

    def __str__(self):
        out = f'Slab(c={self.c}, slab_oc={self.slab_oc})'
        for i in range(len(self.id)):
            out += f'\n  atom {i+1}: {self.id[i]}= {self.el[i]} at '
            out += f'({self.x[i]}, {self.y[i]}, self.z[i]) '
            out += f'u={self.u[i]}, oc={self.oc[i]}, m={self.m[i]}'
        return out

    def add_atom(self, id, element, x, y, z, u=0.0, oc=1.0, m=1.0):
        '''Add an atom to the slab.

        id - a unique id for this atom (string)
        element - the element of this atom has to be found
        within the scatteringlength table.
        x, y, z - position in the assymetricv unit cell (floats)
        u - debye-waller parameter for the atom
        oc - occupancy of the atomic site
        '''
        if id in self.id:
            raise ValueError('The id %s is already defined in the'
                             'slab'%id)
        # TODO: Check the element as well...
        self.x = np.append(self.x, x)
        self.dx = np.append(self.dx, 0.)
        self.y = np.append(self.y, y)
        self.dy = np.append(self.dy, 0.)
        self.z = np.append(self.z, z)
        self.dz = np.append(self.dz, 0.)
        self.u = np.append(self.u, u)
        self.oc = np.append(self.oc, oc)
        self.m = np.append(self.m, m)
        self.id = np.append(self.id, id)
        self.el = np.append(self.el, str(element))
        item = len(self.id)-1
        # Create the set and get functions dynamically
        for par in self.par_names:
            setattr(self, 'set'+id+par, self._make_set_func(par, item))
            setattr(self, 'get'+id+par, self._make_get_func(par, item))
        return AtomGroup(self, id)

    def del_atom(self, id):
        '''Remove atom identified with id
        '''
        if not id in self.id:
            raise ValueError('Can not remove atom with id %s -'
                             'namedoes not exist')
        item = np.argwhere(self.id==id)[0][0]
        if item<len(self.x)-1:
            ar = getattr(self, 'id')
            setattr(self, 'id', np.r_[ar[:item], ar[item+1:]])
            ar = getattr(self, 'el')
            setattr(self, 'el', np.r_[ar[:item], ar[item+1:]])
            ar = getattr(self, 'x')
            setattr(self, 'x', np.r_[ar[:item], ar[item+1:]])
            ar = getattr(self, 'y')
            setattr(self, 'y', np.r_[ar[:item], ar[item+1:]])
            ar = getattr(self, 'z')
            setattr(self, 'z', np.r_[ar[:item], ar[item+1:]])

            for par in self.par_names:
                ar = getattr(self, par)
                setattr(self, par, np.r_[ar[:item], ar[item+1:]])
                delattr(self, 'set'+id+par)
                delattr(self, 'get'+id+par)
        else:
            ar = getattr(self, 'id')
            setattr(self, 'id', ar[:-1])
            ar = getattr(self, 'el')
            setattr(self, 'el', ar[:-1])
            ar = getattr(self, 'x')
            setattr(self, 'x', ar[:-1])
            ar = getattr(self, 'y')
            setattr(self, 'y', ar[:-1])
            ar = getattr(self, 'z')
            setattr(self, 'z', ar[:-1])

            for par in self.par_names:
                ar = getattr(self, par)
                setattr(self, par, ar[:-1])
                delattr(self, 'set'+id+par)
                delattr(self, 'get'+id+par)

    def find_atoms(self, expression):
        '''
        Find the atoms that satisfy the logical expression given in the
        string expression. Expression can also be a list or array of the
        same length as the number of atoms in the slab.

        Allowed variables in expression are:
        x, y, z, u, occ, id, el
        returns an AtomGroup
        '''
        if (type(expression)==type(np.array([])) or
                type(expression)==type(list([]))):
            if len(expression)!=len(self.id):
                raise ValueError('The length of experssion is wrong'
                                 ', it should match the number of atoms')
            ag = AtomGroup()
            [ag.add_atom(self, str(id)) for id, add in
             zip(self.id, expression) if add]
            return ag
        elif type(expression)==type(''):
            choose_list = [eval(expression) for x, y, z, u, oc, el, id in
                           zip(self.x, self.y, self.z, self.u,
                               self.oc, self.el, self.id)]
            # print choose_list
            ag = AtomGroup()
            [ag.add_atom(self, str(name)) for name, add
             in zip(self.id, choose_list) if add]
            return ag
        else:
            raise ValueError('Expression has to be a string, array or list')

    def all_atoms(self):
        '''
        Puts all atoms in the slab to an AtomGroup.

        returns: AtomGroup
        '''
        return self.find_atoms([True]*len(self.id))

    def set_c(self, c):
        '''
        Set the out-of-plane extension of the slab.
        Note that this is in the defined UC coords given in
        the corresponding sample
        '''
        self.c = float(c)

    def get_c(self):
        '''
        Get the out-of-plane extension of the slab in UC coord.
        '''
        return self.c

    def set_oc(self, oc):
        '''
        Set a global occupation parameter for the entire slab.
        should be between 0 and 1. To create the real occupancy this
        value is multiplied with the occupancy for that atom.
        '''
        self.slab_oc = oc

    def get_oc(self):
        '''
        Get the global occupancy of the slab
        '''
        return self.slab_oc

    def __getitem__(self, id):
        '''
        Locate id in slab with a dictonary style.
        Returns a AtomGroup instance
        '''
        return AtomGroup(self, id)

    def __contains__(self, id):
        '''
        Makes it possible to check if id exist in this Slab by using
        the in operator. It is also possible if all atoms in an AtomGroup
        belongs to the slab.
        
        returns True or False
        '''
        if type(id)==type(''):
            return id in self.id
        elif type(id)==type(AtomGroup):
            return np.all([atid in self.id for atid in id.ids])
        else:
            raise ValueError('Can only check for mebership for Atom groups'
                             'or string ids.')

    def _set_in(self, arr, pos, value):
        '''
        Sets a value in an array or list
        '''
        arr[pos] = value

    def _make_set_func(self, par, pos):
        '''
        Creates a set functions for parameter par and at pos.
        Returns a function
        '''

        def set_par(val):
            getattr(self, par)[pos] = val

        return set_par

    def _make_get_func(self, par, pos):
        '''
        Cerates a set function for member par at pos.
        Returns a function.
        '''

        def get_par():
            return getattr(self, par)[pos]

        return get_par

    def _extract_values(self):
        return self.x+self.dx, self.y+self.dy, self.z+self.dz, \
               self.el, self.u, self.oc*self.m*self.slab_oc, self.c

    def _extract_ids(self):
        'Extract the ids of the atoms'
        return [self.name+'.'+str(id) for id in self.id]


class AtomGroup:
    par_names = ['dx', 'dy', 'dz', 'u', 'oc']

    def __init__(self, slab=None, id=None):
        self.ids = []
        self.slabs = []
        # Variable for composition ...
        self.comp = 1.0
        self.oc = 1.0
        if slab is not None and id is not None:
            self.add_atom(slab, id)

    def _set_func(self, par):
        '''create a function that sets all atom paramater par'''
        funcs = [getattr(slab, 'set'+id+par) for id, slab
                 in zip(self.ids, self.slabs)]

        def set_pars(val):
            [func(val) for func in funcs]

        return set_pars

    def _get_func(self, par):
        '''create a function that gets all atom paramater par'''
        funcs = [getattr(slab, 'get'+id+par) for id, slab
                 in zip(self.ids, self.slabs)]

        def get_pars():
            return np.mean([func() for func in funcs])

        return get_pars

    def update_setget_funcs(self):
        '''
        Update all the atomic set and get functions
        '''
        for par in self.par_names:
            setattr(self, 'set'+par, self._set_func(par))
            setattr(self, 'get'+par, self._get_func(par))

    def add_atom(self, slab, id):
        '''
        Add an atom to the group.
        '''
        if not id in slab:
            raise ValueError('The id %s is not a member of the slab'%id)
        self.ids.append(id)
        self.slabs.append(slab)
        self.update_setget_funcs()

    def _copy(self):
        '''
        Creates a copy of self And looses all connection to the
        previously created compositions conenctions
        '''
        cpy = AtomGroup()
        cpy.ids = self.ids[:]
        cpy.slabs = self.slabs[:]
        cpy.update_setget_funcs()
        return cpy

    def comp_coupl(self, other, self_copy=False, exclusive=True):
        '''
        Method to create set-get methods to use compositions
        in the atomic groups. Note that this does not affect
        the slabs global occupancy. If self_copy is True the
        returned value will be a copy of self.
        If exculive is true reomves all methods from the
        previous AtomGroups that are coupled.
        '''
        if not type(self)==type(other):
            raise TypeError('To create a composition function both objects'
                            ' has to be of the type AtomGroup')
        if hasattr(other, '_setoc_'):
            raise AttributeError('The right hand side AtomicGroup has already'
                                 'been coupled to another one before.'
                                 ' Only one connection'
                                 'is allowed')
        if hasattr(self, '_setoc'):
            raise AttributeError('The left hand side AtomicGroup has already'
                                 'been coupled to another one before.'
                                 ' Only one connection'
                                 'is allowed')
        if self_copy:
            s = self._copy()
        else:
            s = self

        def set_comp(comp):
            # print "Executing comp function"
            s.comp = float(comp)
            s._setoc(comp*s.oc)
            other._setoc_((1.0-comp)*s.oc)

        def set_oc(oc):
            # print "Executing oc function"
            s.oc = float(oc)
            s._setoc(s.comp*s.oc)
            other._setoc_((1-s.comp)*s.oc)

        def get_comp():
            return s.comp

        def get_oc():
            return s.oc

        # Functions to couple the other parameters, set
        def create_set_func(par):
            sf_set = getattr(s, 'set'+par)
            of_set = getattr(other, 'set'+par)

            def _set_func(val):
                p = str(par)
                # print 'Setting %s to %s'%(p, val)
                sf_set(val)
                of_set(val)

            return _set_func

        # Functions to couple the other parameters, set
        def create_get_func(par):
            sf_get = getattr(s, 'get'+par)
            of_get = getattr(other, 'get'+par)

            def _get_func():
                p = str(par)
                return (sf_get()+of_get())/2

            return _get_func

        # Do it (couple) for all parameters except the occupations
        if exclusive:
            for par in s.par_names:
                if not str(par)=='oc':
                    # print par
                    setattr(s, 'set'+par, create_set_func(par))
                    setattr(s, 'get'+par, create_get_func(par))

        # Create new set and get methods for the composition
        setattr(s, 'setcomp', set_comp)
        setattr(s, 'getcomp', get_comp)

        # Store the original setoc for future use safely
        setattr(s, '_setoc', s.setoc)
        setattr(other, '_setoc_', getattr(other, 'setoc'))

        setattr(s, 'setoc', set_oc)
        setattr(s, 'getoc', get_oc)

        # Now remove all the coupled attribute from other.
        if exclusive:
            for par in s.par_names:
                delattr(other, 'set'+par)

        s.setcomp(1.0)

        return s

    def __xor__(self, other):
        '''
        Method to create set-get methods to use compositions
        in the atomic groups. Note that this does not affect
        the slabs global occupancy. Note that the
        first element (left hand side of ^) will be copied
        and loose all its previous connections.
        Note that all the move methods that are not coupled will
        be removed.
        '''
        return self.comp_coupl(other, self_copy=True, exclusive=True)

    def __ixor__(self, other):
        '''
        Method to create set-get methods to use compositions
        in the atomic groups. Note that this does not affect
        the slabs global occupancy.
        Note that all the move methods that are not coupled will
        be removed.
        '''
        self.comp_coupl(other, exclusive=True)

    def __or__(self, other):
        '''
        Method to create set-get methods to use compositions
        in the atomic groups. Note that this does not affect
        the slabs global occupancy. Note that the
        first element (left hand side of |) will be copied
        and loose all its previous connections.
        '''
        return self.comp_coupl(other, self_copy=True, exclusive=False)

    def __ior__(self, other):
        '''
        Method to create set-get methods to use compositions
        in the atomic groups. Note that this does not affect
        the slabs global occupancy.
        '''
        self.comp_coupl(other, exclusive=False)

    def __add__(self, other):
        '''
        Adds two Atomic groups togheter
        '''
        if not type(other)==type(self):
            raise TypeError('Adding wrong type to an AtomGroup has to be an'
                            'AtomGroup')
        ids = self.ids+other.ids
        slabs = self.slabs+other.slabs
        out = AtomGroup()
        [out.add_atom(slab, id) for slab, id in zip(slabs, ids)]

        s = self

        def set_oc(oc):
            # print "Executing oc function"
            s.oc = float(oc)
            s.setoc(s.oc)
            other.setoc(s.oc)

        def get_oc():
            return s.oc

        setattr(out, 'setoc', set_oc)
        setattr(out, 'getoc', get_oc)

        return out


class Instrument:
    '''
    Class that keeps tracks of instrument settings.
    '''
    geometries = ['alpha_in fixed', 'alpha_in eq alpha_out',
                  'alpha_out fixed']

    def __init__(self, wavel, alpha, geom='alpha_in fixed', flib=None, rholib=None):
        '''
        Inits the instrument with default parameters
        '''
        if flib is None:
            self.flib = utils.sl.FormFactor(wavel, utils.__lookup_f__)
        else:
            self.flib = flib
        if rholib is None:
            self.rholib = utils.sl.FormFactor(wavel, utils.__lookup_rho__)
        else:
            self.rholib = rholib
        self.set_wavel(wavel)
        self.set_geometry(geom)
        self.alpha = alpha
        self.inten = 1.0

    def set_inten(self, inten):
        '''
        Set the incomming intensity
        '''
        self.inten = inten

    def get_inten(self):
        '''
        retrieves the intensity
        '''
        return self.inten

    def set_wavel(self, wavel):
        '''
        Set the wavelength in AA
        '''
        try:
            self.wavel = float(wavel)
            self.flib.set_wavelength(wavel)
            self.rholib.set_wavelength(wavel)
        except ValueError:
            raise ValueError('%s is not a valid float number needed for the'
                             'wavelength'%wavel)

    def get_wavel(self, wavel):
        '''
        Returns the wavelength in AA
        '''
        return self.wavel

    def set_energy(self, energy):
        '''
        Set the energy in keV
        '''
        try:
            self.set_wavel(12.39842/float(energy))
        except ValueError:
            raise ValueError('%s is not a valid float number needed for the'
                             'energy'%energy)

    def get_energy(self, energy):
        '''Returns the photon energy in keV
        '''
        return 12.39842/self.wavel

    def set_alpha(self, alpha):
        '''
        Sets the freezed angle. The meaning of this angle varies depening
        of the geometry parameter.
        
        geo =  "alpha_in fixed", alpha = alpha_in
        geo = "alpha_in eq alpha_out", alpha = alpha_in = alpha_out
        geo = "alpha_out fixed", alpha = alpha_out
        '''
        self.alpha = alpha

    def get_alpha(self):
        '''Gets the freexed angle. See set_alpha.
        '''
        return self.alpha

    def set_geometry(self, geom):
        '''
        Set the measurement geometry

        Should be one of the items in Instrument.geometry
        '''
        try:
            self.geom = self.geometries.index(geom)
        except ValueError:
            raise ValueError('The geometry  %s does not exist please choose'
                             'one of the following:\n%s'%(geom,
                                                          self.geomeries))

    def set_flib(self, flib):
        '''
        Set the structure factor library
        '''
        self.flib = flib

    def set_rholib(self, rholib):
        '''Set the rho library (electron density shape of the atoms)
        '''
        self.rholib = rholib


# ==============================================================================
# Utillity functions
def scale_sim(data, sim_list, scale_func=None):
    '''
    Scale the data according to a miminimazation of
    sum (data-I_list)**2
    '''
    numerator = sum([(data[i].y*sim_list[i]).sum() for i in range(len(data))
                     if data[i].use])
    denominator = sum([(sim_list[i]**2).sum() for i in range(len(data))
                       if data[i].use])
    scale = numerator/denominator
    scaled_sim_list = [sim*scale for sim in sim_list]
    if not scale_func is None:
        scale_func(scale)
    return scaled_sim_list


def scale_sqrt_sim(data, sim_list, scale_func=None):
    '''
    Scale the data according to a miminimazation of
    sum (sqrt(data)-sqrt(I_list))**2
    '''
    numerator = sum([(np.sqrt(data[i].y*sim_list[i])).sum()
                     for i in range(len(data))
                     if data[i].use])
    denominator = sum([(sim_list[i]).sum() for i in range(len(data))
                       if data[i].use])
    scale = numerator/denominator
    scaled_sim_list = [sim*scale**2 for sim in sim_list]
    if not scale_func is None:
        scale_func(scale)
    return scaled_sim_list


def _get_f(inst, el, dinv):
    '''
    from the elements extract an array with atomic structure factors
    '''
    fdict = {}
    f = np.transpose(np.array([_fatom_eval(inst, fdict, elem, dinv/2.0) for elem in el], dtype=np.complex128))

    return f


def _get_rho(inst, el):
    '''
    Returns the rho functions for all atoms in el
    '''
    rhos = [getattr(inst.rholib, elem) for elem in el]
    return rhos


def _fatom_eval(inst, f, element, s):
    '''
    Smart (fast) evaluation of f_atom. Only evaluates f if not
    evaluated before.
    
    element - element string
    f - dictonary for lookup
    s - sintheta_over_lambda array
    '''
    try:
        fret = f[element]
    except KeyError:
        fret = getattr(inst.flib, element)(s)
        f[element] = fret
        # print element, fret[0]
    return fret


# =============================================================================

if __name__=='__main__':
    inst = Instrument(wavel=0.77, alpha=0.2)
    ss1 = Slab(c=1.00)
    ss1.add_atom('La', 'la', 0.0, 0.0, 0.0, 0.001, 1.0, 1)
    ss1.add_atom('Al', 'al', 0.5, 0.5, 0.5, 0.001, 1.0, 1)
    ss1.add_atom('O1', 'o', 0.5, 0.5, 0.0, 0.001, 1.0, 1)
    ss1.add_atom('O2', 'o', 0.0, 0.5, 0.5, 0.001, 1.0, 1)
    ss1.add_atom('O3', 'o', 0.5, 0.0, 0.5, 0.001, 1.0, 1)

    bulk = Slab()
    bulk.add_atom('Sr', 'sr', 0.0, 0.0, 0.0, 0.001, 1.0)
    bulk.add_atom('Ti', 'ti', 0.5, 0.5, 0.5, 0.001, 1.0)
    bulk.add_atom('O1', 'o', 0.5, 0.0, 0.5, 0.001, 1.0)
    bulk.add_atom('O2', 'o', 0.0, 0.5, 0.5, 0.001, 1.0)
    bulk.add_atom('O3', 'o', 0.5, 0.5, 0.0, 0.001, 1.0)

    sample = Sample(inst, bulk, [ss1]*1,
                    UnitCell(3.945, 3.945, 3.945, 90, 90, 90))
    l = np.arange(0.0, 5, 0.01)
    h = 0.0*np.ones(l.shape)
    k = 1.0*np.ones(l.shape)
    f = sample.calc_f(h, k, l)

    s_sym = Slab(c=1.00)
    s_sym.add_atom('La', 'la', 0.0, 0.0, 0.0, 0.001, 1.0, 1)
    s_sym.add_atom('Al', 'al', 0.5, 0.5, 0.5, 0.001, 1.0, 1)
    s_sym.add_atom('O1', 'o', 0.5, 0.5, 0.0, 0.001, 1.0, 1)
    s_sym.add_atom('O2', 'o', 0.5, 0.0, 0.5, 0.001, 1.0, 2)

    p4 = [SymTrans([[1, 0], [0, 1]]), SymTrans([[-1, 0], [0, -1]]),
          SymTrans([[0, -1], [1, 0]]), SymTrans([[0, 1], [-1, 0]])]
    sample2 = Sample(inst, bulk, [s_sym]*1,
                     UnitCell(3.945, 3.945, 3.945, 90, 90, 90))
    sample2.set_surface_sym(p4)
    # z = np.arange(-0.1, 3.5, 0.01)
    # x = 0*z + 0.5
    # y = 0*z + 0.5
    # rho = sample2.calc_rhos(x, y, z)

    f2 = sample2.calc_f(h, k, l)
    import time


    t1 = time.time()
    sf = sample2.calc_fs(h, k, l)
    t2 = time.time()
    iprint('Python: %f seconds'%(t2-t1))
    t3 = time.time()
    sft = sample2.turbo_calc_fs(h, k, l)
    t4 = time.time()
    iprint('Inline C: %f seconds'%(t4-t3))
