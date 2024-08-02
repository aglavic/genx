"""Utilities for GenX and scattering
=================================

This library contains nice to have functions and classes. Most
noteworthy is the UserVars class. The scattering factor tables can also
come handy.

Classes
-------

UserVars
~~~~~~~~
This class is used to contain user defined variables which is problem
specific and do not warrant a completly new model library. It is
contstructed as: ``vars = UserVars()``.

A new variable is created by calling the new_var method:
``vars.new_var('my_var', 10.0)``

The methods in this class are:

``__init__(self)``
   Create a new variable container
``new_var(self, name, value)``
   Create a new varaible with name [string] name and value value [float]

The created varaibles are accesed through the as [Uservars].set[Name]
and [Uservars].get[Name]. In the example above this would read:

``vars.setMy_var(20)`` to set a value and ``vars.getMy_vars()`` to
retrieve the value.

Scattering factor databases
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following databases exist in GenX. All databases are accesed through
``[name].[element]`` for example: ``fp.fe``

fp
^^
The fp contains the scattering factors at Q = 0 for the 92 first
elements up to U in units of electrons or more precise Thompson
scattering lengths. The data is taken from
http://henke.lbl.gov/optical_constants/asf.html. These tables are also
known as the henke tables. According to cxros homepage: There are 500+
points on a uniform logarithmic mesh from 10 to 30,000 eV with points
added 0.1 eV above and below "sharp" absorption edges. For some elements
data on a finer mesh includes structure around absorption edges. (Below
29 eV f1 is set equal to -9999.)

bc
^^
The bc data base contains the coherent scattering length for neutrons
according to the data published in Neutron News, Vol. 3, No. 3, 1992,
pp. 29-37. The data file is taken from the Dabax library compiled by
esrf http://ftp.esrf.eu/pub/scisoft/xop2.3/DabaxFiles/. To use isotopes
just add "i" in front of the name i.e., bc.i57Fe.

f
^
The f database contains the *scattering vector dependent* isotropic
scattering factors in electrons or more precise Thompson scattering
lengths. The data is taken from the fp and the f0 database.

f0
^^
The f0 database contains the *scattering vector dependent* isotropic
scattering factors in electrons or more precise Thompson scattering
lengths. The data is the so-called Croemer-Mann tables see:
International Tables vol. 4 or vol C; in vol. C refer to pg 500-502.
Note, this is only good out to sin(theta)/lambda < 2.0 [Angstrom^-1].
The data is also fetched from the Dabax library at:
http://ftp.esrf.eu/pub/scisoft/xop2.3/DabaxFiles/ This database is the
non-dispersive (without resonant contribution).

fw
^^
Same thing as f but scaled so that it can be used with a density in
g/cm\ :sup:`3\ .`

bw
^^
Same thing as bc but scaled so that it can be used with a density in
g/cm\ :sup:`3\ .`
"""

import os

from genx.core.custom_logging import iprint

from .lib import scatteringlengths as sl

_head, _tail = os.path.split(__file__)
# Look only after the file name and not the ending since
# the file ending can be pyc if compiled...
__FILENAME__ = _tail.split(".")[0]
# This assumes that plugin is under the current dir may need
# changing
__MODULE_DIR__ = _head
if __MODULE_DIR__ == "":
    __MODULE_DIR__ = "."


class UserVars:
    def __init__(self):
        self._penalty_funcs = []

    def newVar(self, name, value):
        # Adds a new user variable to the class
        setattr(self, name, value)
        setattr(self, "set" + name[0].upper() + name[1:], lambda value: setattr(self, name, value))
        setattr(self, "get" + name[0].upper() + name[1:], lambda: getattr(self, name, value))

    new_var = newVar

    def new_sys_err(self, name, value, error, weight=1.0, correction=0.0):
        # Adds a new systematic error variable to the class that biases the FOM when changed from start value
        setattr(self, name, value)
        setattr(self, "set" + name[0].upper() + name[1:], lambda v: setattr(self, name, v))
        setattr(self, "get" + name[0].upper() + name[1:], lambda: getattr(self, name, value))
        setattr(
            self,
            "penalty" + name[0].upper() + name[1:],
            lambda: weight * (((value - getattr(self, name, value)) / error) ** 2 - correction),
        )
        self._penalty_funcs.append(getattr(self, "penalty" + name[0].upper() + name[1:]))


# ==============================================================================
# Now create default databases for scattering lengths and form factors
# for now only x-rays at single wavelengths
# The non-disersive part but angle dependent
__f0_dict__, __rho0_dict__ = sl.load_f0dabax(__MODULE_DIR__ + "/databases/f0_CromerMann.dat", create_rho=True)
# Workaround for the oxygen
__f0_dict__["o2m"] = __f0_dict__["o2-."]
__rho0_dict__["o2m"] = __rho0_dict__["o2-."]
f0 = sl.ScatteringLength(__f0_dict__)
# Dispersive part at Q = 0
__lookup_fp__ = sl.create_fp_lookup(__MODULE_DIR__ + "/databases/f1f2_nist/")
__lookup_fp_old__ = sl.create_fp_lookup(__MODULE_DIR__ + "/databases/f1f2_cxro/")


def create_fp(wavelength):
    return sl.FormFactor(wavelength, __lookup_fp__)


def create_fp_old(wavelength):
    return sl.FormFactor(wavelength, __lookup_fp_old__)


fp = create_fp(1.54)
fp_old = create_fp_old(1.54)
# The total angle dependent form factor
__lookup_f__ = sl.create_f_lookup(__lookup_fp__, __f0_dict__)
f = sl.FormFactor(1.54, __lookup_f__)
# The electrondensity of an atom
__lookup_rho__ = sl.create_rho_lookup(__lookup_fp__, __rho0_dict__, __f0_dict__)
rho = sl.FormFactor(1.54, __lookup_rho__)
# The coherent scattering length for neutrons
__bc_dict__ = sl.load_bdabax(__MODULE_DIR__ + "/databases/DeBe_NeutronNews.dat")
bc = sl.ScatteringLength(__bc_dict__)
# print 'Loading atomic weights'
__w_dict__ = sl.load_atomic_weights_dabax(__MODULE_DIR__ + "/databases/AtomicWeights.dat")
# print 'Making bw dict'
__bw_dict__ = sl.create_scatt_weight(__bc_dict__, __w_dict__)
# print 'Making bw scattering lengths'
bw = sl.ScatteringLength(__bw_dict__)

__lookup_bl__ = sl.create_bl_lookup(__MODULE_DIR__ + "/databases/geant4/g4xs_", __bc_dict__)


def create_bl(wavelength):
    return sl.FormFactor(wavelength, __lookup_bl__)


bl = create_bl(1.79819)

# print 'Making fw scattering lengths'
__lookup_fw__ = sl.create_fw_lookup(__lookup_fp__, __w_dict__)


def create_fw(wavelength):
    return sl.FormFactor(wavelength, __lookup_fw__)


fw = create_fw(1.54)

__f_chantler_dict__ = sl.read_dabax(__MODULE_DIR__ + "/databases/f1f2_Chantler.dat")
__lookup_fpc__ = sl.load_fdabax(__MODULE_DIR__ + "/databases/f1f2_Chantler.dat")
fpc = sl.FormFactor(15.4, __lookup_fpc__)

__lookup_fdisp__ = sl.create_fpdisp_lookup(__MODULE_DIR__ + "/databases/f1f2_nist/")
fd = sl.Database()
object.__setattr__(fd, "lookup_value", __lookup_fdisp__)

if __name__ == "__main__":
    # MyVars=UserVars()
    # MyVars.newVar('a',3)
    # print 'f0.fe(0)', f0.fe(0)
    # fp.set_wavelength(1.54)
    # print 'fp.fe', fp.fe
    # f.set_wavelength(1.54)
    # print 'f.fe(0)', f.fe(0)
    # print 'f.fe2p(0)', f.fe2p(0)
    # fe = np.array(__f_chantler_dict__['fe'])
    # print fe.reshape(len(fe)/7, 7)
    ##print bc.fe
    # print fpc.Sn
    # print fp.Sn
    import time

    N = 1000
    t1 = time.time()
    for i in range(N):
        fd.Fe(100.0)
    t2 = time.time()
    iprint("Dispersive database access time: ", (t2 - t1) / N)
    t1 = time.time()
    for i in range(N):
        ignore = fp.Fe
    t2 = time.time()
    iprint("Normal database access time: ", (t2 - t1) / N)
