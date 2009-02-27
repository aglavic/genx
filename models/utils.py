'''<h1>Utilities for GenX and scattering</h1>
This library contains nice to have functions and classes. Most noteworthy
is the UserVars class. The scattering factor tables can also come handy.

<h2>Classes</h2>

<h3>UserVars</h3>
This class is used to contain user defined variables which is problem specific
and do not warrant a completly new model library.
It is contstructed as: <code>vars = UserVars()</code>.<br>
A new variable is created by calling the new_var method: 
<code>vars.new_var('my_var', 10.0)</code><br>
The methods in this class are:
<dl>
    <dt><code><b>__init__(self)</b></code></dt>
    <dd>Create a new variable container</dd>
    <dt><code><b>new_var(self, name, value)</b></code></dt>
    <dd>Create a new varaible with name [string] name and value value [float]
    </dd>
</dl>
The created varaibles are accesed through the as [Uservars].set[Name] and
[Uservars].get[Name]. In the example above this would read:
 <code>vars.setMy_var(20)</code> to set a value and 
<code>vars.getMy_vars()</code> to retrieve the value.


<h3>Scattering factor databases</h3>
The following databases exist in GenX. All databases are accesed through
<code>[name].[element]</code> for example: <code>fp.fe</code>

<h4>fp</h4>
The fp contains the scattering factors at Q = 0 for the 92 first elements up
to U in units of electrons or more precise Thompson scattering lengths. 
The data is taken from <a href = "http://henke.lbl.gov/optical_constants/asf.html">
http://henke.lbl.gov/optical_constants/asf.html</a>. These tables are also 
known as the henke tables. According to
cxros homepage: 
There are 500+ points on a uniform logarithmic mesh from 
10 to 30,000 eV with points added 0.1 eV above and 
below "sharp" absorption edges. For some elements data on a 
finer mesh includes structure around absorption edges. 
(Below 29 eV f1 is set equal to -9999.) 

<h4>bc</h4>
The bc data base contains the coherent scattering length for neutrons according
to the data published in Neutron News, Vol. 3, No. 3, 1992, pp.  29-37.
The data file is taken from the Dabax library compiled by esrf
<a href = "http://ftp.esrf.eu/pub/scisoft/xop2.3/DabaxFiles/">
http://ftp.esrf.eu/pub/scisoft/xop2.3/DabaxFiles/</a>.
To use isotopes just add "i" in front of the name i.e., bc.i57Fe.

<h4>f</h4>
The f database contains the <em>scattering vector dependent</em> isotropic
scattering factors in electrons or more precise Thompson scattering lengths.
The data is taken from the fp and the f0 database.

<h4>f0</h4>
The f0 database contains the <em>scattering vector dependent</em> isotropic
scattering factors in electrons or more precise Thompson scattering lengths.
The data is the so-called Croemer-Mann tables see:
International Tables vol. 4 or vol C; in vol. C refer to pg 500-502.
 Note, this is only good out to sin(theta)/lambda < 2.0 [Angstrom^-1].
The data is also fetched from the Dabax library at:
<a href = "http://ftp.esrf.eu/pub/scisoft/xop2.3/DabaxFiles/">
http://ftp.esrf.eu/pub/scisoft/xop2.3/DabaxFiles/</a>
This database is the non-dispersive (without resonant contribution).

<h4>fw<h4>
Same thing as f but scaled so that is can be used with a density
in g/cm<sup>3<sup>.

<h4>bw<h4>
Same thing as bc but scaled so that is can be used with a density
in g/cm<sup>3<sup>.
'''
import numpy as np
import lib.scatteringlengths as sl
import os

_head, _tail = os.path.split(__file__)
# Look only after the file name and not the ending since
# the file ending can be pyc if compiled... 
__FILENAME__ = _tail.split('.')[0]
# This assumes that plugin is under the current dir may need 
# changing
__MODULE_DIR__ = _head
if __MODULE_DIR__ == '':
    __MODULE_DIR__ = '.'


class UserVars:
    def __init__(self):
        pass

    def newVar(self,name,value):
        #name=name.lower()
        setattr(self,name,value)
        setattr(self,'set'+name[0].upper()+name[1:],lambda value:setattr(self,name,value))
        setattr(self,'get'+name[0].upper()+name[1:],lambda :getattr(self,name,value))
        
    def new_var(self,name,value):
        #name=name.lower()
        setattr(self,name,value)
        setattr(self,'set'+name[0].upper()+name[1:],lambda value:setattr(self,name,value))
        setattr(self,'get'+name[0].upper()+name[1:],lambda :getattr(self,name,value))

#==============================================================================
# Now create default databases for scattering lengths and form factors
# for now only x-rays at single wavelengths
# The non-disersive part but angle dependent
__f0_dict__ = sl.load_f0dabax(__MODULE_DIR__+'/databases/f0_CromerMann.dat')
# Workaround for the oxygen
__f0_dict__['o2m'] = __f0_dict__['o2-.']
f0 = sl.ScatteringLength(__f0_dict__)
# Dispersive part at Q = 0
__lookup_fp__ = sl.create_fp_lookup(__MODULE_DIR__+'/databases/f1f2_cxro/')
fp = sl.FormFactor(1.54, __lookup_fp__)
# The total angle dependent form factor
__lookup_f__ = sl.create_f_lookup(__lookup_fp__, __f0_dict__)
f = sl.FormFactor(1.54, __lookup_f__)
# The coherent scattering length for neutrons
__bc_dict__ = sl.load_bdabax(__MODULE_DIR__+'/databases/DeBe_NeutronNews.dat')
bc = sl.ScatteringLength(__bc_dict__)
#print 'Loading atomic weights'
__w_dict__ = sl.load_atomic_weights_dabax(__MODULE_DIR__ +\
                        '/databases/AtomicWeights.dat')
#print 'Making bw dict'
__bw_dict__ = sl.create_scatt_weight(__bc_dict__, __w_dict__)
#print 'Making bw scattering lengths'
bw = sl.ScatteringLength(__bw_dict__)
#print 'Making fw scattering lengths'
__lookup_fw__ = sl.create_fw_lookup(__lookup_fp__, __w_dict__)
fw = sl.FormFactor(1.54, __lookup_fw__)


if __name__=='__main__':
    MyVars=UserVars()
    MyVars.newVar('a',3)
    print 'f0.fe(0)', f0.fe(0)
    fp.set_wavelength(1.54)
    print 'fp.fe', fp.fe
    f.set_wavelength(1.54)
    print 'f.fe(0)', f.fe(0)
    print 'f.fe2p(0)', f.fe2p(0)
