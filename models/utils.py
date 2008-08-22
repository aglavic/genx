import numpy as np
import lib.scatteringlengths as sl
import os

head, tail = os.path.split(__file__)
# Look only after the file name and not the ending since
# the file ending can be pyc if compiled... 
__FILENAME__ = tail.split('.')[0]
# This assumes that plugin is under the current dir may need 
# changing
__MODULE_DIR__ = head
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

#==============================================================================
# Now create default databases for scattering lengths and form factors
# for now only x-rays at single wavelengths
# The non-disersive part but angle dependent
__f0_dict__ = sl.load_f0dabax(__MODULE_DIR__+'/databases/f0_CromerMann.dat')
f0 = sl.ScatteringLength(__f0_dict__)
# Dispersive part at Q = 0
__lookup_fp__ = sl.create_fp_lookup(__MODULE_DIR__+'/databases/f1f2_cxro/')
fp = sl.FormFactor(1.54, __lookup_fp__)
# The total angle dependent form factor
__lookup_f__ = sl.create_f_lookup(__lookup_fp__, __f0_dict__)
f = sl.FormFactor(1.54, __lookup_f__)

if __name__=='__main__':
    MyVars=UserVars()
    MyVars.newVar('a',3)
    print 'f0.fe(0)', f0.fe(0)
    fp.set_wavelength(1.54)
    print 'fp.fe', fp.fe
    f.set_wavelength(1.54)
    print 'f.fe(0)', f.fe(0)
    print 'f.fe2p(0)', f.fe2p(0)