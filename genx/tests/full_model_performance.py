# run performance tests on spec_nx model with CPU numba to see performance of stack
# used to profile the model performance
import sys
import genx.models
sys.modules['models']=sys.modules['genx.models']
from genx.models.lib import paratt_numba, neutron_numba, paratt_cuda, neutron_cuda

import genx.models.spec_nx as model
from genx.models.utils import UserVars, fp, fw, bc, bw
from numpy import sqrt, linspace, pi

# BEGIN Instrument DO NOT CHANGE
inst = model.Instrument(footype = 'gauss beam',probe = 'neutron pol spin flip',beamw = 0.2,resintrange = 3,tthoff = 0.0,
                        pol = 'uu',wavelength = 4.4,respoints = 15,Ibkg = 0.0,I0 = 2,samplelen = 50.0,
                        restype = 'full conv and varying res.',coords = 'q',res = 0.001,incangle = 0.0)
fp.set_wavelength(inst.wavelength)
#Compability issues for pre-fw created gx files
try:
	 fw
except:
	pass
else:
	fw.set_wavelength(inst.wavelength)
# END Instrument

# BEGIN Sample DO NOT CHANGE
Amb = model.Layer(b = 0, d = 0.0, f = (1e-20+1e-20j), dens = 1.0, magn_ang = 0.0, sigma = 0.0, xs_ai = 0.0, magn = 0.0)
SiO = model.Layer(b = bc.Si + bc.O*2, d = 1205, f = (1e-20+1e-20j), dens = 0.026, magn_ang = 0.0, sigma = 2, xs_ai = 0.0, magn = 0.0)
Sub = model.Layer(b = bc.Si, d = 0.0, f = (1e-20+1e-20j), dens = 8/5.443**3, magn_ang = 0.0, sigma = 2, xs_ai = 0.0, magn = 0.0)

surf = model.Stack(Layers=[SiO], Repetitions = 100)

sample = model.Sample(Stacks = [surf], Ambient = Amb, Substrate = Sub)
# END Sample

# BEGIN Parameters DO NOT CHANGE
cp = UserVars()
cp.new_var('dtheta', 0.04)
cp.new_var('dlol', 0.007)
# END Parameters

def Sim(data):
    I = []
    # BEGIN Dataset 0 DO NOT CHANGE
    inst.setRes(sqrt((cp.dlol*data[0].x)**2 + (4*3.1415/4.4*cp.dtheta*pi/360)**2))
    I.append(sample.SimSpecular(data[0].x, inst))
    # END Dataset 0
    return I

class empty():
    pass
e=empty()
e.x=linspace(0.001, 0.3, 500)

print("Numba")
model.Paratt.ReflQ=paratt_numba.ReflQ
model.MatrixNeutron.Refl=neutron_numba.Refl
for i in range(100):
    model.Buffer.parameters=None
    Sim([e])
print("Cuda")
model.Paratt.ReflQ=paratt_cuda.ReflQ
model.MatrixNeutron.Refl=neutron_cuda.Refl
for i in range(100):
    model.Buffer.parameters=None
    Sim([e])
