''' interdiff is a model for specular and off specular simulations including
the effects of interdiffusion in hte calculations. The specular simulations
is conducted with Parrats recursion formula. The off-specular, diffuse
calculations are done with the distorted Born wave approximation (DWBA) as
derived by Holy and with the extensions done by Wormington to include 
diffuse interfaces.
'''

import lib.paratt as Paratt

__offspec__ = 1
try:
    import lib.offspec2_weave
except StandardError,S:
    print 'Failed to import: offspec2_weave, No off-specular simulations possible'
    print S
    __offspec__ = 0
    

from numpy import *
from lib.instrument import *

# Preamble to define the parameters needed for the models outlined below:
ModelID='MingInterdiff'
# Automatic loading of parameters possible by including this list
__pars__ = ['Layer', 'Stack', 'Sample', 'Instrument']
# Used for making choices in the GUI
instrument_string_choices = {'coords': ['q','tth'],\
    'restype': ['no conv', 'fast conv',\
     'full conv and varying res.', 'fast conv + varying res.'],\
    'footype': ['no corr', 'gauss beam', 'square beam']}
    
InstrumentParameters={'wavelength':1.54,'coords':1,'I0':1.0,'res':0.001,\
    'restype':0,'respoints':5,'resintrange':2,'beamw':0.01,'footype': 0,\
    'samplelen':10.0}
# Coordinates=1 => twothetainput
# Coordinates=0 => Q input
#Res stddev of resolution
#ResType 0: No resolution convlution
#               1: Fast convolution
#               2: Full Convolution +varying resolution
#               3: Fast convolution varying resolution
#ResPoints Number of points for the convolution only valid for ResolutionType=2
#ResIntrange Number of standard deviatons to integrate over default 2
# Parameters for footprint coorections
# Footype: 0: No corections for footprint
#          1: Correction for Gaussian beam => Beaw given in mm and stddev
#          2: Correction for square profile => Beaw given in full width mm
# Samlen= Samplelength in mm.

LayerParameters = {'sigmai':0.0, 'sigmar':0.0, 'dens':1.0, 'd':0.0,\
    'f':0.0+0.0j}
StackParameters = {'Layers':[], 'Repetitions':1}
SampleParameters = {'Stacks':[], 'Ambient':None, 'Substrate':None, 'h':1.0,\
    'eta_z':10.0, 'eta_x':10.0}

def Specular(TwoThetaQz, sample, instrument):
    # preamble to get it working with my class interface
    restype = instrument.getRestype()

    if restype == 2 or restype == instrument_string_choices['restype'][2]:
            (TwoThetaQz,weight) = ResolutionVector(TwoThetaQz[:], \
                instrument.getRes(), instrument.getRespoints(),\
                 range=instrument.getResintrange())
    if instrument.getCoords() == 1 or\
        instrument.getCoords() == instrument_string_choices['coords'][1]:
        theta = TwoThetaQz/2
    elif instrument.getCoords() == 0 or\
        instrument.getCoords() == instrument_string_choices['coords'][0]:
        theta = arcsin(TwoThetaQz/4/pi*instrument.getWavelength())*180./pi
    
    lamda = instrument.getWavelength()
    parameters = sample.resolveLayerParameters()
    dens = array(parameters['dens'], dtype = complex64)
    f = array(parameters['f'], dtype = complex64)
    re = 2.82e-13*1e2/1e-10
    n = 1 - dens*re*lamda**2/2/pi*f*1e-4
    d = array(parameters['d'], dtype = float64)
    d = d[1:-1]
    sigmar = array(parameters['sigmar'], dtype = float64)
    sigmar = sigmar[:-1]
    sigmai = array(parameters['sigmai'], dtype = float64)
    sigmai = sigmai[:-1]
    sigma = sqrt(sigmai**2 + sigmar**2)
    #print sigma
    
    R = Paratt.Refl(theta, lamda, n, d, sigma)*instrument.getI0()

    #FootprintCorrections
    
    foocor = 1.0
    footype = instrument.getFootype()
    beamw = instrument.getBeamw()
    samlen = instrument.getSamplelen()
    if footype == 0 or footype == instrument_string_choices['footype'][0]:
        foocor = 1.0
    elif footype == 1 or footype == instrument_string_choices['footype'][1]:
        foocor = GaussIntensity(theta, samlen/2.0, samlen/2.0, beamw)
    elif footype == 2 or footype == instrument_string_choices['footype'][2]:
        foocor = SquareIntensity(theta, samlen, beamw)
    else:
        raise ValueError('Variable footype has an unvalid value')
    
    if restype == 0 or restype == instrument_string_choices['restype'][0]:
        pass
    elif restype == 1 or restype == instrument_string_choices['restype'][1]:
        R = ConvoluteFast(TwoThetaQz,R[:]*foocor, instrument.getRes(),\
            range = instrument.getResintrange())
    elif restype == 2 or restype == instrument_string_choices['restype'][2]:
        R = ConvoluteResolutionVector(TwoThetaQz,R[:]*foocor, weight)
    elif restype == 3 or restype == instrument_string_choices['restype'][3]:
        R = ConvoluteFastVar(TwoThetaQz,R[:]*foocor, instrument.getRes(),\
          range = instrument.getResintrange())
    else:
        raise ValueError('Variable restype has an unvalid value')
    return R

def OffSpecularMingInterdiff(TwoThetaQz, ThetaQx, sample, instrument):
    lamda = instrument.getWavelength()
    if instrument.getCoordinates() == 1: # Sample Coords is theta-2theta
        alphaR1 = ThetaQx
        betaR1 = TwoThetaQz - ThetaQx
        qx = 2*pi/lamda*(cos(alphaR1*pi/180) - cos(betaR1*pi/180))
        qz = 2*pi/lamda*(sin(alphaR1*pi/180) + sin(betaR1*pi/180))
    else:
        qz = TwoThetaQz
        qx = ThetaQx

    #print qx
    #print qz
    parameters = sample.resolveLayerParameters()
    def toarray(a, code):
        a.reverse()
        return array(a, dtype = code)
    dens = array(parameters['dens'], dtype = complex64)
    f = array(parameters['f'], dtype = complex64)
    re = 2.82e-13*1e2/1e-10
    n = 1 - dens*re*lamda**2/2/pi*f*1e-4
    n = toarray(n, code = complex64)
    sigmar = toarray(parameters['sigmar'], code = float64)
    sigmar = sigmar[1:]
    #print sigmar
    sigmai = toarray(parameters['sigmai'],code = float64)
    sigmai = sigmai[1:] + 1e-5
    #print sigmai
    d=toarray(parameters['d'], code = float64)
    d=r_[0, d[1:-1]]
    #print d
    z = -cumsum(d)
    #print z
    eta = sample.getEta_x()
    #print eta
    h = sample.getH()
    #print h
    eta_z = sample.getEta_z()
    #print eta_z
    if __offspec__:
        (I, alpha, omega) = offspec2_weave.DWBA_Interdiff(qx, qz, lamda, n, z,\
            sigmar, sigmai, eta, h, eta_z, d)
    else:
        I=ones(len(qx*qz))
    return real(I)*instrument.getI0()

SimulationFunctions = {'Specular':Specular,\
                        'OffSpecular':OffSpecularMingInterdiff}

import lib.refl as Refl
(Instrument, Layer, Stack, Sample) = Refl.MakeClasses(InstrumentParameters,\
        LayerParameters,StackParameters,\
         SampleParameters, SimulationFunctions, ModelID)


if __name__=='__main__':
    Fe=Layer(d=10,sigmar=3.0,n=1-2.247e-5+2.891e-6j)
    Si=Layer(d=15,sigmar=3.0,n=1-7.577e-6+1.756e-7j)
    sub=Layer(sigmar=3.0,n=1-7.577e-6+1.756e-7j)
    amb=Layer(n=1.0,sigmar=1.0)
    stack=Stack(Layers=[Fe,Si],Repetitions=20)
    sample=Sample(Stacks=[stack],Ambient=amb,Substrate=sub,eta_z=500.0,eta_x=100.0)
    print sample
    inst=Instrument(Wavelength=1.54,Coordinates=1)
