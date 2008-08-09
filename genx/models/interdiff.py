
import lib.paratt as Paratt

__offspec__ = 1
try:
    import offspec2_weave
except StandardError,S:
    print 'Failed to import: offspec2_weave, No off-specular simulations possible'
    print S
    __offspec__ = 0
    

from numpy import *
from lib.instrument import *

# Preamble to define the parameters needed for the models outlined below:
ModelID='MingInterdiff'
# Used for making choices in the GUI
InstrumntGUIChoices = {'Coordinates': ['Q','2Theta'],\
    'Restype': ['No convolution', 'Fast convolution',\
     'Full convolution + varying res.', 'Fast convolution + varying res.'],\
    'Footype': ['No correction', 'Gaussian beam', 'Square beam']}
    
InstrumentParameters={'Wavelength':1.54,'Coordinates':1,'I0':1.0,'Res':0.001,\
    'Restype':0,'Respoints':5,'Resintrange':2,'Beaw':0.01,'Footype':0.0,\
    'Samlen':10.0}
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
#            1: Correction for Gaussian beam => Beaw given in mm and stddev
#            2: Correction for square profile => Beaw given in full width mm
# Samlen= Samplelength in mm.

LayerParameters = {'sigmai':0.0, 'sigmar':0.0, 'reldens':1.0, 'd':0.0,\
    'n':1.0+0.0j}
StackParameters = {'Layers':[], 'Repetitions':1}
SampleParameters = {'Stacks':[], 'Ambient':None, 'Substrate':None, 'h':1.0,\
    'eta_z':10.0, 'eta_x':10.0}

def Specular(TwoThetaQz, sample, instrument):
    # preamble to get it working with my class interface
    restype = instrument.getRestype()

    if restype == 2:
            (TwoThetaQz,weight) = ResolutionVector(TwoThetaQz[:], \
                instrument.getRes(), instrument.getRespoints(),\
                 range=instrument.getResintrange())
    if instrument.getCoordinates() == 1:
        theta = TwoThetaQz/2
    else:
        theta = arcsin(TwoThetaQz/4/pi*instrument.getWavelength())*180./pi
    
    lamda = instrument.getWavelength()
    parameters = sample.resolveLayerParameters()
    n = array(parameters['n'], dtype = complex64)
    reldens = array(parameters['reldens'], dtype = complex64)
    d = array(parameters['d'], dtype = float64)
    d = d[1:-1]
    sigmar = array(parameters['sigmar'], dtype = float64)
    sigmar = sigmar[:-1]
    sigmai = array(parameters['sigmai'], dtype = float64)
    sigmai = sigmai[:-1]
    sigma = sqrt(sigmai**2 + sigmar**2)
    #print sigma
    
    R=Paratt.Refl(theta, lamda,\
            (n-1.0)*reldens + 1.0,d,sigma)*instrument.getI0()

    #FootprintCorrections
    
    foocor = 1.0
    footype = instrument.getFootype()
    beamw = instrument.getBeaw()
    samlen = instrument.getSamlen()
    if footype == 1:
        foocor = GaussIntensity(theta, samlen/2.0, samlen/2.0, beamw)
    elif footype == 2:
        foocor = SquareIntensity(theta, samlen, beamw)
        
    
    if restype == 1:
        R = ConvoluteFast(TwoThetaQz,R[:]*foocor, instrument.getRes(),\
            range = instrument.getResintrange())
    elif restype == 2:
        R = ConvoluteResolutionVector(TwoThetaQz,R[:]*foocor, weight)
    elif restype == 3:
        R = ConvoluteFastVar(TwoThetaQz,R[:]*foocor, instrument.getRes(),\
          range = instrument.getResintrange())
    return R

def OffSpecularMingInterdiff(TwoThetaQz, ThetaQx,sample, instrument):
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
    def toarray(a,code):
        a.reverse()
        return array(a, dtype = code)
    n = toarray(parameters['n'], code = complex64)
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
