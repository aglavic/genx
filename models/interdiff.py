''' <h1> Library for specular and off-specular x-ray reflectivity</h1>
interdiff is a model for specular and off specular simulations including
the effects of interdiffusion in hte calculations. The specular simulations
is conducted with Parrats recursion formula. The off-specular, diffuse
calculations are done with the distorted Born wave approximation (DWBA) as
derived by Holy and with the extensions done by Wormington to include 
diffuse interfaces.
<h2>Classes</h2>
<h3>Layer</h3>
<code> Layer(b = 0.0, d = 0.0, f = 0.0+0.0J, dens = 1.0, magn_ang = 0.0, magn = 0.0, sigma = 0.0)</code>
    <dl>
    <dt><code><b>d</b></code></dt>
    <dd>The thickness of the layer in AA (Angstroms = 1e-10m)</dd>
    <dt><code><b>f</b></code></dt>
    <dd>The x-ray scattering length per formula unit in electrons. To be strict it is the
    number of Thompson scattering lengths for each formula unit.</dd>
    <dt><code><b>dens</b></code></dt>
    <dd>The density of formula units in units per Angstroms. Note the units!</dd>
    <dt><code><b>sigmai</b></code></dt>
    <dd>The root mean square <em>interdiffusion</em> of the top interface of the layer in Angstroms.</dd>		
    <dt><code><b>sigmar</b></code></dt>
    <dd>The root mean square <em>roughness</em> of the top interface of the layer in Angstroms.</dd>
    </dl>
<h3>Stack</h3>
<code> Stack(Layers = [], Repetitions = 1)</code>
    <dl>
    <dt><code><b>Layers</b></code></dt>
    <dd>A <code>list</code> consiting of <code>Layer</code>s in the stack
    the first item is the layer closest to the bottom</dd>
    <dt><code><b>Repetitions</b></code></dt>
    <dd>The number of repsetions of the stack</dd>
    </dl>
<h3>Sample</h3>
<code> Sample(Stacks = [], Ambient = Layer(), Substrate = Layer(), eta_z = 10.0,
    eta_x = 10.0, h = 1.0)</code>
    <dl>
    <dt><code><b>Stacks</b></code></dt>
    <dd>A <code>list</code> consiting of <code>Stack</code>s in the stacks
    the first item is the layer closest to the bottom</dd>
    <dt><code><b>Ambient</b></code></dt>
    <dd>A <code>Layer</code> describing the Ambient (enviroment above the sample).
     Only the scattering lengths and density of the layer is used.</dd>
    <dt><code><b>Substrate</b></code></dt>
    <dd>A <code>Layer</code> describing the substrate (enviroment below the sample).
     Only the scattering lengths, density and  roughness of the layer is used.</dd>
    <dt><code><b>eta_z</b></code></dt>
    <dd>The out-of plane (vertical) correlation length of the roughness
    in the sample. Given in AA. </dd>
    <dt><code><b>eta_x</b></code></dt>
    <dd>The in-plane global correlation length (it is assumed equal for all layers).
    Given in AA.</dd>
    <dt><code><b>h</b></code></dt>
    <dd>The jaggedness parameter, should be between 0 and 1.0. This describes
    how jagged the interfaces are. This is also a global parameter for all
    interfaces.</dd>
    </dl>
    
<h3>Instrument</h3>
<code>Instrument(wavelength = 1.54, coords = 'tth',
     I0 = 1.0 res = 0.001, restype = 'no conv', respoints = 5, resintrange = 2,
     beamw = 0.01, footype = 'no corr', samplelen = 10.0, taylor_n = 1)</code>
    <dl>
    <dt><code><b>wavelength</b></code></dt>
    <dd>The wavalelngth of the radiation givenin AA (Angstroms)</dd>
    <dt><code><b>coords</b></code></dt>
    <dd>The coordinates of the data given to the SimSpecular function.
    The available alternatives are: 'q' or 'tth'. Alternatively the numbers
    0 (q) or 1 (tth) can be used.</dd>
    <dt><code><b>I0</b></code></dt>
    <dd>The incident intensity (a scaling factor)</dd>
    <dt><code><b>Ibkg</b></code></dt>
    <dd>The background intensity. Added as a constant value to the calculated
    reflectivity</dd>
    <dt><code><b>res</b></code></dt>
    <dd>The resolution of the instrument given in the coordinates of
     <code>coords</code>. This assumes a gaussian reloution function and
    <code>res</code> is the standard deviation of that gaussian.</dd>
    <dt><code><b>restype</b></code></dt>
    <dd>Describes the rype of the resolution calculated. One of the alterantives:
    'no conv', 'fast conv', 'full conv and varying res.' or 'fast conv + varying res.'.
    The respective numbers 0-3 also works. Note that fast convolution only alllows
    a single value into res wheras the other can also take an array with the
    same length as the x-data (varying resolution)</dd>
    <dt><code><b>respoints</b></code></dt>
    <dd>The number of points to include in the resolution calculation. This is only
    used for 'full conv and vaying res.' and 'fast conv + varying res'</dd>
    <dt><code><b>resintrange</b></code></dt>
    <dd>Number of standard deviatons to integrate the resolution fucntion times
    the relfectivty over</dd>
    <dt><code><b>footype</b></code></dt>
    <dd>Which type of footprint correction is to be applied to the simulation.
    One of: 'no corr', 'gauss beam' or 'square beam'. Alternatively, 
    the number 0-2 are also valid. The different choices are self explanatory.
    </dd>
    <dt><code><b>beamw</b></code></dt>
    <dd>The width of the beam given in mm. For 'gauss beam' it should be
    the standard deviation. For 'square beam' it is the full width of the beam.</dd>
    <dt><code><b>samplelen</b></code></dt>
    <dd>The length of the sample given in mm</dd>
    <dt><code><b>taylor_n</b></code></dt>
    <dd>The number terms taken into account in the taylor expansion of 
    the fourier integral of the correlation function. More terms more accurate
    calculation but also much slower.</dd>
'''

import lib.paratt as Paratt

__offspec__ = True
try:
    import lib.offspec2_weave
except Exception,S:
    print 'Failed to import: offspec2_weave, No off-specular simulations possible'
    print S
    __offspec__ = False
    

from numpy import *
from scipy.special import erf
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
    
InstrumentParameters={'wavelength':1.54,'coords':'tth','I0':1.0,'res':0.001,\
    'restype':'no conv','respoints':5,'resintrange':2,'beamw':0.01,'footype': 'no corr',\
    'samplelen':10.0, 'Ibkg': 0.0, 'taylor_n': 1}
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
    'f':0.0+1.0j}
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
    #print [type(f) for f in parameters['f']]
    f = array(parameters['f'], dtype = complex64)
    re = 2.82e-13*1e2/1e-10
    n = 1 - dens*re*lamda**2/2/pi*f*1e-4
    d = array(parameters['d'], dtype = float64)
    #d = d[1:-1]
    sigmar = array(parameters['sigmar'], dtype = float64)
    #sigmar = sigmar[:-1]
    sigmai = array(parameters['sigmai'], dtype = float64)
    #sigmai = sigmai[:-1]
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
        R = R[:]*foocor
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
    return R + instrument.getIbkg()

def OffSpecularMingInterdiff(TwoThetaQz, ThetaQx, sample, instrument):
    lamda = instrument.getWavelength()
    if instrument.getCoords() == 1: # Sample Coords is theta-2theta
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
        a = list(a)
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
        (I, alpha, omega) = lib.offspec2_weave.DWBA_Interdiff(qx, qz, lamda, n, z,\
            sigmar, sigmai, eta, h, eta_z, d,\
                taylor_n = instrument.getTaylor_n())
    else:
        I=ones(len(qx*qz))
    return real(I)*instrument.getI0() + instrument.getIbkg()

def SLD_calculations(z, sample, inst):
    ''' Calculates the scatteringlength density as at the positions z
    '''
    parameters = sample.resolveLayerParameters()
    dens = array(parameters['dens'], dtype = complex64)
    f = array(parameters['f'], dtype = complex64)
    sld = dens*f
    d_sld = sld[:-1] - sld[1:]
    d = array(parameters['d'], dtype = float64)
    d = d[1:-1]
    # Include one extra element - the zero pos (substrate/film interface)
    int_pos = cumsum(r_[0,d])
    sigmar = array(parameters['sigmar'], dtype = float64)
    sigmar = sigmar[:-1]
    sigmai = array(parameters['sigmai'], dtype = float64)
    sigmai = sigmai[:-1]
    sigma = sqrt(sigmai**2 + sigmar**2)+1e-7
    if z == None:
        z = arange(-sigma[0]*5, int_pos.max()+sigma[-1]*5, 0.5)
    rho = sum(d_sld*(0.5 - 0.5*erf((z[:,newaxis]-int_pos)/sqrt(2.)/sigma)), 1) + sld[-1]
    return {'real sld': real(rho), 'imag sld': imag(rho), 'z':z}
    

SimulationFunctions = {'Specular':Specular,\
                        'OffSpecular':OffSpecularMingInterdiff,\
                        'SLD': SLD_calculations}

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
