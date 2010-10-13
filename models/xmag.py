''' <h1> Library for specular magnetic x-ray reflectivity</h1>
The magnetic reflectivity is calculated according to: S.A. Stephanov and S.K Shina PRB 61 15304.
Note: The documentation is not updated from the interdiff model!
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

import lib.xrmr

from numpy import *
from scipy.special import erf
from lib.instrument import *

# Preamble to define the parameters needed for the models outlined below:
ModelID='StephanovXRMR'
# Automatic loading of parameters possible by including this list
__pars__ = ['Layer', 'Stack', 'Sample', 'Instrument']
# Used for making choices in the GUI
instrument_string_choices = {'coords': ['q','tth'],
                             'restype': ['no conv', 'fast conv',
                                         'full conv and varying res.', 
                                         'fast conv + varying res.'],
                             'footype': ['no corr', 'gauss beam', 
                                         'square beam'],
                             'pol':['circ+','circ-','tot', 'ass', 'sigma', 'pi']
     }

    
InstrumentParameters={'wavelength':1.54,'coords':'tth','I0':1.0,'res':0.001,\
    'restype':'no conv','respoints':5,'resintrange':2,'beamw':0.01,'footype': 'no corr',\
    'samplelen':10.0, 'Ibkg': 0.0, 'pol':'circ+'}
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
#
#

LayerParameters = {'dens':1.0, 'd':0.0, 'fc':(0.0 + 1e-20J), 
                   'fm1':(0.0 + 1e-20J), 'fm2':(0.0 + 1e-20J), 
                   'phi_m':0.0, 'theta_m':0.0, 'mag_dens':1.0}
StackParameters = {'Layers':[], 'Repetitions':1}
SampleParameters = {'Stacks':[], 'Ambient':None, 'Substrate':None}

# A buffer to save previous calculations for spin-flip calculations
class Buffer:
    W = None
    parameters = None

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
    dens = array(parameters['dens'], dtype = complex128)
    dens = array(parameters['mag_dens'], dtype = complex128)
    #print [type(f) for f in parameters['f']]
    fc = array(parameters['fc'], dtype = complex128) + (1-1J)*1e-20
    fm1 = array(parameters['fm1'], dtype = complex128) + (1-1J)*1e-20
    fm2 = array(parameters['fm2'], dtype = complex128) + (1-1J)*1e-20
    re = 2.8179402894e-5
    A = -lamda**2*re/pi*dens*fc
    B = lamda**2*re/pi*dens*fm1
    C = lamda**2*re/pi*dens*fm2
    d = array(parameters['d'], dtype = float64)
    g_0 = sin(theta*pi/180.0)
    phi = array(parameters['phi_m'], dtype = float64)*pi/180.0
    theta = array(parameters['theta_m'], dtype = float64)*pi/180.0
    M = c_[cos(theta)*cos(phi), cos(theta)*sin(phi), sin(theta)]
    #print A[::-1], B[::-1], d[::-1], M[::-1], lamda, g_0
    W = lib.xrmr.calc_refl(g_0, lamda, A[::-1], 0.0*A[::-1], B[::-1], C[::-1], M[::-1], d[::-1])
    trans = ones(W.shape, dtype = complex128); trans[0,1] = 1.0J; trans[1,1] = -1.0J; trans = trans/sqrt(2)
    Wc = lib.xrmr.dot2(trans, lib.xrmr.dot2(W, lib.xrmr.inv2(trans)))
    #Different polarization channels:
    pol = instrument.getPol()
    if pol == 0 or pol == instrument_string_choices['pol'][0]:
        R = abs(Wc[0,0])**2 + abs(Wc[0,1])**2
    elif pol == 1 or pol == instrument_string_choices['pol'][1]:
        R = abs(Wc[1,0])**2 + abs(Wc[1,1])**2
    elif pol == 2 or pol == instrument_string_choices['pol'][2]:
        R = (abs(W[0,0])**2 + abs(W[1,0])**2 + abs(W[0,1])**2 + abs(W[1,1])**2)/2
    elif pol == 3 or pol == instrument_string_choices['pol'][3]:
        R = 2*(W[0,0]*W[0,1].conj() + W[1,0]*W[1,1].conj()).imag/(abs(W[0,0])**2 + abs(W[1,0])**2 + abs(W[0,1])**2 + abs(W[1,1])**2)
    elif pol == 4 or pol == instrument_string_choices['pol'][4]:
        R = abs(W[0,0])**2 + abs(W[0,1])**2
    elif pol == 5 or pol == instrument_string_choices['pol'][5]:
        R = abs(W[1,0])**2 + abs(W[1,1])**2
    else:
        raise ValueError('Variable pol has an unvalid value')

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
    return R*instrument.getI0() + instrument.getIbkg()

def OffSpecular(TwoThetaQz, ThetaQx, sample, instrument):
   raise NotImplementedError('Off specular calculations are not implemented for magnetic x-ray reflectivity')


def SLD_calculations(z, sample, inst):
    ''' Calculates the scatteringlength density as at the positions z
    '''
    parameters = sample.resolveLayerParameters()
    dens = array(parameters['dens'], dtype = complex64)
    mag_dens = array(parameters['mag_dens'], dtype = complex64)
    fc = array(parameters['fc'], dtype = complex64)
    sldc = dens*fc
    d_sldc = sldc[:-1] - sldc[1:]
    fm1 = array(parameters['fm1'], dtype = complex64)
    fm2 = array(parameters['fm2'], dtype = complex64)
    sldm1 = mag_dens*fm1
    sldm2 = mag_dens*fm2
    d_sldm1 = sldm1[:-1] - sldm1[1:]
    d_sldm2 = sldm2[:-1] - sldm2[1:]
    d = array(parameters['d'], dtype = float64)
    d = d[1:-1]
    # Include one extra element - the zero pos (substrate/film interface)
    int_pos = cumsum(r_[0,d])
    sigma = int_pos*0.0+1e-7
    if z == None:
        z = arange(min(-sigma[0]*5, -5), max(int_pos.max()+sigma[-1]*5, 5), 0.5)
    rho_c = sum(d_sldc*(0.5 - 0.5*erf((z[:,newaxis]-int_pos)/sqrt(2.)/sigma)), 1) + sldc[-1]
    rho_m1 = sum(d_sldm1*(0.5 - 0.5*erf((z[:,newaxis]-int_pos)/sqrt(2.)/sigma)), 1) + sldm1[-1]
    rho_m2 = sum(d_sldm2*(0.5 - 0.5*erf((z[:,newaxis]-int_pos)/sqrt(2.)/sigma)), 1) + sldm2[-1]
    
    return {'real charge sld': real(rho_c), 'imag charge sld': imag(rho_c),
            'real mag_1 sld': real(rho_m1), 'imag mag_1 sld': imag(rho_m1),
            'real mag_2 sld': real(rho_m2), 'imag mag_2 sld': imag(rho_m2),
            'z':z}
    

SimulationFunctions = {'Specular':Specular,\
                        'OffSpecular':OffSpecular,\
                        'SLD': SLD_calculations}

import lib.refl as Refl
(Instrument, Layer, Stack, Sample) = Refl.MakeClasses(InstrumentParameters,\
        LayerParameters,StackParameters,\
         SampleParameters, SimulationFunctions, ModelID)


if __name__=='__main__':
    pass
