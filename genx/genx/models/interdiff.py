'''
Library for specular and off-specular x-ray reflectivity
========================================================
interdiff is a model for specular and off specular simulations including
the effects of interdiffusion in hte calculations. The specular
simulations is conducted with Parrats recursion formula. The
off-specular, diffuse calculations are done with the distorted Born wave
approximation (DWBA) as derived by Holy and with the extensions done by
Wormington to include diffuse interfaces.

Classes
-------

Layer
~~~~~
``Layer(b = 0.0, d = 0.0, f = 0.0+0.0J, dens = 1.0, magn_ang = 0.0, magn = 0.0, sigma = 0.0)``

``d``
   The thickness of the layer in AA (Angstroms = 1e-10m)
``f``
   The x-ray scattering length per formula unit in electrons. To be
   strict it is the number of Thompson scattering lengths for each
   formula unit.
``dens``
   The density of formula units in units per Angstroms. Note the units!
``sigmai``
   The root mean square *interdiffusion* of the top interface of the
   layer in Angstroms.
``sigmar``
   The root mean square *roughness* of the top interface of the layer in
   Angstroms.

Stack
~~~~~
``Stack(Layers = [], Repetitions = 1)``

``Layers``
   A ``list`` consiting of ``Layer``\ s in the stack the first item is
   the layer closest to the bottom
``Repetitions``
   The number of repsetions of the stack

Sample
~~~~~~
``Sample(Stacks = [], Ambient = Layer(), Substrate = Layer(), eta_z = 10.0,     eta_x = 10.0, h = 1.0)``

``Stacks``
   A ``list`` consiting of ``Stack``\ s in the stacks the first item is
   the layer closest to the bottom
``Ambient``
   A ``Layer`` describing the Ambient (enviroment above the sample).
   Only the scattering lengths and density of the layer is used.
``Substrate``
   A ``Layer`` describing the substrate (enviroment below the sample).
   Only the scattering lengths, density and roughness of the layer is
   used.
``eta_z``
   The out-of plane (vertical) correlation length of the roughness in
   the sample. Given in AA.
``eta_x``
   The in-plane global correlation length (it is assumed equal for all
   layers). Given in AA.
``h``
   The jaggedness parameter, should be between 0 and 1.0. This describes
   how jagged the interfaces are. This is also a global parameter for
   all interfaces.

Instrument
~~~~~~~~~~
``Instrument(wavelength = 1.54, coords = 'tth',      I0 = 1.0 res = 0.001, restype = 'no conv', respoints = 5, resintrange = 2,      beamw = 0.01, footype = 'no corr', samplelen = 10.0, taylor_n = 1)``

``wavelength``
    The wavalelngth of the radiation givenin AA (Angstroms)
``coords``
    The coordinates of the data given to the SimSpecular function. The
    available alternatives are: 'q' or 'tth'. Alternatively the numbers 0
    (q) or 1 (tth) can be used.
``I0``
    The incident intensity (a scaling factor)
``Ibkg``
    The background intensity. Added as a constant value to the calculated
    reflectivity
``res``
    The resolution of the instrument given in the coordinates of ``coords``.
    This assumes a gaussian reloution function and ``res`` is the standard
    deviation of that gaussian.
``restype``
    Describes the rype of the resolution calculated. One of the
    alterantives: 'no conv', 'fast conv', 'full conv and varying res.' or
    'fast conv + varying res.'. The respective numbers 0-3 also works. Note
    that fast convolution only alllows a single value into res wheras the
    other can also take an array with the same length as the x-data (varying
    resolution)
``respoints``
    The number of points to include in the resolution calculation. This is
    only used for 'full conv and vaying res.' and 'fast conv + varying res'
``resintrange``
    Number of standard deviatons to integrate the resolution fucntion times
    the relfectivty over
``footype``
    Which type of footprint correction is to be applied to the simulation.
    One of: 'no corr', 'gauss beam' or 'square beam'. Alternatively, the
    number 0-2 are also valid. The different choices are self explanatory.
``beamw``
    The width of the beam given in mm. For 'gauss beam' it should be the
    standard deviation. For 'square beam' it is the full width of the beam.
``samplelen``
    The length of the sample given in mm
``taylor_n``
    The number terms taken into account in the taylor expansion of the
    fourier integral of the correlation function. More terms more accurate
    calculation but also much slower.
'''
from .lib import paratt as Paratt
from .lib import offspec

from .lib.instrument import *

# Preamble to define the parameters needed for the models outlined below:
ModelID='MingInterdiff'
# Automatic loading of parameters possible by including this list
__pars__=['Layer', 'Stack', 'Sample', 'Instrument']
# Used for making choices in the GUI
instrument_string_choices={'coords': ['q', 'tth'],
                           'restype': ['no conv', 'fast conv',
                                       'full conv and varying res.', 'fast conv + varying res.'],
                           'footype': ['no corr', 'gauss beam', 'square beam']}

InstrumentParameters={'wavelength': 1.54, 'coords': 'tth', 'I0': 1.0, 'res': 0.001,
                      'restype': 'no conv', 'respoints': 5, 'resintrange': 2, 'beamw': 0.01, 'footype': 'no corr',
                      'samplelen': 10.0, 'Ibkg': 0.0, 'taylor_n': 1}
# Coordinates=1 => twothetainput
# Coordinates=0 => Q input
# Res stddev of resolution
# ResType 0: No resolution convlution
#               1: Fast convolution
#               2: Full Convolution +varying resolution
#               3: Fast convolution varying resolution
# ResPoints Number of points for the convolution only valid for ResolutionType=2
# ResIntrange Number of standard deviatons to integrate over default 2
# Parameters for footprint coorections
# Footype: 0: No corections for footprint
#          1: Correction for Gaussian beam => Beaw given in mm and stddev
#          2: Correction for square profile => Beaw given in full width mm
# Samlen= Samplelength in mm.

LayerParameters={'sigmai': 0.0, 'sigmar': 0.0, 'dens': 1.0, 'd': 0.0,
                 'f': 0.0+0.0j}
StackParameters={'Layers': [], 'Repetitions': 1}
SampleParameters={'Stacks': [], 'Ambient': None, 'Substrate': None, 'h': 1.0,
                  'eta_z': 10.0, 'eta_x': 10.0}

__xlabel__ = "q [Å$^{-1}$]"
__ylabel__ = "Instnsity [a.u.]"

def Specular(TwoThetaQz, sample, instrument):
    ''' Simulate the specular signal from sample when proped with instrument
    
    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    '''
    # preamble to get it working with my class interface
    restype=instrument.getRestype()
    global __xlabel__
    __xlabel__ = "q [Å$^{-1}$]"

    if restype==2 or restype==instrument_string_choices['restype'][2]:
        (TwoThetaQz, weight)=ResolutionVector(TwoThetaQz[:],
                                              instrument.getRes(), instrument.getRespoints(),
                                              range=instrument.getResintrange())
    if instrument.getCoords()==1 or \
            instrument.getCoords()==instrument_string_choices['coords'][1]:
        theta=TwoThetaQz/2
        __xlabel__ = "2θ [°]"
    elif instrument.getCoords()==0 or \
            instrument.getCoords()==instrument_string_choices['coords'][0]:
        theta=arcsin(TwoThetaQz/4/pi*instrument.getWavelength())*180./pi

    lamda=instrument.getWavelength()
    parameters=sample.resolveLayerParameters()
    dens=array(parameters['dens'], dtype=float64)
    # print [type(f) for f in parameters['f']]
    f=array(parameters['f'], dtype=complex128)
    re=2.82e-13*1e2/1e-10
    n=1-dens*re*lamda**2/2/pi*f*1e-4
    d=array(parameters['d'], dtype=float64)
    # d = d[1:-1]
    sigmar=array(parameters['sigmar'], dtype=float64)
    # sigmar = sigmar[:-1]
    sigmai=array(parameters['sigmai'], dtype=float64)
    # sigmai = sigmai[:-1]
    sigma=sqrt(sigmai**2+sigmar**2)
    # print sigma

    R=Paratt.Refl(theta, lamda, n, d, sigma)*instrument.getI0()

    # FootprintCorrections

    foocor=1.0
    footype=instrument.getFootype()
    beamw=instrument.getBeamw()
    samlen=instrument.getSamplelen()
    if footype==0 or footype==instrument_string_choices['footype'][0]:
        foocor=1.0
    elif footype==1 or footype==instrument_string_choices['footype'][1]:
        foocor=GaussIntensity(theta, samlen/2.0, samlen/2.0, beamw)
    elif footype==2 or footype==instrument_string_choices['footype'][2]:
        foocor=SquareIntensity(theta, samlen, beamw)
    else:
        raise ValueError('Variable footype has an unvalid value')

    if restype==0 or restype==instrument_string_choices['restype'][0]:
        R=R[:]*foocor
    elif restype==1 or restype==instrument_string_choices['restype'][1]:
        R=ConvoluteFast(TwoThetaQz, R[:]*foocor, instrument.getRes(),
                        range=instrument.getResintrange())
    elif restype==2 or restype==instrument_string_choices['restype'][2]:
        R=ConvoluteResolutionVector(TwoThetaQz, R[:]*foocor, weight)
    elif restype==3 or restype==instrument_string_choices['restype'][3]:
        R=ConvoluteFastVar(TwoThetaQz, R[:]*foocor, instrument.getRes(),
                           range=instrument.getResintrange())
    else:
        raise ValueError('Variable restype has an unvalid value')
    return R+instrument.getIbkg()

def OffSpecularMingInterdiff(TwoThetaQz, ThetaQx, sample, instrument):
    ''' Function that simulates the off-specular signal (not implemented)
    
    # BEGIN Parameters
    TwoThetaQz 1.0
    ThetaQx data.x
    # END Parameters
    '''
    lamda=instrument.getWavelength()
    if instrument.getCoords() in [1, instrument_string_choices['coords'][1]]:
        alphaR1=ThetaQx
        betaR1=TwoThetaQz-ThetaQx
        qx=2*pi/lamda*(cos(alphaR1*pi/180)-cos(betaR1*pi/180))
        qz=2*pi/lamda*(sin(alphaR1*pi/180)+sin(betaR1*pi/180))
    else:
        qz=TwoThetaQz
        qx=ThetaQx

    parameters=sample.resolveLayerParameters()

    def toarray(a, code):
        a=list(a)
        a.reverse()
        return array(a, dtype=code)

    dens=array(parameters['dens'], dtype=complex64)
    f=array(parameters['f'], dtype=complex64)
    re=2.82e-13*1e2/1e-10
    n=1-dens*re*lamda**2/2/pi*f*1e-4
    n=toarray(n, code=complex64)
    sigmar=toarray(parameters['sigmar'], code=float64)
    sigmar=sigmar[1:]

    sigmai=toarray(parameters['sigmai'], code=float64)
    sigmai=sigmai[1:]+1e-5

    d=toarray(parameters['d'], code=float64)
    d=r_[0, d[1:-1]]

    z=-cumsum(d)

    eta=sample.getEta_x()

    h=sample.getH()

    eta_z=sample.getEta_z()

    (I, alpha, omega)=offspec.DWBA_Interdiff(qx, qz, lamda, n, z,
                                                 sigmar, sigmai, eta, h, eta_z, d,
                                                 taylor_n=instrument.getTaylor_n())

    restype=instrument.getRestype()
    if restype==0 or restype==instrument_string_choices['restype'][0]:
        # if no resolution is defined, don't include specular peak
        return real(I)*instrument.getI0()+instrument.getIbkg()

    # include specular peak
    instrument.setRestype(0)
    if isinstance(TwoThetaQz, ndarray):
        Ispec=Specular(TwoThetaQz, sample, instrument)
    else:
        Ispec=Specular(array([TwoThetaQz], dtype=float64), sample, instrument)[0]
    instrument.setRestype(restype)

    if instrument.getCoords() in [1, instrument_string_choices['coords'][1]]:
        spec_peak=Ispec*exp(-0.5*(TwoThetaQz/2.-ThetaQx)**2/instrument.getRes()**2)
    else:
        spec_peak=Ispec*exp(-0.5*ThetaQx**2/instrument.getRes()**2)
    return (spec_peak+real(I))*instrument.getI0()+instrument.getIbkg()

def SLD_calculations(z, item, sample, inst):
    ''' Calculates the scatteringlength density as at the positions z
    
    # BEGIN Parameters
    z data.x
    item "Re"
    # END Parameters
    '''
    parameters=sample.resolveLayerParameters()
    dens=array(parameters['dens'], dtype=complex64)
    f=array(parameters['f'], dtype=complex64)
    sld=dens*f
    d_sld=sld[:-1]-sld[1:]
    d=array(parameters['d'], dtype=float64)
    d=d[1:-1]
    # Include one extra element - the zero pos (substrate/film interface)
    int_pos=cumsum(r_[0, d])
    sigmar=array(parameters['sigmar'], dtype=float64)
    sigmar=sigmar[:-1]
    sigmai=array(parameters['sigmai'], dtype=float64)
    sigmai=sigmai[:-1]
    sigma=sqrt(sigmai**2+sigmar**2)+1e-7
    if z is None:
        z=arange(-sigma[0]*5, int_pos.max()+sigma[-1]*5, 0.5)
    rho=sum(d_sld*(0.5-0.5*erf((z[:, newaxis]-int_pos)/sqrt(2.)/sigma)), 1)+sld[-1]
    dic={'Re': real(rho), 'Im': imag(rho), 'z': z,
         'SLD unit': 'r_{e}/\AA^{3}'}
    if item is None or item=='all':
        return dic
    else:
        try:
            return dic[item]
        except:
            raise ValueError('The chosen item, %s, does not exist'%item)

SimulationFunctions={'Specular': Specular,
                     'OffSpecular': OffSpecularMingInterdiff,
                     'SLD': SLD_calculations}

from .lib import refl as Refl

(Instrument, Layer, Stack, Sample)=Refl.MakeClasses(InstrumentParameters,
                                                    LayerParameters, StackParameters,
                                                    SampleParameters, SimulationFunctions, ModelID)

if __name__=='__main__':
    Fe=Layer(d=10, sigmar=3.0, n=1-2.247e-5+2.891e-6j)
    Si=Layer(d=15, sigmar=3.0, n=1-7.577e-6+1.756e-7j)
    sub=Layer(sigmar=3.0, n=1-7.577e-6+1.756e-7j)
    amb=Layer(n=1.0, sigmar=1.0)
    stack=Stack(Layers=[Fe, Si], Repetitions=20)
    sample=Sample(Stacks=[stack], Ambient=amb, Substrate=sub, eta_z=500.0, eta_x=100.0)
    iprint(sample)
    inst=Instrument(Wavelength=1.54, Coordinates=1)
