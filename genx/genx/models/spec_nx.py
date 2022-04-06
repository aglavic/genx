'''
Library for combined x-ray and neutrons simulations.
====================================================
The neutron simulations is capable of handling non-magnetic, magnetic
non-spin flip as well as neutron spin-flip reflectivity.

Classes
-------

Layer
~~~~~
``Layer(b = 0.0, d = 0.0, f = 0.0+0.0J, dens = 1.0, magn_ang = 0.0, magn = 0.0, sigma = 0.0)``

``b``
   The neutron scattering length per formula unit in fm (fermi meter =
   1e-15m)
``d``
   The thickness of the layer in AA (Angstroms = 1e-10m)
``f``
   The x-ray scattering length per formula unit in electrons. To be
   strict it is the number of Thompson scattering lengths for each
   formula unit.
``dens``
   The density of formula units in units per Angstroms. Note the units!
``magn_ang``
   The angle of the magnetic moment in degress. 0 degrees correspond to
   a moment collinear with the neutron spin.
``magn``
   The magnetic moment per formula unit (same formula unit as b and dens
   refer to)
``sigma``
   The root mean square roughness of the top interface of the layer in
   Angstroms.
``xs_ai``
   The sum of the absorption cross section and the incoherent scattering
   cross section in barns for neutrons

Stack
~~~~~
``Stack(Layers = [], Repetitions = 1)``

``Layers``
   A ``list`` consiting of ``Layer``s in the stack the first item is
   the layer closest to the bottom
``Repetitions``
   The number of repsetions of the stack

Sample
~~~~~~
``Sample(Stacks = [], Ambient = Layer(), Substrate = Layer())``

``Stacks``
   A ``list`` consiting of ``Stack``s in the stacks the first item is
   the layer closest to the bottom
``Ambient``
   A ``Layer`` describing the Ambient (enviroment above the sample).
   Only the scattering lengths and density of the layer is used.
``Substrate``
   A ``Layer`` describing the substrate (enviroment below the sample).
   Only the scattering lengths, density and roughness of the layer is
   used.

Instrument
~~~~~~~~~~
``Instrument(probe = 'x-ray', wavelength = 1.54, coords = '2θ', I0 = 1.0 res = 0.001, restype = 'no conv', respoints = 5, resintrange = 2, beamw = 0.01, footype = 'no corr', samplelen = 10.0, incangle = 0.0, pol = 'uu')``

``probe``
    Describes the radiation and measurments used is one of: 'x-ray',
    'neutron', 'neutron pol', 'neutron pol spin flip', 'neutron tof',
    'neutron pol tof' or the respective number 0, 1, 2, 3, 4, 5, 6. The
    calculations for x-rays uses ``f`` for the scattering length for
    neutrons ``b`` for 'neutron pol', 'neutron pol spin flip' and 'neutron
    pol tof' alternatives the ``magn`` is used in the calculations. Note
    that the angle of magnetization ``magn_ang`` is only used in the last
    alternative.
``wavelength``
    The wavelength of the radiation given in AA (Angstroms)
``coords``
    The coordinates of the data given to the SimSpecular function. The
    available alternatives are: 'q' or '2θ'. Alternatively the numbers 0 (q)
    or 1 (tth) can be used.
``I0``
    The incident intensity (a scaling factor)
``Ibkg``
    The background intensity. Added as a constant value to the calculated
    reflectivity
``res``
    The resolution of the instrument given in the coordinates of ``coords``.
    This assumes a gaussian resolution function and ``res`` is the standard
    deviation of that gaussian. If ``restype`` has (dx/x) in its name the
    gaussian standard deviation is given by res*x where x is either in tth
    or q.
``restype``
    Describes the rype of the resolution calculated. One of the
    alterantives: 'no conv', 'fast conv', 'full conv and varying res.',
    'fast conv + varying res.', 'full conv and varying res. (dx/x)', 'fast
    conv + varying res. (dx/x)'. The respective numbers 0-3 also works. Note
    that fast convolution only alllows a single value into res wheras the
    other can also take an array with the same length as the x-data (varying
    resolution)
``respoints``
    The number of points to include in the resolution calculation. This is
    only used for 'full conv and vaying res.', 'fast conv + varying res',
    'full conv and varying res. (dx/x)' and 'fast conv + varying res.
    (dx/x)'.
``resintrange``
    Number of standard deviatons to integrate the resolution function times
    the reflectivity over
``footype``
    Which type of footprint correction is to be applied to the simulation.
    One of: 'no corr', 'gauss beam' or 'square beam'. Alternatively, the
    number 0-2 are also valid. The different choices are self expnalatory.
``beamw``
    The width of the beam given in mm. For 'gauss beam' it should be the
    standard deviation. For 'square beam' it is the full width of the beam.
``samplelen``
    The length of the sample given in mm
``incangle``
    The incident angle of the neutrons, only valid in tof mode
``pol``
    The measured polarization of the instrument. Valid options are:
    'uu','dd', 'ud', 'du' or 'ass' the respective number 0-3 also works.
'''
import numpy as np

from .lib import paratt as Paratt
from .lib import neutron_refl as MatrixNeutron
from .lib.instrument import *
from .lib import refl as refl
from .lib.physical_constants import r_e, muB_to_SL


# Preamble to define the parameters needed for the models outlined below:

# import core.paratt as slow_paratt

ModelID = 'SpecNX'
# InstrumentParameters={'Wavelength':1.54, 'Coordinates':1, 'I0':1.0, 'Sim': 0,
#    'Res':0.001, 'Restype':0, 'Respoints':5, 'Resintrange':2, 'Beaw':0.01,
#    'Footype':0.0, 'Samlen':10.0, 'Incangle':0.0}
__pars__ = ['Layer', 'Stack', 'Sample', 'Instrument']

instrument_string_choices = {
    'probe': ['x-ray', 'neutron', 'neutron pol',
              'neutron pol spin flip', 'neutron tof', 'neutron pol tof'],
    'coords': ['q', '2θ', 'tth'],
    'restype': ['no conv', 'fast conv',
                'full conv and varying res.', 'fast conv + varying res.',
                'full conv and varying res. (dx/x)', 'fast conv + varying res. (dx/x)'],
    'footype': ['no corr', 'gauss beam', 'square beam'],
    'pol': ['uu', 'dd', 'ud', 'ass', 'du']
    }
InstrumentParameters = {
    'probe': 'x-ray', 'wavelength': 1.54, 'coords': '2θ',
    'I0': 1.0, 'res': 0.001,
    'restype': 'no conv', 'respoints': 5, 'resintrange': 2.0, 'beamw': 0.01,
    'footype': 'no corr', 'samplelen': 10.0, 'incangle': 0.5, 'pol': 'uu',
    'Ibkg': 0.0, 'tthoff': 0.0
    }
InstrumentGroups = [('General', ['wavelength', 'coords', 'I0', 'Ibkg', 'tthoff']),
                    ('Resolution', ['restype', 'res', 'respoints', 'resintrange']),
                    ('Probe', ['probe', 'pol', 'incangle']),
                    ('Footprint', ['footype', 'beamw', 'samplelen', ]),
                    ]
InstrumentUnits = {
    'probe': '', 'wavelength': 'Å', 'coords': '',
    'I0': 'arb.', 'res': '[coord]',
    'restype': '', 'respoints': 'pts.', 'resintrange': '[coord]', 'beamw': 'mm',
    'footype': '', 'samplelen': 'mm', 'incangle': '°', 'pol': '',
    'Ibkg': 'arb.', 'tthoff': '°',
    '2θ': '°', 'tth': '°', 'q': 'Å$^-1$'
    }
# Coordinates=1 or '2θ' => twothetainput
# Coordinates=0 or 'q'=> Q input
# probe: Type of simulation
#   'x-ray' or 0: X-rays (One output)
#   'neutron'or 1: Neutrons (One output, ignoring magn, magn_ang)
#   'neutron pol' or 2: Neutrons polarized (Two outputs Ruu,Rdd)
#   'neutron pol spin flip' or 3: Neutrons polarized with spin-flip 
#            (Three outputs Ruu,Rdd,Rud=Rdu, ignoring sigma!)
#   'neutron tof' or 4: Neutrons non-polarized TOF, Inc Angle must be set
#   'neutron pol tof'or 5: Neutrons polarized TOF (non-spin flip),
#         Inc Angle must be set
#
# res stddev of resolution
# restype 0 'none': No resolution convlution
#               1 or 'fast': Fast convolution
#               2 or 'full': Full Convolution +varying resolution
#               3 or 'ffull': Fast convolution varying resolution
# respoints Number of points for the convolution only valid for ResolutionType=2
# resintrange Number of standard deviatons to integrate over default 2
# Parameters for footprint coorections
# footype: 0 or 'no corr': No corections for footprint
#            1 or 'gauss beam': Correction for Gaussian beam => Beaw given in mm and stddev
#            2 or 'square beam': Correction for square profile => Beaw given in full width mm
# samlen= Samplelength in mm.

LayerParameters = {
    'sigma': 0.0, 'dens': 1.0, 'd': 0.0, 'f': (1.0+1.0j)*1e-20,
    'b': 0.0+0.0J, 'xs_ai': 0.0, 'magn': 0.0, 'magn_ang': 0.0
    }
LayerUnits = {
    'sigma': 'AA', 'dens': 'at./AA', 'd': 'AA', 'f': 'el./at.',
    'b': 'fm/at.', 'xs_ai': 'barn/at.', 'magn': 'mu_B/at.', 'magn_ang': 'deg.'
    }
LayerGroups = [('Standard', ['f', 'dens', 'd', 'sigma']),
               ('Neutron', ['b', 'xs_ai', 'magn', 'magn_ang'])]
StackParameters = {'Layers': [], 'Repetitions': 1}
SampleParameters = {'Stacks': [], 'Ambient': None, 'Substrate': None}

AA_to_eV = 12398.5
''' Conversion from Angstrom to eV E = AA_to_eV/lamda.'''

q_limit = 1e-10
''' Minimum allowed q-value '''

__xlabel__ = "q [Å$^{-1}$]"
__ylabel__ = "Instnsity [a.u.]"


# A buffer to save previous calculations for spin-flip calculations
class Buffer:
    Ruu = 0
    Rdd = 0
    Rdu = 0
    Rud = 0
    parameters = None
    TwoThetaQz = None


def footprintcorr(Q, instrument):
    foocor = 1.0
    footype = instrument.getFootype()
    beamw = instrument.getBeamw()
    samlen = instrument.getSamplelen()

    if instrument.getProbe() in [instrument_string_choices['probe'][4],
                                 instrument_string_choices['probe'][5], 4, 5]:
        theta = instrument.getIncangle()
        # if ai is an array, make sure it gets repeated for every resolution point
        if type(theta) is ndarray and instrument.getRestype() in [2, 4,
                                                                  instrument_string_choices['restype'][2],
                                                                  instrument_string_choices['restype'][4]]:
            theta = (theta*ones(instrument.getRespoints())[:, newaxis]).flatten()

    else:
        # theta =  180./pi*arcsin(instrument.getWavelength() / 4.0 / pi * Q)
        theta = QtoTheta(instrument.getWavelength(), Q)
    if footype==1 or footype==instrument_string_choices['footype'][1]:
        foocor = GaussIntensity(theta, samlen/2.0, samlen/2.0, beamw)
    elif footype==2 or footype==instrument_string_choices['footype'][2]:
        foocor = SquareIntensity(theta, samlen, beamw)
    elif footype==0 or footype==instrument_string_choices['footype'][0]:
        pass
    else:
        raise ValueError('The choice of footprint correction, footype,'
                         'is WRONG')

    return foocor


def resolutioncorr(R, TwoThetaQz, foocor, instrument, weight):
    ''' Do the convolution of the reflectivity to account for resolution effects.'''
    restype = instrument.getRestype()
    if restype==instrument_string_choices['restype'][1] or restype==1:
        R = ConvoluteFast(TwoThetaQz, R[:]*foocor, instrument.getRes(),
                          range=instrument.getResintrange())
    elif (restype==instrument_string_choices['restype'][2] or restype==2 or
          restype==instrument_string_choices['restype'][4] or restype==4):
        R = ConvoluteResolutionVector(TwoThetaQz, R[:]*foocor, weight)
    elif restype==instrument_string_choices['restype'][3] or restype==3:
        R = ConvoluteFastVar(TwoThetaQz, R[:]*foocor, instrument.getRes(), range=instrument.getResintrange())
    elif restype==instrument_string_choices['restype'][5] or restype==5:
        R = ConvoluteFastVar(TwoThetaQz, R[:]*foocor, instrument.getRes()*TwoThetaQz,
                             range=instrument.getResintrange())
    elif restype==instrument_string_choices['restype'][0] or restype==0:
        R = R[:]*foocor
    else:
        raise ValueError('The choice of resolution type, restype,'
                         'is WRONG')
    return R


def resolution_init(TwoThetaQz, instrument):
    ''' Inits the dependet variable with regards to coordinates and resolution.'''
    global __xlabel__
    __xlabel__ = "q [Å$^{-1}$]"

    restype = instrument.getRestype()
    weight = 0
    if restype==2 or restype==instrument_string_choices['restype'][2]:
        (TwoThetaQz, weight) = ResolutionVector(TwoThetaQz[:],
                                                instrument.getRes(), instrument.getRespoints(),
                                                range=instrument.getResintrange())
    elif restype==4 or restype==instrument_string_choices['restype'][4]:
        (TwoThetaQz, weight) = ResolutionVector(TwoThetaQz[:],
                                                instrument.getRes()*TwoThetaQz, instrument.getRespoints(),
                                                range=instrument.getResintrange())
    # TTH values given as x
    if instrument.getCoords() in [instrument_string_choices['coords'][1],
                                  instrument_string_choices['coords'][2], 1, 2]:
        # Q = 4 * pi / instrument.getWavelength() * sin((TwoThetaQz + instrument.getTthoff()) * pi / 360.0)
        Q = TwoThetatoQ(instrument.getWavelength(), TwoThetaQz+instrument.getTthoff())
        __xlabel__ = "2θ [°]"
    # Q vector given....
    elif instrument.getCoords() in [instrument_string_choices['coords'][0], 0]:
        # if there is no tth offset, nothing to be done for Q
        if instrument.getTthoff()==0:
            Q = TwoThetaQz
        # for tof the q-values are not angles but wavelenght, so tth-offset is a scaling factor
        elif instrument.getProbe() in [instrument_string_choices['probe'][4],
                                       instrument_string_choices['probe'][5], 4, 5]:
            ai = instrument.getIncangle()
            # if ai is an array, make sure it gets repeated for every resolution point
            if type(ai) is ndarray and restype in [2, 4,
                                                   instrument_string_choices['restype'][2],
                                                   instrument_string_choices['restype'][4]]:
                ai = (ai*ones(instrument.getRespoints())[:, newaxis]).flatten()
            Q = TwoThetaQz*(sin((ai+instrument.getTthoff()/2.)*pi/180.)/sin(ai*pi/180.))
        else:
            Q = TwoThetatoQ(instrument.getWavelength(),
                            QtoTheta(instrument.getWavelength(), TwoThetaQz)*2.0+instrument.getTthoff())
    else:
        raise ValueError('The value for coordinates, coords, is WRONG! should be q(0) or tth(1).')
    return Q, TwoThetaQz, weight


def neutron_sld(abs_xs, dens, fb, wl):
    return dens*(wl**2/2/pi*fb-1.0J*abs_xs*wl/4/pi)


def Specular(TwoThetaQz, sample, instrument):
    """ Simulate the specular signal from sample when probed with instrument

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """
    return specular_calcs(TwoThetaQz, sample, instrument, return_int=True)


def SpecularField(TwoThetaQz, sample, instrument):
    """ Simulate the specular signal from sample when probed with instrument

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """
    return specular_calcs(TwoThetaQz, sample, instrument, return_int=False)


def specular_calcs(TwoThetaQz, sample, instrument, return_int=True):
    """ Simulate the specular signal from sample when probed with instrument

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """

    # preamble to get it working with my class interface
    restype = instrument.getRestype()
    Q, TwoThetaQz, weight = resolution_init(TwoThetaQz, instrument)
    # often an issue with resolution etc. so just replace Q values < q_limit
    # if any(Q < q_limit):
    #    raise ValueError('The q vector has to be above %.1e'%q_limit)
    Q = maximum(Q, q_limit)

    ptype = instrument.getProbe()
    pol = instrument.getPol()

    parameters = sample.resolveLayerParameters()
    if ptype==instrument_string_choices['probe'][0] or ptype==0:
        # fb = array(parameters['f'], dtype = complex64)
        e = AA_to_eV/instrument.getWavelength()
        fb = refl.cast_to_array(parameters['f'], e).astype(complex128)
    else:
        fb = array(parameters['b'], dtype=complex128)*1e-5
        abs_xs = array(parameters['xs_ai'], dtype=complex128)*1e-4**2

    dens = array(parameters['dens'], dtype=float64)
    d = array(parameters['d'], dtype=float64)
    magn = array(parameters['magn'], dtype=float64)
    # Transform to radians
    magn_ang = array(parameters['magn_ang'], dtype=float64)*pi/180.0

    sigma = array(parameters['sigma'], dtype=float64)

    if ptype==instrument_string_choices['probe'][0] or ptype==0:
        sld = dens*fb*instrument.getWavelength()**2/2/pi
    else:
        wl = instrument.getWavelength()
        # sld = dens*(wl**2/2/pi*sqrt(fb**2 - (abs_xs/2.0/wl)**2) -
        #                       1.0J*abs_xs*wl/4/pi)
        sld = neutron_sld(abs_xs, dens, fb, wl)
    # Ordinary Paratt X-rays
    if ptype==instrument_string_choices['probe'][0] or ptype==0:
        R = Paratt.ReflQ(Q, instrument.getWavelength(), 1.0-r_e*sld, d, sigma, return_int=return_int)
        # print 2.82e-5*sld
    # Ordinary Paratt Neutrons
    elif ptype==instrument_string_choices['probe'][1] or ptype==1:
        R = Paratt.ReflQ(Q, instrument.getWavelength(), 1.0-sld, d, sigma, return_int=return_int)
    # Ordinary Paratt but with magnetization
    elif ptype==instrument_string_choices['probe'][2] or ptype==2:
        msld = muB_to_SL*magn*dens*instrument.getWavelength()**2/2/pi
        # Polarization uu or ++
        if pol==instrument_string_choices['pol'][0] or pol==0:
            R = Paratt.ReflQ(Q, instrument.getWavelength(), 1.0-sld-msld, d, sigma, return_int=return_int)
        # Polarization dd or --
        elif pol==instrument_string_choices['pol'][1] or pol==1:
            R = Paratt.ReflQ(Q, instrument.getWavelength(), 1.0-sld+msld, d, sigma, return_int=return_int)
        elif pol==instrument_string_choices['pol'][3] or pol==3:
            Rp = Paratt.ReflQ(Q, instrument.getWavelength(), 1.0-sld-msld, d, sigma, return_int=return_int)
            Rm = Paratt.ReflQ(Q, instrument.getWavelength(), 1.0-sld+msld, d, sigma, return_int=return_int)
            R = (Rp-Rm)/(Rp+Rm)

        else:
            raise ValueError('The value of the polarization is WRONG.'
                             ' It should be uu(0) or dd(1)')
    # Spin flip
    elif ptype in [3, instrument_string_choices['probe'][3]]:
        # Check if we have calcluated the same sample previous:
        if Buffer.TwoThetaQz is not None:
            Q_ok = Buffer.TwoThetaQz.shape==Q.shape
            if Q_ok:
                Q_ok = not (Buffer.TwoThetaQz!=Q).any()
        else:
            Q_ok = False
        if Buffer.parameters!=parameters or not Q_ok:
            msld = muB_to_SL*magn*dens*instrument.getWavelength()**2/2/pi
            # renormalize SLDs if ambient layer is not vacuum
            if msld[-1]!=0. or sld[-1]!=0:
                msld -= msld[-1]
                sld -= sld[-1]
            sld_p = sld+msld
            sld_m = sld-msld
            Vp = (2*pi/instrument.getWavelength())**2*(sld_p*(2.+sld_p))  # (1-np**2) - better numerical accuracy
            Vm = (2*pi/instrument.getWavelength())**2*(sld_m*(2.+sld_m))  # (1-nm**2)
            (Ruu, Rdd, Rud, Rdu) = MatrixNeutron.Refl(Q, Vp, Vm, d, magn_ang, sigma, return_int=return_int)
            Buffer.Ruu = Ruu
            Buffer.Rdd = Rdd
            Buffer.Rud = Rud
            Buffer.parameters = parameters.copy()
            Buffer.TwoThetaQz = Q.copy()
        else:
            pass
        # Polarization uu or ++
        if pol==instrument_string_choices['pol'][0] or pol==0:
            R = Buffer.Ruu
        # Polarization dd or --
        elif pol==instrument_string_choices['pol'][1] or pol==1:
            R = Buffer.Rdd
        # Polarization ud or +-
        elif (pol==instrument_string_choices['pol'][2] or pol==2 or
              pol==instrument_string_choices['pol'][4] or pol==4):
            R = Buffer.Rud
        # Calculating the asymmetry ass
        elif pol==instrument_string_choices['pol'][3] or pol==3:
            R = (Buffer.Ruu-Buffer.Rdd)/(Buffer.Ruu+Buffer.Rdd+2*Buffer.Rud)
        else:
            raise ValueError('The value of the polarization is WRONG.'
                             ' It should be uu(0), dd(1) or ud(2)')

    # tof
    elif ptype==instrument_string_choices['probe'][4] or ptype==4:
        ai = instrument.getIncangle()
        # if ai is an array, make sure it gets repeated for every resolution point
        if type(ai) is ndarray and restype in [2, 4,
                                               instrument_string_choices['restype'][2],
                                               instrument_string_choices['restype'][4]]:
            ai = (ai*ones(instrument.getRespoints())[:, newaxis]).flatten()
        else:
            ai = ai*ones(Q.shape)
        wl = 4*pi*sin(ai*pi/180)/Q
        sld = neutron_sld(abs_xs[:, newaxis], dens[:, newaxis], fb[:, newaxis], wl)
        R = Paratt.Refl_nvary2(ai, wl,
                               1.0-sld, d, sigma, return_int=return_int)
    # tof spin polarized
    elif ptype==instrument_string_choices['probe'][5] or ptype==5:
        wl = 4*pi*sin(instrument.getIncangle()*pi/180)/Q
        sld = neutron_sld(abs_xs[:, newaxis], dens[:, newaxis], fb[:, newaxis], wl)
        msld = muB_to_SL*magn[:, newaxis]*dens[:, newaxis]*(4*pi*sin(instrument.getIncangle()*pi/180)/Q)**2/2/pi
        # polarization uu or ++
        if pol==instrument_string_choices['pol'][0] or pol==0:
            R = Paratt.Refl_nvary2(instrument.getIncangle()*ones(Q.shape),
                                   (4*pi*sin(instrument.getIncangle()*pi/180)/Q), 1.0-sld-msld, d, sigma,
                                   return_int=return_int)
        # polarization dd or --
        elif pol==instrument_string_choices['pol'][1] or pol==1:
            R = Paratt.Refl_nvary2(instrument.getIncangle()*ones(Q.shape),
                                   (4*pi*sin(instrument.getIncangle()*pi/180)/Q), 1.0-sld+msld, d, sigma,
                                   return_int=return_int)
        # Calculating the asymmetry
        elif pol==instrument_string_choices['pol'][3] or pol==3:
            Rd = Paratt.Refl_nvary2(instrument.getIncangle()*ones(Q.shape),
                                    (4*pi*sin(instrument.getIncangle()*pi/180)/Q),
                                    1.0-sld+msld, d, sigma, return_int=return_int)
            Ru = Paratt.Refl_nvary2(instrument.getIncangle()*ones(Q.shape),
                                    (4*pi*sin(instrument.getIncangle()*pi/180)/Q),
                                    1.0-sld-msld, d, sigma, return_int=return_int)
            R = (Ru-Rd)/(Ru+Rd)

        else:
            raise ValueError('The value of the polarization is WRONG.'
                             ' It should be uu(0) or dd(1) or ass')
    else:
        raise ValueError('The choice of probe is WRONG')
    if return_int:
        # FootprintCorrections
        foocor = footprintcorr(Q, instrument)
        # Resolution corrections
        R = resolutioncorr(R, TwoThetaQz, foocor, instrument, weight)

        return R*instrument.getI0()+instrument.getIbkg()
    else:
        return R


def EnergySpecular(Energy, TwoThetaQz, sample, instrument):
    ''' Simulate the specular signal from sample when probed with instrument. Energy should be in eV.

    # BEGIN Parameters
    Energy data.x
    TwoThetaQz 3.0
    # END Parameters
    '''
    # preamble to get it working with my class interface
    restype = instrument.getRestype()
    # TODO: Fix so that resolution can be included.
    if restype!=0 and restype!=instrument_string_choices['restype'][0]:
        raise ValueError('Only no resolution is allowed for energy scans.')

    wl = AA_to_eV/Energy
    global __xlabel__
    __xlabel__ = 'E [eV]'

    # TTH values given as x
    if instrument.getCoords()==instrument_string_choices['coords'][1] \
            or instrument.getCoords()==1:
        theta = TwoThetaQz/2.0
    # Q vector given....
    elif instrument.getCoords()==instrument_string_choices['coords'][0] \
            or instrument.getCoords()==0:
        theta = arcsin(TwoThetaQz*wl/4/pi)*180.0/pi

    else:
        raise ValueError('The value for coordinates, coords, is WRONG!'
                         'should be q(0) or tth(1).')
    Q = 4*pi/wl*sin((2*theta+instrument.getTthoff())*pi/360.0)

    ptype = instrument.getProbe()

    parameters = sample.resolveLayerParameters()
    if ptype==instrument_string_choices['probe'][0] or ptype==0:
        fb = refl.cast_to_array(parameters['f'], Energy).astype(complex128)
    else:
        fb = array(parameters['b'], dtype=complex128)*1e-5
        abs_xs = array(parameters['xs_ai'], dtype=complex128)*1e-4**2

    dens = array(parameters['dens'], dtype=float64)
    d = array(parameters['d'], dtype=float64)
    sigma = array(parameters['sigma'], dtype=float64)

    if ptype==instrument_string_choices['probe'][0] or ptype==0:
        sld = dens[:, newaxis]*fb*wl**2/2/pi
    else:
        wl = instrument.getWavelength()
        sld = dens*(wl**2/2/pi*sqrt(fb**2-(abs_xs/2.0/wl)**2)-1.0J*abs_xs*wl/4/pi)
    # Ordinary Paratt X-rays
    if ptype==instrument_string_choices['probe'][0] or ptype==0:
        # R = Paratt.ReflQ(Q,instrument.getWavelength(),1.0-2.82e-5*sld,d,sigma)
        R = Paratt.Refl_nvary2(theta, wl, 1.0-r_e*sld, d, sigma)
    else:
        raise ValueError('The choice of probe is WRONG')
    # TODO: Fix corrections
    # FootprintCorrections
    # foocor = footprintcorr(Q, instrument)
    # Resolution corrections
    # R = resolutioncorr(R, TwoThetaQz, foocor, instrument, weight)

    return R*instrument.getI0()+instrument.getIbkg()


def OffSpecular(TwoThetaQz, ThetaQx, sample, instrument):
    ''' Function that simulates the off-specular signal (not implemented)
    
    # BEGIN Parameters
    TwoThetaQz 1.0
    ThetaQx data.x
    # END Parameters
    '''
    raise NotImplementedError('Not implemented use model interdiff insteads')
    return TwoThetaQz, ThetaQx


def SLD_calculations(z, item, sample, inst):
    ''' Calculates the scatteringlength density as at the positions z
    if item is None or "all" the function returns a dictonary of values.
    Otherwise it returns the item as identified by its string.
    
    # BEGIN Parameters
    z data.x
    item 'Re'
    # END Parameters
    '''
    parameters = sample.resolveLayerParameters()
    if hasattr(sample, 'crop_sld') and sample.crop_sld!=0:
        crop_top_bottom = abs(sample.crop_sld)
        inter = dict([(key, 0.) for key in parameters])
        if sample.crop_sld>0:
            inter['d'] = sum(parameters['d'][crop_top_bottom:-crop_top_bottom])
        else:
            inter['d'] = 5.0
            inter['dens'] = 0.1
            inter['b'] = 12.0+0j
            inter['f'] = 100.0+0j
        if len(parameters['dens'])>2*crop_top_bottom:
            for key, value in parameters.items():
                val_start = value[:crop_top_bottom]
                val_end = value[-crop_top_bottom:]
                parameters[key] = val_start+[inter[key]]+val_end
    dens = array(parameters['dens'], dtype=float32)
    # f = array(parameters['f'], dtype = complex64)
    e = AA_to_eV/inst.getWavelength()
    f = refl.cast_to_array(parameters['f'], e).astype(complex64)
    b = array(parameters['b'], dtype=complex64)*1e-5
    abs_xs = array(parameters['xs_ai'], dtype=float32)*1e-4**2
    wl = inst.getWavelength()
    ptype = inst.getProbe()
    magnetic = False
    mag_sld = 0
    sld_unit = r'r_{e}/\AA^{3}'
    if ptype==instrument_string_choices['probe'][0] or ptype==0:
        sld = dens*f
    elif ptype==instrument_string_choices['probe'][1] or ptype==1 or \
            ptype==instrument_string_choices['probe'][4] or ptype==4:
        sld = dens*(wl**2/2/pi*b-1.0J*abs_xs*wl/4/pi)/1e-6/(wl**2/2/pi)
        sld_unit = r'10^{-6}\AA^{-2}'
    else:
        magnetic = True
        sld = dens*(wl**2/2/pi*b-1.0J*abs_xs*wl/4/pi)/1e-6/(wl**2/2/pi)
        magn = array(parameters['magn'], dtype=float64)
        # Transform to radians
        magn_ang = array(parameters['magn_ang'], dtype=float64)*pi/180.0
        mag_sld = 2.645*magn*dens*10.
        mag_sld_x = mag_sld*cos(magn_ang)
        mag_sld_y = mag_sld*sin(magn_ang)
        sld_unit = r'10^{-6}\AA^{-2}'

    d = array(parameters['d'], dtype=float64)
    d = d[1:-1]
    # Include one extra element - the zero pos (substrate/film interface)
    int_pos = cumsum(r_[0, d])
    sigma = array(parameters['sigma'], dtype=float64)[:-1]+1e-7
    if z is None:
        z = arange(-sigma[0]*5, int_pos.max()+sigma[-1]*5, 0.5)
    if not magnetic:
        rho = sum((sld[:-1]-sld[1:])*(0.5-
                                      0.5*erf((z[:, newaxis]-int_pos)/sqrt(2.)/sigma)), 1)+sld[-1]
        dic = {
            'Re': real(rho), 'Im': imag(rho), 'z': z,
            'SLD unit': sld_unit
            }
    else:
        sld_p = sld+mag_sld
        sld_m = sld-mag_sld
        rho_p = sum((sld_p[:-1]-sld_p[1:])*(0.5-
                                            0.5*erf((z[:, newaxis]-int_pos)/sqrt(2.)/sigma)), 1)+sld_p[-1]
        rho_m = sum((sld_m[:-1]-sld_m[1:])*(0.5-
                                            0.5*erf((z[:, newaxis]-int_pos)/sqrt(2.)/sigma)), 1)+sld_m[-1]
        rho_nucl = (rho_p+rho_m)/2.
        if (magn_ang!=0.).any():
            rho_mag_x = sum((mag_sld_x[:-1]-mag_sld_x[1:])*
                            (0.5-0.5*erf((z[:, newaxis]-int_pos)/sqrt(2.)/sigma)), 1)+mag_sld_x[-1]
            rho_mag_y = sum((mag_sld_y[:-1]-mag_sld_y[1:])*
                            (0.5-0.5*erf((z[:, newaxis]-int_pos)/sqrt(2.)/sigma)), 1)+mag_sld_y[-1]
            dic = {
                'Re non-mag': real(rho_nucl), 'Im non-mag': imag(rho_nucl),
                'mag': real(rho_p-rho_m)/2, 'z': z, 'mag_x': rho_mag_x, 'mag_y': rho_mag_y,
                'SLD unit': sld_unit
                }
        else:
            dic = {
                'Re non-mag': real(rho_nucl), 'Im non-mag': imag(rho_nucl),
                'mag': real(rho_p-rho_m)/2, 'z': z,
                'SLD unit': sld_unit
                }
    if item is None or item=='all':
        return dic
    else:
        try:
            return dic[item]
        except:
            raise ValueError('The chosen item, %s, does not exist'%item)

POL_CHANNELS = ['uu', 'ud', 'du', 'dd']

def PolSpecular(TwoThetaQz, p1, p2, F1, F2, sample, instrument):
    """
    Specular reflectivity of polarized measurement with finite polarization.
    The polarization parameters are in accordance to the definition used in
    A.R. Wildes publication, Review of Scientific Instruments 70, 11 (1999)
    https://doi.org/10.1063/1.1150060

    pol_params = (p1, p2, f1, f2)
    p1: polarizer efficiency, 0: 100% spin-up, 1: 100% spin-down, 0.5: unpolarized
    p2: analyzer efficiency,  0: 100% spin-up, 1: 100% spin-down, 0.5: unpolairzed
    F1/F2: Flipper efficienty, 0: 100% efficient, 1: no flipping

    # BEGIN Parameters
    TwoThetaQz data.x
    p1 0.
    p2 0.
    F1 0.
    F2 0.
    # END Parameters

    """
    inst_pol = instrument.pol
    if not inst_pol in POL_CHANNELS:
        raise ValueError(f"Instrument polarization as to be one of {POL_CHANNELS}.")
    if instrument.probe not in [3, instrument_string_choices['probe'][3]]:
        raise ValueError("Polarization corrected simulation requires probe to be 'neutron pol spin flip'")

    instrument.pol = 'uu'
    uu = Specular(TwoThetaQz, sample, instrument)
    instrument.pol = 'dd'
    dd = Specular(TwoThetaQz, sample, instrument)
    instrument.pol = 'ud'
    ud = Specular(TwoThetaQz, sample, instrument)
    du = ud
    instrument.pol = inst_pol

    P = get_pol_matrix(p1, p2, F1, F2)
    Pline = P[POL_CHANNELS.index(instrument.pol)]
    I = Pline[:, newaxis] * np.vstack([uu, ud, du, dd])
    return I.sum(axis=0)

SimulationFunctions = {
    'Specular': Specular,
    'PolSpecular': PolSpecular,
    'SpecularField': SpecularField,
    'OffSpecular': OffSpecular,
    'SLD': SLD_calculations,
    'EnergySpecular': EnergySpecular,
    }

(Instrument, Layer, Stack, Sample) = refl.MakeClasses(InstrumentParameters,
                                                      LayerParameters, StackParameters, SampleParameters,
                                                      SimulationFunctions,
                                                      ModelID)

if __name__=='__main__':
    from .utils import UserVars, fp, fw


    # BEGIN Instrument DO NOT CHANGE
    inst = Instrument(footype='gauss beam', probe='x-ray', beamw=0.04, resintrange=2, pol='uu', wavelength=1.54,
                      respoints=5, Ibkg=0.0, I0=2, samplelen=10.0, restype='no conv', coords='2θ', res=0.001,
                      incangle=0.0)
    fp.set_wavelength(inst.wavelength)
    # Compability issues for pre-fw created gx files
    try:
        fw
    except:
        pass
    else:
        fw.set_wavelength(inst.wavelength)
    # END Instrument

    # BEGIN Sample DO NOT CHANGE
    Amb = Layer(b=0, d=0.0, f=(1e-20+1e-20j), dens=1.0, magn_ang=0.0, sigma=0.0, xs_ai=0.0, magn=0.0)
    topPt = Layer(b=0, d=11.0, f=fp.Pt, dens=4/3.92**3, magn_ang=0.0, sigma=3.0, xs_ai=0.0, magn=0.0)
    TopFe = Layer(b=0, d=11.0, f=fp.Fe, dens=2/2.866**3, magn_ang=0.0, sigma=2.0, xs_ai=0.0, magn=0.0)
    Pt = Layer(b=0, d=11.0, f=fp.Pt, dens=4/3.92**3, magn_ang=0.0, sigma=2.0, xs_ai=0.0, magn=0.0)
    Fe = Layer(b=0, d=11, f=fp.Fe, dens=2/2.866**3, magn_ang=0.0, sigma=2.0, xs_ai=0.0, magn=0.0)
    bufPt = Layer(b=0, d=45, f=fp.Pt, dens=4/3.92**3, magn_ang=0.0, sigma=2, xs_ai=0.0, magn=0.0)
    bufFe = Layer(b=0, d=2, f=fp.Fe, dens=2/2.866**3, magn_ang=0.0, sigma=2, xs_ai=0.0, magn=0.0)
    Sub = Layer(b=0, d=0.0, f=fp.Mg+fp.O, dens=2/4.2**3, magn_ang=0.0, sigma=4.0, xs_ai=0.0, magn=0.0)

    ML = Stack(Layers=[Fe, Pt], Repetitions=100)

    sample = Sample(Stacks=[ML], Ambient=Amb, Substrate=Sub)
    # END Sample

    # BEGIN Parameters DO NOT CHANGE
    cp = UserVars()
    cp.new_var('Lambda', 33)
    # END Parameters

    import cProfile


    tth = np.arange(0.01, 8.0, 0.01)
    cProfile.run('[sample.SimSpecular(tth, inst) for i in range(10)]', sort='tottime')


    def Sim(data):
        I = []
        # BEGIN Dataset 0 DO NOT CHANGE
        TopFe.setD(Fe.d)
        TopFe.setSigma(Fe.sigma)
        Pt.setD(cp.Lambda-Fe.d)
        I.append(sample.SimSpecular(data[0].x, inst))
        # END Dataset 0
        return I
