'''
Library for specular magnetic x-ray and neutron reflectivity
============================================================
The magnetic reflectivity is calculated according to: S.A. Stephanov and
S.K Shina PRB 61 15304. for the full anisotropic model. It also one
simpler model where the media is considered to be isotropic but with
different refractive indices for left and right circular light. The
model also has the possibility to calculate the neutron reflectivity
from the same sample structure. This model includes a interface layer
for each ``Layer``. This means that the model is suitable for refining
data that looks for interfacial changes of the magnetic moment. Note!
This model should be considered as a gamma version. It is still under
heavy development and the api can change significantly from version to
version. Should only be used by expert users.

Classes
-------

Layer
~~~~~
``Layer(fr = 1e-20j, b = 1e-20j, dd_u = 0.0, d = 0.0, f = 1e-20j, dens = 1.0, resmag = 1.0,              theta_m = 0.0, fm2 = 1e-20j, xs_ai = 0.0,              sigma_mu = 0.0, fm1 = 1e-20j, dmag_u = 0.0,              mag = 0.0, sigma_ml = 0.0, sigma_c = 0.0,               resdens = 1.0, phi_m = 0.0, dd_l = 0.0, dmag_l = 0.0)``

``d``
   The thickness of the layer in AA (Angstroms = 1e-10m)
``dens``
   The density of formula units in units per Angstroms. Note the units!
``sigma``
   The root mean square roughness of the top interface for the layer in
   Angstroms.
``f``
   The non-resonant x-ray scattering length per formula unit in
   electrons. To be strict it is the number of Thompson scattering
   lengths for each formula unit.
``fr``
   The resonant x-ray scattering length of the resonant species in
   electrons. This is multiplied by ``resdens*dens`` to form the
   resonant scattering length. The total non-magnetic scattering length
   is ``(f + fr*resdens)*dens``.
``fm1``
   The resonant magnetic part of the scattering length - refers to the
   magnetic circular dichroic part. Same units as ``f``
``fm2``
   The resonant magnetic part of the scattering length - refers to the
   magnetic linear dichroic part.
``b``
   The neutron scattering length in fm.
``xs_ai``
   The sum of the absorption cross section and the incoherent scattering
   cross section in barns per formula unit for the neutrons
``mag``
   The magnetic moment per formula unit. The magnetic density is
   ``mag*dens``.
``phi_m``
   The in-plane angle of the magnetic moment of the layer relative the
   projected incident beam for x-rays and relative the polarization axis
   for neutrons.
``theta_m``
   The out-of-plane angle of the magnetic moment. ``theta_m = 0``
   corresponds to an in-plane magnetic moment and ``theta_m``
   corresponds to an out-of-plane magnetic moment.
``dmag_u``
   The relative increase of the magnetic moment in the interface layer.
   Total magnetic moment is ``mag*(1 + dmag_u)``.
``dmag_l``
   As ``dmag_u`` but for the lower interface layer.
``dd_u``
   The width of the upper interface layer in Angstroms.
``sigma_mu``
   The roughness of the upper magnetic interface.
``sigma_ml``
   The roughness of the lower magnetic interface.
``dd_l``
   The width of the lower interface in Angstroms.
``resmag``
   The relative amount of magnetic resonant atoms the total resonant
   magnetic atoms. The total magnetic scattering length is calculated as
   (for the circular dichroic term) ``fm1*resmag*mag*resdens*dens``

Stack
~~~~~
``Stack(Layers = [], Repetitions = 1)``

``Layers``
   A ``list`` consiting of ``Layer``\ s in the stack the first item is
   the layer closest to the bottom
``Repetitions``
   The number of repetitions of the stack

Sample
~~~~~~
``Sample(Stacks = [], dsld_max = 0.1, dsld_offdiag_max = 0.1,              compress = 'yes', slicing = 'no', dsld_n_max = 0.01,              dabs_n_max = 0.01, sld_buffer = 20.0, sld_delta = 5.0,              dmag_max = 0.01, sld_mult = 4.0, slice_depth = 1.0,              Ambient = Amb, Substrate = Sub)``

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
   The maximum allowed step in the scattering length density for x-rays
   (diagonal terms)
``dsld_offdiag_max``
   The maximum allowed step in the scattering length density for the
   offdiagonal terms of the scattering length (magnetic part)
``compress``
   A flag that signals if the sliced composition profile should be
   compressed.
``slicing``
   A flag that signals if the composition profile should be sliced up.
``dsld_n_max``
   The maximum allowed step (in compression) for the neutron scattering
   length.
``dabs_n_max``
   The maximum allowed step (in compression) for the neutron absorption
   (in units of barn/AA^3)
``sld_buffer``
   A buffer for the slicing calculations (to assure convergence in the
   sld profile.
``sld_delta``
   An extra buffer - needed at all?
``dmag_max``
   The maximum allowed step (in compression) for the magnetization.
   Primarily intended to limit the steps in the magnetic profile for
   neutrons.
``sld_mult``
   A multiplication factor for a buffer that takes the roughness into
   account.
``slice_depth``
   The depth of the slices in the calculation of the sliced scattering
   length density profile.

Instrument
~~~~~~~~~~
``model.Instrument(res = 0.001,theory = 'neutron spin-pol',                         footype = 'no corr',beamw = 0.01,                         wavelength = 4.4,respoints = 5,xpol = 'circ+',Ibkg = 0.0,                         I0 = 1.0,samplelen = 10.0,npol = '++',restype = 'no conv',                         coords = 'tth',resintrange = 2)``

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
``theory``
    Defines which theory (code) that should calcualte the reflectivity.
    Should be one of: 'x-ray anis.', 'x-ray simpl. anis.', 'x-ray iso.',
    'neutron spin-pol' or 'neutron spin-flip'.
``xpol``
    The polarization state of the x-ray beam. Should be one of:
    'circ+','circ-','tot', 'ass', 'sigma', 'pi', 'sigma-sigma', 'sigma-pi',
    'pi-pi' or 'pi-sigma'
``npol``
    The neutron polarization state. Should be '++', '--' or '+-','-+' for
    spin flip.
'''

from .lib import refl
from .lib import xrmr
from .lib import edm_slicing as edm
from .lib import paratt as Paratt

from .lib import ables as ables
from .lib import neutron_refl as neutron_refl
from .lib.physical_constants import muB_to_SL
from .lib.instrument import *

mag_limit=1e-8
mpy_limit=1e-8
theta_limit=1e-8

re=2.8179402894e-5

# Preamble to define the parameters needed for the models outlined below:
ModelID='MAGrefl'
# Automatic loading of parameters possible by including this list
__pars__=['Layer', 'Stack', 'Sample', 'Instrument']
# Used for making choices in the GUI
instrument_string_choices={'coords': ['q', 'tth'],
                           'restype': ['no conv', 'fast conv',
                                       'full conv and varying res.',
                                       'fast conv + varying res.'],
                           'footype': ['no corr', 'gauss beam',
                                       'square beam'],
                           'xpol': ['circ+', 'circ-', 'tot', 'ass', 'sigma', 'pi',
                                    'sigma-sigma', 'sigma-pi', 'pi-pi', 'pi-sigma'],
                           'npol': ['++', '--', '+-', 'ass', '-+'],
                           'theory': ['x-ray anis.', 'x-ray simpl. anis.',
                                      'neutron spin-pol', 'neutron spin-flip',
                                      'neutron spin-pol tof', 'x-ray iso.'],
                           # 'compress':['yes', 'no'],
                           # 'slicing':['yes', 'no'],
                           }

InstrumentParameters={'wavelength': 1.54, 'coords': 'tth', 'I0': 1.0, 'res': 0.001,
                      'restype': 'no conv', 'respoints': 5, 'resintrange': 2.0, 'beamw': 0.01, 'footype': 'no corr',
                      'samplelen': 10.0, 'Ibkg': 0.0, 'xpol': 'circ+', 'npol': '++', 'theory': 'x-ray anis.',
                      'incang': 0.2}
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
#
#
InstrumentGroups=[('General', ['wavelength', 'coords', 'I0', 'Ibkg', 'incang']),
                  ('Resolution', ['restype', 'res', 'respoints', 'resintrange']),
                  ('Misc.', ['theory', 'xpol', 'npol', ]),
                  ('Footprint', ['footype', 'beamw', 'samplelen', ]),
                  ]
InstrumentUnits={'wavelength': 'AA', 'coords': '', 'I0': 'arb.', 'res': '[coord]',
                 'restype': '', 'respoints': 'pts.', 'resintrange': '[coord]',
                 'beamw': 'mm', 'footype': '',
                 'samplelen': 'mm', 'Ibkg': 'arb.', 'xpol': '',
                 'theory': '', 'npol': '', 'incang': 'deg'}

LayerParameters={'dens': 1.0, 'd': 0.0, 'f': (0.0+1e-20J),
                 'fr': (0.0+1e-20J),
                 'fm1': (0.0+1e-20J), 'fm2': (0.0+1e-20J),
                 'phi_m': 0.0, 'theta_m': 0.0, 'resdens': 1.0,
                 'resmag': 1.0,
                 'sigma_c': 0.0, 'sigma_ml': 0.0, 'sigma_mu': 0.0,
                 'mag': 0.0,
                 'dmag_l': 0.0, 'dmag_u': 0.0, 'dd_l': 0.0,
                 'dd_u': 0.0, 'b': 1e-20J, 'xs_ai': 0.0,
                 # 'dtheta_l': 0.0, 'dtheta_u':0.0, 'dphi_l':0.0, 'dphi_u':0.0,
                 }
LayerUnits={'dens': 'at./AA^3', 'd': 'AA', 'f': 'el.',
            'fr': 'el.',
            'fm1': 'el./mu_B', 'fm2': 'el./mu_B^2',
            'phi_m': 'deg.', 'theta_m': 'deg.', 'resdens': 'rel.',
            'resmag': 'rel.',
            'sigma_c': 'AA', 'sigma_ml': 'AA', 'sigma_mu': 'AA',
            'mag': 'mu_B',
            'dmag_l': 'rel.', 'dmag_u': 'rel.', 'dd_l': 'AA',
            'dd_u': 'AA', 'b': 'fm', 'xs_ai': 'barn/at.',
            # 'dtheta_l': 0.0, 'dtheta_u':0.0, 'dphi_l':0.0, 'dphi_u':0.0,
            }
LayerGroups=[('Scatt. len.', ['b', 'xs_ai', 'f', 'fr', 'fm1', 'fm2']),
             ('Magnetism', ['mag', 'resmag', 'phi_m', 'theta_m']),
             ('Misc.', ['sigma_c', 'dens', 'resdens', 'd']),
             ('Interf. Mag. Mom.', ['dmag_l', 'dmag_u', 'sigma_ml', 'sigma_mu',
                                    'dd_l', 'dd_u'])
             ]
# ('Interf. Mag. Ang.', ('dtheta_l', 'dtheta_u', 'dphi_l', 'dphi_u'))
StackParameters={'Layers': [], 'Repetitions': 1}
SampleParameters={'Stacks': [], 'Ambient': None, 'Substrate': None,
                  'compress': 'yes', 'slicing': 'no', 'slice_depth': 1.0,
                  'sld_mult': 4.0, 'sld_buffer': 20.0, 'sld_delta': 5.0,
                  'dsld_max': 0.1, 'dsld_offdiag_max': 0.1, 'dsld_n_max': 0.01,
                  'dmag_max': 0.01, 'dabs_n_max': 0.01}

SampleGroups=[['Slicing', ['slicing', 'slice_depth', 'sld_mult', 'sld_buffer',
                           'sld_delta']],
              ['Compression', ['compress', 'dsld_max', 'dsld_offdiag_max',
                               'dmag_max', 'dsld_n_max', 'dabs_n_max']],
              ]

sample_string_choices={'compress': ['yes', 'no'],
                       'slicing': ['yes', 'no'],
                       }

__xlabel__ = "q [Å$^{-1}$]"
__ylabel__ = "Instnsity [a.u.]"

# A buffer to save previous calculations for XRMR calculations
class XBuffer:
    W=None
    parameters=None
    g_0=None
    coords=None
    wavelength=None

# A buffer to save previous calculations for spin-flip calculations
class NBuffer:
    Ruu=0
    Rdd=0
    Rdu=0
    Rud=0
    parameters=None
    TwoThetaQz=None

AA_to_eV=12398.5
''' Conversion from Angstrom to eV E = AA_to_eV/lamda.'''

def correct_reflectivity(R, TwoThetaQz, instrument, theta, weight):
    pol=instrument.getXpol()
    theory=instrument.getTheory()
    if not ((pol in [3, instrument_string_choices['xpol'][3]]) and
            (theory in [0, 1]+instrument_string_choices['theory'][:2])):
        # FootprintCorrections
        foocor=footprint_correction(instrument, theta)
        R=convolute_reflectivity(R, instrument, foocor, TwoThetaQz, weight)
        R=R*instrument.getI0()+instrument.getIbkg()
    else:
        foocor=footprint_correction(instrument, theta)*0+1.0
        R=convolute_reflectivity(R, instrument, foocor, TwoThetaQz, weight)
    return R

def Specular(TwoThetaQz, sample, instrument):
    ''' Simulate the specular signal from sample when proped with instrument
    
    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    '''
    global __xlabel__
    __xlabel__ = "q [Å$^{-1}$]"

    # preamble to get it working with my class interface
    restype=instrument.getRestype()
    xray_energy=AA_to_eV/instrument.getWavelength()
    weight=None
    if restype==2 or restype==instrument_string_choices['restype'][2]:
        (TwoThetaQz, weight)=ResolutionVector(TwoThetaQz[:], instrument.getRes(), instrument.getRespoints(),
                                              range=instrument.getResintrange())
    if instrument.getCoords()==1 or instrument.getCoords()==instrument_string_choices['coords'][1]:
        theta=TwoThetaQz/2
        __xlabel__ = "2θ [°]"
    elif instrument.getCoords()==0 or instrument.getCoords()==instrument_string_choices['coords'][0]:
        theta=arcsin(TwoThetaQz/4/pi*instrument.getWavelength())*180./pi
    if any(theta<theta_limit):
        iprint('Theta min: ', theta.min())
        raise ValueError('The incident angle has to be above %.1e'%theta_limit)

    R=reflectivity_xmag(sample, instrument, theta, TwoThetaQz, xray_energy)

    R=correct_reflectivity(R, TwoThetaQz, instrument, theta, weight)

    return R

def SpecularElectricField(TwoThetaQz, sample, instrument):
    ''' Simulate the specular signal from sample when probed with instrument.
    Returns the wave field (complex number) of the reflected wave.
    No resolution is taken into account.

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    '''
    global __xlabel__
    __xlabel__ = "q [Å$^{-1}$]"

    xray_energy=AA_to_eV/instrument.getWavelength()
    if instrument.getCoords()==1 or instrument.getCoords()==instrument_string_choices['coords'][1]:
        theta=TwoThetaQz/2
        __xlabel__ = "2θ [°]"
    elif instrument.getCoords()==0 or instrument.getCoords()==instrument_string_choices['coords'][0]:
        theta=arcsin(TwoThetaQz/4/pi*instrument.getWavelength())*180./pi
    if any(theta<theta_limit):
        raise ValueError('The incident angle has to be above %.1e'%theta_limit)

    R=reflectivity_xmag(sample, instrument, theta, TwoThetaQz, xray_energy, return_amplitude=False)

    return R

def EnergySpecular(Energy, TwoThetaQz, sample, instrument):
    """ Simulate the specular signal from sample when probed with instrument. Energy should be in eV.

    # BEGIN Parameters
    Energy data.x
    TwoThetaQz 3.0
    # END Parameters
    """
    global __xlabel__
    __xlabel__ = "E [eV]"

    restype=instrument.getRestype()
    # TODO: Fix so that resolution can be included.
    if restype!=0 and restype!=instrument_string_choices['restype'][0]:
        raise ValueError('Only no resolution is allowed for energy scans.')

    wl=AA_to_eV/Energy
    if isscalar(TwoThetaQz):
        TwoThetaQz=TwoThetaQz*ones(Energy.shape)
    # TTH values given as x
    if instrument.getCoords()==instrument_string_choices['coords'][1] \
            or instrument.getCoords()==1:
        theta=TwoThetaQz/2.0
    # Q vector given....
    elif instrument.getCoords()==instrument_string_choices['coords'][0] \
            or instrument.getCoords()==0:
        theta=arcsin(TwoThetaQz*wl/4/pi)*180.0/pi

    else:
        raise ValueError('The value for coordinates, coords, is WRONG!'
                         'should be q(0) or tth(1).')

    if any(theta<theta_limit):
        raise ValueError('The incident angle has to be above %.1e'%theta_limit)

    R=reflectivity_xmag(sample, instrument, theta, TwoThetaQz, Energy)

    # TODO: Fix Corrections
    # R = correct_reflectivity(R, TwoThetaQz, instrument, theta, weight)
    R=R*instrument.getI0()+instrument.getIbkg()

    return R

def EnergySpecularField(Energy, TwoThetaQz, sample, instrument):
    """ Simulate the specular signal from sample when probed with instrument. Energy should be in eV.
    Returns the wave field (complex number) of the reflected wave.
    No resolution is taken into account.

    # BEGIN Parameters
    Energy data.x
    TwoThetaQz 3.0
    # END Parameters
    """
    global __xlabel__
    __xlabel__ = "E [eV]"

    restype=instrument.getRestype()
    # TODO: Fix so that resolution can be included.
    if restype!=0 and restype!=instrument_string_choices['restype'][0]:
        raise ValueError('Only no resolution is allowed for energy scans.')

    wl=AA_to_eV/Energy
    if isscalar(TwoThetaQz):
        TwoThetaQz=TwoThetaQz*ones(Energy.shape)
    # TTH values given as x
    if instrument.getCoords()==instrument_string_choices['coords'][1] \
            or instrument.getCoords()==1:
        theta=TwoThetaQz/2.0
    # Q vector given....
    elif instrument.getCoords()==instrument_string_choices['coords'][0] \
            or instrument.getCoords()==0:
        theta=arcsin(TwoThetaQz*wl/4/pi)*180.0/pi

    else:
        raise ValueError('The value for coordinates, coords, is WRONG!'
                         'should be q(0) or tth(1).')

    if any(theta<theta_limit):
        raise ValueError('The incident angle has to be above %.1e'%theta_limit)

    R=reflectivity_xmag(sample, instrument, theta, TwoThetaQz, Energy, return_amplitude=False)

    # TODO: Fix Corrections
    # R = correct_reflectivity(R, TwoThetaQz, instrument, theta, weight)
    return R

def OffSpecular(TwoThetaQz, ThetaQx, sample, instrument):
    ''' Function that simulates the off-specular signal (not implemented)
    
    # BEGIN Parameters
    TwoThetaQz 1.0
    ThetaQx data.x
    # END Parameters
    '''
    raise NotImplementedError('Off specular calculations are not implemented for magnetic x-ray reflectivity')

def SLD_calculations(z, item, sample, inst):
    ''' Calculates the scatteringlength density as at the positions z
    if item is None or "all" the function returns a dictonary of values.
    Otherwise it returns the item as identified by its string.
    
    # BEGIN Parameters
    z data.x
    item "Re sld_c"
    # END Parameters
    '''
    use_slicing=sample.getSlicing()
    if use_slicing==1 or use_slicing==sample_string_choices['slicing'][1]:
        return compose_sld_anal(z, sample, inst)
    lamda=inst.getWavelength()
    theory=inst.getTheory()
    xray_energy=AA_to_eV/inst.getWavelength()
    (d, sl_c, sl_m1, sl_m2, M, chi, non_mag, mpy,
     sl_n, abs_n, mag_dens, mag_dens_x, mag_dens_y, z0)=compose_sld(sample, inst, array([0.0, ]), xray_energy)
    z=zeros(len(d)*2)
    z[::2]=cumsum(r_[0, d[:-1]])
    z[1::2]=cumsum(r_[d])
    z+=z0

    dic={'z': z}

    if theory in [0, instrument_string_choices['theory'][0]]:
        # Full theory return the suceptibility matrix

        new_size=len(d)*2
        sl_cp=zeros(new_size, dtype=complex128)
        sl_cp[::2]=sl_c
        sl_cp[1::2]=sl_c
        sl_m1p=zeros(new_size, dtype=complex128)
        sl_m1p[::2]=sl_m1
        sl_m1p[1::2]=sl_m1
        sl_m2p=zeros(new_size, dtype=complex128)
        sl_m2p[::2]=sl_m2
        sl_m2p[1::2]=sl_m2

        def interleave(a):
            new_a=zeros(len(a)*2, dtype=complex128)
            new_a[::2]=a
            new_a[1::2]=a
            return new_a

        chi=[[interleave(c) for c in ch] for ch in chi]

        # return {'real sld_c': sl_cp.real, 'imag sld_c': sl_cp.imag,
        #        'real sld_m1': sl_m1p.real, 'imag sld_m1': sl_m1p.imag,
        #        'real sld_m2': sl_m2p.real, 'imag sld_m2': sl_m2p.imag,
        #        'z':z}
        re=2.8179402894e-5
        c=1/(lamda**2*re/pi)
        dic={'Re sl_xx': chi[0][0].real*c, 'Re sl_xy': chi[0][1].real*c, 'Re sl_xz': chi[0][2].real*c,
             'Re sl_yy': chi[1][1].real*c, 'Re sl_yz': chi[1][2].real*c, 'Re sl_zz': chi[2][2].real*c,
             'Im sl_xx': chi[0][0].imag*c, 'Im sl_xy': chi[0][1].imag*c, 'Im sl_xz': chi[0][2].imag*c,
             'Im sl_yy': chi[1][1].imag*c, 'Im sl_yz': chi[1][2].imag*c, 'Im sl_zz': chi[2][2].imag*c,
             'z': z, 'SLD unit': 'r_e/\AA^{3}'}
    else:
        new_size=len(d)*2

        def parray(ar):
            tmp=zeros(new_size, dtype=complex128)
            tmp[::2]=ar
            tmp[1::2]=ar
            return tmp

        sl_cp=parray(sl_c)
        sl_m1p=parray(sl_m1)
        sl_np=parray(sl_n)
        mag_densp=parray(mag_dens)
        mag_dens_xp=parray(mag_dens_x)
        mag_dens_yp=parray(mag_dens_y)

        abs_np=parray(abs_n)
        if theory in [1, instrument_string_choices['theory'][1]]:
            # Simplified anisotropic
            # print sl_cp.shape, sl_np.shape, abs_np.shape, mag_densp.shape, z.shape
            dic={'Re sld_c': sl_cp.real, 'Im sld_c': sl_cp.imag,
                 'Re sld_m': sl_m1p.real, 'Im sld_m': sl_m1p.imag,
                 'mag_dens': mag_densp,
                 'z': z, 'SLD unit': 'r_{e}/\AA^{3},\,\mu_{B}/\AA^{3}'}
        elif theory in [2, instrument_string_choices['theory'][2]]:
            # Neutron spin pol
            dic={'sld_n': sl_np, 'abs_n': abs_np, 'mag_dens': mag_densp,
                 'z': z, 'SLD unit': 'fm/\AA^{3}, b/\AA^{3},\,\mu_{B}/\AA^{3}'}
        elif theory in [3, instrument_string_choices['theory'][3]]:
            # Neutron spin pol with spin flip
            dic={'sld_n': sl_np, 'abs_n': abs_np, 'mag_dens': mag_densp, 'mag_dens_x': mag_dens_xp,
                 'mag_dens_y': mag_dens_yp,
                 'z': z, 'SLD unit': 'fm/\AA^{3}, b/\AA^{3},\,\mu_{B}/\AA^{3}'}
        elif theory in [4, instrument_string_choices['theory'][4]]:
            # Neutron spin pol
            dic={'sld_n': sl_np, 'abs_n': abs_np, 'mag_dens': mag_densp,
                 'z': z, 'SLD unit': 'fm/\AA^{3}, b/\AA^{3},\,\mu_{B}/\AA^{3}'}
        if theory in [5, instrument_string_choices['theory'][5]]:
            # isotropic (normal x-ray reflectivity)
            dic={'Re sld_c': sl_cp.real, 'Im sld_c': sl_cp.imag,
                 'z': z, 'SLD unit': 'r_{e}/\AA^{3}'}

    if item is None or item=='all':
        return dic
    else:
        try:
            return dic[item]
        except:
            raise ValueError('The chosen item, %s, does not exist'%item)

def neturon_sld(abs_xs, b, dens, wl):
    return dens*(wl**2/2/pi*b-1.0J*abs_xs*wl/4/pi)/1e-5/(wl**2/2/pi)

def compose_sld_anal(z, sample, instrument):
    '''Compose a analytical profile funciton'''

    def sld_interface(z, drho_jm1_l, drho_j, drho_j_u, sigma_jm1_l, sigma_j, sigma_j_u, dd_jm1_l, dd_j_u):
        ''' Calculate the sld of one interface '''
        sld=drho_j_u*(0.5+0.5*erf((z-dd_j_u)/sqrt(2*(sigma_j_u**2+sigma_j**2))))
        sld+=drho_jm1_l*(0.5+0.5*erf((z+dd_jm1_l)/sqrt(2*(sigma_jm1_l**2+sigma_j**2))))
        sld+=drho_j*(0.5+0.5*erf(z/sqrt(2)/sigma_j))
        return sld

    def calc_sld(z, int_pos, sld, sld_l, sld_u, sigma_l, sigma_c, sigma_u, dd_l, dd_u):
        return (sum(sld_interface(-(z[:, newaxis]-int_pos), -(sld[1:]-sld_l[1:]),
                                  -(sld_l[1:]-sld_u[:-1]), -(sld_u[:-1]-sld[:-1]),
                                  sigma_l[1:], sigma_c[:-1], sigma_u[:-1],
                                  dd_l[1:], dd_u[:-1]), 1)+sld[-1])

    re=2.8179402894e-5
    lamda=instrument.getWavelength()
    parameters=sample.resolveLayerParameters()
    dens=array(parameters['dens'], dtype=float64)
    resdens=array(parameters['resdens'], dtype=float64)
    resmag=array(parameters['resmag'], dtype=float64)
    mag=array(parameters['mag'], dtype=float64)

    dmag_l=array(parameters['dmag_l'], dtype=float64)
    dmag_u=array(parameters['dmag_u'], dtype=float64)
    dd_l=array(parameters['dd_l'], dtype=float64)
    dd_u=array(parameters['dd_u'], dtype=float64)
    # print [type(f) for f in parameters['f']]
    xray_energy=AA_to_eV/instrument.getWavelength()
    f=refl.cast_to_array(parameters['f'], xray_energy)
    fr=refl.cast_to_array(parameters['fr'], xray_energy)
    fm1=refl.cast_to_array(parameters['fm1'], xray_energy)
    fm2=refl.cast_to_array(parameters['fm2'], xray_energy)

    d=array(parameters['d'], dtype=float64)

    phi=array(parameters['phi_m'], dtype=float64)*pi/180.0
    theta_m=array(parameters['theta_m'], dtype=float64)*pi/180.0
    # Create an offset of phi to handle negative mag values and negate theta_m
    phi=phi+where(mag<0, pi, 0)
    theta_m=where(mag<0, -theta_m, theta_m)
    mag=abs(mag)
    m_x=cos(theta_m)*cos(phi)
    sl_c=(dens*(f+resdens*fr))
    sl_m1=dens*resdens*resmag*mag*fm1
    sl_m2=dens*resdens*resmag*mag**2*fm2
    sigma_c=array(parameters['sigma_c'], dtype=float64)+1e-20
    sigma_l=array(parameters['sigma_ml'], dtype=float64)+1e-20
    sigma_u=array(parameters['sigma_mu'], dtype=float64)+1e-20
    sl_m1_l=sl_m1*(1.+dmag_l)
    sl_m1_u=sl_m1*(1.+dmag_u)
    sl_m2_l=sl_m2*(1.+dmag_l)**2
    sl_m2_u=sl_m2*(1.+dmag_u)**2

    b=(array(parameters['b'], dtype=complex128))*1e-5
    abs_xs=(array(parameters['xs_ai'], dtype=complex128))*1e-4**2
    wl=instrument.getWavelength()
    sl_n=neturon_sld(abs_xs, b, dens, wl)
    mag_d=mag*dens
    mag_d_l=mag_d*(1.+dmag_l)
    mag_d_u=mag_d*(1.+dmag_u)

    int_pos=cumsum(r_[0, d[1:-1]])
    if z is None:
        z=arange(-sigma_c[0]*10-50, int_pos.max()+sigma_c.max()*10+50, 0.5)
    # Note: First layer substrate and last ambient
    sld_c=calc_sld(z, int_pos, sl_c, sl_c, sl_c, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
    sld_m=calc_sld(z, int_pos, sl_m1*m_x, sl_m1_l*m_x, sl_m1_u*m_x, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
    sld_n=calc_sld(z, int_pos, sl_n, sl_n, sl_n, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
    mag_dens=calc_sld(z, int_pos, mag_d, mag_d_l, mag_d_u, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
    mag_dens_x=calc_sld(z, int_pos, mag_d*cos(theta_m)*cos(phi), mag_d_l*cos(theta_m)*cos(phi),
                        mag_d_u*cos(theta_m)*cos(phi), sigma_l, sigma_c, sigma_u, dd_l, dd_u)
    mag_dens_y=calc_sld(z, int_pos, mag_d*cos(theta_m)*sin(phi), mag_d_l*cos(theta_m)*sin(phi),
                        mag_d_u*cos(theta_m)*sin(phi), sigma_l, sigma_c, sigma_u, dd_l, dd_u)

    theory=instrument.getTheory()
    if theory in [0, instrument_string_choices['theory'][0]]:
        # Full polarization calc
        c=1/(lamda**2*re/pi)
        A=-sl_c/c
        B=sl_m1/c
        C=sl_m2/c

        M=c_[cos(theta_m)*cos(phi), cos(theta_m)*sin(phi), sin(theta_m)]
        chi=xrmr.create_chi(None, None, A*0, A, B, C, M, None)[0]
        chi_l=xrmr.create_chi(None, None, A*0, A, B*(1+dmag_l), C*(1+dmag_l)**2, M, None)[0]
        chi_u=xrmr.create_chi(None, None, A*0, A, B*(1+dmag_u), C*(1+dmag_u)**2, M, None)[0]

        chi_xx, chi_xy, chi_xz=chi[0];
        chi_yx, chi_yy, chi_yz=chi[1];
        chi_zx, chi_zy, chi_zz=chi[2]
        chi_l_xx, chi_l_xy, chi_l_xz=chi_l[0];
        chi_l_yx, chi_l_yy, chi_l_yz=chi_l[1];
        chi_l_zx, chi_l_zy, chi_l_zz=chi_l[2]
        chi_u_xx, chi_u_xy, chi_u_xz=chi_u[0];
        chi_u_yx, chi_u_yy, chi_u_yz=chi_u[1];
        chi_u_zx, chi_u_zy, chi_u_zz=chi_u[2]
        c_xx=calc_sld(z, int_pos, chi_xx, chi_l_xx, chi_u_xx, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
        c_xy=calc_sld(z, int_pos, chi_xy, chi_l_xy, chi_u_xy, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
        c_xz=calc_sld(z, int_pos, chi_xz, chi_l_xz, chi_u_xz, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
        c_yy=calc_sld(z, int_pos, chi_yy, chi_l_yy, chi_u_yy, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
        c_yz=calc_sld(z, int_pos, chi_yz, chi_l_yz, chi_u_yz, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
        c_zz=calc_sld(z, int_pos, chi_zz, chi_l_zz, chi_u_zz, sigma_l, sigma_c, sigma_u, dd_l, dd_u)

        return {'Re sl_xx': c_xx.real*c, 'Re sl_xy': c_xy.real*c, 'Re sl_xz': c_xz.real*c,
                'Re sl_yy': c_yy.real*c, 'Re sl_yz': c_yz.real*c, 'Re sl_zz': c_zz.real*c,
                'Im sl_xx': c_xx.imag*c, 'Im sl_xy': c_xy.imag*c, 'Im sl_xz': c_xz.imag*c,
                'Im sl_yy': c_yy.imag*c, 'Im sl_yz': c_yz.imag*c, 'Im sl_zz': c_zz.imag*c,
                'z': z, 'SLD unit': 'r_e/\AA^{3}'}
    elif theory in [1, instrument_string_choices['theory'][1]]:
        # Simplified anisotropic
        return {'Re sld_c': sld_c.real, 'Im sld_c': sld_c.imag,
                'Re sld_m': sld_m.real, 'Im sld_m': sld_m.imag,
                'mag_dens': mag_dens,
                'z': z, 'SLD unit': 'r_{e}/\AA^{3},\,\mu_{B}/\AA^{3}'}
    elif theory in [2, instrument_string_choices['theory'][2]]:
        # Neutron spin pol
        return {'sld_n': sld_n, 'mag_dens': mag_dens,
                'z': z, 'SLD unit': 'fm/\AA^{3}, \mu_{B}/\AA^{3}'}
    elif theory in [3, instrument_string_choices['theory'][3]]:
        # Neutron spin pol with spin flip
        return {'sld_n': sld_n, 'mag_dens': mag_dens, 'mag_dens_x': mag_dens_x, 'mag_dens_y': mag_dens_y,
                'z': z, 'SLD unit': 'fm/\AA^{3}, \mu_{B}/\AA^{3}'}
    elif theory in [4, instrument_string_choices['theory'][4]]:
        # Neutron spin pol tof
        return {'sld_n': sld_n, 'mag_dens': mag_dens,
                'z': z, 'SLD unit': 'fm/\AA^{3}, \mu_{B}/\AA^{3}'}
    elif theory in [5, instrument_string_choices['theory'][5]]:
        # x-ray isotropic (normal x-ray reflectivity)
        return {'Re sld_c': sld_c.real, 'Im sld_c': sld_c.imag,
                'z': z, 'SLD unit': 'r_{e}/\AA^{3},\,\mu_{B}/\AA^{3}'}
    else:
        raise ValueError('Wrong value of theory given. Value: %s'%theory)

def compose_sld(sample, instrument, theta, xray_energy, layer=None):
    """ Composes the sld for a slicing model

    Parameters:
        sample: The sample
        instrument: The instrument
        theta: The incident angle
        xray_energy: The xray energy either scalar or array
        layer: Defines which layer number to return. If None (default) returns the entire profile.
    """
    re=2.8179402894e-5
    parameters=sample.resolveLayerParameters()
    dmag_l=array(parameters['dmag_l'], dtype=float64)
    dmag_u=array(parameters['dmag_u'], dtype=float64)
    dd_u=array(parameters['dd_u'], dtype=float64)
    dd_l=array(parameters['dd_l'], dtype=float64)
    d=array(parameters['d'], dtype=float64)
    mag=array(parameters['mag'], dtype=float64)

    if isscalar(xray_energy):
        shape=None
    else:
        shape=(d.shape[0], xray_energy.shape[0])
    lamda=AA_to_eV/xray_energy
    dens=refl.harm_sizes(parameters['dens'], shape, dtype=float64)
    resdens=refl.harm_sizes(parameters['resdens'], shape, dtype=float64)
    resmag=refl.harm_sizes(parameters['resmag'], shape, dtype=float64)

    f=refl.harm_sizes(refl.cast_to_array(parameters['f'], xray_energy), shape, dtype=complex128)
    fr=refl.harm_sizes(refl.cast_to_array(parameters['fr'], xray_energy), shape, dtype=complex128)
    fm1=refl.harm_sizes(refl.cast_to_array(parameters['fm1'], xray_energy), shape, dtype=complex128)
    fm2=refl.harm_sizes(refl.cast_to_array(parameters['fm2'], xray_energy), shape, dtype=complex128)

    sl_c=dens*(f+resdens*fr)
    sl_m1=dens*resdens*resmag*fm1
    sl_m2=dens*resdens*resmag*fm2  # mag is multiplied in later

    theory=instrument.getTheory()
    if theory in [1, instrument_string_choices['theory'][1]]:
        # If simplified theory set sl_m2 to zero to be able to back calculate B
        sl_m2*=0

    phi=array(parameters['phi_m'], dtype=float64)*pi/180.0
    theta_m=array(parameters['theta_m'], dtype=float64)*pi/180.0
    # Create an offset of phi to handle negative mag values and negate theta_m
    phi=phi+where(mag<0, pi, 0)
    theta_m=where(mag<0, -theta_m, theta_m)
    mag=abs(mag)

    M=c_[cos(theta_m)*cos(phi), cos(theta_m)*sin(phi), sin(theta_m)]

    sigma_c=array(parameters['sigma_c'], dtype=float64)
    sigma_mu=sqrt(array(parameters['sigma_mu'], dtype=float64)[:-1]**2+sigma_c[:-1]**2)
    sigma_ml=sqrt(array(parameters['sigma_ml'], dtype=float64)[1:]**2+sigma_c[:-1]**2)

    # Neutrons
    wl=instrument.getWavelength()
    abs_xs=array(parameters['xs_ai'], dtype=complex128)  # *(1e-4)**2
    b=array(parameters['b'], dtype=complex128)  # *1e-5
    dens_n=array(parameters['dens'], dtype=float64)
    sl_n=dens_n*b
    abs_n=dens_n*abs_xs

    if sample.getSlicing()==sample_string_choices['slicing'][0]:
        dz=sample.getSlice_depth()
        reply=edm.create_profile_cm2(d[1:-1], sigma_c[:-1].real,
                                     sigma_ml.real, sigma_mu.real,
                                     [edm.erf_profile]*len(d),
                                     [edm.erf_interf]*len(d),
                                     dmag_l, dmag_u, mag, dd_l, dd_u,
                                     dz=dz, mult=sample.getSld_mult(),
                                     buffer=sample.getSld_buffer(),
                                     delta=sample.getSld_delta())
        z, comp_prof, mag_prof=reply
        if not shape is None:
            new_shape=(shape[0], comp_prof.shape[1], shape[1])
        else:
            new_shape=None
        comp_prof_x=refl.harm_sizes(comp_prof, new_shape, dtype=float64)
        mag_prof_x=refl.harm_sizes(mag_prof, new_shape, dtype=float64)
        sl_c_lay=comp_prof_x*sl_c[:, newaxis]
        sl_m1_lay=comp_prof_x*mag_prof_x*sl_m1[:, newaxis]
        sl_m2_lay=comp_prof_x*mag_prof_x**2*sl_m2[:, newaxis]

        # Neutrons
        sl_n_lay=comp_prof*sl_n[:, newaxis]
        abs_n_lay=comp_prof*abs_n[:, newaxis]
        mag_dens_lay=comp_prof*mag_prof*dens_n[:, newaxis]

        if not shape is None:
            M=rollaxis(array((ones(comp_prof_x.shape)*M[:, 0][:, newaxis, newaxis],
                              ones(comp_prof_x.shape)*M[:, 1][:, newaxis, newaxis],
                              ones(comp_prof_x.shape)*M[:, 2][:, newaxis, newaxis])), 0, 2)
        else:
            M=rollaxis(array((ones(comp_prof_x.shape)*M[:, 0][:, newaxis],
                              ones(comp_prof_x.shape)*M[:, 1][:, newaxis],
                              ones(comp_prof_x.shape)*M[:, 2][:, newaxis])), 0, 2)

        A=-lamda**2*re/pi*sl_c_lay
        B=lamda**2*re/pi*sl_m1_lay
        C=lamda**2*re/pi*sl_m2_lay
        g_0=sin(theta*pi/180.0)

        chi, non_mag, mpy=xrmr.create_chi(g_0, lamda, A, 0.0*A, B, C, M, d)
        if layer is not None:
            sl_c=sl_c_lay[layer]
            sl_m1=sl_m1_lay[layer]
            sl_m2=sl_m2_lay[layer]
            sl_n=sl_n_lay[layer]
            abs_n=abs_n_lay[layer]
            mag_dens=mag_dens_lay[layer]
            mag_dens_x=(comp_prof*mag_prof*(dens_n*cos(theta_m)*cos(phi))[:, newaxis])[layer]
            mag_dens_y=(comp_prof*mag_prof*(dens_n*cos(theta_m)*sin(phi))[:, newaxis])[layer]
            chi=tuple([c[layer] for c in chi[0]+chi[1]+chi[2]])
        else:
            sl_c=sl_c_lay.sum(0)
            sl_m1=sl_m1_lay.sum(0)
            sl_m2=sl_m2_lay.sum(0)
            sl_n=sl_n_lay.sum(0)
            abs_n=abs_n_lay.sum(0)
            mag_dens=mag_dens_lay.sum(0)
            mag_dens_x=(comp_prof*mag_prof*(dens_n*cos(theta_m)*cos(phi))[:, newaxis]).sum(0)
            mag_dens_y=(comp_prof*mag_prof*(dens_n*cos(theta_m)*sin(phi))[:, newaxis]).sum(0)
            chi=tuple([c.sum(0) for c in chi[0]+chi[1]+chi[2]])

        if sample.getCompress()==sample_string_choices['compress'][0]:
            # Compressing the profile..
            lamda_max=lamda if isscalar(lamda) else lamda.max()
            dsld_max=sample.getDsld_max()
            dchi_max=dsld_max*lamda_max**2*re/pi
            dsld_offdiag_max=sample.getDsld_offdiag_max()
            dsld_n_max=sample.getDsld_n_max()
            dabs_n_max=sample.getDabs_n_max()
            dmag_max=sample.getDmag_max()
            dchi_od_max=dsld_offdiag_max*lamda_max**2*re/pi

            index, z=edm.compress_profile_index_n(z, chi+(sl_n, mag_dens, abs_n, mag_dens_x, mag_dens_y),
                                                  (dchi_max, dchi_od_max, dchi_od_max,
                                                   dchi_od_max, dchi_max, dchi_od_max,
                                                   dchi_od_max, dchi_od_max, dchi_max,
                                                   dsld_n_max, dmag_max, dabs_n_max, dmag_max, dmag_max,
                                                   ))
            reply=edm.create_compressed_profile((sl_c, sl_m1, sl_m2)+
                                                chi+(sl_n, mag_dens, abs_n, mag_dens_x, mag_dens_y),
                                                index)
            (sl_c, sl_m1, sl_m2, chi_xx, chi_xy, chi_xz, chi_yx, chi_yy, chi_yz, chi_zx, chi_zy, chi_zz,
             sl_n, mag_dens, abs_n, mag_dens_x, mag_dens_y)=reply
            non_mag=((abs(chi_xy)<mag_limit)
                     *(abs(chi_xz)<mag_limit)
                     *(abs(chi_yz)<mag_limit))
            mpy=(abs(chi_yz)/abs(chi_xx)<mpy_limit)*(abs(chi_xy)/abs(chi_xx)<mpy_limit)*bitwise_not(non_mag)
            # print mpy
            chi=((chi_xx, chi_xy, chi_xz), (chi_yx, chi_yy, chi_yz), (chi_zx, chi_zy, chi_zz))
        else:
            (chi_xx, chi_xy, chi_xz, chi_yx, chi_yy, chi_yz, chi_zx, chi_zy, chi_zz)=chi
            non_mag=((abs(chi_xy)<mag_limit)
                     *(abs(chi_xz)<mag_limit)
                     *(abs(chi_yz)<mag_limit))
            non_mag[0]=True
            mpy=(abs(chi_yz)/abs(chi_xx)<mpy_limit)*(abs(chi_xy)/abs(chi_xx)<mpy_limit)*bitwise_not(non_mag)
            chi=((chi_xx, chi_xy, chi_xz), (chi_yx, chi_yy, chi_yz), (chi_zx, chi_zy, chi_zz))
        d=r_[z[1:]-z[:-1], 1]
    else:
        re=2.8179402894e-5
        A=-lamda**2*re/pi*sl_c
        B=lamda**2*re/pi*sl_m1
        C=lamda**2*re/pi*sl_m2
        g_0=sin(theta*pi/180.0)
        chi, non_mag, mpy=xrmr.create_chi(g_0, lamda, A, 0.0*A,
                                          B, C, M, d)

    return d, sl_c, sl_m1, sl_m2, M, chi, non_mag, mpy, sl_n, abs_n, mag_dens, mag_dens_x, mag_dens_y, z[0]

def extract_anal_iso_pars(sample, instrument, theta, xray_energy, pol='+', Q=None):
    ''' Note Q is only used for Neutron TOF
    :param lamda:
    '''
    re=2.8179402894e-5
    parameters=sample.resolveLayerParameters()
    dmag_l=array(parameters['dmag_l'], dtype=float64)
    dmag_u=array(parameters['dmag_u'], dtype=float64)
    dd_u=array(parameters['dd_u'], dtype=float64)
    dd_l=array(parameters['dd_l'], dtype=float64)
    d=array(parameters['d'], dtype=float64)
    mag=array(parameters['mag'], dtype=float64)

    shape=(d.shape[0], theta.shape[0])
    lamda=AA_to_eV/xray_energy
    dens=refl.harm_sizes(parameters['dens'], shape, dtype=float64)
    resdens=refl.harm_sizes(parameters['resdens'], shape, dtype=float64)
    resmag=refl.harm_sizes(parameters['resmag'], shape, dtype=float64)

    f=refl.harm_sizes(refl.cast_to_array(parameters['f'], xray_energy), shape, dtype=complex128)
    fr=refl.harm_sizes(refl.cast_to_array(parameters['fr'], xray_energy), shape, dtype=complex128)
    fm1=refl.harm_sizes(refl.cast_to_array(parameters['fm1'], xray_energy), shape, dtype=complex128)
    fm2=refl.harm_sizes(refl.cast_to_array(parameters['fm2'], xray_energy), shape, dtype=complex128)

    d=array(parameters['d'], dtype=float64)

    theta=theta*pi/180.0
    phi=array(parameters['phi_m'], dtype=float64)*pi/180.0
    theta_m=array(parameters['theta_m'], dtype=float64)*pi/180.0
    # Create an offset of phi to handle negative mag values and negate theta_m
    phi=phi+where(mag<0, pi, 0)
    theta_m=where(mag<0, -theta_m, theta_m)
    mag=abs(mag)
    sl_c=(dens*(f+resdens*fr))
    m_x=refl.harm_sizes((cos(theta_m)*cos(phi)), shape)
    sl_m1=(dens*resdens*resmag*refl.harm_sizes(mag, shape)*fm1)*cos(theta)*m_x

    sigma_c=array(parameters['sigma_c'], dtype=float64)
    sigma_l=array(parameters['sigma_ml'], dtype=float64)
    sigma_u=array(parameters['sigma_mu'], dtype=float64)

    theory=instrument.getTheory()

    if theory in [0, 1]+instrument_string_choices['theory'][:2]:
        if pol=='+':
            n=1-lamda**2*re/pi*(sl_c+sl_m1)/2.0
            n_l=1-lamda**2*re/pi*(sl_c+sl_m1*(1.+dmag_l)[:, newaxis])/2.0
            n_u=1-lamda**2*re/pi*(sl_c+sl_m1*(1.+dmag_u)[:, newaxis])/2.0
        elif pol=='-':
            n=1-lamda**2*re/pi*(sl_c-sl_m1)/2.0
            n_l=1-lamda**2*re/pi*(sl_c-sl_m1*(1.+dmag_l)[:, newaxis])/2.0
            n_u=1-lamda**2*re/pi*(sl_c-sl_m1*(1.+dmag_u)[:, newaxis])/2.0
    elif theory in [2, 3]+instrument_string_choices['theory'][2:4]:
        b=(array(parameters['b'], dtype=complex128)*1e-5)[:, newaxis]*ones(theta.shape)
        abs_xs=(array(parameters['xs_ai'], dtype=complex128)*1e-4**2)[:, newaxis]*ones(theta.shape)
        wl=instrument.getWavelength()*1.0
        sld=dens*(wl**2/2.0/pi*sqrt(b**2-(abs_xs/2.0/wl)**2)-1.0J*abs_xs*wl/4/pi)
        # print mag.shape, dens.shape, theta_m.shape, phi.shape, theta.shape
        msld=(muB_to_SL*mag*wl**2/2/pi*cos(theta_m)*cos(phi))[:, newaxis]*dens*ones(theta.shape)
        if pol in ['++', 'uu']:
            n=1.0-sld-msld
            n_l=1.0-sld-msld*(1.0+dmag_l)[:, newaxis]
            n_u=1.0-sld-msld*(1.0+dmag_u)[:, newaxis]
        if pol in ['--', 'dd']:
            n=1.0-sld+msld
            n_l=1.0-sld+msld*(1.0+dmag_l)[:, newaxis]
            n_u=1.0-sld+msld*(1.0+dmag_u)[:, newaxis]
    elif theory in [4, instrument_string_choices['theory'][4]]:
        wl=4*pi*sin(instrument.getIncang()*pi/180)/Q
        b=(array(parameters['b'], dtype=complex128)*1e-5)[:, newaxis]*ones(wl.shape)
        abs_xs=(array(parameters['xs_ai'], dtype=complex128)*1e-4**2)[:, newaxis]*ones(wl.shape)
        sld=dens[:, newaxis]*(wl**2/2/pi*sqrt(b**2-(abs_xs/2.0/wl)**2)-
                              1.0J*abs_xs*wl/4/pi)
        msld=(muB_to_SL*(mag*dens)[:, newaxis]*wl**2/2/pi)*(cos(theta_m)*cos(phi))[:, newaxis]

        if pol in ['++', 'uu']:
            n=1.0-sld-msld
            n_l=1.0-sld-msld*(1.0+dmag_l)[:, newaxis]
            n_u=1.0-sld-msld*(1.0+dmag_u)[:, newaxis]
        elif pol in ['--', 'dd']:
            n=1.0-sld+msld
            n_l=1.0-sld+msld*(1.0+dmag_l)[:, newaxis]
            n_u=1.0-sld+msld*(1.0+dmag_u)[:, newaxis]
        else:
            raise ValueError('An unexpected value of pol was given. Value: %s'%(pol,))
    elif theory in [5, instrument_string_choices['theory'][5]]:
        n=1-lamda**2*re/pi*sl_c/2.0
        n_l=1-lamda**2*re/pi*sl_c/2.0
        n_u=1-lamda**2*re/pi*sl_c/2.0
    else:
        raise ValueError('An unexpected value of theory was given. Value: %s'%(theory,))
    d=d-dd_u-dd_l
    d*=(d>=0)
    return n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l

def reflectivity_xmag(sample, instrument, theta, TwoThetaQz, xray_energy, return_amplitude=True):
    use_slicing=sample.getSlicing()
    if use_slicing==0 or use_slicing==sample_string_choices['slicing'][0]:
        R=slicing_reflectivity(sample, instrument, theta, TwoThetaQz, xray_energy, return_amplitude=return_amplitude)
    elif use_slicing==1 or use_slicing==sample_string_choices['slicing'][1]:
        R=analytical_reflectivity(sample, instrument, theta, TwoThetaQz, xray_energy,
                                  return_amplitude=return_amplitude)
    else:
        raise ValueError('Unkown input to the slicing parameter')
    return R

def analytical_reflectivity(sample, instrument, theta, TwoThetaQz, xray_energy, return_amplitude=True):
    # lamda = instrument.getWavelength()
    theory=instrument.getTheory()
    re=2.8179402894e-5
    parameters=sample.resolveLayerParameters()
    dmag_l=array(parameters['dmag_l'], dtype=float64)
    dmag_u=array(parameters['dmag_u'], dtype=float64)
    dd_u=array(parameters['dd_u'], dtype=float64)
    dd_l=array(parameters['dd_l'], dtype=float64)

    sigma=array(parameters['sigma_c'], dtype=float64)+1e-9
    sigma_u=array(parameters['sigma_mu'], dtype=float64)+1e-9
    sigma_l=array(parameters['sigma_ml'], dtype=float64)+1e-9

    d=array(parameters['d'], dtype=float64)
    # lamda = instrument.getWavelength()
    if isscalar(xray_energy):
        shape=None
    else:
        shape=(d.shape[0], xray_energy.shape[0])
    lamda=AA_to_eV/xray_energy
    dens=refl.harm_sizes(parameters['dens'], shape, dtype=float64)
    resdens=refl.harm_sizes(parameters['resdens'], shape, dtype=float64)
    resmag=refl.harm_sizes(parameters['resmag'], shape, dtype=float64)
    mag=array(parameters['mag'], dtype=float64)

    f=refl.harm_sizes(refl.cast_to_array(parameters['f'], xray_energy), shape, dtype=complex128)
    fr=refl.harm_sizes(refl.cast_to_array(parameters['fr'], xray_energy), shape, dtype=complex128)
    fm1=refl.harm_sizes(refl.cast_to_array(parameters['fm1'], xray_energy), shape, dtype=complex128)
    fm2=refl.harm_sizes(refl.cast_to_array(parameters['fm2'], xray_energy), shape, dtype=complex128)

    phi=array(parameters['phi_m'], dtype=float64)*pi/180.0
    theta_m=array(parameters['theta_m'], dtype=float64)*pi/180.0
    # Create an offset of phi to handle negative mag values and negate theta_m
    phi=phi+where(mag<0, pi, 0)
    theta_m=where(mag<0, -theta_m, theta_m)
    mag=refl.harm_sizes(abs(mag), shape, dtype=float64)

    if theory in [0, instrument_string_choices['theory'][0]]:
        sl_c=dens*(f+resdens*fr)
        sl_m1=dens*resdens*resmag*mag*fm1
        sl_m2=dens*resdens*resmag*mag**2*fm2

        A=-lamda**2*re/pi*sl_c
        B=lamda**2*re/pi*sl_m1
        C=lamda**2*re/pi*sl_m2

        if not shape is None:
            M_shape=(shape[0], 3, shape[1])
        else:
            M_shape=None
        M=refl.harm_sizes(c_[cos(theta_m)*cos(phi), cos(theta_m)*sin(phi), sin(theta_m)], M_shape)

        g_0=sin(theta*pi/180.0)

        # Full theory
        if XBuffer.g_0 is not None:
            g0_ok=XBuffer.g_0.shape==g_0.shape
            if g0_ok:
                g0_ok=any(not_equal(XBuffer.g_0, g_0))
        else:
            g0_ok=False
        if True or (XBuffer.parameters!=parameters or XBuffer.coords!=instrument.getCoords()
                    or not g0_ok or XBuffer.wavelength!=lamda):
            # print g_0.shape, lamda.shape, A.shape, B.shape, C.shape, M.shape,
            W=xrmr.calc_refl_int_lay(g_0, lamda, A*0, A[::-1], B[::-1], C[::-1], M[::-1, ...]
                                     , d[::-1], sigma[::-1], sigma_l[::-1], sigma_u[::-1]
                                     , dd_l[::-1], dd_u[::-1], dmag_l[::-1], dmag_u[::-1])
            XBuffer.W=W
            XBuffer.parameters=parameters.copy()
            XBuffer.coords=instrument.getCoords()
            XBuffer.g_0=g_0.copy()
            XBuffer.wavelength=lamda
        else:
            # print 'Reusing W'
            W=XBuffer.W
        trans=ones(W.shape, dtype=complex128);
        trans[0, 1]=1.0J;
        trans[1, 1]=-1.0J;
        trans=trans/sqrt(2)
        # Wc = xrmr.dot2(trans, xrmr.dot2(W, xrmr.inv2(trans)))
        Wc=xrmr.dot2(trans, xrmr.dot2(W, conj(xrmr.inv2(trans))))
        # Different polarization channels:
        pol=instrument.getXpol()
        if pol==0 or pol==instrument_string_choices['xpol'][0]:
            # circ +
            R=abs(Wc[0, 0])**2+abs(Wc[1, 0])**2
        elif pol==1 or pol==instrument_string_choices['xpol'][1]:
            # circ -
            R=abs(Wc[1, 1])**2+abs(Wc[0, 1])**2
        elif pol==2 or pol==instrument_string_choices['xpol'][2]:
            # tot
            R=(abs(W[0, 0])**2+abs(W[1, 0])**2+abs(W[0, 1])**2+abs(W[1, 1])**2)/2
        elif pol==3 or pol==instrument_string_choices['xpol'][3]:
            # ass
            R=2*(W[0, 0]*W[0, 1].conj()+W[1, 0]*W[1, 1].conj()).imag/(
                        abs(W[0, 0])**2+abs(W[1, 0])**2+abs(W[0, 1])**2+abs(W[1, 1])**2)
        elif pol==4 or pol==instrument_string_choices['xpol'][4]:
            # sigma
            R=abs(W[0, 0])**2+abs(W[1, 0])**2
        elif pol==5 or pol==instrument_string_choices['xpol'][5]:
            # pi
            R=abs(W[0, 1])**2+abs(W[1, 1])**2
        elif pol==6 or pol==instrument_string_choices['xpol'][6]:
            # sigma-sigma
            R=abs(W[0, 0])**2
        elif pol==7 or pol==instrument_string_choices['xpol'][7]:
            # sigma-pi
            R=abs(W[1, 0])**2
        elif pol==8 or pol==instrument_string_choices['xpol'][8]:
            # pi-pi
            R=abs(W[1, 1])**2
        elif pol==9 or pol==instrument_string_choices['xpol'][9]:
            # pi-sigma
            R=abs(W[0, 1])**2
        else:
            raise ValueError('Variable pol has an unvalid value')
        # Override if we should return the complex amplitude (in this case a 2x2 matrix)
        if not return_amplitude:
            R=W

    elif theory in [1, instrument_string_choices['theory'][1]]:
        pol=instrument.getXpol()
        re=2.82e-13*1e2/1e-10
        Q=4*pi/lamda*sin(theta*pi/180)
        if pol==0 or pol==instrument_string_choices['xpol'][0]:
            # circ +
            pars=extract_anal_iso_pars(sample, instrument, theta, xray_energy, '+')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l=pars
            R=ables.ReflQ_mag(Q, lamda, n.T[:, ::-1], d[::-1], sigma_c[::-1], n_u.T[:, ::-1],
                              dd_u[::-1], sigma_u[::-1], n_l.T[:, ::-1], dd_l[::-1], sigma_l[::-1])
        elif pol==1 or pol==instrument_string_choices['xpol'][1]:
            # circ -
            pars=extract_anal_iso_pars(sample, instrument, theta, xray_energy, '-')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l=pars
            R=ables.ReflQ_mag(Q, lamda, n.T[:, ::-1], d[::-1], sigma_c[::-1], n_u.T[:, ::-1],
                              dd_u[::-1], sigma_u[::-1], n_l.T[:, ::-1], dd_l[::-1], sigma_l[::-1])
        elif pol==2 or pol==instrument_string_choices['xpol'][2]:
            # tot
            pars=extract_anal_iso_pars(sample, instrument, theta, xray_energy, '-')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l=pars
            Rm=ables.ReflQ_mag(Q, lamda, n.T[:, ::-1], d[::-1], sigma_c[::-1], n_u.T[:, ::-1],
                               dd_u[::-1], sigma_u[::-1], n_l.T[:, ::-1], dd_l[::-1], sigma_l[::-1])
            pars=extract_anal_iso_pars(sample, instrument, theta, xray_energy, '+')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l=pars
            Rp=ables.ReflQ_mag(Q, lamda, n.T[:, ::-1], d[::-1], sigma_c[::-1], n_u.T[:, ::-1],
                               dd_u[::-1], sigma_u[::-1], n_l.T[:, ::-1], dd_l[::-1], sigma_l[::-1])
            R=(Rp+Rm)/2.0
        elif pol==3 or pol==instrument_string_choices['xpol'][3]:
            # ass
            pars=extract_anal_iso_pars(sample, instrument, theta, xray_energy, '-')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l=pars
            Rm=ables.ReflQ_mag(Q, lamda, n.T[:, ::-1], d[::-1], sigma_c[::-1], n_u.T[:, ::-1],
                               dd_u[::-1], sigma_u[::-1], n_l.T[:, ::-1], dd_l[::-1], sigma_l[::-1])
            pars=extract_anal_iso_pars(sample, instrument, theta, xray_energy, '+')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l=pars
            Rp=ables.ReflQ_mag(Q, lamda, n.T[:, ::-1], d[::-1], sigma_c[::-1], n_u.T[:, ::-1],
                               dd_u[::-1], sigma_u[::-1], n_l.T[:, ::-1], dd_l[::-1], sigma_l[::-1])
            R=(Rp-Rm)/(Rp+Rm)
        else:
            raise ValueError('Variable pol has an unvalid value')

    elif theory in [2, instrument_string_choices['theory'][2]]:
        # neutron spin-pol calcs
        wl=instrument.getWavelength()
        Q=4*pi/wl*sin(theta*pi/180)
        pol=instrument.getNpol()
        if pol==instrument_string_choices['npol'][3] or pol==3:
            # Calculating the asymmetry
            pars=extract_anal_iso_pars(sample, instrument, theta, xray_energy, '++')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l=pars
            Rp=ables.ReflQ_mag(Q, lamda, n.T[:, ::-1], d[::-1], sigma_c[::-1], n_u.T[:, ::-1], dd_u[::-1],
                               sigma_u[::-1], n_l.T[:, ::-1], dd_l[::-1], sigma_l[::-1])
            pars=extract_anal_iso_pars(sample, instrument, theta, xray_energy, '--')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l=pars
            Rm=ables.ReflQ_mag(Q, lamda, n.T[:, ::-1], d[::-1], sigma_c[::-1], n_u.T[:, ::-1], dd_u[::-1],
                               sigma_u[::-1], n_l.T[:, ::-1], dd_l[::-1], sigma_l[::-1])
            R=(Rp-Rm)/(Rp+Rm)
        else:
            pars=extract_anal_iso_pars(sample, instrument, theta, xray_energy, instrument.getNpol())
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l=pars
            R=ables.ReflQ_mag(Q, lamda, n.T[:, ::-1], d[::-1], sigma_c[::-1], n_u.T[:, ::-1], dd_u[::-1],
                              sigma_u[::-1], n_l.T[:, ::-1], dd_l[::-1], sigma_l[::-1])
    elif theory in [3, instrument_string_choices['theory'][3]]:
        # neutron spin-flip calcs
        wl=instrument.getWavelength()
        Q=4*pi/wl*sin(theta*pi/180)
        # Check if we have calcluated the same sample previous:
        if NBuffer.TwoThetaQz is not None:
            Q_ok=NBuffer.TwoThetaQz.shape==Q.shape
            if Q_ok:
                Q_ok=any(not_equal(NBuffer.TwoThetaQz, Q))
        if NBuffer.parameters!=parameters or not Q_ok:
            # print 'Reloading buffer'
            b=array(parameters['b'], dtype=complex128)*1e-5
            abs_xs=array(parameters['xs_ai'], dtype=complex128)*1e-4**2
            # Bulk of the layers
            # sld = dens*(wl**2/2/pi*sqrt(fb**2 - (abs_xs/2.0/wl)**2) -
            #                   1.0J*abs_xs*wl/4/pi)

            V0=2*2*pi*dens*(b-1.0J*abs_xs/2.0/wl)
            Vmag=2*2*pi*muB_to_SL*mag*dens

            (Ruu, Rdd, Rud, Rdu)=neutron_refl.Refl_int_lay(Q, V0[::-1], Vmag[::-1], d[::-1], phi[::-1], sigma[::-1],
                                                           dmag_u[::-1], dd_u[::-1], phi[::-1], sigma_u[::-1],
                                                           dmag_l[::-1], dd_l[::-1], phi[::-1], sigma_l[::-1])
            NBuffer.Ruu=Ruu.copy();
            NBuffer.Rdd=Rdd.copy();
            NBuffer.Rud=Rud.copy()
            NBuffer.parameters=parameters.copy()
            NBuffer.TwoThetaQz=Q.copy()
        else:
            pass

        pol=instrument.getNpol()
        if pol==instrument_string_choices['npol'][0] or pol==0:
            R=NBuffer.Ruu
        # Polarization dd or --
        elif pol==instrument_string_choices['npol'][1] or pol==1:
            R=NBuffer.Rdd
        # Polarization ud or +-
        elif (pol==instrument_string_choices['npol'][2] or pol==2 or
              pol==instrument_string_choices['npol'][4] or pol==4):
            R=NBuffer.Rud
        # Polarisation is ass (asymmetry)
        elif pol==instrument_string_choices['npol'][3] or pol==3:
            R=(NBuffer.Ruu-NBuffer.Rdd)/(NBuffer.Ruu+NBuffer.Rdd+2*NBuffer.Rud)
        else:
            raise ValueError('The value of the polarization is WRONG.'
                             ' It should be ++(0), --(1) or +-(2)')
        # raise NotImplementedError('Neutron calcs not implemented')
    elif theory in [4, instrument_string_choices['theory'][4]]:
        if (instrument.getCoords()!=0 and
                instrument.getCoords()!=instrument_string_choices['coords'][0]):
            raise ValueError('Neutron TOF calculation only supports q as coordinate (x - axis)!')
        Q=TwoThetaQz
        wl=4*pi*sin(instrument.getIncang()*pi/180)/Q
        pars=extract_anal_iso_pars(sample, instrument, theta, xray_energy, instrument.getNpol(), Q=TwoThetaQz)
        n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l=pars
        R=ables.ReflQ_mag(TwoThetaQz, wl, n.T[:, ::-1], d[::-1], sigma_c[::-1], n_u.T[:, ::-1], dd_u[::-1],
                          sigma_u[::-1], n_l.T[:, ::-1], dd_l[::-1], sigma_l[::-1])
    elif theory in [5, instrument_string_choices['theory'][5]]:
        # x-ray isotropic (normal)
        # re = 2.82e-13*1e2/1e-10
        # Q = 4*pi/lamda*sin(theta*pi/180)
        pars=extract_anal_iso_pars(sample, instrument, theta, xray_energy, '+')
        n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l=pars
        R=Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, sigma_c)
        # R = ables.ReflQ_mag(Q, lamda, n.T[:,::-1], d[::-1], sigma_c[::-1], n_u.T[:,::-1],
        #                        dd_u[::-1], sigma_u[::-1], n_l.T[:,::-1], dd_l[::-1], sigma_l[::-1])

    else:
        raise ValueError('The given theory mode does not exist')

    return R

def slicing_reflectivity(sample, instrument, theta, TwoThetaQz, xray_energy, return_amplitude=True):
    lamda=AA_to_eV/xray_energy
    parameters=sample.resolveLayerParameters()

    (d, sl_c, sl_m1, sl_m2, M, chi, non_mag, mpy,
     sl_n, abs_n, mag_dens, mag_dens_x, mag_dens_y, z0)=compose_sld(sample, instrument, theta, xray_energy)
    re=2.8179402894e-5
    g_0=sin(theta*pi/180.0)
    theory=instrument.getTheory()
    # Full theory
    if not XBuffer.g_0 is None:
        g0_ok=XBuffer.g_0.shape==g_0.shape
        if g0_ok:
            g0_ok=allclose(XBuffer.g_0, g_0)
    else:
        g0_ok=False

    buffer_wl=array(XBuffer.wavelength)
    current_wl=array(lamda)
    if not XBuffer.wavelength is None:
        wl_ok=buffer_wl.shape==current_wl.shape
        if wl_ok:
            wl_ok=allclose(buffer_wl, lamda)
    else:
        wl_ok=False

    if theory in [0, instrument_string_choices['theory'][0]]:
        if (XBuffer.parameters!=parameters or XBuffer.coords!=instrument.getCoords() or
                not g0_ok or not wl_ok):
            # print 'Calculating W'
            chi=tuple([tuple([item[::-1] for item in row]) for row in chi])
            # print chi[0][0].shape
            d=d[::-1]
            non_mag=non_mag[::-1]
            mpy=mpy[::-1]
            W=xrmr.do_calc(g_0, lamda, chi, d, non_mag, mpy)
            XBuffer.W=W
            XBuffer.parameters=parameters.copy()
            XBuffer.coords=instrument.getCoords()
            XBuffer.g_0=g_0.copy()
            XBuffer.wavelength=lamda
        else:
            # print "Reusing W"
            W=XBuffer.W
        trans=ones(W.shape, dtype=complex128);
        trans[0, 1]=1.0J;
        trans[1, 1]=-1.0J;
        trans=trans/sqrt(2)
        Wc=xrmr.dot2(trans, xrmr.dot2(W, conj(xrmr.inv2(trans))))
        # Different polarization channels:
        pol=instrument.getXpol()
        if pol==0 or pol==instrument_string_choices['xpol'][0]:
            # circ +
            R=abs(Wc[0, 0])**2+abs(Wc[1, 0])**2
        elif pol==1 or pol==instrument_string_choices['xpol'][1]:
            # circ -
            R=abs(Wc[1, 1])**2+abs(Wc[0, 1])**2
        elif pol==2 or pol==instrument_string_choices['xpol'][2]:
            # tot
            R=(abs(W[0, 0])**2+abs(W[1, 0])**2+abs(W[0, 1])**2+abs(W[1, 1])**2)/2
        elif pol==3 or pol==instrument_string_choices['xpol'][3]:
            # ass
            R=2*(W[0, 0]*W[0, 1].conj()+W[1, 0]*W[1, 1].conj()).imag/(
                        abs(W[0, 0])**2+abs(W[1, 0])**2+abs(W[0, 1])**2+abs(W[1, 1])**2)
        elif pol==4 or pol==instrument_string_choices['xpol'][4]:
            # sigma
            R=abs(W[0, 0])**2+abs(W[1, 0])**2
        elif pol==5 or pol==instrument_string_choices['xpol'][5]:
            # pi
            R=abs(W[0, 1])**2+abs(W[1, 1])**2
        elif pol==6 or pol==instrument_string_choices['xpol'][6]:
            # sigma-sigma
            R=abs(W[0, 0])**2
        elif pol==7 or pol==instrument_string_choices['xpol'][7]:
            # sigma-pi
            R=abs(W[1, 0])**2
        elif pol==8 or pol==instrument_string_choices['xpol'][8]:
            # pi-pi
            R=abs(W[1, 1])**2
        elif pol==9 or pol==instrument_string_choices['xpol'][9]:
            # pi-sigma
            R=abs(W[0, 1])**2
        else:
            raise ValueError('Variable pol has an unvalid value')
        if not return_amplitude:
            R=W
    # Simplified theory
    elif theory in [1, instrument_string_choices['theory'][1]]:
        pol=instrument.getXpol()
        re=2.8179402894e-5
        c=1/(lamda**2*re/pi)
        sl_c=-chi[0][0]*c
        sl_m1=-1.0J*chi[2][1]*c
        if pol==0 or pol==instrument_string_choices['xpol'][0]:
            # circ +
            n=1-lamda**2*re/pi*(sl_c[:, newaxis]+sl_m1[:, newaxis]*cos(theta*pi/180))/2.0
            R=Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
        elif pol==1 or pol==instrument_string_choices['xpol'][1]:
            # circ -
            n=1-lamda**2*re/pi*(sl_c[:, newaxis]-sl_m1[:, newaxis]*cos(theta*pi/180))/2.0
            R=Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
        elif pol==2 or pol==instrument_string_choices['xpol'][2]:
            # tot
            n=1-lamda**2*re/pi*(sl_c[:, newaxis]-sl_m1[:, newaxis]*cos(theta*pi/180))/2.0
            Rm=Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
            # Hack to check kinematical approx
            n=1-lamda**2*re/pi*(sl_c[:, newaxis]+sl_m1[:, newaxis]*cos(theta*pi/180))/2.0
            Rp=Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
            # Hack to check kinematical approx
            R=(Rp+Rm)/2.0
        elif pol==3 or pol==instrument_string_choices['xpol'][3]:
            # ass
            n=1-lamda**2*re/pi*(sl_c[:, newaxis]-sl_m1[:, newaxis]*cos(theta*pi/180))/2.0
            Rm=Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
            # Hack to check kinematical approx
            n=1-lamda**2*re/pi*(sl_c[:, newaxis]+sl_m1[:, newaxis]*cos(theta*pi/180))/2.0
            Rp=Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
            # Hack to check kinematical approx
            R=(Rp-Rm)/(Rp+Rm)
        else:
            raise ValueError('Variable pol has an unvalid value')
    # Neutron spin pol calculations normal mode
    elif theory in [2, instrument_string_choices['theory'][2]]:
        lamda=instrument.getWavelength()
        sl_n=sl_n*1e-5
        abs_n=abs_n*1e-8
        sl_n=(lamda**2/2/pi*sl_n-1.0J*abs_n*lamda/4/pi)
        sl_nm=muB_to_SL*mag_dens*lamda**2/2/pi
        pol=instrument.getNpol()
        if pol in ['++', 'uu']:
            n=1.0-sl_n-sl_nm
            R=Paratt.Refl_nvary2(theta, lamda*ones(theta.shape),
                                 n[:, newaxis]*ones(theta.shape), d, zeros(d.shape))
        elif pol in ['--', 'dd']:
            n=1.0-sl_n+sl_nm
            R=Paratt.Refl_nvary2(theta, lamda*ones(theta.shape),
                                 n[:, newaxis]*ones(theta.shape), d, zeros(d.shape))
        elif pol=='ass':
            n=1.0-sl_n+sl_nm
            Rm=Paratt.Refl_nvary2(theta, lamda*ones(theta.shape),
                                  n[:, newaxis]*ones(theta.shape), d, zeros(d.shape))
            n=1.0-sl_n-sl_nm
            Rp=Paratt.Refl_nvary2(theta, lamda*ones(theta.shape),
                                  n[:, newaxis]*ones(theta.shape), d, zeros(d.shape))
            R=(Rp-Rm)/(Rp+Rm)

    # Neutron calculations spin-flip
    elif theory in [3, instrument_string_choices['theory'][3]]:
        # neutron spin-flip calcs
        lamda=instrument.getWavelength()
        sl_n=sl_n*1e-5
        abs_n=abs_n*1e-8
        Q=4*pi/lamda*sin(theta*pi/180)
        # Check if we have calcluated the same sample previous:
        if NBuffer.TwoThetaQz is not None:
            Q_ok=NBuffer.TwoThetaQz.shape==Q.shape
            if Q_ok:
                Q_ok=any(not_equal(NBuffer.TwoThetaQz, Q))
        if NBuffer.parameters!=parameters or not Q_ok or True:
            iprint('Reloading buffer')
            # Bulk of the layers
            # V0 = 2*2*pi*dens*(sqrt(b**2 - (abs_xs/2.0/wl)**2) -
            #                   1.0J*abs_xs/2.0/wl)
            # These rows are added to always have an ambient in the structure
            # large roughness messes up the spin-flip channel otherwise.
            sl_n=append(sl_n, sample.Ambient.dens*sample.Ambient.b*1e-5)
            abs_n=append(abs_n, sample.Ambient.dens*sample.Ambient.xs_ai*1e-4**2)
            mag_dens=append(mag_dens, 0.0)
            mag_dens_x=append(mag_dens_x, 0.0)
            mag_dens_y=append(mag_dens_y, 0.0)
            d=append(d, 0.0)

            V0=2*2*pi*(sl_n-1.0J*abs_n/2.0/lamda)
            mag=sqrt(mag_dens_x**2+mag_dens_y**2)
            Vmag=2*2*pi*muB_to_SL*mag
            phi_tmp=arccos(mag_dens_x/mag)
            phi=where(mag<1e-20, zeros_like(mag), phi_tmp)
            (Ruu, Rdd, Rud, Rdu)=neutron_refl.Refl(Q, V0[::1]+Vmag[::1], V0[::1]-Vmag[::1], d[::1], phi[::1])

            NBuffer.Ruu=Ruu.copy();
            NBuffer.Rdd=Rdd.copy();
            NBuffer.Rud=Rud.copy()
            NBuffer.parameters=parameters.copy()
            NBuffer.TwoThetaQz=Q.copy()
        else:
            pass

        pol=instrument.getNpol()
        if pol==instrument_string_choices['npol'][0] or pol==0:
            R=NBuffer.Ruu
        # Polarization dd or --
        elif pol==instrument_string_choices['npol'][1] or pol==1:
            R=NBuffer.Rdd
        # Polarization ud or +-
        elif (pol==instrument_string_choices['npol'][2] or pol==2 or
              pol==instrument_string_choices['npol'][4] or pol==4):
            R=NBuffer.Rud
        # Polarisation is ass (asymmetry)
        elif pol==instrument_string_choices['npol'][3] or pol==3:
            R=(NBuffer.Ruu-NBuffer.Rdd)/(NBuffer.Ruu+NBuffer.Rdd+2*NBuffer.Rud)
        else:
            raise ValueError('The value of the polarization is WRONG.'
                             ' It should be ++(0), --(1) or +-(2)')
        # raise NotImplementedError('Neutron calcs not implemented')

    elif theory in [4, instrument_string_choices['theory'][4]]:
        # neutron TOF calculations
        incang=instrument.getIncang()
        lamda=4*pi*sin(incang*pi/180)/TwoThetaQz
        sl_n=sl_n[:, newaxis]*1e-5
        abs_n=abs_n[:, newaxis]*1e-8
        sl_n=(lamda**2/2/pi*sl_n-1.0J*abs_n*lamda/4/pi)
        sl_nm=muB_to_SL*mag_dens[:, newaxis]*lamda**2/2/pi
        pol=instrument.getNpol()

        if pol in ['++', 'uu']:
            n=1.0-sl_n-sl_nm
            R=Paratt.Refl_nvary2(incang*ones(lamda.shape), lamda,
                                 n, d, zeros(d.shape))
        elif pol in ['--', 'dd']:
            n=1.0-sl_n+sl_nm
            R=Paratt.Refl_nvary2(incang*ones(lamda.shape), lamda,
                                 n, d, zeros(d.shape))
        elif pol=='ass':
            n=1.0-sl_n+sl_nm
            Rm=Paratt.Refl_nvary2(incang*ones(lamda.shape), lamda,
                                  n, d, zeros(d.shape))
            n=1.0-sl_n-sl_nm
            Rp=Paratt.Refl_nvary2(incang*ones(lamda.shape), lamda,
                                  n, d, zeros(d.shape))
            R=(Rp-Rm)/(Rp+Rm)
    # Isotropic x-rays (normal x-ray reflectivity no magnetism)
    elif theory in [5, instrument_string_choices['theory'][5]]:
        pol=instrument.getXpol()
        re=2.8179402894e-5
        c=1/(lamda**2*re/pi)
        sl_c=-chi[0][0]*c
        n=1-lamda**2*re/pi*sl_c[:, newaxis]/2.0*ones(theta.shape)
        R=Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
    else:
        raise ValueError('The given theory mode deos not exist')
    return R

def footprint_correction(instrument, theta):
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
    return foocor

def convolute_reflectivity(R, instrument, foocor, TwoThetaQz, weight):
    restype=instrument.getRestype()
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
    return R

SimulationFunctions={'Specular': Specular, 'OffSpecular': OffSpecular, 'EnergySpecular': EnergySpecular,
                     'EnergySpecularField': EnergySpecularField,
                     'SLD': SLD_calculations, 'SpecularElectricField': SpecularElectricField}

(Instrument, Layer, Stack, Sample)=refl.MakeClasses(InstrumentParameters, LayerParameters, StackParameters,
                                                    SampleParameters, SimulationFunctions, ModelID)

if __name__=='__main__':
    pass

# CODE for analytical reflectivity calcs for spin flip calcs with TOF
# if (instrument.getCoords() != 0 and
#             instrument.getCoords() != instrument_string_choices['coords'][0]):
#             raise ValueError('Neutron TOF calculation only supports q as coordinate (x - axis)!')
#         # neutron spin-flip calcs
#         Q = TwoThetaQz
#         wl = 4*pi*sin(instrument.getIncang()*pi/180)/Q
#         # Check if we have calcluated the same sample previous:
#         if NBuffer.parameters != parameters or not all(equal(NBuffer.TwoThetaQz, Q)):
#             #print 'Reloading buffer'
#             b = array(parameters['b'], dtype = complex64).real*1e-5
#             abs_xs = array(parameters['xs_ai'], dtype = complex64)*(1e-4)**2
#             # Bulk of the layers
#             V0 = 2*2*pi*dens*(sqrt(b**2 - (abs_xs/2.0/wl[:,newaxis])**2) -
#                                1.0J*abs_xs/2.0/wl[:,newaxis])
#             Vmag = 2*2*pi*2.645e-5*mag*dens*ones(wl.shape)[:,newaxis]
#
#             (Ruu,Rdd,Rud,Rdu) = neutron_refl.Refl_int_lay(Q, V0[:,::-1], Vmag[:,::-1], d[::-1], phi[::-1], sigma[::-1],
#                                                         dmag_u[::-1], dd_u[::-1], phi[::-1], sigma_u[::-1],
#                                                         dmag_l[::-1], dd_l[::-1], phi[::-1], sigma_l[::-1])
#             NBuffer.Ruu = Ruu; NBuffer.Rdd = Rdd; NBuffer.Rud = Rud
#             NBuffer.parameters = parameters.copy()
#             NBuffer.TwoThetaQz = Q.copy()
#         else:
#             pass
#
#         pol = instrument.getNpol()
#         if pol == instrument_string_choices['npol'][0] or pol == 0:
#             R = NBuffer.Ruu
#         # Polarization dd or --
#         elif pol == instrument_string_choices['npol'][1] or pol == 1:
#             R = NBuffer.Rdd
#         # Polarization ud or +-
#         elif pol == instrument_string_choices['npol'][2] or pol == 2:
#             R = NBuffer.Rud
#         else:
#             raise ValueError('The value of the polarization is WRONG.'
#                 ' It should be ++(0), --(1) or +-(2)')
#         #raise NotImplementedError('TOF Neutron calcs not implemented')
