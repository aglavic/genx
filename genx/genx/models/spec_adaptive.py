'''
Library for combined x-ray and neutrons simulations with adaptive layer segmentation
====================================================================================
Library for specular neutron and x-ray reflectometry of more complex structures where elemental composition
and/or magnetism is better described separately than within one slap model. The model sums up a set of
*Elements* to calculate a total SLD profile and then uses adaptive layer segmentation to model it.

The actual modeling of the result structure is done with the same function as in spec_nx.

Classes
-------
'''

import numpy as np
from typing import List
from dataclasses import dataclass, field, fields

from .lib import paratt as Paratt
from .lib import neutron_refl as MatrixNeutron
from .lib.instrument import *
from .lib.base import AltStrEnum
from .lib import refl_new as refl
from .lib.physical_constants import r_e, muB_to_SL
# import all special footprint functions
from .lib import footprint as footprint_module
from .lib.footprint import *
from .lib import resolution as resolution_module
from .lib.resolution import *

from .spec_nx import (Probe, Coords, ResType, FootType, Polarization, Instrument as NXInstrument,
                      footprintcorr, resolutioncorr, resolution_init, neutron_sld, q_limit, AA_to_eV)

ModelID='SpecAdaptive'

__xlabel__ = "q [Å$^{-1}$]"
__ylabel__ = "Instnsity [a.u.]"

class RoughType(AltStrEnum):
    gauss = 'gauss'
    linear = 'linear'
    exp1 = 'exp-1'
    exp2 = 'exp-2'

@dataclass
class Layer(refl.ReflBase):
    """
    Representing a layer in the sample structur.

    ``b``
       The neutron scattering length per formula unit in fm (femtometer = 1e-15m)
    ``d``
        The thickness of the layer in AA (Angstroms = 1e-10m)
    ``f``
       The x-ray scattering length per formula unit in electrons. To be
       strict it is the number of Thompson scattering lengths for each
       formula unit.
    ``dens``
        The density of formula units in units per Angstroms. Note the units!
    ``sigma``
        The root-mean-square roughness of the top interface of the layer in Angstroms.
    ``magn``
        The magnetic moment per formula unit (same formula unit as b and dens refer to)
    ``magn_ang``
        The angle of the magnetic moment in degress. 0 degrees correspond to
        a moment collinear with the neutron spin.
    ``magn_void``
       If true this layer has no magnetization. In case of *sigma_mag* beging larger then 0 the additional
       roughness is only applied to the magnetic layer and inside this layer follows the chemical profile.
    ``sigma_mag``
       A different roughness parameter for the magnetization of the layer, 0 is ignored
    ``rough_type``
       Used model to get the SLD profile of the interface, *gauss* is an error function profile (gaussian roughness),
       *linear* is a linear profile, *exp-1* and *exp-2* are exponential decays from bottom or top side.
    ``xs_ai``
       The sum of the absorption cross-section and the incoherent scattering
       cross-section in barns for neutrons
    """
    sigma: float = 0.0
    dens: float = 1.0
    d: float = 0.0
    f: complex = 1e-20j
    b: complex = 0j
    xs_ai: float = 0.0

    rough_type: RoughType = 'gauss'

    magn: float = 0.0
    magn_ang: float = 0.0
    magn_void: bool = False
    sigma_mag: float = 0.0


    Units = {
        'sigma': 'AA', 'dens': 'at./AA', 'd': 'AA', 'f': 'el./at.',
        'b': 'fm/at.', 'xs_ai': 'barn/at.',
        'magn': 'mu_B/at.', 'magn_ang': 'deg.', 'sigma_mag': 'AA',
        }

    Groups = [('Standard', ['f', 'dens', 'd', 'sigma', 'sigma_mag', 'rough_type']),
              ('Neutron', ['b', 'xs_ai', 'magn', 'magn_ang', 'magn_void'])]

@dataclass
class LayerParameters:
    sigma: List[float]
    dens: List[float]
    d: List[float]
    f: List[complex]
    b: List[complex]
    xs_ai: List[float]
    rough_type: List[RoughType]
    magn: List[float]
    magn_ang: List[float]
    magn_void: List[bool]
    sigma_mag: List[float]

@dataclass
class Stack(refl.StackBase):
    """
    A collection of Layer objects that can be repeated.

    ``Layers``
       A ``list`` consiting of ``Layers`` in the stack the first item is
       the layer closest to the bottom
    ``Repetitions``
       The number of repsetions of the stack
    ``Element``
       The Element of the model that this stack belongs to. There has to be at least one stack with Element 0.
       For every *Element* the layers are stacked *on top* of the substrate separately and then all *Elements* are
       summed up to calculate the total SLD.
       The main use case for this is either to separate magnetic from nuclear structure
       (nuclear Element=0, magnetic Element=1) or two or more elemental contributions for element specific diffusion.
       For layers that have no contribution at a certain depth one can add a layer with 0 density as spacer.
    """
    Layers: List[Layer] = field(default_factory=list)
    Repetitions: int = 1
    Element: int = 0

@dataclass
class Sample(refl.SampleBase):
    """
    Describe global sample by listing ambient, substrate and layer parameters.

    ``Stacks``
       A ``list`` consiting of ``Stacks`` in the stacks the first item is
       the layer closest to the bottom
    ``Ambient``
       A ``Layer`` describing the Ambient (enviroment above the sample).
       Only the scattering lengths and density of the layer is used.
    ``Substrate``
       A ``Layer`` describing the substrate (enviroment below the sample).
       Only the scattering lengths, density and roughness of the layer is
       used.
    ``minimal_steps``
       The thickness of the minimal step between layers. Smaller values make the model more precise but slower.
       For data with larger q-range a smaller minimal_step is required. Try to start with 0.5-1 Å step size and
       increase the value until you see differences in the simulated data.
    ``max_diff_n``
       Maximum neutron SLD deviation to be allowed for layers to be combined in the adaptive procedure
    ``max_diff_x``
       Maximum x-ray SLD deviation to be allowed for layers to be combined in the adaptive procedure
    ``smoothen``
       Default is to not use any roughness for the segmentation. If True this will add roughnesses between all
       segments to make the curve more smooth.
    ``crop_sigma``
       For cases where roughness of an interface is in the order of layer thickness will limit the extent
       of roughness to the neighboring interfaces to avoid strange SLD profiles.
       Note: This means that the sigma parameter strictly is not an actual rms roughness anymore.
    """
    Stacks: List[Stack] = field(default_factory=list)
    Ambient: Layer = field(default_factory=Layer)
    Substrate: Layer = field(default_factory=Layer)

    minimal_steps: float = 0.5
    max_diff_n: float = 0.01
    max_diff_x: float = 0.01
    smoothen: bool = False
    crop_sigma: bool = False
    _layer_parameter_class = LayerParameters

class Zeeman(AltStrEnum):
    none = 'no corr'
    field = 'field only'
    pos_sf = 'SF q (+)'
    neg_sf = 'SF q (-)'

@dataclass
class Instrument(NXInstrument):
    zeeman: Zeeman = 'no corr'
    mag_field: float = 0.0

    Units = NXInstrument.Units | {'mag_field': "T"}
    Groups = NXInstrument.Groups + [("Zeeman correction", ['zeeman', 'mag_field'])]


# A buffer to save previous calculations for spin-flip calculations
class Buffer:
    Ruu = 0
    Rdd = 0
    Rdu = 0
    Rud = 0
    parameters = None
    TwoThetaQz = None


def Specular(TwoThetaQz, sample: Sample, instrument: Instrument):
    """ Simulate the specular signal from sample when probed with instrument

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """
    return specular_calcs(TwoThetaQz, sample, instrument, return_int=True)

def specular_calcs(TwoThetaQz, sample: Sample, instrument: Instrument, return_int=True):
    """ Simulate the specular signal from sample when probed with instrument

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """

    # preamble to get it working with my class interface
    restype = instrument.restype
    Q, TwoThetaQz, weight = resolution_init(TwoThetaQz, instrument)
    # often an issue with resolution etc. so just replace Q values < q_limit
    # if any(Q < q_limit):
    #    raise ValueError('The q vector has to be above %.1e, please verify all your x-axis points fulfill this criterion, including possible resolution smearing.'%q_limit)
    Q = maximum(Q, q_limit)

    ptype = instrument.probe
    pol = instrument.pol

    parameters:LayerParameters = sample.resolveLayerParameters()

    dens = array(parameters.dens, dtype=float64)
    d = array(parameters.d, dtype=float64)
    magn = array(parameters.magn, dtype=float64)
    # Transform to radians
    magn_ang = array(parameters.magn_ang, dtype=float64)*pi/180.0

    sigma = array(parameters.sigma, dtype=float64)

    if ptype==Probe.xray:
        # fb = array(parameters['f'], dtype = complex64)
        e = AA_to_eV/instrument.wavelength
        fb = refl.cast_to_array(parameters.f, e).astype(complex128)
        sld = dens*fb*instrument.wavelength**2/2/pi
    else:
        fb = array(parameters.b, dtype=complex128)*1e-5
        abs_xs = array(parameters.xs_ai, dtype=complex128)*1e-4**2
        wl = instrument.wavelength
        # sld = dens*(wl**2/2/pi*sqrt(fb**2 - (abs_xs/2.0/wl)**2) -
        #                       1.0J*abs_xs*wl/4/pi)
        sld = neutron_sld(abs_xs, dens, fb, wl)
    # Ordinary Paratt X-rays
    if ptype==Probe.xray:
        R = Paratt.ReflQ(Q, instrument.wavelength, 1.0-r_e*sld, d, sigma, return_int=return_int)
        # print 2.82e-5*sld
    # Ordinary Paratt Neutrons
    elif ptype==Probe.neutron:
        R = Paratt.ReflQ(Q, instrument.wavelength, 1.0-sld, d, sigma, return_int=return_int)
    # Ordinary Paratt but with magnetization
    elif ptype==Probe.npol:
        msld = muB_to_SL*magn*dens*instrument.wavelength**2/2/pi
        # Polarization uu or ++
        if pol==Polarization.up_up:
            R = Paratt.ReflQ(Q, instrument.wavelength, 1.0-sld-msld, d, sigma, return_int=return_int)
        # Polarization dd or --
        elif pol==Polarization.down_down:
            R = Paratt.ReflQ(Q, instrument.wavelength, 1.0-sld+msld, d, sigma, return_int=return_int)
        elif pol==Polarization.asymmetry:
            Rp = Paratt.ReflQ(Q, instrument.wavelength, 1.0-sld-msld, d, sigma, return_int=return_int)
            Rm = Paratt.ReflQ(Q, instrument.wavelength, 1.0-sld+msld, d, sigma, return_int=return_int)
            R = (Rp-Rm)/(Rp+Rm)

        else:
            raise ValueError('The value of the polarization is WRONG.'
                             ' It should be uu(0) or dd(1)')
    # Spin flip
    elif ptype==Probe.npolsf:
        # Check if we have calcluated the same sample previous:
        if Buffer.TwoThetaQz is not None:
            Q_ok = Buffer.TwoThetaQz.shape==Q.shape
            if Q_ok:
                Q_ok = not (Buffer.TwoThetaQz!=Q).any()
        else:
            Q_ok = False
        if Buffer.parameters!=parameters or not Q_ok:
            msld = muB_to_SL*magn*dens*instrument.wavelength**2/2/pi
            # renormalize SLDs if ambient layer is not vacuum
            if msld[-1]!=0. or sld[-1]!=0:
                msld -= msld[-1]
                sld -= sld[-1]
            sld_p = sld+msld
            sld_m = sld-msld
            Vp = (2*pi/instrument.wavelength)**2*(sld_p*(2.+sld_p))  # (1-np**2) - better numerical accuracy
            Vm = (2*pi/instrument.wavelength)**2*(sld_m*(2.+sld_m))  # (1-nm**2)
            (Ruu, Rdd, Rud, Rdu) = MatrixNeutron.Refl(Q, Vp, Vm, d, magn_ang, sigma, return_int=return_int)
            Buffer.Ruu = Ruu
            Buffer.Rdd = Rdd
            Buffer.Rud = Rud
            Buffer.parameters = parameters
            Buffer.TwoThetaQz = Q.copy()
        else:
            pass
        if pol==Polarization.up_up:
            R = Buffer.Ruu
        elif pol==Polarization.down_down:
            R = Buffer.Rdd
        elif pol in [Polarization.up_down, Polarization.down_up]:
            R = Buffer.Rud
        # Calculating the asymmetry ass
        elif pol==Polarization.asymmetry:
            R = (Buffer.Ruu-Buffer.Rdd)/(Buffer.Ruu+Buffer.Rdd+2*Buffer.Rud)
        else:
            raise ValueError('The value of the polarization is WRONG.'
                             ' It should be uu(0), dd(1) or ud(2)')

    # tof
    elif ptype==Probe.ntof:
        ai = instrument.incangle
        # if ai is an array, make sure it gets repeated for every resolution point
        if type(ai) is ndarray and restype in [ResType.full_conv_rel, ResType.full_conv_abs]:
            ai = (ai*ones(instrument.respoints)[:, newaxis]).flatten()
        else:
            ai = ai*ones(Q.shape)
        wl = 4*pi*sin(ai*pi/180)/Q
        sld = neutron_sld(abs_xs[:, newaxis], dens[:, newaxis], fb[:, newaxis], wl)
        R = Paratt.Refl_nvary2(ai, wl,
                               1.0-sld, d, sigma, return_int=return_int)
    # tof spin polarized
    elif ptype==Probe.npolsf:
        wl = 4*pi*sin(instrument.incangle*pi/180)/Q
        sld = neutron_sld(abs_xs[:, newaxis], dens[:, newaxis], fb[:, newaxis], wl)
        msld = muB_to_SL*magn[:, newaxis]*dens[:, newaxis]*(4*pi*sin(instrument.incangle*pi/180)/Q)**2/2/pi
        # polarization uu or ++
        if pol==Polarization.up_up:
            R = Paratt.Refl_nvary2(instrument.incangle*ones(Q.shape),
                                   (4*pi*sin(instrument.incangle*pi/180)/Q), 1.0-sld-msld, d, sigma,
                                   return_int=return_int)
        # polarization dd or --
        elif pol==Polarization.down_down:
            R = Paratt.Refl_nvary2(instrument.incangle*ones(Q.shape),
                                   (4*pi*sin(instrument.incangle*pi/180)/Q), 1.0-sld+msld, d, sigma,
                                   return_int=return_int)
        # Calculating the asymmetry
        elif pol==Polarization.asymmetry:
            Rd = Paratt.Refl_nvary2(instrument.incangle*ones(Q.shape),
                                    (4*pi*sin(instrument.incangle*pi/180)/Q),
                                    1.0-sld+msld, d, sigma, return_int=return_int)
            Ru = Paratt.Refl_nvary2(instrument.incangle*ones(Q.shape),
                                    (4*pi*sin(instrument.incangle*pi/180)/Q),
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

        return R*instrument.I0+instrument.Ibkg
    else:
        return R


def SLD_calculations(z, item, sample: Sample, inst: Instrument):
    ''' Calculates the scatteringlength density as at the positions z
    if item is None or "all" the function returns a dictonary of values.
    Otherwise it returns the item as identified by its string.

    # BEGIN Parameters
    z data.x
    item 'Re'
    # END Parameters
    '''
    parameters: LayerParameters = sample.resolveLayerParameters()
    if hasattr(sample, 'crop_sld') and sample.crop_sld!=0:
        crop_top_bottom = abs(sample.crop_sld)
        inter = Layer()
        if sample.crop_sld>0:
            inter.d = sum(parameters.d[crop_top_bottom:-crop_top_bottom])
        else:
            inter.d = 5.0
            inter.dens = 0.1
            inter.b = 12.0+0j
            inter.f = 100.0+0j
        if len(parameters.dens)>2*crop_top_bottom:
            for fi in fields(Layer):
                key = fi.name
                value = getattr(parameters, key)
                val_start = value[:crop_top_bottom]
                val_end = value[-crop_top_bottom:]
                setattr(parameters, key, val_start+[inter[key]]+val_end)
    dens = array(parameters.dens, dtype=float32)
    # f = array(parameters['f'], dtype = complex64)
    e = AA_to_eV/inst.wavelength
    f = refl.cast_to_array(parameters.f, e).astype(complex64)
    b = array(parameters.b, dtype=complex64)*1e-5
    abs_xs = array(parameters.xs_ai, dtype=float32)*1e-4**2
    wl = inst.wavelength
    ptype = inst.probe
    magnetic = False
    mag_sld = 0
    sld_unit = r'r_{e}/\AA^{3}'
    if ptype==Probe.xray:
        sld = dens*f
    elif ptype in [Probe.neutron, Probe.ntof]:
        sld = dens*(wl**2/2/pi*b-1.0J*abs_xs*wl/4/pi)/1e-6/(wl**2/2/pi)
        sld_unit = r'10^{-6}\AA^{-2}'
    else:
        magnetic = True
        sld = dens*(wl**2/2/pi*b-1.0J*abs_xs*wl/4/pi)/1e-6/(wl**2/2/pi)
        magn = array(parameters.magn, dtype=float64)
        # Transform to radians
        magn_ang = array(parameters.magn_ang, dtype=float64)*pi/180.0
        mag_sld = 2.645*magn*dens*10.
        mag_sld_x = mag_sld*cos(magn_ang)
        mag_sld_y = mag_sld*sin(magn_ang)
        sld_unit = r'10^{-6}\AA^{-2}'

    d = array(parameters.d, dtype=float64)
    d = d[1:-1]
    # Include one extra element - the zero pos (substrate/film interface)
    int_pos = cumsum(r_[0, d])
    sigma = array(parameters.sigma, dtype=float64)[:-1]+1e-7
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


POL_CHANNELS = [Polarization.up_up, Polarization.down_down, Polarization.up_down, Polarization.down_up]


def PolSpecular(TwoThetaQz, p1, p2, F1, F2, sample: Sample, instrument: Instrument):
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
    if instrument.probe!=Probe.npolsf:
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
    I = Pline[:, newaxis]*np.vstack([uu, ud, du, dd])
    return I.sum(axis=0)


SimulationFunctions = {
    'Specular': Specular,
    'PolSpecular': PolSpecular,
    'SLD': SLD_calculations,
    }

Sample.setSimulationFunctions(SimulationFunctions)
