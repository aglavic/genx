"""
Library for combined x-ray and neutrons simulations.
====================================================
The neutron simulations is capable of handling non-magnetic, magnetic
non-spin flip as well as neutron spin-flip reflectivity.

Classes
-------

"""

from dataclasses import dataclass, field, fields
from typing import List

import numpy as np

# import all special footprint functions
from .lib import footprint as footprint_module
from .lib import neutron_refl as MatrixNeutron
from .lib import paratt as Paratt
from .lib import refl_new as refl
from .lib import resolution as resolution_module
from .lib.base import AltStrEnum
from .lib.footprint import *
from .lib.instrument import *
from .lib.physical_constants import muB_to_SL, r_e
from .lib.resolution import *

# Preamble to define the parameters needed for the models outlined below:

ModelID = "SpecNX"

AA_to_eV = 12398.5
""" Conversion from Angstrom to eV E = AA_to_eV/lamda."""

q_limit = 1e-10
""" Minimum allowed q-value """

__xlabel__ = "q [Å$^{-1}$]"
__ylabel__ = "Instnsity [a.u.]"


class Probe(AltStrEnum):
    xray = "x-ray"
    neutron = ("neutron",)
    npol = "neutron pol"
    npolsf = "neutron pol spin flip"
    ntof = "neutron tof"
    ntofpol = "neutron pol tof"


class Coords(AltStrEnum):
    q = "q"
    tth = "2θ"
    alternate_tth = "tth"


class ResType(AltStrEnum):
    none = "no conv"
    fast_conv = "fast conv"
    fast_conv_var = "fast conv + varying res."
    fast_conv_rel = "fast conv + varying res. (dx/x)"
    full_conv_abs = "full conv and varying res."
    full_conv_rel = "full conv and varying res. (dx/x)"


class FootType(AltStrEnum):
    none = "no corr"
    gauss = "gauss beam"
    square = "square beam"


class Polarization(AltStrEnum):
    up_up = "uu"
    down_down = "dd"
    up_down = "ud"
    down_up = "du"
    asymmetry = "ass"

    alternate_up_up = "++"
    alternate_down_down = "--"
    alternate_up_down = "+-"
    alternate_down_up = "-+"


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
    ``magn_ang``
        The angle of the magnetic moment in degress. 0 degrees correspond to
        a moment collinear with the neutron spin.
    ``magn``
        The magnetic moment per formula unit (same formula unit as b and dens refer to)
    ``sigma``
        The root mean square roughness of the top interface of the layer in Angstroms.
    ``xs_ai``
       The sum of the absorption cross section and the incoherent scattering
       cross section in barns for neutrons
    """

    sigma: float = 0.0
    dens: float = 1.0
    d: float = 0.0
    f: complex = 1e-20j
    b: complex = 0j
    xs_ai: float = 0.0
    magn: float = 0.0
    magn_ang: float = 0.0

    Units = {
        "sigma": "AA",
        "dens": "at./AA",
        "d": "AA",
        "f": "el./at.",
        "b": "fm/at.",
        "xs_ai": "barn/at.",
        "magn": "mu_B/at.",
        "magn_ang": "deg.",
    }

    Groups = [("Standard", ["f", "dens", "d", "sigma"]), ("Neutron", ["b", "xs_ai", "magn", "magn_ang"])]


@dataclass
class LayerParameters:
    sigma: List[float]
    dens: List[float]
    d: List[float]
    f: List[complex]
    b: List[complex]
    xs_ai: List[float]
    magn: List[float]
    magn_ang: List[float]


@dataclass
class Stack(refl.StackBase):
    """
    A collection of Layer objects that can be repeated.

    ``Layers``
       A ``list`` consiting of ``Layers`` in the stack the first item is
       the layer closest to the bottom
    ``Repetitions``
       The number of repsetions of the stack
    """

    Layers: List[Layer] = field(default_factory=list)
    Repetitions: int = 1


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
    ``crop_sld``
       For samples with many layers this limits the number of layers that
       are shown on SLD graphs. Negative values just remove these layers
       and show a spike in SLD, instead, while postivie values exchange
       them by one layer of same thickness.
    """

    Stacks: List[Stack] = field(default_factory=list)
    Ambient: Layer = field(default_factory=Layer)
    Substrate: Layer = field(default_factory=Layer)
    crop_sld: int = 0
    _layer_parameter_class = LayerParameters


@dataclass
class Instrument(refl.ReflBase):
    """
    Specify parameters of the probe and reflectometry instrument.

    ``probe``
        Describes the radiation and measurments used, it is one of:
        'x-ray', 'neutron', 'neutron pol', 'neutron pol spin flip',
        'neutron tof', 'neutron pol tof'.
        The calculations for x-rays uses ``f`` for the scattering length for
        neutrons ``b`` for 'neutron pol', 'neutron pol spin flip' and 'neutron
        pol tof' alternatives the ``magn`` is used in the calculations. Note
        that the angle of magnetization ``magn_ang`` is only used in the spin
        flip model.
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
    """

    probe: Probe = "x-ray"
    wavelength: float = 1.54
    coords: Coords = "2θ"
    I0: float = 1.0
    res: float = 0.001
    restype: ResType = "no conv"
    respoints: int = 5
    resintrange: float = 2.0
    beamw: float = 0.01
    footype: FootType = "no corr"
    samplelen: float = 10.0
    incangle: float = 0.5
    pol: Polarization = "uu"
    Ibkg: float = 0.0
    tthoff: float = 0.0

    Units = {
        "probe": "",
        "wavelength": "Å",
        "coords": "",
        "I0": "arb.",
        "res": "[coord]",
        "restype": "",
        "respoints": "pts.",
        "resintrange": "[coord]",
        "beamw": "mm",
        "footype": "",
        "samplelen": "mm",
        "incangle": "°",
        "pol": "",
        "Ibkg": "arb.",
        "tthoff": "°",
        "2θ": "°",
        "tth": "°",
        "q": "Å$^-1$",
    }

    Groups = [
        ("General", ["wavelength", "coords", "I0", "Ibkg", "tthoff"]),
        ("Resolution", ["restype", "res", "respoints", "resintrange"]),
        ("Probe", ["probe", "pol", "incangle"]),
        (
            "Footprint",
            [
                "footype",
                "beamw",
                "samplelen",
            ],
        ),
    ]


# A buffer to save previous calculations for spin-flip calculations
class Buffer:
    Ruu = 0
    Rdd = 0
    Rdu = 0
    Rud = 0
    parameters = None
    TwoThetaQz = None


def footprintcorr(Q, instrument: Instrument):
    foocor = 1.0
    footype = instrument.footype
    beamw = instrument.beamw
    samlen = instrument.samplelen

    if instrument.probe in [Probe.ntof, Probe.ntofpol]:
        theta = instrument.incangle
        # if ai is an array, make sure it gets repeated for every resolution point
        if type(theta) is ndarray and instrument.restype in [ResType.full_conv_rel, ResType.full_conv_abs]:
            theta = (theta * ones(instrument.respoints)[:, newaxis]).flatten()

    else:
        # theta =  180./pi*arcsin(instrument.wavelength / 4.0 / pi * Q)
        theta = QtoTheta(instrument.wavelength, Q)
    if footype == FootType.gauss:
        foocor = GaussIntensity(theta, samlen / 2.0, samlen / 2.0, beamw)
    elif footype == FootType.square:
        foocor = SquareIntensity(theta, samlen, beamw)
    elif footype == FootType.none:
        pass
    elif isinstance(footype, footprint_module.Footprint):
        foocor = footype(theta, samlen)
    else:
        raise ValueError("The choice of footprint correction, footype," "is WRONG")

    return foocor


def resolutioncorr(R, TwoThetaQz, foocor, instrument: Instrument, weight):
    """Do the convolution of the reflectivity to account for resolution effects."""
    R = R[:] * foocor
    restype = instrument.restype
    if restype == ResType.fast_conv:
        R = ConvoluteFast(TwoThetaQz, R, instrument.res, range=instrument.resintrange)
    elif restype in [ResType.full_conv_rel, ResType.full_conv_abs] or isinstance(restype, resolution_module.Resolution):
        R = ConvoluteResolutionVector(TwoThetaQz, R, weight)
    elif restype == ResType.fast_conv_var:
        R = ConvoluteFastVar(TwoThetaQz, R, instrument.res, range=instrument.resintrange)
    elif restype == ResType.fast_conv_rel:
        R = ConvoluteFastVar(TwoThetaQz, R, instrument.res * TwoThetaQz, range=instrument.resintrange)
    elif restype == ResType.none:
        pass
    else:
        raise ValueError("The choice of resolution type, restype," "is WRONG")
    return R


def resolution_init(TwoThetaQz, instrument: Instrument):
    """Inits the dependet variable with regards to coordinates and resolution."""
    global __xlabel__
    __xlabel__ = "q [Å$^{-1}$]"

    restype = instrument.restype
    weight = 0
    if isinstance(restype, resolution_module.Resolution):
        (TwoThetaQz, weight) = restype(TwoThetaQz[:], instrument.respoints, instrument.resintrange)
    elif restype == ResType.full_conv_abs:
        res_function = GaussianResolution(sigma=instrument.res)
        (TwoThetaQz, weight) = res_function(TwoThetaQz[:], instrument.respoints, instrument.resintrange)
    elif restype == ResType.full_conv_rel:
        res_function = GaussianResolution(sigma=instrument.res * TwoThetaQz)
        (TwoThetaQz, weight) = res_function(TwoThetaQz[:], instrument.respoints, instrument.resintrange)
    # TTH values given as x
    if instrument.coords == Coords.tth:
        Q = TwoThetatoQ(instrument.wavelength, TwoThetaQz + instrument.tthoff)
        __xlabel__ = "2θ [°]"
    # Q vector given....
    elif instrument.coords in [Coords.q]:
        # if there is no tth offset, nothing to be done for Q
        if instrument.tthoff == 0:
            Q = TwoThetaQz
        # for tof the q-values are not angles but wavelenght, so tth-offset is a scaling factor
        elif instrument.probe in [Probe.ntof, Probe.ntofpol]:
            ai = instrument.incangle
            # if ai is an array, make sure it gets repeated for every resolution point
            if type(ai) is ndarray and restype in [ResType.full_conv_abs, ResType.full_conv_rel]:
                ai = (ai * ones(instrument.respoints)[:, newaxis]).flatten()
            Q = TwoThetaQz * (sin((ai + instrument.tthoff / 2.0) * pi / 180.0) / sin(ai * pi / 180.0))
        else:
            Q = TwoThetatoQ(
                instrument.wavelength, QtoTheta(instrument.wavelength, TwoThetaQz) * 2.0 + instrument.tthoff
            )
    else:
        raise ValueError("The value for coordinates, coords, is WRONG! should be q(0) or tth(1).")
    return Q, TwoThetaQz, weight


def neutron_sld(abs_xs, dens, fb, wl):
    return dens * (wl**2 / 2 / pi * fb - 1.0j * abs_xs * wl / 4 / pi)


def Specular(TwoThetaQz, sample: Sample, instrument: Instrument):
    """Simulate the specular signal from sample when probed with instrument

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """
    return specular_calcs(TwoThetaQz, sample, instrument, return_int=True)


def SpecularField(TwoThetaQz, sample, instrument):
    """Simulate the specular signal from sample when probed with instrument

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """
    return specular_calcs(TwoThetaQz, sample, instrument, return_int=False)


def specular_calcs(TwoThetaQz, sample: Sample, instrument: Instrument, return_int=True):
    """Simulate the specular signal from sample when probed with instrument

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

    parameters: LayerParameters = sample.resolveLayerParameters()

    dens = array(parameters.dens, dtype=float64)
    d = array(parameters.d, dtype=float64)
    magn = array(parameters.magn, dtype=float64)
    # Transform to radians
    magn_ang = array(parameters.magn_ang, dtype=float64) * pi / 180.0

    sigma = array(parameters.sigma, dtype=float64)

    if ptype == Probe.xray:
        # fb = array(parameters['f'], dtype = complex64)
        e = AA_to_eV / instrument.wavelength
        fb = refl.cast_to_array(parameters.f, e).astype(complex128)
        sld = dens * fb * instrument.wavelength**2 / 2 / pi
    else:
        fb = array(parameters.b, dtype=complex128) * 1e-5
        abs_xs = array(parameters.xs_ai, dtype=complex128) * 1e-4**2
        wl = instrument.wavelength
        # sld = dens*(wl**2/2/pi*sqrt(fb**2 - (abs_xs/2.0/wl)**2) -
        #                       1.0J*abs_xs*wl/4/pi)
        sld = neutron_sld(abs_xs, dens, fb, wl)
    # Ordinary Paratt X-rays
    if ptype == Probe.xray:
        R = Paratt.ReflQ(Q, instrument.wavelength, 1.0 - r_e * sld, d, sigma, return_int=return_int)
        # print 2.82e-5*sld
    # Ordinary Paratt Neutrons
    elif ptype == Probe.neutron:
        R = Paratt.ReflQ(Q, instrument.wavelength, 1.0 - sld, d, sigma, return_int=return_int)
    # Ordinary Paratt but with magnetization
    elif ptype == Probe.npol:
        msld = muB_to_SL * magn * dens * instrument.wavelength**2 / 2 / pi
        # Polarization uu or ++
        if pol == Polarization.up_up:
            R = Paratt.ReflQ(Q, instrument.wavelength, 1.0 - sld - msld, d, sigma, return_int=return_int)
        # Polarization dd or --
        elif pol == Polarization.down_down:
            R = Paratt.ReflQ(Q, instrument.wavelength, 1.0 - sld + msld, d, sigma, return_int=return_int)
        elif pol == Polarization.asymmetry:
            Rp = Paratt.ReflQ(Q, instrument.wavelength, 1.0 - sld - msld, d, sigma, return_int=return_int)
            Rm = Paratt.ReflQ(Q, instrument.wavelength, 1.0 - sld + msld, d, sigma, return_int=return_int)
            R = (Rp - Rm) / (Rp + Rm)

        else:
            raise ValueError("The value of the polarization is WRONG." " It should be uu(0) or dd(1)")
    # Spin flip
    elif ptype == Probe.npolsf:
        # Check if we have calcluated the same sample previous:
        if Buffer.TwoThetaQz is not None:
            Q_ok = Buffer.TwoThetaQz.shape == Q.shape
            if Q_ok:
                Q_ok = not (Buffer.TwoThetaQz != Q).any()
        else:
            Q_ok = False
        if Buffer.parameters != parameters or not Q_ok:
            msld = muB_to_SL * magn * dens * instrument.wavelength**2 / 2 / pi
            # renormalize SLDs if ambient layer is not vacuum
            if msld[-1] != 0.0 or sld[-1] != 0:
                msld -= msld[-1]
                sld -= sld[-1]
            sld_p = sld + msld
            sld_m = sld - msld
            Vp = (2 * pi / instrument.wavelength) ** 2 * (
                sld_p * (2.0 + sld_p)
            )  # (1-np**2) - better numerical accuracy
            Vm = (2 * pi / instrument.wavelength) ** 2 * (sld_m * (2.0 + sld_m))  # (1-nm**2)
            (Ruu, Rdd, Rud, Rdu) = MatrixNeutron.Refl(Q, Vp, Vm, d, magn_ang, sigma, return_int=return_int)
            Buffer.Ruu = Ruu
            Buffer.Rdd = Rdd
            Buffer.Rud = Rud
            Buffer.parameters = parameters
            Buffer.TwoThetaQz = Q.copy()
        else:
            pass
        if pol == Polarization.up_up:
            R = Buffer.Ruu
        elif pol == Polarization.down_down:
            R = Buffer.Rdd
        elif pol in [Polarization.up_down, Polarization.down_up]:
            R = Buffer.Rud
        # Calculating the asymmetry ass
        elif pol == Polarization.asymmetry:
            R = (Buffer.Ruu - Buffer.Rdd) / (Buffer.Ruu + Buffer.Rdd + 2 * Buffer.Rud)
        else:
            raise ValueError("The value of the polarization is WRONG." " It should be uu(0), dd(1) or ud(2)")

    # tof
    elif ptype == Probe.ntof:
        ai = instrument.incangle
        # if ai is an array, make sure it gets repeated for every resolution point
        if type(ai) is ndarray and restype in [ResType.full_conv_rel, ResType.full_conv_abs]:
            ai = (ai * ones(instrument.respoints)[:, newaxis]).flatten()
        else:
            ai = ai * ones(Q.shape)
        wl = 4 * pi * sin(ai * pi / 180) / Q
        sld = neutron_sld(abs_xs[:, newaxis], dens[:, newaxis], fb[:, newaxis], wl)
        R = Paratt.Refl_nvary2(ai, wl, 1.0 - sld, d, sigma, return_int=return_int)
    # tof spin polarized
    elif ptype == Probe.ntofpol:
        wl = 4 * pi * sin(instrument.incangle * pi / 180) / Q
        sld = neutron_sld(abs_xs[:, newaxis], dens[:, newaxis], fb[:, newaxis], wl)
        msld = (
            muB_to_SL
            * magn[:, newaxis]
            * dens[:, newaxis]
            * (4 * pi * sin(instrument.incangle * pi / 180) / Q) ** 2
            / 2
            / pi
        )
        # polarization uu or ++
        if pol == Polarization.up_up:
            R = Paratt.Refl_nvary2(
                instrument.incangle * ones(Q.shape),
                (4 * pi * sin(instrument.incangle * pi / 180) / Q),
                1.0 - sld - msld,
                d,
                sigma,
                return_int=return_int,
            )
        # polarization dd or --
        elif pol == Polarization.down_down:
            R = Paratt.Refl_nvary2(
                instrument.incangle * ones(Q.shape),
                (4 * pi * sin(instrument.incangle * pi / 180) / Q),
                1.0 - sld + msld,
                d,
                sigma,
                return_int=return_int,
            )
        # Calculating the asymmetry
        elif pol == Polarization.asymmetry:
            Rd = Paratt.Refl_nvary2(
                instrument.incangle * ones(Q.shape),
                (4 * pi * sin(instrument.incangle * pi / 180) / Q),
                1.0 - sld + msld,
                d,
                sigma,
                return_int=return_int,
            )
            Ru = Paratt.Refl_nvary2(
                instrument.incangle * ones(Q.shape),
                (4 * pi * sin(instrument.incangle * pi / 180) / Q),
                1.0 - sld - msld,
                d,
                sigma,
                return_int=return_int,
            )
            R = (Ru - Rd) / (Ru + Rd)

        else:
            raise ValueError("The value of the polarization is WRONG." " It should be uu(0) or dd(1) or ass")
    else:
        raise ValueError("The choice of probe is WRONG")
    if return_int:
        # FootprintCorrections
        foocor = footprintcorr(Q, instrument)
        # Resolution corrections
        R = resolutioncorr(R, TwoThetaQz, foocor, instrument, weight)

        return R * instrument.I0 + instrument.Ibkg
    else:
        return R


def EnergySpecular(Energy, TwoThetaQz, sample: Sample, instrument: Instrument):
    """Simulate the specular signal from sample when probed with instrument. Energy should be in eV.

    # BEGIN Parameters
    Energy data.x
    TwoThetaQz 3.0
    # END Parameters
    """
    # preamble to get it working with my class interface
    restype = instrument.restype
    # TODO: Fix so that resolution can be included.
    if restype != 0 and restype != ResType.none:
        raise ValueError("Only no resolution is allowed for energy scans.")

    wl = AA_to_eV / Energy
    global __xlabel__
    __xlabel__ = "E [eV]"

    # TTH values given as x
    if instrument.coords == Coords.tth:
        theta = TwoThetaQz / 2.0
    # Q vector given....
    elif instrument.coords == Coords.q:
        theta = arcsin(TwoThetaQz * wl / 4 / pi) * 180.0 / pi

    else:
        raise ValueError("The value for coordinates, coords, is WRONG!" "should be q(0) or tth(1).")
    Q = 4 * pi / wl * sin((2 * theta + instrument.tthoff) * pi / 360.0)

    ptype = instrument.probe

    parameters: LayerParameters = sample.resolveLayerParameters()
    if ptype == Probe.xray:
        fb = refl.cast_to_array(parameters.f, Energy).astype(complex128)
    else:
        fb = array(parameters.b, dtype=complex128) * 1e-5
        abs_xs = array(parameters.xs_ai, dtype=complex128) * 1e-4**2

    dens = array(parameters.dens, dtype=float64)
    d = array(parameters.d, dtype=float64)
    sigma = array(parameters.sigma, dtype=float64)

    if ptype == Probe.xray:
        sld = dens[:, newaxis] * fb * wl**2 / 2 / pi
    else:
        wl = instrument.wavelength
        sld = dens * (wl**2 / 2 / pi * sqrt(fb**2 - (abs_xs / 2.0 / wl) ** 2) - 1.0j * abs_xs * wl / 4 / pi)
    # Ordinary Paratt X-rays
    if ptype == Probe.xray:
        # R = Paratt.ReflQ(Q,instrument.wavelength,1.0-2.82e-5*sld,d,sigma)
        R = Paratt.Refl_nvary2(theta, wl, 1.0 - r_e * sld, d, sigma)
    else:
        raise ValueError("The choice of probe is WRONG")
    # TODO: Fix corrections
    # FootprintCorrections
    # foocor = footprintcorr(Q, instrument)
    # Resolution corrections
    # R = resolutioncorr(R, TwoThetaQz, foocor, instrument, weight)

    return R * instrument.I0 + instrument.Ibkg


def OffSpecular(TwoThetaQz, ThetaQx, sample: Sample, instrument: Instrument):
    """Function that simulates the off-specular signal (not implemented)

    # BEGIN Parameters
    TwoThetaQz 1.0
    ThetaQx data.x
    # END Parameters
    """
    raise NotImplementedError("Not implemented use model interdiff insteads")


def SLD_calculations(z, item, sample: Sample, inst: Instrument):
    """Calculates the scatteringlength density as at the positions z
    if item is None or "all" the function returns a dictonary of values.
    Otherwise, it returns the item as identified by its string.

    # BEGIN Parameters
    z data.x
    item 'Re'
    # END Parameters
    """
    parameters: LayerParameters = sample.resolveLayerParameters()
    if hasattr(sample, "crop_sld") and sample.crop_sld != 0:
        crop_top_bottom = abs(sample.crop_sld)
        inter = Layer()
        if sample.crop_sld > 0:
            inter.d = sum(parameters.d[crop_top_bottom:-crop_top_bottom])
        else:
            inter.d = 5.0
            inter.dens = 0.1
            inter.b = 12.0 + 0j
            inter.f = 100.0 + 0j
        if len(parameters.dens) > 2 * crop_top_bottom:
            for fi in fields(Layer):
                key = fi.name
                value = getattr(parameters, key)
                val_start = value[:crop_top_bottom]
                val_end = value[-crop_top_bottom:]
                setattr(parameters, key, val_start + [inter[key]] + val_end)
    dens = array(parameters.dens, dtype=float32)
    # f = array(parameters['f'], dtype = complex64)
    e = AA_to_eV / inst.wavelength
    f = refl.cast_to_array(parameters.f, e).astype(complex64)
    b = array(parameters.b, dtype=complex64) * 1e-5
    abs_xs = array(parameters.xs_ai, dtype=float32) * 1e-4**2
    wl = inst.wavelength
    ptype = inst.probe
    magnetic = False
    mag_sld = 0
    sld_unit = r"r_{e}/\AA^{3}"
    if ptype == Probe.xray:
        sld = dens * f
    elif ptype in [Probe.neutron, Probe.ntof]:
        sld = dens * (wl**2 / 2 / pi * b - 1.0j * abs_xs * wl / 4 / pi) / 1e-6 / (wl**2 / 2 / pi)
        sld_unit = r"10^{-6}\AA^{-2}"
    else:
        magnetic = True
        sld = dens * (wl**2 / 2 / pi * b - 1.0j * abs_xs * wl / 4 / pi) / 1e-6 / (wl**2 / 2 / pi)
        magn = array(parameters.magn, dtype=float64)
        # Transform to radians
        magn_ang = array(parameters.magn_ang, dtype=float64) * pi / 180.0
        mag_sld = 2.645 * magn * dens * 10.0
        mag_sld_x = mag_sld * cos(magn_ang)
        mag_sld_y = mag_sld * sin(magn_ang)
        sld_unit = r"10^{-6}\AA^{-2}"

    d = array(parameters.d, dtype=float64)
    d = d[1:-1]
    # Include one extra element - the zero pos (substrate/film interface)
    int_pos = cumsum(r_[0, d])
    sigma = array(parameters.sigma, dtype=float64)[:-1] + 1e-7
    if z is None:
        z = arange(-sigma[0] * 5, int_pos.max() + sigma[-1] * 5, 0.5)
    if not magnetic:
        rho = sum((sld[:-1] - sld[1:]) * (0.5 - 0.5 * erf((z[:, newaxis] - int_pos) / sqrt(2.0) / sigma)), 1) + sld[-1]
        dic = {"Re": real(rho), "Im": imag(rho), "z": z, "SLD unit": sld_unit}
    else:
        sld_p = sld + mag_sld
        sld_m = sld - mag_sld
        rho_p = (
            sum((sld_p[:-1] - sld_p[1:]) * (0.5 - 0.5 * erf((z[:, newaxis] - int_pos) / sqrt(2.0) / sigma)), 1)
            + sld_p[-1]
        )
        rho_m = (
            sum((sld_m[:-1] - sld_m[1:]) * (0.5 - 0.5 * erf((z[:, newaxis] - int_pos) / sqrt(2.0) / sigma)), 1)
            + sld_m[-1]
        )
        rho_nucl = (rho_p + rho_m) / 2.0
        if (magn_ang != 0.0).any():
            rho_mag_x = (
                sum(
                    (mag_sld_x[:-1] - mag_sld_x[1:]) * (0.5 - 0.5 * erf((z[:, newaxis] - int_pos) / sqrt(2.0) / sigma)),
                    1,
                )
                + mag_sld_x[-1]
            )
            rho_mag_y = (
                sum(
                    (mag_sld_y[:-1] - mag_sld_y[1:]) * (0.5 - 0.5 * erf((z[:, newaxis] - int_pos) / sqrt(2.0) / sigma)),
                    1,
                )
                + mag_sld_y[-1]
            )
            dic = {
                "Re non-mag": real(rho_nucl),
                "Im non-mag": imag(rho_nucl),
                "mag": real(rho_p - rho_m) / 2,
                "z": z,
                "mag_x": rho_mag_x,
                "mag_y": rho_mag_y,
                "SLD unit": sld_unit,
            }
        else:
            dic = {
                "Re non-mag": real(rho_nucl),
                "Im non-mag": imag(rho_nucl),
                "mag": real(rho_p - rho_m) / 2,
                "z": z,
                "SLD unit": sld_unit,
            }
    if item is None or item == "all":
        return dic
    else:
        try:
            return dic[item]
        except:
            raise ValueError("The chosen item, %s, does not exist" % item)


POL_CHANNELS = [Polarization.up_up, Polarization.up_down, Polarization.down_up, Polarization.down_down]


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
    if instrument.probe != Probe.npolsf:
        raise ValueError("Polarization corrected simulation requires probe to be 'neutron pol spin flip'")

    instrument.pol = "uu"
    uu = Specular(TwoThetaQz, sample, instrument)
    instrument.pol = "dd"
    dd = Specular(TwoThetaQz, sample, instrument)
    instrument.pol = "ud"
    ud = Specular(TwoThetaQz, sample, instrument)
    du = ud
    instrument.pol = inst_pol

    P = get_pol_matrix(p1, p2, F1, F2)
    Pline = P[POL_CHANNELS.index(instrument.pol)]
    I = Pline[:, newaxis] * np.vstack([uu, ud, du, dd])
    return I.sum(axis=0)


SimulationFunctions = {
    "Specular": Specular,
    "PolSpecular": PolSpecular,
    "SpecularField": SpecularField,
    "OffSpecular": OffSpecular,
    "SLD": SLD_calculations,
    "EnergySpecular": EnergySpecular,
}

Sample.setSimulationFunctions(SimulationFunctions)
