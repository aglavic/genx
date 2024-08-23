"""
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

"""

from dataclasses import dataclass, field, fields
from typing import List

from .lib import ables as ables
from .lib import edm_slicing as edm
from .lib import neutron_refl as neutron_refl
from .lib import paratt as Paratt
from .lib import refl_base as refl
from .lib import xrmr
from .lib.base import AltStrEnum
from .lib.instrument import *
from .lib.physical_constants import AA_to_eV, muB_to_SL, r_e
from .lib.testing import ModelTestCase


# Preamble to define the parameters needed for the models outlined below:
ModelID = "MAGrefl"

mag_limit = 1e-8
mpy_limit = 1e-8
theta_limit = 1e-8

"""

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
"""


class XRayPol(AltStrEnum):
    total = "tot"
    circ_plus = "circ+"
    circ_minus = "circ-"

    sigma = "σ"
    pi = "π"
    sigma_sigma = "σ-σ"
    sigma_pi = "σ-π"
    pi_pi = "π-π"
    pi_sigma = "π-σ"

    asymmetry = "ass"

    alternate_sigma = "sigma"
    alternate_pi = "pi"
    alternate_sigma_sigma = "sigma-sigma"
    alternate_sigma_pi = "sigma-pi"
    alternate_pi_pi = "pi-pi"
    alternate_pi_sigma = "pi-sigma"


class NeutronPol(AltStrEnum):
    up_up = "uu"
    down_down = "dd"
    up_down = "ud"
    down_up = "du"
    asymmetry = "ass"

    alternate_up_up = "++"
    alternate_down_down = "--"
    alternate_up_down = "+-"
    alternate_down_up = "-+"


class ProbeTheory(AltStrEnum):
    xray_iso = "x-ray"
    xray_simple_aniso = "x-ray simpl. anis."
    xray_aniso = "x-ray anis."
    npol = "neutron pol"
    ntofpol = "neutron pol tof"
    npolsf = "neutron pol spin flip"


class Coords(AltStrEnum):
    q = "q"
    tth = "2θ"
    alternate_tth = "tth"


class ResType(AltStrEnum):
    none = "no conv"
    fast_conv = "fast conv"
    fast_conv_var = "fast conv + varying res."
    full_conv_var = "full conv and varying res."


class FootType(AltStrEnum):
    none = "no corr"
    gauss = "gauss beam"
    square = "square beam"


@dataclass
class Layer(refl.ReflBase):
    """
    Representing a layer in the sample structur.

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
    ``resdens``
       Relative fraction of resonant species in the formula unit. See
       ``fr`` for details.
    ``resmag``
       The relative amount of magnetic resonant atoms the total resonant
       magnetic atoms. The total magnetic scattering length is calculated as
       (for the circular dichroic term) ``fm1*resmag*mag*resdens*dens``
    ``dd_u``
       The width of the upper interface layer in Angstroms.
    ``dmag_u``
       The relative increase of the magnetic moment in the upper interface layer.
       Total magnetic moment is ``mag*(1 + dmag_u)``.
    ``sigma_mu``
       The roughness of the upper magnetic interface.
    ``dd_l``
       The width of the lower interface in Angstroms.
    ``dmag_l``
       As ``dmag_u`` but for the lower interface layer.
    ``sigma_ml``
       The roughness of the lower magnetic interface.
    ``b``
       The neutron scattering length in fm.
    ``xs_ai``
       The sum of the absorption cross section and the incoherent scattering
       cross section in barns per formula unit for the neutrons
    ``magn``
       The magnetic moment per formula unit. The magnetic density is
       ``mag*dens``.
    ``magn_ang``
       The in-plane angle of the magnetic moment of the layer relative the
       projected incident beam for x-rays and relative the polarization axis
       for neutrons.
    ``magn_theta``
       The out-of-plane angle of the magnetic moment. ``magn_theta = 0``
       corresponds to an in-plane magnetic moment and ``magn_theta = 90``
       corresponds to an out-of-plane magnetic moment.
    """

    d: float = 0.0
    dens: float = 1.0
    sigma: float = 0.0

    f: complex = 1e-20j
    fr: complex = 0j
    fm1: complex = 0j
    fm2: complex = 0j
    resdens: float = 1.0
    resmag: float = 1.0

    dd_l: float = 0.0
    dd_u: float = 0.0
    dmag_l: float = 0.0
    dmag_u: float = 0.0
    sigma_ml: float = 0.0
    sigma_mu: float = 0.0

    b: complex = 0j
    xs_ai: float = 0.0

    magn: float = 0.0
    magn_ang: float = 0.0
    magn_theta: float = 0.0

    Units = {
        "sigma": "AA",
        "dens": "at./AA",
        "d": "AA",
        "f": "el./at.",
        "b": "fm/at.",
        "xs_ai": "barn/at.",
        "magn": "mu_B/at.",
        "magn_ang": "deg.",
        "fr": "el.",
        "fm1": "el./mu_B",
        "fm2": "el./mu_B^2",
        "phi_m": "deg.",
        "theta_m": "deg.",
        "resdens": "rel.",
        "resmag": "rel.",
        "sigma_c": "AA",
        "sigma_ml": "AA",
        "sigma_mu": "AA",
        "mag": "mu_B",
        "dmag_l": "rel.",
        "dmag_u": "rel.",
        "dd_l": "AA",
        "dd_u": "AA",
    }

    Groups = [
        ("General", ["d", "dens", "sigma"]),
        ("Neutron", ["b", "xs_ai"]),
        ("X-Ray", ["f", "fr", "fm1", "fm2", "resdens", "resmag"]),
        ("Magnetic", ["magn", "magn_ang", "magn_theta"]),
        ("Magn. Interfaces", ["dd_u", "dmag_u", "sigma_mu", "dd_l", "dmag_l", "sigma_ml"]),
    ]


@dataclass
class LayerParameters:
    d: List[float]
    dens: List[float]
    sigma: List[float]

    magn: List[float]
    magn_ang: List[float]
    magn_theta: List[float]

    f: List[complex]
    fr: List[complex]
    fm1: List[complex]
    fm2: List[complex]
    resdens: List[float]
    resmag: List[float]

    b: List[complex]
    xs_ai: List[float]
    magn: List[float]
    magn_ang: List[float]

    dd_l: List[float]
    dd_u: List[float]
    dmag_l: List[float]
    dmag_u: List[float]
    sigma_ml: List[float]
    sigma_mu: List[float]


@dataclass
class Instrument(refl.ReflBase):
    """
    Specify parameters of the probe and reflectometry instrument.

    ``probe``
        Defines the theory (model) that should calcualte the reflectivity.
        Should be one of: 'x-ray anis.', 'x-ray simpl. anis.', 'x-ray',
        'neutron pol', 'neutron pol tof' or 'neutron pol spin flip'.
        Neutron models use either Parratt's formaism (non spin-flip models) or
        optical matrix method.
    ``wavelength``
        The wavelength of the radiation given in AA (Angstroms)
    ``I0``
        The incident intensity (a scaling factor)
    ``Ibkg``
        The background intensity. Added as a constant value to the calculated
        reflectivity
    ``xpol``
        The polarization state of the x-ray beam. Should be one of:
        'circ+','circ-','tot', 'ass', 'sigma', 'pi', 'sigma-sigma', 'sigma-pi',
        'pi-pi' or 'pi-sigma'
    ``npol``
        The neutron polarization state. Should be '++', '--' or '+-','-+' for
        spin flip.
    ``coords``
        The coordinates of the data given to the SimSpecular function. The
        available alternatives are: 'q' or '2θ'. Alternatively the numbers 0 (q)
        or 1 (tth) can be used.
    ``tthoff``
        Linear offset to the scattering angle calibration
    ``incangle``
        The incident angle of the neutrons, only valid in tof mode
    ``restype``
        Describes the rype of the resolution calculated. One of the
        alterantives: 'no conv', 'fast conv', 'full conv and varying res.',
        'fast conv + varying res.', 'full conv and varying res. (dx/x)', 'fast
        conv + varying res. (dx/x)'. The respective numbers 0-3 also works. Note
        that fast convolution only alllows a single value into res wheras the
        other can also take an array with the same length as the x-data (varying
        resolution)
    ``res``
        The resolution of the instrument given in the coordinates of ``coords``.
        This assumes a gaussian resolution function and ``res`` is the standard
        deviation of that gaussian. If ``restype`` has (dx/x) in its name the
        gaussian standard deviation is given by res*x where x is either in tth
        or q.
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
    """

    probe: ProbeTheory = "x-ray"
    wavelength: float = 1.54
    I0: float = 1.0
    Ibkg: float = 0.0
    xpol: XRayPol = "tot"
    npol: NeutronPol = "uu"
    coords: Coords = "2θ"
    incangle: float = 0.5
    restype: ResType = "no conv"
    res: float = 0.001
    respoints: int = 5
    resintrange: float = 2.0
    footype: FootType = "no corr"
    beamw: float = 0.01
    samplelen: float = 10.0

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
        "xpol": "",
        "npol": "",
        "Ibkg": "arb.",
        "2θ": "°",
        "tth": "°",
        "q": "Å$^-1$",
    }

    Groups = [
        ("Radiation", ["probe", "wavelength", "I0", "Ibkg", "xpol", "npol"]),
        ("X-Resolution", ["restype", "res", "respoints", "resintrange"]),
        ("X-Coordinates", ["coords", "incangle"]),
        ("Footprint", ["footype", "beamw", "samplelen"]),
    ]


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
    ``slicing``
       A flag that signals if the composition profile should be sliced up.
    ``slice_depth``
       The depth of the slices in the calculation of the sliced scattering
       length density profile.
    ``sld_mult``
       A multiplication factor for a buffer that takes the roughness into
       account.
    ``sld_buffer``
       A buffer for the slicing calculations (to assure convergence in the
       sld profile.
    ``sld_delta``
       An extra buffer - needed at all?
    ``compress``
       A flag that signals if the sliced composition profile should be
       compressed.
    ``dsld_max``
       The maximum allowed step in the scattering length density for x-rays
       (diagonal terms)
    ``dsld_offdiag_max``
       The maximum allowed step in the scattering length density for the
       offdiagonal terms of the scattering length (magnetic part)
    ``dmag_max``
       The maximum allowed step (in compression) for the magnetization.
       Primarily intended to limit the steps in the magnetic profile for
       neutrons.
    ``dsld_n_max``
       The maximum allowed step (in compression) for the neutron scattering
       length.
    ``dabs_n_max``
       The maximum allowed step (in compression) for the neutron absorption
       (in units of barn/AA^3)
    """

    Stacks: List[Stack] = field(default_factory=list)
    Ambient: Layer = field(default_factory=Layer)
    Substrate: Layer = field(default_factory=Layer)

    slicing: bool = False
    slice_depth: float = 1.0
    sld_mult: float = 4.0
    sld_buffer: float = 20.0
    sld_delta: float = 5.0

    compress: bool = True
    dsld_max: float = 0.1
    dsld_offdiag_max: float = 0.1
    dmag_max: float = 0.01
    dsld_n_max: float = 0.01
    dabs_n_max: float = 0.01

    Groups = [
        ("Slicing", ["slicing", "slice_depth", "sld_mult", "sld_buffer", "sld_delta"]),
        ("Compression", ["compress", "dsld_max", "dsld_offdiag_max", "dmag_max", "dsld_n_max", "dabs_n_max"]),
    ]
    Units = {
        "slice_depth": "AA",
    }
    _layer_parameter_class = LayerParameters


# A buffer to save previous calculations for XRMR calculations
class XBuffer:
    W = None
    parameters = None
    g_0 = None
    coords = None
    wavelength = None


# A buffer to save previous calculations for spin-flip calculations
class NBuffer:
    Ruu = 0
    Rdd = 0
    Rdu = 0
    Rud = 0
    parameters = None
    TwoThetaQz = None


def correct_reflectivity(R, TwoThetaQz, instrument: Instrument, theta, weight):
    pol = instrument.xpol
    theory = instrument.probe
    if not (pol == XRayPol.asymmetry and (theory in [ProbeTheory.xray_aniso, ProbeTheory.xray_simple_aniso])):
        # FootprintCorrections
        foocor = footprint_correction(instrument, theta)
        R = convolute_reflectivity(R, instrument, foocor, TwoThetaQz, weight)
        R = R * instrument.I0 + instrument.Ibkg
    else:
        foocor = footprint_correction(instrument, theta) * 0 + 1.0
        R = convolute_reflectivity(R, instrument, foocor, TwoThetaQz, weight)
    return R


def Specular(TwoThetaQz, sample: Sample, instrument: Instrument):
    """Simulate the specular signal from sample when proped with instrument

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """
    global __xlabel__
    __xlabel__ = "q [Å$^{-1}$]"

    # preamble to get it working with my class interface
    restype = instrument.restype
    xray_energy = AA_to_eV / instrument.wavelength
    weight = None
    if restype == ResType.full_conv_var:
        (TwoThetaQz, weight) = ResolutionVector(
            TwoThetaQz[:], instrument.res, instrument.respoints, range=instrument.resintrange
        )
    if instrument.coords == Coords.tth:
        theta = TwoThetaQz / 2
        __xlabel__ = "2θ [°]"
    else:
        theta = arcsin(TwoThetaQz / 4 / pi * instrument.wavelength) * 180.0 / pi
    theta = maximum(theta, theta_limit)

    R = reflectivity_xmag(sample, instrument, theta, TwoThetaQz, xray_energy)

    R = correct_reflectivity(R, TwoThetaQz, instrument, theta, weight)

    return R


def SpecularElectricField(TwoThetaQz, sample: Sample, instrument: Instrument):
    """Simulate the specular signal from sample when probed with instrument.
    Returns the wave field (complex number) of the reflected wave.
    No resolution is taken into account.

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """
    global __xlabel__
    __xlabel__ = "q [Å$^{-1}$]"

    xray_energy = AA_to_eV / instrument.wavelength
    if instrument.coords == Coords.tth:
        theta = TwoThetaQz / 2
        __xlabel__ = "2θ [°]"
    else:
        theta = arcsin(TwoThetaQz / 4 / pi * instrument.wavelength) * 180.0 / pi
    if any(theta < theta_limit):
        raise ValueError("The incident angle has to be above %.1e" % theta_limit)

    R = reflectivity_xmag(sample, instrument, theta, TwoThetaQz, xray_energy, return_amplitude=False)

    return R


def EnergySpecular(Energy, TwoThetaQz, sample: Sample, instrument: Instrument):
    """Simulate the specular signal from sample when probed with instrument. Energy should be in eV.

    # BEGIN Parameters
    Energy data.x
    TwoThetaQz 3.0
    # END Parameters
    """
    global __xlabel__
    __xlabel__ = "E [eV]"

    restype = instrument.restype
    # TODO: Fix so that resolution can be included.
    if restype != ResType.none:
        raise ValueError("Only no resolution is allowed for energy scans.")

    wl = AA_to_eV / Energy
    if isscalar(TwoThetaQz):
        TwoThetaQz = TwoThetaQz * ones(Energy.shape)
    if instrument.coords == Coords.tth:
        theta = TwoThetaQz / 2.0
    else:
        theta = arcsin(TwoThetaQz * wl / 4 / pi) * 180.0 / pi

    theta = maximum(theta, theta_limit)

    R = reflectivity_xmag(sample, instrument, theta, TwoThetaQz, Energy)

    # TODO: Fix Corrections
    # R = correct_reflectivity(R, TwoThetaQz, instrument, theta, weight)
    R = R * instrument.I0 + instrument.Ibkg

    return R


def EnergySpecularField(Energy, TwoThetaQz, sample: Sample, instrument: Instrument):
    """Simulate the specular signal from sample when probed with instrument. Energy should be in eV.
    Returns the wave field (complex number) of the reflected wave.
    No resolution is taken into account.

    # BEGIN Parameters
    Energy data.x
    TwoThetaQz 3.0
    # END Parameters
    """
    global __xlabel__
    __xlabel__ = "E [eV]"

    restype = instrument.restype
    # TODO: Fix so that resolution can be included.
    if restype != ResType.none:
        raise ValueError("Only no resolution is allowed for energy scans.")

    wl = AA_to_eV / Energy
    if isscalar(TwoThetaQz):
        TwoThetaQz = TwoThetaQz * ones(Energy.shape)
    # TTH values given as x
    if instrument.coords == Coords.tth:
        theta = TwoThetaQz / 2.0
    # Q vector given....
    else:
        theta = arcsin(TwoThetaQz * wl / 4 / pi) * 180.0 / pi
    theta = maximum(theta, theta_limit)

    R = reflectivity_xmag(sample, instrument, theta, TwoThetaQz, Energy, return_amplitude=False)

    # TODO: Fix Corrections
    # R = correct_reflectivity(R, TwoThetaQz, instrument, theta, weight)
    return R


def OffSpecular(TwoThetaQz, ThetaQx, sample: Sample, instrument: Instrument):
    """Function that simulates the off-specular signal (not implemented)

    # BEGIN Parameters
    TwoThetaQz 1.0
    ThetaQx data.x
    # END Parameters
    """
    raise NotImplementedError("Off specular calculations are not implemented for magnetic x-ray reflectivity")


def SLD_calculations(z, item, sample: Sample, inst: Instrument):
    """Calculates the scatteringlength density as at the positions z
    if item is None or "all" the function returns a dictonary of values.
    Otherwise it returns the item as identified by its string.

    # BEGIN Parameters
    z data.x
    item "Re sld_c"
    # END Parameters
    """
    use_slicing = sample.slicing
    if use_slicing:
        return compose_sld_anal(z, sample, inst)
    lamda = inst.wavelength
    theory = inst.probe
    xray_energy = AA_to_eV / inst.wavelength
    (d, sl_c, sl_m1, sl_m2, M, chi, non_mag, mpy, sl_n, abs_n, mag_dens, mag_dens_x, mag_dens_y, z0) = compose_sld(
        sample,
        inst,
        array(
            [
                0.0,
            ]
        ),
        xray_energy,
    )
    z = zeros(len(d) * 2)
    z[::2] = cumsum(r_[0, d[:-1]])
    z[1::2] = cumsum(r_[d])
    z += z0

    dic = {"z": z}

    if theory == ProbeTheory.xray_aniso:
        new_size = len(d) * 2
        sl_cp = zeros(new_size, dtype=complex128)
        sl_cp[::2] = sl_c
        sl_cp[1::2] = sl_c
        sl_m1p = zeros(new_size, dtype=complex128)
        sl_m1p[::2] = sl_m1
        sl_m1p[1::2] = sl_m1
        sl_m2p = zeros(new_size, dtype=complex128)
        sl_m2p[::2] = sl_m2
        sl_m2p[1::2] = sl_m2

        def interleave(a):
            new_a = zeros(len(a) * 2, dtype=complex128)
            new_a[::2] = a
            new_a[1::2] = a
            return new_a

        chi = [[interleave(c) for c in ch] for ch in chi]

        # return {'real sld_c': sl_cp.real, 'imag sld_c': sl_cp.imag,
        #        'real sld_m1': sl_m1p.real, 'imag sld_m1': sl_m1p.imag,
        #        'real sld_m2': sl_m2p.real, 'imag sld_m2': sl_m2p.imag,
        #        'z':z}
        c = 1 / (lamda**2 * r_e / pi)
        dic = {
            "Re sl_xx": chi[0][0].real * c,
            "Re sl_xy": chi[0][1].real * c,
            "Re sl_xz": chi[0][2].real * c,
            "Re sl_yy": chi[1][1].real * c,
            "Re sl_yz": chi[1][2].real * c,
            "Re sl_zz": chi[2][2].real * c,
            "Im sl_xx": chi[0][0].imag * c,
            "Im sl_xy": chi[0][1].imag * c,
            "Im sl_xz": chi[0][2].imag * c,
            "Im sl_yy": chi[1][1].imag * c,
            "Im sl_yz": chi[1][2].imag * c,
            "Im sl_zz": chi[2][2].imag * c,
            "z": z,
            "SLD unit": "r_e/\\AA^{3}",
        }
    else:
        new_size = len(d) * 2

        def parray(ar):
            tmp = zeros(new_size, dtype=complex128)
            tmp[::2] = ar
            tmp[1::2] = ar
            return tmp

        sl_cp = parray(sl_c)
        sl_m1p = parray(sl_m1)
        sl_np = parray(sl_n)
        mag_densp = parray(mag_dens)
        mag_dens_xp = parray(mag_dens_x)
        mag_dens_yp = parray(mag_dens_y)

        abs_np = parray(abs_n)
        if theory == ProbeTheory.xray_simple_aniso:
            dic = {
                "Re sld_c": sl_cp.real,
                "Im sld_c": sl_cp.imag,
                "Re sld_m": sl_m1p.real,
                "Im sld_m": sl_m1p.imag,
                "mag_dens": mag_densp,
                "z": z,
                "SLD unit": "r_{e}/\\AA^{3},\\,\\mu_{B}/\\AA^{3}",
            }
        elif theory in [ProbeTheory.npol, ProbeTheory.ntofpol]:
            dic = {
                "sld_n": sl_np,
                "abs_n": abs_np,
                "mag_dens": mag_densp,
                "z": z,
                "SLD unit": "fm/\\AA^{3}, b/\\AA^{3},\\,\\mu_{B}/\\AA^{3}",
            }
        elif theory == ProbeTheory.npolsf:
            dic = {
                "sld_n": sl_np,
                "abs_n": abs_np,
                "mag_dens": mag_densp,
                "mag_dens_x": mag_dens_xp,
                "mag_dens_y": mag_dens_yp,
                "z": z,
                "SLD unit": "fm/\\AA^{3}, b/\\AA^{3},\\,\\mu_{B}/\\AA^{3}",
            }
        elif theory == ProbeTheory.xray_iso:
            dic = {"Re sld_c": sl_cp.real, "Im sld_c": sl_cp.imag, "z": z, "SLD unit": "r_{e}/\\AA^{3}"}

    if item is None or item == "all":
        return dic
    else:
        try:
            return dic[item]
        except:
            raise ValueError("The chosen item, %s, does not exist" % item)


def neturon_sld(abs_xs, b, dens, wl):
    return dens * (wl**2 / 2 / pi * b - 1.0j * abs_xs * wl / 4 / pi) / 1e-5 / (wl**2 / 2 / pi)


def compose_sld_anal(z, sample: Sample, instrument: Instrument):
    """Compose a analytical profile funciton"""

    def sld_interface(z, drho_jm1_l, drho_j, drho_j_u, sigma_jm1_l, sigma_j, sigma_j_u, dd_jm1_l, dd_j_u):
        """Calculate the sld of one interface"""
        sld = drho_j_u * (0.5 + 0.5 * erf((z - dd_j_u) / sqrt(2 * (sigma_j_u**2 + sigma_j**2))))
        sld += drho_jm1_l * (0.5 + 0.5 * erf((z + dd_jm1_l) / sqrt(2 * (sigma_jm1_l**2 + sigma_j**2))))
        sld += drho_j * (0.5 + 0.5 * erf(z / sqrt(2) / sigma_j))
        return sld

    def calc_sld(z, int_pos, sld, sld_l, sld_u, sigma_l, sigma_c, sigma_u, dd_l, dd_u):
        return (
            sum(
                sld_interface(
                    -(z[:, newaxis] - int_pos),
                    -(sld[1:] - sld_l[1:]),
                    -(sld_l[1:] - sld_u[:-1]),
                    -(sld_u[:-1] - sld[:-1]),
                    sigma_l[1:],
                    sigma_c[:-1],
                    sigma_u[:-1],
                    dd_l[1:],
                    dd_u[:-1],
                ),
                1,
            )
            + sld[-1]
        )

    lamda = instrument.wavelength
    parameters: LayerParameters = sample.resolveLayerParameters()
    dens = array(parameters.dens, dtype=float64)
    resdens = array(parameters.resdens, dtype=float64)
    resmag = array(parameters.resmag, dtype=float64)
    mag = array(parameters.magn, dtype=float64)

    dmag_l = array(parameters.dmag_l, dtype=float64)
    dmag_u = array(parameters.dmag_u, dtype=float64)
    dd_l = array(parameters.dd_l, dtype=float64)
    dd_u = array(parameters.dd_u, dtype=float64)
    # print [type(f) for f in parameters['f']]
    xray_energy = AA_to_eV / instrument.wavelength
    f = refl.cast_to_array(parameters.f, xray_energy)
    fr = refl.cast_to_array(parameters.fr, xray_energy)
    fm1 = refl.cast_to_array(parameters.fm1, xray_energy)
    fm2 = refl.cast_to_array(parameters.fm2, xray_energy)

    d = array(parameters.d, dtype=float64)

    phi = array(parameters.magn_ang, dtype=float64) * pi / 180.0
    theta_m = array(parameters.magn_theta, dtype=float64) * pi / 180.0
    # Create an offset of phi to handle negative mag values and negate theta_m
    phi = phi + where(mag < 0, pi, 0)
    theta_m = where(mag < 0, -theta_m, theta_m)
    mag = abs(mag)
    m_x = cos(theta_m) * cos(phi)
    sl_c = dens * (f + resdens * fr)
    sl_m1 = dens * resdens * resmag * mag * fm1
    sl_m2 = dens * resdens * resmag * mag**2 * fm2
    sigma_c = array(parameters.sigma, dtype=float64) + 1e-20
    sigma_l = array(parameters.sigma_ml, dtype=float64) + 1e-20
    sigma_u = array(parameters.sigma_mu, dtype=float64) + 1e-20
    sl_m1_l = sl_m1 * (1.0 + dmag_l)
    sl_m1_u = sl_m1 * (1.0 + dmag_u)
    sl_m2_l = sl_m2 * (1.0 + dmag_l) ** 2
    sl_m2_u = sl_m2 * (1.0 + dmag_u) ** 2

    b = (array(parameters.b, dtype=complex128)) * 1e-5
    abs_xs = (array(parameters.xs_ai, dtype=complex128)) * 1e-4**2
    wl = instrument.wavelength
    sl_n = neturon_sld(abs_xs, b, dens, wl)
    mag_d = mag * dens
    mag_d_l = mag_d * (1.0 + dmag_l)
    mag_d_u = mag_d * (1.0 + dmag_u)

    int_pos = cumsum(r_[0, d[1:-1]])
    if z is None:
        z = arange(-sigma_c[0] * 10 - 50, int_pos.max() + sigma_c.max() * 10 + 50, 0.5)
    # Note: First layer substrate and last ambient
    sld_c = calc_sld(z, int_pos, sl_c, sl_c, sl_c, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
    sld_m = calc_sld(z, int_pos, sl_m1 * m_x, sl_m1_l * m_x, sl_m1_u * m_x, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
    sld_n = calc_sld(z, int_pos, sl_n, sl_n, sl_n, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
    mag_dens = calc_sld(z, int_pos, mag_d, mag_d_l, mag_d_u, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
    mag_dens_x = calc_sld(
        z,
        int_pos,
        mag_d * cos(theta_m) * cos(phi),
        mag_d_l * cos(theta_m) * cos(phi),
        mag_d_u * cos(theta_m) * cos(phi),
        sigma_l,
        sigma_c,
        sigma_u,
        dd_l,
        dd_u,
    )
    mag_dens_y = calc_sld(
        z,
        int_pos,
        mag_d * cos(theta_m) * sin(phi),
        mag_d_l * cos(theta_m) * sin(phi),
        mag_d_u * cos(theta_m) * sin(phi),
        sigma_l,
        sigma_c,
        sigma_u,
        dd_l,
        dd_u,
    )

    theory = instrument.probe
    if theory == ProbeTheory.xray_aniso:
        c = 1 / (lamda**2 * r_e / pi)
        A = -sl_c / c
        B = sl_m1 / c
        C = sl_m2 / c

        M = c_[cos(theta_m) * cos(phi), cos(theta_m) * sin(phi), sin(theta_m)]
        chi = xrmr.create_chi(None, None, A * 0, A, B, C, M, None)[0]
        chi_l = xrmr.create_chi(None, None, A * 0, A, B * (1 + dmag_l), C * (1 + dmag_l) ** 2, M, None)[0]
        chi_u = xrmr.create_chi(None, None, A * 0, A, B * (1 + dmag_u), C * (1 + dmag_u) ** 2, M, None)[0]

        chi_xx, chi_xy, chi_xz = chi[0]
        chi_yx, chi_yy, chi_yz = chi[1]
        chi_zx, chi_zy, chi_zz = chi[2]
        chi_l_xx, chi_l_xy, chi_l_xz = chi_l[0]
        chi_l_yx, chi_l_yy, chi_l_yz = chi_l[1]
        chi_l_zx, chi_l_zy, chi_l_zz = chi_l[2]
        chi_u_xx, chi_u_xy, chi_u_xz = chi_u[0]
        chi_u_yx, chi_u_yy, chi_u_yz = chi_u[1]
        chi_u_zx, chi_u_zy, chi_u_zz = chi_u[2]
        c_xx = calc_sld(z, int_pos, chi_xx, chi_l_xx, chi_u_xx, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
        c_xy = calc_sld(z, int_pos, chi_xy, chi_l_xy, chi_u_xy, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
        c_xz = calc_sld(z, int_pos, chi_xz, chi_l_xz, chi_u_xz, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
        c_yy = calc_sld(z, int_pos, chi_yy, chi_l_yy, chi_u_yy, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
        c_yz = calc_sld(z, int_pos, chi_yz, chi_l_yz, chi_u_yz, sigma_l, sigma_c, sigma_u, dd_l, dd_u)
        c_zz = calc_sld(z, int_pos, chi_zz, chi_l_zz, chi_u_zz, sigma_l, sigma_c, sigma_u, dd_l, dd_u)

        return {
            "Re sl_xx": c_xx.real * c,
            "Re sl_xy": c_xy.real * c,
            "Re sl_xz": c_xz.real * c,
            "Re sl_yy": c_yy.real * c,
            "Re sl_yz": c_yz.real * c,
            "Re sl_zz": c_zz.real * c,
            "Im sl_xx": c_xx.imag * c,
            "Im sl_xy": c_xy.imag * c,
            "Im sl_xz": c_xz.imag * c,
            "Im sl_yy": c_yy.imag * c,
            "Im sl_yz": c_yz.imag * c,
            "Im sl_zz": c_zz.imag * c,
            "z": z,
            "SLD unit": "r_e/\\AA^{3}",
        }
    elif theory == ProbeTheory.xray_simple_aniso:
        return {
            "Re sld_c": sld_c.real,
            "Im sld_c": sld_c.imag,
            "Re sld_m": sld_m.real,
            "Im sld_m": sld_m.imag,
            "mag_dens": mag_dens,
            "z": z,
            "SLD unit": "r_{e}/\\AA^{3},\\,\\mu_{B}/\\AA^{3}",
        }
    elif theory in [ProbeTheory.npol, ProbeTheory.ntofpol]:
        return {"sld_n": sld_n, "mag_dens": mag_dens, "z": z, "SLD unit": "fm/\\AA^{3}, \\mu_{B}/\\AA^{3}"}
    elif theory == ProbeTheory.npolsf:
        return {
            "sld_n": sld_n,
            "mag_dens": mag_dens,
            "mag_dens_x": mag_dens_x,
            "mag_dens_y": mag_dens_y,
            "z": z,
            "SLD unit": "fm/\\AA^{3}, \\mu_{B}/\\AA^{3}",
        }
    elif theory == ProbeTheory.xray_iso:
        return {
            "Re sld_c": sld_c.real,
            "Im sld_c": sld_c.imag,
            "z": z,
            "SLD unit": "r_{e}/\\AA^{3},\\,\\mu_{B}/\\AA^{3}",
        }
    else:
        raise ValueError("Wrong value of theory given. Value: %s" % theory)


def compose_sld(sample: Sample, instrument: Instrument, theta, xray_energy, layer=None):
    """Composes the sld for a slicing model

    Parameters:
        sample: The sample
        instrument: The instrument
        theta: The incident angle
        xray_energy: The xray energy either scalar or array
        layer: Defines which layer number to return. If None (default) returns the entire profile.
    """
    parameters: LayerParameters = sample.resolveLayerParameters()
    dmag_l = array(parameters.dmag_l, dtype=float64)
    dmag_u = array(parameters.dmag_u, dtype=float64)
    dd_u = array(parameters.dd_u, dtype=float64)
    dd_l = array(parameters.dd_l, dtype=float64)
    d = array(parameters.d, dtype=float64)
    mag = array(parameters.magn, dtype=float64)

    if isscalar(xray_energy):
        shape = None
    else:
        shape = (d.shape[0], xray_energy.shape[0])
    lamda = AA_to_eV / xray_energy
    dens = refl.harm_sizes(parameters.dens, shape, dtype=float64)
    resdens = refl.harm_sizes(parameters.resdens, shape, dtype=float64)
    resmag = refl.harm_sizes(parameters.resmag, shape, dtype=float64)

    f = refl.harm_sizes(refl.cast_to_array(parameters.f, xray_energy), shape, dtype=complex128)
    fr = refl.harm_sizes(refl.cast_to_array(parameters.fr, xray_energy), shape, dtype=complex128)
    fm1 = refl.harm_sizes(refl.cast_to_array(parameters.fm1, xray_energy), shape, dtype=complex128)
    fm2 = refl.harm_sizes(refl.cast_to_array(parameters.fm2, xray_energy), shape, dtype=complex128)

    sl_c = dens * (f + resdens * fr)
    sl_m1 = dens * resdens * resmag * fm1
    sl_m2 = dens * resdens * resmag * fm2  # mag is multiplied in later

    theory = instrument.probe
    if theory == ProbeTheory.xray_simple_aniso:
        # If simplified theory set sl_m2 to zero to be able to back calculate B
        sl_m2 *= 0

    phi = array(parameters.magn_ang, dtype=float64) * pi / 180.0
    theta_m = array(parameters.magn_theta, dtype=float64) * pi / 180.0
    # Create an offset of phi to handle negative mag values and negate theta_m
    phi = phi + where(mag < 0, pi, 0)
    theta_m = where(mag < 0, -theta_m, theta_m)
    mag = abs(mag)

    M = c_[cos(theta_m) * cos(phi), cos(theta_m) * sin(phi), sin(theta_m)]

    sigma_c = array(parameters.sigma, dtype=float64)
    sigma_mu = sqrt(array(parameters.sigma_mu, dtype=float64)[:-1] ** 2 + sigma_c[:-1] ** 2)
    sigma_ml = sqrt(array(parameters.sigma_ml, dtype=float64)[1:] ** 2 + sigma_c[:-1] ** 2)

    # Neutrons
    wl = instrument.wavelength
    abs_xs = array(parameters.xs_ai, dtype=complex128)  # *(1e-4)**2
    b = array(parameters.b, dtype=complex128)  # *1e-5
    dens_n = array(parameters.dens, dtype=float64)
    sl_n = dens_n * b
    abs_n = dens_n * abs_xs
    mag_dens = 0.0
    mag_dens_x = 0.0
    mag_dens_y = 0.0

    if sample.slicing:
        dz = sample.slice_depth
        reply = edm.create_profile_cm2(
            d[1:-1],
            sigma_c[:-1].real,
            sigma_ml.real,
            sigma_mu.real,
            [edm.erf_profile] * len(d),
            [edm.erf_interf] * len(d),
            dmag_l,
            dmag_u,
            mag,
            dd_l,
            dd_u,
            dz=dz,
            mult=sample.sld_mult,
            buffer=sample.sld_buffer,
            delta=sample.sld_delta,
        )
        z, comp_prof, mag_prof = reply
        if not shape is None:
            new_shape = (shape[0], comp_prof.shape[1], shape[1])
        else:
            new_shape = None
        comp_prof_x = refl.harm_sizes(comp_prof, new_shape, dtype=float64)
        mag_prof_x = refl.harm_sizes(mag_prof, new_shape, dtype=float64)
        sl_c_lay = comp_prof_x * sl_c[:, newaxis]
        sl_m1_lay = comp_prof_x * mag_prof_x * sl_m1[:, newaxis]
        sl_m2_lay = comp_prof_x * mag_prof_x**2 * sl_m2[:, newaxis]

        # Neutrons
        sl_n_lay = comp_prof * sl_n[:, newaxis]
        abs_n_lay = comp_prof * abs_n[:, newaxis]
        mag_dens_lay = comp_prof * mag_prof * dens_n[:, newaxis]

        if not shape is None:
            M = rollaxis(
                array(
                    (
                        ones(comp_prof_x.shape) * M[:, 0][:, newaxis, newaxis],
                        ones(comp_prof_x.shape) * M[:, 1][:, newaxis, newaxis],
                        ones(comp_prof_x.shape) * M[:, 2][:, newaxis, newaxis],
                    )
                ),
                0,
                2,
            )
        else:
            M = rollaxis(
                array(
                    (
                        ones(comp_prof_x.shape) * M[:, 0][:, newaxis],
                        ones(comp_prof_x.shape) * M[:, 1][:, newaxis],
                        ones(comp_prof_x.shape) * M[:, 2][:, newaxis],
                    )
                ),
                0,
                2,
            )

        A = -(lamda**2) * r_e / pi * sl_c_lay
        B = lamda**2 * r_e / pi * sl_m1_lay
        C = lamda**2 * r_e / pi * sl_m2_lay
        g_0 = sin(theta * pi / 180.0)

        chi, non_mag, mpy = xrmr.create_chi(g_0, lamda, A, 0.0 * A, B, C, M, d)
        if layer is not None:
            sl_c = sl_c_lay[layer]
            sl_m1 = sl_m1_lay[layer]
            sl_m2 = sl_m2_lay[layer]
            sl_n = sl_n_lay[layer]
            abs_n = abs_n_lay[layer]
            mag_dens = mag_dens_lay[layer]
            mag_dens_x = (comp_prof * mag_prof * (dens_n * cos(theta_m) * cos(phi))[:, newaxis])[layer]
            mag_dens_y = (comp_prof * mag_prof * (dens_n * cos(theta_m) * sin(phi))[:, newaxis])[layer]
            chi = tuple([c[layer] for c in chi[0] + chi[1] + chi[2]])
        else:
            sl_c = sl_c_lay.sum(0)
            sl_m1 = sl_m1_lay.sum(0)
            sl_m2 = sl_m2_lay.sum(0)
            sl_n = sl_n_lay.sum(0)
            abs_n = abs_n_lay.sum(0)
            mag_dens = mag_dens_lay.sum(0)
            mag_dens_x = (comp_prof * mag_prof * (dens_n * cos(theta_m) * cos(phi))[:, newaxis]).sum(0)
            mag_dens_y = (comp_prof * mag_prof * (dens_n * cos(theta_m) * sin(phi))[:, newaxis]).sum(0)
            chi = tuple([c.sum(0) for c in chi[0] + chi[1] + chi[2]])

        if sample.compress:
            # Compressing the profile..
            lamda_max = lamda if isscalar(lamda) else lamda.max()
            dsld_max = sample.dsld_max
            dchi_max = dsld_max * lamda_max**2 * r_e / pi
            dsld_offdiag_max = sample.dsld_offdiag_max
            dsld_n_max = sample.dsld_n_max
            dabs_n_max = sample.dabs_n_max
            dmag_max = sample.dmag_max
            dchi_od_max = dsld_offdiag_max * lamda_max**2 * r_e / pi

            index, z = edm.compress_profile_index_n(
                z,
                chi + (sl_n, mag_dens, abs_n, mag_dens_x, mag_dens_y),
                (
                    dchi_max,
                    dchi_od_max,
                    dchi_od_max,
                    dchi_od_max,
                    dchi_max,
                    dchi_od_max,
                    dchi_od_max,
                    dchi_od_max,
                    dchi_max,
                    dsld_n_max,
                    dmag_max,
                    dabs_n_max,
                    dmag_max,
                    dmag_max,
                ),
            )
            reply = edm.create_compressed_profile(
                (sl_c, sl_m1, sl_m2) + chi + (sl_n, mag_dens, abs_n, mag_dens_x, mag_dens_y), index
            )
            (
                sl_c,
                sl_m1,
                sl_m2,
                chi_xx,
                chi_xy,
                chi_xz,
                chi_yx,
                chi_yy,
                chi_yz,
                chi_zx,
                chi_zy,
                chi_zz,
                sl_n,
                mag_dens,
                abs_n,
                mag_dens_x,
                mag_dens_y,
            ) = reply
            non_mag = (abs(chi_xy) < mag_limit) * (abs(chi_xz) < mag_limit) * (abs(chi_yz) < mag_limit)
            mpy = (
                (abs(chi_yz) / abs(chi_xx) < mpy_limit) * (abs(chi_xy) / abs(chi_xx) < mpy_limit) * bitwise_not(non_mag)
            )
            # print mpy
            chi = ((chi_xx, chi_xy, chi_xz), (chi_yx, chi_yy, chi_yz), (chi_zx, chi_zy, chi_zz))
        else:
            (chi_xx, chi_xy, chi_xz, chi_yx, chi_yy, chi_yz, chi_zx, chi_zy, chi_zz) = chi
            non_mag = (abs(chi_xy) < mag_limit) * (abs(chi_xz) < mag_limit) * (abs(chi_yz) < mag_limit)
            non_mag[0] = True
            mpy = (
                (abs(chi_yz) / abs(chi_xx) < mpy_limit) * (abs(chi_xy) / abs(chi_xx) < mpy_limit) * bitwise_not(non_mag)
            )
            chi = ((chi_xx, chi_xy, chi_xz), (chi_yx, chi_yy, chi_yz), (chi_zx, chi_zy, chi_zz))
        d = r_[z[1:] - z[:-1], 1]
    else:
        A = -(lamda**2) * r_e / pi * sl_c
        B = lamda**2 * r_e / pi * sl_m1
        C = lamda**2 * r_e / pi * sl_m2
        g_0 = sin(theta * pi / 180.0)
        chi, non_mag, mpy = xrmr.create_chi(g_0, lamda, A, 0.0 * A, B, C, M, d)
        z = [0.0]
    return d, sl_c, sl_m1, sl_m2, M, chi, non_mag, mpy, sl_n, abs_n, mag_dens, mag_dens_x, mag_dens_y, z[0]


def extract_anal_iso_pars(sample: Sample, instrument: Instrument, theta, xray_energy, pol, Q=None):
    """Note Q is only used for Neutron TOF
    :param lamda:
    """
    parameters: LayerParameters = sample.resolveLayerParameters()
    dmag_l = array(parameters.dmag_l, dtype=float64)
    dmag_u = array(parameters.dmag_u, dtype=float64)
    dd_u = array(parameters.dd_u, dtype=float64)
    dd_l = array(parameters.dd_l, dtype=float64)
    d = array(parameters.d, dtype=float64)
    mag = array(parameters.magn, dtype=float64)

    shape = (d.shape[0], theta.shape[0])
    lamda = AA_to_eV / xray_energy
    dens = refl.harm_sizes(parameters.dens, shape, dtype=float64)
    resdens = refl.harm_sizes(parameters.resdens, shape, dtype=float64)
    resmag = refl.harm_sizes(parameters.resmag, shape, dtype=float64)

    f = refl.harm_sizes(refl.cast_to_array(parameters.f, xray_energy), shape, dtype=complex128)
    fr = refl.harm_sizes(refl.cast_to_array(parameters.fr, xray_energy), shape, dtype=complex128)
    fm1 = refl.harm_sizes(refl.cast_to_array(parameters.fm1, xray_energy), shape, dtype=complex128)
    fm2 = refl.harm_sizes(refl.cast_to_array(parameters.fm2, xray_energy), shape, dtype=complex128)

    d = array(parameters.d, dtype=float64)

    theta = theta * pi / 180.0
    phi = array(parameters.magn_ang, dtype=float64) * pi / 180.0
    theta_m = array(parameters.magn_theta, dtype=float64) * pi / 180.0
    # Create an offset of phi to handle negative mag values and negate theta_m
    phi = phi + where(mag < 0, pi, 0)
    theta_m = where(mag < 0, -theta_m, theta_m)
    mag = abs(mag)
    sl_c = dens * (f + resdens * fr)
    m_x = refl.harm_sizes((cos(theta_m) * cos(phi)), shape)
    sl_m1 = (dens * resdens * resmag * refl.harm_sizes(mag, shape) * fm1) * cos(theta) * m_x

    sigma_c = array(parameters.sigma, dtype=float64)
    sigma_l = array(parameters.sigma_ml, dtype=float64)
    sigma_u = array(parameters.sigma_mu, dtype=float64)

    theory = instrument.probe

    if theory in [ProbeTheory.xray_aniso, ProbeTheory.xray_simple_aniso]:
        if pol == XRayPol.circ_plus:
            n = 1 - lamda**2 * r_e / pi * (sl_c + sl_m1) / 2.0
            n_l = 1 - lamda**2 * r_e / pi * (sl_c + sl_m1 * (1.0 + dmag_l)[:, newaxis]) / 2.0
            n_u = 1 - lamda**2 * r_e / pi * (sl_c + sl_m1 * (1.0 + dmag_u)[:, newaxis]) / 2.0
        elif pol == XRayPol.circ_minus:
            n = 1 - lamda**2 * r_e / pi * (sl_c - sl_m1) / 2.0
            n_l = 1 - lamda**2 * r_e / pi * (sl_c - sl_m1 * (1.0 + dmag_l)[:, newaxis]) / 2.0
            n_u = 1 - lamda**2 * r_e / pi * (sl_c - sl_m1 * (1.0 + dmag_u)[:, newaxis]) / 2.0
        else:
            raise NotImplementedError("extract_anal_iso_pars is only defined for x-ray circular polarization")
    elif theory in [ProbeTheory.npol, ProbeTheory.npolsf]:
        b = (array(parameters.b, dtype=complex128) * 1e-5)[:, newaxis] * ones(theta.shape)
        abs_xs = (array(parameters.xs_ai, dtype=complex128) * 1e-4**2)[:, newaxis] * ones(theta.shape)
        wl = instrument.wavelength * 1.0
        sld = dens * (wl**2 / 2.0 / pi * sqrt(b**2 - (abs_xs / 2.0 / wl) ** 2) - 1.0j * abs_xs * wl / 4 / pi)
        msld = (muB_to_SL * mag * wl**2 / 2 / pi * cos(theta_m) * cos(phi))[:, newaxis] * dens * ones(theta.shape)
        if pol == NeutronPol.up_up:
            n = 1.0 - sld - msld
            n_l = 1.0 - sld - msld * (1.0 + dmag_l)[:, newaxis]
            n_u = 1.0 - sld - msld * (1.0 + dmag_u)[:, newaxis]
        elif pol == NeutronPol.down_down:
            n = 1.0 - sld + msld
            n_l = 1.0 - sld + msld * (1.0 + dmag_l)[:, newaxis]
            n_u = 1.0 - sld + msld * (1.0 + dmag_u)[:, newaxis]
        else:
            raise NotImplementedError("extract_anal_iso_pars is only defined for neutron ++ and -- polarization")
    elif theory == ProbeTheory.ntofpol:
        wl = 4 * pi * sin(instrument.incangle * pi / 180) / Q
        b = (array(parameters.b, dtype=complex128) * 1e-5)[:, newaxis] * ones(wl.shape)
        abs_xs = (array(parameters.xs_ai, dtype=complex128) * 1e-4**2)[:, newaxis] * ones(wl.shape)
        sld = dens * (wl**2 / 2 / pi * sqrt(b**2 - (abs_xs / 2.0 / wl) ** 2) - 1.0j * abs_xs * wl / 4 / pi)
        msld = (muB_to_SL * mag[:, newaxis] * dens * wl**2 / 2 / pi) * (cos(theta_m) * cos(phi))[:, newaxis]

        if pol == NeutronPol.up_up:
            n = 1.0 - sld - msld
            n_l = 1.0 - sld - msld * (1.0 + dmag_l)[:, newaxis]
            n_u = 1.0 - sld - msld * (1.0 + dmag_u)[:, newaxis]
        elif pol == NeutronPol.down_down:
            n = 1.0 - sld + msld
            n_l = 1.0 - sld + msld * (1.0 + dmag_l)[:, newaxis]
            n_u = 1.0 - sld + msld * (1.0 + dmag_u)[:, newaxis]
        else:
            raise ValueError("An unexpected value of pol was given. Value: %s" % (pol,))
    elif theory == ProbeTheory.xray_iso:
        n = 1 - lamda**2 * r_e / pi * sl_c / 2.0
        n_l = 1 - lamda**2 * r_e / pi * sl_c / 2.0
        n_u = 1 - lamda**2 * r_e / pi * sl_c / 2.0
    else:
        raise ValueError("An unexpected value of theory was given. Value: %s" % (theory,))
    d = d - dd_u - dd_l
    d *= d >= 0
    return n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l


def reflectivity_xmag(sample: Sample, instrument: Instrument, theta, TwoThetaQz, xray_energy, return_amplitude=True):
    if sample.slicing:
        R = slicing_reflectivity(sample, instrument, theta, TwoThetaQz, xray_energy, return_amplitude=return_amplitude)
    else:
        R = analytical_reflectivity(
            sample, instrument, theta, TwoThetaQz, xray_energy, return_amplitude=return_amplitude
        )
    return R


def analytical_reflectivity(
    sample: Sample, instrument: Instrument, theta, TwoThetaQz, xray_energy, return_amplitude=True
):
    theory = instrument.probe
    parameters: LayerParameters = sample.resolveLayerParameters()

    dmag_l = array(parameters.dmag_l, dtype=float64)
    dmag_u = array(parameters.dmag_u, dtype=float64)
    dd_u = array(parameters.dd_u, dtype=float64)
    dd_l = array(parameters.dd_l, dtype=float64)

    sigma = array(parameters.sigma, dtype=float64) + 1e-9
    sigma_u = array(parameters.sigma_mu, dtype=float64) + 1e-9
    sigma_l = array(parameters.sigma_ml, dtype=float64) + 1e-9

    d = array(parameters.d, dtype=float64)
    if isscalar(xray_energy):
        shape = None
    else:
        shape = (d.shape[0], xray_energy.shape[0])
    lamda = AA_to_eV / xray_energy
    dens = refl.harm_sizes(parameters.dens, shape, dtype=float64)
    resdens = refl.harm_sizes(parameters.resdens, shape, dtype=float64)
    resmag = refl.harm_sizes(parameters.resmag, shape, dtype=float64)
    mag = array(parameters.magn, dtype=float64)

    f = refl.harm_sizes(refl.cast_to_array(parameters.f, xray_energy), shape, dtype=complex128)
    fr = refl.harm_sizes(refl.cast_to_array(parameters.fr, xray_energy), shape, dtype=complex128)
    fm1 = refl.harm_sizes(refl.cast_to_array(parameters.fm1, xray_energy), shape, dtype=complex128)
    fm2 = refl.harm_sizes(refl.cast_to_array(parameters.fm2, xray_energy), shape, dtype=complex128)

    phi = array(parameters.magn_ang, dtype=float64) * pi / 180.0
    theta_m = array(parameters.magn_theta, dtype=float64) * pi / 180.0
    # Create an offset of phi to handle negative mag values and negate theta_m
    phi = phi + where(mag < 0, pi, 0)
    theta_m = where(mag < 0, -theta_m, theta_m)
    mag = refl.harm_sizes(abs(mag), shape, dtype=float64)

    if theory == ProbeTheory.xray_aniso:
        sl_c = dens * (f + resdens * fr)
        sl_m1 = dens * resdens * resmag * mag * fm1
        sl_m2 = dens * resdens * resmag * mag**2 * fm2

        A = -(lamda**2) * r_e / pi * sl_c
        B = lamda**2 * r_e / pi * sl_m1
        C = lamda**2 * r_e / pi * sl_m2

        if not shape is None:
            M_shape = (shape[0], 3, shape[1])
        else:
            M_shape = None
        M = refl.harm_sizes(c_[cos(theta_m) * cos(phi), cos(theta_m) * sin(phi), sin(theta_m)], M_shape)

        g_0 = sin(theta * pi / 180.0)

        # Full theory
        if XBuffer.g_0 is not None:
            g0_ok = XBuffer.g_0.shape == g_0.shape
            if g0_ok:
                g0_ok = any(not_equal(XBuffer.g_0, g_0))
        else:
            g0_ok = False
        # TODO: fix buffering for aniso model
        if True or (
            XBuffer.parameters != parameters
            or XBuffer.coords != instrument.coords
            or not g0_ok
            or XBuffer.wavelength != lamda
        ):
            # print g_0.shape, lamda.shape, A.shape, B.shape, C.shape, M.shape,
            W = xrmr.calc_refl_int_lay(
                g_0,
                lamda,
                A * 0,
                A[::-1],
                B[::-1],
                C[::-1],
                M[::-1, ...],
                d[::-1],
                sigma[::-1],
                sigma_l[::-1],
                sigma_u[::-1],
                dd_l[::-1],
                dd_u[::-1],
                dmag_l[::-1],
                dmag_u[::-1],
            )
            XBuffer.W = W
            XBuffer.parameters = parameters
            XBuffer.coords = instrument.coords
            XBuffer.g_0 = g_0.copy()
            XBuffer.wavelength = lamda
        else:
            # print 'Reusing W'
            W = XBuffer.W
        trans = ones(W.shape, dtype=complex128)
        trans[0, 1] = 1.0j
        trans[1, 1] = -1.0j
        trans = trans / sqrt(2)
        # Wc = xrmr.dot2(trans, xrmr.dot2(W, xrmr.inv2(trans)))
        Wc = xrmr.dot2(trans, xrmr.dot2(W, conj(xrmr.inv2(trans))))
        # Different polarization channels:
        pol = instrument.xpol
        if pol == XRayPol.circ_plus:
            R = abs(Wc[0, 0]) ** 2 + abs(Wc[1, 0]) ** 2
        elif pol == XRayPol.circ_minus:
            R = abs(Wc[1, 1]) ** 2 + abs(Wc[0, 1]) ** 2
        elif pol == XRayPol.total:
            R = (abs(W[0, 0]) ** 2 + abs(W[1, 0]) ** 2 + abs(W[0, 1]) ** 2 + abs(W[1, 1]) ** 2) / 2
        elif pol == XRayPol.asymmetry:
            R = (
                2
                * (W[0, 0] * W[0, 1].conj() + W[1, 0] * W[1, 1].conj()).imag
                / (abs(W[0, 0]) ** 2 + abs(W[1, 0]) ** 2 + abs(W[0, 1]) ** 2 + abs(W[1, 1]) ** 2)
            )
        elif pol == XRayPol.sigma:
            R = abs(W[0, 0]) ** 2 + abs(W[1, 0]) ** 2
        elif pol == XRayPol.pi:
            R = abs(W[0, 1]) ** 2 + abs(W[1, 1]) ** 2
        elif pol == XRayPol.sigma_sigma:
            R = abs(W[0, 0]) ** 2
        elif pol == XRayPol.sigma_pi:
            R = abs(W[1, 0]) ** 2
        elif pol == XRayPol.pi_pi:
            R = abs(W[1, 1]) ** 2
        elif pol == XRayPol.pi_sigma:
            R = abs(W[0, 1]) ** 2
        else:
            raise ValueError("Variable pol has an unvalid value")
        # Override if we should return the complex amplitude (in this case a 2x2 matrix)
        if not return_amplitude:
            R = W

    elif theory == ProbeTheory.xray_simple_aniso:
        pol = instrument.xpol
        Q = 4 * pi / lamda * sin(theta * pi / 180)
        if pol == XRayPol.circ_plus:
            pars = extract_anal_iso_pars(sample, instrument, theta, xray_energy, XRayPol.circ_plus)
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            R = ables.ReflQ_mag(
                Q,
                lamda,
                n.T[:, ::-1],
                d[::-1],
                sigma_c[::-1],
                n_u.T[:, ::-1],
                dd_u[::-1],
                sigma_u[::-1],
                n_l.T[:, ::-1],
                dd_l[::-1],
                sigma_l[::-1],
            )
        elif pol == XRayPol.circ_minus:
            pars = extract_anal_iso_pars(sample, instrument, theta, xray_energy, XRayPol.circ_minus)
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            R = ables.ReflQ_mag(
                Q,
                lamda,
                n.T[:, ::-1],
                d[::-1],
                sigma_c[::-1],
                n_u.T[:, ::-1],
                dd_u[::-1],
                sigma_u[::-1],
                n_l.T[:, ::-1],
                dd_l[::-1],
                sigma_l[::-1],
            )
        elif pol == XRayPol.total:
            pars = extract_anal_iso_pars(sample, instrument, theta, xray_energy, XRayPol.circ_minus)
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            Rm = ables.ReflQ_mag(
                Q,
                lamda,
                n.T[:, ::-1],
                d[::-1],
                sigma_c[::-1],
                n_u.T[:, ::-1],
                dd_u[::-1],
                sigma_u[::-1],
                n_l.T[:, ::-1],
                dd_l[::-1],
                sigma_l[::-1],
            )
            pars = extract_anal_iso_pars(sample, instrument, theta, xray_energy, XRayPol.circ_plus)
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            Rp = ables.ReflQ_mag(
                Q,
                lamda,
                n.T[:, ::-1],
                d[::-1],
                sigma_c[::-1],
                n_u.T[:, ::-1],
                dd_u[::-1],
                sigma_u[::-1],
                n_l.T[:, ::-1],
                dd_l[::-1],
                sigma_l[::-1],
            )
            R = (Rp + Rm) / 2.0
        elif pol == XRayPol.asymmetry:
            pars = extract_anal_iso_pars(sample, instrument, theta, xray_energy, XRayPol.circ_minus)
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            Rm = ables.ReflQ_mag(
                Q,
                lamda,
                n.T[:, ::-1],
                d[::-1],
                sigma_c[::-1],
                n_u.T[:, ::-1],
                dd_u[::-1],
                sigma_u[::-1],
                n_l.T[:, ::-1],
                dd_l[::-1],
                sigma_l[::-1],
            )
            pars = extract_anal_iso_pars(sample, instrument, theta, xray_energy, XRayPol.circ_plus)
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            Rp = ables.ReflQ_mag(
                Q,
                lamda,
                n.T[:, ::-1],
                d[::-1],
                sigma_c[::-1],
                n_u.T[:, ::-1],
                dd_u[::-1],
                sigma_u[::-1],
                n_l.T[:, ::-1],
                dd_l[::-1],
                sigma_l[::-1],
            )
            R = (Rp - Rm) / (Rp + Rm)
        else:
            raise ValueError("Variable pol has an unvalid value")

    elif theory == ProbeTheory.npol:
        # neutron spin-pol calcs
        wl = instrument.wavelength
        Q = 4 * pi / wl * sin(theta * pi / 180)
        pol = instrument.npol
        if pol == NeutronPol.asymmetry:
            pars = extract_anal_iso_pars(sample, instrument, theta, xray_energy, NeutronPol("++"))
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            Rp = ables.ReflQ_mag(
                Q,
                lamda,
                n.T[:, ::-1],
                d[::-1],
                sigma_c[::-1],
                n_u.T[:, ::-1],
                dd_u[::-1],
                sigma_u[::-1],
                n_l.T[:, ::-1],
                dd_l[::-1],
                sigma_l[::-1],
            )
            pars = extract_anal_iso_pars(sample, instrument, theta, xray_energy, NeutronPol("--"))
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            Rm = ables.ReflQ_mag(
                Q,
                lamda,
                n.T[:, ::-1],
                d[::-1],
                sigma_c[::-1],
                n_u.T[:, ::-1],
                dd_u[::-1],
                sigma_u[::-1],
                n_l.T[:, ::-1],
                dd_l[::-1],
                sigma_l[::-1],
            )
            R = (Rp - Rm) / (Rp + Rm)
        else:
            pars = extract_anal_iso_pars(sample, instrument, theta, xray_energy, instrument.npol)
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            R = ables.ReflQ_mag(
                Q,
                lamda,
                n.T[:, ::-1],
                d[::-1],
                sigma_c[::-1],
                n_u.T[:, ::-1],
                dd_u[::-1],
                sigma_u[::-1],
                n_l.T[:, ::-1],
                dd_l[::-1],
                sigma_l[::-1],
            )
    elif theory == ProbeTheory.npolsf:
        # neutron spin-flip calcs
        wl = instrument.wavelength
        Q = 4 * pi / wl * sin(theta * pi / 180)
        # Check if we have calcluated the same sample previous:
        Q_ok = False
        if NBuffer.TwoThetaQz is not None:
            Q_ok = NBuffer.TwoThetaQz.shape == Q.shape
            if Q_ok:
                Q_ok = any(not_equal(NBuffer.TwoThetaQz, Q))
        if NBuffer.parameters != parameters or not Q_ok:
            b = array(parameters.b, dtype=complex128) * 1e-5
            abs_xs = array(parameters.xs_ai, dtype=complex128) * 1e-4**2
            # Bulk of the layers
            # sld = dens*(wl**2/2/pi*sqrt(fb**2 - (abs_xs/2.0/wl)**2) -
            #                   1.0J*abs_xs*wl/4/pi)

            V0 = 2 * 2 * pi * dens * (b - 1.0j * abs_xs / 2.0 / wl)
            Vmag = 2 * 2 * pi * muB_to_SL * mag * dens

            (Ruu, Rdd, Rud, Rdu) = neutron_refl.Refl_int_lay(
                Q,
                V0[::-1],
                Vmag[::-1],
                d[::-1],
                phi[::-1],
                sigma[::-1],
                dmag_u[::-1],
                dd_u[::-1],
                phi[::-1],
                sigma_u[::-1],
                dmag_l[::-1],
                dd_l[::-1],
                phi[::-1],
                sigma_l[::-1],
            )
            NBuffer.Ruu = Ruu.copy()
            NBuffer.Rdd = Rdd.copy()
            NBuffer.Rud = Rud.copy()
            NBuffer.parameters = parameters
            NBuffer.TwoThetaQz = Q.copy()
        else:
            pass

        pol = instrument.npol
        if pol == NeutronPol.up_up:
            R = NBuffer.Ruu
        elif pol == NeutronPol.down_down:
            R = NBuffer.Rdd
        elif pol in [NeutronPol.up_down or NeutronPol.down_up]:
            R = NBuffer.Rud
        elif pol == NeutronPol.asymmetry:
            R = (NBuffer.Ruu - NBuffer.Rdd) / (NBuffer.Ruu + NBuffer.Rdd + 2 * NBuffer.Rud)
        else:
            raise ValueError("The value of the polarization is WRONG." " It should be ++, -- or +-")
    elif theory == ProbeTheory.ntofpol:
        if instrument.coords != Coords.q:
            raise ValueError("Neutron TOF calculation only supports q as coordinate (x - axis)!")
        Q = TwoThetaQz
        wl = 4 * pi * sin(instrument.incangle * pi / 180) / Q
        pars = extract_anal_iso_pars(sample, instrument, theta, xray_energy, instrument.npol, Q=TwoThetaQz)
        n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
        R = ables.ReflQ_mag(
            TwoThetaQz,
            wl,
            n.T[:, ::-1],
            d[::-1],
            sigma_c[::-1],
            n_u.T[:, ::-1],
            dd_u[::-1],
            sigma_u[::-1],
            n_l.T[:, ::-1],
            dd_l[::-1],
            sigma_l[::-1],
        )
    elif theory == ProbeTheory.xray_iso:
        pars = extract_anal_iso_pars(sample, instrument, theta, xray_energy, "+")
        n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
        R = Paratt.Refl_nvary2(theta, lamda * ones(theta.shape), n, d, sigma_c)
    else:
        raise ValueError("The given theory mode does not exist")

    return R


def slicing_reflectivity(sample: Sample, instrument: Instrument, theta, TwoThetaQz, xray_energy, return_amplitude=True):
    lamda = AA_to_eV / xray_energy
    parameters = sample.resolveLayerParameters()

    (d, sl_c, sl_m1, sl_m2, M, chi, non_mag, mpy, sl_n, abs_n, mag_dens, mag_dens_x, mag_dens_y, z0) = compose_sld(
        sample, instrument, theta, xray_energy
    )
    g_0 = sin(theta * pi / 180.0)
    theory = instrument.probe
    # Full theory
    if not XBuffer.g_0 is None:
        g0_ok = XBuffer.g_0.shape == g_0.shape
        if g0_ok:
            g0_ok = allclose(XBuffer.g_0, g_0)
    else:
        g0_ok = False

    buffer_wl = array(XBuffer.wavelength)
    current_wl = array(lamda)
    if not XBuffer.wavelength is None:
        wl_ok = buffer_wl.shape == current_wl.shape
        if wl_ok:
            wl_ok = allclose(buffer_wl, lamda)
    else:
        wl_ok = False

    if theory == ProbeTheory.xray_aniso:
        if XBuffer.parameters != parameters or XBuffer.coords != instrument.coords or not g0_ok or not wl_ok:
            chi = tuple([tuple([item[::-1] for item in row]) for row in chi])
            d = d[::-1]
            non_mag = non_mag[::-1]
            mpy = mpy[::-1]
            W = xrmr.do_calc(g_0, lamda, chi, d, non_mag, mpy)
            XBuffer.W = W
            XBuffer.parameters = parameters
            XBuffer.coords = instrument.coords
            XBuffer.g_0 = g_0.copy()
            XBuffer.wavelength = lamda
        else:
            # print "Reusing W"
            W = XBuffer.W
        trans = ones(W.shape, dtype=complex128)
        trans[0, 1] = 1.0j
        trans[1, 1] = -1.0j
        trans = trans / sqrt(2)
        Wc = xrmr.dot2(trans, xrmr.dot2(W, conj(xrmr.inv2(trans))))
        # Different polarization channels:
        pol = instrument.xpol
        if pol == XRayPol.circ_plus:
            R = abs(Wc[0, 0]) ** 2 + abs(Wc[1, 0]) ** 2
        elif pol == XRayPol.circ_minus:
            R = abs(Wc[1, 1]) ** 2 + abs(Wc[0, 1]) ** 2
        elif pol == XRayPol.total:
            R = (abs(W[0, 0]) ** 2 + abs(W[1, 0]) ** 2 + abs(W[0, 1]) ** 2 + abs(W[1, 1]) ** 2) / 2
        elif pol == XRayPol.asymmetry:
            R = (
                2
                * (W[0, 0] * W[0, 1].conj() + W[1, 0] * W[1, 1].conj()).imag
                / (abs(W[0, 0]) ** 2 + abs(W[1, 0]) ** 2 + abs(W[0, 1]) ** 2 + abs(W[1, 1]) ** 2)
            )
        elif pol == XRayPol.sigma:
            R = abs(W[0, 0]) ** 2 + abs(W[1, 0]) ** 2
        elif pol == XRayPol.pi:
            R = abs(W[0, 1]) ** 2 + abs(W[1, 1]) ** 2
        elif pol == XRayPol.sigma_sigma:
            R = abs(W[0, 0]) ** 2
        elif pol == XRayPol.sigma_pi:
            R = abs(W[1, 0]) ** 2
        elif pol == XRayPol.pi_pi:
            R = abs(W[1, 1]) ** 2
        elif pol == XRayPol.pi_sigma:
            R = abs(W[0, 1]) ** 2
        else:
            raise ValueError("Variable pol has an unvalid value")
        if not return_amplitude:
            R = W
    elif theory == ProbeTheory.xray_simple_aniso:
        pol = instrument.xpol
        c = 1 / (lamda**2 * r_e / pi)
        sl_c = -chi[0][0] * c
        sl_m1 = -1.0j * chi[2][1] * c
        if pol == XRayPol.circ_plus:
            n = 1 - lamda**2 * r_e / pi * (sl_c[:, newaxis] + sl_m1[:, newaxis] * cos(theta * pi / 180)) / 2.0
            R = Paratt.Refl_nvary2(theta, lamda * ones(theta.shape), n, d, zeros(d.shape))
        elif pol == XRayPol.circ_minus:
            n = 1 - lamda**2 * r_e / pi * (sl_c[:, newaxis] - sl_m1[:, newaxis] * cos(theta * pi / 180)) / 2.0
            R = Paratt.Refl_nvary2(theta, lamda * ones(theta.shape), n, d, zeros(d.shape))
        elif pol == XRayPol.total:
            n = 1 - lamda**2 * r_e / pi * (sl_c[:, newaxis] - sl_m1[:, newaxis] * cos(theta * pi / 180)) / 2.0
            Rm = Paratt.Refl_nvary2(theta, lamda * ones(theta.shape), n, d, zeros(d.shape))
            # Hack to check kinematical approx
            n = 1 - lamda**2 * r_e / pi * (sl_c[:, newaxis] + sl_m1[:, newaxis] * cos(theta * pi / 180)) / 2.0
            Rp = Paratt.Refl_nvary2(theta, lamda * ones(theta.shape), n, d, zeros(d.shape))
            # Hack to check kinematical approx
            R = (Rp + Rm) / 2.0
        elif pol == XRayPol.asymmetry:
            # ass
            n = 1 - lamda**2 * r_e / pi * (sl_c[:, newaxis] - sl_m1[:, newaxis] * cos(theta * pi / 180)) / 2.0
            Rm = Paratt.Refl_nvary2(theta, lamda * ones(theta.shape), n, d, zeros(d.shape))
            # Hack to check kinematical approx
            n = 1 - lamda**2 * r_e / pi * (sl_c[:, newaxis] + sl_m1[:, newaxis] * cos(theta * pi / 180)) / 2.0
            Rp = Paratt.Refl_nvary2(theta, lamda * ones(theta.shape), n, d, zeros(d.shape))
            # Hack to check kinematical approx
            R = (Rp - Rm) / (Rp + Rm)
        else:
            raise ValueError("Variable pol has an unvalid value")
    elif theory == ProbeTheory.npol:
        lamda = instrument.wavelength
        sl_n = sl_n * 1e-5
        abs_n = abs_n * 1e-8
        sl_n = lamda**2 / 2 / pi * sl_n - 1.0j * abs_n * lamda / 4 / pi
        sl_nm = muB_to_SL * mag_dens * lamda**2 / 2 / pi
        pol = instrument.npol
        if pol == NeutronPol.up_up:
            n = 1.0 - sl_n - sl_nm
            R = Paratt.Refl_nvary2(
                theta, lamda * ones(theta.shape), n[:, newaxis] * ones(theta.shape), d, zeros(d.shape)
            )
        elif pol == NeutronPol.down_down:
            n = 1.0 - sl_n + sl_nm
            R = Paratt.Refl_nvary2(
                theta, lamda * ones(theta.shape), n[:, newaxis] * ones(theta.shape), d, zeros(d.shape)
            )
        elif pol == NeutronPol.asymmetry:
            n = 1.0 - sl_n + sl_nm
            Rm = Paratt.Refl_nvary2(
                theta, lamda * ones(theta.shape), n[:, newaxis] * ones(theta.shape), d, zeros(d.shape)
            )
            n = 1.0 - sl_n - sl_nm
            Rp = Paratt.Refl_nvary2(
                theta, lamda * ones(theta.shape), n[:, newaxis] * ones(theta.shape), d, zeros(d.shape)
            )
            R = (Rp - Rm) / (Rp + Rm)
        else:
            raise ValueError(f"For simple neutron polarized model, polarization {pol} is not possible")
    elif theory == ProbeTheory.npolsf:
        lamda = instrument.wavelength
        sl_n = sl_n * 1e-5
        abs_n = abs_n * 1e-8
        Q = 4 * pi / lamda * sin(theta * pi / 180)
        # Check if we have calcluated the same sample previous:
        Q_ok = False
        if NBuffer.TwoThetaQz is not None:
            Q_ok = NBuffer.TwoThetaQz.shape == Q.shape
            if Q_ok:
                Q_ok = any(not_equal(NBuffer.TwoThetaQz, Q))
        if NBuffer.parameters != parameters or not Q_ok or True:
            # Bulk of the layers
            # V0 = 2*2*pi*dens*(sqrt(b**2 - (abs_xs/2.0/wl)**2) -
            #                   1.0J*abs_xs/2.0/wl)
            # These rows are added to always have an ambient in the structure
            # large roughness messes up the spin-flip channel otherwise.
            sl_n = append(sl_n, sample.Ambient.dens * sample.Ambient.b * 1e-5)
            abs_n = append(abs_n, sample.Ambient.dens * sample.Ambient.xs_ai * 1e-4**2)
            mag_dens = append(mag_dens, 0.0)
            mag_dens_x = append(mag_dens_x, 0.0)
            mag_dens_y = append(mag_dens_y, 0.0)
            d = append(d, 0.0)

            V0 = 2 * 2 * pi * (sl_n - 1.0j * abs_n / 2.0 / lamda)
            mag = sqrt(mag_dens_x**2 + mag_dens_y**2)
            Vmag = 2 * 2 * pi * muB_to_SL * mag
            phi_tmp = arccos(mag_dens_x / mag)
            phi = where(mag < 1e-20, zeros_like(mag), phi_tmp)
            (Ruu, Rdd, Rud, Rdu) = neutron_refl.Refl(Q, V0[::1] + Vmag[::1], V0[::1] - Vmag[::1], d[::1], phi[::1])

            NBuffer.Ruu = Ruu.copy()
            NBuffer.Rdd = Rdd.copy()
            NBuffer.Rud = Rud.copy()
            NBuffer.parameters = parameters
            NBuffer.TwoThetaQz = Q.copy()
        else:
            pass

        pol = instrument.npol
        if pol == NeutronPol.up_up:
            R = NBuffer.Ruu
        elif pol == NeutronPol.down_down:
            R = NBuffer.Rdd
        elif pol in [NeutronPol.up_down, NeutronPol.down_up]:
            R = NBuffer.Rud
        elif pol == NeutronPol.asymmetry:
            R = (NBuffer.Ruu - NBuffer.Rdd) / (NBuffer.Ruu + NBuffer.Rdd + 2 * NBuffer.Rud)
        else:
            raise ValueError("The value of the polarization is WRONG." " It should be ++(0), --(1) or +-(2)")

    elif theory == ProbeTheory.ntofpol:
        incang = instrument.incangle
        lamda = 4 * pi * sin(incang * pi / 180) / TwoThetaQz
        sl_n = sl_n[:, newaxis] * 1e-5
        abs_n = abs_n[:, newaxis] * 1e-8
        sl_n = lamda**2 / 2 / pi * sl_n - 1.0j * abs_n * lamda / 4 / pi
        sl_nm = muB_to_SL * mag_dens[:, newaxis] * lamda**2 / 2 / pi
        pol = instrument.npol

        if pol in NeutronPol.up_up:
            n = 1.0 - sl_n - sl_nm
            R = Paratt.Refl_nvary2(incang * ones(lamda.shape), lamda, n, d, zeros(d.shape))
        elif pol in NeutronPol.down_down:
            n = 1.0 - sl_n + sl_nm
            R = Paratt.Refl_nvary2(incang * ones(lamda.shape), lamda, n, d, zeros(d.shape))
        elif pol == NeutronPol.asymmetry:
            n = 1.0 - sl_n + sl_nm
            Rm = Paratt.Refl_nvary2(incang * ones(lamda.shape), lamda, n, d, zeros(d.shape))
            n = 1.0 - sl_n - sl_nm
            Rp = Paratt.Refl_nvary2(incang * ones(lamda.shape), lamda, n, d, zeros(d.shape))
            R = (Rp - Rm) / (Rp + Rm)
        else:
            raise ValueError(f"For simple neutron polarized model, polarization {pol} is not possible")
    elif theory == ProbeTheory.xray_iso:
        c = 1 / (lamda**2 * r_e / pi)
        sl_c = -chi[0][0] * c
        n = 1 - lamda**2 * r_e / pi * sl_c[:, newaxis] / 2.0 * ones(theta.shape)
        R = Paratt.Refl_nvary2(theta, lamda * ones(theta.shape), n, d, zeros(d.shape))
    else:
        raise ValueError("The given theory mode deos not exist")
    return R


def footprint_correction(instrument: Instrument, theta):
    footype = instrument.footype
    beamw = instrument.beamw
    samlen = instrument.samplelen
    if footype == FootType.none:
        foocor = 1.0
    elif footype == FootType.gauss:
        foocor = GaussIntensity(theta, samlen / 2.0, samlen / 2.0, beamw)
    elif footype == FootType.square:
        foocor = SquareIntensity(theta, samlen, beamw)
    else:
        raise ValueError("Variable footype has an unvalid value")
    return foocor


def convolute_reflectivity(R, instrument: Instrument, foocor, TwoThetaQz, weight):
    restype = instrument.restype
    if restype == ResType.none:
        R = R[:] * foocor
    elif restype == ResType.fast_conv:
        R = ConvoluteFast(TwoThetaQz, R[:] * foocor, instrument.res, range=instrument.resintrange)
    elif restype == ResType.fast_conv_var:
        R = ConvoluteFastVar(TwoThetaQz, R[:] * foocor, instrument.res, range=instrument.resintrange)
    elif restype == ResType.full_conv_var:
        R = ConvoluteResolutionVector(TwoThetaQz, R[:]*foocor, weight)
    else:
        raise ValueError("Variable restype has an unvalid value")
    return R


SimulationFunctions = {
    "Specular": Specular,
    "OffSpecular": OffSpecular,
    "EnergySpecular": EnergySpecular,
    "EnergySpecularField": EnergySpecularField,
    "SLD": SLD_calculations,
    "SpecularElectricField": SpecularElectricField,
}

Sample.setSimulationFunctions(SimulationFunctions)

class TestSpecNX(ModelTestCase):
    # TODO: currently this only checks for raise conditions in the code above, check of results should be added

    def test_spec_xray(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, f=3e-5 + 1e-7j, dens=0.1,
                                        magn=0.1, magn_ang=24.0, magn_theta=10.)])],
            Ambient=Layer(),
            Substrate=Layer(f=5e-5 + 2e-7j, dens=0.1),
            slicing=False,
        )
        instrument = Instrument(
            probe=ProbeTheory.xray_iso,
            coords=Coords.tth,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            wavelength=1.54,
        )
        with self.subTest("x-ray tth"):
            Specular(self.tth, sample, instrument)
        with self.subTest("x-ray tth-field"):
            SpecularElectricField(self.tth, sample, instrument)
        with self.subTest("x-ray q"):
            instrument.coords = Coords.q
            Specular(self.qz, sample, instrument)

        # resolution corrections
        with self.subTest("x-ray q-res-fast"):
            instrument.restype = ResType.fast_conv
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray q-res-fast-var"):
            instrument.restype = ResType.fast_conv_var
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray q-res-full"):
            instrument.restype = ResType.full_conv_var
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray q-res-wrong"):
            instrument.restype = 123
            with self.assertRaises(ValueError):
                Specular(self.qz, sample, instrument)
        instrument.restype = ResType.none

        # resonant models
        with self.subTest("x-ray res simple +"):
            instrument.probe = ProbeTheory.xray_simple_aniso
            instrument.xpol = XRayPol.circ_plus
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res simple -"):
            instrument.xpol = XRayPol.circ_minus
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res simple tot"):
            instrument.xpol = XRayPol.total
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res simple asym"):
            instrument.xpol = XRayPol.asymmetry
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full +"):
            instrument.probe = ProbeTheory.xray_aniso
            instrument.xpol = XRayPol.circ_plus
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full -"):
            instrument.xpol = XRayPol.circ_minus
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full tot"):
            instrument.xpol = XRayPol.total
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full asym"):
            instrument.xpol = XRayPol.asymmetry
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full sigma"):
            instrument.xpol = XRayPol.sigma
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full pi"):
            instrument.xpol = XRayPol.pi
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full sigma-sigma"):
            instrument.xpol = XRayPol.sigma_sigma
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full pi-pi"):
            instrument.xpol = XRayPol.pi_pi
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full sigma-pi"):
            instrument.xpol = XRayPol.sigma_pi
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full pi-sigma"):
            instrument.xpol = XRayPol.pi_sigma
            Specular(self.qz, sample, instrument)

        # footprint corrections
        with self.subTest("x-ray q-footprint-square"):
            instrument.footype = FootType.square
            Specular(self.qz, sample, instrument)
        instrument.coords = Coords.tth
        with self.subTest("x-ray tth-footprint-square"):
            Specular(self.tth, sample, instrument)
        with self.subTest("x-ray tth-footprint-gauss"):
            instrument.footype = FootType.gauss
            Specular(self.tth, sample, instrument)
        with self.subTest("x-ray tth-footprint-wrong"):
            instrument.footype = 123
            with self.assertRaises(ValueError):
                Specular(self.qz, sample, instrument)


    def test_spec_neutron(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, b=3e-6, dens=0.1, magn=0.1, magn_ang=24.0)])],
            Ambient=Layer(b=1e-7, dens=0.1),
            Substrate=Layer(b=4e-6, dens=0.1),
            slicing=False,
        )
        instrument = Instrument(
            probe=ProbeTheory.npol,
            coords=Coords.tth,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            wavelength=4.5,
            incangle=0.5,
        )
        with self.subTest("neutron tth"):
            instrument.npol = NeutronPol.up_up
            Specular(self.tth, sample, instrument)
        with self.subTest("neutron q +"):
            instrument.coords = Coords.q
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron q -"):
            instrument.npol = NeutronPol.down_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-tof q-footprint +"):
            instrument.npol = NeutronPol.up_up
            instrument.probe = ProbeTheory.ntofpol
            instrument.footype = FootType.square
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-tof q-footprint -"):
            instrument.npol = NeutronPol.down_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-tof q-footprint"):
            instrument.npol = NeutronPol.up_up
            instrument.footype = FootType.none
            instrument.restype = ResType.full_conv_var
            Specular(self.qz, sample, instrument)
        instrument.restype = ResType.none
        with self.subTest("neutron-pol++ q"):
            instrument.probe = ProbeTheory.npol
            instrument.pol = NeutronPol.up_up
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-pol-- q"):
            instrument.pol = NeutronPol.down_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-polsa q"):
            instrument.pol = NeutronPol.asymmetry
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-tofpol++ q"):
            instrument.probe = ProbeTheory.ntofpol
            instrument.pol = NeutronPol.up_up
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-tofpol-- q"):
            instrument.pol = NeutronPol.down_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-tofpolsa q"):
            instrument.pol = NeutronPol.asymmetry
            Specular(self.qz, sample, instrument)

        with self.subTest("neutron tthoffset"):
            instrument.probe = ProbeTheory.npol
            instrument.tthoff = 0.1
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-tof tthoffset"):
            instrument.probe = ProbeTheory.ntofpol
            instrument.tthoff = 0.1
            Specular(self.qz, sample, instrument)

        with self.subTest("neutron-sf++ q"):
            instrument.probe = ProbeTheory.npolsf
            instrument.pol = NeutronPol.up_up
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-sf-- q"):
            instrument.pol = NeutronPol.down_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-sf+- q"):
            instrument.pol = NeutronPol.up_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-sf-+ q"):
            instrument.pol = NeutronPol.down_up
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-sfsa q"):
            instrument.pol = NeutronPol.asymmetry
            Specular(self.qz, sample, instrument)

    def test_spec_slicing(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, f=3e-5 + 1e-7j, dens=0.1,
                                        magn=0.1, magn_ang=24.0, magn_theta=10.)])],
            Ambient=Layer(),
            Substrate=Layer(f=5e-5 + 2e-7j, dens=0.1),
            slicing=True,
        )
        instrument = Instrument(
            probe=ProbeTheory.xray_iso,
            coords=Coords.q,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            wavelength=1.54,
        )
        with self.subTest("x-ray iso"):
            Specular(self.qz, sample, instrument)

        with self.subTest("x-ray iso nocompress"):
            sample.compress = False
            Specular(self.qz, sample, instrument)
        sample.compress = True

        # resonant models
        with self.subTest("x-ray res simple +"):
            instrument.probe = ProbeTheory.xray_simple_aniso
            instrument.xpol = XRayPol.circ_plus
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res simple -"):
            instrument.xpol = XRayPol.circ_minus
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res simple tot"):
            instrument.xpol = XRayPol.total
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res simple asym"):
            instrument.xpol = XRayPol.asymmetry
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full +"):
            instrument.probe = ProbeTheory.xray_aniso
            instrument.xpol = XRayPol.circ_plus
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full -"):
            instrument.xpol = XRayPol.circ_minus
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full tot"):
            instrument.xpol = XRayPol.total
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full asym"):
            instrument.xpol = XRayPol.asymmetry
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full sigma"):
            instrument.xpol = XRayPol.sigma
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full pi"):
            instrument.xpol = XRayPol.pi
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full sigma-sigma"):
            instrument.xpol = XRayPol.sigma_sigma
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full pi-pi"):
            instrument.xpol = XRayPol.pi_pi
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full sigma-pi"):
            instrument.xpol = XRayPol.sigma_pi
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray res full pi-sigma"):
            instrument.xpol = XRayPol.pi_sigma
            Specular(self.qz, sample, instrument)

        # neutron models
        with self.subTest("neutron pol +"):
            instrument.probe = ProbeTheory.npol
            instrument.npol = NeutronPol.up_up
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron pol -"):
            instrument.npol = NeutronPol.down_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron pol asym"):
            instrument.npol = NeutronPol.asymmetry
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron tofpol +"):
            instrument.probe = ProbeTheory.ntofpol
            instrument.npol = NeutronPol.up_up
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron tofpol -"):
            instrument.npol = NeutronPol.down_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron tofpol asym"):
            instrument.npol = NeutronPol.asymmetry
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron polsf ++"):
            instrument.probe = ProbeTheory.npolsf
            instrument.npol = NeutronPol.up_up
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron polsf --"):
            instrument.npol = NeutronPol.down_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron polsf +-"):
            instrument.npol = NeutronPol.up_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron polsf -+"):
            instrument.npol = NeutronPol.down_up
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron polsf asym"):
            instrument.npol = NeutronPol.asymmetry
            Specular(self.qz, sample, instrument)


    def test_energy(self):
        from .utils import fp

        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, b=3e-6, f=fp.Fe, dens=0.1, magn=0.1, magn_ang=24.0)])],
            Ambient=Layer(b=1e-7, dens=0.1),
            Substrate=Layer(b=4e-6, f=fp.Si, dens=0.1),
            slicing=False,
        )
        instrument = Instrument(
            probe=ProbeTheory.xray_iso,
            coords=Coords.tth,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            wavelength=4.5,
            incangle=0.5,
        )
        energy = linspace(8000.0, 5000.0, 20)

        with self.subTest("energy xray"):
            EnergySpecular(energy, 1.5, sample, instrument)
        with self.subTest("energy xray q"):
            instrument.coords = Coords.q
            EnergySpecular(energy, 1.5, sample, instrument)

    def test_sld(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, f=2e-5 + 1e-7j, b=3e-6, dens=0.1, magn=0.1, magn_ang=24.0)])],
            Ambient=Layer(b=1e-7, dens=0.1),
            Substrate=Layer(b=4e-6, dens=0.1),
        )
        instrument = Instrument(
            probe=ProbeTheory.xray_iso,
            coords=Coords.tth,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            wavelength=4.5,
            incangle=0.5,
        )
        with self.subTest("sld xray"):
            SLD_calculations(None, None, sample, instrument)
        with self.subTest("sld xray slicing"):
            sample.slicing = True
            SLD_calculations(None, None, sample, instrument)
        sample.slicing = False
        with self.subTest("sld xray aniso"):
            instrument.probe = ProbeTheory.xray_aniso
            SLD_calculations(None, None, sample, instrument)
        with self.subTest("sld xray aniso slicing"):
            sample.slicing = True
            SLD_calculations(None, None, sample, instrument)
        sample.slicing = False
        with self.subTest("sld neutron"):
            instrument.probe = ProbeTheory.npol
            SLD_calculations(None, None, sample, instrument)
        with self.subTest("sld neutron pol"):
            instrument.probe = ProbeTheory.npolsf
            SLD_calculations(None, None, sample, instrument)
        with self.subTest("sld neutron pol2"):
            sample.Stacks[0].Layers[0].magn_ang = 0.0
            SLD_calculations(None, None, sample, instrument)

def standard_xray():
    """
        return the defied standard x-ray reflectivity to compare against other models
    """
    qz = linspace(0.01, 0.3, 15)
    return Specular(
        qz,
        Sample(
            Ambient=Layer(),
            Substrate=Layer(d=150.0, f=1e-5 + 1e-8j, dens=0.1),
            Stacks=[Stack(Layers=[Layer(d=150.0, f=2e-5 + 2e-8j, dens=0.1)])],
        ),
        Instrument(probe=ProbeTheory.xray_iso, coords=Coords.q, wavelength=1.54,
                   footype=FootType.none, restype=ResType.none),
    )
