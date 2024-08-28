"""
Library for combined x-ray and neutrons simulations with adaptive layer segmentation
====================================================================================
Library for specular neutron and x-ray reflectometry of more complex structures where elemental composition
and/or magnetism is better described separately than within one slap model. The model sums up a set of
*Elements* to calculate a total SLD profile and then uses adaptive layer segmentation to model it.

The actual modeling of the result structure is done with the same function as in spec_nx.

Classes
-------
"""

from dataclasses import dataclass, field, fields
from typing import List

import numpy as np

from . import spec_nx
# import all special footprint functions
from .lib import footprint as footprint_module
from .lib import neutron_refl as MatrixNeutron
from .lib import paratt as Paratt
from .lib import refl_base as refl
from .lib import resolution as resolution_module
from .lib.base import AltStrEnum
from .lib.footprint import *
from .lib.instrument import *
from .lib.physical_constants import T_to_SL, muB_to_SL, r_e
from .lib.resolution import *
from .lib.testing import ModelTestCase
from .spec_nx import AA_to_eV, Coords, FootType
from .spec_nx import Instrument as NXInstrument
from .spec_nx import LayerParameters, Polarization, Probe, ResType, q_limit

ModelID = "SpecAdaptive"

__xlabel__ = "q [Å$^{-1}$]"
__ylabel__ = "Instnsity [a.u.]"


class RoughType(AltStrEnum):
    gauss = "gauss"
    linear = "linear"
    exp1 = "exp-1"
    exp2 = "exp-2"


_rough_mapping = {RoughType.gauss: 0, RoughType.linear: 1, RoughType.exp1: 2, RoughType.exp2: 3}


@dataclass
class Layer(refl.ReflBase):
    """
    Representing a layer in the sample structur.

    ``d``
        The thickness of the layer in AA (Angstroms = 1e-10m)
    ``dens``
        The density of formula units in units per Angstroms. Note the units!
    ``sigma``
        The root-mean-square roughness of the top interface of the layer in Angstroms.
    ``rough_type``
       Used model to get the SLD profile of the interface, *gauss* is an error function profile (gaussian roughness),
       *linear* is a linear profile, *exp-1* and *exp-2* are exponential decays from bottom or top side.
    ``magn``
        The magnetic moment per formula unit (same formula unit as b and dens refer to)
    ``magn_ang``
        The angle of the magnetic moment in degress. 0 degrees correspond to
        a moment collinear with the neutron spin.
    ``magn_void``
       If true this layer has no magnetization. In case of *sigma_mag* beging larger than 0, the additional
       roughness is only applied to the magnetic layer and inside this layer follows the chemical profile.
    ``sigma_mag``
       A different roughness parameter for the magnetization of the layer, 0 is ignored

    ``f``
       The x-ray scattering length per formula unit in electrons. To be
       strict it is the number of Thompson scattering lengths for each
       formula unit.
    ``b``
       The neutron scattering length per formula unit in fm (femtometer = 1e-15m)
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

    rough_type: RoughType = "gauss"

    magn: float = 0.0
    magn_ang: float = 0.0
    magn_void: bool = False
    sigma_mag: float = 0.0

    Units = {
        "sigma": "AA",
        "dens": "at./AA",
        "d": "AA",
        "f": "el./at.",
        "b": "fm/at.",
        "xs_ai": "barn/at.",
        "magn": "mu_B/at.",
        "magn_ang": "deg.",
        "sigma_mag": "AA",
    }

    Groups = [
        ("General", ["d", "dens", "sigma", "rough_type"]),
        ("Neutron Magnetic", ["magn", "magn_ang", "magn_void", "sigma_mag"]),
        ("X-Ray", ["f"]),
        ("Neutron Nuclear", ["b", "xs_ai"]),
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

    def resolveLayerParameters(self):
        # resolve parameters by creating layers automatically
        # adapting changes to the SLDs
        par = {}
        for fi in fields(LayerParameters):
            par[fi.name] = [self._resolve_parameter(self.Substrate, fi.name)]
        par["sigma"][0] = 0.0
        d, rho_x, rho_n, rho_m, xs_ai, magn_ang = calculate_segmentation(self)
        par["d"] += d
        par["f"] += rho_x
        par["b"] += rho_n
        par["magn"] += rho_m
        par["xs_ai"] += xs_ai
        if self.smoothen:
            par["sigma"] += [(d[i] + d[i + 1]) / 4.0 for i in range(len(d) - 1)] + [0.0]
        else:
            par["sigma"] += [0.0 for ignore in d]
        par["magn_ang"] += magn_ang
        par["dens"] += [1.0 for ignore in d]
        for fi in fields(LayerParameters):
            par[fi.name].append(self._resolve_parameter(self.Ambient, fi.name))
        return LayerParameters(**par)


class Zeeman(AltStrEnum):
    none = "no corr"
    field = "field only"
    pos_sf = "SF q (+)"
    neg_sf = "SF q (-)"


@dataclass
class Instrument(NXInstrument):
    """
    Specify parameters of the probe and reflectometry instrument.

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
    ``tthoff``
        Linear offset to the scattering angle calibration
    ``probe``
        Describes the radiation and measurments used, it is one of:
        'x-ray', 'neutron', 'neutron pol', 'neutron pol spin flip',
        'neutron tof', 'neutron pol tof'.
        The calculations for x-rays uses ``f`` for the scattering length for
        neutrons ``b`` for 'neutron pol', 'neutron pol spin flip' and 'neutron
        pol tof' alternatives the ``magn`` is used in the calculations. Note
        that the angle of magnetization ``magn_ang`` is only used in the spin
        flip model.
    ``pol``
        The measured polarization of the instrument. Valid options are:
        'uu','dd', 'ud', 'du' or 'ass' the respective number 0-3 also works.
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
    ``zeeman``
        Apply corrections for Zeeman-effect when using neutron pol spin-flip model with elevated magnetic field
        and canted magnetic moments. The configuration can be one of 'no corr', 'field only', 'SF q (+)' or 'SF q (-)'.
        With 'field only', the q-values for calculation are not changed but the magnetic SLD for all layers
        is modified by the external field (including substrate and ambient layer). The 'SF q (+/-)' make an
        additional correct to the q-value for spin-flip channels that assumes the q-value was calculated
        from the incident angle (+) or the outgoing angle (-). (Direction of the beam changes due to
        the energy loss/gain from spin-flip in a magnetic field.)
    ``mag_field``
        Strength of the external magnetic field in T.
    """

    zeeman: Zeeman = "no corr"
    mag_field: float = 0.0

    Units = NXInstrument.Units | {"mag_field": "T"}
    Groups = NXInstrument.Groups + [("Zeeman correction", ["zeeman", "mag_field"])]


# A buffer to save previous calculations for spin-flip calculations
class Buffer:
    Ruu = 0
    Rdd = 0
    Rdu = 0
    Rud = 0
    parameters = None
    TwoThetaQz = None


def specular_calc_zeemann(TwoThetaQz, sample: Sample, instrument: Instrument):
    """For details see spec_nx implementation with more comments.

    This implements the concepts of [1] by correcting the q-position for spin-flip and
    adding a magnetic sld to all layers along the external field diretion.

    [1] Brian B. Maranville et al., "Polarized specular neutron reflectivity", J. Appl. Cryst. (2016). 49, 1121–1129
    """

    restype = instrument.restype
    pol = instrument.pol
    Q, TwoThetaQz, weight = spec_nx.resolution_init(TwoThetaQz, instrument)

    rho_Z = instrument.mag_field * T_to_SL
    # apply Zeeman correction to spin-flip channel q
    if (
        instrument.zeeman == Zeeman.pos_sf
        and pol == Polarization.up_down
        or instrument.zeeman == Zeeman.neg_sf
        and pol == Polarization.down_up
    ):
        Q = Q / 2.0 + np.sqrt((Q / 2.0) ** 2 + 8 * pi * rho_Z)
    elif (
        instrument.zeeman == Zeeman.pos_sf
        and pol == Polarization.down_up
        or instrument.zeeman == Zeeman.neg_sf
        and pol == Polarization.up_down
    ):
        Q = Q / 2.0 + np.sqrt(maximum(0.0, (Q / 2.0) ** 2 - 8 * pi * rho_Z))

    Q = maximum(Q, q_limit)

    parameters: LayerParameters = sample.resolveLayerParameters()

    dens = array(parameters.dens, dtype=float64)
    d = array(parameters.d, dtype=float64)
    magn = array(parameters.magn, dtype=float64)
    # Transform to radians
    magn_ang = array(parameters.magn_ang, dtype=float64) * pi / 180.0

    sigma = array(parameters.sigma, dtype=float64)

    fb = array(parameters.b, dtype=complex128) * 1e-5
    abs_xs = array(parameters.xs_ai, dtype=complex128) * 1e-4**2
    wl = instrument.wavelength
    # sld = dens*(wl**2/2/pi*sqrt(fb**2 - (abs_xs/2.0/wl)**2) -
    #                       1.0J*abs_xs*wl/4/pi)
    sld = spec_nx.neutron_sld(abs_xs, dens, fb, wl)

    # Check if we have calcluated the same sample previous:
    if Buffer.TwoThetaQz is not None:
        Q_ok = Buffer.TwoThetaQz.shape == Q.shape
        if Q_ok:
            Q_ok = not (Buffer.TwoThetaQz != Q).any()
    else:
        Q_ok = False
    if Buffer.parameters != (parameters, instrument.mag_field) or not Q_ok:
        msld = muB_to_SL * magn * dens
        # apply Zeeman correction to magnetic parameters
        magn_x = msld * cos(magn_ang)  # M parallel to polarization
        magn_y = msld * sin(magn_ang)  # M perpendicular to polarization
        magn_x += rho_Z  # Apply magnetization from external field
        # calculate new magnitudes and angles
        msld = sqrt(magn_x**2 + magn_y**2)
        magn_ang = np.arctan2(magn_y, magn_x)
        # renormalize SLDs if ambient layer is not vacuum
        if msld[-1] != 0.0 or sld[-1] != 0:
            msld -= msld[-1]
            sld -= sld[-1]
        sld_p = sld * 2 * pi / wl**2 + msld
        sld_m = sld * 2 * pi / wl**2 - msld
        Vp = (2 * pi) * (sld_p * (2.0 + sld_p))  # (1-np**2) - better numerical accuracy
        Vm = (2 * pi) * (sld_m * (2.0 + sld_m))  # (1-nm**2)
        (Ruu, Rdd, Rud, Rdu) = MatrixNeutron.Refl(Q, Vp, Vm, d, magn_ang, sigma, return_int=True)
        Buffer.Ruu = Ruu
        Buffer.Rdd = Rdd
        Buffer.Rud = Rud
        Buffer.parameters = (parameters, instrument.mag_field)
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

    # FootprintCorrections
    foocor = spec_nx.footprintcorr(Q, instrument)
    # Resolution corrections
    R = spec_nx.resolutioncorr(R, TwoThetaQz, foocor, instrument, weight)

    return R * instrument.I0 + instrument.Ibkg


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
    if not inst_pol in spec_nx.POL_CHANNELS:
        raise ValueError(f"Instrument polarization as to be one of {spec_nx.POL_CHANNELS}.")
    if instrument.probe != Probe.npolsf:
        raise ValueError("Polarization corrected simulation requires probe to be 'neutron pol spin flip'")

    instrument.pol = "uu"
    uu = Specular(TwoThetaQz, sample, instrument)
    instrument.pol = "dd"
    dd = Specular(TwoThetaQz, sample, instrument)
    instrument.pol = "ud"
    ud = Specular(TwoThetaQz, sample, instrument)
    instrument.pol = "du"
    du = Specular(TwoThetaQz, sample, instrument)
    instrument.pol = inst_pol

    P = get_pol_matrix(p1, p2, F1, F2)
    Pline = P[spec_nx.POL_CHANNELS.index(instrument.pol)]
    I = Pline[:, newaxis] * np.vstack([uu, ud, du, dd])
    return I.sum(axis=0)


def SLD_calculations(z, item, sample, inst):
    """Calculates the scatteringlength density as at the positions z
    if item is None or "all" the function returns a dictonary of values.
    Otherwise, it returns the item as identified by its string.

    # BEGIN Parameters
    z data.x
    item 'Re'
    # END Parameters
    """
    res = spec_nx.SLD_calculations(z, item, sample, inst)
    res["z"] -= 5 * sample._resolve_parameter(sample.Substrate, "sigma")
    return res


def Specular(TwoThetaQz, sample: Sample, instrument: Instrument):
    """Simulate the specular signal from sample when probed with instrument

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """
    if instrument.probe == Probe.npolsf and instrument.zeeman != Zeeman.none:
        # additional Zeeman correction
        out = specular_calc_zeemann(TwoThetaQz, sample, instrument)
    else:
        # use standard calculation
        out = spec_nx.specular_calcs(TwoThetaQz, sample, instrument, return_int=True)

    global __xlabel__
    __xlabel__ = spec_nx.__xlabel__
    return out


SimulationFunctions = {
    "Specular": Specular,
    "PolSpecular": PolSpecular,
    "SLD": SLD_calculations,
}

Sample.setSimulationFunctions(SimulationFunctions)


def calculate_segmentation(sample):
    """
    Calculate segmentation steps inside a sample defined by
    a maximum SLD slope and minimum step size. It first
    calculates the nuclear and magnetic SLD profile
    from the model and than separates it according to the
    given parameters.
    """
    parameters = resolve_parameters_by_element(sample)
    dens = array(parameters["dens"], dtype=float64)
    f = array(parameters["f"], dtype=complex128)
    b = array(parameters["b"], dtype=complex128)
    xs_ai = array(parameters["xs_ai"], dtype=float64)
    magn = array(parameters["magn"], dtype=float64)
    magn_ang = array(parameters["magn_ang"], dtype=float64) / 180.0 * pi
    magn_void = array(parameters["magn_void"], dtype=float64)

    crop_sigma = sample.crop_sigma
    sld_x = dens * f
    sld_n = dens * b
    sld_xs = dens * xs_ai
    # the new magnetization density and angle will be
    # calculated from perpendicular and parallel components to allow
    # a smooth transition in magnetic angle
    mag_sld_nsf = magn * dens * cos(magn_ang)
    mag_sld_sf = magn * dens * sin(magn_ang)

    d = array(parameters["d"], dtype=float64)
    d = d[1:-1]
    # Include one extra element - the zero pos (substrate/film interface)
    int_pos = cumsum(r_[0, d])
    rough_type = array([_rough_mapping[rti] for rti in parameters["rough_type"]], dtype=int)[:-1]
    sigma_n = array(parameters["sigma"], dtype=float64)[:-1] + 1e-7
    sigma_m = array(parameters["sigma_mag"], dtype=float64)[:-1] + 1e-7
    z = arange(-sigma_n[0] * 5, int_pos.max() + sigma_n[-1] * 5, sample.minimal_steps / 5.0)
    # interface transition functions, products for different per-layer rough_type cases
    # Gaussian (starts at 1 goes to 0, interface is at 0.5)
    trans_n = (0.5 - 0.5 * erf((z[:, newaxis] - int_pos) / sqrt(2.0) / sigma_n)) * (rough_type == 0)
    # Linear
    trans_n += maximum(0.0, minimum(1.0, (1.0 + (int_pos - z[:, newaxis]) / 2.0 / sigma_n) / 2.0)) * (rough_type == 1)
    # Exponential decrease or increase
    trans_n += maximum(0.0, minimum(1.0, exp((int_pos - z[:, newaxis]) / sigma_n))) * (rough_type == 2)
    trans_n += maximum(0.0, minimum(1.0, 1.0 - exp((z[:, newaxis] - int_pos) / sigma_n))) * (rough_type == 3)
    trans_m = (0.5 - 0.5 * erf((z[:, newaxis] - int_pos) / sqrt(2.0) / sigma_m)) * (rough_type == 0)
    trans_m += maximum(0.0, minimum(1.0, (1.0 + (int_pos - z[:, newaxis]) / 2.0 / sigma_m) / 2.0)) * (rough_type == 1)
    trans_m += maximum(0.0, minimum(1.0, exp((int_pos - z[:, newaxis]) / sigma_m))) * (rough_type == 2)
    trans_m += maximum(0.0, minimum(1.0, 1.0 - exp((z[:, newaxis] - int_pos) / sigma_m))) * (rough_type == 3)
    if crop_sigma:
        # Cop the roughness tails above and below the interface to avoid overspill.
        # This is done by introducing a split-location within a layer where the
        # influence of that layers roughness function is faded out with an error function.
        # The location of the split is defined by the ratio of adjacent roughness values.
        # I.e. the split is closer to the interface that has lower roughness.

        # sigma ratio -1 to 1
        sratio = (sigma_n[1:] - sigma_n[:-1]) / (sigma_n[1:] + sigma_n[:-1])
        # location of split relative to bottom interface
        delta_pos = (0.5 + 0.5 * sratio) * d
        sigma_pos = (1.0 - abs(sratio)) * d / 2.0
        # the fade out function applied to the interface from below and above located within a layer
        interface = 0.5 - 0.5 * erf((z[:, newaxis] - int_pos[1:] + delta_pos) * sqrt(2.0) / sigma_pos)

        # Fade out roughness influence close to top and bottom split location
        trans_n[:, 0] = trans_n[:, 0] * interface[:, 0]
        trans_n[:, 1:-1] = interface[:, :-1] + (1.0 - interface[:, :-1]) * trans_n[:, 1:-1] * interface[:, 1:]
        trans_n[:, -1] = interface[:, -1] + (1.0 - interface[:, -1]) * trans_n[:, -1]
        trans_m[:, 0] = trans_m[:, 0] * interface[:, 0]
        trans_m[:, 1:-1] = interface[:, :-1] + (1.0 - interface[:, :-1]) * trans_m[:, 1:-1] * interface[:, 1:]
        trans_m[:, -1] = interface[:, -1] + (1.0 - interface[:, -1]) * trans_m[:, -1]
    # SLD calculations
    rho_x = sum((sld_x[:-1] - sld_x[1:]) * trans_n, 1) + sld_x[-1]
    rho_n = sum((sld_n[:-1] - sld_n[1:]) * trans_n, 1) + sld_n[-1]
    rho_m_nsf = sum((mag_sld_nsf[:-1] - mag_sld_nsf[1:]) * trans_m, 1) + mag_sld_nsf[-1]
    rho_m_sf = sum((mag_sld_sf[:-1] - mag_sld_sf[1:]) * trans_m, 1) + mag_sld_sf[-1]
    rho_void = sum((magn_void[:-1] - magn_void[1:]) * trans_n, 1) + magn_void[-1]
    xs_ai_comb = sum((sld_xs[:-1] - sld_xs[1:]) * trans_n, 1) + sld_xs[-1]
    # add more elements to the SLDs
    for params in parameters["Elements"][1:]:
        dens = array(params["dens"], dtype=float64)
        f = array(params["f"], dtype=complex128)
        b = array(params["b"], dtype=complex128)
        xs_ai = array(params["xs_ai"], dtype=float64)
        magn = array(params["magn"], dtype=float64)
        magn_ang = array(params["magn_ang"], dtype=float64) / 180.0 * pi
        sld_x = dens * f
        sld_n = dens * b
        sld_xs = dens * xs_ai
        mag_sld_nsf = magn * dens * cos(magn_ang)
        mag_sld_sf = magn * dens * sin(magn_ang)

        d = array(params["d"], dtype=float64)
        d = d[1:-1]
        # Include one extra element - the zero pos (substrate/film interface)
        int_pos = cumsum(r_[0, d])
        rough_type = array([_rough_mapping[rti] for rti in params["rough_type"]], dtype=int)[:-1]
        sigma_n = array(params["sigma"], dtype=float64)[:-1] + 1e-7
        sigma_m = array(params["sigma_mag"], dtype=float64)[:-1] + 1e-7
        # interface transition functions
        trans_n = (0.5 - 0.5 * erf((z[:, newaxis] - int_pos) / sqrt(2.0) / sigma_n)) * (rough_type == 0)
        trans_n += maximum(0.0, minimum(1.0, (1.0 + (int_pos - z[:, newaxis]) / 2.0 / sigma_n) / 2.0)) * (
            rough_type == 1
        )
        trans_n += maximum(0.0, minimum(1.0, exp((int_pos - z[:, newaxis]) / sigma_n))) * (rough_type == 2)
        trans_n += maximum(0.0, minimum(1.0, 1.0 - exp((z[:, newaxis] - int_pos) / sigma_n))) * (rough_type == 3)
        trans_m = (0.5 - 0.5 * erf((z[:, newaxis] - int_pos) / sqrt(2.0) / sigma_m)) * (rough_type == 0)
        trans_m += maximum(0.0, minimum(1.0, (1.0 + (int_pos - z[:, newaxis]) / 2.0 / sigma_m) / 2.0)) * (
            rough_type == 1
        )
        trans_m += maximum(0.0, minimum(1.0, exp((int_pos - z[:, newaxis]) / sigma_m))) * (rough_type == 2)
        trans_m += maximum(0.0, minimum(1.0, 1.0 - exp((z[:, newaxis] - int_pos) / sigma_m))) * (rough_type == 3)
        # SLD calculations
        rho_x += sum((sld_x[:-1] - sld_x[1:]) * trans_n, 1) + sld_x[-1]
        rho_n += sum((sld_n[:-1] - sld_n[1:]) * trans_n, 1) + sld_n[-1]
        rho_m_nsf += sum((mag_sld_nsf[:-1] - mag_sld_nsf[1:]) * trans_m, 1) + mag_sld_nsf[-1]
        rho_m_sf += sum((mag_sld_sf[:-1] - mag_sld_sf[1:]) * trans_m, 1) + mag_sld_sf[-1]
        xs_ai_comb += sum((sld_xs[:-1] - sld_xs[1:]) * trans_n, 1) + sld_xs[-1]
    # calculate the segmentation
    d_segments = [0.0]
    i = 0
    rho_x_out = [rho_x[0]]
    rho_n_out = [rho_n[0]]
    rho_nsf_out = [rho_m_nsf[0]]
    rho_sf_out = [rho_m_sf[0]]
    xs_ai_out = [xs_ai_comb[0]]
    rho_x_r = rho_x.real
    rho_n_p = rho_n.real + rho_m_nsf
    rho_n_m = rho_n.real - rho_m_nsf
    while i < (len(z) - 1):
        j = next_adaptive_segment(i, rho_x_r, rho_n_p, rho_n_m, rho_m_sf, sample.max_diff_n, sample.max_diff_x, z)
        d_segments.append(z[j] - z[i])
        rho_x_out.append(rho_x[i:j].mean())
        rho_n_out.append(rho_n[i:j].mean())
        rho_nsf_out.append(
            rho_m_nsf[i:j].mean() * (1.0 - rho_void[i:j].mean())
        )  # averadge magn taking voids into account
        rho_sf_out.append(
            rho_m_sf[i:j].mean() * (1.0 - rho_void[i:j].mean())
        )  # averadge magn taking voids into account
        xs_ai_out.append(xs_ai_comb[i:j].mean())  # averadge mang angle
        i = j
    rho_nsf_out = array(rho_nsf_out)
    rho_sf_out = array(rho_sf_out)
    rho_m_out = sqrt(rho_nsf_out**2 + rho_sf_out**2).tolist()
    magn_ang_out = (arctan2(rho_nsf_out, -rho_sf_out) * 180.0 / pi - 90.0).tolist()
    return (d_segments[1:], rho_x_out[1:], rho_n_out[1:], rho_m_out[1:], xs_ai_out[1:], magn_ang_out[1:])


def next_adaptive_segment(i, rho_x_r, rho_n_p, rho_n_m, rho_m_sf, max_diff_n, max_diff_x, z):
    # calculate the maximum variation of SLD up to given index
    diff_x = abs(maximum.accumulate(rho_x_r[i + 1 :]) - minimum.accumulate(rho_x_r[i + 1 :]))
    diff_n_p = abs(maximum.accumulate(rho_n_p[i + 1 :]) - minimum.accumulate(rho_n_p[i + 1 :]))
    diff_n_m = abs(maximum.accumulate(rho_n_m[i + 1 :]) - minimum.accumulate(rho_n_m[i + 1 :]))
    diff_m_sf = abs(maximum.accumulate(rho_m_sf[i + 1 :]) - minimum.accumulate(rho_m_sf[i + 1 :]))
    diff_idx = where(
        logical_not(
            (diff_n_p < max_diff_n) & (diff_x < max_diff_x) & (diff_n_m < max_diff_n) & (diff_m_sf < max_diff_n)
        )
    )[0]
    if len(diff_idx) > 0:
        j = min(len(z) - 1, max(i + diff_idx[0] + 1, i + 5))
    else:
        j = len(z) - 1  # last position
    return j


def resolve_parameters_by_element(sample):
    """
    Resolve the model standard parameters for each element.
    The first element used for normal parameter names,
    every other element does ignore substrate and ambience sld
    and is assigned to the 'elements' keyword as list.
    This makes it possible to build SLD profiles as sum of all
    elements.
    """
    elements = list(set([stack.Element for stack in sample.Stacks]))
    elements.sort()
    par = sample.Substrate._parameters.copy()
    for k in par:
        par[k] = [sample._resolve_parameter(sample.Substrate, k)]
    for k in sample.Substrate._parameters:
        for stack in sample.Stacks:
            if stack.Element != 0:
                continue
            par[k] = par[k] + stack.resolveLayerParameter(k)
        par[k] = par[k] + [sample._resolve_parameter(sample.Ambient, k)]
    par["sigma_mag"] = where(array(par["sigma_mag"]) != 0.0, par["sigma_mag"], par["sigma"]).tolist()
    output = par
    output["Elements"] = []
    for element in elements:
        par = sample.Substrate._parameters.copy()
        for k in list(par.keys()):
            par[k] = [sample._resolve_parameter(sample.Substrate, k)]
        # zero substrat SLD
        par["f"] = [0j]
        par["b"] = [0j]
        par["magn"] = [0.0]
        par["xs_ai"] = [0.0]
        par["magn_ang"] = [0.0]
        for k in sample.Substrate._parameters:
            for stack in sample.Stacks:
                if stack.Element != element:
                    continue
                par[k] = par[k] + stack.resolveLayerParameter(k)
            par[k] = par[k] + [sample._resolve_parameter(sample.Ambient, k)]
        # zero ambience SLD for Element
        par["f"][-1] = 0j
        par["b"][-1] = 0j
        par["magn"][-1] = 0.0
        par["xs_ai"][-1] = 0.0
        par["magn_ang"][-1] = 0.0
        # if magnetic roughness is set to zero use structural roughness
        par["sigma_mag"] = where(array(par["sigma_mag"]) != 0.0, par["sigma_mag"], par["sigma"]).tolist()
        output["Elements"].append(par)
    return output


class TestSpecAdaptive(ModelTestCase):
    # TODO: currently this only checks for raise conditions in the code above, check of results should be added

    def test_spec_neutron(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, b=3e-6, dens=0.1, magn=0.1, magn_ang=24.0)])],
            Ambient=Layer(b=1e-7, dens=0.1),
            Substrate=Layer(b=4e-6, dens=0.1),
        )
        instrument = Instrument(
            probe=Probe.neutron,
            coords=Coords.tth,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            tthoff=0.0,
            wavelength=4.5,
            incangle=0.5,
        )
        with self.subTest("neutron tth"):
            Specular(self.tth, sample, instrument)
        with self.subTest("neutron q"):
            instrument.coords = Coords.q
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-tof q-footprint"):
            instrument.probe = Probe.ntof
            instrument.footype = FootType.square
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-tof q-footprint"):
            instrument.footype = FootType.none
            instrument.restype = ResType.full_conv_rel
            Specular(self.qz, sample, instrument)
            instrument.incangle = ones_like(self.qz) * 0.5
            Specular(self.qz, sample, instrument)
        instrument.incangle = 0.5
        instrument.restype = ResType.none
        with self.subTest("neutron-pol++ q"):
            instrument.probe = Probe.npol
            instrument.pol = Polarization.up_up
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-pol-- q"):
            instrument.pol = Polarization.down_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-polsa q"):
            instrument.pol = Polarization.asymmetry
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-tofpol++ q"):
            instrument.probe = Probe.ntofpol
            instrument.pol = Polarization.up_up
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-tofpol-- q"):
            instrument.pol = Polarization.down_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-tofpolsa q"):
            instrument.pol = Polarization.asymmetry
            Specular(self.qz, sample, instrument)

        with self.subTest("neutron-sf++ q"):
            instrument.probe = Probe.npolsf
            instrument.pol = Polarization.up_up
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-sf-- q"):
            instrument.pol = Polarization.down_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-sf+- q"):
            instrument.pol = Polarization.up_down
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-sf-+ q"):
            instrument.pol = Polarization.down_up
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-sfsa q"):
            instrument.pol = Polarization.asymmetry
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-polspec q"):
            instrument.pol = Polarization.up_up
            PolSpecular(self.qz, 0.01, 0.01, 0.01, 0.01, sample, instrument)

    def test_zeemann(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, b=3e-6, dens=0.1, magn=0.1, magn_ang=24.0)])],
            Ambient=Layer(b=1e-7, dens=0.1),
            Substrate=Layer(b=4e-6, dens=0.1),
        )
        instrument = Instrument(
            probe=Probe.npolsf,
            coords=Coords.tth,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            tthoff=0.0,
            wavelength=4.5,
            incangle=0.5,
            zeeman=Zeeman.field,
            mag_field=1.0,
        )
        with self.subTest("zeemann-sf++ q"):
            instrument.pol = Polarization.up_up
            Specular(self.qz, sample, instrument)
        with self.subTest("zeemann-sf-- q"):
            instrument.pol = Polarization.down_down
            Specular(self.qz, sample, instrument)
        with self.subTest("zeemann-sf+- q"):
            instrument.pol = Polarization.up_down
            Specular(self.qz, sample, instrument)
        with self.subTest("zeemann-sf-+ q"):
            instrument.pol = Polarization.down_up
            Specular(self.qz, sample, instrument)
        with self.subTest("zeemann-sf+- q+"):
            instrument.pol = Polarization.up_down
            instrument.zeeman = Zeeman.pos_sf
            Specular(self.qz, sample, instrument)
        with self.subTest("zeemann-sf+- q-"):
            instrument.pol = Polarization.up_down
            instrument.zeeman = Zeeman.neg_sf
            Specular(self.qz, sample, instrument)

    def test_sld(self):
        sample = Sample(
            Stacks=[
                Stack(Layers=[Layer(d=150, sigma=2.0, f=2e-5 + 1e-7j, b=3e-6, dens=0.1, magn=0.1, magn_ang=24.0)]),
                Stack(
                    Layers=[Layer(d=150, sigma=2.0, f=2e-5 + 1e-7j, b=3e-6, dens=0.1, magn=0.1, magn_ang=24.0)],
                    Element=1,
                ),
            ],
            crop_sigma=True,
            Ambient=Layer(b=1e-7, dens=0.1),
            Substrate=Layer(b=4e-6, dens=0.1),
            minimal_steps=5.0,
        )
        instrument = Instrument(
            probe=Probe.xray,
            coords=Coords.tth,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            tthoff=0.0,
            wavelength=4.5,
            incangle=0.5,
        )
        with self.subTest("sld xray"):
            SLD_calculations(None, None, sample, instrument)
        with self.subTest("sld neutron"):
            instrument.probe = Probe.neutron
            SLD_calculations(None, None, sample, instrument)
        with self.subTest("sld neutron pol"):
            instrument.probe = Probe.npolsf
            SLD_calculations(None, None, sample, instrument)
        with self.subTest("sld neutron pol2"):
            sample.Stacks[0].Layers[0].magn_ang = 0.0
            SLD_calculations(None, None, sample, instrument)
        with self.subTest("sld neutron crop"):
            instrument.probe = Probe.neutron
            sample.crop_sld = 15
            sample.Stacks[0].Repetitions = 100
            SLD_calculations(None, None, sample, instrument)
            sample.crop_sld = -15
            sample.Stacks[0].Repetitions = 100
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
        Instrument(probe=Probe.xray, coords=Coords.q, wavelength=1.54, footype=FootType.none, restype=ResType.none),
    )
