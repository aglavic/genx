"""
Library for specular and off-specular x-ray reflectivity
========================================================
interdiff is a model for specular and off specular simulations including
the effects of interdiffusion in the calculations. The specular
simulations is conducted with Parrats recursion formula. The
off-specular, diffuse calculations are done with the distorted wave Born
approximation (DWBA) as derived by Holy and with the extensions done by
Wormington to include diffuse interfaces.

Classes
-------

"""

from dataclasses import dataclass, field, fields
from typing import List

import numpy as np

# import all special footprint functions
from .lib import footprint as footprint_module
from .lib import neutron_refl as MatrixNeutron
from .lib import offspec
from .lib import paratt as Paratt
from .lib import refl_base as refl
from .lib import resolution as resolution_module
from .lib.base import AltStrEnum
from .lib.footprint import *
from .lib.instrument import *
from .lib.physical_constants import AA_to_eV, muB_to_SL, r_e
from .lib.resolution import *
from .lib.testing import ModelTestCase

# Preamble to define the parameters needed for the models outlined below:
ModelID = "MingInterdiff"

q_limit = 1e-10
""" Minimum allowed q-value """

__xlabel__ = "q [Å$^{-1}$]"
__ylabel__ = "Instnsity [a.u.]"


class Coords(AltStrEnum):
    q = "q"
    tth = "2θ"
    alternate_tth = "tth"


class ResType(AltStrEnum):
    none = "no conv"
    fast_conv = "fast conv"
    fast_conv_var = "fast conv + varying res."
    full_conv_abs = "full conv and varying res."


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
       The total root mean square *roughness* of the top interface of the layer in
       Angstroms. This includes both the extended "wavy" roughness sigmar and the atomic-scale
       interface broadening due to interdiffusion sigmai. It is defined as sqrt(sigmar**2+sigmai**2).
    ``sigmai``
       The root mean square *interdiffusion* of the top interface of the
       layer in Angstroms.
    ``f``
       The x-ray scattering length per formula unit in electrons. To be
       strict it is the number of Thompson scattering lengths for each
       formula unit.
    """

    d: float = 0.0
    dens: float = 1.0
    sigma: float = 5.0
    sigmai: float = 0.0
    f: complex = 1e-20j

    Units = {
        "d": "AA",
        "dens": "at./AA",
        "sigma": "AA",
        "sigmai": "AA",
        "f": "el./at.",
    }

    Groups = [("General", ["d", "dens", "sigma", "sigmai"]), ("X-Ray", ["f"])]


@dataclass
class LayerParameters:
    d: List[float]
    dens: List[float]
    sigma: List[float]
    sigmai: List[float]
    f: List[complex]


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
    ``eta_z``
       The out-of plane (vertical) correlation length of the roughness in
       the sample. Given in AA.
    ``eta_x``
       The in-plane global correlation length (it is assumed equal for all
       layers). Given in AA.
    ``h``
       The jaggedness parameter, should be between 0 (jagged) and 1.0 (smooth). This describes
       how jagged the interfaces are. This is also a global parameter for
       all interfaces. (See e.g.: [book](https://doi.org/10.1007/978-1-4612-3784-6_2) )
    """

    Stacks: List[Stack] = field(default_factory=list)
    Ambient: Layer = field(default_factory=Layer)
    Substrate: Layer = field(default_factory=Layer)
    eta_z: float = 100.0
    eta_x: float = 100.0
    h: float = 1.0

    _layer_parameter_class = LayerParameters


@dataclass
class Instrument(refl.ReflBase):
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
        only used for 'full conv and vaying res.' and 'fast conv + varying res'.
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
    ``taylor_n``
        The number terms taken into account in the taylor expansion of the
        fourier integral of the correlation function. More terms more accurate
        calculation but also much slower.
    """

    wavelength: float = 1.54
    coords: Coords = "2θ"
    I0: float = 1.0
    Ibkg: float = 0.0
    restype: ResType = "no conv"
    res: float = 0.001
    respoints: int = 5
    resintrange: float = 2.0
    footype: FootType = "no corr"
    beamw: float = 0.01
    samplelen: float = 10.0
    taylor_n: int = 2

    Units = {
        "wavelength": "Å",
        "coords": "",
        "I0": "arb.",
        "Ibkg": "arb.",
        "restype": "",
        "res": "[coord]",
        "respoints": "pts.",
        "resintrange": "[coord]",
        "footype": "",
        "beamw": "mm",
        "samplelen": "mm",
        "2θ": "°",
        "tth": "°",
        "q": "Å$^-1$",
    }

    Groups = [
        ("Radiation", ["wavelength", "I0", "Ibkg"]),
        ("X-Resolution", ["restype", "res", "respoints", "resintrange"]),
        ("X-Coordinates", ["coords"]),
        ("DWBA", ["taylor_n"]),
        ("Footprint", ["footype", "beamw", "samplelen"]),
    ]


def Specular(TwoThetaQz, sample: Sample, instrument: Instrument):
    """Simulate the specular signal from sample when proped with instrument

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """
    # preamble to get it working with my class interface
    global __xlabel__
    __xlabel__ = "q [Å$^{-1}$]"

    restype = instrument.restype
    weight = 0
    if isinstance(restype, resolution_module.Resolution):
        (TwoThetaQz, weight) = restype(TwoThetaQz[:], instrument.respoints, instrument.resintrange)
    elif restype == ResType.full_conv_abs:
        res_function = GaussianResolution(sigma=instrument.res)
        (TwoThetaQz, weight) = res_function(TwoThetaQz[:], instrument.respoints, instrument.resintrange)

    if instrument.coords == Coords.tth:
        theta = TwoThetaQz / 2
        __xlabel__ = "2θ [°]"
    elif instrument.coords == Coords.q:
        theta = arcsin(TwoThetaQz / 4 / pi * instrument.wavelength) * 180.0 / pi

    lamda = instrument.wavelength
    parameters: LayerParameters = sample.resolveLayerParameters()
    dens = array(parameters.dens, dtype=float64)
    # print [type(f) for f in parameters['f']]
    f = array(parameters.f, dtype=complex128)
    n = 1 - dens * r_e * lamda**2 / 2 / pi * f
    d = array(parameters.d, dtype=float64)

    sigma = array(parameters.sigma, dtype=float64)

    R = Paratt.Refl(theta, lamda, n, d, sigma) * instrument.I0

    # FootprintCorrections
    foocor = 1.0
    footype = instrument.footype
    beamw = instrument.beamw
    samlen = instrument.samplelen
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

    R = R[:] * foocor
    restype = instrument.restype
    if restype == ResType.fast_conv:
        R = ConvoluteFast(TwoThetaQz, R, instrument.res, range=instrument.resintrange)
    elif restype == ResType.full_conv_abs or isinstance(restype, resolution_module.Resolution):
        R = ConvoluteResolutionVector(TwoThetaQz, R, weight)
    elif restype == ResType.fast_conv_var:
        R = ConvoluteFastVar(TwoThetaQz, R, instrument.res, range=instrument.resintrange)
    elif restype == ResType.none:
        pass
    else:
        raise ValueError("The choice of resolution type, restype," "is WRONG")
    return R + instrument.Ibkg


def OffSpecularMingInterdiff(TwoThetaQz, ThetaQx, sample: Sample, instrument: Instrument):
    """Function that simulates the off-specular signal (not implemented)

    # BEGIN Parameters
    TwoThetaQz 1.0
    ThetaQx data.x
    # END Parameters
    """
    lamda = instrument.wavelength
    k = 2 * pi / lamda
    if instrument.coords == Coords.tth:
        alphaR1 = ThetaQx
        betaR1 = TwoThetaQz - ThetaQx
        cen_idx = where(betaR1 >= 0)[0][0]
        qx = k * (cos(alphaR1 * pi / 180) - cos(betaR1 * pi / 180))
        qz = k * (sin(alphaR1 * pi / 180) + sin(betaR1 * pi / 180))
    else:
        qz = TwoThetaQz
        qx = ThetaQx
        cen_idx = where(qx >= 0)[0][0]
    if isinstance(qz, ndarray):
        qz_c = qz[cen_idx]
    else:
        qz_c = qz
    tth = arcsin(qz_c / 4.0 / pi * lamda) * 2.0
    qx_max = k * (1.0 - cos(tth))

    if instrument.coords == Coords.q:
        # calculate incident angle from qx for footprint correction
        alphaR1 = (arctan2(qx, qz) + tth / 2.0) * 180.0 / pi

    parameters: LayerParameters = sample.resolveLayerParameters()

    def toarray(a, code):
        a = list(a)
        a.reverse()
        return array(a, dtype=code)

    dens = array(parameters.dens, dtype=complex64)
    f = array(parameters.f, dtype=complex64)
    re = 2.82e-13 * 1e2 / 1e-10
    n = 1 - dens * re * lamda**2 / 2 / pi * f * 1e-4
    n = toarray(n, code=complex64)
    sigmai = toarray(parameters.sigmai, code=float64)
    sigma = toarray(parameters.sigma, code=float64)
    sigmar = sqrt(sigma**2 - np.minimum(sigma, sigmai) ** 2)
    sigmar = sigmar[1:]
    sigmai = sigmai[1:] + 1e-5

    d = toarray(parameters.d, code=float64)
    d = r_[0, d[1:-1]]

    z = -cumsum(d)

    eta = sample.eta_x

    h = sample.h
    if h < 0 or h > 1:
        raise ValueError("The Hurst parameter h has to be between 0 and 1")

    eta_z = sample.eta_z

    (I, alpha, omega) = offspec.DWBA_Interdiff(
        qx, qz, lamda, n, z, sigmar, sigmai, eta, h, eta_z, d, taylor_n=instrument.taylor_n
    )
    I = real(I) * ((qx > -qx_max) & (qx < qx_max)) * instrument.I0

    # FootprintCorrections
    foocor = 1.0
    footype = instrument.footype
    beamw = instrument.beamw
    samlen = instrument.samplelen
    if footype == FootType.gauss:
        foocor = GaussIntensity(alphaR1, samlen / 2.0, samlen / 2.0, beamw)
    elif footype == FootType.square:
        foocor = SquareIntensity(alphaR1, samlen, beamw)
    elif footype == FootType.none:
        pass
    elif isinstance(footype, footprint_module.Footprint):
        foocor = footype(alphaR1, samlen)
    else:
        raise ValueError("The choice of footprint correction, footype," "is WRONG")
    # off-specular footprint correction of a homogeneous beam larger than the sample would be 1
    I *= foocor / alphaR1 * tth / 2.0

    restype = instrument.restype
    if restype == ResType.none:
        # if no resolution is defined, don't include specular peak
        return I + instrument.Ibkg

    # include specular peak
    Ibgk = instrument.Ibkg
    instrument.Ibkg = 0.0
    instrument.restype = ResType(0)
    if isinstance(TwoThetaQz, ndarray):
        Ispec = Specular(array([TwoThetaQz[cen_idx]], dtype=float64), sample, instrument)
    else:
        Ispec = Specular(array([TwoThetaQz], dtype=float64), sample, instrument)[0]
    instrument.Ibkg = Ibgk
    instrument.restype = restype

    if instrument.coords == Coords.tth:
        spec_peak = Ispec * exp(-0.5 * (TwoThetaQz / 2.0 - ThetaQx) ** 2 / instrument.res**2)
    else:
        # angular resolution converted to qx grid
        qx_res = k * (cos(tth / 2) - cos(tth / 2 * (1 + instrument.res / qz_c)))
        spec_peak = Ispec * exp(-0.5 * ThetaQx**2 / qx_res**2)

    return spec_peak + I + Ibgk


def SLD_calculations(z, item, sample: Sample, inst: Instrument):
    """Calculates the scatteringlength density as at the positions z

    # BEGIN Parameters
    z data.x
    item "Re"
    # END Parameters
    """
    parameters: LayerParameters = sample.resolveLayerParameters()
    dens = array(parameters.dens, dtype=complex64)
    f = array(parameters.f, dtype=complex64)
    sld = dens * f
    d_sld = sld[:-1] - sld[1:]
    d = array(parameters.d, dtype=float64)
    d = d[1:-1]
    # Include one extra element - the zero pos (substrate/film interface)
    int_pos = cumsum(r_[0, d])
    sigma = array(parameters.sigma, dtype=float64)[:-1] + 1e-7
    if z is None:
        z = arange(-sigma[0] * 5, int_pos.max() + sigma[-1] * 5, 0.5)
    rho = sum(d_sld * (0.5 - 0.5 * erf((z[:, newaxis] - int_pos) / sqrt(2.0) / sigma)), 1) + sld[-1]
    dic = {"Re": real(rho), "Im": imag(rho), "z": z, "SLD unit": "r_{e}/\\AA^{3}"}
    if item is None or item == "all":
        return dic
    else:
        try:
            return dic[item]
        except:
            raise ValueError("The chosen item, %s, does not exist" % item)


SimulationFunctions = {"Specular": Specular, "OffSpecular": OffSpecularMingInterdiff, "SLD": SLD_calculations}

Sample.setSimulationFunctions(SimulationFunctions)

class TestInterdiff(ModelTestCase):
    # TODO: currently this only checks for raise conditions in the code above, check of results should be added

    def test_spec(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, f=3e-5 + 1e-7j, dens=0.1)])],
            Ambient=Layer(),
            Substrate=Layer(f=5e-5 + 2e-7j, dens=0.1),
        )
        instrument = Instrument(
            coords=Coords.tth,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            wavelength=1.54,
        )
        with self.subTest("x-ray tth"):
            Specular(self.tth, sample, instrument)
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
            instrument.restype = ResType.full_conv_abs
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray q-res-gauss"):
            instrument.restype = resolution_module.GaussianResolution(sigma=0.001)
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray q-res-trapezoid"):
            instrument.restype = resolution_module.TrapezoidResolution(inner_width=0.001, outer_width=0.002)
            instrument.restype.get_weight_example()
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray q-res-wrong"):
            instrument.restype = 123
            with self.assertRaises(ValueError):
                Specular(self.qz, sample, instrument)
        instrument.restype = ResType.none

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
        with self.subTest("x-ray tth-footprint-gauss-offset"):
            instrument.footype = GaussianBeamOffset(sigma=0.5, offset=0.1)
            Specular(self.tth, sample, instrument)
        with self.subTest("x-ray tth-footprint-square-offset"):
            instrument.footype = SquareBeamOffset(width=0.1, offset=0.05)
            Specular(self.tth, sample, instrument)
        with self.subTest("x-ray tth-footprint-trapezoid"):
            instrument.footype = TrapezoidBeam(inner_width=0.1, outer_width=0.2)
            Specular(self.tth, sample, instrument)
            instrument.footype = TrapezoidBeam(inner_width=0.1, outer_width=0.1)
            Specular(self.tth, sample, instrument)
            instrument.footype = TrapezoidBeam(inner_width=0.0, outer_width=0.2)
            Specular(self.tth, sample, instrument)
        with self.subTest("x-ray tth-footprint-wrong"):
            instrument.footype = 123
            with self.assertRaises(ValueError):
                Specular(self.qz, sample, instrument)


    def test_offspec(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, f=3e-5 + 1e-7j, dens=0.1)])],
            Ambient=Layer(),
            Substrate=Layer(f=5e-5 + 2e-7j, dens=0.1),
        )
        instrument = Instrument(
            coords=Coords.tth,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            wavelength=1.54,
        )
        with self.subTest("x-ray tth"):
            OffSpecularMingInterdiff(self.tth, self.tth, sample, instrument)
        with self.subTest("x-ray q"):
            instrument.coords = Coords.q
            OffSpecularMingInterdiff(0.01*self.qz, self.qz, sample, instrument)

        # resolution corrections
        with self.subTest("x-ray q-res-fast"):
            instrument.restype = ResType.fast_conv
            OffSpecularMingInterdiff(0.01*self.qz, self.qz, sample, instrument)
            OffSpecularMingInterdiff(0.001, self.qz, sample, instrument)
        with self.subTest("x-ray tth-res-fast"):
            instrument.coords = Coords.tth
            OffSpecularMingInterdiff(self.tth, self.tth, sample, instrument)
            OffSpecularMingInterdiff(0.1, self.tth, sample, instrument)
        instrument.restype = ResType.none
        instrument.coords = Coords.q

        # footprint corrections
        with self.subTest("x-ray q-footprint-square"):
            instrument.footype = FootType.square
            OffSpecularMingInterdiff(0.01*self.qz, self.qz, sample, instrument)
        instrument.coords = Coords.tth
        with self.subTest("x-ray tth-footprint-square"):
            OffSpecularMingInterdiff(self.tth, self.tth, sample, instrument)
        with self.subTest("x-ray tth-footprint-gauss"):
            instrument.footype = FootType.gauss
            OffSpecularMingInterdiff(self.tth, self.tth, sample, instrument)
        with self.subTest("x-ray tth-footprint-gauss-offset"):
            instrument.footype = GaussianBeamOffset(sigma=0.5, offset=0.1)
            OffSpecularMingInterdiff(self.tth, self.tth, sample, instrument)
        with self.subTest("x-ray tth-footprint-square-offset"):
            instrument.footype = SquareBeamOffset(width=0.1, offset=0.05)
            OffSpecularMingInterdiff(self.tth, self.tth, sample, instrument)
        with self.subTest("x-ray tth-footprint-trapezoid"):
            instrument.footype = TrapezoidBeam(inner_width=0.1, outer_width=0.2)
            OffSpecularMingInterdiff(self.tth, self.tth, sample, instrument)
            instrument.footype = TrapezoidBeam(inner_width=0.1, outer_width=0.1)
            OffSpecularMingInterdiff(self.tth, self.tth, sample, instrument)
            instrument.footype = TrapezoidBeam(inner_width=0.0, outer_width=0.2)
            OffSpecularMingInterdiff(self.tth, self.tth, sample, instrument)

    def test_sld(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, f=3e-5 + 1e-7j, dens=0.1)])],
            Ambient=Layer(),
            Substrate=Layer(f=5e-5 + 2e-7j, dens=0.1),
        )
        instrument = Instrument(
            coords=Coords.tth,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            wavelength=4.5,
        )
        with self.subTest("sld xray"):
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
        Instrument(coords=Coords.q, wavelength=1.54, footype=FootType.none, restype=ResType.none),
    )
