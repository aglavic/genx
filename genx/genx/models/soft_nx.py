"""
Library for combined x-ray and neutrons simulations.
====================================================
The neutron simulations is capable of handling non-magnetic, magnetic
non-spin flip as well as neutron spin-flip reflectivity. The model works
with scattering lengths densities directly.

The model is equivalent to spec_nx in calculation but uses different
parameterization that is more suitable for soft-matter applications.

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
from .lib import refl_base as refl
from .lib import resolution as resolution_module
from .lib.base import AltStrEnum
from .lib.footprint import *
from .lib.instrument import *
from .lib.physical_constants import muB_to_SL, r_e
from .lib.resolution import *
from .lib.testing import ModelTestCase
from .spec_nx import (AA_to_eV, Coords, FootType, Instrument, Polarization, Probe, ResType, footprintcorr, q_limit,
                      resolution_init, resolutioncorr)

# Preamble to define the parameters needed for the models outlined below:

ModelID = "SoftNX"

__xlabel__ = "q [Å$^{-1}$]"
__ylabel__ = "Instnsity [a.u.]"


@dataclass
class Layer(refl.ReflBase):
    """
    Representing a layer in the sample structur.

    ``d``
       The thickness of the layer in AA (Angstroms = 1e-10m)
    ``sigma``
       The root mean square roughness of the top interface of the layer in
       Angstroms.
    ``sld_x``
       The x-ray scattering length density in 1e-6 1/AA^2
    ``sld_n``
       The neutron scattering length density in 1e-6 1/AA^2
    ``sld_m``
       The neutron magnetic scattering length density in 1e-6 1/AA^2
    ``magn_ang``
       The angle of the magnetic moment in degress. 0 degrees correspond to
       a moment collinear with the neutron spin.
    """

    sigma: float = 0.0
    d: float = 0.0
    sld_n: complex = 1e-20j
    sld_x: complex = 0j
    sld_m: float = 0.0
    magn_ang: float = 0.0

    Units = {
        "sigma": "AA",
        "d": "AA",
        "sld_n": "1e-6 1/AA^2",
        "sld_x": "1e-6 1/AA^2",
        "sld_m": "1e-6 1/AA^2",
        "magn_ang": "deg.",
    }

    Groups = [("General", ["d", "sigma"]), ("Neutron", ["sld_n", "sld_m", "magn_ang"]), ("X-Ray", ["sld_x"])]


@dataclass
class LayerParameters:
    sigma: List[float]
    d: List[float]
    sld_n: List[complex]
    sld_x: List[complex]
    sld_m: List[float]
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


# A buffer to save previous calculations for spin-flip calculations
class Buffer:
    Ruu = 0
    Rdd = 0
    Rdu = 0
    Rud = 0
    parameters = None
    TwoThetaQz = None


def Specular(TwoThetaQz, sample: Sample, instrument: Instrument):
    """Simulate the specular signal from sample when probed with instrument

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """
    global __xlabel__
    __xlabel__ = "q [Å$^{-1}$]"
    if instrument.coords == Coords.tth:
        __xlabel__ = "2θ [°]"

    # preamble to get it working with my class interface
    restype = instrument.restype
    Q, TwoThetaQz, weight = resolution_init(TwoThetaQz, instrument)
    # often an issue with resolution etc. so just replace Q values < q_limit
    Q = maximum(Q, q_limit)

    ptype = instrument.probe
    pol = instrument.pol

    parameters: LayerParameters = sample.resolveLayerParameters()

    if ptype == Probe.xray:
        # fb = array(parameters['f'], dtype = complex64)
        e = AA_to_eV / instrument.wavelength
        sld = refl.cast_to_array(parameters.sld_x, e) * 1e-6
    else:
        sld = array(parameters.sld_n, dtype=complex128) * 1e-6

    d = array(parameters.d, dtype=float64)
    sld_m = array(parameters.sld_m, dtype=float64) * 1e-6
    # Transform to radians
    magn_ang = array(parameters.magn_ang, dtype=float64) * pi / 180.0

    sigma = array(parameters.sigma, dtype=float64)

    l2pi = instrument.wavelength**2 / 2 / 3.141592
    # Ordinary Paratt X-rays
    if ptype == Probe.xray:
        R = Paratt.ReflQ(Q, instrument.wavelength, 1.0 - l2pi * sld, d, sigma)
    # Ordinary Paratt Neutrons
    elif ptype == Probe.neutron:
        R = Paratt.ReflQ(Q, instrument.wavelength, 1.0 - l2pi * sld, d, sigma)
    # Ordinary Paratt but with magnetization
    elif ptype == Probe.npol:
        # Polarization uu or ++
        if pol == Polarization.up_up:
            R = Paratt.ReflQ(Q, instrument.wavelength, 1.0 - l2pi * (sld + sld_m), d, sigma)
        # Polarization dd or --
        elif pol == Polarization.down_down:
            R = Paratt.ReflQ(Q, instrument.wavelength, 1.0 - l2pi * (sld - sld_m), d, sigma)
        elif pol == Polarization.asymmetry:
            Rp = Paratt.ReflQ(Q, instrument.wavelength, 1.0 - l2pi * (sld - sld_m), d, sigma)
            Rm = Paratt.ReflQ(Q, instrument.wavelength, 1.0 - l2pi * (sld + sld_m), d, sigma)
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
            if sld_m[-1] != 0.0 or sld[-1] != 0:
                sld_m -= sld_m[-1]
                sld -= sld[-1]
            sld_p = l2pi * (sld + sld_m)
            sld_m = l2pi * (sld - sld_m)
            Vp = (2 * pi / instrument.wavelength) ** 2 * (
                sld_p * (2.0 + sld_p)
            )  # (1-np**2) - better numerical accuracy
            Vm = (2 * pi / instrument.wavelength) ** 2 * (sld_m * (2.0 + sld_m))  # (1-nm**2)
            (Ruu, Rdd, Rud, Rdu) = MatrixNeutron.Refl(Q, Vp, Vm, d, magn_ang, sigma)
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
    # TODO: Check the following to cases
    # tof
    elif ptype == Probe.ntof:
        ai = instrument.incangle
        # if ai is an array, make sure it gets repeated for every resolution point
        if type(ai) is ndarray and restype in [ResType.full_conv_rel, ResType.full_conv_abs]:
            ai = (ai * ones(instrument.respoints)[:, newaxis]).flatten()
        else:
            ai = ai * ones(Q.shape)
        wl = 4 * pi * sin(ai * pi / 180) / Q
        l2pi = wl**2 / 2 / 3.141592
        R = Paratt.Refl_nvary2(ai, wl, 1.0 - l2pi * sld[:, newaxis], d, sigma, return_int=True)
    # tof spin polarized
    elif ptype == Probe.ntofpol:
        wl = 4 * pi * sin(instrument.incangle * pi / 180) / Q
        l2pi = wl**2 / 2 / 3.141592
        msld = sld_m[:, newaxis] * (4 * pi * sin(instrument.incangle * pi / 180) / Q) ** 2 / 2 / pi
        # polarization uu or ++
        if pol == Polarization.up_up:
            R = Paratt.Refl_nvary2(
                instrument.incangle * ones(Q.shape),
                (4 * pi * sin(instrument.incangle * pi / 180) / Q),
                1.0 - l2pi * (sld[:, newaxis] + msld),
                d,
                sigma,
                return_int=True,
            )
        # polarization dd or --
        elif pol == Polarization.down_down:
            R = Paratt.Refl_nvary2(
                instrument.incangle * ones(Q.shape),
                (4 * pi * sin(instrument.incangle * pi / 180) / Q),
                1.0 - l2pi * (sld[:, newaxis] - msld),
                d,
                sigma,
                return_int=True,
            )
        # Calculating the asymmetry
        elif pol == Polarization.asymmetry:
            Rd = Paratt.Refl_nvary2(
                instrument.incangle * ones(Q.shape),
                (4 * pi * sin(instrument.incangle * pi / 180) / Q),
                1.0 - l2pi * (sld[:, newaxis] - msld),
                d,
                sigma,
                return_int=True,
            )
            Ru = Paratt.Refl_nvary2(
                instrument.incangle * ones(Q.shape),
                (4 * pi * sin(instrument.incangle * pi / 180) / Q),
                1.0 - l2pi * (sld[:, newaxis] + msld),
                d,
                sigma,
                return_int=True,
            )
            R = (Ru - Rd) / (Ru + Rd)

        else:
            raise ValueError("The value of the polarization is WRONG." " It should be uu(0) or dd(1) or ass")
    else:
        raise ValueError("The choice of probe is WRONG")
    # FootprintCorrections

    foocor = footprintcorr(Q, instrument)
    # Resolution corrections
    R = resolutioncorr(R, TwoThetaQz, foocor, instrument, weight)

    return R * instrument.I0 + instrument.Ibkg


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
    if instrument.probe != Probe.xray:
        raise NotImplementedError("EnergySpecular only implemented for xray, for neutron ToF use Specular.")

    wl = AA_to_eV / Energy
    global __xlabel__
    __xlabel__ = "E [eV]"

    # TTH values given as x
    if instrument.coords == Coords.tth:
        theta = (TwoThetaQz + instrument.tthoff) / 2.0 * ones_like(Energy)
    # Q vector given....
    elif instrument.coords == Coords.q:
        theta = ((arcsin(TwoThetaQz * wl / 4 / pi) * 180.0 / pi) + instrument.tthoff / 2.0) * ones_like(Energy)

    parameters: LayerParameters = sample.resolveLayerParameters()
    sld = refl.cast_to_array(parameters.sld_x, Energy) * 1e-6

    d = array(parameters.d, dtype=float64)
    sigma = array(parameters.sigma, dtype=float64)

    l2pi = wl**2 / 2 / 3.141592
    # Ordinary Paratt X-rays
    R = Paratt.Refl_nvary2(theta, wl, 1.0 - l2pi * sld[:, newaxis], d, sigma)

    # TODO: Fix corrections
    # FootprintCorrections
    # foocor = footprintcorr(Q, instrument)
    # Resolution corrections
    # R = resolutioncorr(R, TwoThetaQz, foocor, instrument, weight)

    return R * instrument.I0 + instrument.Ibkg


def OffSpecular(TwoThetaQz, ThetaQx, sample, instrument):
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
    Otherwise it returns the item as identified by its string.

    # BEGIN Parameters
    z data.x
    item 'Re'
    # END Parameters
    """
    parameters: LayerParameters = sample.resolveLayerParameters()
    # f = array(parameters['f'], dtype = complex64)
    e = AA_to_eV / inst.wavelength
    sld_x = refl.cast_to_array(parameters.sld_x, e)
    sld_n = array(parameters.sld_n, dtype=complex64)
    ptype = inst.probe
    magnetic = False
    mag_sld = 0
    sld_unit = "10^{-6}\AA^{2}"
    if ptype == Probe.xray:
        sld = sld_x
    elif ptype == Probe.neutron:
        sld = sld_n
        sld_unit = "10^{-6}/\AA^{2}"
    else:
        magnetic = True
        sld = sld_n
        sld_m = array(parameters.sld_m, dtype=float64)
        # Transform to radians
        magn_ang = array(parameters.magn_ang, dtype=float64) * pi / 180.0
        mag_sld = sld_m
        mag_sld_x = mag_sld * cos(magn_ang)
        mag_sld_y = mag_sld * sin(magn_ang)
        sld_unit = "10^{-6}/\AA^{2}"

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


SimulationFunctions = {
    "Specular": Specular,
    "OffSpecular": OffSpecular,
    "SLD": SLD_calculations,
    "EnergySpecular": EnergySpecular,
}

Sample.setSimulationFunctions(SimulationFunctions)


class TestSoftNX(ModelTestCase):
    # TODO: currently this only checks for raise conditions in the code above, check of results should be added

    def test_spec_xray(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, sld_x=3e-5 + 1e-7j)])],
            Ambient=Layer(),
            Substrate=Layer(sld_x=5e-5 + 2e-7j),
        )
        instrument = Instrument(
            probe=Probe.xray,
            coords=Coords.tth,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            tthoff=0.0,
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
        with self.subTest("x-ray q-res-fast-rel"):
            instrument.restype = ResType.fast_conv_rel
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray q-res-fast-var"):
            instrument.restype = ResType.fast_conv_var
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray q-res-full"):
            instrument.restype = ResType.full_conv_abs
            Specular(self.qz, sample, instrument)
        with self.subTest("x-ray q-res-full-rel"):
            instrument.restype = ResType.full_conv_rel
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
        with self.subTest("x-ray tth-footprint-wrong"):
            instrument.footype = 123
            with self.assertRaises(ValueError):
                Specular(self.qz, sample, instrument)

    def test_spec_neutron(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, sld_n=3e-6, sld_m=1e-6)])],
            Ambient=Layer(sld_n=1e-7),
            Substrate=Layer(sld_n=4e-6),
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

        with self.subTest("neutron tthoffset"):
            instrument.probe = Probe.neutron
            instrument.tthoff = 0.1
            Specular(self.qz, sample, instrument)
        with self.subTest("neutron-tof tthoffset"):
            instrument.probe = Probe.ntof
            instrument.tthoff = 0.1
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

    def test_energy(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, sld_x=3e-6)])],
            Ambient=Layer(sld_x=1e-7),
            Substrate=Layer(sld_x=4e-6),
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
    scl = 0.1*AA_to_eV * 1.54 / 2 / pi
    return Specular(
        qz,
        Sample(
            Ambient=Layer(),
            Substrate=Layer(d=150.0, sld_x=(1e-5 + 1e-8j)*scl),
            Stacks=[Stack(Layers=[Layer(d=150.0, sld_x=(2e-5 + 2e-8j)*scl)])],
        ),
        Instrument(probe=Probe.xray, coords=Coords.q, wavelength=1.54, footype=FootType.none, restype=ResType.none),
    )
