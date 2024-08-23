"""
Library for combined x-ray and neutrons simulations for inhomogeneous samples.
==============================================================================
In addition to the options from spec_nx this allows to model thickness variation over
the sample surface and gradients in the roughness and thickness over a repetition of layers.

Classes
-------
"""

from copy import deepcopy
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
from .lib.sm_hayter_mook import sm_layers
from .lib.testing import ModelTestCase
from .spec_nx import Coords, FootType, Instrument
from .spec_nx import Layer as NXLayer
from .spec_nx import LayerParameters as NXLayerParameters
from .spec_nx import Polarization, Probe, ResType, q_limit

# Preamble to define the parameters needed for the models outlined below:
ModelID = "SpecInhom"

Buffer = spec_nx.Buffer

__xlabel__ = "q [Å$^{-1}$]"
__ylabel__ = "Instnsity [a.u.]"

@dataclass
class Layer(NXLayer):
    """
    Representing a layer in the sample structur.

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
    ``sigma_gradient``
       Increase of roughness of this layer from bottom to top in stack repetitions
    ``d_gradient``
       Increase of the thickness of this layer from bottom to top in stack repetitions
    """

    sigma_gradient: float = 0.0
    d_gradient: float = 0.0

    Units = NXLayer.Units | {"sigma_gradient": "1", "d_gradient": "1"}

    Groups = [
        ("General", ["d", "dens", "sigma"]),
        ("Neutron", ["b", "xs_ai", "magn", "magn_ang"]),
        ("Inhom.", ["sigma_gradient", "d_gradient"]),
        ("X-Ray", ["f"]),
    ]


@dataclass
class LayerParameters(NXLayerParameters):
    sigma_gradient: List[float]
    d_gradient: List[float]


class SigmaGradientType(AltStrEnum):
    rel_lin = "relative linear"
    rel_sqrt = "relative sqrt"
    abs_lin = "absolute linear"
    abs_sqrt = "absolute sqrt"


@dataclass
class Stack(refl.StackBase):
    """
    A collection of Layer objects that can be repeated.

    ``Layers``
       A ``list`` consiting of ``Layer``\ s in the stack the first item is
       the layer closest to the bottom
    ``Repetitions``
       The number of repsetions of the stack or m-value for analytical super-mirror model.
    ``sigma_gtype``
       model for this increase in roughness (relative linlinea, relative sqrt, absolute linear, absolute sqrt)
    ``sigma_gradient``
       amount of increase in roughness from bottom to top applied to all layers in the stack
    ``d_gradient``
       thickness increase from bottom to top
    ``dens_gradient``
       density increase from bottom to top
    ``beta_sm``
       Parameter used to model neutron super-mirror coatings. There are two models implemented.
       If *beta_sm* is positive, a simplified analystical model from J. Schelten and K. Mika (Nuc. Inst. Metho. 160 (1979))
       is used and the parameter represents the quality paramter for super-mirror sequences.
       A super-mirror sequence is generated and Repetitions is interpreted as the m-value for
       the sequence to calculate actual repetitions automatically.

       If *beta_sm* is negative it is interpretated as the zeta parameter of an iterative build-up of super-mirror
       layers using the method described by J.B. Hayter and H.A. Mook (J. Appl. Cryst. (1989), 22, 35-41). Here,
       the actual given number of repetitions are used to build up a super-mirror from the critical edge
       upowards. The m-value follows from the number of repetitions, material parameters and zeta.

       This method only works acuratly with two layers in the Stack, as the maximum and
       minimum SLD layers are used for the SLD calculation needed to get the layer sequences.
       (Other layers are simulated but their SLDs are ignored for the
       number of repetitions and super-lattice repetition period).

       For the Schelten/Mika method, all layers are scaled to the relative size needed to produce the
       right super-lattice period. In Hayter/Mook only the two min/max components are set to the
       optimal thickness values while the rest of the stack stays with constant width.

       The value of d_gradient is interpreted as a relative change from bottom to top in this case. Together
       with the *sm_scale*, that is applied to all layers.
    ``sm_scale``
       A scaling parameter applied to all thicknesses in a supermirror. This can account for thickness
       differences introduced by the manufacturing process due to imperfect calibration.
    """

    Layers: List[Layer] = field(default_factory=list)
    Repetitions: float = 1.0
    sigma_gtype: SigmaGradientType = 0  # model for this increas (rel lin, rel sqrt, abs lin, abs sqrt)
    sigma_gradient: float = 0.0  # amount of increase in roughness from bottom to top
    d_gradient: float = 0.0  # change in thickness bottom to top
    dens_gradient: float = 0.0  # change in density bottom to top
    beta_sm: float = 0.0  # function applied to change thickness (linear,  super-mirror (d=beta SM))
    sm_scale: float = 1.0  # scaling factor for thicknesses in super-mirror

    Groups = [
        ("General", ["Repetitions"]),
        ("Roughness Gradient", ["sigma_gtype", "sigma_gradient"]),
        ("Supermirror", ["beta_sm", "sm_scale"]),
        ("Other Gradients", ["d_gradient", "dens_gradient"]),
    ]

    def resolveLayerParameter(self, name):
        """
        Add gradient for sigma and thickness to multilayers
        """
        if self.beta_sm > 0:
            # user has selected super-mirror coating sequence
            m = self.Repetitions
            beta = self.beta_sm
            rho_all = [li.dens * (li.b.real + li.magn) * 1e-5 for li in self.Layers]
            rhoA = max(rho_all)
            rhoB = min(rho_all)
            # critical wave vector values (in paper this includes an addition factor of 1/2)
            QcA = 4.0 * pi * sqrt(rhoA / pi + 0j)
            QcB = 4.0 * pi * sqrt(rhoB / pi + 0j)
            Qcm = 0.022 * m  # critical q for given m-value
            g = Qcm / QcA
            QAB = QcA / sqrt(QcA**2 - QcB**2)
            b = beta * QAB**4 + 2.0 * beta * QAB**2 - 1
            N = int(abs(beta * ((g**2 * QAB**2 + 1) ** 2 - 1) - b))
            if N > 100000:
                raise ValueError(
                    "the number of bi-layers calculated is greater than 100 000, please check beta_sm parameter"
                )
        elif self.beta_sm < 0:
            # super-mirror from Hayter and Mook iterative method was selected, fixed number of layers
            if abs(self.beta_sm) > 1.0 or abs(self.beta_sm) < 0.6:
                raise ValueError("negative beta_sm should be between -1 and -0.6")
            N = int(abs(self.Repetitions))
        else:
            N = int(self.Repetitions)
        if name == "sigma":
            sigma_gradient = self.sigma_gradient
            # parameters for layers with roughness gradient
            par = [li.sigma for li in self.Layers]
            for i in range(1, N):
                if self.sigma_gtype == SigmaGradientType.rel_lin:
                    # linear increase of roughness to (1+sigma_gradient) times bottom roughness)
                    par += [li.sigma * (1.0 + (sigma_gradient + li.sigma_gradient) * i / (N - 1)) for li in self.Layers]
                elif self.sigma_gtype == SigmaGradientType.rel_sqrt:
                    # add roughness using rms
                    par += [
                        li.sigma * sqrt(1.0 + ((sigma_gradient + li.sigma_gradient) * i / (N - 1)) ** 2)
                        for li in self.Layers
                    ]
                elif self.sigma_gtype == SigmaGradientType.abs_lin:
                    # linear increase of roughness to bottom roughness + sigma_gradient)
                    par += [li.sigma + (sigma_gradient + li.sigma_gradient) * i / (N - 1) for li in self.Layers]
                elif self.sigma_gtype == SigmaGradientType.abs_sqrt:
                    # add roughness using rms
                    par += [
                        sqrt(li.sigma**2 + ((sigma_gradient + li.sigma_gradient) * i / (N - 1)) ** 2)
                        for li in self.Layers
                    ]
                else:
                    raise NotImplementedError("sigma_gtype must be in SigmaGradientType Enum")
        elif name == "d":
            d_gradient = self.d_gradient
            if self.beta_sm > 0:
                # layer thickness sequence based on calculated bi-layer thicknesses for super-mirror
                L_thicknesses = [li.d + 0.0 for li in self.Layers]
                D_start = sum(L_thicknesses)
                rel_thickness = [di / D_start for di in L_thicknesses]
                sm_scale = self.sm_scale
                idx = arange(N)
                D_SL = (
                    abs((2.0 * QAB * pi / QcA) / sqrt(sqrt(1 + (N - idx + b) / beta) - 1))
                    * (1.0 + d_gradient * idx / (N - 1))
                    * sm_scale
                )
                par = []
                for i in range(N):
                    par += [di * D_SL[i] for di in rel_thickness]
            elif self.beta_sm < 0:
                zeta = -self.beta_sm
                rho_all = [li.dens * (li.b.real + li.magn) * 1e-5 for li in self.Layers]
                rho1 = max(rho_all)  # in paper, SLD is called alpha
                rho2 = min(rho_all)
                # if two layers are chosen, use scale from olgorithm, else only change main 2 layers size
                L_thicknesses = [li.d + 0.0 for li in self.Layers]
                D_start = sum(L_thicknesses)
                idx1, idx2 = rho_all.index(rho1), rho_all.index(rho2)
                rest_size = D_start - L_thicknesses[idx1] - L_thicknesses[idx2]
                # layer thicknesses according to Hayter+Mook algorithm
                D_SL = sm_layers(rho1, rho2, N, zeta)
                # sign on repetitions defines direction of multilayer
                if self.Repetitions > 0:
                    D_SL.reverse()
                par = []
                sm_scale = self.sm_scale
                for i, (d1, d2) in enumerate(D_SL):
                    # variation from user parameters
                    var_scale = (1.0 + d_gradient * i / (N - 1)) * sm_scale
                    # scale the size of main two layer but remove residual layer thicknesses
                    total_bilayer = d1 + d2
                    scaled_bilayer = (total_bilayer - rest_size) / total_bilayer
                    for i, di in enumerate(L_thicknesses):
                        if i == idx1:
                            par.append(d1 * scaled_bilayer * var_scale)
                        elif i == idx2:
                            par.append(d2 * scaled_bilayer * var_scale)
                        else:
                            par.append(di * var_scale)
            else:
                # parameters for layers with thickness gradient
                par = []
                for i in range(N):
                    par += [
                        li.d * (1.0 - (d_gradient + li.d_gradient) * (1.0 / 2.0 - float(i) / N)) for li in self.Layers
                    ]
        elif name == "dens":
            dens_gradient = self.dens_gradient
            # parameters for layers with roughness gradient
            par = []
            for i in range(N):
                par += [
                    li.dens * (1.0 - dens_gradient * (1.0 / 2.0 - float(i) / self.Repetitions)) for li in self.Layers
                ]
        else:
            par = [getattr(li, name) for li in self.Layers] * N
        return par


class InhomType(AltStrEnum):
    gauss = "gauss"
    semi_gauss = "semi-gauss"
    empiric = "empiric PLD"


@dataclass
class Sample(refl.SampleBase):
    """
    Describe global sample by listing ambient, substrate and layer parameters.

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
    ``sigma_inhom``
       Width of the thickness distribution
    ``lscale_inhom``
       For empirical PLD model this defines the width of the increas peak
    ``flatwidth_inhom``
       For empirical PLD model this defines the trnsition from flat to increased probability.
    ``steps_inhom``
       Number of simulations to be performed to sample the thickness distribution model
    ``type_inhom``
       Function for the thickness probability. Either symmetric gaussian, semi-gaussian which does not have any probability
       for a higher thickness and an empirical model developed from PLD samples.
    ``crop_sld``
       Useful for multilayers with a very large number of repetitions. Only keeps crop_sld number of layers
       on top and bottom and removes all in the center for the SLD plot. The removed layers are replace with
       one layer of empty space. If the parameter is negative this gap can be replaced by a "peak" that
       separates top and bottom, this does invalidate the x-axis of the SLD plot, though.

    The empricial PLD model is base of simulations from x-ray line focus plume shapes to get thickness
    distribution probabilities and reduced to a smaller number of parameters.
    See PhD thesis *Multiferroicity in oxide thin films and heterostructures, A.Glavic, RWTH Aachen, (2012)* for a
    more detailed description of the empirical PLD model and an example application.
    """

    Stacks: List[Stack] = field(default_factory=list)
    Ambient: Layer = field(default_factory=Layer)
    Substrate: Layer = field(default_factory=Layer)
    crop_sld: int = 200
    sigma_inhom: float = 0.0
    lscale_inhom: float = 0.9
    flatwidth_inhom: float = 0.3
    steps_inhom: int = 20
    type_inhom: InhomType = "empiric PLD"

    _layer_parameter_class = LayerParameters


def Specular(TwoThetaQz, sample: Sample, instrument: Instrument):
    """
    The model function. Averadging the intensities for different
    layer thicknesses as found for e.g. large PLD samples.
    """
    global __xlabel__

    Q, TwoThetaQz, weight = spec_nx.resolution_init(TwoThetaQz, instrument)
    # often an issue with resolution etc. so just replace Q values < q_limit
    # if any(Q < q_limit):
    #    raise ValueError('The q vector has to be above %.1e, please verify all your x-axis points fulfill this criterion, including possible resolution smearing.'%q_limit)
    Q = np.maximum(Q, q_limit)

    # for better performance, perform all simulations without resolution convolution or footprint correction
    restype = instrument.restype
    foottype = instrument.footype
    Ibkg = instrument.Ibkg
    instrument.restype = ResType("no conv")
    instrument.footype = FootType("no corr")
    instrument.Ibkg = 0.0
    sampcall = deepcopy(sample)

    # average thicknesses before inhomogeniety average
    d0 = [array([Layer.d for Layer in Stack.Layers]) for Stack in sample.Stacks]
    sigma_d = sample.sigma_inhom * 0.01  # Inhomogeniety in \% (gamma for type 2)
    lorentz_scale = sample.lscale_inhom
    flat_width = np.maximum(1e-4, sample.flatwidth_inhom * 0.01)
    type_inhom = sample.type_inhom
    # Define the thicknesses to calculate and their propability
    if sigma_d == 0:  # no inhomogeniety
        d_fact = array([1.0])
        P = array([1.0])
    elif type_inhom == InhomType.semi_gauss:  # half gaussian shape inhomogeniety
        d_fact = 1.0 + linspace(-2.0 * sigma_d, 0, sample.steps_inhom)
        P = exp(-0.5 * (d_fact - sigma_d - 1.0) ** 2 / sigma_d**2)
        P /= P.sum()
        mean_d = (P * d_fact).sum()
        d_fact += 1.0 - mean_d
    elif type_inhom == InhomType.empiric:  # inhomogeniety of a PLD line focus, better approximation
        d_fact = 1.0 + linspace(-1.0 * max(2.0 * sigma_d, flat_width), 0, sample.steps_inhom)
        Pg = where(d_fact > flat_width, lorentz_scale * 1.0 / (1.0 + ((d_fact - 1.0) / sigma_d) ** 2), 0.0)
        Pf = (1.0 - lorentz_scale) * where(d_fact > flat_width, 1.0, 0.0)
        P = Pg + Pf
        P /= P.sum()
        mean_d = (P * d_fact).sum()
        d_fact += 1.0 - mean_d
    else:  # gaussian inhomegeniety
        d_fact = 1.0 + linspace(-sigma_d, sigma_d, sample.steps_inhom)
        P = exp(-0.5 * (d_fact - 1.0) ** 2 / sigma_d**2)
        P /= P.sum()
    # list for reflectivities to average
    Rlist = []
    # Iterate over thicknesses
    for d_facti, Pi in zip(d_fact, P):
        di = [d_facti * d0i for d0i in d0]
        for i, Stack in enumerate(sampcall.Stacks):
            for j, Layer in enumerate(Stack.Layers):
                # Layer.setD(di[i][j])
                Layer.d = di[i][j]
        Rlist.append(Pi * spec_nx.Specular(TwoThetaQz, sampcall, instrument))
    R = array(Rlist).sum(axis=0)

    # apply resolution convolution and footprint correction to result
    instrument.restype = restype
    instrument.footype = foottype
    instrument.Ibkg = Ibkg

    # footprint correction
    foocor = spec_nx.footprintcorr(Q, instrument)
    # resolution correction
    R = spec_nx.resolutioncorr(R, TwoThetaQz, foocor, instrument, weight)

    __xlabel__ = spec_nx.__xlabel__
    return R + instrument.Ibkg



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
    du = ud
    instrument.pol = inst_pol

    P = get_pol_matrix(p1, p2, F1, F2)
    Pline = P[spec_nx.POL_CHANNELS.index(instrument.pol)]
    I = Pline[:, newaxis] * np.vstack([uu, ud, du, dd])
    return I.sum(axis=0)


SLD_calculations = spec_nx.SLD_calculations

SimulationFunctions = {"Specular": Specular, "PolSpecular": PolSpecular, "SLD": SLD_calculations}

Sample.setSimulationFunctions(SimulationFunctions)


class TestSpecInhom(ModelTestCase):
    # TODO: currently this only checks for raise conditions in the code above, check of results should be added

    def test_spec_xray(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, f=3e-5 + 1e-7j, dens=0.1)])],
            Ambient=Layer(),
            Substrate=Layer(f=5e-5 + 2e-7j, dens=0.1),
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
        with self.subTest("neutron-polspec q"):
            instrument.pol = Polarization.up_up
            PolSpecular(self.qz, 0.01, 0.01, 0.01, 0.01, sample, instrument)

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

    def test_inhom_types(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, f=3e-5 + 1e-7j, dens=0.1)])],
            Ambient=Layer(),
            Substrate=Layer(f=5e-5 + 2e-7j, dens=0.1),
            sigma_inhom=0.5,
            type_inhom=InhomType.gauss,
        )
        instrument = Instrument(
            probe=Probe.xray,
            coords=Coords.q,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            tthoff=0.0,
            wavelength=1.54,
        )
        with self.subTest("inhom gauss"):
            Specular(self.qz, sample, instrument)
        with self.subTest("inhom semi-gauss"):
            sample.type_inhom = InhomType.semi_gauss
            Specular(self.qz, sample, instrument)
        with self.subTest("inhom empiric"):
            sample.type_inhom = InhomType.empiric
            Specular(self.qz, sample, instrument)

    def test_gradients(self):
        sample = Sample(
            Stacks=[Stack(Layers=[Layer(d=150, sigma=2.0, f=3e-5 + 1e-7j, dens=0.1)], Repetitions=10)],
            Ambient=Layer(),
            Substrate=Layer(f=5e-5 + 2e-7j, dens=0.1),
            sigma_inhom=0.5,
            type_inhom=InhomType.gauss,
        )
        instrument = Instrument(
            probe=Probe.xray,
            coords=Coords.q,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            tthoff=0.0,
            wavelength=1.54,
        )
        with self.subTest("stack d gradient"):
            sample.Stacks[0].d_gradient = 1.0
            Specular(self.qz, sample, instrument)
        with self.subTest("stack sigma gradient abs_lin"):
            sample.Stacks[0].d_gradient = 0.0
            sample.Stacks[0].sigma_gradient = 5.0
            sample.Stacks[0].sigma_gtype = SigmaGradientType.abs_lin
            Specular(self.qz, sample, instrument)
        with self.subTest("stack sigma gradient abs_sqrt"):
            sample.Stacks[0].sigma_gradient = 5.0
            sample.Stacks[0].sigma_gtype = SigmaGradientType.abs_sqrt
            Specular(self.qz, sample, instrument)
        with self.subTest("stack sigma gradient rel_lin"):
            sample.Stacks[0].sigma_gradient = 0.5
            sample.Stacks[0].sigma_gtype = SigmaGradientType.rel_lin
            Specular(self.qz, sample, instrument)
        with self.subTest("stack sigma gradient rel_sqrt"):
            sample.Stacks[0].sigma_gradient = 0.5
            sample.Stacks[0].sigma_gtype = SigmaGradientType.rel_sqrt
            Specular(self.qz, sample, instrument)

    def test_supermirrors(self):
        from .utils import bc
        sample = Sample(
            Stacks=[
                Stack(
                    Layers=[
                        Layer(d=50, sigma=2.0, b=bc.Ni, dens=8.908*0.602214/58.6934),
                        Layer(d=50, sigma=2.0, b=bc.Ti, dens=4.506*0.602214/47.867),
                    ],
                    Repetitions=2.0,
                )
            ],
            Ambient=Layer(),
            Substrate=Layer(b=bc.Si, dens=0.04995982812477898),
        )
        instrument = Instrument(
            probe=Probe.neutron,
            coords=Coords.q,
            res=0.001,
            restype=ResType.none,
            beamw=0.1,
            footype=FootType.none,
            tthoff=0.0,
            wavelength=1.54,
        )
        with self.subTest("analytic supermirror"):
            sample.Stacks[0].beta_sm = 0.75
            sample.Stacks[0].Repetitions = 2.5 # m-value
            Specular(self.qz, sample, instrument)
        with self.subTest("analytic supermirror"):
            sample.Stacks[0].beta_sm = -0.99
            sample.Stacks[0].Repetitions = 2000  # number layers
            Specular(self.qz, sample, instrument)

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
