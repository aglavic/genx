"""
Library for specular and off-specular x-ray reflectivity
========================================================
implements the interdiff model but using BornAgain as simulation library.

Classes
-------

"""

from .interdiff import *
import bornagain as ba
ba.require_versions("bornagain>=25,<26")


def get_ba_sample(sample: Sample, lamda: float):
    parameters: LayerParameters = sample.resolveLayerParameters()
    dens = array(parameters.dens, dtype=float64)
    f = array(parameters.f, dtype=complex128)
    sldfac = lamda**2 / 2 / pi
    dib = r_e * dens * sldfac * f # delta - i beta
    d = array(parameters.d, dtype=float64)

    sigma = array(parameters.sigma, dtype=float64)

    BA_sample = ba.Sample()
    transient = ba.ErfTransient()
    crosscorr = ba.CommonDepthCrosscorrelation(sample.eta_z*ba.angstrom)
    # iterate over all layers, BornAgain starts with the substrate
    for i, (dibi, di, sigmai) in reversed(tuple(enumerate(
            zip(dib,d,sigma)))):
        if i==(len(d)-1):
            BA_sample.addLayer(
                # ba.Layer(ba.RefractiveMaterial(f"Layer_{i}", (1.0,0.,0.), dibi.real, -dibi.imag)))
                ba.Layer(ba.SLDMaterial(f"Layer_{i}", (1.0,0.,0.), dibi.real/sldfac, -dibi.imag/sldfac)))
        elif i==0:
            autocorr = ba.SelfAffineFractalModel(sigmai*ba.angstrom, sample.h, sample.eta_x*ba.angstrom, 0.02)
            roughness = ba.Roughness(autocorr, transient, crosscorr)
            BA_sample.addLayer(
                # ba.Layer(ba.RefractiveMaterial(f"Layer_{i}", (1.0,0.,0.), dibi.real, -dibi.imag), roughness))
                ba.Layer(ba.SLDMaterial(f"Layer_{i}", (1.0,0.,0.), dibi.real/sldfac, -dibi.imag/sldfac), roughness))
        else:
            autocorr = ba.SelfAffineFractalModel(sigmai*ba.angstrom, sample.h, sample.eta_x*ba.angstrom, 0.02)
            roughness = ba.Roughness(autocorr, transient, crosscorr)
            # BA_sample.addLayer(ba.Layer(ba.RefractiveMaterial(f"Layer_{i}", (1.0,0.,0.), dibi.real, -dibi.imag), di*ba.angstrom, roughness))
            BA_sample.addLayer(ba.Layer(ba.SLDMaterial(f"Layer_{i}", (1.0,0.,0.), dibi.real/sldfac, -dibi.imag/sldfac), di*ba.angstrom, roughness))

    return BA_sample

def SpecularBA(TwoThetaQz, sample: Sample, instrument: Instrument):
    """Simulate the specular signal from sample when proped with instrument

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """
    # preamble to get it working with my class interface
    global __xlabel__
    __xlabel__ = "q [Å$^{-1}$]"

    if instrument.coords == Coords.tth:
        theta = TwoThetaQz / 2
        __xlabel__ = "2θ [°]"
    elif instrument.coords == Coords.q:
        theta = arcsin(TwoThetaQz / 4 / pi * instrument.wavelength) * 180.0 / pi
    else:
        raise ValueError("Invalid coordinate system, instrument.coords must be tth or q")

    lamda = instrument.wavelength
    BA_sample = get_ba_sample(sample, lamda)

    theta_sorted, unsort = np.unique(theta, return_inverse=True)

    alpha_list = ba.ListScan("alpha_i (rad)", (theta_sorted*ba.deg).tolist())  # all points
    BA_scan = ba.AlphaScan(alpha_list)
    BA_scan.setWavelength(lamda*ba.angstrom)
    BA_scan.setIntensity(instrument.I0)

    footype = instrument.footype
    bw_p_sl = instrument.beamw/instrument.samplelen
    if footype == FootType.gauss:
        BA_scan.setFootprint(ba.FootprintGauss(bw_p_sl*2.))
    elif footype == FootType.square:
        BA_scan.setFootprint(ba.FootprintSquare(bw_p_sl))
    elif footype == FootType.none:
        pass
    else:
        raise ValueError("The choice of footprint correction, footype," "is WRONG")

    restype = instrument.restype
    if restype in [ResType.fast_conv, ResType.fast_conv_var]:
        raise ValueError("Fast convolution not supported for BornAgain model")
    elif restype == ResType.full_conv_abs or isinstance(restype, resolution_module.Resolution):
        alpha_distr = ba.DistributionGaussian(0, instrument.res/2.0*ba.deg, int(instrument.respoints), instrument.resintrange)
        BA_scan.setGrazingAngleDistribution(alpha_distr)
    elif restype == ResType.none:
        pass
    else:
        raise ValueError("The choice of resolution type, restype," "is WRONG")


    BA_sim = ba.SpecularSimulation(BA_scan, BA_sample)
    BA_sim.setBackground(ba.ConstantBackground(instrument.Ibkg))
    ####--- BornAgain part ---####

    R = BA_sim.simulate().intensities()[unsort]
    return R*instrument.I0


def SLD_calculation_BA(z, item, sample: Sample, inst: Instrument):
    """Calculates the scatteringlength density as at the positions z

    # BEGIN Parameters
    z data.x
    item "Re"
    # END Parameters
    """
    lamda = inst.wavelength
    BA_sample = get_ba_sample(sample, lamda)
    parameters: LayerParameters = sample.resolveLayerParameters()
    d = array(parameters.d, dtype=float64)
    int_pos = cumsum(r_[0, d])
    sigma = array(parameters.sigma, dtype=float64)[:-1] + 1e-7

    if z is None:
        z = arange(-sigma[0] * 5, int_pos.max() + sigma[-1] * 5, 0.5)
    rho = ba.materialProfileSLD(BA_sample, z.shape[0], -z[-1]*ba.angstrom, -z[0]*ba.angstrom)/r_e
    dic = {"Re": real(rho), "Im": imag(rho), "z": z, "SLD unit": "r_{e}/\\AA^{3}"}
    if item is None or item == "all":
        return dic
    else:
        try:
            return dic[item]
        except:
            raise ValueError("The chosen item, %s, does not exist" % item)

USE_GISAS_DETECTOR = False

def OffSpecularMingInterdiffBA(TwoThetaQz, ThetaQx, sample: Sample, instrument: Instrument):
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
        if (betaR1 > 0).all() or (betaR1 < 0).all():
            longitudinal = True
        else:
            longitudinal = False
            cen_idx = where(betaR1 >= 0)[0][0]
        qx = k * (cos(alphaR1 * pi / 180) - cos(betaR1 * pi / 180))
        qz = k * (sin(alphaR1 * pi / 180) + sin(betaR1 * pi / 180))
        tth = TwoThetaQz
    else:
        qz = TwoThetaQz
        qx = ThetaQx
        if (qx > 0).all() or (qx < 0).all():
            longitudinal = True
        else:
            longitudinal = False
            cen_idx = where(qx >= 0)[0][0]
        tth = arcsin(qz / 4.0 / pi * lamda) * 2.0
        alphaR1 = (arctan2(qx, qz) + tth / 2.0) * 180.0 / pi
        betaR1 = tth - alphaR1

    lamda = instrument.wavelength
    BA_sample = get_ba_sample(sample, lamda)

    alpha_i, unsort = np.unique(np.maximum(alphaR1, instrument.res*instrument.resintrange), return_inverse=True)


    footype = instrument.footype
    bw_p_sl = instrument.beamw/instrument.samplelen
    if footype == FootType.gauss:
        fp = ba.FootprintGauss(bw_p_sl)
    elif footype == FootType.square:
        fp = ba.FootprintSquare(bw_p_sl)
    elif footype == FootType.none:
        fp = None
    else:
        raise ValueError("The choice of footprint correction, footype," "is WRONG")

    restype = instrument.restype
    if restype in [ResType.fast_conv, ResType.fast_conv_var]:
        raise ValueError("Fast convolution not supported for BornAgain model")
    elif restype == ResType.full_conv_abs or isinstance(restype, resolution_module.Resolution):
        alpha_distr = ba.DistributionGaussian(0, instrument.res/2.0*ba.deg, int(instrument.respoints), instrument.resintrange)
    elif restype == ResType.none:
        pass
    else:
        raise ValueError("The choice of resolution type, restype," "is WRONG")

    I=[]
    #detres_HWHM = instrument.detres*np.sqrt(2*np.log(2))
    resscale = np.exp(-0.5*np.arange(-3., 3.5, 0.5)**2)
    for ai, af in zip(np.maximum(alphaR1, instrument.res*instrument.resintrange), betaR1):
        alpha_f_min = af-instrument.detres*3.25 # alpha_f range to have pixel centers at -2 to 2 sigma
        alpha_f_max = af+instrument.detres*3.25
        if USE_GISAS_DETECTOR:
            BA_det = ba.SphericalDetector(1, -1.0*ba.deg, 1.0*ba.deg, 13, alpha_f_min*ba.deg, alpha_f_max*ba.deg)
            BA_beam = ba.Beam(instrument.I0, lamda*ba.angstrom, ai*ba.deg)
            if fp is not None:
                BA_beam.setFootprint(fp)
            BA_sim = ba.ScatteringSimulation(BA_beam, BA_sample, BA_det)
            if restype != ResType.none:
                BA_sim.addParameterDistribution(ba.ParameterDistribution.BeamGrazingAngle,
                    ba.DistributionGaussian(ai*ba.deg, instrument.res/2.0*ba.deg,
                                            int(instrument.respoints), instrument.resintrange))
            BA_sim.options().setIncludeSpecular(True)
            I.append((BA_sim.simulate().intensities()*resscale[:, newaxis]).sum())
        else:
            BA_det = ba.OffspecDetector(13, alpha_f_min*ba.deg, alpha_f_max*ba.deg, -1*ba.deg, +1*ba.deg)
            alpha_list = ba.ListScan("alpha_i (rad)", [ai*ba.deg])  # all points
            BA_scan = ba.AlphaScan(alpha_list)
            BA_scan.setWavelength(lamda*ba.angstrom)
            BA_scan.setIntensity(instrument.I0)
            if fp is not None:
                BA_scan.setFootprint(fp)
            if restype != ResType.none:
                BA_scan.setGrazingAngleDistribution(alpha_distr)
            BA_sim = ba.OffspecSimulation(BA_scan, BA_sample, BA_det)
            BA_sim.options().setIncludeSpecular(True)
            I.append((BA_sim.simulate().intensities()*resscale[:,newaxis]).sum())
    ####--- BornAgain part ---####
    return np.array(I)+instrument.Ibkg


SimulationFunctions = {"Specular": SpecularBA, "OffSpecular": OffSpecularMingInterdiffBA, "SLD": SLD_calculation_BA}

Sample.setSimulationFunctions(SimulationFunctions)
