# -*- coding: utf8 -*-
'''
Library for combined x-ray and neutrons simulations for inhomogeneous samples.
==============================================================================
In addition to the options from spec_nx this allows to model thickness variation over
the sample surface and gradients in the roughness and thickness over a repetition of layers.

Classes
-------

Layer
~~~~~
``Layer(b = 0.0, d = 0.0, f = 0.0+0.0J, dens = 1.0, magn_ang = 0.0, magn = 0.0, sigma = 0.0, sigma_gradient=0.0, d_gradient=0.0)``

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

Stack
~~~~~
``Stack(Layers = [], Repetitions = 1, sigma_gradient=0.0, sigma_gtype=3, d_gradient=0.0, dens_gradient=0.0)``

``Layers``
   A ``list`` consiting of ``Layer``\ s in the stack the first item is
   the layer closest to the bottom
``Repetitions``
   The number of repsetions of the stack
``sigma_gradient``
   amount of increase in roughness from bottom to top applied to all layers in the stack
``sigma_gtype``
   model for this increase in roughness (0: relative linlinea, 1: relative sqrt, 2:absolute linear, 3: absolute sqrt)
``d_gradient``
   thickness increase from bottom to top
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
``dens_gradient``
   density increase from bottom to top

Sample
~~~~~~
``Sample(Stacks = [], Ambient = Layer(), Substrate = Layer(), sigma_inhom=0.0, lscale_inhom=0.9, flatwidth_inhom=0.3, steps_inhom=20, type_inhom='semi-gauss')``

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
from copy import deepcopy
from numpy import *
from scipy.special import wofz
from . import spec_nx
from .lib.sm_hayter_mook import sm_layers
from .spec_nx import refl
from .lib.instrument import *

# Preamble to define the parameters needed for the models outlined below:
ModelID='SpecInhom'
__pars__=spec_nx.__pars__.copy()

instrument_string_choices=spec_nx.instrument_string_choices.copy()
InstrumentParameters=spec_nx.InstrumentParameters.copy()
InstrumentGroups=spec_nx.InstrumentGroups.copy()
InstrumentUnits=spec_nx.InstrumentUnits.copy()

LayerParameters=spec_nx.LayerParameters.copy()
LayerParameters.update({'sigma_gradient': 0.0, 'd_gradient': 0.0})
LayerUnits=spec_nx.LayerUnits.copy()
LayerUnits.update({'sigma_gradient': '1', 'd_gradient': '1'})
LayerGroups=[('Standard', ['f', 'dens', 'd', 'sigma']),
             ('Neutron', ['b', 'xs_ai', 'magn', 'magn_ang']),
             ('Inhom.', ['sigma_gradient', 'd_gradient'])]

StackParameters={'Layers': [], 'Repetitions': 1,
                 'sigma_gradient': 0.,  # amount of increase in roughness from bottom to top
                 'sigma_gtype': 3,  # model for this increas (rel lin, rel sqrt, abs lin, abs sqrt)
                 'd_gradient': 0.,  # change in thickness bottom to top
                 'beta_sm' : 0.0, # function applied to change thickness (linear,  super-mirror (d=beta SM))
                 'sm_scale': 1.0, # scaling factor for thicknesses in super-mirror
                 'dens_gradient': 0.,  # change in density bottom to top
                 }
SampleParameters=spec_nx.SampleParameters.copy()
SampleParameters.update({'sigma_inhom': 0.0, 'lscale_inhom': 0.9, 'flatwidth_inhom': 0.3,
                         'steps_inhom': 20, 'type_inhom': 'empiric PLD', 'crop_sld': 200})
sample_string_choices={
    'type_inhom': ['gauss', 'semi-gauss', 'empiric PLD'],
    }

AA_to_eV=spec_nx.AA_to_eV
q_limit=spec_nx.q_limit
Buffer=spec_nx.Buffer

__xlabel__ = "q [Å$^{-1}$]"
__ylabel__ = "Instnsity [a.u.]"

def Specular(TwoThetaQz, sample, instrument):
    '''
      The model function. Averadging the intensities for different
      layer thicknesses as found for e.g. large PLD samples.
    '''
    global __xlabel__

    Q, TwoThetaQz, weight=spec_nx.resolution_init(TwoThetaQz, instrument)
    if any(Q<q_limit):
        raise ValueError('The q vector has to be above %.1e'%q_limit)
    restype=instrument.getRestype()
    foottype=instrument.getFootype()
    Ibkg=instrument.getIbkg()
    instrument.setRestype('no conv')
    instrument.setFootype('no corr')
    instrument.Ibkg=0.
    sampcall=deepcopy(sample)

    # average thicknesses before inhomogeniety average
    d0=[array([Layer.getD() for Layer in Stack.Layers]) for Stack in sample.Stacks]
    sigma_d=sample.getSigma_inhom()*0.01  # Inhomogeniety in \% (gamma for type 2)
    lorentz_scale=sample.getLscale_inhom()
    flat_width=maximum(1e-4, sample.getFlatwidth_inhom()*0.01)
    type_inhom=sample.getType_inhom()
    # Define the thicknesses to calculate and their propability
    if type_inhom in sample_string_choices['type_inhom']:
        type_inhom=sample_string_choices['type_inhom'].index(type_inhom)
    else:
        type_inhom=0
    if sigma_d==0:  # no inhomogeniety
        d_fact=array([1.])
        P=array([1.])
    elif type_inhom==1:  # half gaussian shape inhomogeniety
        d_fact=1.+linspace(-2.*sigma_d, 0, sample.getSteps_inhom())
        P=exp(-0.5*(d_fact-sigma_d-1.)**2/sigma_d**2)
        P/=P.sum()
        mean_d=(P*d_fact).sum()
        d_fact+=1.-mean_d
    elif type_inhom==2:  # inhomogeniety of a PLD line focus, better approximation
        d_fact=1.+linspace(-1.*max(2.*sigma_d, flat_width), 0, sample.getSteps_inhom())
        Pg=where(d_fact>flat_width, lorentz_scale*1./(1.+((d_fact-1.)/sigma_d)**2), 0.)
        Pf=(1.-lorentz_scale)*where(d_fact>flat_width, 1., 0.)
        P=Pg+Pf
        P/=P.sum()
        mean_d=(P*d_fact).sum()
        d_fact+=1.-mean_d
    else:  # gaussian inhomegeniety
        d_fact=1.+linspace(-sigma_d, sigma_d, sample.getSteps_inhom())
        P=exp(-0.5*(d_fact-1.)**2/sigma_d**2)
        P/=P.sum()
    # list for reflectivities to average
    Rlist=[]
    # Iterate over thicknesses
    for d_facti, Pi in zip(d_fact, P):
        di=[d_facti*d0i for d0i in d0]
        for i, Stack in enumerate(sampcall.Stacks):
            for j, Layer in enumerate(Stack.Layers):
                # Layer.setD(di[i][j])
                Layer.d=di[i][j]
        Rlist.append(Pi*spec_nx.Specular(TwoThetaQz, sampcall, instrument))
    R=array(Rlist).sum(axis=0)

    instrument.setRestype(restype)
    instrument.setFootype(foottype)
    instrument.Ibkg=Ibkg

    # footprint correction
    foocor=spec_nx.footprintcorr(Q, instrument)
    # resolution correction
    R=spec_nx.resolutioncorr(R, TwoThetaQz, foocor, instrument, weight)

    __xlabel__ = spec_nx.__xlabel__
    return R+instrument.getIbkg()

def ResolutionVectorAsymetric(Q, dQ, points, dLambda, asymmetry, range_=3):
    '''
      Resolution vector for a asymmetric wavelength distribution found in
      neutron experiments with multilayer monochromator.
    '''
    Qrange=max(range_*dQ, range_*dLambda*Q.max())
    Qstep=2*Qrange/points
    Qres=Q+(arange(points)-(points-1)/2)[:, newaxis]*Qstep
    Quse=transpose(Q[:, newaxis])

    gamma_asym=2.*dLambda*Quse/(1+exp(asymmetry*(Quse-Qres)))
    z=(Quse-Qres+(abs(gamma_asym)*1j))/abs(dQ)/sqrt(2.)
    z0=(0.+(abs(gamma_asym)*1j))/abs(dQ)/sqrt(2)
    weight=wofz(z).real/wofz(z0).real
    Qret=Qres.flatten()
    return Qret, weight

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
    I = Pline[:, newaxis] * vstack([uu, ud, du, dd])
    return I.sum(axis=0)

SLD_calculations=spec_nx.SLD_calculations

SimulationFunctions={'Specular': Specular,
                     'PolSpecular': PolSpecular,
                     'SLD': SLD_calculations
                     }

(Instrument, Layer, Stack, Sample)=refl.MakeClasses(InstrumentParameters,
                                                    LayerParameters, StackParameters, SampleParameters,
                                                    SimulationFunctions,
                                                    ModelID)

# Add gradient for sigma and thickness to multilayers
def resolveLayerParameter(self, parameter):
    if self.beta_sm>0:
        # user has selected super-mirror coating sequence
        m = self.Repetitions
        beta = self.beta_sm
        rho_all = [refl.resolve_par(L, 'dens')*(refl.resolve_par(L, 'b').real+refl.resolve_par(L, 'magn'))*1e-5
                   for L in self.Layers]
        rhoA = max(rho_all)*1e-5
        rhoB = min(rho_all)*1e-5
        # critical wave vector values (in paper this is includes an addition factor of 1/2)
        QcA = 4.*pi*sqrt(rhoA/pi+0j)
        QcB = 4.*pi*sqrt(rhoB/pi+0j)
        Qcm = 0.022*m # critical q for given m-value
        g=Qcm/QcA
        QAB = QcA / sqrt(QcA**2-QcB**2)
        b = beta*QAB**4 + 2.*beta*QAB**2 - 1
        N = int(abs(beta*((g**2*QAB**2+1)**2-1)-b))
    elif self.beta_sm<0:
        # super-mirror from Hayter and Mook iterative method was selected, fixed number of layers
        N = int(abs(self.Repetitions))
    else:
        N = int(self.Repetitions)
    if parameter=='sigma':
        sigma_gradient=self.sigma_gradient
        # parameters for layers with roughness gradient
        par=[refl.resolve_par(lay, parameter) for lay in self.Layers]
        for i in range(1, N):
            if self.sigma_gtype==0:
                # linear increase of roughness to (1+sigma_gradient) times bottom roughness)
                par+=[refl.resolve_par(lay, parameter)*(1.+(sigma_gradient+lay.sigma_gradient)*i/(N-1))
                      for lay in self.Layers]
            elif self.sigma_gtype==1:
                # add roughness using rms
                par+=[refl.resolve_par(lay, parameter)*sqrt(
                    1.+((sigma_gradient+lay.sigma_gradient)*i/(N-1))**2)
                      for lay in self.Layers]
            elif self.sigma_gtype==2:
                # linear increase of roughness to bottom roughness + sigma_gradient)
                par+=[refl.resolve_par(lay, parameter)+(sigma_gradient+lay.sigma_gradient)*i/(N-1)
                      for lay in self.Layers]
            elif self.sigma_gtype==3:
                # add roughness using rms
                par+=[sqrt(
                    refl.resolve_par(lay, parameter)**2+((sigma_gradient+lay.sigma_gradient)*i/(N-1))**2)
                      for lay in self.Layers]
            else:
                raise NotImplementedError('sigma_gtype must be between 0 and 3')
    elif parameter=='d':
        d_gradient = self.d_gradient
        if self.beta_sm>0:
            # layer thickness sequence based on calculated bi-layer thicknesses for super-mirror
            L_thicknesses = [refl.resolve_par(lay, 'd')+0.0 for lay in self.Layers]
            D_start = sum(L_thicknesses)
            rel_thickness = [di/D_start for di in L_thicknesses]
            sm_scale=self.sm_scale
            idx=arange(N)
            D_SL=abs((2.*QAB*pi/QcA) / sqrt(sqrt(1+(N-idx+b)/beta)-1))*(1.0+d_gradient*idx/(N-1))*sm_scale
            par = []
            for i in range(N):
                par += [di*D_SL[i] for di in rel_thickness]
        elif self.beta_sm<0:
            zeta = -self.beta_sm
            rho_all = [refl.resolve_par(L, 'dens')*(refl.resolve_par(L, 'b').real+refl.resolve_par(L, 'magn'))*1e-5
                       for L in self.Layers]
            rho1 = max(rho_all) # in paper, SLD is called alpha
            rho2 = min(rho_all)
            # if two layers are chosen, use scale from olgorithm, else only change main 2 layers size
            L_thicknesses = [refl.resolve_par(lay, 'd')+0.0 for lay in self.Layers]
            D_start = sum(L_thicknesses)
            idx1,idx2=rho_all.index(rho1),rho_all.index(rho2)
            rest_size=D_start-L_thicknesses[idx1]-L_thicknesses[idx2]
            # layer thicknesses according to Hayter+Mook algorithm
            D_SL=sm_layers(rho1, rho2, N, zeta)
            # sign on repetitions defines direction of multilayer
            if self.Repetitions>0:
                D_SL.reverse()
            par=[]
            sm_scale=self.sm_scale
            for i, (d1, d2) in enumerate(D_SL):
                # variation from user parameters
                var_scale=(1.0+d_gradient*i/(N-1))*sm_scale
                # scale the size of main two layer but remove residual layer thicknesses
                total_bilayer=d1+d2
                scaled_bilayer=(total_bilayer-rest_size)/total_bilayer
                for i, di in enumerate(L_thicknesses):
                    if i==idx1:
                        par.append(d1*scaled_bilayer*var_scale)
                    elif i==idx2:
                        par.append(d2*scaled_bilayer*var_scale)
                    else:
                        par.append(di*var_scale)
        else:
            # parameters for layers with thickness gradient
            par=[]
            for i in range(N):
                par+=[refl.resolve_par(lay, parameter)*(1.-(d_gradient+lay.d_gradient)*(1./2.-float(i)/N))
                      for lay in self.Layers]
    elif parameter in ['dens']:
        dens_gradient=self.dens_gradient
        # parameters for layers with roughness gradient
        par=[]
        for i in range(N):
            par+=[refl.resolve_par(lay, parameter)*(1.-dens_gradient*(1./2.-float(i)/self.Repetitions)) for lay in
                  self.Layers]
    else:
        par=[refl.resolve_par(lay, parameter)+0.0 for lay in self.Layers]*N
    return par

Stack.resolveLayerParameter=resolveLayerParameter
