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
                 'd_gradient': 0., 'dens_gradient': 0.,  # change in thickness or density bottom to top
                 }
SampleParameters=spec_nx.SampleParameters.copy()
SampleParameters.update({'sigma_inhom': 0.0, 'lscale_inhom': 0.9, 'flatwidth_inhom': 0.3,
                         'steps_inhom': 20, 'type_inhom': 'empiric PLD'})
sample_string_choices={
    'type_inhom': ['gauss', 'semi-gauss', 'empiric PLD'],
    }

AA_to_eV=spec_nx.AA_to_eV
q_limit=spec_nx.q_limit
Buffer=spec_nx.Buffer

def Specular(TwoThetaQz, sample, instrument):
    '''
      The model function. Averadging the intensities for different
      layer thicknesses as found for e.g. large PLD samples.
    '''

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

SLD_calculations=spec_nx.SLD_calculations

SimulationFunctions={'Specular': Specular,
                     'SLD': SLD_calculations
                     }

(Instrument, Layer, Stack, Sample)=refl.MakeClasses(InstrumentParameters,
                                                    LayerParameters, StackParameters, SampleParameters,
                                                    SimulationFunctions,
                                                    ModelID)

# Add gradient for sigma and thickness to multilayers
def resolveLayerParameter(self, parameter):
    if parameter=='sigma':
        sigma_gradient=self.sigma_gradient
        # parameters for layers with roughness gradient
        par=[refl.resolve_par(lay, parameter) for lay in self.Layers]
        for i in range(1, self.Repetitions):
            if self.sigma_gtype==0:
                # linear increase of roughness to (1+sigma_gradient) times bottom roughness)
                par+=[refl.resolve_par(lay, parameter)*(1.+(sigma_gradient+lay.sigma_gradient)*i/(self.Repetitions-1))
                      for lay in self.Layers]
            elif self.sigma_gtype==1:
                # add roughness using rms
                par+=[refl.resolve_par(lay, parameter)*sqrt(
                    1.+((sigma_gradient+lay.sigma_gradient)*i/(self.Repetitions-1))**2)
                      for lay in self.Layers]
            elif self.sigma_gtype==2:
                # linear increase of roughness to bottom roughness + sigma_gradient)
                par+=[refl.resolve_par(lay, parameter)+(sigma_gradient+lay.sigma_gradient)*i/(self.Repetitions-1)
                      for lay in self.Layers]
            elif self.sigma_gtype==3:
                # add roughness using rms
                par+=[sqrt(
                    refl.resolve_par(lay, parameter)**2+((sigma_gradient+lay.sigma_gradient)*i/(self.Repetitions-1))**2)
                      for lay in self.Layers]
            else:
                raise NotImplementedError('sigma_gtype must be between 0 and 3')
    elif parameter=='d':
        d_gradient=self.d_gradient
        # parameters for layers with roughness gradient
        par=[]
        for i in range(self.Repetitions):
            par+=[refl.resolve_par(lay, parameter)*(1.-(d_gradient+lay.d_gradient)*(1./2.-float(i)/self.Repetitions))
                  for lay in self.Layers]
    elif parameter in ['dens']:
        dens_gradient=self.dens_gradient
        # parameters for layers with roughness gradient
        par=[]
        for i in range(self.Repetitions):
            par+=[refl.resolve_par(lay, parameter)*(1.-dens_gradient*(1./2.-float(i)/self.Repetitions)) for lay in
                  self.Layers]
    else:
        par=[refl.resolve_par(lay, parameter)+0.0 for lay in self.Layers]*self.Repetitions
    return par

Stack.resolveLayerParameter=resolveLayerParameter
