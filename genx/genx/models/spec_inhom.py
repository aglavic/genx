# -*- coding: utf8 -*-
'''<h1>Library for combined x-ray and neutrons simulations for inhomogeneous samples.</h1>
<p>Most of the model is the same as in spec_nx.</p>
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
