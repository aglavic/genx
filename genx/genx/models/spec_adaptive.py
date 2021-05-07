# -*- coding: utf8 -*-
'''<h1>Library for combined x-ray and neutrons simulations with adaptive layer segmentation.</h1>
<p>Most of the model is the same as in spec_nx.</p>
'''
from copy import deepcopy
from numpy import *
from scipy.special import wofz
from . import spec_nx
from .spec_nx import refl
from .lib.instrument import *

# Preamble to define the parameters needed for the models outlined below:
ModelID='SpecAdaptive'
__pars__=spec_nx.__pars__.copy()

instrument_string_choices=spec_nx.instrument_string_choices.copy()
InstrumentParameters=spec_nx.InstrumentParameters.copy()
InstrumentGroups=spec_nx.InstrumentGroups.copy()
InstrumentUnits=spec_nx.InstrumentUnits.copy()

LayerParameters=spec_nx.LayerParameters.copy()
LayerParameters.update({'magn_void': False, 'rough_type': 0, 'sigma_mag': 0.0})
LayerUnits=spec_nx.LayerUnits.copy()
LayerUnits.update({'magn_void': 'True/False',
                   'rough_type': '0-gauss/1-lin\n2/3-exp', 'sigma_mag': 'AA'})
LayerGroups=[('Standard', ['f', 'dens', 'd', 'sigma']),
             ('Neutron', ['b', 'xs_ai', 'magn', 'magn_ang', 'sigma_mag']),
             ('Special', ['magn_void', 'rough_type'])]

StackParameters={'Layers': [], 'Repetitions': 1, 'Element': 0}
SampleParameters=spec_nx.SampleParameters.copy()
SampleParameters.update({'minimal_steps': 0.5, 'max_diff_n': 0.01,
                         'max_diff_x': 0.01, 'smoothen': 0})
sample_string_choices={}

AA_to_eV=spec_nx.AA_to_eV
q_limit=spec_nx.q_limit
Buffer=spec_nx.Buffer

def SLD_calculations(z, item, sample, inst):
    res=spec_nx.SLD_calculations(z, item, sample, inst)
    res['z']-=5*refl.resolve_par(sample.Substrate, 'sigma')
    return res

SimulationFunctions={'Specular': spec_nx.Specular,
                     'SLD': SLD_calculations
                     }

(Instrument, Layer, Stack, Sample)=refl.MakeClasses(InstrumentParameters,
                                                    LayerParameters, StackParameters, SampleParameters,
                                                    SimulationFunctions,
                                                    ModelID)

def calculate_segmentation(sample):
    '''
      Calculate segmentation steps inside a sample defined by
      a maximum SLD slope and minimum step size. It first
      calculates the nuclear and magnetic SLD profile
      from the model and than separates it according to the
      given parameters.
    '''
    parameters=resolve_parameters_by_element(sample)
    dens=array(parameters['dens'], dtype=float64)
    f=array(parameters['f'], dtype=complex128)
    b=array(parameters['b'], dtype=complex128)
    xs_ai=array(parameters['xs_ai'], dtype=float64)
    magn=array(parameters['magn'], dtype=float64)
    magn_ang=array(parameters['magn_ang'], dtype=float64)/180.*pi
    magn_void=array(parameters['magn_void'], dtype=float64)

    sld_x=dens*f
    sld_n=dens*b
    sld_xs=dens*xs_ai
    # the new magnetization density and angle will be
    # calculated from perpendicular and parallel components to allow
    # a smooth transition in magnetic angle
    mag_sld_nsf=magn*dens*cos(magn_ang)
    mag_sld_sf=magn*dens*sin(magn_ang)

    d=array(parameters['d'], dtype=float64)
    d=d[1:-1]
    # Include one extra element - the zero pos (substrate/film interface)
    int_pos=cumsum(r_[0, d])
    rough_type=array(parameters['rough_type'], dtype=int)[:-1]
    sigma_n=array(parameters['sigma'], dtype=float64)[:-1]+1e-7
    sigma_m=array(parameters['sigma_mag'], dtype=float64)[:-1]+1e-7
    z=arange(-sigma_n[0]*5, int_pos.max()+sigma_n[-1]*5, sample.minimal_steps/5.0)
    # interface transition functions
    trans_n=(0.5-0.5*erf((z[:, newaxis]-int_pos)/sqrt(2.)/sigma_n))*(rough_type==0)
    trans_n+=maximum(0., minimum(1., (1.+(int_pos-z[:, newaxis])/2./sigma_n)/2.))*(rough_type==1)
    trans_n+=maximum(0., minimum(1., exp((int_pos-z[:, newaxis])/sigma_n)))*(rough_type==2)
    trans_n+=maximum(0., minimum(1., 1.-exp((z[:, newaxis]-int_pos)/sigma_n)))*(rough_type==3)
    trans_m=(0.5-0.5*erf((z[:, newaxis]-int_pos)/sqrt(2.)/sigma_m))*(rough_type==0)
    trans_m+=maximum(0., minimum(1., (1.+(int_pos-z[:, newaxis])/2./sigma_m)/2.))*(rough_type==1)
    trans_m+=maximum(0., minimum(1., exp((int_pos-z[:, newaxis])/sigma_m)))*(rough_type==2)
    trans_m+=maximum(0., minimum(1., 1.-exp((z[:, newaxis]-int_pos)/sigma_m)))*(rough_type==3)
    # SLD calculations
    rho_x=sum((sld_x[:-1]-sld_x[1:])*trans_n, 1)+sld_x[-1]
    rho_n=sum((sld_n[:-1]-sld_n[1:])*trans_n, 1)+sld_n[-1]
    rho_m_nsf=sum((mag_sld_nsf[:-1]-mag_sld_nsf[1:])*trans_m, 1)+mag_sld_nsf[-1]
    rho_m_sf=sum((mag_sld_sf[:-1]-mag_sld_sf[1:])*trans_m, 1)+mag_sld_sf[-1]
    rho_void=sum((magn_void[:-1]-magn_void[1:])*trans_n, 1)+magn_void[-1]
    xs_ai_comb=sum((sld_xs[:-1]-sld_xs[1:])*trans_n, 1)+sld_xs[-1]
    # add more elements to the SLDs
    for params in parameters['Elements'][1:]:
        dens=array(params['dens'], dtype=float64)
        f=array(params['f'], dtype=complex128)
        b=array(params['b'], dtype=complex128)
        xs_ai=array(params['xs_ai'], dtype=float64)
        magn=array(params['magn'], dtype=float64)
        magn_ang=array(params['magn_ang'], dtype=float64)/180.*pi
        sld_x=dens*f
        sld_n=dens*b
        sld_xs=dens*xs_ai
        mag_sld_nsf=magn*dens*cos(magn_ang)
        mag_sld_sf=magn*dens*sin(magn_ang)

        d=array(params['d'], dtype=float64)
        d=d[1:-1]
        # Include one extra element - the zero pos (substrate/film interface)
        int_pos=cumsum(r_[0, d])
        rough_type=array(params['rough_type'], dtype=int)[:-1]
        sigma_n=array(params['sigma'], dtype=float64)[:-1]+1e-7
        sigma_m=array(params['sigma_mag'], dtype=float64)[:-1]+1e-7
        # interface transition functions
        trans_n=(0.5-0.5*erf((z[:, newaxis]-int_pos)/sqrt(2.)/sigma_n))*(rough_type==0)
        trans_n+=maximum(0., minimum(1., (1.+(int_pos-z[:, newaxis])/2./sigma_n)/2.))*(rough_type==1)
        trans_n+=maximum(0., minimum(1., exp((int_pos-z[:, newaxis])/sigma_n)))*(rough_type==2)
        trans_n+=maximum(0., minimum(1., 1.-exp((z[:, newaxis]-int_pos)/sigma_n)))*(rough_type==3)
        trans_m=(0.5-0.5*erf((z[:, newaxis]-int_pos)/sqrt(2.)/sigma_m))*(rough_type==0)
        trans_m+=maximum(0., minimum(1., (1.+(int_pos-z[:, newaxis])/2./sigma_m)/2.))*(rough_type==1)
        trans_m+=maximum(0., minimum(1., exp((int_pos-z[:, newaxis])/sigma_m)))*(rough_type==2)
        trans_m+=maximum(0., minimum(1., 1.-exp((z[:, newaxis]-int_pos)/sigma_m)))*(rough_type==3)
        # SLD calculations
        rho_x+=sum((sld_x[:-1]-sld_x[1:])*trans_n, 1)+sld_x[-1]
        rho_n+=sum((sld_n[:-1]-sld_n[1:])*trans_n, 1)+sld_n[-1]
        rho_m_nsf+=sum((mag_sld_nsf[:-1]-mag_sld_nsf[1:])*trans_m, 1)+mag_sld_nsf[-1]
        rho_m_sf+=sum((mag_sld_sf[:-1]-mag_sld_sf[1:])*trans_m, 1)+mag_sld_sf[-1]
        xs_ai_comb+=sum((sld_xs[:-1]-sld_xs[1:])*trans_n, 1)+sld_xs[-1]
    # calculate the segmentation
    d_segments=[0.]
    i=0
    rho_x_out=[rho_x[0]]
    rho_n_out=[rho_n[0]]
    rho_nsf_out=[rho_m_nsf[0]]
    rho_sf_out=[rho_m_sf[0]]
    xs_ai_out=[xs_ai_comb[0]]
    while i<(len(z)-1):
        # calculate SLD differeces for all subsequent positions
        #        diff_x=abs(rho_x[i+1:]-rho_x_out[-1])
        #        diff_n_p=abs(rho_n[i+1:]+rho_m_nsf[i+1:]-rho_n_out[-1]-rho_nsf_out[-1])
        #        diff_n_m=abs(rho_n[i+1:]-rho_m_nsf[i+1:]-rho_n_out[-1]+rho_nsf_out[-1])
        #        diff_m_sf=abs(rho_m_sf[i+1:]-rho_sf_out[-1])
        #        diff_idx=where((diff_n_p>sample.max_diff_n)|(diff_x>sample.max_diff_x)|
        #                       (diff_n_m>sample.max_diff_n)|(diff_m_sf>sample.max_diff_n))[0]
        # calculate the maximum variation of SLD up to given index
        diff_x=abs(maximum.accumulate(rho_x[i+1:])-minimum.accumulate(rho_x[i+1:]))
        diff_n_p=abs(maximum.accumulate(rho_n[i+1:]+rho_m_nsf[i+1:])-minimum.accumulate(rho_n[i+1:]+rho_m_nsf[i+1:]))
        diff_n_m=abs(maximum.accumulate(rho_n[i+1:]-rho_m_nsf[i+1:])-minimum.accumulate(rho_n[i+1:]-rho_m_nsf[i+1:]))
        diff_m_sf=abs(maximum.accumulate(rho_m_sf[i+1:])-minimum.accumulate(rho_m_sf[i+1:]))
        diff_idx=where(logical_not((diff_n_p<sample.max_diff_n) & (diff_x<sample.max_diff_x) &
                                   (diff_n_m<sample.max_diff_n) & (diff_m_sf<sample.max_diff_n)))[0]
        if len(diff_idx)>0:
            j=min(len(z)-1, max(i+diff_idx[0]+1, i+5))
        else:
            j=len(z)-1  # last position
        d_segments.append(z[j]-z[i])
        rho_x_out.append(rho_x[i:j].mean())
        rho_n_out.append(rho_n[i:j].mean())
        rho_nsf_out.append(rho_m_nsf[i:j].mean()*(1.-rho_void[i:j].mean()))  # averadge magn taking voids into account
        rho_sf_out.append(rho_m_sf[i:j].mean()*(1.-rho_void[i:j].mean()))  # averadge magn taking voids into account
        xs_ai_out.append(xs_ai_comb[i:j].mean())  # averadge mang angle
        i=j
    rho_nsf_out=array(rho_nsf_out)
    rho_sf_out=array(rho_sf_out)
    rho_m_out=sqrt(rho_nsf_out**2+rho_sf_out**2).tolist()
    magn_ang_out=(arctan2(rho_nsf_out, -rho_sf_out)*180./pi-90.).tolist()
    return (d_segments[1:], rho_x_out[1:],
            rho_n_out[1:], rho_m_out[1:], xs_ai_out[1:], magn_ang_out[1:])

def resolve_parameters_by_element(sample):
    """
      Resolve the model standard parameters for each element.
      The first element used for normal parameter names,
      every other element does ignore substrate and ambience sld
      and is assigned to the 'elements' keyword as list.
      This makes it possible to build SLD profiles as sum of all
      elements.
    """
    elements=list(set([stack.Element for stack in sample.Stacks]))
    elements.sort()
    par=sample.Substrate._parameters.copy()
    for k in par:
        par[k]=[refl.resolve_par(sample.Substrate, k)]
    for k in sample.Substrate._parameters:
        for stack in sample.Stacks:
            if stack.Element!=0:
                continue
            par[k]=par[k]+stack.resolveLayerParameter(k)
        par[k]=par[k]+[refl.resolve_par(sample.Ambient, k)]
    par['sigma_mag']=where(array(par['sigma_mag'])!=0.,
                           par['sigma_mag'], par['sigma']).tolist()
    output=par
    output['Elements']=[]
    for element in elements:
        par=sample.Substrate._parameters.copy()
        for k in list(par.keys()):
            par[k]=[refl.resolve_par(sample.Substrate, k)]
        # zero substrat SLD
        par['f']=[0j]
        par['b']=[0j]
        par['magn']=[0.]
        par['xs_ai']=[0.]
        par['magn_ang']=[0.]
        for k in sample.Substrate._parameters:
            for stack in sample.Stacks:
                if stack.Element!=element:
                    continue
                par[k]=par[k]+stack.resolveLayerParameter(k)
            par[k]=par[k]+[refl.resolve_par(sample.Ambient, k)+0.0]
        # zero ambience SLD for Element
        par['f'][-1]=0j
        par['b'][-1]=0j
        par['magn'][-1]=0.
        par['xs_ai'][-1]=0.
        par['magn_ang'][-1]=0.
        # if magnetic roughness is set to zero use structural roughness
        par['sigma_mag']=where(array(par['sigma_mag'])!=0.,
                               par['sigma_mag'], par['sigma']).tolist()
        output['Elements'].append(par)
    return output

def resolveLayerParameters(self):
    # resolve parameters by creating layers automatically
    # adapting changes to the SLDs
    par={}
    for k in ['b', 'd', 'dens', 'f', 'magn', 'magn_ang', 'sigma', 'xs_ai']:
        par[k]=[refl.resolve_par(self.Substrate, k)]
    par['sigma'][0]=0.
    d, rho_x, rho_n, rho_m, xs_ai, magn_ang=calculate_segmentation(self)
    par['d']+=d
    par['f']+=rho_x
    par['b']+=rho_n
    par['magn']+=rho_m
    par['xs_ai']+=xs_ai
    if self.smoothen:
        par['sigma']+=[(d[i]+d[i+1])/4. for i in range(len(d)-1)]+[0.]
    else:
        par['sigma']+=[0. for ignore in d]
    par['magn_ang']+=magn_ang
    par['dens']+=[1. for ignore in d]
    for k in ['b', 'd', 'dens', 'f', 'magn', 'magn_ang', 'sigma', 'xs_ai']:
        par[k].append(refl.resolve_par(self.Ambient, k))
    return par

Sample.resolveLayerParameters=resolveLayerParameters

if __name__=='__main__':
    pass
