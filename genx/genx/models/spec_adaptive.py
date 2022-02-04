# -*- coding: utf8 -*-
'''
Library for combined x-ray and neutrons simulations with adaptive layer segmentation
====================================================================================
Library for specular neutron and x-ray reflectometry of more complex structures where elemental composition
and/or magnetism is better described separately than within one slap model. The model sums up a set of
*Elements* to calculate a total SLD profile and then uses adaptive layer segmentation to model it.

The actual modeling of the result structure is done with the same function as in spec_nx.

Classes
-------

Layer
~~~~~
``Layer(b = 0.0, d = 0.0, f = 0.0+0.0J, b=0j, dens = 1.0, magn_ang = 0.0, magn = 0.0, sigma = 0.0, xs_ai=0.0, rough_type=0, sigma_magn=0.0, magn_void=False)``

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
``sigma_mag``
   A different roughness parameter for the magnetization of the layer, 0 is ignored
``xs_ai``
   The sum of the absorption cross section and the incoherent scattering
   cross section in barns for neutrons
``magn_void``
   If true this layer has no magnetization. In case of *sigma_mag* beging larger then 0 the additional
   roughness is only applied to the magnetic layer and inside this layer follows the chemical profile.
``rough_type``
   Used model to get the SLD profile of the interface, *0* is an error function profile (gaussian roughness),
   *1* is a linear profile, *2* and *3* are exponential decays from bottom or top side.


Stack
~~~~~
``Stack(Layers = [], Repetitions = 1, Element = 0)``

``Layers``
   A ``list`` consiting of ``Layer``\ s in the stack the first item is
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

Sample
~~~~~~
``Sample(Stacks = [], Ambient = Layer(), Substrate = Layer(), minimal_steps=0.5, max_diff_n=0.01, max_diff_x=0.01, smoothen=0)``

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
``minimal_steps``
   The thickness of the minimal step between layers. Smaller values make the model more precise but slower.
   For data with larger q-range a smaller minimal_step is required. Try to start with 0.5-1 Å step size and
   increase the value until you see differences in the simulated data.
``max_diff_n``
   Maximum neutron SLD deviation to be allowed for layers to be combined in the adaptive procedure
``max_diff_x``
   Maximum x-ray SLD deviation to be allowed for layers to be combined in the adaptive procedure
``smoothen``
   Default is to not use any roughness for the segmentation. If set to 1 this will add roughnesses between all
   segments to make the curve more smooth.

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

__xlabel__ = "q [Å$^{-1}$]"
__ylabel__ = "Instnsity [a.u.]"

def SLD_calculations(z, item, sample, inst):
    res=spec_nx.SLD_calculations(z, item, sample, inst)
    res['z']-=5*refl.resolve_par(sample.Substrate, 'sigma')
    return res

def Specular(TwoThetaQz, sample, instrument):
    out = spec_nx.Specular(TwoThetaQz, sample, instrument)
    global __xlabel__
    __xlabel__ = spec_nx.__xlabel__
    return out

def PolSpecular(TwoThetaQz, p1, p2, F1, F2, sample, instrument):
    out = spec_nx.PolSpecular(TwoThetaQz, p1, p2, F1, F2, sample, instrument)
    global __xlabel__
    __xlabel__ = spec_nx.__xlabel__
    return out


SimulationFunctions={'Specular': Specular,
                     'PolSpecular': PolSpecular,
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
    rho_x_r=rho_x.real
    rho_n_p=rho_n.real+rho_m_nsf
    rho_n_m=rho_n.real-rho_m_nsf
    while i<(len(z)-1):
        j=next_adaptive_segment(i, rho_x_r, rho_n_p, rho_n_m, rho_m_sf,
                                sample.max_diff_n, sample.max_diff_x, z)
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

def next_adaptive_segment(i, rho_x_r, rho_n_p, rho_n_m, rho_m_sf, max_diff_n, max_diff_x, z):
    # calculate the maximum variation of SLD up to given index
    diff_x=abs(maximum.accumulate(rho_x_r[i+1:])-minimum.accumulate(rho_x_r[i+1:]))
    diff_n_p=abs(maximum.accumulate(rho_n_p[i+1:])-minimum.accumulate(rho_n_p[i+1:]))
    diff_n_m=abs(maximum.accumulate(rho_n_m[i+1:])-minimum.accumulate(rho_n_m[i+1:]))
    diff_m_sf=abs(maximum.accumulate(rho_m_sf[i+1:])-minimum.accumulate(rho_m_sf[i+1:]))
    diff_idx=where(logical_not((diff_n_p<max_diff_n) & (diff_x<max_diff_x) &
                               (diff_n_m<max_diff_n) & (diff_m_sf<max_diff_n)))[0]
    if len(diff_idx)>0:
        j=min(len(z)-1, max(i+diff_idx[0]+1, i+5))
    else:
        j=len(z)-1  # last position
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
