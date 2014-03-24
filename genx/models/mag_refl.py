''' <h1> Library for specular magnetic x-ray and neutron reflectivity</h1>
The magnetic reflectivity is calculated according to: S.A. Stephanov and S.K Shina PRB 61 15304. for 
the full anisotropic model. It also one simpler model where the media is considered to be isotropic but 
with different refractive indices for left and right circular light.
The model also has the possibility to calculate the neutron reflectivity from the same sample structure. 
This model includes a interface layer for each <code>Layer</code>. This means that the model is suitable for 
refining data that looks for interfacial changes of the magnetic moment. 

Note! This model should be considered as a gamma version. It is still under heavy development and 
the api can change significantly from version to version. Should only be used by expert users. 
<h2>Classes</h2>
<h3>Layer</h3>
<code> Layer(fr = 1e-20j, b = 1e-20j, dd_u = 0.0, d = 0.0, f = 1e-20j, dens = 1.0, resmag = 1.0, 
            theta_m = 0.0, fm2 = 1e-20j, xs_ai = 0.0, 
            sigma_mu = 0.0, fm1 = 1e-20j, dmag_u = 0.0,
             mag = 0.0, sigma_ml = 0.0, sigma_c = 0.0,
              resdens = 1.0, phi_m = 0.0, dd_l = 0.0, dmag_l = 0.0)</code>
    <dl>
    <dt><code><b>d</b></code></dt>
    <dd>The thickness of the layer in AA (Angstroms = 1e-10m)</dd>
    <dt><code><b>dens</b></code></dt>
    <dd>The density of formula units in units per Angstroms. Note the units!</dd>
    <dt><code><b>sigma</b></code></dt>
    <dd>The root mean square roughness of the top interface for the layer in Angstroms.</dd>        
    <dt><code><b>f</b></code></dt>
    <dd>The non-resonant x-ray scattering length per formula unit in electrons. To be strict it is the
    number of Thompson scattering lengths for each formula unit.</dd>
    <dt><code><b>fr</b></code></dt>
    <dd>The resonant x-ray scattering length of the resonant species in electrons. This is multiplied by
    <code>resdens*dens</code> to form the resonant scattering length. The total non-magnetic scattering length is
    <code>(f + fr*resdens)*dens</code>.</dd>
    <dt><code><b>fm1</b></code></dt>
    <dd>The resonant magnetic part of the scattering length - refers to the magnetic circular dichroic part.
    Same units as <code>f</code></dd>
    <dt><code><b>fm2</b></code></dt>
    <dd>The resonant magnetic part of the scattering length - refers to the magnetic linear dichroic part.</dd>
    <dt><code><b>b</b></code></dt>
    <dd>The neutron scattering length in fm.</dd>
    <dt><code><b>xs_ai</b></code></dt>
    <dd>The sum of the absorption cross section and the incoherent scattering cross section
        in barns per formula unit for the neutrons</dd>
    <dt><code><b>mag</b></code></dt>
    <dd>The magnetic moment per formula unit. The magnetic density is <code>mag*dens</code>.</dd>
    <dt><code><b>phi_m</b></code></dt>
    <dd>The in-plane angle of the magnetic moment of the layer relative the projected incident beam for 
    x-rays and relative the polarization axis for neutrons.</dd>
    <dt><code><b>theta_m</b></code></dt>
    <dd>The out-of-plane angle of the magnetic moment. <code>theta_m = 0</code> corresponds to an in-plane 
    magnetic moment and <code>theta_m</code> corresponds to an out-of-plane magnetic moment.</dd>
    <dt><code><b>dmag_u</b></code></dt>
    <dd>The relative increase of the magnetic moment in the interface layer. Total magnetic moment is
    <code>mag*(1 + dmag_u)</code>.</dd>
    <dt><code><b>dmag_l</b></code></dt>
    <dd>As <code>dmag_u</code> but for the lower interface layer.</dd>
    <dt><code><b>dd_u</b></code></dt>
    <dd>The width of the upper interface layer in Angstroms.</dd>
    <dt><code><b>sigma_mu</b></code></dt>
    <dd>The roughness of the upper magnetic interface.</dd>
    <dt><code><b>sigma_ml</b></code></dt>
    <dd>The roughness of the lower magnetic interface.</dd>
    <dt><code><b>dd_l</b></code></dt>
    <dd>The width of the lower interface in Angstroms.</dd>
    <dt><code><b>resmag</b></code></dt>
    <dd>The relative amount of magnetic resonant atoms  the total resonant magnetic atoms. The total magnetic scattering
    length is calculated 
    as (for the circular dichroic term) <code>fm1*resmag*mag*resdens*dens</code></dd>
    </dl>
<h3>Stack</h3>
<code> Stack(Layers = [], Repetitions = 1)</code>
    <dl>
    <dt><code><b>Layers</b></code></dt>
    <dd>A <code>list</code> consiting of <code>Layer</code>s in the stack
    the first item is the layer closest to the bottom</dd>
    <dt><code><b>Repetitions</b></code></dt>
    <dd>The number of repetitions of the stack</dd>
    </dl>
<h3>Sample</h3>
<code> Sample(Stacks = [], dsld_max = 0.1, dsld_offdiag_max = 0.1, 
            compress = 'yes', slicing = 'no', dsld_n_max = 0.01, 
            dabs_n_max = 0.01, sld_buffer = 20.0, sld_delta = 5.0, 
            dmag_max = 0.01, sld_mult = 4.0, slice_depth = 1.0, 
            Ambient = Amb, Substrate = Sub)</code>
    <dl>
    <dt><code><b>Stacks</b></code></dt>
    <dd>A <code>list</code> consiting of <code>Stack</code>s in the stacks
    the first item is the layer closest to the bottom</dd>
    <dt><code><b>Ambient</b></code></dt>
    <dd>A <code>Layer</code> describing the Ambient (enviroment above the sample).
     Only the scattering lengths and density of the layer is used.</dd>
    <dt><code><b>Substrate</b></code></dt>
    <dd>A <code>Layer</code> describing the substrate (enviroment below the sample).
     Only the scattering lengths, density and  roughness of the layer is used.</dd>
    dt><code><b>dsld_max</b></code></dt>
    <dd>The maximum allowed step in the scattering length density for x-rays (diagonal terms)</dd>
    <dt><code><b>dsld_offdiag_max</b></code></dt>
    <dd>The maximum allowed step in the scattering length density for the offdiagonal terms of the 
    scattering length (magnetic part)</dd>
    <dt><code><b>compress</b></code></dt>
    <dd>A flag that signals if the sliced composition profile should be compressed.</dd>
    <dt><code><b>slicing</b></code></dt>
    <dd>A flag that signals if the composition profile should be sliced up.</dd>
    <dt><code><b>dsld_n_max</b></code></dt>
    <dd>The maximum allowed step (in compression) for the neutron scattering length.</dd>
    <dt><code><b>dabs_n_max</b></code></dt>
    <dd>The maximum allowed step (in compression) for the neutron absorption (in units of barn/AA^3)</dd>
    <dt><code><b>sld_buffer</b></code></dt>
    <dd>A buffer for the slicing calculations (to assure convergence in the sld profile. </dd>
    <dt><code><b>sld_delta</b></code></dt>
    <dd>An extra buffer - needed at all?</dd>
    <dt><code><b>dmag_max</b></code></dt>
    <dd>The maximum allowed step (in compression) for the magnetization. Primarily intended to limit the 
    steps in the magnetic profile for neutrons.</dd>
    <dt><code><b>sld_mult</b></code></dt>
    <dd>A multiplication factor for a buffer that takes the roughness into account.</dd>
    <dt><code><b>slice_depth</b></code></dt>
    <dd>The depth of the slices in the calculation of the sliced scattering length density profile.</dd>
    </dl>
    
<h3>Instrument</h3>
<code>model.Instrument(res = 0.001,theory = 'neutron spin-pol',
                        footype = 'no corr',beamw = 0.01,
                        wavelength = 4.4,respoints = 5,xpol = 'circ+',Ibkg = 0.0,
                        I0 = 1.0,samplelen = 10.0,npol = '++',restype = 'no conv',
                        coords = 'tth',resintrange = 2)</code>
    <dl>
    <dt><code><b>wavelength</b></code></dt>
    <dd>The wavalelngth of the radiation givenin AA (Angstroms)</dd>
    <dt><code><b>coords</b></code></dt>
    <dd>The coordinates of the data given to the SimSpecular function.
    The available alternatives are: 'q' or 'tth'. Alternatively the numbers
    0 (q) or 1 (tth) can be used.</dd>
    <dt><code><b>I0</b></code></dt>
    <dd>The incident intensity (a scaling factor)</dd>
    <dt><code><b>Ibkg</b></code></dt>
    <dd>The background intensity. Added as a constant value to the calculated
    reflectivity</dd>
    <dt><code><b>res</b></code></dt>
    <dd>The resolution of the instrument given in the coordinates of
     <code>coords</code>. This assumes a gaussian reloution function and
    <code>res</code> is the standard deviation of that gaussian.</dd>
    <dt><code><b>restype</b></code></dt>
    <dd>Describes the rype of the resolution calculated. One of the alterantives:
    'no conv', 'fast conv', 'full conv and varying res.' or 'fast conv + varying res.'.
    The respective numbers 0-3 also works. Note that fast convolution only alllows
    a single value into res wheras the other can also take an array with the
    same length as the x-data (varying resolution)</dd>
    <dt><code><b>respoints</b></code></dt>
    <dd>The number of points to include in the resolution calculation. This is only
    used for 'full conv and vaying res.' and 'fast conv + varying res'</dd>
    <dt><code><b>resintrange</b></code></dt>
    <dd>Number of standard deviatons to integrate the resolution fucntion times
    the relfectivty over</dd>
    <dt><code><b>footype</b></code></dt>
    <dd>Which type of footprint correction is to be applied to the simulation.
    One of: 'no corr', 'gauss beam' or 'square beam'. Alternatively, 
    the number 0-2 are also valid. The different choices are self explanatory.
    </dd>
    <dt><code><b>beamw</b></code></dt>
    <dd>The width of the beam given in mm. For 'gauss beam' it should be
    the standard deviation. For 'square beam' it is the full width of the beam.</dd>
    <dt><code><b>samplelen</b></code></dt>
    <dd>The length of the sample given in mm</dd>
    <dt><code><b>theory</b></code></dt>
    <dd>Defines which theory (code) that should calcualte the reflectivity. Should be one of: 
    'x-ray anis.', 'x-ray simpl. anis.', 'neutron spin-pol' or 'neutron spin-flip'.
    </dd>
    <dt><code><b>xpol</b></code></dt>
    <dd>The polarization state of the x-ray beam. Should be one of: 'circ+','circ-','tot', 'ass', 'sigma' or 'pi'</dd>
    <dt><code><b>npol</b></code></dt>
    <dd>The neutron polarization state. Should be '++','uu', '--', 'dd' alt. '+-','ud' for spin flip.</dd>
    
'''

import lib.xrmr
import lib.edm_slicing as edm
try:
    import lib.paratt_weave as Paratt
except StandardError,S:
    print 'Not using inline c code for reflectivity calcs - can not import module'
    print S
    import lib.paratt as Paratt
#import lib.paratt_weave as Paratt
#import lib.paratt as Paratt

import lib.ables as ables

from numpy import *
from scipy.special import erf
from lib.instrument import *

mag_limit = 1e-8
mpy_limit = 1e-8
theta_limit = 1e-8

re = 2.8179402894e-5

# Preamble to define the parameters needed for the models outlined below:
ModelID='MAGrefl'
# Automatic loading of parameters possible by including this list
__pars__ = ['Layer', 'Stack', 'Sample', 'Instrument']
# Used for making choices in the GUI
instrument_string_choices = {'coords': ['q','tth'],
                             'restype': ['no conv', 'fast conv',
                                         'full conv and varying res.', 
                                         'fast conv + varying res.'],
                             'footype': ['no corr', 'gauss beam', 
                                         'square beam'],
                             'xpol':['circ+','circ-','tot', 'ass', 'sigma', 'pi'],
                             'npol':['++', '--', '+-'],
                             'theory': ['x-ray anis.', 'x-ray simpl. anis.', 
                                        'neutron spin-pol', 'neutron spin-flip',
                                        'neutron spin-pol tof'],
                             #'compress':['yes', 'no'],
                             #'slicing':['yes', 'no'],
     }


    
InstrumentParameters={'wavelength':1.54,'coords':'tth','I0':1.0,'res':0.001,\
    'restype':'no conv','respoints':5,'resintrange':2,'beamw':0.01,'footype': 'no corr',\
    'samplelen':10.0, 'Ibkg': 0.0, 'xpol':'circ+', 'npol': '++', 'theory':'x-ray anis.',
    'incang':0.2, }
# Coordinates=1 => twothetainput
# Coordinates=0 => Q input
#Res stddev of resolution
#ResType 0: No resolution convlution
#               1: Fast convolution
#               2: Full Convolution +varying resolution
#               3: Fast convolution varying resolution
#ResPoints Number of points for the convolution only valid for ResolutionType=2
#ResIntrange Number of standard deviatons to integrate over default 2
# Parameters for footprint coorections
# Footype: 0: No corections for footprint
#          1: Correction for Gaussian beam => Beaw given in mm and stddev
#          2: Correction for square profile => Beaw given in full width mm
# Samlen= Samplelength in mm.
#
#
InstrumentGroups = [('General', ['wavelength', 'coords', 'I0', 'Ibkg', 'incang']),
                    ('Resolution', ['restype', 'res', 'respoints', 'resintrange']),
                    ('Misc.', ['theory', 'xpol', 'npol',]),
                    ('Footprint', ['footype', 'beamw', 'samplelen',]),
                    ]
InstrumentUnits={'wavelength':'AA','coords':'','I0':'arb.','res':'[coord]',
                 'restype':'','respoints':'pts.','resintrange': '[coord]',
                 'beamw':'mm','footype': '',\
                 'samplelen':'mm', 'Ibkg': 'arb.', 'xpol':'', 
                 'theory':'','npol': '','incang':'deg'}

LayerParameters = {'dens':1.0, 'd':0.0, 'f': (0.0 + 1e-20J), 
                   'fr':(0.0 + 1e-20J),
                   'fm1':(0.0 + 1e-20J), 'fm2':(0.0 + 1e-20J), 
                   'phi_m': 0.0, 'theta_m': 0.0, 'resdens': 1.0,
                   'resmag': 1.0,
                   'sigma_c': 0.0, 'sigma_ml': 0.0, 'sigma_mu': 0.0,
                   'mag':0.0,
                   'dmag_l': 0.0, 'dmag_u': 0.0, 'dd_l':0.0,
                    'dd_u':0.0, 'b': 1e-20J, 'xs_ai': 0.0,
                   #'dtheta_l': 0.0, 'dtheta_u':0.0, 'dphi_l':0.0, 'dphi_u':0.0,
                   }
LayerUnits = {'dens':'at./AA^3', 'd':'AA', 'f': 'el.', 
                   'fr':'el.',
                   'fm1':'el./mu_B', 'fm2':'el./mu_B', 
                   'phi_m': 'deg.', 'theta_m': 'deg.', 'resdens': 'rel.',
                   'resmag': 'rel.',
                   'sigma_c': 'AA', 'sigma_ml': 'AA', 'sigma_mu': 'AA',
                   'mag': 'mu_B',
                   'dmag_l': 'rel.', 'dmag_u': 'rel.', 'dd_l':'AA',
                   'dd_u':'AA', 'b': 'fm','xs_ai': 'barn/at.',
                   #'dtheta_l': 0.0, 'dtheta_u':0.0, 'dphi_l':0.0, 'dphi_u':0.0,
                   }
LayerGroups = [('Scatt. len.', ['b', 'xs_ai', 'f', 'fr', 'fm1', 'fm2']), 
                ('Magnetism', ['mag', 'resmag', 'phi_m','theta_m']), 
                ('Misc.', ['sigma_c', 'dens', 'resdens', 'd']),
                ('Interf. Mag. Mom.', ['dmag_l', 'dmag_u', 'sigma_ml', 'sigma_mu',
                                       'dd_l', 'dd_u'])
                ]
#('Interf. Mag. Ang.', ('dtheta_l', 'dtheta_u', 'dphi_l', 'dphi_u'))
StackParameters = {'Layers':[], 'Repetitions':1}
SampleParameters = {'Stacks':[], 'Ambient':None, 'Substrate':None, 
                    'compress':'yes', 'slicing':'no', 'slice_depth':1.0,
                    'sld_mult':4.0, 'sld_buffer': 20.0, 'sld_delta': 5.0,
                    'dsld_max':0.1, 'dsld_offdiag_max':0.1, 'dsld_n_max': 0.01,
                    'dmag_max': 0.01, 'dabs_n_max': 0.01}
                    
SampleGroups = [['Slicing', [ 'slicing', 'slice_depth', 'sld_mult', 'sld_buffer', 
                             'sld_delta']],
                ['Compression', ['compress', 'dsld_max', 'dsld_offdiag_max', 
                                 'dmag_max', 'dsld_n_max', 'dabs_n_max']],
                ]
                
sample_string_choices = {'compress':['yes', 'no'],
                          'slicing':['yes', 'no'],
                          }

# A buffer to save previous calculations for spin-flip calculations
class Buffer:
    W = None
    parameters = None
    g_0 = None
    coords = None
    wavelength = None

def Specular(TwoThetaQz, sample, instrument):
    ''' Simulate the specular signal from sample when proped with instrument
    
    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    '''
    # preamble to get it working with my class interface
    restype = instrument.getRestype()
    weight = None
    if restype == 2 or restype == instrument_string_choices['restype'][2]:
            (TwoThetaQz,weight) = ResolutionVector(TwoThetaQz[:], \
                instrument.getRes(), instrument.getRespoints(),\
                 range=instrument.getResintrange())
    if instrument.getCoords() == 1 or\
        instrument.getCoords() == instrument_string_choices['coords'][1]:
        theta = TwoThetaQz/2
    elif instrument.getCoords() == 0 or\
        instrument.getCoords() == instrument_string_choices['coords'][0]:
        theta = arcsin(TwoThetaQz/4/pi*instrument.getWavelength())*180./pi
    if any(theta < theta_limit):
        raise ValueError('The incident angle has to be above %.1e'%theta_limit)
    
    R = reflectivity_xmag(sample, instrument, theta, TwoThetaQz)
    pol = instrument.getXpol()
    theory = instrument.getTheory()
    if not ((pol == 3 or pol == instrument_string_choices['xpol'][3]) and 
        (theory < 2 or theory in instrument_string_choices['theory'][:2])):
        #FootprintCorrections
        foocor = footprint_correction(instrument, theta)
        R = correct_reflectivity(R, instrument, foocor, TwoThetaQz, weight)
        return R*instrument.getI0() + instrument.getIbkg()
    else:
        foocor = footprint_correction(instrument, theta)*0 + 1.0
        R = correct_reflectivity(R, instrument, foocor, TwoThetaQz, weight)
        return R

def OffSpecular(TwoThetaQz, ThetaQx, sample, instrument):
    ''' Function that simulates the off-specular signal (not implemented)
    
    # BEGIN Parameters
    TwoThetaQz 1.0
    ThetaQx data.x
    # END Parameters
    '''
    raise NotImplementedError('Off specular calculations are not implemented for magnetic x-ray reflectivity')


def SLD_calculations(z, sample, inst):
    ''' Calculates the scatteringlength density as at the positions z
    
    # BEGIN Parameters
    z data.x
    # END Parameters
    '''
    use_slicing = sample.getSlicing()
    if use_slicing == 1 or use_slicing == sample_string_choices['slicing'][1]:
        return compose_sld_anal(z, sample, inst)
    lamda = inst.getWavelength()
    theory = inst.getTheory()
    d, sl_c, sl_m1, sl_m2, M, chi, non_mag, mpy, sl_n, abs_n, mag_dens = compose_sld(sample, inst, array([0.0,]))
    if (theory == 0 or theory == instrument_string_choices['theory'][0]): 
        # Full theory return the suceptibility matrix
        
        new_size = len(d)*2
        sl_cp = zeros(new_size, dtype = complex128)
        sl_cp[::2] = sl_c
        sl_cp[1::2] = sl_c
        sl_m1p = zeros(new_size, dtype = complex128)
        sl_m1p[::2] = sl_m1
        sl_m1p[1::2] = sl_m1
        sl_m2p = zeros(new_size, dtype = complex128)
        sl_m2p[::2] = sl_m2
        sl_m2p[1::2] = sl_m2
        #print sl_m2p
        z = zeros(len(d)*2)
        z[::2] = cumsum(r_[0,d[:-1]])
        z[1::2] = cumsum(r_[d])
        #print d, z
        #print z.shape, sl_c.shape
        def interleave(a):
            new_a = zeros(len(a)*2, dtype = complex128)
            new_a[::2] = a
            new_a[1::2] = a
            return new_a
        chi = [[interleave(c) for c in ch] for ch in chi]
        
    #return {'real sld_c': sl_cp.real, 'imag sld_c': sl_cp.imag,
    #        'real sld_m1': sl_m1p.real, 'imag sld_m1': sl_m1p.imag,
    #        'real sld_m2': sl_m2p.real, 'imag sld_m2': sl_m2p.imag,
    #        'z':z}
        re = 2.8179402894e-5
        c = 1/(lamda**2*re/pi)
        return {'Re sl_xx':chi[0][0].real*c, 'Re sl_xy':chi[0][1].real*c, 'Re sl_xz':chi[0][2].real*c,
                'Re sl_yy':chi[1][1].real*c,'Re sl_yz':chi[1][2].real*c,'Re sl_zz':chi[2][2].real*c,
                'Im sl_xx':chi[0][0].imag*c, 'Im sl_xy':chi[0][1].imag*c, 'Im sl_xz':chi[0][2].imag*c,
                'Im sl_yy':chi[1][1].imag*c,'Im sl_yz':chi[1][2].imag*c,'Im sl_zz':chi[2][2].imag*c,
                'z':z, 'SLD unit': 'r_e/\AA^{3}'}
    else:
        z = zeros(len(d)*2)
        z[::2] = cumsum(r_[0,d[:-1]])
        z[1::2] = cumsum(r_[d])
        
        new_size = len(d)*2
        def parray(ar):
            tmp = zeros(new_size, dtype = complex128)
            tmp[::2] = ar
            tmp[1::2] = ar
            return tmp
        sl_cp = parray(sl_c)
        sl_m1p = parray(sl_m1)
        sl_np = parray(sl_n)
        mag_densp = parray(mag_dens)
        abs_np = parray(abs_n)
        if (theory == 1 or theory == instrument_string_choices['theory'][1]):
            # Simplified anisotropic
            #print sl_cp.shape, sl_np.shape, abs_np.shape, mag_densp.shape, z.shape
            return {'Re sld_c': sl_cp.real, 'Im sld_c': sl_cp.imag,
                    'Re sld_m': sl_m1p.real, 'Im sld_m': sl_m1p.imag,
                    'mag_dens': mag_densp,
                    'z':z, 'SLD unit': 'r_{e}/\AA^{3},\,\mu_{B}/\AA^{3}'}
        elif (theory == 2 or theory == instrument_string_choices['theory'][2]):
            # Neutron spin pol
            return {'sld_n': sl_np, 'abs_n': abs_np, 'mag_dens': mag_densp,
                    'z':z, 'SLD unit': 'fm/\AA^{3}, b/\AA^{3},\,\mu_{B}/\AA^{3}'}
        elif (theory == 3 or theory == instrument_string_choices['theory'][3]):
            # Neutron spin pol with spin flip
            return {'sld_n': sl_np, 'abs_n': abs_np, 'mag_dens': mag_densp,
                    'z':z, 'SLD unit': 'fm/\AA^{3}, b/\AA^{3},\,\mu_{B}/\AA^{3}'}
        
    
def compose_sld_anal(z, sample, instrument):
    '''Compose a analytical profile funciton'''
    def sld_interface(z, drho_jm1_l, drho_j, drho_j_u,
                  sigma_jm1_l, sigma_j, sigma_j_u,
                  dd_jm1_l, dd_j_u):
        ''' Calcualte the sld of one interface '''
        sld = drho_j_u*(0.5 + 0.5*erf((z - dd_j_u)/sqrt(2*(sigma_j_u**2 + sigma_j**2))))
        sld += drho_jm1_l*(0.5 + 0.5*erf((z + dd_jm1_l)/sqrt(2*(sigma_jm1_l**2 + sigma_j**2))))
        sld += drho_j*(0.5 + 0.5*erf((z)/sqrt(2)/sigma_j))
        return sld
    re = 2.8179402894e-5
    lamda = instrument.getWavelength()
    parameters = sample.resolveLayerParameters()
    dens = array(parameters['dens'], dtype = float64)
    resdens = array(parameters['resdens'], dtype = float64)
    resmag = array(parameters['resmag'], dtype = float64)
    mag = abs(array(parameters['mag'], dtype = float64))
    dmag_l = array(parameters['dmag_l'], dtype = float64)
    dmag_u = array(parameters['dmag_u'], dtype = float64)
    dd_l = array(parameters['dd_l'], dtype = float64)
    dd_u = array(parameters['dd_u'], dtype = float64)
    #print [type(f) for f in parameters['f']]
    f = array(parameters['f'], dtype = complex128) + (1-1J)*1e-20
    fr = array(parameters['fr'], dtype = complex128) + (1-1J)*1e-20
    fm1 = array(parameters['fm1'], dtype = complex128) + (1-1J)*1e-20
    fm2 = array(parameters['fm2'], dtype = complex128) + (1-1J)*1e-20
    
    d = array(parameters['d'], dtype = float64)
    
    #sl_m2 = dens*resdens*resmag*fm2 #mag is multiplied in later
    
    #g_0 = sin(theta*pi/180.0)
    
    phi = array(parameters['phi_m'], dtype = float64)*pi/180.0
    theta_m = array(parameters['theta_m'], dtype = float64)*pi/180.0
    sl_c = (dens*(f + resdens*fr))
    #print sl_c
    # This is wrong!!! I should have a cos theta dep.
    #sl_m1 = (dens*resdens*resmag*fm1)[:, newaxis]*cos(theta - theta_m[:,newaxis])*cos(phi[:,newaxis])
    sl_m1 = (dens*resdens*resmag*mag*fm1)
    #print sl_m1
    #print M
    #print sl_c.shape, sl_m1.shape
    sigma_c = array(parameters['sigma_c'], dtype = float64) + 1e-20
    sigma_l = array(parameters['sigma_ml'], dtype = float64)+ 1e-20
    sigma_u = array(parameters['sigma_mu'], dtype = float64)+ 1e-20
    sl_m1_l = (sl_m1*(1. + dmag_l))
    sl_m1_u = (sl_m1*(1. + dmag_u))
    
    b = (array(parameters['b'], dtype = complex128))
    abs_xs = (array(parameters['xs_ai'], dtype = complex128))
    wl = instrument.getWavelength()
    #print b
    #print b.shape, abs_xs.shape, theta.shape
    sl_n = dens*b
    mag_d = mag*dens
    #print mag_d
    mag_d_l = mag_d*(1. + dmag_l)
    mag_d_u = mag_d*(1. + dmag_u)
               
    int_pos = cumsum(r_[0,d[1:-1]])
    if z == None:
        z = arange(-sigma_c[0]*10 - 50, int_pos.max()+sigma_c.max()*10+50, 0.5)
        #print 'autoz'
    sld_c = -(sum(sld_interface(z[:,newaxis]-int_pos, 0.0J, sl_c[:-1] - sl_c[1:], 0.0J,
                  sigma_l[1:], sigma_c[:-1], sigma_u[:-1],
                  dd_l[1:], dd_u[:-1]),1) - sl_c[0])
    sld_m = -(sum(sld_interface(z[:,newaxis]-int_pos, sl_m1[:-1] - sl_m1_l[:-1], 
                              sl_m1_l[:-1]  - sl_m1_u[1:], sl_m1_u[1:] - sl_m1[1:],
                  sigma_l[1:], sigma_c[:-1], sigma_u[:-1],
                  dd_l[1:], dd_u[:-1]),1) - sl_m1[0])
    sld_n = -(sum(sld_interface(z[:,newaxis]-int_pos, 0.0J, sl_n[:-1] - sl_n[1:], 0.0J,
                              sigma_l[1:], sigma_c[:-1], sigma_u[:-1],
                              dd_l[1:], dd_u[:-1]),1) - sl_n[0])
    mag_dens = -(sum(sld_interface(z[:,newaxis]-int_pos, mag_d[:-1] - mag_d_l[:-1], 
                              mag_d_l[:-1]  - mag_d_u[1:], mag_d_u[1:] - mag_d[1:],
                  sigma_l[1:], sigma_c[:-1], sigma_u[:-1],
                  dd_l[1:], dd_u[:-1]),1) - mag_d[0])
    
    #print z.shape, sld_c.shape
    #print sld_m
    #print 'he'
    #print mag_dens
    #print sld_n.shape, mag_dens.shape, z.shape, sld_c.real.shape
    return {'z':z, 'Re sld_c': sld_c.real, 'Im sld_c': sld_c.imag,
            'Re sld_m': sld_m.real, 'Im sld_m': sld_m.imag,
            'sld_n': sld_n, 'mag_dens': mag_dens}
    
    if (theory == 0 or theory == instrument_string_choices['theory'][0]):
        # Full polarization calc
        #print sl_cp.shape, sl_np.shape, abs_np.shape, mag_densp.shape, z.shape
        return {'Re sld_c': sl_c.real, 'Im sld_c': sl_c.imag,
                'Re sld_m': sl_m.real, 'Im sld_m': sl_m.imag,
                'mag_dens': mag_dens,
                'z':z, 'SLD unit': 'r_{e}/\AA^{3},\,\mu_{B}/\AA^{3}'}
    elif (theory == 1 or theory == instrument_string_choices['theory'][1]):
        # Simplified anisotropic
        #print sl_cp.shape, sl_np.shape, abs_np.shape, mag_densp.shape, z.shape
        return {'Re sld_c': sl_c.real, 'Im sld_c': sl_c.imag,
                'Re sld_m': sl_m.real, 'Im sld_m': sl_m.imag,
                'mag_dens': mag_dens,
                'z':z, 'SLD unit': 'r_{e}/\AA^{3},\,\mu_{B}/\AA^{3}'}
    elif (theory == 2 or theory == instrument_string_choices['theory'][2]):
        # Neutron spin pol
        return {'sld_n': sl_n, 'mag_dens': mag_dens,
                'z':z, 'SLD unit': 'fm/\AA^{3}, \mu_{B}/\AA^{3}'}
    elif (theory == 3 or theory == instrument_string_choices['theory'][3]):
        # Neutron spin pol with spin flip
        return {'sld_n': sl_n, 'mag_dens': mag_dens,
                'z':z, 'SLD unit': 'fm/\AA^{3}, \mu_{B}/\AA^{3}'}
    

def compose_sld(sample, instrument, theta):
    re = 2.8179402894e-5
    lamda = instrument.getWavelength()
    parameters = sample.resolveLayerParameters()
    dens = array(parameters['dens'], dtype = float64)
    resdens = array(parameters['resdens'], dtype = float64)
    resmag = array(parameters['resmag'], dtype = float64)
    mag = abs(array(parameters['mag'], dtype = float64))
    dmag_l = array(parameters['dmag_l'], dtype = float64)
    dmag_u = array(parameters['dmag_u'], dtype = float64)
    dd_u = array(parameters['dd_u'], dtype = float64)
    dd_l = array(parameters['dd_l'], dtype = float64)
    
    #print [type(f) for f in parameters['f']]
    f = array(parameters['f'], dtype = complex128) + (1-1J)*1e-20
    fr = array(parameters['fr'], dtype = complex128) + (1-1J)*1e-20
    fm1 = array(parameters['fm1'], dtype = complex128) + (1-1J)*1e-20
    fm2 = array(parameters['fm2'], dtype = complex128) + (1-1J)*1e-20
    
    d = array(parameters['d'], dtype = float64)
    sl_c = dens*(f + resdens*fr) 
    sl_m1 = dens*resdens*resmag*fm1
    sl_m2 = dens*resdens*resmag*fm2 #mag is multiplied in later
    
    #g_0 = sin(theta*pi/180.0)
    phi = array(parameters['phi_m'], dtype = float64)*pi/180.0
    theta_m = array(parameters['theta_m'], dtype = float64)*pi/180.0
    M = c_[cos(theta_m)*cos(phi), cos(theta_m)*sin(phi), sin(theta_m)]
    #print M
    sigma_c = array(parameters['sigma_c'], dtype = float64)
    sigma_mu = sqrt(array(parameters['sigma_mu'], dtype = float64)[:-1]**2 + sigma_c[:-1]**2)
    sigma_ml = sqrt(array(parameters['sigma_ml'], dtype = float64)[1:]**2 + sigma_c[:-1]**2)
    #print sigma_c
    #print sigma_ml
    #print sigma_mu
    
    #Neutrons
    wl = instrument.getWavelength()
    abs_xs = array(parameters['xs_ai'], dtype = complex64)#*(1e-4)**2
    b = array(parameters['b'], dtype = complex64).real#*1e-5
    #sl_n = dens*(wl**2/2/pi*sqrt(b**2 - (abs_xs/2.0/wl)**2) - 
    #                           1.0J*abs_xs*wl/4/pi)
    sl_n = dens*b
    abs_n = dens*abs_xs
    #mag_dens = mag*dens
    #sl_nm = 2.645e-5*mag*dens*instrument.getWavelength()**2/2/pi
    
    #print A, B
    #print type(sample.getSld_buffer())
    if sample.getSlicing() == sample_string_choices['slicing'][0]:
        dz = sample.getSlice_depth()
        reply= edm.create_profile_cm2(d[1:-1], sigma_c[:-1].real, 
                                      sigma_ml.real, sigma_mu.real, 
                                     [edm.erf_profile]*len(sl_c),
                                     [edm.erf_interf]*len(sigma_c[:]),
                                     dmag_l, dmag_u, mag, dd_l, dd_u,
                                     dz = dz, mult = sample.getSld_mult(), 
                                     buffer = sample.getSld_buffer(), 
                                     delta = sample.getSld_delta())
        z, comp_prof, mag_prof = reply
        sl_c_lay = comp_prof*sl_c[:, newaxis]
        sl_c = sl_c_lay.sum(0)
        sl_m1_lay = comp_prof*mag_prof*sl_m1[:, newaxis]
        sl_m1 = sl_m1_lay.sum(0)
        sl_m2_lay = comp_prof*mag_prof*sl_m2[:, newaxis]
        sl_m2 = sl_m2_lay.sum(0)
        
        # Neutrons
        sl_n_lay = comp_prof*sl_n[:, newaxis]
        sl_n = sl_n_lay.sum(0)
        abs_n_lay = comp_prof*abs_n[:,newaxis]
        abs_n = abs_n_lay.sum(0)
        mag_dens_lay = comp_prof*mag_prof*dens[:, newaxis]
        mag_dens = mag_dens_lay.sum(0)     
        
        #print comp_prof.shape, sl_m1_lay.shape, sl_c_lay.shape
        M = rollaxis(array((ones(comp_prof.shape)*M[:,0][:, newaxis], 
               ones(comp_prof.shape)*M[:,1][:, newaxis], 
               ones(comp_prof.shape)*M[:,2][:, newaxis])),0, 3)
        
        #print 'M', M
        #print M[...,1].shape
        
        A = -lamda**2*re/pi*sl_c_lay
        B = lamda**2*re/pi*sl_m1_lay
        C = lamda**2*re/pi*sl_m2_lay
        g_0 = sin(theta*pi/180.0)
        #M = c_[ones(sl_c.shape), zeros(sl_c.shape), zeros(sl_c.shape)]
        chi, non_mag, mpy = lib.xrmr.create_chi(g_0, lamda, A, 0.0*A, 
                                       B, C, M, d)
        chi = tuple([c.sum(0) for c in chi[0] + chi[1] + chi[2]])
        #print chi[0]
        #M = c_[(M[:,0][:,newaxis]*sl_m1_tmp).sum(0)/sl_m1,
        #       (M[:,1][:,newaxis]*sl_m1_tmp).sum(0)/sl_m1,
        #       (M[:,2][:,newaxis]*sl_m1_tmp).sum(0)/sl_m1
        #       ].real
        #print M
        #print sl_m2
        #print sigma_c, sigma_m, A, B, d
        #print 'Uncompressed:', z.shape
        if sample.getCompress() == sample_string_choices['compress'][0]:
            #Compressing the profile..
            #z, pdens_c, pdens_m = edm.compress_profile2(z, sl_c, sl_m1, sample.getDsld_max())
            dsld_max = sample.getDsld_max()
            dchi_max = dsld_max*lamda**2*re/pi
            dsld_offdiag_max = sample.getDsld_offdiag_max()
            dsld_n_max = sample.getDsld_n_max()
            dabs_n_max = sample.getDabs_n_max()
            dmag_max = sample.getDmag_max() 
            dchi_od_max = dsld_offdiag_max*lamda**2*re/pi
            #z, pdens = edm.compress_profile_n(z, (sl_c, sl_m1, sl_m2), 
            #                                  (dsld_max, dsld_max, dsld_max))
            #sl_c, sl_m1, sl_m2 = pdens
            
            #print chi[0].shape
            #print sl_n.shape, mag_dens.shape, abs_n.shape, chi[0].shape
            index, z = edm.compress_profile_index_n(z, chi + (sl_n, mag_dens, abs_n), 
                                                (dchi_max, dchi_od_max, dchi_od_max,
                                                 dchi_od_max, dchi_max, dchi_od_max,
                                                 dchi_od_max, dchi_od_max, dchi_max,
                                                 dsld_n_max, dmag_max, dabs_n_max
                                                 ))
            reply = edm.create_compressed_profile((sl_c, sl_m1, sl_m2) + 
                                                  chi + (sl_n, mag_dens, abs_n), 
                                                  index)
            sl_c, sl_m1, sl_m2, chi_xx, chi_xy, chi_xz, chi_yx, chi_yy, chi_yz, chi_zx, chi_zy, chi_zz, sl_n, mag_dens, abs_n = reply
            non_mag = ((abs(chi_xy) < mag_limit)
                       *(abs(chi_xz) < mag_limit)
                       *(abs(chi_yz) < mag_limit))
            mpy = (abs(chi_yz) < mpy_limit)*(abs(chi_xy) < mpy_limit)*bitwise_not(non_mag)
            #print mpy
            chi = ((chi_xx, chi_xy, chi_xz),(chi_yx, chi_yy, chi_yz),(chi_zx, chi_zy, chi_zz))
        else:
            (chi_xx, chi_xy, chi_xz, chi_yx, chi_yy, chi_yz,chi_zx, chi_zy, chi_zz) = chi
            non_mag = ((abs(chi_xy) < mag_limit)
                       *(abs(chi_xz) < mag_limit)
                       *(abs(chi_yz) < mag_limit))
            non_mag[-1] = True
            mpy = (abs(chi_yz) < mpy_limit)*(abs(chi_xy) < mpy_limit)*bitwise_not(non_mag)
            chi = ((chi_xx, chi_xy, chi_xz),(chi_yx, chi_yy, chi_yz),(chi_zx, chi_zy, chi_zz))
        d = r_[z[1:] - z[:-1],1]
        #print 'Compressed: ', z.shape, sl_c.shape
        #print 'WARNING: M is ignored!'
    else:
        #print 'test'
        re = 2.8179402894e-5
        A = -lamda**2*re/pi*sl_c  
        B = lamda**2*re/pi*sl_m1
        C = lamda**2*re/pi*sl_m2
        g_0 = sin(theta*pi/180.0)
        #M = c_[ones(sl_c.shape), zeros(sl_c.shape), zeros(sl_c.shape)]
        chi, non_mag, mpy = lib.xrmr.create_chi(g_0, lamda, A, 0.0*A, 
                                                    B, C, M, d)
        #chi = ((chi_xx, chi_xy, chi_xz),(chi_yx, chi_yy, chi_yz),(chi_zx, chi_zy, chi_zz))
        #sl_c = pdens_c
        #sl_m1 = pdens_m
        #sl_m2 = sl_m1*0
        
        
        #print d.shape, A.shape
        #print A, B
        #M = c_[ones(sl_c.shape), zeros(sl_c.shape), zeros(sl_c.shape)]
        #print 'Sl_m2: ', sl_m2, 'END'
    return d, sl_c, sl_m1, sl_m2, M, chi, non_mag, mpy, sl_n, abs_n, mag_dens

def extract_anal_iso_pars(sample, instrument, theta, pol = '+', Q = None):
    ''' Note Q is only used for Neutron TOF'''
    re = 2.8179402894e-5
    lamda = instrument.getWavelength()
    parameters = sample.resolveLayerParameters()
    dens = array(parameters['dens'], dtype = float64)
    resdens = array(parameters['resdens'], dtype = float64)
    resmag = array(parameters['resmag'], dtype = float64)
    mag = abs(array(parameters['mag'], dtype = float64))
    dmag_l = array(parameters['dmag_l'], dtype = float64)
    dmag_u = array(parameters['dmag_u'], dtype = float64)
    dd_l = array(parameters['dd_l'], dtype = float64)
    dd_u = array(parameters['dd_u'], dtype = float64)
    #print [type(f) for f in parameters['f']]
    f = array(parameters['f'], dtype = complex128) + (1-1J)*1e-20
    fr = array(parameters['fr'], dtype = complex128) + (1-1J)*1e-20
    fm1 = array(parameters['fm1'], dtype = complex128) + (1-1J)*1e-20
    fm2 = array(parameters['fm2'], dtype = complex128) + (1-1J)*1e-20
    
    d = array(parameters['d'], dtype = float64)
    
    #sl_m2 = dens*resdens*resmag*fm2 #mag is multiplied in later
    
    #g_0 = sin(theta*pi/180.0)
    theta = theta*pi/180.0
    phi = array(parameters['phi_m'], dtype = float64)*pi/180.0
    theta_m = array(parameters['theta_m'], dtype = float64)*pi/180.0
    sl_c = (dens*(f + resdens*fr))[:, newaxis]*ones(theta.shape)
    # This is wrong!!! I should have a cos theta dep.
    #sl_m1 = (dens*resdens*resmag*fm1)[:, newaxis]*cos(theta - theta_m[:,newaxis])*cos(phi[:,newaxis])
    sl_m1 = (dens*resdens*resmag*mag*fm1)[:, newaxis]*cos(theta)
    #print M
    #print sl_c.shape, sl_m1.shape
    sigma_c = array(parameters['sigma_c'], dtype = float64)
    sigma_l = array(parameters['sigma_ml'], dtype = float64)
    sigma_u = array(parameters['sigma_mu'], dtype = float64)
    
    theory = instrument.getTheory()
    
    if (theory == 0 or theory == instrument_string_choices['theory'][0] or
        theory == 1 or theory == instrument_string_choices['theory'][1]):
        if pol == '+':
            n = 1 - lamda**2*re/pi*(sl_c + sl_m1)/2.0
            n_l = 1 - lamda**2*re/pi*(sl_c + sl_m1*(1. + dmag_l)[:,newaxis])/2.0
            n_u = 1 - lamda**2*re/pi*(sl_c + sl_m1*(1. + dmag_u)[:,newaxis])/2.0
        elif pol == '-':
            n = 1 - lamda**2*re/pi*(sl_c - sl_m1)/2.0
            n_l = 1 - lamda**2*re/pi*(sl_c - sl_m1*(1. + dmag_l)[:,newaxis])/2.0
            n_u = 1 - lamda**2*re/pi*(sl_c - sl_m1*(1. + dmag_u)[:,newaxis])/2.0
    elif (theory == 2 or theory == instrument_string_choices['theory'][2] or
          theory == 3 or theory == instrument_string_choices['theory'][3]):
        b = (array(parameters['b'], dtype = complex128)*1e-5)[:, newaxis]*ones(theta.shape)
        abs_xs = (array(parameters['xs_ai'], dtype = complex128)*(1e-4)**2)[:, newaxis]*ones(theta.shape)
        wl = instrument.getWavelength()
        #print b
        #print b.shape, abs_xs.shape, theta.shape
        sld = dens[:, newaxis]*(wl**2/2/pi*sqrt(b**2 - (abs_xs/2.0/wl)**2) - 
                               1.0J*abs_xs*wl/4/pi)
        msld = (2.645e-5*mag*dens*wl**2/2/pi)[:,newaxis]*ones(theta.shape)
        if pol in ['++', 'uu']:
            n = 1.0 - sld - msld
            n_l = 1.0 - sld - msld*(1.0 + dmag_l)[:, newaxis]
            n_u = 1.0 - sld - msld*(1.0 + dmag_u)[:, newaxis]
        if pol in ['--', 'dd']:
            n = 1.0 - sld + msld
            n_l = 1.0 - sld + msld*(1.0 + dmag_l)[:, newaxis]
            n_u = 1.0 - sld + msld*(1.0 + dmag_u)[:, newaxis]
    elif (theory == 4 or theory == instrument_string_choices['theory'][4]):
        wl = 4*pi*sin(instrument.getIncang()*pi/180)
        b = (array(parameters['b'], dtype = complex128)*1e-5)[:, newaxis]*ones(wl.shape)
        abs_xs = (array(parameters['xs_ai'], dtype = complex128)*(1e-4)**2)[:, newaxis]*ones(wl.shape)
        #print b
        #print b.shape, abs_xs.shape, theta.shape
        sld = dens[:, newaxis]*(wl**2/2/pi*sqrt(b**2 - (abs_xs/2.0/wl)**2) - 
                               1.0J*abs_xs*wl/4/pi)
        msld = (2.645e-5*(mag*dens)[:,newaxis]*wl**2/2/pi)
        if pol in ['++', 'uu']:
            n = 1.0 - sld - msld
            n_l = 1.0 - sld - msld*(1.0 + dmag_l)[:, newaxis]
            n_u = 1.0 - sld - msld*(1.0 + dmag_u)[:, newaxis]
        if pol in ['--', 'dd']:
            n = 1.0 - sld + msld
            n_l = 1.0 - sld + msld*(1.0 + dmag_l)[:, newaxis]
            n_u = 1.0 - sld + msld*(1.0 + dmag_u)[:, newaxis]
    else:
        raise ValueError('An unexpected value of pol was given. Value: %s'%(pol,)) 
    #print n.shape, d.shape
    #print 'test'
    #print all(n_u == n), all(n_l == n) 
    d = d - dd_u - dd_l
    d = d*(d >= 0)
    return n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l

def reflectivity_xmag(sample, instrument, theta, TwoThetaQz):
    use_slicing = sample.getSlicing()
    if use_slicing == 0 or use_slicing == sample_string_choices['slicing'][0]:
        R = slicing_reflectivity(sample, instrument, theta, TwoThetaQz)
    elif use_slicing == 1 or use_slicing == sample_string_choices['slicing'][1]:
        R = analytical_reflectivity(sample, instrument, theta, TwoThetaQz)
    else:
        raise ValueError('Unkown input to the slicing parameter')
    return R
    
def analytical_reflectivity(sample, instrument, theta, TwoThetaQz):
    lamda = instrument.getWavelength()
    theory = instrument.getTheory()
    if theory == 0 or theory == instrument_string_choices['theory'][0]:
        raise NotImplementedError('Full calculations only implemented for slicing so far')
    elif theory == 1 or theory == instrument_string_choices['theory'][1]:
        pol = instrument.getXpol()
        re = 2.82e-13*1e2/1e-10
        Q = 4*pi/lamda*sin(theta*pi/180)
        if pol == 0 or pol == instrument_string_choices['xpol'][0]:
            # circ +
            pars = extract_anal_iso_pars(sample, instrument, theta, '+')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            #chi_temp = chi[0][0][:,newaxis] - 1.0J*chi[2][1][:,newaxis]*cos(theta*pi/180)
            #n = 1 + chi_temp/2.0
            #print n.shape, theta.shape, d.shape
            #print 'Ord'
            #R = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), pars[0], pars[1], zeros(pars[1].shape))
            R = ables.ReflQ_mag(Q, lamda, n.T, d, sigma_c, n_u.T, dd_u, sigma_u, n_l.T, dd_l, sigma_l)
        elif pol == 1 or pol == instrument_string_choices['xpol'][1]:
            # circ -
            pars = extract_anal_iso_pars(sample, instrument, theta, '-')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            #chi_temp = chi[0][0][:,newaxis] + 1.0J*chi[2][1][:,newaxis]*cos(theta*pi/180)
            #n = 1 + chi_temp/2.0
            #R = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), pars[0], pars[1], zeros(pars[1].shape))
            R = ables.ReflQ_mag(Q, lamda, n.T, d, sigma_c, n_u.T, dd_u, sigma_u, n_l.T, dd_l, sigma_l)
        elif pol == 2 or pol == instrument_string_choices['xpol'][2]:
            # tot
            pars = extract_anal_iso_pars(sample, instrument, theta, '-')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            #chi_temp = chi[0][0][:,newaxis] + 1.0J*chi[2][1][:,newaxis]*cos(theta*pi/180)
            #n = 1 + chi_temp/2.0
            #Rm = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), pars[0], pars[1], zeros(pars[1].shape))
            Rm = ables.ReflQ_mag(Q, lamda, n.T, d, sigma_c, n_u.T, dd_u, sigma_u, n_l.T, dd_l, sigma_l)
            pars = extract_anal_iso_pars(sample, instrument, theta, '+')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            #chi_temp = chi[0][0][:,newaxis] - 1.0J*chi[2][1][:,newaxis]*cos(theta*pi/180)
            #n = 1 + chi_temp/2.0
            #Rp = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), pars[0], pars[1], zeros(pars[1].shape))
            Rp = ables.ReflQ_mag(Q, lamda, n.T, d, sigma_c, n_u.T, dd_u, sigma_u, n_l.T, dd_l, sigma_l)
            R = (Rp + Rm)/2.0
            #raise ValueError('Variable pol has an unvalid value')
        elif pol == 3 or pol == instrument_string_choices['xpol'][3]:
            # ass
            pars = extract_anal_iso_pars(sample, instrument, theta, '-')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            
            #chi_temp = chi[0][0][:,newaxis] + 1.0J*chi[2][1][:,newaxis]*cos(theta*pi/180)
            #n = 1 + chi_temp/2.0
            #Rm = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), pars[0], pars[1], zeros(pars[1].shape))
            Rm = ables.ReflQ_mag(Q, lamda, n.T, d, sigma_c, n_u.T, dd_u, sigma_u, n_l.T, dd_l, sigma_l)
            pars = extract_anal_iso_pars(sample, instrument, theta, '+')
            n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
            #chi_temp = chi[0][0][:,newaxis] - 1.0J*chi[2][1][:,newaxis]*cos(theta*pi/180)
            #n = 1 + chi_temp/2.0
            #Rp = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), pars[0], pars[1], zeros(pars[1].shape))
            Rp = ables.ReflQ_mag(Q, lamda, n.T, d, sigma_c, n_u.T, dd_u, sigma_u, n_l.T, dd_l, sigma_l)
            R = (Rp - Rm)/(Rp + Rm)
            #raise ValueError('Variable pol has an unvalid value')
        else:
            raise ValueError('Variable pol has an unvalid value')

    elif theory == 2 or theory == instrument_string_choices['theory'][2]:
        # neutron spin-pol calcs
        Q = 4*pi/lamda*sin(theta*pi/180)
        pars = extract_anal_iso_pars(sample, instrument, theta, instrument.getNpol())
        n, d, sigma_c, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l = pars
        R = ables.ReflQ_mag(Q, lamda, n.T, d, sigma_c, n_u.T, dd_u, sigma_u, n_l.T, dd_l, sigma_l)
        #raise NotImplementedError('Neutron calcs not implemented')
    elif theory == 3 or theory == instrument_string_choices['theory'][3]:
        # neutron spin-flip calcs
        raise NotImplementedError('Neutron calcs not implemented')
    elif theory == 4 or theory == instrument_string_choices['theory'][4]:
        # neutron spin-flip calcs
        raise NotImplementedError('TOF Neutron calcs not implemented')
    else:
        raise ValueError('The given theory mode deos not exist')
    return R

def slicing_reflectivity(sample, instrument, theta, TwoThetaQz):
    lamda = instrument.getWavelength()
    parameters = sample.resolveLayerParameters()
    
    d, sl_c, sl_m1, sl_m2, M, chi, non_mag, mpy, sl_n, abs_n, mag_dens = compose_sld(sample, instrument, theta)
    #re = 2.8179402894e-5
    #A = -lamda**2*re/pi*sl_c
    #B = lamda**2*re/pi*sl_m1
    #C = lamda**2*re/pi*sl_m2
    
    g_0 = sin(theta*pi/180.0)
    #print A[::-1], B[::-1], d[::-1], M[::-1], lamda, g_0
    theory = instrument.getTheory()
    # Full theory
    if Buffer.g_0 != None:
        g0_ok = Buffer.g_0.shape == g_0.shape
        if g0_ok:
            g0_ok = any(not_equal(Buffer.g_0,g_0))
    else:
        g0_ok = False
    if  theory == 0 or theory == instrument_string_choices['theory'][0]:
        if (Buffer.parameters != parameters or Buffer.coords != instrument.getCoords()
            or not g0_ok or Buffer.wavelength != lamda):
            #W = lib.xrmr.calc_refl(g_0, lamda, A[::-1], 0.0*A[::-1], B[::-1], C[::-1], M[::-1], d[::-1])
            #print 'Calc W'
            W = lib.xrmr.do_calc(g_0, lamda, chi, d, non_mag, mpy)
            Buffer.W = W
            Buffer.parameters = parameters.copy()
            Buffer.coords = instrument.getCoords()
            Buffer.g_0 = g_0.copy()
            Buffer.wavelength = lamda
        else:
            #print 'Reusing W'
            W = Buffer.W
        trans = ones(W.shape, dtype = complex128); trans[0,1] = 1.0J; trans[1,1] = -1.0J; trans = trans/sqrt(2)
        #Wc = lib.xrmr.dot2(trans, lib.xrmr.dot2(W, lib.xrmr.inv2(trans)))
        Wc = lib.xrmr.dot2(trans, lib.xrmr.dot2(W, conj(lib.xrmr.inv2(trans))))
        #Different polarization channels:
        pol = instrument.getXpol()
        if pol == 0 or pol == instrument_string_choices['xpol'][0]:
            # circ +
            R = abs(Wc[0,0])**2 + abs(Wc[1,0])**2
        elif pol == 1 or pol == instrument_string_choices['xpol'][1]:
            # circ -
            R = abs(Wc[1,1])**2 + abs(Wc[0,1])**2
        elif pol == 2 or pol == instrument_string_choices['xpol'][2]:
            # tot
            R = (abs(W[0,0])**2 + abs(W[1,0])**2 + abs(W[0,1])**2 + abs(W[1,1])**2)/2
        elif pol == 3 or pol == instrument_string_choices['xpol'][3]:
            # ass
            R = 2*(W[0,0]*W[0,1].conj() + W[1,0]*W[1,1].conj()).imag/(abs(W[0,0])**2 + abs(W[1,0])**2 + abs(W[0,1])**2 + abs(W[1,1])**2)
        elif pol == 4 or pol == instrument_string_choices['xpol'][4]:
            # sigma
            R = abs(W[0,0])**2 + abs(W[1,0])**2
        elif pol == 5 or pol == instrument_string_choices['xpol'][5]:
            # pi
            R = abs(W[0,1])**2 + abs(W[1,1])**2
        else:
            raise ValueError('Variable pol has an unvalid value')
    # Simplified theory
    elif theory == 1 or theory == instrument_string_choices['theory'][1]:
        pol = instrument.getXpol()
        re = 2.8179402894e-5
        if pol == 0 or pol == instrument_string_choices['xpol'][0]:
            # circ +
            n = 1 - lamda**2*re/pi*(sl_c[:,newaxis] + sl_m1[:, newaxis]*cos(theta*pi/180))/2.0
            #print n.shape, theta.shape, d.shape
            R = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
        elif pol == 1 or pol == instrument_string_choices['xpol'][1]:
            # circ -
            n = 1 - lamda**2*re/pi*(sl_c[:,newaxis] - sl_m1[:, newaxis]*cos(theta*pi/180))/2.0
            R = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
        elif pol == 2 or pol == instrument_string_choices['xpol'][2]:
            # tot
            n = 1 - lamda**2*re/pi*(sl_c[:,newaxis] - sl_m1[:, newaxis]*cos(theta*pi/180))/2.0
            Rm = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
            n = 1 - lamda**2*re/pi*(sl_c[:,newaxis] + sl_m1[:, newaxis]*cos(theta*pi/180))/2.0
            Rp = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
            R = (Rp + Rm)/2.0
            #raise ValueError('Variable pol has an unvalid value')
        elif pol == 3 or pol == instrument_string_choices['xpol'][3]:
            # ass
            n = 1 - lamda**2*re/pi*(sl_c[:,newaxis] - sl_m1[:, newaxis]*cos(theta*pi/180))/2.0
            Rm = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
            n = 1 - lamda**2*re/pi*(sl_c[:,newaxis] + sl_m1[:, newaxis]*cos(theta*pi/180))/2.0
            Rp = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
            R = (Rp - Rm)/(Rp + Rm)
            #raise ValueError('Variable pol has an unvalid value')
        else:
            raise ValueError('Variable pol has an unvalid value')
    # Neutron spin pol calculations normal mode
    elif theory == 2 or theory == instrument_string_choices['theory'][2]:
        sl_n = sl_n*1e-5
        abs_n = abs_n*1e-8
        sl_n = (lamda**2/2/pi*sqrt(sl_n**2 - (abs_n/2.0/lamda)**2) - 
                               1.0J*abs_n*lamda/4/pi)
        sl_nm = 2.645e-5*mag_dens*lamda**2/2/pi
        pol = instrument.getNpol()
        if pol in ['++', 'uu']:
            n = 1.0 - sl_n - sl_nm
            R = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), 
                                   n[:, newaxis]*ones(theta.shape), d, zeros(d.shape))
        if pol in ['--', 'dd']:
            n = 1.0 - sl_n + sl_nm
            R = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), 
                                   n[:,newaxis]*ones(theta.shape), d, zeros(d.shape))
    # Neutron calcs spin-flip 
    elif theory == 3 or theory == instrument_string_choices['theory'][3]:
        raise NotImplementedError('Spin flip calculations not implemented yet')
    
    elif theory == 4 or theory == instrument_string_choices['theory'][4]:
        # neutron TOF calculations
        incang = instrument.getIncang()
        lamda = 4*pi*sin(incang*pi/180)/TwoThetaQz
        sl_n = sl_n[:,newaxis]*1e-5
        abs_n = abs_n[:,newaxis]*1e-8
        sl_n = (lamda**2/2/pi*sqrt(sl_n**2 - (abs_n/2.0/lamda)**2) - 
                               1.0J*abs_n*lamda/4/pi)
        sl_nm = 2.645e-5*mag_dens[:,newaxis]*lamda**2/2/pi
        pol = instrument.getNpol()
        
        if pol in ['++', 'uu']:
            n = 1.0 - sl_n - sl_nm
            R = Paratt.Refl_nvary2(incang*ones(lamda.shape), lamda, 
                                   n, d, zeros(d.shape))
        if pol in ['--', 'dd']:
            n = 1.0 - sl_n + sl_nm
            R = Paratt.Refl_nvary2(incang*ones(lamda.shape), lamda, 
                                   n, d, zeros(d.shape))
        #raise NotImplementedError('TOF Neutron calcs not implemented')
    else:
        raise ValueError('The given theory mode deos not exist')
    return R

def footprint_correction(instrument, theta):
    foocor = 1.0
    footype = instrument.getFootype()
    beamw = instrument.getBeamw()
    samlen = instrument.getSamplelen()
    if footype == 0 or footype == instrument_string_choices['footype'][0]:
        foocor = 1.0
    elif footype == 1 or footype == instrument_string_choices['footype'][1]:
        foocor = GaussIntensity(theta, samlen/2.0, samlen/2.0, beamw)
    elif footype == 2 or footype == instrument_string_choices['footype'][2]:
        foocor = SquareIntensity(theta, samlen, beamw)
    else:
        raise ValueError('Variable footype has an unvalid value')
    return foocor
    
def correct_reflectivity(R, instrument, foocor, TwoThetaQz, weight):
    restype = instrument.getRestype()
    if restype == 0 or restype == instrument_string_choices['restype'][0]:
        R = R[:]*foocor
    elif restype == 1 or restype == instrument_string_choices['restype'][1]:
        R = ConvoluteFast(TwoThetaQz,R[:]*foocor, instrument.getRes(),\
            range = instrument.getResintrange())
    elif restype == 2 or restype == instrument_string_choices['restype'][2]:
        R = ConvoluteResolutionVector(TwoThetaQz,R[:]*foocor, weight)
    elif restype == 3 or restype == instrument_string_choices['restype'][3]:
        R = ConvoluteFastVar(TwoThetaQz,R[:]*foocor, instrument.getRes(),\
          range = instrument.getResintrange())
    else:
        raise ValueError('Variable restype has an unvalid value')
    return R
    

SimulationFunctions = {'Specular':Specular,\
                        'OffSpecular':OffSpecular,\
                        'SLD': SLD_calculations}

import lib.refl as Refl
(Instrument, Layer, Stack, Sample) = Refl.MakeClasses(InstrumentParameters,\
        LayerParameters,StackParameters,\
         SampleParameters, SimulationFunctions, ModelID)


if __name__=='__main__':
    pass