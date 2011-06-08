''' <h1> Library for specular magnetic x-ray reflectivity</h1>
The magnetic reflectivity is calculated according to: S.A. Stephanov and S.K Shina PRB 61 15304.
Note: The documentation is not updated from the interdiff model!
<h2>Classes</h2>
<h3>Layer</h3>
<code> Layer(b = 0.0, d = 0.0, f = 0.0+0.0J, dens = 1.0, magn_ang = 0.0, magn = 0.0, sigma = 0.0)</code>
    <dl>
    <dt><code><b>d</b></code></dt>
    <dd>The thickness of the layer in AA (Angstroms = 1e-10m)</dd>
    <dt><code><b>f</b></code></dt>
    <dd>The x-ray scattering length per formula unit in electrons. To be strict it is the
    number of Thompson scattering lengths for each formula unit.</dd>
    <dt><code><b>dens</b></code></dt>
    <dd>The density of formula units in units per Angstroms. Note the units!</dd>
    <dt><code><b>sigmai</b></code></dt>
    <dd>The root mean square <em>interdiffusion</em> of the top interface of the layer in Angstroms.</dd>		
    <dt><code><b>sigmar</b></code></dt>
    <dd>The root mean square <em>roughness</em> of the top interface of the layer in Angstroms.</dd>
    </dl>
<h3>Stack</h3>
<code> Stack(Layers = [], Repetitions = 1)</code>
    <dl>
    <dt><code><b>Layers</b></code></dt>
    <dd>A <code>list</code> consiting of <code>Layer</code>s in the stack
    the first item is the layer closest to the bottom</dd>
    <dt><code><b>Repetitions</b></code></dt>
    <dd>The number of repsetions of the stack</dd>
    </dl>
<h3>Sample</h3>
<code> Sample(Stacks = [], Ambient = Layer(), Substrate = Layer(), eta_z = 10.0,
    eta_x = 10.0, h = 1.0)</code>
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
    <dt><code><b>eta_z</b></code></dt>
    <dd>The out-of plane (vertical) correlation length of the roughness
    in the sample. Given in AA. </dd>
    <dt><code><b>eta_x</b></code></dt>
    <dd>The in-plane global correlation length (it is assumed equal for all layers).
    Given in AA.</dd>
    <dt><code><b>h</b></code></dt>
    <dd>The jaggedness parameter, should be between 0 and 1.0. This describes
    how jagged the interfaces are. This is also a global parameter for all
    interfaces.</dd>
    </dl>
    
<h3>Instrument</h3>
<code>Instrument(wavelength = 1.54, coords = 'tth',
     I0 = 1.0 res = 0.001, restype = 'no conv', respoints = 5, resintrange = 2,
     beamw = 0.01, footype = 'no corr', samplelen = 10.0, taylor_n = 1)</code>
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
    <dt><code><b>taylor_n</b></code></dt>
    <dd>The number terms taken into account in the taylor expansion of 
    the fourier integral of the correlation function. More terms more accurate
    calculation but also much slower.</dd>
'''

import lib.xrmr
import lib.edm_slicing as edm
try:
    import lib.paratt_weave as Paratt
except StandardError,S:
    print 'Not using inline c code for reflectivity calcs - can not import module'
    print S
    import lib.paratt as Paratt
    


from numpy import *
from scipy.special import erf
from lib.instrument import *

mag_limit = 1e-8
mpy_limit = 1e-8
# Preamble to define the parameters needed for the models outlined below:
ModelID='StephanovXRMR'
# Automatic loading of parameters possible by including this list
__pars__ = ['Layer', 'Stack', 'Sample', 'Instrument']
# Used for making choices in the GUI
instrument_string_choices = {'coords': ['q','tth'],
                             'restype': ['no conv', 'fast conv',
                                         'full conv and varying res.', 
                                         'fast conv + varying res.'],
                             'footype': ['no corr', 'gauss beam', 
                                         'square beam'],
                             'pol':['circ+','circ-','tot', 'ass', 'sigma', 'pi'],
                             'theory': ['full', 'non-anisotropic'],
                             #'compress':['yes', 'no'],
                             #'slicing':['yes', 'no'],
     }


    
InstrumentParameters={'wavelength':1.54,'coords':'tth','I0':1.0,'res':0.001,\
    'restype':'no conv','respoints':5,'resintrange':2,'beamw':0.01,'footype': 'no corr',\
    'samplelen':10.0, 'Ibkg': 0.0, 'pol':'circ+', 'theory':'full',}
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
InstrumentGroups = [('General', ['wavelength', 'coords', 'I0', 'Ibkg']),
                    ('Resolution', ['restype', 'res', 'respoints', 'resintrange']),
                    ('XRMR', ['pol', 'theory']),
                    ('Footprint', ['footype', 'beamw', 'samplelen',]),
                    ]
InstrumentUnits={'wavelength':'AA','coords':'','I0':'arb.','res':'[coord]',
                 'restype':'','respoints':'pts.','resintrange': '[coord]',
                 'beamw':'mm','footype': '',\
                 'samplelen':'mm', 'Ibkg': 'arb.', 'pol':'', 
                 'theory':'',}

LayerParameters = {'dens':1.0, 'd':0.0, 'f': (0.0 + 1e-20J), 
                   'fr':(0.0 + 1e-20J),
                   'fm1':(0.0 + 1e-20J), 'fm2':(0.0 + 1e-20J), 
                   'phi_m': 0.0, 'theta_m': 0.0, 'resdens': 1.0,
                   'resmag': 1.0,
                   'sigma_c': 0.0, 'sigma_m': 0.0, 'mag':1.0,
                   'dmag_l': 1.0, 'dmag_u': 1.0, 'dd_m':0.0,
                   'b': 1e-20J
                   #'dtheta_l': 0.0, 'dtheta_u':0.0, 'dphi_l':0.0, 'dphi_u':0.0,
                   }
LayerUnits = {'dens':'at./AA^3', 'd':'AA', 'f': 'el.', 
                   'fr':'el.',
                   'fm1':'el./mu_B', 'fm2':'el./mu_B', 
                   'phi_m': 'deg.', 'theta_m': 'deg.', 'resdens': 'rel.',
                   'resmag': 'rel.',
                   'sigma_c': 'AA', 'sigma_m': 'AA', 'mag': 'mu_B',
                   'dmag_l': 'rel.', 'dmag_u': 'rel.', 'dd_m':'AA',
                   'b': 'fm'
                   #'dtheta_l': 0.0, 'dtheta_u':0.0, 'dphi_l':0.0, 'dphi_u':0.0,
                   }
LayerGroups = [('Scatt. len.', ['b', 'f', 'fr', 'fm1', 'fm2']), 
                ('Magnetism', ['mag', 'resmag', 'phi_m','theta_m']), 
                ('Misc.', ['sigma_c', 'dens', 'resdens', 'd']),
                ('Interf. Mag. Mom.', ['dmag_l', 'dmag_u', 'sigma_m', 'dd_m'])
                ]
#('Interf. Mag. Ang.', ('dtheta_l', 'dtheta_u', 'dphi_l', 'dphi_u'))
StackParameters = {'Layers':[], 'Repetitions':1}
SampleParameters = {'Stacks':[], 'Ambient':None, 'Substrate':None, 
                    'compress':'yes', 'slicing':'no', 'slice_depth':1.0,
                    'sld_mult':4.0, 'sld_buffer': 20.0, 'sld_delta': 5.0,
                    'dsld_max':0.1, 'dsld_offdiag_max':0.1,
                    }
SampleGroups = [['Slicing', [ 'slicing', 'slice_depth', 'sld_mult', 'sld_buffer', 
                             'sld_delta']],
                ['Compression', ['compress', 'dsld_max', 'dsld_offdiag_max']],
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
    
    R = reflectivity_xmag(sample, instrument, theta)
    pol = instrument.getPol()
    if pol != 3 and pol != instrument_string_choices['pol'][3]:
        #FootprintCorrections
        foocor = footprint_correction(instrument, theta)
        R = correct_reflectivity(R, instrument, foocor, TwoThetaQz, weight)
        return R*instrument.getI0() + instrument.getIbkg()
    else:
        foocor = footprint_correction(instrument, theta)*0 + 1.0
        R = correct_reflectivity(R, instrument, foocor, TwoThetaQz, weight)
        return R

def OffSpecular(TwoThetaQz, ThetaQx, sample, instrument):
   raise NotImplementedError('Off specular calculations are not implemented for magnetic x-ray reflectivity')


def SLD_calculations(z, sample, inst):
    ''' Calculates the scatteringlength density as at the positions z
    '''
    lamda = inst.getWavelength()
    d, sl_c, sl_m1, sl_m2, M, chi, non_mag, mpy = compose_sld(sample, inst, array([0.0,]))
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
    return {'sl_xx':chi[0][0].real*c, 'sl_xy':chi[0][1].real*c, 'sl_xz':chi[0][2].real*c,
            'sl_yy':chi[1][1].real*c,'sl_yz':chi[1][2].real*c,'sl_zz':chi[2][2].real*c,
            'z':z}

def compose_sld(sample, instrument, theta):
    lamda = instrument.getWavelength()
    parameters = sample.resolveLayerParameters()
    dens = array(parameters['dens'], dtype = float64)
    resdens = array(parameters['resdens'], dtype = float64)
    resmag = array(parameters['resmag'], dtype = float64)
    mag = array(parameters['mag'], dtype = float64)
    dmag_l = array(parameters['dmag_l'], dtype = float64)
    dmag_u = array(parameters['dmag_u'], dtype = float64)
    dd_m = array(parameters['dd_m'], dtype = float64)
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
    sigma_m = sqrt(array(parameters['sigma_m'], dtype = float64)**2 + sigma_c**2)
    #print A, B
    #print type(sample.getSld_buffer())
    if sample.getSlicing() == sample_string_choices['slicing'][0]:
        dz = sample.getSlice_depth()
        reply= edm.create_profile_cm(d[1:-1], sigma_c[:-1].real, sigma_m[:-1].real, 
                                     [edm.erf_profile]*len(sl_c),
                                     [edm.erf_interf]*len(sigma_c[:]),
                                     dmag_l, dmag_u, mag, dd_m,
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
        #print comp_prof.shape, sl_m1_lay.shape, sl_c_lay.shape
        M = rollaxis(array((ones(comp_prof.shape)*M[:,0][:, newaxis], 
               ones(comp_prof.shape)*M[:,1][:, newaxis], 
               ones(comp_prof.shape)*M[:,2][:, newaxis])),0, 3)
        
        #print 'M', M
        #print M[...,1].shape
        re = 2.8179402894e-5
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
            dchi_od_max = dsld_offdiag_max*lamda**2*re/pi
            #z, pdens = edm.compress_profile_n(z, (sl_c, sl_m1, sl_m2), 
            #                                  (dsld_max, dsld_max, dsld_max))
            #sl_c, sl_m1, sl_m2 = pdens
            
            #print chi[0].shape
            index, z = edm.compress_profile_index_n(z, chi, 
                                                (dchi_max, dchi_od_max, dchi_od_max,
                                                 dchi_od_max, dsld_max, dchi_od_max,
                                                 dchi_od_max, dchi_od_max, dsld_max))
            reply = edm.create_compressed_profile((sl_c, sl_m1, sl_m2) + 
                                                  chi, 
                                                  index)
            sl_c, sl_m1, sl_m2, chi_xx, chi_xy, chi_xz, chi_yx, chi_yy, chi_yz, chi_zx, chi_zy, chi_zz = reply
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
    return d, sl_c, sl_m1, sl_m2, M, chi, non_mag, mpy

def reflectivity_xmag(sample, instrument, theta):
    lamda = instrument.getWavelength()
    parameters = sample.resolveLayerParameters()
    
    d, sl_c, sl_m1, sl_m2, M, chi, non_mag, mpy = compose_sld(sample, instrument, theta)
    #re = 2.8179402894e-5
    #A = -lamda**2*re/pi*sl_c
    #B = lamda**2*re/pi*sl_m1
    #C = lamda**2*re/pi*sl_m2
    g_0 = sin(theta*pi/180.0)
    #print A[::-1], B[::-1], d[::-1], M[::-1], lamda, g_0
    theory = instrument.getTheory()
    # Full theory
    if  theory == 0 or theory == instrument_string_choices['theory'][0]:
        if (Buffer.parameters != parameters or Buffer.coords != instrument.getCoords()
            or any(not_equal(Buffer.g_0,g_0)) or Buffer.wavelength != lamda):
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
        Wc = lib.xrmr.dot2(trans, lib.xrmr.dot2(W, lib.xrmr.inv2(trans)))
        #Different polarization channels:
        pol = instrument.getPol()
        if pol == 0 or pol == instrument_string_choices['pol'][0]:
            # circ +
            R = abs(Wc[0,0])**2 + abs(Wc[0,1])**2
        elif pol == 1 or pol == instrument_string_choices['pol'][1]:
            # circ -
            R = abs(Wc[1,1])**2 + abs(Wc[1,0])**2
        elif pol == 2 or pol == instrument_string_choices['pol'][2]:
            # tot
            R = (abs(W[0,0])**2 + abs(W[1,0])**2 + abs(W[0,1])**2 + abs(W[1,1])**2)/2
        elif pol == 3 or pol == instrument_string_choices['pol'][3]:
            # ass
            R = 2*(W[0,0]*W[0,1].conj() + W[1,0]*W[1,1].conj()).imag/(abs(W[0,0])**2 + abs(W[1,0])**2 + abs(W[0,1])**2 + abs(W[1,1])**2)
        elif pol == 4 or pol == instrument_string_choices['pol'][4]:
            # sigma
            R = abs(W[0,0])**2 + abs(W[0,1])**2
        elif pol == 5 or pol == instrument_string_choices['pol'][5]:
            # pi
            R = abs(W[1,0])**2 + abs(W[1,1])**2
        else:
            raise ValueError('Variable pol has an unvalid value')
    # Simplified theory
    elif theory == 1 or theory == instrument_string_choices['theory'][1]:
        pol = instrument.getPol()
        re = 2.82e-13*1e2/1e-10
        if pol == 0 or pol == instrument_string_choices['pol'][0]:
            # circ +
            chi_temp = chi[0][0][:,newaxis] - 1.0J*chi[2][1][:,newaxis]*cos(theta*pi/180)
            n = 1 + chi_temp/2.0
            #print n.shape, theta.shape, d.shape
            R = Paratt.Refl_nvary2(theta, lamda, n, d, zeros(d.shape))
        elif pol == 1 or pol == instrument_string_choices['pol'][1]:
            # circ -
            chi_temp = chi[0][0][:,newaxis] + 1.0J*chi[2][1][:,newaxis]*cos(theta*pi/180)
            n = 1 + chi_temp/2.0
            R = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
        elif pol == 2 or pol == instrument_string_choices['pol'][2]:
            # tot
            chi_temp = chi[0][0][:,newaxis] + 1.0J*chi[2][1][:,newaxis]*cos(theta*pi/180)
            n = 1 + chi_temp/2.0
            Rm = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
            chi_temp = chi[0][0][:,newaxis] - 1.0J*chi[2][1][:,newaxis]*cos(theta*pi/180)
            n = 1 + chi_temp/2.0
            Rp = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
            R = (Rp + Rm)/2.0
            #raise ValueError('Variable pol has an unvalid value')
        elif pol == 3 or pol == instrument_string_choices['pol'][3]:
            # ass
            chi_temp = chi[0][0][:,newaxis] + 1.0J*chi[2][1][:,newaxis]*cos(theta*pi/180)
            n = 1 + chi_temp/2.0
            Rm = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
            chi_temp = chi[0][0][:,newaxis] - 1.0J*chi[2][1][:,newaxis]*cos(theta*pi/180)
            n = 1 + chi_temp/2.0
            Rp = Paratt.Refl_nvary2(theta, lamda*ones(theta.shape), n, d, zeros(d.shape))
            R = (Rp - Rm)/(Rp + Rm)
            #raise ValueError('Variable pol has an unvalid value')
        else:
            raise ValueError('Variable pol has an unvalid value')
    else:
        raise ValueError('Variable theory has an unvalid value')
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