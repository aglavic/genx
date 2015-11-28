''' <h1>Library for combined x-ray and neutrons simulations.</h1>
<p> The neutron simulations is capable of handling non-magnetic,
magnetic non-spin flip as well as neutron spin-flip reflectivity.
The model works with scattering lengths densities directly.
</p>
<h2>Classes</h2>
<h3>Layer</h3>
<code> Layer(sld_n=0.0, sld_x=1e-20J, d = 0.0, magn_ang = 0.0, sld_m = 0.0, sigma = 0.0)</code>
    <dl>
    <dt><code><b>sld_n</b></code></dt>
    <dd>The neutron scattering length density in 1e-6 1/AA^2</dd>
    <dt><code><b>d</b></code></dt>
    <dd>The thickness of the layer in AA (Angstroms = 1e-10m)</dd>
    <dt><code><b>sld_x</b></code></dt>
    <dd>The x-ray scattering length density in 1e-6 1/AA^2</dd>
    <dt><code><b>magn_ang</b></code></dt>
    <dd>The angle of the magnetic moment in degress. 0 degrees correspond to
    a moment collinear with the neutron spin.</dd>
    <dt><code><b>sld_m</b></code></dt>
    <dd>The neutron magnetic scattering length density in 1e-6 1/AA^2</dd>
    <dt><code><b>sigma</b></code></dt>
    <dd>The root mean square roughness of the top interface of the layer in Angstroms.</dd>
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
<code> Sample(Stacks = [], Ambient = Layer(), Substrate = Layer())</code>
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
    </dl>
    
<h3>Instrument</h3>
<code>Instrument(probe = 'x-ray', wavelength = 1.54, coords = 'tth',
     I0 = 1.0 res = 0.001, restype = 'no conv', respoints = 5, resintrange = 2,
     beamw = 0.01, footype = 'no corr', samplelen = 10.0, incangle = 0.0, pol = 'uu')</code>
    <dl>
    <dt><code><b>probe</b></code></dt>
    <dd>Describes the radiation and measurments used is one of:
    'x-ray', 'neutron', 'neutron pol', 'neutron pol spin flip'
    or the respective
    number 0, 1, 2, 3. The calculations for x-rays uses <code>sld_x</code> for the scattering
    length for neutrons <code>sld_n</code> for 'neutron pol', 'neutron pol spin flip' and
    alternatives the <code>sld_m</code>
    is used in the calculations. Note that the angle of magnetization <code>magn_ang</code>
    is only used in the last alternative.</dd>
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
     <code>coords</code>. This assumes a gaussian resolution function and
    <code>res</code> is the standard deviation of that gaussian.
    If <code>restype</code> has (dx/x) in its name the gaussian standard deviation is given by res*x where x is
    either in tth or q.</dd>
    <dt><code><b>restype</b></code></dt>
    <dd>Describes the rype of the resolution calculated. One of the alterantives:
    'no conv', 'fast conv', 'full conv and varying res.', 'fast conv + varying res.',
    'full conv and varying res. (dx/x)', 'fast conv + varying res. (dx/x)'.
    The respective numbers 0-3 also works. Note that fast convolution only alllows
    a single value into res wheras the other can also take an array with the
    same length as the x-data (varying resolution)</dd>
    <dt><code><b>respoints</b></code></dt>
    <dd>The number of points to include in the resolution calculation. This is only
    used for 'full conv and vaying res.', 'fast conv + varying res', 'full conv and varying res. (dx/x)' and
    'fast conv + varying res. (dx/x)'.</dd>
    <dt><code><b>resintrange</b></code></dt>
    <dd>Number of standard deviatons to integrate the resolution function times
    the reflectivity over</dd>
    <dt><code><b>footype</b></code></dt>
    <dd>Which type of footprint correction is to be applied to the simulation.
    One of: 'no corr', 'gauss beam' or 'square beam'. Alternatively, 
    the number 0-2 are also valid. The different choices are self expnalatory.</dd>
    <dt><code><b>beamw</b></code></dt>
    <dd>The width of the beam given in mm. For 'gauss beam' it should be
    the standard deviation. For 'square beam' it is the full width of the beam.</dd>
    <dt><code><b>samplelen</b></code></dt>
    <dd>The length of the sample given in mm</dd>
    <dt><code><b>incangle</b></code></dt>
    <dd>The incident angle of the neutrons, only valid in tof mode</dd>
    <dt><code><b>pol</b></code></dt>
    <dd>The measured polarization of the instrument. Valid options are:
    'uu','dd', 'ud', 'du' or 'ass' the respective number 0-3 also works.</dd>
'''
from numpy import *
try:
    import lib.paratt_weave as Paratt
except StandardError,S:
    print 'Not using inline c code for reflectivity calcs - can not import module'
    print S
    import lib.paratt as Paratt
import lib.neutron_refl as MatrixNeutron
from lib.instrument import *
import lib.refl as refl
# Preamble to define the parameters needed for the models outlined below:

ModelID='SoftNX'
#InstrumentParameters={'Wavelength':1.54, 'Coordinates':1, 'I0':1.0, 'Sim': 0,\
#    'Res':0.001, 'Restype':0, 'Respoints':5, 'Resintrange':2, 'Beaw':0.01,\
#    'Footype':0.0, 'Samlen':10.0, 'Incangle':0.0}
__pars__ = ['Layer', 'Stack', 'Sample', 'Instrument']

instrument_string_choices = {'probe': ['x-ray', 'neutron', 'neutron pol',
    'neutron pol spin flip', 'neutron tof', 'neutron pol tof'], 'coords': ['q', 'tth'],
    'restype': ['no conv', 'fast conv',
     'full conv and varying res.', 'fast conv + varying res.',
     'full conv and varying res. (dx/x)', 'fast conv + varying res. (dx/x)'],
    'footype': ['no corr', 'gauss beam', 'square beam'],
    'pol': ['uu', 'dd', 'ud', 'ass', 'du']}
InstrumentParameters = {'probe':'x-ray', 'wavelength':1.54, 'coords':'tth',
                        'I0':1.0, 'res':0.001,
                        'restype':'no conv', 'respoints':5, 'resintrange':2, 'beamw':0.01,
                        'footype': 'no corr', 'samplelen':10.0, 'incangle':0.0, 'pol': 'uu',
                        'Ibkg': 0.0, 'tthoff':0.0}
InstrumentGroups = [('General', ['wavelength', 'coords', 'I0', 'Ibkg', 'tthoff']),
                    ('Resolution', ['restype', 'res', 'respoints', 'resintrange']),
                    ('Neutron', ['probe', 'pol', 'incangle']),
                    ('Footprint', ['footype', 'beamw', 'samplelen',]),
                    ]
InstrumentUnits = {'probe':'', 'wavelength': 'AA', 'coords':'',
                   'I0': 'arb.', 'res': '[coord]',
                   'restype':'', 'respoints':'pts.', 'resintrange':'[coord]', 'beamw':'mm',\
                   'footype': '', 'samplelen':'mm', 'incangle':'deg.', 'pol': '',\
                   'Ibkg': 'arb.', 'tthoff':'deg.'}
# Coordinates=1 or 'tth' => twothetainput
# Coordinates=0 or 'q'=> Q input
# probe: Type of simulation
#   'x-ray' or 0: X-rays (One output)
#   'neutron'or 1: Neutrons (One output, ignoring magn, magn_ang)
#   'neutron pol' or 2: Neutrons polarized (Two outputs Ruu,Rdd)
#   'neutron pol spin flip' or 3: Neutrons polarized with spin-flip 
#            (Three outputs Ruu,Rdd,Rud=Rdu, ignoring sigma!)
#   'neutron tof' or 4: Neutrons non-polarized TOF, Inc Angle must be set
#   'neutron pol tof'or 5: Neutrons polarized TOF (non-spin flip),
#         Inc Angle must be set
#
# res stddev of resolution
# restype 0 'none': No resolution convlution
#               1 or 'fast': Fast convolution
#               2 or 'full': Full Convolution +varying resolution
#               3 or 'ffull': Fast convolution varying resolution
# respoints Number of points for the convolution only valid for ResolutionType=2
# resintrange Number of standard deviatons to integrate over default 2
# Parameters for footprint coorections
# footype: 0 or 'no corr': No corections for footprint
#            1 or 'gauss beam': Correction for Gaussian beam => Beaw given in mm and stddev
#            2 or 'square beam': Correction for square profile => Beaw given in full width mm
# samlen= Samplelength in mm.

LayerParameters = {'sigma': 0.0, 'd': 0.0, 'sld_x': (1.0+1.0j)*1e-20,
                   'sld_n':  0.0 + 0.0J, 'sld_m': 0.0, 'magn_ang': 0.0}
LayerUnits = {'sigma': 'AA', 'd': 'AA', 'sld_x':'1e-6 1/AA^2',
              'sld_n': '1e-6 1/AA^2', 'sld_m': '1e-6 1/AA^2', 'magn_ang': 'deg.'}
LayerGroups = [('Standard', ['sld_x', 'd', 'sigma']),
               ('Neutron', ['sld_n', 'sld_m', 'magn_ang'])]
StackParameters = {'Layers': [], 'Repetitions': 1}
SampleParameters = {'Stacks': [], 'Ambient': None, 'Substrate': None}

AA_to_eV = 12398.5
''' Conversion from Angstrom to eV E = AA_to_eV/lamda.'''


q_limit = 1e-10
''' Minimum allowed q-value '''

# A buffer to save previous calculations for spin-flip calculations
class Buffer:
    Ruu = 0
    Rdd = 0
    Rdu = 0
    Rud = 0
    parameters = None
    TwoThetaQz = None


def footprintcorr(Q, instrument):
    foocor = 1.0
    footype = instrument.getFootype()
    beamw = instrument.getBeamw()
    samlen = instrument.getSamplelen()
    theta = arcsin(Q * instrument.getWavelength() / 4.0 / pi) * 180 / pi
    if footype == 1 or footype == instrument_string_choices['footype'][1]:
        foocor = GaussIntensity(theta, samlen / 2.0, samlen / 2.0, beamw)
    elif footype == 2 or footype == instrument_string_choices['footype'][2]:
        foocor = SquareIntensity(theta, samlen, beamw)
    elif footype == 0 or footype == instrument_string_choices['footype'][0]:
        pass
    else:
        raise ValueError('The choice of footprint correction, footype,'
                         'is WRONG')

    return foocor


def resolutioncorr(R, TwoThetaQz, foocor, instrument, weight):
    ''' Do the convolution of the reflectivity to account for resolution effects.'''
    restype = instrument.getRestype()
    if restype == instrument_string_choices['restype'][1] or restype == 1:
        R = ConvoluteFast(TwoThetaQz, R[:] * foocor, instrument.getRes(), \
                          range=instrument.getResintrange())
    elif (restype == instrument_string_choices['restype'][2] or restype == 2 or
          restype == instrument_string_choices['restype'][4] or restype == 4):
        R = ConvoluteResolutionVector(TwoThetaQz, R[:] * foocor, weight)
    elif restype == instrument_string_choices['restype'][3] or restype == 3:
        R = ConvoluteFastVar(TwoThetaQz, R[:] * foocor, instrument.getRes(), range=instrument.getResintrange())
    elif restype == instrument_string_choices['restype'][5] or restype == 5:
        R = ConvoluteFastVar(TwoThetaQz, R[:] * foocor, instrument.getRes()*TwoThetaQz,
                             range=instrument.getResintrange())
    elif restype == instrument_string_choices['restype'][0] or restype == 0:
        R = R[:] * foocor
    else:
        raise ValueError('The choice of resolution type, restype,'
                         'is WRONG')
    return R


def resolution_init(TwoThetaQz, instrument):
    ''' Inits the dependet variable with regards to coordinates and resolution.'''
    restype = instrument.getRestype()
    weight = 0
    if restype == 2 or restype == instrument_string_choices['restype'][2]:
        (TwoThetaQz, weight) = ResolutionVector(TwoThetaQz[:],
                                                instrument.getRes(), instrument.getRespoints(),
                                                range=instrument.getResintrange())
    elif restype == 4 or restype == instrument_string_choices['restype'][4]:
        (TwoThetaQz, weight) = ResolutionVector(TwoThetaQz[:],
                                                instrument.getRes()*TwoThetaQz, instrument.getRespoints(),
                                                range=instrument.getResintrange())
    # TTH values given as x
    if instrument.getCoords() == instrument_string_choices['coords'][1] \
            or instrument.getCoords() == 1:
        Q = 4 * pi / instrument.getWavelength() * sin((TwoThetaQz + instrument.getTthoff()) * pi / 360.0)
    # Q vector given....
    elif instrument.getCoords() == instrument_string_choices['coords'][0] \
            or instrument.getCoords() == 0:
        Q = 4 * pi / instrument.getWavelength() * sin(
            arcsin(TwoThetaQz * instrument.getWavelength() / 4 / pi) + instrument.getTthoff() * pi / 360.)
    else:
        raise ValueError('The value for coordinates, coords, is WRONG! should be q(0) or tth(1).')
    return Q, TwoThetaQz, weight


def Specular(TwoThetaQz, sample, instrument):
    """ Simulate the specular signal from sample when probed with instrument

    # BEGIN Parameters
    TwoThetaQz data.x
    # END Parameters
    """

    # preamble to get it working with my class interface
    restype = instrument.getRestype()
    Q, TwoThetaQz, weight = resolution_init(TwoThetaQz, instrument)
    if any(Q < q_limit):
        raise ValueError('The q vector has to be above %.1e'%q_limit)

    type = instrument.getProbe()
    pol = instrument.getPol()

    parameters = sample.resolveLayerParameters()
    if type ==  instrument_string_choices['probe'][0] or type==0:
        #fb = array(parameters['f'], dtype = complex64)
        e = AA_to_eV/instrument.getWavelength()
        sld = refl.cast_to_array(parameters['sld_x'], e)*1e-6
    else: 
        sld = array(parameters['sld_n'], dtype = complex128)*1e-6

    d = array(parameters['d'], dtype = float64)
    sld_m = array(parameters['sld_m'], dtype = float64)*1e-6
    #Transform to radians
    magn_ang = array(parameters['magn_ang'], dtype = float64)*pi/180.0 
    
    sigma = array(parameters['sigma'], dtype=float64)

    wl = instrument.getWavelength()
    l2pi = wl**2/2/3.141592
    # Ordinary Paratt X-rays
    if type == instrument_string_choices['probe'][0] or type == 0:
        R = Paratt.ReflQ(Q,instrument.getWavelength(), 1.0 - l2pi*sld, d, sigma)
    #Ordinary Paratt Neutrons
    elif type == instrument_string_choices['probe'][1] or type == 1:
        R = Paratt.ReflQ(Q,instrument.getWavelength(), 1.0 - l2pi*sld, d, sigma)
    #Ordinary Paratt but with magnetization
    elif type == instrument_string_choices['probe'][2] or type == 2:
        # Polarization uu or ++
        if pol == instrument_string_choices['pol'][0] or pol == 0:
            R = Paratt.ReflQ(Q,instrument.getWavelength(),\
                1.0 - l2pi*(sld + sld_m), d, sigma)
        # Polarization dd or --
        elif pol == instrument_string_choices['pol'][1] or pol == 1:
            R = Paratt.ReflQ(Q,instrument.getWavelength(),\
                 1.0 - l2pi*(sld - sld_m), d, sigma)
        elif pol == instrument_string_choices['pol'][3] or pol == 3:
            Rp = Paratt.ReflQ(Q, instrument.getWavelength(), 1.0 - l2pi*(sld - sld_m), d, sigma)
            Rm = Paratt.ReflQ(Q, instrument.getWavelength(), 1.0 - l2pi*(sld + sld_m), d, sigma)
            R = (Rp - Rm)/(Rp + Rm)

        else:
            raise ValueError('The value of the polarization is WRONG.'
                ' It should be uu(0) or dd(1)')
    # Spin flip
    elif type == instrument_string_choices['probe'][3] or type == 3:
        # Check if we have calcluated the same sample previous:
        if Buffer.TwoThetaQz is not None:
            Q_ok = Buffer.TwoThetaQz.shape == Q.shape
            if Q_ok:
                Q_ok = any(not_equal(Buffer.TwoThetaQz, Q))
        if Buffer.parameters != parameters or not Q_ok:
            #msld = 2.645e-5*magn*dens*instrument.getWavelength()**2/2/pi
            np = 1.0 - l2pi*(sld + sld_m)
            nm = 1.0 - l2pi*(sld - sld_m)
            Vp = (2*pi/instrument.getWavelength())**2*(1 - np**2)
            Vm = (2*pi/instrument.getWavelength())**2*(1 - nm**2)
            (Ruu, Rdd, Rud, Rdu) = MatrixNeutron.Refl(Q, Vp, Vm, d, magn_ang, sigma)
            Buffer.Ruu = Ruu; Buffer.Rdd = Rdd; Buffer.Rud = Rud
            Buffer.parameters = parameters.copy()
            Buffer.TwoThetaQz = Q.copy()
        else:
            pass
        # Polarization uu or ++
        if pol == instrument_string_choices['pol'][0] or pol == 0:
            R = Buffer.Ruu
        # Polarization dd or --
        elif pol == instrument_string_choices['pol'][1] or pol == 1:
            R = Buffer.Rdd
        # Polarization ud or +-
        elif (pol == instrument_string_choices['pol'][2] or pol == 2 or
              pol == instrument_string_choices['pol'][4] or pol == 4):
            R = Buffer.Rud
        # Calculating the asymmetry ass
        elif pol == instrument_string_choices['pol'][3] or pol == 3:
            R = (Buffer.Ruu - Buffer.Rdd)/(Buffer.Ruu + Buffer.Rdd + 2*Buffer.Rud)
        else:
            raise ValueError('The value of the polarization is WRONG. It should be uu(0), dd(1) or ud(2)')
    else:
        raise ValueError('The choice of probe is WRONG')
    #FootprintCorrections

    foocor = footprintcorr(Q, instrument)
    #Resolution corrections
    R = resolutioncorr(R, TwoThetaQz, foocor, instrument, weight)
    
    return R*instrument.getI0() + instrument.getIbkg()
    
def EnergySpecular(Energy, TwoThetaQz,sample,instrument):
    ''' Simulate the specular signal from sample when probed with instrument. Energy should be in eV.

    # BEGIN Parameters
    Energy data.x
    TwoThetaQz 3.0
    # END Parameters
    '''
    # preamble to get it working with my class interface
    restype = instrument.getRestype()
    #TODO: Fix so that resolution can be included.
    if restype != 0 and restype != instrument_string_choices['restype'][0]:
        raise ValueError('Only no resolution is allowed for energy scans.')

    wl = AA_to_eV/Energy

    # TTH values given as x
    if instrument.getCoords() == instrument_string_choices['coords'][1] \
            or instrument.getCoords() == 1:
        theta = TwoThetaQz/2.0
    # Q vector given....
    elif instrument.getCoords() == instrument_string_choices['coords'][0] \
            or instrument.getCoords() == 0:
        theta = arcsin(TwoThetaQz * wl / 4 / pi)*180.0/pi

    else:
        raise ValueError('The value for coordinates, coords, is WRONG!'
                         'should be q(0) or tth(1).')
    Q = 4 * pi / wl * sin((2*theta + instrument.getTthoff()) * pi / 360.0)

    type = instrument.getProbe()

    parameters = sample.resolveLayerParameters()
    if type ==  instrument_string_choices['probe'][0] or type==0:
        sld = refl.cast_to_array(parameters['sld_x'], Energy)*1e-6
    else:
        sld = array(parameters['sld_n'], dtype = complex64).real*1e-6

    d = array(parameters['d'], dtype = float64)
    sigma = array(parameters['sigma'], dtype = float64)

    wl = instrument.getWavelength()
    l2pi = wl**2/2/3.141592
    # Ordinary Paratt X-rays
    if type == instrument_string_choices['probe'][0] or type == 0:
        #R = Paratt.ReflQ(Q,instrument.getWavelength(),1.0-2.82e-5*sld,d,sigma)
        R = Paratt.Refl_nvary2(theta, wl, 1.0 - l2pi*sld, d, sigma)
    else:
        raise ValueError('The choice of probe is WRONG')
    #TODO: Fix corrections
    #FootprintCorrections
    #foocor = footprintcorr(Q, instrument)
    #Resolution corrections
    #R = resolutioncorr(R, TwoThetaQz, foocor, instrument, weight)

    return R*instrument.getI0() + instrument.getIbkg()


def OffSpecular(TwoThetaQz,ThetaQx,sample,instrument):
    ''' Function that simulates the off-specular signal (not implemented)
    
    # BEGIN Parameters
    TwoThetaQz 1.0
    ThetaQx data.x
    # END Parameters
    '''
    raise NotImplementedError('Not implemented use model interdiff insteads')
    return TwoThetaQz,ThetaQx

def SLD_calculations(z, item, sample, inst):
    ''' Calculates the scatteringlength density as at the positions z
    if item is None or "all" the function returns a dictonary of values.
    Otherwise it returns the item as identified by its string.
    
    # BEGIN Parameters
    z data.x
    item 'Re'
    # END Parameters
    '''
    parameters = sample.resolveLayerParameters()
    #f = array(parameters['f'], dtype = complex64)
    e = AA_to_eV/inst.getWavelength()
    sld_x = refl.cast_to_array(parameters['sld_x'], e)
    sld_n = array(parameters['sld_n'], dtype = complex64)
    type = inst.getProbe()
    magnetic = False
    mag_sld = 0
    sld_unit = '10^{-6}\AA^{2}'
    if type == instrument_string_choices['probe'][0] or type == 0:
        sld = sld_x
    elif type == instrument_string_choices['probe'][1] or type == 1 or\
        type == instrument_string_choices['probe'][4] or type == 4:
        sld = sld_n
        sld_unit = '10^{-6}/\AA^{2}'
    else:
        magnetic = True
        sld = sld_n
        sld_m = array(parameters['sld_m'], dtype = float64)
        #Transform to radians
        magn_ang = array(parameters['magn_ang'], dtype = float64)*pi/180.0
        mag_sld = sld_m
        mag_sld_x = mag_sld*cos(magn_ang)
        mag_sld_y = mag_sld*sin(magn_ang)
        sld_unit = '10^{-6}/\AA^{2}'
        
    d = array(parameters['d'], dtype = float64)
    d = d[1:-1]
    # Include one extra element - the zero pos (substrate/film interface)
    int_pos = cumsum(r_[0,d])
    sigma = array(parameters['sigma'], dtype = float64)[:-1] + 1e-7
    if z == None:
        z = arange(-sigma[0]*5, int_pos.max()+sigma[-1]*5, 0.5)
    if not magnetic:
        rho = sum((sld[:-1] - sld[1:])*(0.5 -\
            0.5*erf((z[:,newaxis]-int_pos)/sqrt(2.)/sigma)), 1) + sld[-1]
        dic = {'Re': real(rho), 'Im': imag(rho), 'z':z, 
               'SLD unit': sld_unit}
    else:
        sld_p = sld + mag_sld
        sld_m = sld - mag_sld
        rho_p = sum((sld_p[:-1] - sld_p[1:])*(0.5 -\
            0.5*erf((z[:,newaxis]-int_pos)/sqrt(2.)/sigma)), 1) + sld_p[-1]
        rho_m = sum((sld_m[:-1] - sld_m[1:])*(0.5 -\
            0.5*erf((z[:,newaxis]-int_pos)/sqrt(2.)/sigma)), 1)  + sld_m[-1]
        rho_mag_x = sum((mag_sld_x[:-1] - mag_sld_x[1:])*
                        (0.5 - 0.5*erf((z[:,newaxis]-int_pos)/sqrt(2.)/sigma)), 1) + mag_sld_x[-1]
        rho_mag_y = sum((mag_sld_y[:-1] - mag_sld_y[1:])*
                        (0.5 - 0.5*erf((z[:,newaxis]-int_pos)/sqrt(2.)/sigma)), 1) + mag_sld_y[-1]
        #dic = {'Re sld +': real(rho_p), 'Im sld +': imag(rho_p),\
        #        'Re sld -': real(rho_m), 'Im sld -': imag(rho_m), 'z':z,
        #        'SLD unit': sld_unit}
        rho_nucl = (rho_p + rho_m)/2.
        dic = {'Re non-mag': real(rho_nucl), 'Im non-mag': imag(rho_nucl),
                'mag': real(rho_p - rho_m)/2, 'mag_x': rho_mag_x, 'mag_y': rho_mag_y,
                'z':z, 'SLD unit': sld_unit}
    if item == None or item == 'all':
        return dic
    else:
        try:
            return dic[item]
        except:
            raise ValueError('The chosen item, %s, does not exist'%item)

SimulationFunctions={'Specular':Specular,
                     'OffSpecular':OffSpecular,
                     'SLD': SLD_calculations,
                     'EnergySpecular': EnergySpecular,
                    }


(Instrument, Layer, Stack, Sample) = refl.MakeClasses(InstrumentParameters,\
    LayerParameters, StackParameters, SampleParameters, SimulationFunctions,\
    ModelID)


if __name__=='__main__':
    pass
