''' Library for combined x-ray and neutrons simulations.

The neutron simulations is capable of handling non-magnetic, 
magnetic non-spin flip as well as neutron spin-flip reflectivity. 
'''
from numpy import *

import lib.paratt as Paratt
import lib.neutron_refl as MatrixNeutron
from lib.instrument import *
# Preamble to define the parameters needed for the models outlined below:

ModelID='SpecNX'
#InstrumentParameters={'Wavelength':1.54, 'Coordinates':1, 'I0':1.0, 'Sim': 0,\
#    'Res':0.001, 'Restype':0, 'Respoints':5, 'Resintrange':2, 'Beaw':0.01,\
#    'Footype':0.0, 'Samlen':10.0, 'Incangle':0.0}
__pars__ = ['Layer', 'Stack', 'Sample', 'Instrument']
instrument_string_choices = {'probe': ['x-ray', 'neutron', 'neutron pol',\
    'neutron pol spin flip', 'neutron tof', 'n pol tof'], 'coords': ['q','tth'],\
    'restype': ['no conv', 'fast conv',\
     'full conv and varying res.', 'fast conv + varying res.'],\
    'footype': ['no corr', 'gauss beam', 'square beam'],\
    'pol': ['uu','dd','ud']}
    
InstrumentParameters={'probe':'x-ray', 'wavelength':1.54, 'coords':'tth',\
     'I0':1.0, 'res':0.001,\
    'restype':'no conv', 'respoints':5, 'resintrange':2, 'beamw':0.01,\
     'footype': 'no corr', 'samplelen':10.0, 'incangle':0.0, 'pol': 'uu'}
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

LayerParameters={'sigma':0.0, 'dens':1.0, 'd':0.0, 'f':0.0+0.0j,\
     'b':0.0+0.0j, 'magn':0.0, 'magn_ang':0.0}
StackParameters={'Layers':[], 'Repetitions':1}
SampleParameters={'Stacks':[], 'Ambient':None, 'Substrate':None}

# A buffer to save previous calculations for spin-flip calculations
class Buffer:
    Ruu = 0
    Rdd = 0
    Rdu = 0
    Rud = 0
    parameters = None

def Specular(TwoThetaQz,sample,instrument):
    # preamble to get it working with my class interface
    restype = instrument.getRestype()

    if restype == 2 or restype == instrument_string_choices['restype'][2]:
        (TwoThetaQz,weight) = ResolutionVector(TwoThetaQz[:], \
              instrument.getRes(), instrument.getRespoints(),\
               range = instrument.getResintrange())
    # TTH values given as x
    if instrument.getCoords() == instrument_string_choices['coords'][1]\
     or instrument.getCoords() == 1:
        Q = 4*pi/instrument.getWavelength()*sin(TwoThetaQz*pi/360.0)
    # Q vector given....
    elif instrument.getCoords() == instrument_string_choices['coords'][0]\
     or instrument.getCoords() == 0:
        Q = TwoThetaQz
    else:
        raise ValueError('The value for coordinates, coords, is WRONG!'
                        'should be q(0) or tth(1).')
            
    type = instrument.getProbe()
    pol = instrument.getPol()
    
    lamda = instrument.getWavelength()
    parameters = sample.resolveLayerParameters()
    if type ==  instrument_string_choices['probe'][0] or type==0:
        fb = array(parameters['f'], dtype = complex64)
    else: 
        fb = array(parameters['b'], dtype = complex64)
    
    dens = array(parameters['dens'], dtype = complex64)
    d = array(parameters['d'], dtype = float64)
    magn = array(parameters['magn'], dtype = float64)
    #Transform to radians
    magn_ang = array(parameters['magn_ang'], dtype = float64)*pi/180.0 
    
    sigma = array(parameters['sigma'], dtype = float64)
    sld = dens*fb*instrument.getWavelength()**2/2/pi

    # Ordinary Paratt X-rays
    if type == instrument_string_choices['probe'][0] or type == 0:
        R = Paratt.ReflQ(Q,instrument.getWavelength(),1.0-2.82e-5*sld,d,sigma)\
                *instrument.getI0()
    #Ordinary Paratt Neutrons
    elif type == instrument_string_choices['probe'][1] or type == 1:
        R = Paratt.ReflQ(Q,instrument.getWavelength(),1.0-sld,d,sigma)\
            *instrument.getI0()
    #Ordinary Paratt but with magnetization
    elif type == instrument_string_choices['probe'][2] or type == 2:
        msld = 2.645e-5*magn*dens*instrument.getWavelength()**2/2/pi
        # Polarization uu or ++
        if pol == instrument_string_choices['pol'][0] or pol == 0:
            R = Paratt.ReflQ(Q,instrument.getWavelength(),\
                1.0-sld-msld,d,sigma)*instrument.getI0()
        # Polarization dd or --
        elif pol == instrument_string_choices['pol'][1] or pol == 1:
            R = Paratt.ReflQ(Q,instrument.getWavelength(),\
                 1.0-sld+msld,d,sigma)*instrument.getI0()
        else:
            raise ValueError('The value of the polarization is WRONG.'
                ' It should be uu(0) or dd(1)')
    # Spin flip
    elif type == instrument_string_choices['probe'][3] or type == 3:
        # Check if we have calcluated the same sample previous:
        if Buffer.parameters != parameters:
            msld = 2.645e-5*magn*dens*instrument.getWavelength()**2/2/pi
            np = 1.0-sld-msld
            nm = 1.0-sld+msld
            Vp = (2*pi/instrument.getWavelength())**2*(1-np**2)
            Vm = (2*pi/instrument.getWavelength())**2*(1-nm**2)
            (Ruu,Rdd,Rud,Rdu) = MatrixNeutron.Refl(Q,Vp,Vm,d,magn_ang)
            Buffer.Ruu = Ruu; Buffer.Rdd = Rdd; Buffer.Rud = Rud
            Buffer.parameters = parameters.copy()
        else:
            pass
        # Polarization uu or ++
        if pol == instrument_string_choices['pol'][0] or pol == 0:
            R = Buffer.Ruu
        # Polarization dd or --
        elif pol == instrument_string_choices['pol'][1] or pol == 1:
            R = Buffer.Rdd
        # Polarization ud or +-
        elif pol == instrument_string_choices['pol'][2] or pol == 2:
            R = Buffer.Rud
        else:
            raise ValueError('The value of the polarization is WRONG.'
                ' It should be uu(0), dd(1) or ud(2)')
        R = R*instrument.getI0()
        
    # tof
    elif type == instrument_string_choices['probe'][4] or type == 4:
        sld = dens[:,newaxis]*fb[:,newaxis]*\
                (4*pi*sin(instrument.getIncangle()*pi/180)/Q)**2/2/pi
        R = Paratt.Refl_nvary2(instrument.getIncangle()*ones(Q.shape),\
            (4*pi*sin(instrument.getIncangle()*pi/180)/Q),\
                1.0-sld,d,sigma)*instrument.getI0()
    # tof spin polarized
    elif type == instrument_string_choices['probe'][5] or type == 5:
        sld = dens[:,newaxis]*fb[:,newaxis]*\
            (4*pi*sin(instrument.getIncangle()*pi/180)/Q)**2/2/pi
        msld = 2.645e-5*magn[:,newaxis]*dens[:,newaxis]\
                *(4*pi*sin(instrument.getIncangle()*pi/180)/Q)**2/2/pi
        # polarization uu or ++
        if pol == instrument_string_choices['pol'][0] or pol == 0:
            R = Paratt.Refl_nvary2(instrument.getIncangle()*ones(Q.shape),\
                (4*pi*sin(instrument.getIncangle()*pi/180)/Q),\
                 1.0-sld-msld,d,sigma)*instrument.getI0()
        # polarization dd or --
        elif pol == instrument_string_choices['pol'][1] or pol == 1:
            R = Paratt.Refl_nvary2(instrument.getIncangle()*ones(Q.shape),\
             (4*pi*sin(instrument.getIncangle()*pi/180)/Q),\
              1.0-sld+msld,d,sigma)*instrument.getI0()
        else:
            raise ValueError('The value of the polarization is WRONG.'
                ' It should be uu(0) or dd(1)')
    else:
        raise ValueError('The choice of probe is WRONG')
    #FootprintCorrections
    
    foocor = 1.0
    footype = instrument.getFootype()
    beamw = instrument.getBeamw()
    samlen = instrument.getSamplelen()
    theta = arcsin(Q*instrument.getWavelength()/4.0/pi)*180/pi
    if footype == 1 or instrument_string_choices['footype'][1]:
        foocor = GaussIntensity(theta, samlen/2.0, samlen/2.0, beamw)
    elif footype == 2 or instrument_string_choices['footype'][2]:
        foocor=SquareIntensity(theta, samlen, beamw)
    elif footype == 0 or instrument_string_choices['footype'][0]:
        pass
    else:
        raise ValueError('The choice of footprint correction, footype,'
            'is WRONG')
        
    #Resolution corrections
    if restype == instrument_string_choices['restype'][1] or restype == 1:
        R = ConvoluteFast(TwoThetaQz,R[:]*foocor,instrument.getRes(),\
             range=instrument.getResintrange())
    elif restype == instrument_string_choices['restype'][2] or restype == 2:
        R = ConvoluteResolutionVector(TwoThetaQz,R[:]*foocor,weight)
    elif restype == instrument_string_choices['restype'][3] or restype == 3:
        R = ConvoluteFastVar(TwoThetaQz,R[:]*foocor,instrument.getRes(),\
            range = instrument.getResintrange())
    elif restype == instrument_string_choices['restype'][0] or restype == 0:
        pass
    else:
        raise ValueError('The choice of resolution type, restype,'
            'is WRONG')
    
    return R
    

def OffSpecularMingInterdiff(TwoThetaQz,ThetaQx,sample,instrument):
    raise NotImplementedError('Not implemented use model interdiff insteads')
    return TwoThetaQz,ThetaQx

SimulationFunctions={'Specular':Specular, 'OffSpecular':OffSpecularMingInterdiff}

import lib.refl as Refl
(Instrument, Layer, Stack, Sample) = Refl.MakeClasses(InstrumentParameters,\
    LayerParameters, StackParameters, SampleParameters, SimulationFunctions,\
    ModelID)


if __name__=='__main__':
    pass
