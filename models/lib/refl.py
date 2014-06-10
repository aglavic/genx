import shelve, copy
import numpy as np


class ReflBase:
    # _parameters is a dict of parameter names with their defualt values
    _parameters = {}

    def __init__(self, **kargs):
        # Setup all the parameters in the class
        for par in self._parameters:
            setattr(self, par, self._parameters[par])
            iscomplex = type(self._parameters[par]) is complex
            # Making the set function
            self._make_set_func(par, iscomplex)
            # Creating the get function
            self._make_get_func(par, iscomplex)

        # Set all parameters given as keyword arguments
        for k in kargs:
            if not k in self._parameters and not k in dir(self):
                raise ValueError('%s is not an parameter in %s' %
                                 (k, self.__class__))
            else:
                setattr(self, k, kargs[k])

    def _make_get_func(self, par, iscomplex = False):
        ''' Creates a get function for parameter par and binds it to the object
        '''

        def get_func():
            return getattr(self, par)

        get_func.__name__ = 'get' + par.capitalize()
        setattr(self, get_func.__name__, get_func)

        if iscomplex:
            def get_real_func():
                return getattr(self, par).real

            get_real_func.__name__ = get_func.__name__ + 'real'
            setattr(self, get_real_func.__name__, get_real_func)

            def get_imag_func():
                return getattr(self, par).imag

            get_imag_func.__name__ = get_func.__name__ + 'imag'
            setattr(self, get_imag_func.__name__, get_imag_func)

    def _make_set_func(self, par, iscomplex = False):
        ''' Creates a set function for parameter par and binds it to the object
            '''

        def set_func(val):
            setattr(self, par, val)

        set_func.__name__ = 'set' + par.capitalize()
        setattr(self, set_func.__name__, set_func)

        if iscomplex:
            def set_real_func(val):
                setattr(self, par, val + getattr(self, par).imag*1J)

            set_real_func.__name__ = set_func.__name__ + 'real'
            setattr(self, set_real_func.__name__, set_real_func)

            def set_imag_func(val):
                setattr(self, par, val*1J + getattr(self, par).real)

            set_imag_func.__name__ = set_func.__name__ + 'imag'
            setattr(self, set_imag_func.__name__, set_imag_func)


    def get_parameters(self):
        ''' Returns all the parameters of the current object '''

        return self.parameters


class InstrumentBase(ReflBase):
    _parameters = {}

    def __repr__(self):
        s = 'Instrument('

        for k in self._parameters:
            # if the type is a string...
            if type(getattr(self, k)) == type(''):
                stemp = "%s = '%s'," % (k, str(getattr(self, k)))
            else:
                stemp = '%s = %s,' % (k, str(getattr(self, k)))
            s = s + stemp
        return s[:-1] + ')'


class LayerBase(ReflBase):
    _parameters = {}

    def __repr__(self):
        s = 'Layer('
        for k in self._parameters:
            s += '%s = %s, ' % (k, str(getattr(self, k)))
        return s[:-2] + ')'


class StackBase(ReflBase):
    _parameters = {'Repetitions': 1}
    Layers = []
    Repetitions = 1

    def __init__(self, **kargs):
        ReflBase.__init__(self, **kargs)
        if 'Layers' in kargs:
            self.Layers = kargs['Layers']
            kargs.pop('Layers')
        else:
            self.Layers = []


    def __repr__(self):
        s = 'Stack: '
        for k in self._parameters:
            s += '%s = %s, ' % (k, str(getattr(self, k)))
        s = s[:-2] + '\n'
        it = len(self.Layers)
        for lay in range(it - 1, -1, -1):
            s += '\t' + repr(self.Layers[lay]) + '\n'
        return s

    def resolveLayerParameter(self, parameter):
        par = [getattr(lay, parameter)
               for lay in self.Layers] * self.Repetitions
        return par


# Class Sample

class SampleBase(ReflBase):
    _parameters = {}
    Ambient = None
    Substrate = None
    Stacks = []

    def __init__(self, **kargs):
        ReflBase.__init__(self, **kargs)
        special_objects = ['Ambient', 'Substrate', 'Stacks']
        for p in special_objects:
            if p in kargs:
                setattr(self, p, kargs[p])
            else:
                raise ValueError('%s has to be defined to create a Sample object' % p)
        # Create a new dictonary for the new arguments
        nkargs = {}
        for p in kargs:
            if not p in special_objects:
                nkargs[p] = kargs[p]


    def __repr__(self):
        Add = 'Sample: '
        for k in self._parameters:
            Add += '%s = %s, ' % (k, getattr(self, k).__repr__())
        if len(self._parameters) > 0:
            Add = Add[:-2] + '\n'
        else:
            Add = Add[:-1] + '\n'
        temp = [repr(x) for x in self.Stacks]
        temp.reverse()
        return (Add + 'Ambient:\n\t' + repr(self.Ambient) + '\n' + ''.join(temp)
                + 'Substrate:\n\t' + repr(self.Substrate))

    def resolveLayerParameters(self):
        par = self.Substrate._parameters.copy()
        for k in par:
            par[k] = [getattr(self.Substrate, k)]
        for k in self.Substrate._parameters:
            for stack in self.Stacks:
                par[k] = par[k] + stack.resolveLayerParameter(k)
            par[k] = par[k] + [getattr(self.Ambient, k)]
        return par


# Factory needs to:
#            add members and their names in the _parameter list
#            Add set and get methods for each parameter
class ModelFactory:
    ''' Class that creates new classes to implement certain parameters 
        dynamically.
    '''

    class Layer(LayerBase): _parameters = LayerBase._parameters.copy()

    class Stack(StackBase): _parameters = StackBase._parameters.copy()

    class Sample(SampleBase): _parameters = SampleBase._parameters.copy()

    class Instrument(InstrumentBase): _parameters = InstrumentBase._parameters.copy()

    def __init__(self):
        pass

    @staticmethod
    def _add_parameter(cls, par_name, def_val):
        cls._parameters.update({par_name: def_val})

    @staticmethod
    def _add_sim_method(cls, name, func):
        def method(self, *args):
            nargs = args[:-1] + (self, ) + (args[-1], )
            return func(*nargs)

        method.__name__ = 'Sim' + name
        setattr(cls, method.__name__, method)

    def set_layer_parameters(self, parameters):
        '''Adds extra parameters (dict with name: value entries) to the
            Layer class'''
        [self._add_parameter(self.Layer, k, parameters[k]) for k in parameters]

    def set_stack_parameters(self, parameters):
        '''Adds extra parameters (dict with name: value entries) to the
            Stack class'''
        # The if not is a hook to so that everything works with the old implementation
        [self._add_parameter(self.Stack, k, parameters[k]) for k in parameters
         if not k in ['Layers', 'Repetitions']]

    def set_sample_parameters(self, parameters):
        '''Adds extra parameters (dict with name: value entries) to the
            Sample class'''
        # The if not is a hook to so that everything works with the old implementation
        [self._add_parameter(self.Sample, k, parameters[k]) for k in parameters
            if not k in ['Ambient', 'Substrate', 'Stacks']]

    def set_simulation_functions(self, functions):
        ''' Adds the simulation functions to the sample class
        '''
        [self._add_sim_method(self.Sample, k, functions[k]) for k in functions]

    def set_instrument_parameters(self, parameters):
        '''Adds extra parameters (dict with name: value entries) to the
            Instrument class'''
        [self._add_parameter(self.Instrument, k, parameters[k]) for k in parameters]

    def get_layer_class(self):
        return self.Layer

    def get_stack_class(self):
        return self.Stack

    def get_sample_class(self):
        return self.Sample

    def get_instrument_class(self):
        return self.Instrument


# Function! to create classes with the model given by the input parameters
def MakeClasses(InstrumentParameters={'Wavelength': 1.54, 'Coordinates': 1}, \
                LayerParameters={'d': 0.0, 'sigma': 0.0, 'n': 0.0 + 0.0j}, \
                StackParameters={}, \
                SampleParameters={}, \
                SimulationFunctions={'Specular': lambda x: x, 'OffSpecular': lambda x: x}, \
                ModelID='Standard'):
    factory = ModelFactory()
    factory.set_layer_parameters(LayerParameters)
    factory.set_stack_parameters(StackParameters)
    factory.set_sample_parameters(SampleParameters)
    factory.set_simulation_functions(SimulationFunctions)
    factory.set_instrument_parameters(InstrumentParameters)
    Layer = factory.get_layer_class()
    Stack = factory.get_stack_class()
    Sample = factory.get_sample_class()
    Instrument = factory.get_instrument_class()
    return (Instrument, Layer, Stack, Sample)


if __name__ == '__main__':
    InstrumentParameters = {'probe': 'x-ray', 'wavelength': 1.54, 'coords': 'tth',
                            'I0': 1.0, 'res': 0.001,
                            'restype': 'no conv', 'respoints': 5, 'resintrange': 2, 'beamw': 0.01,
                            'footype': 'no corr', 'samplelen': 10.0, 'incangle': 0.0, 'pol': 'uu',
                            'Ibkg': 0.0, 'tthoff': 0.0}

    LayerParameters = {'sigma': 0.0, 'dens': 1.0, 'd': 0.0, 'f': (1.0 + 1.0j) * 1e-20,
                       'b': 0.0 + 0.0J, 'xs_ai': 0.0, 'magn': 0.0, 'magn_ang': 0.0}
    StackParameters = {'Layers': [], 'Repetitions': 1}
    SampleParameters = {'Stacks': [], 'Ambient': None, 'Substrate': None}

    (Instrument, Layer, Stack, Sample) = MakeClasses(InstrumentParameters, \
                                                          LayerParameters, StackParameters, SampleParameters,
                                                            {'Specualr':lambda x:x}, \
                                                          'test')
    inst = Instrument(footype = 'gauss beam',probe = 'x-ray',beamw = 0.04,resintrange = 2,pol = 'uu',wavelength = 1.54,respoints = 5,Ibkg = 0.0,I0 = 2,samplelen = 10.0,restype = 'fast conv',coords = 'tth',res = 0.001,incangle = 0.0)

    # BEGIN Sample DO NOT CHANGE
    Amb = Layer(b = 0, d = 0.0, f = (1e-20+1e-20j), dens = 1.0, magn_ang = 0.0, sigma = 0.0, xs_ai = 0.0, magn = 0.0)
    topPt = Layer(b = 0, d = 11.0, f = 58, dens = 4/3.92**3, magn_ang = 0.0, sigma = 3.0, xs_ai = 0.0, magn = 0.0)
    TopFe = Layer(b = 0, d = 11.0, f = 26, dens = 2/2.866**3, magn_ang = 0.0, sigma = 2.0, xs_ai = 0.0, magn = 0.0)
    Pt = Layer(b = 0, d = 11.0, f = 58, dens = 4/3.92**3, magn_ang = 0.0, sigma = 2.0, xs_ai = 0.0, magn = 0.0)
    Fe = Layer(b = 0, d = 11, f = 26, dens = 2/2.866**3, magn_ang = 0.0, sigma = 2.0, xs_ai = 0.0, magn = 0.0)
    bufPt = Layer(b = 0, d = 45, f = 58, dens = 4/3.92**3, magn_ang = 0.0, sigma = 2, xs_ai = 0.0, magn = 0.0)
    bufFe = Layer(b = 0, d = 2, f = 26, dens = 2/2.866**3, magn_ang = 0.0, sigma = 2, xs_ai = 0.0, magn = 0.0)
    Sub = Layer(b = 0, d = 0.0, f = 12 + 16, dens = 2/4.2**3, magn_ang = 0.0, sigma = 4.0, xs_ai = 0.0, magn = 0.0)

    top = Stack(Layers=[TopFe , topPt], Repetitions = 1)
    ML = Stack(Layers=[Fe , Pt], Repetitions = 19)
    buffer = Stack(Layers=[bufFe , bufPt], Repetitions = 1)

    sample = Sample(Stacks = [buffer ,ML ,top], Ambient = Amb, Substrate = Sub)