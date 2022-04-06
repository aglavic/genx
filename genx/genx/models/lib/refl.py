''' Module to provide the Layer - Stack - Sample classes to build a sample for reflectivity modelling.

Classes:
ReflFunction - A function class that can be used as a parameter in the other classes.
is_reflfunction - Funtion that checks if an object belongs to the class ReflFunction
ReflBase - Base class for all the physical classes.
LayerBase - Base Layer class.
StackBase - Base Stack class.
SampleBase - Base Sample class.
InstrumentBase - Base Instrument class.
ModelFactory - A factory class that creates specific instances of the Base classes for a certain set of parameters.
MakeClasses - Function that creates the classes (backward comparability).
'''

import numpy as np
from genx.core.custom_logging import iprint

class ReflFunction:
    def __init__(self, function, validation_args, validation_kwargs, id=None):
        """ Creates the Refl Function given function. The arguments validation_args and
        validation_kwargs will be used to the validate the returned type of the function by passing
        them to function. The variable id should be a unique string to identify the type ReflFunction.
        """
        self.__func__=function
        self.validation_args=validation_args
        self.validation_kwargs=validation_kwargs
        self.id=id

    def __call__(self, *args, **kwargs):
        return self.__func__(*args, **kwargs)

    def validate(self):
        ''' Function to test that the function returns the anticipated type
        '''
        return self.__call__(*self.validation_args, **self.validation_kwargs)

    def _check_obj(self, other):
        """ Checks the object other so that it fulfills the demands for arithmetic operations.
        """
        supported_types=[int, float, int, complex, np.float64, np.float32]
        if is_reflfunction(other):
            if self.id!=other.id:
                raise TypeError("Two ReflFunction objects must have identical id's to conduct arithmetic operations")
        elif not type(other) in supported_types:
            raise TypeError("%s is not supported for arithmetic operations "%(repr(type(other)))+
                            "of a ReflFunction. It has to int, float, long or complex")

    def __mul__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):
            def new_func(*args, **kwargs):
                return self(*args, **kwargs)*other(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return self(*args, **kwargs)*other
        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __rmul__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):
            def new_func(*args, **kwargs):
                return other(*args, **kwargs)*self(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return other*self(*args, **kwargs)
        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __add__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):
            def new_func(*args, **kwargs):
                return self(*args, **kwargs)+other(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return self(*args, **kwargs)+other
        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __radd__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):
            def new_func(*args, **kwargs):
                return other(*args, **kwargs)+self(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return other+self(*args, **kwargs)
        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __sub__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):
            def new_func(*args, **kwargs):
                return self(*args, **kwargs)-other(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return self(*args, **kwargs)-other
        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __rsub__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):
            def new_func(*args, **kwargs):
                return other(*args, **kwargs)-self(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return other-self(*args, **kwargs)
        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __div__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):
            def new_func(*args, **kwargs):
                return self(*args, **kwargs)/other(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return self(*args, **kwargs)/other
        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __rdiv__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):
            def new_func(*args, **kwargs):
                return other(*args, **kwargs)/self(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return other/self(*args, **kwargs)
        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __neg__(self):
        def new_func(*args, **kwargs):
            return -self(*args, **kwargs)

        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __pos__(self):
        def new_func(*args, **kwargs):
            return self(*args, **kwargs)

        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __pow__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):
            def new_func(*args, **kwargs):
                return self(*args, **kwargs)**other(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return self(*args, **kwargs)**other
        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __rpow__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):
            def new_func(*args, **kwargs):
                return other(*args, **kwargs)**self(*args, **kwargs)
        else:
            def new_func(*args, **kwargs):
                return other**self(*args, **kwargs)
        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

def cast_to_array(list_of_obj, *args, **kwargs):
    ''' Casts an list_of_obj, can be a number or an ReflFunction, into an list of evaluated values'''
    id=''
    shape=False
    ret_list=[]
    for obj in list_of_obj:
        if is_reflfunction(obj):
            if id=='':
                id=obj.id
                ret_list.append(obj(*args, **kwargs))
                # Check if we have got an array
                if not np.isscalar(ret_list[-1]):
                    shape=ret_list[-1].shape
            elif id==obj.id:
                ret_list.append(obj(*args, **kwargs))
            else:
                TypeError("Two ReflFunction objects must have identical id's in order to merge them into an array")
        else:
            # We assume that this is an object that can be transformed into an array later on.
            ret_list.append(obj)
    # if we have an array make sure that all the objects have the same shape
    if shape:
        base_array=np.ones(shape)
        nret_list=[]
        for item in ret_list:
            if np.isscalar(item):
                nret_list.append(item*base_array)
            else:
                nret_list.append(item)
        ret_list=nret_list

    return np.array(ret_list)

def harm_sizes(ar, shape, dtype=np.float64):
    '''Utility function to add an additional axis if needed to fulfill the size in shape'''
    ar=np.array(ar, dtype=dtype)
    if shape is None:
        return ar
    elif len(ar.shape)<len(shape):
        return np.array(ar[..., np.newaxis]*np.ones(shape), dtype=dtype)
    elif ar.shape==shape:
        return ar
    else:
        raise TypeError('The size of the array, %s, can not be changed to shape %s'%(ar.shape, shape))

def is_reflfunction(obj):
    ''' Convenience function to determine whether obj belongs to the ReflFunction class.
    Return boolean.
    '''
    return obj.__class__.__name__=='ReflFunction'

def resolve_par(obj, par_name, max_recur=100):
    """ Resolves a layer parameter and takes into account a coupling to another object.
    If obj.par_name is of type obj the relation will be recursive, obj.par_name.par_name.
    A maximum of 100 levels are conducted before an error is thrown.

    :param obj: An object
    :param par_name: A name of a member
    :param max_recur: Maximum number of recursions.
    :return: a value
    """
    val=getattr(obj, par_name)
    i=0
    while type(val) is type(obj) and i<max_recur and not is_reflfunction(val):
        val=getattr(val, par_name)
        i+=1

    return val

class ReflBase:
    # _parameters is a dict of parameter names with their default values
    _parameters={}

    def __init__(self, **kargs):
        # Setup all the parameters in the class
        for par in self._parameters:
            setattr(self, par, self._parameters[par])
            iscomplex=type(self._parameters[par]) is complex
            # Making the set function
            self._make_set_func(par, iscomplex)
            # Creating the get function
            self._make_get_func(par, iscomplex)

        # Set all parameters given as keyword arguments
        for k in kargs:
            if not k in self._parameters and not k in dir(self):
                iprint('%s is not an parameter in %s so it is ignored'%
                       (k, self.__class__.__name__))
            else:
                setattr(self, k, kargs[k])

    def _make_get_func(self, par, iscomplex=False):
        ''' Creates a get function for parameter par and binds it to the object
        '''

        def get_func():
            return getattr(self, par)

        get_func.__name__='get'+par.capitalize()
        setattr(self, get_func.__name__, get_func)

        if iscomplex:
            def get_real_func():
                return getattr(self, par).real

            get_real_func.__name__=get_func.__name__+'real'
            setattr(self, get_real_func.__name__, get_real_func)

            def get_imag_func():
                return getattr(self, par).imag

            get_imag_func.__name__=get_func.__name__+'imag'
            setattr(self, get_imag_func.__name__, get_imag_func)

    def _make_set_func(self, par, iscomplex=False):
        ''' Creates a set function for parameter par and binds it to the object
            '''

        def set_func(val):
            setattr(self, par, val)

        set_func.__name__='set'+par.capitalize()
        setattr(self, set_func.__name__, set_func)

        if iscomplex:
            def set_real_func(val):
                setattr(self, par, val+getattr(self, par).imag*1J)

            set_real_func.__name__=set_func.__name__+'real'
            setattr(self, set_real_func.__name__, set_real_func)

            def set_imag_func(val):
                setattr(self, par, val*1J+getattr(self, par).real)

            set_imag_func.__name__=set_func.__name__+'imag'
            setattr(self, set_imag_func.__name__, set_imag_func)

    def get_parameters(self):
        ''' Returns all the parameters of the current object '''

        return self.parameters

class InstrumentBase(ReflBase):
    _parameters={}

    def __repr__(self):
        s='Instrument('

        for k in self._parameters:
            # if the type is a string...
            if type(getattr(self, k))==type(''):
                stemp="%s='%s', "%(k, str(getattr(self, k)))
            else:
                stemp='%s=%s, '%(k, str(getattr(self, k)))
            s=s+stemp
        return s[:-1]+')'

class LayerBase(ReflBase):
    _parameters={}

    def __repr__(self):
        s='Layer('
        for k in self._parameters:
            s+='%s=%s, '%(k, str(getattr(self, k)))
        return s[:-2]+')'

class StackBase(ReflBase):
    _parameters={'Repetitions': 1}
    Layers=[]
    Repetitions=1

    def __init__(self, **kargs):
        ReflBase.__init__(self, **kargs)
        if 'Layers' in kargs:
            self.Layers=kargs['Layers']
            kargs.pop('Layers')
        else:
            self.Layers=[]

    def __repr__(self):
        s='Stack: '
        for k in self._parameters:
            s+='%s=%s, '%(k, str(getattr(self, k)))
        s=s[:-2]+'\n'
        it=len(self.Layers)
        for lay in range(it-1, -1, -1):
            s+='\t'+repr(self.Layers[lay])+'\n'
        return s

    def resolveLayerParameter(self, parameter):
        par=[resolve_par(lay, parameter)
             for lay in self.Layers]*int(self.Repetitions)
        return par

# Class Sample

class SampleBase(ReflBase):
    _parameters={}
    Ambient=None
    Substrate=None
    Stacks=[]

    def __init__(self, **kargs):
        ReflBase.__init__(self, **kargs)
        special_objects=['Ambient', 'Substrate', 'Stacks']
        for p in special_objects:
            if p in kargs:
                setattr(self, p, kargs[p])
            else:
                raise ValueError('%s has to be defined to create a Sample object'%p)
        # Create a new dictonary for the new arguments
        nkargs={}
        for p in kargs:
            if not p in special_objects:
                nkargs[p]=kargs[p]

    def __repr__(self):
        Add='Sample: '
        for k in self._parameters:
            if type(getattr(self, k))==type(''):
                Add+="%s='%s', "%(k, str(getattr(self, k)))
            else:
                Add+='%s=%s, '%(k, str(getattr(self, k)))
            # Add += '%s=%s, ' % (k, getattr(self, k).__repr__())
        if len(self._parameters)>0:
            Add=Add[:-2]+'\n'
        else:
            Add=Add[:-1]+'\n'
        temp=[repr(x) for x in self.Stacks]
        temp.reverse()
        return (Add+'Ambient:\n\t'+repr(self.Ambient)+'\n'+''.join(temp)
                +'Substrate:\n\t'+repr(self.Substrate))

    def resolveLayerParameters(self):
        par=self.Substrate._parameters.copy()
        for k in par:
            par[k]=[resolve_par(self.Substrate, k)]
        for k in self.Substrate._parameters:
            for stack in self.Stacks:
                par[k]=par[k]+stack.resolveLayerParameter(k)
            par[k]=par[k]+[resolve_par(self.Ambient, k)]
        return par

# Factory needs to:
#            add members and their names in the _parameter list
#            Add set and get methods for each parameter
class ModelFactory:
    ''' Class that creates new classes to implement certain parameters 
        dynamically.
    '''

    def __init__(self):
        class Layer(LayerBase): _parameters=LayerBase._parameters.copy()

        self.Layer=Layer

        class Stack(StackBase): _parameters=StackBase._parameters.copy()

        self.Stack=Stack

        class Sample(SampleBase): _parameters=SampleBase._parameters.copy()

        self.Sample=Sample

        class Instrument(InstrumentBase): _parameters=InstrumentBase._parameters.copy()

        self.Instrument=Instrument

    @staticmethod
    def _add_parameter(cls, par_name, def_val):
        cls._parameters.update({par_name: def_val})

    @staticmethod
    def _add_sim_method(cls, name, func):
        def method(self, *args):
            nargs=args[:-1]+(self,)+(args[-1],)
            return func(*nargs)

        method.__name__='Sim'+name
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
def MakeClasses(InstrumentParameters=None,
                LayerParameters=None,
                StackParameters=None,
                SampleParameters=None,
                SimulationFunctions=None,
                ModelID='Standard'):
    if InstrumentParameters is None:
        InstrumentParameters={'Wavelength': 1.54, 'Coordinates': 1}
    if LayerParameters is None:
        LayerParameters={'d': 0.0, 'sigma': 0.0, 'n': 0.0+0.0j}
    if StackParameters is None:
        StackParameters={}
    if SampleParameters is None:
        SampleParameters={}
    if SimulationFunctions is None:
        SimulationFunctions={'Specular': lambda x: x, 'OffSpecular': lambda x: x}
    factory=ModelFactory()
    factory.set_layer_parameters(LayerParameters)
    factory.set_stack_parameters(StackParameters)
    factory.set_sample_parameters(SampleParameters)
    factory.set_simulation_functions(SimulationFunctions)
    factory.set_instrument_parameters(InstrumentParameters)
    Layer=factory.get_layer_class()
    Stack=factory.get_stack_class()
    Sample=factory.get_sample_class()
    Instrument=factory.get_instrument_class()
    return Instrument, Layer, Stack, Sample

if __name__=='__main__':
    InstrumentParameters={'probe': 'x-ray', 'wavelength': 1.54, 'coords': 'tth',
                          'I0': 1.0, 'res': 0.001,
                          'restype': 'no conv', 'respoints': 5, 'resintrange': 2.0, 'beamw': 0.01,
                          'footype': 'no corr', 'samplelen': 10.0, 'incangle': 0.0, 'pol': 'uu',
                          'Ibkg': 0.0, 'tthoff': 0.0}

    LayerParameters={'sigma': 0.0, 'dens': 1.0, 'd': 0.0, 'f': (1.0+1.0j)*1e-20,
                     'b': 0.0+0.0J, 'xs_ai': 0.0, 'magn': 0.0, 'magn_ang': 0.0}
    StackParameters={'Layers': [], 'Repetitions': 1}
    SampleParameters={'Stacks': [], 'Ambient': None, 'Substrate': None}

    (Instrument, Layer, Stack, Sample)=MakeClasses(InstrumentParameters,
                                                   LayerParameters, StackParameters, SampleParameters,
                                                   {'Specualr': lambda x: x},
                                                   'test')
    inst=Instrument(footype='gauss beam', probe='x-ray', beamw=0.04, resintrange=2, pol='uu', wavelength=1.54,
                    respoints=5, Ibkg=0.0, I0=2, samplelen=10.0, restype='fast conv', coords='2θ', res=0.001,
                    incangle=0.0)

    # BEGIN Sample DO NOT CHANGE
    Amb=Layer(b=0, d=0.0, f=(1e-20+1e-20j), dens=1.0, magn_ang=0.0, sigma=0.0, xs_ai=0.0, magn=0.0)
    topPt=Layer(b=0, d=11.0, f=58, dens=4/3.92**3, magn_ang=0.0, sigma=3.0, xs_ai=0.0, magn=0.0)
    TopFe=Layer(b=0, d=11.0, f=26, dens=2/2.866**3, magn_ang=0.0, sigma=2.0, xs_ai=0.0, magn=0.0)
    Pt=Layer(b=0, d=11.0, f=58, dens=4/3.92**3, magn_ang=0.0, sigma=2.0, xs_ai=0.0, magn=0.0)
    Fe=Layer(b=0, d=11, f=26, dens=2/2.866**3, magn_ang=0.0, sigma=2.0, xs_ai=0.0, magn=0.0)
    bufPt=Layer(b=0, d=45, f=58, dens=4/3.92**3, magn_ang=0.0, sigma=2, xs_ai=0.0, magn=0.0)
    bufFe=Layer(b=0, d=2, f=26, dens=2/2.866**3, magn_ang=0.0, sigma=2, xs_ai=0.0, magn=0.0)
    Sub=Layer(b=0, d=0.0, f=12+16, dens=2/4.2**3, magn_ang=0.0, sigma=4.0, xs_ai=0.0, magn=0.0)

    top=Stack(Layers=[TopFe, topPt], Repetitions=1)
    ML=Stack(Layers=[Fe, Pt], Repetitions=19)
    buffer=Stack(Layers=[bufFe, bufPt], Repetitions=1)

    sample=Sample(Stacks=[buffer, ML, top], Ambient=Amb, Substrate=Sub)

    # Test case for ReflFunction class
    from scipy import interpolate

    def create_dispersion_func(name='Fe'):
        path='../databases/f1f2_nist/'
        e, f1, f2=np.loadtxt(path+'%s.nff'%name.lower(), skiprows=1,
                             unpack=True)
        f1interp=interpolate.interp1d(e, f1, kind='linear')
        f2interp=interpolate.interp1d(e, f2, kind='linear')

        def f(energy):
            return f1interp(energy)-1.0J*f2interp(energy)

        return f

    fFe=ReflFunction(create_dispersion_func('Fe'), (1000,), {}, id='f(E)')
    fCo=ReflFunction(create_dispersion_func('Co'), (1000,), {}, id='f(E)')
    iprint(fFe.validate())
    iprint('Cast to array tests:')
    ltest=[fFe, 3.0, 3.0]
    iprint('Test single value: ', cast_to_array(ltest, 1000))
    iprint('Test array: ', cast_to_array(ltest, np.arange(1000, 1010)))
    iprint('Cast to array finished.')
