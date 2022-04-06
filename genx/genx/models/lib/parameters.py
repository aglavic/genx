#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
.. module:: parameters
   :synopsis: Classes to work with parameters and expressions thereof.

.. moduleauthor:: Matts Björck <matts.bjorck@gmail.com>

"""
import numpy as np

from collections.abc import MutableSequence
from genx.core.custom_logging import iprint

class Parameter(object):
    """Base class for all Parameters"""

    def __init__(self, help="", unit=""):
        self.help=help
        self.unit=unit

    def _get_value(self, **kwargs):
        raise NotImplementedError("Callback parameter method not implemented")

    def __call__(self, **kwargs):
        return self._get_value(**kwargs)

    def validate(self, value):
        """ Function to test that value is valid. Should raise a ValueError otherwise.
        """
        raise NotImplementedError("Validate method not implemented")

    def couple_parameter(self, parameter):
        """Function to call when a parameter is coupled (replaced) by another.
        This is used in HasParameters classes.
        """
        raise AttributeError('This parameter can not be set to another parameter')

    def has_coupled_parameter(self):
        """Returns True if the parameter is coupled to another parameter."""
        return False

def is_parameter(obj):
    """Returns true if object is a subclass of the class Parameter"""
    return isinstance(obj, Parameter)

def is_hasparameter(obj):
    """Returns true if object is a subclass of the class HasParameter"""
    return isinstance(obj, HasParameters)

class ArithmeticParameter(Parameter):
    """ A parameter that supports arithmetic calculations"""

    def _check_obj(self, other):
        """ Checks the object other so that it fulfills the demands for arithmetic operations.
        """
        supported_types=[int, float, int, complex, np.float64, np.float32]
        if is_parameter(other):
            pass
        elif not type(other) in supported_types:
            raise TypeError("%s is not supported for arithmetic operations "%(repr(type(other)))+
                            "of a Parameter. It has to be int, float, long or complex")

    def _new_object_from_func(self, value_func, help=""):
        """Create a new object from function"""
        new=ArithmeticParameter(help=help)
        new._get_value=value_func
        return new

    def __mul__(self, other):
        self._check_obj(other)
        if is_parameter(other):
            def new_func(**kwargs):
                return self(**kwargs)*other(**kwargs)
        else:
            def new_func(**kwargs):
                return self(**kwargs)*other
        return self._new_object_from_func(new_func)

    def __rmul__(self, other):
        self._check_obj(other)
        if is_parameter(other):
            def new_func(**kwargs):
                return other(**kwargs)*self(**kwargs)
        else:
            def new_func(**kwargs):
                return other*self(**kwargs)
        return self._new_object_from_func(new_func)

    def __add__(self, other):
        self._check_obj(other)
        if is_parameter(other):
            def new_func(**kwargs):
                return self(**kwargs)+other(**kwargs)
        else:
            def new_func(**kwargs):
                return self(**kwargs)+other
        return self._new_object_from_func(new_func)

    def __radd__(self, other):
        self._check_obj(other)
        if is_parameter(other):
            def new_func(**kwargs):
                return other(**kwargs)+self(**kwargs)
        else:
            def new_func(**kwargs):
                return other+self(**kwargs)
        return self._new_object_from_func(new_func)

    def __sub__(self, other):
        self._check_obj(other)
        if is_parameter(other):
            def new_func(**kwargs):
                return self(**kwargs)-other(**kwargs)
        else:
            def new_func(**kwargs):
                return self(**kwargs)-other
        return self._new_object_from_func(new_func)

    def __rsub__(self, other):
        self._check_obj(other)
        if is_parameter(other):
            def new_func(**kwargs):
                return other(**kwargs)-self(**kwargs)
        else:
            def new_func(**kwargs):
                return other-self(**kwargs)
        return self._new_object_from_func(new_func)

    def __div__(self, other):
        self._check_obj(other)
        if is_parameter(other):
            def new_func(**kwargs):
                return self(**kwargs)/other(**kwargs)
        else:
            def new_func(**kwargs):
                return self(**kwargs)/other
        return self._new_object_from_func(new_func)

    def __rdiv__(self, other):
        self._check_obj(other)
        if is_parameter(other):
            def new_func(**kwargs):
                return other(**kwargs)/self(**kwargs)
        else:
            def new_func(*args, **kwargs):
                return other/self(*args, **kwargs)
        return self._new_object_from_func(new_func)

    def __neg__(self):
        def new_func(**kwargs):
            return -self(**kwargs)

        return self._new_object_from_func(new_func)

    def __pos__(self):
        def new_func(**kwargs):
            return self(**kwargs)

        return self._new_object_from_func(new_func)

    def __pow__(self, other):
        self._check_obj(other)
        if is_parameter(other):
            def new_func(**kwargs):
                return self(**kwargs)**other(**kwargs)
        else:
            def new_func(**kwargs):
                return self(**kwargs)**other
        return self._new_object_from_func(new_func)

    def __rpow__(self, other):
        self._check_obj(other)
        if is_parameter(other):
            def new_func(**kwargs):
                return other(**kwargs)**self(**kwargs)
        else:
            def new_func(**kwargs):
                return other**self(**kwargs)
        return self._new_object_from_func(new_func)

class NumericParameter(ArithmeticParameter):
    """A parameter holding a numeric value"""

    def __init__(self, value, help="", unit=""):
        ArithmeticParameter.__init__(self, help, unit=unit)
        self.value=value
        self._coupled_parameter=None

    def validate(self, value):
        raise NotImplementedError("Validation of value not implemented")

    def __setattr__(self, key, value):
        if key=='value':
            object.__setattr__(self, 'value', self.validate(value))
            self._coupled_parameter=None
        else:
            object.__setattr__(self, key, value)

    def _get_value(self, **kwargs):
        """Callback for getting the value of the parameter"""
        if not self.has_coupled_parameter():
            return self.value
        else:
            return self.validate(self._coupled_parameter(**kwargs))

    def couple_parameter(self, parameter):
        """Couple a parameter to this parameter, replacing the _get_value output with its output"""
        object.__setattr__(self, '_coupled_parameter', parameter)

    def has_coupled_parameter(self):
        """Returns True if the parameter is coupled to another parameter."""
        return self._coupled_parameter is not None

class Enum(Parameter):
    """An enumeration object"""

    def __init__(self, allowed_values=None, help=""):
        Parameter.__init__(self, help=help)
        if allowed_values is None:
            allowed_values=[]
        self.allowed_values=allowed_values
        self.value=self.allowed_values[0]

    def __setattr__(self, key, value):
        if key=='value':
            object.__setattr__(self, 'value', self.validate(value))
        else:
            object.__setattr__(self, key, value)

    def validate(self, value):
        """Validates the value. Returns a proper value else raises an ValueError"""
        if value in self.allowed_values:
            return value
        elif isinstance(value, int):
            if len(self.allowed_values)>value>0:
                return self.allowed_values[0]
        raise ValueError("The value to an enum must be either an integer"
                         " between 0 and %d or one of %s"%(len(self.allowed_values), str(self.allowed_values)))

    def _get_value(self, **kwargs):
        return self.value

class List(MutableSequence):
    """A List object with type checking"""
    protected=True

    def __init__(self, allowed_type=None, value=None, help=""):
        if value is None:
            value=[]
        self.help=help
        self.allowed_type=allowed_type
        self.data=list()
        for item in value:
            self._check(item)
            self.data.append(item)

    def _check(self, value):
        if self.allowed_type is not None and not isinstance(value, self.allowed_type):
            raise ValueError("List object can only consist of type %s, trying to set a type: %s"%
                             (self.allowed_type, type(value)))

    def validate(self, value):
        self._check(value)
        return value

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        # print "Setting ", key, " to ", value
        if isinstance(key, slice):
            for item in value:
                self._check(item)
        else:
            self.validate(value)
        self.data.__setitem__(key, value)

    def __delitem__(self, key):
        self.data.__delitem__(key)

    def __len__(self):
        return len(self.data)

    def insert(self, index, value):
        self._check(value)
        self.data.insert(index, value)

class HasParameters(object):
    """Mixin class to handle the setting of parameters in a correct manner.

    Attributes that are parameter can not be set to an incompatible type. This is done by overloading the __setattr__
    method.

    Also if an attribute implements a boolean attribute protected which is True then this attribute can not be set.
    """
    validation_kwargs={}

    def __init__(self, **kwargs):
        """The given allowed keyword arguments allow to initialise parameters"""
        for name in kwargs:
            if hasattr(self, name):
                self.__setattr__(name, kwargs[name])
            else:
                raise TypeError("Creation of %s got an unexpected keyword argument %s"%
                                (self.__class__.__name__, name))

    def __setattr__(self, name, value):
        try:
            attr=self.__getattribute__(name)
        except AttributeError:
            # Attribute does not exist
            object.__setattr__(self, name, value)
        else:
            if is_parameter(attr) and not self._is_protected(attr):
                if is_parameter(value) or is_hasparameter(value):
                    # Trying to set a Parameter to a Parameter...
                    # check so that the type is correct:
                    try:
                        attr.validate(value(parameter=name, **self.validation_kwargs.copy()))
                    except ValueError as e:
                        raise ValueError('Can not set attribute %s. %s'%(name, str(e)))
                    else:
                        # print "Coupling parameter: %s" % name
                        attr.couple_parameter(value)
                elif isinstance(attr, NumericParameter):
                    attr.value=value
                else:
                    raise AttributeError("Setting attribute %s is not allowed"%name)
            else:
                if self._is_protected(attr):
                    raise AttributeError("Setting attribute %s is not allowed"%name)
                else:
                    object.__setattr__(self, name, value)

    def __call__(self, **kwargs):
        """A call to the object. Will try to look up parameter as a member.

        Parameters:
            parameter(string): Name of parameter to look up

        """
        if 'parameter' not in kwargs:
            raise TypeError('Expected argument "parameter" to of a HasParameters object')

        return self.__getattribute__(kwargs['parameter'])(**kwargs)

    def _is_protected(self, attr):
        """Check if an attribute is marked as protected (implements variable protected and set to True)"""
        try:
            protected=attr.__getattribute__("protected")
        except AttributeError:
            protected=False
        return protected

    def get_parameter_list(self, numeric_types_only=False):
        """Returns a list of all the Parameters that exists

        Members that are "private", the name starting with a "_" will not be listed.

        Parameters:
            numeric_types_only (bool): if true only returns NumericParameter types.


        Returns:
            par_list (list): A list of strings with names of all the parameters.
        """
        allowed_parameter_classes=NumericParameter if numeric_types_only else Parameter
        # Loop thorough all member objects to find the allowed parameters.
        #  Note that if it is not included if it inherits HasParameters or beginning with a _
        members=self.__dict__
        par_list=[name for name in members
                  if isinstance(members[name], allowed_parameter_classes)
                  and not isinstance(members[name], HasParameters) and name[0]!='_'
                  ]
        # Loop through all the objects that inherit from HasParameter classes and add the full parameter name
        [[par_list.append(name+par_name) for par_name in
          members[name].get_parameter_list(numeric_types_only=numeric_types_only)
          ]
         for name in members if isinstance(members[name], HasParameters)
         ]
        return par_list

class Var(ArithmeticParameter):
    """ A model specific variable that will be included in the arguments to the evaluation."""

    def __init__(self, name, default_val, help=""):
        ArithmeticParameter.__init__(self, help)
        self.name=name
        self.default_val=default_val

    def _get_value(self, **kwargs):
        try:
            return kwargs[self.name]
        except KeyError:
            return self.default_val

class Calc(ArithmeticParameter):
    """A calculation done by a function of kwargs that is used as a parameter"""

    def __init__(self, function, help=""):
        ArithmeticParameter.__init__(self, help)
        self.function=function

    def _get_value(self, **kwargs):
        return self.function(**kwargs)

class Func(Calc, HasParameters):
    """ Simulates a function of parameters"""

    def __init__(self, **kwargs):
        HasParameters.__init__(self, **kwargs)

    def function(self, **kwargs):
        raise NotImplementedError("Method function in Func object not implemented.")

class Wrap(object):
    """ Defines a function where the argument can consist of parameters."""

    def __init__(self, function):
        self.function=function

    def __call__(self, *args):
        def wrap(**kwargs):
            new_args=[]
            for arg in args:
                if is_parameter(arg):
                    new_args.append(arg(**kwargs))
                else:
                    new_args.append(arg)
            return self.function(*new_args)

        wrap.__name__=self.function.__name__

        return Calc(wrap, help=wrap.__name__)

class Int(NumericParameter):
    """An integer parameter"""

    def validate(self, value):
        try:
            value=int(value)
        except Exception:
            raise ValueError('%s can not be cast to int'%value.__repr__())
        return value

class Float(NumericParameter):
    """An floating point parameter"""

    def validate(self, value):
        try:
            value=float(value)
        except Exception:
            raise ValueError('%s can not be cast to float'%value.__repr__())
        return value

class Complex(NumericParameter, HasParameters):
    """A complex number parameter"""

    def __init__(self, value, **kwargs):
        self.real=Float(0)
        self.imag=Float(0)
        NumericParameter.__init__(self, value, **kwargs)

    def validate(self, value):
        try:
            value=complex(value)
        except Exception:
            raise ValueError('%s can not be cast to complex'%value.__repr__())
        return value

    def __setattr__(self, key, value):
        if key=='value':
            value=self.validate(value)
            self.real.value=value.real
            self.imag.value=value.imag
        else:
            HasParameters.__setattr__(self, key, value)

    def _get_value(self, **kwargs):
        """Callback for getting the value of the parameter"""
        if self.has_coupled_parameter():
            return self.validate(self._coupled_parameter(**kwargs))
        else:
            return self.real(**kwargs)+1.0J*self.imag(**kwargs)

class FloatArray(NumericParameter):
    """A float array parameter"""

    # TODO: Add an array interface to support slicing as well...
    def validate(self, value):
        try:
            value=np.array(value, dtype=np.float64)
        except Exception:
            raise ValueError('%s can not be cast to float'%value.__repr__())
        return value

class ComplexArray(Complex):
    # TODO: Add an array interface to support slicing as well...
    """A complex array parameter"""

    def validate(self, value):
        try:
            value=np.array(value, dtype=np.complex128)
        except Exception:
            raise ValueError('%s can not be cast to complex array'%value.__repr__())
        return value

def get_parameters(obj, numeric_types_only=False, group_parameters=True):
    """Searches obj for members that are inherited from Parameters or HasParameters and places them in a dictonary.

    Parameters:
        obj: An object that has __dict__ (as all classes has.
        numeric_types_only (bool): returns only NumericParameters.
        group_parameters (bool): Flag to control if parameters should be grouped by type, default True.

    Returns:
        par_dict (dict): A Dictonary with class names as keys and items as lists of parameter names.
    """
    par_dict={}

    def add_parameter_to_dict(d, key, item):
        try:
            d[key].append(item)
        except KeyError:
            d[key]=[item]

    def add_object_to_dict(d, cls, name, object):
        try:
            dcls=d[cls]
        except KeyError:
            d[cls]={}
            dcls=d[cls]

        dcls[name]=object

    members=obj.__dict__
    allowed_parameter_classes=NumericParameter if numeric_types_only else Parameter
    # Loop thorough all member objects to find the allowed parameters. Note that if it is not included if it inherits
    # HasParameters
    [add_parameter_to_dict(par_dict, members[name].__class__.__name__, name)
     for name in members
     if name[0]!='_' and
     isinstance(members[name], allowed_parameter_classes) and
     not isinstance(members[name], HasParameters)
     ]
    # Loop through all the objects that inherit from HasParameter classes and do the same thing
    [add_object_to_dict(par_dict, members[name].__class__.__name__, name,
                        members[name].get_parameter_list(numeric_types_only=numeric_types_only)
                        )
     for name in members
     if isinstance(members[name], HasParameters)
     ]
    return par_dict
