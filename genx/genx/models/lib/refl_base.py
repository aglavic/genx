"""
Module to provide the Layer - Stack - Sample classes to build a sample for reflectivity modelling.
The new implementation is based on python dataclasses.

Classes:
ReflFunction - A function class that can be used as a parameter in the other classes.
is_reflfunction - Funtion that checks if an object belongs to the class ReflFunction
ReflBase - Base class for all the physical classes.
LayerBase - Base Layer class.
StackBase - Base Stack class.
SampleBase - Base Sample class.
InstrumentBase - Base Instrument class.

A model should derive its own data classes as base classe of the above like:
::
    @dataclass
    class Instrument(ReflBase):
        wavelength: float = 0.0

Choices should use appropriate Enum to simplify comparison. An example would be:
::
    class Polarization(str, enum.Enum):
        up_up = 'uu'
        down_down = 'dd'
        up_down = 'ud'
        down_up = 'du'

    @dataclass
    class Instrument(ReflBase):
        pol: Polarization = Polarization.up_up
"""

from dataclasses import dataclass, field, fields
from typing import List

import numpy as np

from .base import ModelParamBase


class ReflBase(ModelParamBase):
    """
    ModelParamBased derived class for use in reflectometry model.
    This class is required for the GUI to recognize classes for use with the Reflectometry plugin.
    """

    Groups = []

    @property
    def _parameters(self):
        # for legacy parameter lookup
        return dict([(fi.name, getattr(self, fi.name)) for fi in fields(self)])

    def _parameter_info(self):
        # field information for all parameters
        return dict([(fi.name, fi) for fi in fields(self)])


@dataclass
class StackBase(ReflBase):
    Layers: List[ReflBase] = field(default_factory=list)
    Repetitions: int = 1

    def resolveLayerParameter(self, name):
        return [getattr(li, name) for li in self.Layers] * self.Repetitions


@dataclass
class SampleBase(ReflBase):
    Stacks: List[StackBase] = field(default_factory=list)
    Ambient: ReflBase = None
    Substrate: ReflBase = None

    # following attribute can be used in sub-class to specify dataclass to return on resolveLayerParameters
    _layer_parameter_class = None

    def _resolve_parameter(self, obj, key):
        return getattr(obj, key)

    def resolveLayerParameters(self):
        par = {}
        for fi in fields(self.Substrate):
            par[fi.name] = []
            par[fi.name] = [self._resolve_parameter(self.Substrate, fi.name)]
            for stack in self.Stacks:
                par[fi.name] += stack.resolveLayerParameter(fi.name)
            par[fi.name].append(self._resolve_parameter(self.Ambient, fi.name))
        if self._layer_parameter_class:
            return self._layer_parameter_class(**par)
        else:
            return par

    @classmethod
    def _add_sim_method(cls, name, func):
        def method(self, *args):
            nargs = args[:-1] + (self,) + (args[-1],)
            return func(*nargs)

        method.__name__ = "Sim" + name
        setattr(cls, method.__name__, method)

    @classmethod
    def setSimulationFunctions(cls, sim_functions):
        [cls._add_sim_method(k, func) for k, func in sim_functions.items()]


class ReflFunction:
    def __init__(self, function, validation_args, validation_kwargs, id=None):
        """Creates the Refl Function given function. The arguments validation_args and
        validation_kwargs will be used to the validate the returned type of the function by passing
        them to function. The variable id should be a unique string to identify the type ReflFunction.
        """
        self.__func__ = function
        self.validation_args = validation_args
        self.validation_kwargs = validation_kwargs
        self.id = id

    def __call__(self, *args, **kwargs):
        return self.__func__(*args, **kwargs)

    def validate(self):
        """Function to test that the function returns the anticipated type"""
        return self.__call__(*self.validation_args, **self.validation_kwargs)

    def _check_obj(self, other):
        """Checks the object other so that it fulfills the demands for arithmetic operations."""
        supported_types = [int, float, int, complex, np.float64, np.float32]
        if is_reflfunction(other):
            if self.id != other.id:
                raise TypeError("Two ReflFunction objects must have identical id's to conduct arithmetic operations")
        elif not type(other) in supported_types:
            raise TypeError(
                "%s is not supported for arithmetic operations " % (repr(type(other)))
                + "of a ReflFunction. It has to int, float, long or complex"
            )

    def __mul__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):

            def new_func(*args, **kwargs):
                return self(*args, **kwargs) * other(*args, **kwargs)

        else:

            def new_func(*args, **kwargs):
                return self(*args, **kwargs) * other

        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __rmul__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):

            def new_func(*args, **kwargs):
                return other(*args, **kwargs) * self(*args, **kwargs)

        else:

            def new_func(*args, **kwargs):
                return other * self(*args, **kwargs)

        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __add__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):

            def new_func(*args, **kwargs):
                return self(*args, **kwargs) + other(*args, **kwargs)

        else:

            def new_func(*args, **kwargs):
                return self(*args, **kwargs) + other

        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __radd__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):

            def new_func(*args, **kwargs):
                return other(*args, **kwargs) + self(*args, **kwargs)

        else:

            def new_func(*args, **kwargs):
                return other + self(*args, **kwargs)

        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __sub__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):

            def new_func(*args, **kwargs):
                return self(*args, **kwargs) - other(*args, **kwargs)

        else:

            def new_func(*args, **kwargs):
                return self(*args, **kwargs) - other

        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __rsub__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):

            def new_func(*args, **kwargs):
                return other(*args, **kwargs) - self(*args, **kwargs)

        else:

            def new_func(*args, **kwargs):
                return other - self(*args, **kwargs)

        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __div__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):

            def new_func(*args, **kwargs):
                return self(*args, **kwargs) / other(*args, **kwargs)

        else:

            def new_func(*args, **kwargs):
                return self(*args, **kwargs) / other

        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __rdiv__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):

            def new_func(*args, **kwargs):
                return other(*args, **kwargs) / self(*args, **kwargs)

        else:

            def new_func(*args, **kwargs):
                return other / self(*args, **kwargs)

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
                return self(*args, **kwargs) ** other(*args, **kwargs)

        else:

            def new_func(*args, **kwargs):
                return self(*args, **kwargs) ** other

        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)

    def __rpow__(self, other):
        self._check_obj(other)
        if is_reflfunction(other):

            def new_func(*args, **kwargs):
                return other(*args, **kwargs) ** self(*args, **kwargs)

        else:

            def new_func(*args, **kwargs):
                return other ** self(*args, **kwargs)

        return ReflFunction(new_func, self.validation_args, self.validation_kwargs, self.id)


def is_reflfunction(obj):
    """Convenience function to determine whether obj belongs to the ReflFunction class.
    Return boolean.
    """
    return obj.__class__.__name__ == "ReflFunction"


def cast_to_array(list_of_obj, *args, **kwargs):
    """Casts a list_of_obj, can be a number or an ReflFunction, into a list of evaluated values"""
    id = ""
    shape = False
    ret_list = []
    for obj in list_of_obj:
        if is_reflfunction(obj):
            if id == "":
                id = obj.id
                ret_list.append(obj(*args, **kwargs))
                # Check if we have got an array
                if not np.isscalar(ret_list[-1]):
                    shape = ret_list[-1].shape
            elif id == obj.id:
                ret_list.append(obj(*args, **kwargs))
            else:
                TypeError("Two ReflFunction objects must have identical id's in order to merge them into an array")
        else:
            # We assume that this is an object that can be transformed into an array later on.
            ret_list.append(obj)
    # if we have an array make sure that all the objects have the same shape
    if shape:
        base_array = np.ones(shape)
        nret_list = []
        for item in ret_list:
            if np.isscalar(item):
                nret_list.append(item * base_array)
            else:
                nret_list.append(item)
        ret_list = nret_list

    return np.array(ret_list)


def harm_sizes(ar, shape, dtype=np.float64):
    """Utility function to add an additional axis if needed to fulfill the size in shape"""
    ar = np.array(ar, dtype=dtype)
    if shape is None:
        return ar
    elif len(ar.shape) < len(shape):
        return np.array(ar[..., np.newaxis] * np.ones(shape), dtype=dtype)
    elif ar.shape == shape:
        return ar
    else:
        raise TypeError("The size of the array, %s, can not be changed to shape %s" % (ar.shape, shape))
