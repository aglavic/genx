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
from typing import List, Tuple

from .base import ModelParamBase
from .refl import cast_to_array


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
