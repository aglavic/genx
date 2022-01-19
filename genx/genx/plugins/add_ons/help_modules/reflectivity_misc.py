"""
Typing support for reflectivity plugins.
Could be extended with other small helpers.
"""

from genx.models.lib.refl import InstrumentBase, LayerBase, SampleBase, StackBase


try:
    # noinspection PyUnresolvedReferences
    from typing import Protocol, Iterable, Dict, Any, Type
except ImportError:
    ReflectivityModule = object
else:
    class ReflectivityModule(Protocol):
        """
        Defines the attributes that are expected in a python module that defines a reflectivity model.
        This is used to type check what the classes in this file use.
        """
        sample_string_choices: Dict[str, Iterable[str]]
        SampleParameters: Dict[str, Any]
        SampleGroups: Iterable
        SampleUnits: Dict[str, str]

        instrument_string_choices: Dict[str, Iterable[str]]
        InstrumentParameters: Dict[str, Any]
        InstrumentGroups: Iterable
        InstrumentUnits: Dict[str, str]

        LayerParameters: Dict[str, Any]
        LayerGroups: Iterable
        LayerUnits: Dict[str, str]

        StackParameters: Dict[str, Any]
        StackGroups: Iterable
        StackUnits: Dict[str, str]

        Instrument: Type[InstrumentBase]
        Sample: Type[SampleBase]
        Stack: Type[StackBase]
        Layer: Type[LayerBase]
