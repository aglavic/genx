"""
Package containing all elemental data used for SLD calculations.

Main function is get_element(value) that returns an Element object for a given Z-number or identifier.
The objects are stored in a dictionary for faster lookup as well as keeping the objects for each element the same.
"""

from .element import Element


_ELEMENTS = {}


def get_element(value):
    if value in _ELEMENTS:
        return _ELEMENTS[value]
    if type(value) is int:
        res = Element(Z=value)
        if res.symbol in _ELEMENTS:
            return _ELEMENTS[res.symbol]
        else:
            _ELEMENTS[res.symbol] = res
        return res
    else:
        res = Element(symbol=value)
        _ELEMENTS[value] = res
    return res
