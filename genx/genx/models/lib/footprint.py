"""
Classes for advanced footprint correction to be used instead of the default square / gaussian beam profile.

All footprint classes follow the initial Footprint protocol to allow the model to evaluate them within the simulation.
"""

import numpy as np
from dataclasses import dataclass, fields

from .instrument import GaussIntensity
from .base import ModelParamBase

__all__ = ['GaussianBeamOffset', 'SquareBeamOffset']

class Footprint(ModelParamBase):
    """
    Base class for footprint corrections, needs to be sub-classed to provide additional functionality.

    Any subclass has to implement the __call__ method that applies its correction to an angle and sample length.
    """

    def __call__(self, theta, samplen):
        raise NotImplementedError("Subclass must implement __call__ method")

@dataclass
class GaussianBeamOffset(Footprint):
    """
    The footpring for a gaussian beam with sample off-centered.
    """
    sigma: float = 0.5
    offset: float = 0.0

    def __call__(self, theta, samplen):
        sinalpha = np.sin(theta*np.pi/180.)

        left = (samplen/2.0)+self.offset/sinalpha
        right = (samplen/2.0)-self.offset/sinalpha
        return GaussIntensity(theta, left, right, self.sigma)

@dataclass
class SquareBeamOffset(Footprint):
    """
    The footpring for a square beam with sample off-centered.
    Offset has to be smaller than width.
    """
    width: float = 0.1
    offset: float = 0.0

    def __call__(self, theta, samplen):
        sinalpha = np.sin(theta*np.pi/180.)
        cross_section = samplen*sinalpha
        c_max = cross_section/2.0+self.offset
        c_min = -cross_section/2.0+self.offset
        FP = np.maximum(0., (np.minimum(c_max, self.width/2.)-np.maximum(c_min, -self.width/2.)))/self.width
        return FP
