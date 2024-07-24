"""
Classes for advanced resolution functions to be used instead of gaussian convolution.

All resolution classes follow the initial Resolution protocol to allow the model to evaluate them within the simulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Union

from .instrument import ResolutionVector
from .base import ModelParamBase

__all__ = ['GaussianResolution']

class Resolution(ModelParamBase):
    """
    Base class for resolution calculations, needs to be sub-classed to provide additional functionality.

    Any subclass has to implement the __call__ method that applies its correction to an angle and sample length.
    The method returns a tuple of the 2 dimensional Q-positions for each calculation point and
    the weights for each. Later integration is done along the first axis. (See ResolutionVector for example.)
    """

    def __call__(self, TwoThetaQz, respoints=15, resintrange=2.0):
        raise NotImplementedError("Subclass must implement __call__ method")

@dataclass
class GaussianResolution(Resolution):
    sigma: Union[float, np.ndarray]

    def __call__(self, TwoThetaQz, respoints=15, resintrange=2.0):
        # (TwoThetaQz, weight) =
        return ResolutionVector(TwoThetaQz[:], self.sigma, respoints, range=resintrange)

@dataclass
class TrapezoidResolution(Resolution):
    inner_width: Union[float, np.ndarray] = 0.2
    outer_width: Union[float, np.ndarray] = 0.3

    def __call__(self, TwoThetaQz, respoints=15, resintrange=2.0):
        # (TwoThetaQz, weight) =
        a = -self.outer_width/2.
        b = -self.inner_width/2.
        c = self.inner_width/2.
        d = self.outer_width/2.
        scale = np.linspace(-0.5, 0.5, respoints)[:,np.newaxis]
        Qres = TwoThetaQz+self.outer_width*scale
        norm = 1./(self.inner_width+self.outer_width)
        inner_rel = self.inner_width/self.outer_width
        #weight = norm*np.where(2*abs(scale)<=inner_rel, 1.0, (x-a)/(b-a))
        return (Qres.flatten(), weight)
    # Qres = Q+dQ*linspace(-range, range, points)[:, newaxis]
    # weight = 1/sqrt(2*pi)/dQ*exp(-(transpose(Q[:, newaxis])-Qres)**2/dQ**2/2)
    # Qret = Qres.flatten()
