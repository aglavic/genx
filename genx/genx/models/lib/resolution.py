"""
Classes for advanced resolution functions to be used instead of gaussian convolution.

All resolution classes follow the initial Resolution protocol to allow the model to evaluate them within the simulation.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np

from .base import ModelParamBase
from .instrument import ResolutionVector

__all__ = ["GaussianResolution", "TrapezoidResolution"]


class Resolution(ModelParamBase):
    """
    Base class for resolution calculations, needs to be sub-classed to provide additional functionality.

    Any subclass has to implement the __call__ method that applies its correction to an angle and sample length.
    The method returns a tuple of the 2 dimensional Q-positions for each calculation point and
    the weights for each. Later integration is done along the first axis. (See ResolutionVector for example.)
    """

    def __call__(self, TwoThetaQz, respoints=15, resintrange=2.0):
        raise NotImplementedError("Subclass must implement __call__ method")

    def get_weight_example(self):
        # Return a test sample of the resolution function for a q=0.1 and default parameters
        qres, weight = self(np.array([0.1]))
        weight = weight.flatten() / weight.max()
        return qres, weight.flatten()


@dataclass
class GaussianResolution(Resolution):
    sigma: Union[float, np.ndarray] = 0.005

    def __call__(self, TwoThetaQz, respoints=15, resintrange=2.0):
        # (TwoThetaQz, weight) =
        return ResolutionVector(TwoThetaQz[:], self.sigma, respoints, range=resintrange)


@dataclass
class TrapezoidResolution(Resolution):
    inner_width: Union[float, np.ndarray] = 0.02
    outer_width: Union[float, np.ndarray] = 0.0

    def __call__(self, TwoThetaQz, respoints=15, resintrange=2.0):
        full_width = np.maximum(self.inner_width, self.outer_width)
        scale = np.linspace(-0.5, 0.5, respoints)[:, np.newaxis]
        Qres = TwoThetaQz + full_width * scale
        # relative size of inner and outer widths, make sure the shape is like TwoThetaQz (widths can be float or array)
        inner_rel = self.inner_width / full_width * np.ones_like(TwoThetaQz)
        weight = np.where(2.0 * abs(scale) <= inner_rel, 1.0, 1.0 - (2.0 * abs(scale) - inner_rel) / (1.0 - inner_rel))
        return (Qres.flatten(), weight)
