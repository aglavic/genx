"""
Classes for advanced footprint correction to be used instead of the default square / gaussian beam profile.

All footprint classes follow the initial Footprint protocol to allow the model to evaluate them within the simulation.
"""

from dataclasses import dataclass

import numpy as np

from .base import ModelParamBase
from .instrument import GaussIntensity

__all__ = ["GaussianBeamOffset", "SquareBeamOffset", "TrapezoidBeam"]


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
        sinalpha = np.sin(theta * np.pi / 180.0)

        left = (samplen / 2.0) + self.offset / sinalpha
        right = (samplen / 2.0) - self.offset / sinalpha
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
        sinalpha = np.sin(theta * np.pi / 180.0)
        cross_section = samplen * sinalpha
        c_max = cross_section / 2.0 + self.offset
        c_min = -cross_section / 2.0 + self.offset
        FP = np.maximum(0.0, (np.minimum(c_max, self.width / 2.0) - np.maximum(c_min, -self.width / 2.0))) / self.width
        return FP


@dataclass
class TrapezoidBeam(Footprint):
    """
    The footpring for a trapezoid beam. The intensity of the beam is constant
    on the inner_width and then reduces linearly until outer_width is reached.

    This footprint describes the typical neutron instrument situation, where the beam
    is defined by two slits, very well.
    """

    inner_width: float = 0.2
    outer_width: float = 0.3

    def __call__(self, theta, samplen):
        sinalpha = np.sin(theta * np.pi / 180.0)
        cross_section = samplen * sinalpha
        in_out_ratio = min(1.0, 2.0 * self.inner_width / (self.inner_width + self.outer_width))
        if self.inner_width > 0:
            FP_in = np.minimum(1.0, cross_section / self.inner_width)
        else:
            FP_in = 0.0
        if self.outer_width > self.inner_width:
            FP_out = np.minimum(
                1.0,
                np.maximum(
                    0.0,
                    1.0
                    - (
                        1.0
                        - np.minimum(1.0, (cross_section - self.inner_width) / (self.outer_width - self.inner_width))
                    )
                    ** 2,
                ),
            )
        else:
            FP_out = 0.0
        return FP_in * in_out_ratio + FP_out * (1.0 - in_out_ratio)
