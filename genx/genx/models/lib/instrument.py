# Functions for instrumental resolution corrections and area of illumination
# these are to be included in the Models class and Refl class
# Note that these function assumes a Gaussian spread in both the angular as well
# as in real space

# THis one really needs scipy
from numpy import (abs, allclose, append, arange, arccos, arcsin, arctan, arctan2, array, bitwise_not, c_, complex64,
                   complex128, conj, convolve, cos, cumsum, dot, exp, float32, float64, imag, isscalar, linspace,
                   logical_not, maximum, minimum, ndarray, newaxis, not_equal, ones, ones_like, pi, r_, real, rollaxis,
                   sin, sqrt, sum, tan, transpose, where, zeros, zeros_like)

try:
    # trapz removed in numpy 2.x
    from numpy import trapezoid as trapz
except ImportError:
    from numpy import trapz

from scipy.special import erf

from genx.core.custom_logging import iprint

rad = pi / 180.0
sqrt2 = sqrt(2.0)


# Intensity correction Gaussian beamprofile
def GaussIntensity(alpha, s1, s2, sigma_x):
    sinalpha = sin(alpha * rad)
    if s1 == s2:
        return erf(s2 / sqrt2 / sigma_x * sinalpha)
    else:
        common = sinalpha / sqrt2 / sigma_x
        return (erf(s2 * common) + erf(s1 * common)) / 2.0


# Specular foorprintcorrections square beamprofile
def SquareIntensity(alpha, slen, beamwidth):
    F = slen / beamwidth * sin(alpha * rad)
    return where(F <= 1.0, F, ones(F.shape))


####################################################################
## Resolution Functions (Normal Distributed=Gaussian)
#####################################################################

# Full Convlutions - vayring resolution


# Function to create a 1D vector for the resolution with the
# positions to calculate the reflectivity Qret and the weight
# of each point weight
# Inputs: Q - the Q values
#         dQ - the resolution
#         points - the number of points for the convolution
#         range how far the gaussian should be convoluted
def ResolutionVector(Q, dQ, points, range=3):
    Qres = Q + dQ * linspace(-range, range, points)[:, newaxis]
    weight = 1 / sqrt(2 * pi) / dQ * exp(-((transpose(Q[:, newaxis]) - Qres) ** 2) / dQ**2 / 2)
    Qret = Qres.flatten()
    return Qret, weight


# Include the resolution with Qret and weight calculated from ResolutionVector
# and I the calculated intensity at each point. returns the intensity
def ConvoluteResolutionVector(Qret, I, weight):
    Qret2 = Qret.reshape(weight.shape[0], weight.shape[1])
    I2 = I.reshape(weight.shape[0], weight.shape[1])
    norm_fact = trapz(weight, x=Qret2, axis=0)
    Int = trapz(I2 * weight, x=Qret2, axis=0) / norm_fact
    return Int


# Fast convlution - constant resolution
# constant spacing between data!
def ConvoluteFast(Q, I, dQ, range=3):
    Qstep = Q[1] - Q[0]
    resvector = arange(-range * dQ, range * dQ + Qstep, Qstep)
    weight = 1 / sqrt(2 * pi) / dQ * exp(-(resvector**2) / dQ**2 / 2)
    Iconv = convolve(
        r_[ones(resvector.shape) * I[0], I, ones(resvector.shape) * I[-1]], weight / weight.sum(), mode="same"
    )[resvector.shape[0] : -resvector.shape[0]]
    return Iconv


# Fast convolution - varying resolution
# constant spacing between the dat.
def ConvoluteFastVar(Q, I, dQ, range=3):
    Qstep = Q[1] - Q[0]
    steps = max(dQ * ones(Q.shape)) * range / Qstep
    weight = 1 / sqrt(2 * pi) / dQ * exp(-((Q[:, newaxis] - Q) ** 2) / dQ**2 / 2)
    Itemp = I[:, newaxis] * ones(I.shape)
    norm_fact = trapz(weight, axis=0)
    Int = trapz(Itemp * weight, axis=0) / norm_fact
    return Int


def QtoTheta(wavelength, Q):
    return arcsin(wavelength / 4.0 / pi * Q) / rad


def TwoThetatoQ(wavelength, TwoTheta):
    return 4.0 * pi / wavelength * sin(rad / 2.0 * TwoTheta)


########## Neutron Polarization #################
def get_pol_matrix(p1, p2, F1, F2):
    # @formatter:off
    P1 = array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [F1, 0, (1.0 - F1), 0],
            [0, F1, 0, (1.0 - F1)],
        ]
    )
    P2 = array(
        [
            [1, 0, 0, 0],
            [F2, (1.0 - F2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, F2, (1.0 - F2)],
        ]
    )
    P3 = array(
        [
            [(1.0 - p1), 0, p1, 0],
            [0, (1.0 - p1), 0, p1],
            [p1, 0, (1.0 - p1), 0],
            [0, p1, 0, (1.0 - p1)],
        ]
    )
    P4 = array(
        [
            [(1.0 - p2), p2, 0, 0],
            [p2, (1.0 - p2), 0, 0],
            [0, 0, (1.0 - p2), p2],
            [0, 0, p2, (1.0 - p2)],
        ]
    )
    # @formatter:on
    P = dot(dot(P1, P2), dot(P3, P4))
    return P


from . import USE_NUMBA

if USE_NUMBA:
    # try to use numba to speed up the calculation intensive functions:
    try:
        from .instrument_numba import GaussIntensity, QtoTheta, ResolutionVector, SquareIntensity, TwoThetatoQ
    except Exception as e:
        iprint("Could not use numba, no speed up from JIT compiler:\n" + str(e))
