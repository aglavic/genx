import math
import numba
from numpy import empty, ndarray, empty_like

pi=math.pi
rad=pi/180.
sqrt2=math.sqrt(2.)

@numba.jit(
    numba.float64[:](numba.float64[:], numba.float64, numba.float64, numba.float64),
    nopython=True, parallel=True, cache=True)
def GaussIntensity(alpha, s1, s2, sigma_x):
    I=empty_like(alpha)
    if s1==s2:
        for ai in numba.prange(alpha.shape[0]):
            sinalpha=math.sin(alpha[ai]*rad)
            I[ai]=math.erf(s2/sqrt2/sigma_x*sinalpha)
    else:
        for ai in numba.prange(alpha.shape[0]):
            sinalpha=math.sin(alpha[ai]*rad)
            I[ai]=(math.erf(s1/sqrt2/sigma_x*sinalpha)+math.erf(s2/sqrt2/sigma_x*sinalpha))/2.
    return I

@numba.jit(
    numba.float64[:](numba.float64[:], numba.float64, numba.float64),
    nopython=True, parallel=True, cache=True)
def SquareIntensity(alpha, slen, beamwidth):
    I=empty_like(alpha)
    scale=slen/beamwidth
    for ai in numba.prange(alpha.shape[0]):
        F=scale*math.sin(alpha[ai]*rad)
        I[ai]=min(1.0, F)
    return I

@numba.jit(
    numba.float64[:](numba.float64, numba.float64[:]),
    nopython=True, parallel=True, cache=True)
def TwoThetatoQScalar(wavelength, TwoTheta):
    Q=empty_like(TwoTheta)
    scale=4.*pi/wavelength
    thscale=rad/2.0
    for li in numba.prange(Q.shape[0]):
        Q[li]=math.sin(TwoTheta[li]*thscale)*scale
    return Q

@numba.jit(
    numba.float64[:](numba.float64[:], numba.float64[:]),
    nopython=True, parallel=True, cache=True)
def TwoThetatoQVector(wavelength, TwoTheta):
    Q=empty_like(TwoTheta)
    scale=4.*pi
    thscale=rad/2.0
    for li in numba.prange(Q.shape[0]):
        Q[li]=math.sin(TwoTheta[li]*thscale)*scale/wavelength[li]
    return Q

def TwoThetatoQ(wavelength, TwoTheta):
    if type(wavelength) is ndarray:
        return TwoThetatoQVector(wavelength, TwoTheta)
    else:
        return TwoThetatoQScalar(wavelength, TwoTheta)

@numba.jit(
    numba.float64[:](numba.float64, numba.float64[:]),
    nopython=True, parallel=True, cache=True)
def QtoThetaScalar(wavelength, Q):
    Theta=empty_like(Q)
    scale=wavelength/4.0/pi
    for li in numba.prange(Q.shape[0]):
        Theta[li]=math.asin(scale*Q[li])/rad
    return Theta

@numba.jit(
    numba.float64[:](numba.float64[:], numba.float64[:]),
    nopython=True, parallel=True, cache=True)
def QtoThetaVector(wavelength, Q):
    Theta=empty_like(Q)
    scale=1.0/4.0/pi
    for li in numba.prange(Q.shape[0]):
        Theta[li]=math.asin(scale*wavelength[li]*Q[li])/rad
    return Theta

def QtoTheta(wavelength, Q):
    if type(wavelength) is ndarray:
        return QtoThetaVector(wavelength, Q)
    else:
        return QtoThetaScalar(wavelength, Q)

#
@numba.jit(
    numba.types.Tuple((numba.float64[:], numba.float64[:, :]))(numba.float64[:], numba.float64, numba.int16,
                                                               numba.float64),
    nopython=True, parallel=True, cache=True)
def ResolutionVectorScalar(Q, dQ, points, range):
    Qres=empty(Q.shape[0]*points)
    weight=empty((points, Q.shape[0]))
    Qstep=2.0*range/(points-1)*dQ
    NQ=Q.shape[0]

    wscale=1.0/math.sqrt(2.0*pi)/dQ

    for ri in numba.prange(points):
        dq=(ri-(points-1.0)/2.0)*Qstep
        for qi in numba.prange(NQ):
            Qi=Q[qi]
            Qj=Qi+dq
            Qres[qi+ri*NQ]=Qj
            weight[ri, qi]=wscale*math.exp(-(Qi-Qj)**2/(dQ)**2/2.0)

    return (Qres, weight)

@numba.jit(
    numba.types.Tuple((numba.float64[:], numba.float64[:, :]))(numba.float64[:], numba.float64[:], numba.int16,
                                                               numba.float64),
    nopython=True, parallel=True, cache=True)
def ResolutionVectorVector(Q, dQ, points, range):
    Qres=empty(Q.shape[0]*points)
    weight=empty((points, Q.shape[0]))
    Qstep_scale=2.0*range/(points-1.0)
    NQ=Q.shape[0]

    wscale=1.0/math.sqrt(2.0*pi)

    for ri in numba.prange(points):
        dq_scale=(ri-(points-1.0)/2.0)*Qstep_scale
        for qi in numba.prange(NQ):
            dq=dq_scale*dQ[qi]
            Qi=Q[qi]
            Qj=Qi+dq
            Qres[qi+ri*NQ]=Qj
            weight[ri, qi]=wscale*math.exp(-(Qi-Qj)**2/(dQ[qi])**2/2.0)/dQ[qi]

    return (Qres, weight)

def ResolutionVector(Q, dQ, points, range=3):
    if type(dQ) is ndarray:
        return ResolutionVectorVector(Q, dQ, points, range)
    else:
        return ResolutionVectorScalar(Q, dQ, points, range)
