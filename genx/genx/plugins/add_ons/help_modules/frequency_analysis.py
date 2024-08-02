"""
Helper functions used to analyse reflectivity using various frequency estimation techniques.
"""

from enum import Enum

from numpy import cumsum, exp, linspace, log, maximum, newaxis, pi, sqrt, trapz, zeros, zeros_like


def NDFT(q, y, d=None):
    """
    Non-uniform Discrete Fourier Transform.

    There is a faster NFFT approximation but for typical reflectivity datasets the computation time is low.
    """
    if d is None:
        # if no output grid is defined, use a regular spaced grid with same number of points
        d = linspace(0, len(q) * pi / q.max(), len(q))
    FT = zeros(d.shape, dtype=complex)
    for qi, yi in zip(q, y):
        FT += yi * exp(1j * qi * d)
    return d, FT


def STFT(q, y, d=None, tau=1.0):
    """
    Short Time Frourier Transform.
    """
    return_d = False
    if d is None:
        return_d = True
        d = linspace(0, len(q) * 2.0 * pi / (q.max() - q.min()), len(q))
    FT = zeros(len(d), dtype=complex)
    for i, di in enumerate(d):
        FT[i] = trapz(y * _G(y / 2.0 / pi - di, tau) * exp(1j * di * q), x=q)
    if return_d:
        return d, FT
    else:
        return FT


def _G(t, tau):
    return exp(-pi / 2.0 * (t / tau) ** 2)


def MexicanHat(t):
    return (1.0 - t**2) * exp(-(t**2) / 2.0)


def Morlet(t, sigma=1.0):
    return pi ** (-0.25) * exp(-(t**2) / 2.0) * (exp(1j * sigma * t) - exp(-(sigma**2) / 2.0))


def CWT(q, y, wavelet=MexicanHat, d=None, N=20):
    """
    Wavelet transform using a mexican hat function.
    """
    return_d = False
    if d is None:
        return_d = True
        d = linspace(0, len(q) * 2.0 * pi / (q.max() - q.min()), len(q))

    #    b=10**linspace(log10(pi*len(d)/(d.max()-d.min())), log10(d.max()), N)
    b = linspace(5.0, d.max(), N)

    WT = zeros((len(d), len(b)), dtype=complex)
    for i, di in enumerate(d):
        Phi = (y * exp(-1j * q * di))[:, newaxis] * wavelet((q[:, newaxis] - di) / b)
        WT[i] = trapz(Phi, x=q, axis=0)
    WT /= sqrt(b)[newaxis, :]

    if return_d:
        return d, WT
    else:
        return WT


def derivative(x, y, N):
    """
    Take discrete derivative using N neighbors with N>0. For each additional
    point needed for the derivative one edge point derivative is set to 0.
    """
    dydx = zeros_like(y)
    # calculate central derivative for each point
    dx_center = x[2:] - x[:-2]
    dxmin = dx_center[dx_center > 0].min()
    dydx_center = (y[2:] - y[:-2]) / maximum(dxmin, dx_center)
    if N == 1:
        dydx[1:-1] = dydx_center
    else:
        # average N derivatives of neighboring points
        dydx_csum = cumsum(dydx_center)
        dydx[N:-N] = (dydx_csum[2 * N - 2 :] - dydx_csum[: (2 - 2 * N) or None]) / 2.0 / (N - 1)
        for i in range(2, N):
            dydx[i] = (dydx_csum[2 * i - 2] - dydx_csum[0]) / 2.0 / (i - 1)
            dydx[-i] = (dydx_csum[2 - 2 * i] - dydx_csum[-1]) / 2.0 / (i - 1)
        dydx[1] = dydx_center[0]
        dydx[-2] = dydx_center[-1]

    return dydx


def average(y, N=20):
    N = N + N % 2 - 1
    avg = y.copy()
    csum = cumsum(y)
    if N < len(avg):
        avg[N // 2 + 1 : (-N // 2) or None] = (csum[N:-1] - csum[: -N - 1]) / N
    for i in range(1, N // 2 + 1):
        avg[i] = csum[2 * i] / (2.0 * i + 1.0)
        avg[-i - 1] = (csum[-1] - csum[-2 * i - 2]) / (2.0 * i + 1.0)
    return avg


class TransformType(str, Enum):
    fourier_transform = "FT"
    # short_time_FT = 'STFT'
    mexican_hat_WT = "MH"
    morlet_WT = "MO"


def transform(
    q,
    I,
    Qc=0.0,
    trans_type: TransformType = TransformType.fourier_transform,
    avg_correction=None,
    avgN=10,
    logI=False,
    Q4=True,
    derivate=True,
    derivN=3,
    Qmin=0.0,
    Qmax=None,
    D=None,
    wavelet_scaling=0.5,
    return_corrected=False,
):
    """
    Transform a reflectivity dataset to frequency domain.
    Allows some corrections to be made before the actual transform as
    logarithmic/Q^4 compression of the dataset, taking the derivative
    and correcting q for the critical wave vector Qc.

    :param q: Q coordinate of the measurement
    :param I: Intensity of the measurement
    :keyword Qc: Critical wave vector for the correction
    :keyword trans_type: Kind of transform to be used, at the moment either 'FT','STFT', 'MH', 'MO'
                         for Fourier Transform/short-time FT, Mexican Hat WT, Morlet WT, respectively
    :keyword avg_correction: Type of average curve correction to be made, either 'subtract' or ''devide'
    :keyword log: Perform logarithmic compression
    :keyword Q4: Perform Q^4 compression
    :keyword derivate: Take the derivative of the data
    :keyword derivN: Number of points to average the derivative for smoothing
    :keyword Qmin/Qmax: Crop dataset to given Q range, this is applied after correction for Qc
    :keyword D: Optionally give an array of D values to calculate the transform for
    :keyword wavelet_offset: Value to be used to offset Q for the wavelet transform
    :keyword return_corrected: Also return the corrected Q and I values

    :returns: D, transform  or Q, I, D, transform when return_corrected is True
    """
    Q = sqrt(maximum(0.0, q**2 - Qc**2))
    if Qmax is None:
        Qmax = Q.max()
    # if no D-range is given derive it from Q
    if D is None:
        D = linspace(0.0, len(Q) * 2.0 * pi / (Q.max() - Q.min()))

    if Q4:
        I = Q**4 * I
    if logI:
        I = log(maximum(I, I[I > 0].min()))

    if avg_correction:
        avg = average(I, N=avgN)
        if avg_correction == "subtract":
            I -= avg
        else:
            I /= avg

    if derivate:
        Drange = Q > 0
        I = I[Drange]
        Q = Q[Drange]
        I = derivative(Q, I, derivN)

    posvals = (Q > Qmin) & (Q <= Qmax)
    Quse = Q[posvals]
    Iuse = I[posvals]

    if trans_type == TransformType.fourier_transform:
        D, T = NDFT(Quse, Iuse, d=D)
        T = abs(T)
    # elif trans_type==TransformType.short_time_FT:
    #     T = abs(STFT(Quse, Iuse, D))
    elif trans_type == TransformType.mexican_hat_WT:
        T = abs(CWT(Quse, Iuse, d=D, wavelet=MexicanHat)) ** 2.0 + 1e-20
        T = T[:, int(T.shape[1] * wavelet_scaling)]
    elif trans_type == TransformType.morlet_WT:
        T = abs(CWT(Quse, Iuse, d=D, wavelet=Morlet)) ** 2.0 + 1e-20
        T = T[:, int(T.shape[1] * wavelet_scaling)]
    else:
        raise ValueError("Transformation type not known: %s" % trans_type)

    if return_corrected:
        return Q, I, D, T
    else:
        return D, T
