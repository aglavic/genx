"""
DWBA extension for offspecular from rough surfaces.

Rebuild of old C++ extension module.
"""

import numba

from numpy import arcsin, arctan, complex128, conj, cumsum, fabs, float64, inf, real, zeros, linspace, log, hstack, searchsorted
from scipy import integrate, LowLevelCallable
from scipy.special import factorial
from functools import lru_cache

from .elfield import *


@numba.jit(
    numba.float64[:](
        numba.float64[:],
        numba.complex128[:, :, :],
        numba.complex128[:, :, :],
        numba.float64,
        numba.float64,
        numba.float64[:],
        numba.float64[:],
        numba.complex128[:],
        numba.float64[:],
        numba.complex128[:],
        numba.float64[:],
        numba.float64[:],
        numba.float64,
    ),
    nopython=True,
    parallel=True,
    cache=True,
)
def dwba_interdiff_sum(qx, G, q, eta, h, sigma, sigmaid, sqn, z, table, q_min, fn, eta_z):
    Layers = G.shape[1]
    Points = qx.shape[0]

    s = 0.0j
    I = zeros(qx.shape, dtype=float64)

    half_o_h =1.0 / 2.0 / h

    for m in numba.prange(Points):
        Itemp = 0.0j
        qx_eta = qx[m] * eta
        # F(q) and scaled qx values are the same for all terms in the inner loop
        F = []
        for n in range(1, fn.shape[0]+1):
            # calculate the x value needed in the tabulated function F(q)
            xnew = abs(qx_eta/(n**half_o_h))
            # index of the q value within the table
            lower = searchsorted(q_min,xnew, side='right')-1 # works on non-uniform q_min array
            #lower = int((xnew-q_min[0])/(q_min[1]-q_min[0]))
            F.append(( # linea interpolate between the F(q) points closest to xnew
                                    (table[lower + 1] - table[lower])
                                    / (q_min[lower + 1] - q_min[lower])
                                    * (xnew - q_min[lower])
                                    + table[lower]
                                ))
        for i in range(Layers):
            for j in range(Layers):
                # roughness times vertical cross-correlation term, the same for all terms in the inner loop
                sigma_exp_zizj =  sigma[i] * sigma[j] * exp(-abs(z[i] - z[j]) / eta_z)
                for k in range(4):
                    for l in range(4):
                        nind = (k, i, m)  # DWBA component, Layer, q-point for incident
                        npind = (l, j, m)  # DWBA component, Layer, q-point for outgoing
                        # This loop corresponds to the Fourier transform series expansion given in the Pape paper
                        # as equation (11) and pasted into equation (10).
                        # There are some discrepancies like the power of n, not 2n for q q* (could be mistake in paper)
                        # and using two sigmas like in equation (9).
                        for n in range(1, fn.shape[0]+1):
                            s += (
                                pow(
                                    q[nind] * conj(q[npind]) * sigma_exp_zizj,
                                    n,
                                )
                                / fn[n-1] # n!
                                * eta
                                / (n**half_o_h) # this factor also is not in the paper
                                * F[n-1]
                            )
                        s *= 2.0
                        Itemp += (
                            # Pl,i.j , here the integral is somehow replaeced by a simple gaussian
                            (sqn[i] - sqn[i + 1])
                            * G[nind]
                            * exp(-0.5 * pow(sigmaid[i] * q[nind], 2))
                            # P*l',i',j'
                            * conj(sqn[j] - sqn[j + 1])
                            * conj(G[npind] * exp(-0.5 * pow(sigmaid[j] * q[npind], 2)))
                            # second term in equation (10)
                            * exp(-0.5 * (pow(q[nind] * sigma[i], 2) + pow(conj(q[npind]) * sigma[j], 2)))
                            / q[nind]
                            / conj(q[npind])
                            # third term in equation (10) replaced by (11) above
                            * s
                        )
                        s = 0.0j
        I[m] = Itemp.real
    return I


@numba.jit(
    numba.float64[:](
        numba.float64[:],
        numba.complex128[:, :, :],
        numba.complex128[:, :, :],
        numba.float64,
        numba.float64,
        numba.float64[:],
        numba.complex128[:],
        numba.float64[:],
        numba.complex128[:],
        numba.float64[:],
        numba.float64[:],
        numba.float64,
    ),
    nopython=True,
    parallel=True,
    cache=True,
)
def dwba_sum(qx, G, q, eta, h, sigma, sqn, z, table, q_min, fn, eta_z):
    Layers = G.shape[1]
    Points = qx.shape[0]
    s = 0.0j
    I = zeros(qx.shape, dtype=float64)

    for m in numba.prange(Points):
        Itemp = 0.0j
        for i in range(Layers):
            for j in range(Layers):
                for k in range(4):
                    for l in range(4):
                        nind = (k, i, m)  # DWBA component, Layer, q-point for incident
                        npind = (l, j, m)  # DWBA component, Layer, q-point for outgoing
                        for p in range(fn.shape[0]):
                            xnew = fabs(qx[m] * eta / pow(p + 1.0, 1.0 / 2.0 / h))
                            lower = searchsorted(q_min, xnew, side='right')-1  # works on non-uniform q_min array
                            #lower = int((xnew - q_min[0]) / (q_min[1] - q_min[0]))
                            s += (
                                2.0
                                * pow(
                                    q[nind] * conj(q[npind]) * sigma[i] * sigma[j] * exp(-abs(z[i] - z[j]) / eta_z),
                                    p + 1.0,
                                )
                                / fn[p]
                                * eta
                                / pow(p + 1.0, 1.0 / 2.0 / h)
                                * (
                                    (table[lower + 1] - table[lower])
                                    / (q_min[lower + 1] - q_min[lower])
                                    * (xnew - q_min[lower])
                                    + table[lower]
                                )
                            )
                        Itemp += (
                            (sqn[i] - sqn[i + 1])
                            * conj(sqn[j] - sqn[j + 1])
                            * G[nind]
                            * conj(G[npind])
                            * exp(-0.5 * (pow(q[nind] * sigma[i], 2.0) + pow(conj(q[npind]) * sigma[j], 2.0)))
                            / q[nind]
                            / conj(q[npind])
                            * s
                        )
                        s = 0.0j
        I[m] = Itemp.real
    return I


@numba.jit(
    numba.float64[:](
        numba.float64[:],
        numba.float64[:],
        numba.float64,
        numba.float64,
        numba.float64[:],
        numba.complex128[:],
        numba.float64[:],
        numba.complex128[:],
        numba.float64[:],
        numba.float64[:],
        numba.float64,
    ),
    nopython=True,
    parallel=True,
    cache=True,
)
def born_ext(qx, qz, eta, h, sigma, sqn, z, table, q_min, fn, eta_z):
    Layers = sigma.shape[0]
    Points = qx.shape[0]
    s = 0.0j
    I = zeros(qx.shape, dtype=float64)

    for m in numba.prange(Points):
        Itemp = 0.0j
        for i in range(Layers):
            for j in range(Layers):
                for p in range(fn.shape[0]):
                    xnew = qx[m] * eta / pow(p + 1.0, 1.0 / 2.0 / h)
                    lower = int((xnew - q_min[0]) / (q_min[1] - q_min[0]))
                    s += (
                        2.0
                        * pow(qz[m] * qz[m] * sigma[i] * sigma[j] * exp(-abs(z[i] - z[j]) / eta_z), p + 1.0)
                        / fn[p]
                        * eta
                        / pow(p + 1.0, 1.0 / 2.0 / h)
                        * (
                            (table[lower + 1] - table[lower])
                            / (q_min[lower + 1] - q_min[lower])
                            * (xnew - q_min[lower])
                            + table[lower]
                        )
                    )

                Itemp += (
                    (sqn[i] - sqn[i + 1])
                    * conj(sqn[j] - sqn[j + 1])
                    * exp(-0.5 * (pow(qz[m] * sigma[i], 2.0) + pow(qz[m] * sigma[j], 2.0)))
                    / qz[m]
                    / qz[m]
                    * s
                )
                s = 0.0j
        I[m] = Itemp.real
    return I


def vec_realsymint(F, omega, eta, h, eta_z, qz_n, qz_np, sigma_n, sigma_np, max_n):
    I = 2 * sum(
        [
            ((qz_n * conj(qz_np) * (sigma_n * sigma_np)) ** n)
            / factorial(n)
            * eta
            / n ** (1.0 / 2.0 / h)
            * F(omega * eta / (n ** (1.0 / 2.0 / h)))
            for n in arange(1.0, max_n + 1.0)
        ]
    )
    I = array(I)
    return I


def make_F(qstart, qend, qstep, h):
    """
    Calculate the tabulated Fourier integral (11) from Pape1998.
    """
    q_min = arange(qstart, qend, qstep, dtype=float64)

    @numba.cfunc('float64(float64)')
    def f(x):
        return exp(-abs(x) ** (2 * h))
    llc = LowLevelCallable(f.ctypes)

    q = q_min * (abs(q_min) > 0.003) + 0.003 * (abs(q_min) < 0.003)
    I = [integrate.quad(llc, 0, inf, weight="cos", wvar=x)[0] for x in q]
    I = array(I, dtype=complex128)
    return I, q_min

q_min_fixed = None
q_eta_min = 1e-3
q_eta_max = 5000. # 0.01/Å * 50µm

@lru_cache(maxsize=16)
def make_F_fixed(h):
    """
    Calculate the tabulated Fourier integral (11) from Pape1998. The
    values only depend on the q*eta range and Hurst parameter, thus this
    version uses a fixed q_min array and caches results in case hurst isn't fitted.
    """
    global q_min_fixed
    if q_min_fixed is None:
        # generate a q_min that is non-evenly spaced to keep length in manageable size
        q_min_fixed = hstack([-q_eta_min*2, -q_eta_min, 0., exp(linspace(log(q_eta_min), log(q_eta_max), 5000))], dtype=float64)
    q_min = q_min_fixed

    @numba.cfunc('float64(float64)')
    def f(x):
        return exp(-abs(x) ** (2 * h))
    llc = LowLevelCallable(f.ctypes)

    q = q_min * (abs(q_min) > 0.003) + 0.003 * (abs(q_min) < 0.003)
    I = [integrate.quad(llc, 0, inf, weight="cos", wvar=x)[0] for x in q]
    I = array(I, dtype=complex128)
    return I, q_min


def DWBA_Interdiff(qx, qz, lamda, n, z, sigma, sigmaid, eta, h, eta_z,
                   d=[0], taylor_n=1, use_fixed_q=False):
    # sigmaid-interdiffused part
    # sigma-roughness
    k = 2 * pi / lamda
    qx = array(qx, dtype=float64)
    # Calculating electrical fields
    omega = arctan(qx / qz)  # Not the real omega given by Fewster
    omegap = arcsin(sqrt(qx**2 + qz**2) / 2 / k)
    kxi = k * cos(omega + omegap)
    kxf = k * cos(omegap - omega)

    # Calculate the electric fields
    (T, R, kz) = AmpElfield_q(k, r_[kxi, kxf], lamda, n[::-1], r_[0, d[1:][::-1]])

    lenq = len(qx)
    Ri = R[:, :lenq]
    Rf = R[:, lenq:]
    Ti = T[:, :lenq]
    Tf = T[:, lenq:]
    ki = kz[:, :lenq]
    kf = kz[:, lenq:]
    G = array([Ti * Tf, Ti * Rf, Ri * Tf, Ri * Rf], dtype=complex128)
    q = array([ki + kf, ki - kf, -ki + kf, -ki - kf], dtype=complex128)

    # Setting up for the Fourier integral as given by Pape et.al. (DOI:10.1016/s0921-4526(98)00395-0)
    maxn = taylor_n
    dqmin = min(abs(qx[:-1] - qx[1:])) * eta / (maxn ** (1.0 / 2.0 / h))
    if abs(dqmin) < 1e-12:
        dqmin = 1e-3

    if use_fixed_q:
        global q_eta_max, q_min_fixed
        if (max(abs(qx)) * eta + 2 * dqmin) > q_eta_max:
            # model parameters are outside of fixed range, expand the range
            q_eta_max = 2.0* (max(abs(qx)) * eta + 2 * dqmin)
            q_min_fixed = None
            make_F_fixed.cache_clear()
        table, q_min = make_F_fixed(h)
    else:
        table, q_min = make_F(min(abs(qx)) * eta / (maxn ** (1.0 / 2.0 / h)) - dqmin,
                          max(abs(qx)) * eta + 2 * dqmin, dqmin, h)

    fn = factorial(arange(1, maxn + 1))
    sqn = n**2
    sqn = array(sqn, dtype=complex128)
    sigma = array(sigma, dtype=float64)
    sigmaid = array(sigmaid, dtype=float64)
    s = dwba_interdiff_sum(qx, G, q, eta, h, sigma, sigmaid, sqn, z, table, q_min, fn, eta_z)
    # I was missing a k³/8/pi factor from equation (10) in dwba_interdiff_sum
    # the presence makes sense to me as the formula includes three terms with a unit, 1/q² and eta
    # which combine to a unit of length³.
    # another factor present in both Pape1998 and Wormington1996 is 1/sin(omega)

    return (s*k**3/8/pi/sin((omegap-omega)), omega + omegap, omegap - omega)


def DWBA(qx, qz, lamda, n, z, sigma, eta, h, eta_z, d=[0], taylor_n=1):
    k = 2 * pi / lamda
    qx = array(qx, dtype=float64)
    sqn = n**2
    sqn = array(sqn, dtype=complex)
    # Calculating electrical fields
    omega = arctan(qx / qz)  # Not the real omega given by Fewster
    omegap = arcsin(sqrt(qx**2 + qz**2) / 2 / k)
    kxi = k * cos(omega + omegap)
    kxf = k * cos(omegap - omega)
    (T, R, kz) = AmpElfield_q(k, r_[kxi, kxf], lamda, n[::-1], r_[0, d[1:][::-1]])

    lenq = len(qx)
    Ri = R[:, :lenq]
    Rf = R[:, lenq:]
    Ti = T[:, :lenq]
    Tf = T[:, lenq:]
    ki = kz[:, :lenq]
    kf = kz[:, lenq:]
    G = array([Ti * Tf, Ti * Rf, Ri * Tf, Ri * Rf], dtype=complex)
    q = array([ki + kf, ki - kf, -ki + kf, -ki - kf], dtype=complex)

    # Setting up for the Fourier integral as given by Pape et.al.
    maxn = taylor_n
    dqmin = (qx[1] - qx[0]) * eta / (maxn ** (1.0 / 2.0 / h))
    if abs(dqmin) < 1e-12:
        dqmin = 1e-3
    table, q_min = make_F(min(abs(qx)) * eta / (maxn ** (1.0 / 2.0 / h)) - dqmin,
                          max(abs(qx)) * eta + 2 * dqmin, dqmin, h)

    fn = factorial(arange(1, maxn + 1))
    s = dwba_sum(qx, G, q, eta, h, sigma, sqn, z, table, q_min, fn, eta_z)
    return (s, omega + omegap, omegap - omega)


def Born(qx, qz, lamda, n, z, sigma, eta, h, eta_z, d=[0], taylor_n=1):
    k = 2 * pi / lamda
    qx = array(qx, dtype=float64)
    sqn = n**2
    sqn = array(sqn, dtype=complex)
    # print 'Setup Complete'
    # Calculating electrical fields
    omega = arctan(qx / qz)  # Not the real omega given by Fewster
    omegap = arcsin(sqrt(qx**2 + qz**2) / 2 / k)
    # Setting up for the Fourier integral as given by Pape et.al.
    maxn = taylor_n
    dqmin = (qx[1] - qx[0]) * eta / (maxn ** (1.0 / 2.0 / h))
    q_min = arange(qx[0] * eta / (maxn ** (1.0 / 2.0 / h)), qx[-1] * eta + 2 * dqmin, dqmin, dtype=float64)

    table = make_F(q_min, h)
    table = array(table, dtype=complex128)

    fn = factorial(arange(1, maxn + 1))
    s = born_ext(qx, qz, eta, h, sigma, sqn, z, table, q_min, fn, eta_z)

    return (s, omega + omegap, omegap - omega)


if __name__ == "__main__":
    n = array([1] + [1 - 7.57e-6 + 1.73e-7j, 1 - 2.24e-5 + 2.89e-6j] * 10 + [1 - 7.57e-6 + 1.73e-7j])
    # n=array([1,1-2.25e-5+2.9e-6j,1-7.5e-6+1.75e-7j])
    # n=array([1,1-2.25e-5,1-7.5e-6])
    d = array([0.0] + [80.0, 20.0] * 10)
    # d=array([0.0,1000.0])
    z = -cumsum(d)
    sigma = array([5.0] + [5.0, 5.0] * 10)
    # sigma=array([3.0,3.0])

    qx = arange(0.1e-4, 0.005, 0.00002)
    qz = 0.2 * ones(qx.shape)

    # qx=arange(1e-5,3e-5,1e-5)
    eta = 200.0
    eta_z = 1.0
    h = 1.0

    (I1, ain, aout) = Born(qx, qz, 1.540562, n, z, sigma, eta, h, eta_z)
    (I2, ain, aout) = DWBA(qx, qz, 1.540562, n, z, sigma, eta, h, eta_z, d)
    (I3, ain, aout) = DWBA_Interdiff(qx, qz, 1.540562, n, z, sigma, 0.3 * sigma, eta, h, eta_z, d)

    from matplotlib import pyplot

    pyplot.plot(qx, real(I1), qx, real(I2), qx, real(I3))
    pyplot.show()
