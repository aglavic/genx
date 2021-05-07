from numpy import *
import numba
import math, cmath

@numba.jit(numba.float64[:](numba.float64[:], numba.float64, numba.complex128[:], numba.float64[:], numba.float64[:]),
           nopython=True, parallel=True, cache=True)
def ReflNB(theta, lamda, n, d, sigma):
    layers=d.shape[0]
    angles=theta.shape[0]
    n0=n[-1]

    k=2.*pi/lamda
    R=empty_like(theta)

    pre1=2.*n0*k
    pre2=(pi/180.)

    for ai in numba.prange(angles):
        Qi=pre1*cmath.sqrt((n[0]/n0)**2-math.cos(theta[ai]*pre2)**2)

        Qj=pre1*cmath.sqrt((n[1]/n0)**2-math.cos(theta[ai]*pre2)**2)
        # Fresnel reflectivity for the interfaces
        rpj=(Qj-Qi)/(Qj+Qi)*cmath.exp(-Qj*Qi/2.*sigma[0]**2)
        Aj=rpj
        Qi=Qj

        for lj in range(2, layers):
            Qj=pre1*cmath.sqrt((n[lj]/n0)**2-math.cos(theta[ai]*pre2)**2)
            # Fresnel reflectivity for the interfaces
            rpj=(Qj-Qi)/(Qj+Qi)*cmath.exp(-Qj*Qi/2.*sigma[lj-1]**2)

            pj=cmath.exp(1.0j*d[lj-1]*Qi)
            part=Aj*pj
            Aj=(rpj+part)/(1.+part*rpj)

            Qi=Qj
        R[ai]=abs(Aj)**2
    return R

@numba.jit(
    numba.complex128[:](numba.float64[:], numba.float64, numba.complex128[:], numba.float64[:], numba.float64[:]),
    nopython=True, parallel=True, cache=True)
def AmpNB(theta, lamda, n, d, sigma):
    layers=d.shape[0]
    angles=theta.shape[0]
    n0=n[-1]

    k=2.*pi/lamda
    A=empty(theta.shape, dtype=complex128)

    pre1=2.*n0*k
    pre2=(pi/180.)

    for ai in numba.prange(angles):
        Qi=pre1*cmath.sqrt((n[0]/n0)**2-math.cos(theta[ai]*pre2)**2)

        Qj=pre1*cmath.sqrt((n[1]/n0)**2-math.cos(theta[ai]*pre2)**2)
        # Fresnel reflectivity for the interfaces
        rpj=(Qj-Qi)/(Qj+Qi)*cmath.exp(-Qj*Qi/2.*sigma[0]**2)
        Aj=rpj
        Qi=Qj

        for lj in range(2, layers):
            Qj=pre1*cmath.sqrt((n[lj]/n0)**2-math.cos(theta[ai]*pre2)**2)
            # Fresnel reflectivity for the interfaces
            rpj=(Qj-Qi)/(Qj+Qi)*cmath.exp(-Qj*Qi/2.*sigma[lj-1]**2)

            pj=cmath.exp(1.0j*d[lj-1]*Qi)
            part=Aj*pj
            Aj=(rpj+part)/(1.+part*rpj)

            Qi=Qj
        A[ai]=Aj
    return A

def Refl(theta, lamda, n, d, sigma, return_int=True):
    if return_int:
        return ReflNB(theta, lamda, n, d, sigma)
    else:
        return AmpNB(theta, lamda, n, d, sigma)

@numba.jit(numba.float64[:](numba.float64[:], numba.float64, numba.complex128[:], numba.float64[:], numba.float64[:]),
           nopython=True, parallel=True, cache=True)
def ReflQNB(Q, lamda, n, d, sigma):
    layers=d.shape[0]
    points=Q.shape[0]
    n0=n[-1]

    Q0=4.*pi/lamda
    R=empty_like(Q)

    for qi in numba.prange(points):
        Qi=cmath.sqrt((n[0]**2-n0**2)*Q0**2+n0**2*Q[qi]**2)

        Qj=cmath.sqrt((n[1]**2-n0**2)*Q0**2+n0**2*Q[qi]**2)
        # Fresnel reflectivity for the interfaces
        rpj=(Qj-Qi)/(Qj+Qi)*cmath.exp(-Qj*Qi/2.*sigma[0]**2)
        Aj=rpj
        Qi=Qj

        for lj in range(2, layers):
            Qj=cmath.sqrt((n[lj]**2-n0**2)*Q0**2+n0**2*Q[qi]**2)
            # Fresnel reflectivity for the interfaces
            rpj=(Qj-Qi)/(Qj+Qi)*cmath.exp(-Qj*Qi/2.*sigma[lj-1]**2)

            pj=cmath.exp(1.0j*d[lj-1]*Qi)
            part=Aj*pj
            Aj=(rpj+part)/(1.+part*rpj)

            Qi=Qj
        R[qi]=abs(Aj)**2
    return R

@numba.jit(
    numba.complex128[:](numba.float64[:], numba.float64, numba.complex128[:], numba.float64[:], numba.float64[:]),
    nopython=True, parallel=True, cache=True)
def AmpQNB(Q, lamda, n, d, sigma):
    layers=d.shape[0]
    points=Q.shape[0]
    n0=n[-1]

    Q0=4.*pi/lamda
    A=empty(Q.shape, dtype=complex128)

    for qi in numba.prange(points):
        Qi=cmath.sqrt((n[0]**2-n0**2)*Q0**2+n0**2*Q[qi]**2)

        Qj=cmath.sqrt((n[1]**2-n0**2)*Q0**2+n0**2*Q[qi]**2)
        # Fresnel reflectivity for the interfaces
        rpj=(Qj-Qi)/(Qj+Qi)*cmath.exp(-Qj*Qi/2.*sigma[0]**2)
        Aj=rpj
        Qi=Qj

        for lj in range(2, layers):
            Qj=cmath.sqrt((n[lj]**2-n0**2)*Q0**2+n0**2*Q[qi]**2)
            # Fresnel reflectivity for the interfaces
            rpj=(Qj-Qi)/(Qj+Qi)*cmath.exp(-Qj*Qi/2.*sigma[lj-1]**2)

            pj=cmath.exp(1.0j*d[lj-1]*Qi)
            part=Aj*pj
            Aj=(rpj+part)/(1.+part*rpj)

            Qi=Qj
        A[qi]=Aj
    return A

def ReflQ(Q, lamda, n, d, sigma, return_int=True):
    if return_int:
        return ReflQNB(Q, lamda, n, d, sigma)
    else:
        return AmpQNB(Q, lamda, n, d, sigma)

@numba.jit(
    numba.float64[:](numba.float64[:], numba.float64[:], numba.complex128[:, :], numba.float64[:], numba.float64[:]),
    nopython=True, parallel=True, cache=True)
def Refl_nvary2NB(theta, lamda, n, d, sigma):
    layers=d.shape[0]
    angles=theta.shape[0]

    R=empty_like(theta)

    pre2=(pi/180.)

    for ai in numba.prange(angles):
        n0=n[-1, ai]
        ki=2.*pi/lamda[ai]
        pre1=2.*n0*ki

        Qi=pre1*cmath.sqrt((n[0, ai]/n0)**2-math.cos(theta[ai]*pre2)**2)

        Qj=pre1*cmath.sqrt((n[1, ai]/n0)**2-math.cos(theta[ai]*pre2)**2)
        # Fresnel reflectivity for the interfaces
        rpj=(Qj-Qi)/(Qj+Qi)*cmath.exp(-Qj*Qi/2.*sigma[0]**2)
        Aj=rpj
        Qi=Qj

        for lj in range(2, layers):
            Qj=pre1*cmath.sqrt((n[lj, ai]/n0)**2-math.cos(theta[ai]*pre2)**2)
            # Fresnel reflectivity for the interfaces
            rpj=(Qj-Qi)/(Qj+Qi)*cmath.exp(-Qj*Qi/2.*sigma[lj-1]**2)

            pj=cmath.exp(1.0j*d[lj-1]*Qi)
            part=Aj*pj
            Aj=(rpj+part)/(1.+part*rpj)

            Qi=Qj
        R[ai]=abs(Aj)**2
    return R

@numba.jit(
    numba.complex128[:](numba.float64[:], numba.float64[:], numba.complex128[:, :], numba.float64[:], numba.float64[:]),
    nopython=True, parallel=True, cache=True)
def Amp_nvary2NB(theta, lamda, n, d, sigma):
    layers=d.shape[0]
    angles=theta.shape[0]

    A=empty(theta.shape, dtype=complex128)

    pre2=(pi/180.)

    for ai in numba.prange(angles):
        n0=n[-1, ai]
        ki=2.*pi/lamda[ai]
        pre1=2.*n0*ki

        Qi=pre1*cmath.sqrt((n[0, ai]/n0)**2-math.cos(theta[ai]*pre2)**2)

        Qj=pre1*cmath.sqrt((n[1, ai]/n0)**2-math.cos(theta[ai]*pre2)**2)
        # Fresnel reflectivity for the interfaces
        rpj=(Qj-Qi)/(Qj+Qi)*cmath.exp(-Qj*Qi/2.*sigma[0]**2)
        Aj=rpj
        Qi=Qj

        for lj in range(2, layers):
            Qj=pre1*cmath.sqrt((n[lj, ai]/n0)**2-math.cos(theta[ai]*pre2)**2)
            # Fresnel reflectivity for the interfaces
            rpj=(Qj-Qi)/(Qj+Qi)*cmath.exp(-Qj*Qi/2.*sigma[lj-1]**2)

            pj=cmath.exp(1.0j*d[lj-1]*Qi)
            part=Aj*pj
            Aj=(rpj+part)/(1.+part*rpj)

            Qi=Qj
        A[ai]=Aj
    return A

def Refl_nvary2(theta, lamda, n, d, sigma, return_int=True):
    if return_int:
        return Refl_nvary2NB(theta, lamda, n, d, sigma)
    else:
        return Amp_nvary2NB(theta, lamda, n, d, sigma)
