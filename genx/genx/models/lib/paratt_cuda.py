from numpy import *
from numba import cuda
import numba
import math, cmath

if numba.__version__.split('.')>['0', '55', '0']:
    JIT_OPTIONS = dict(cache=True)
else:
    JIT_OPTIONS = {}

@cuda.jit(numba.void(numba.float64[:], numba.float64, numba.complex128[:],
                     numba.float64[:], numba.float64[:], numba.float64[:]), **JIT_OPTIONS)
def ReflNB(theta, lamda, n, d, sigma, Rout):
    # Thread id in a 1D block
    tx=cuda.threadIdx.x
    # Block id in a 1D grid
    ty=cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw=cuda.blockDim.x
    # Compute flattened index inside the array
    ai=tx+ty*bw
    if ai>=theta.shape[0]:
        return

    layers=d.shape[0]
    n0=n[-1]

    k=2.*pi/lamda

    pre1=2.*n0*k
    pre2=(pi/180.)

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
    Rout[ai]=abs(Aj)**2

@cuda.jit(numba.void(numba.float64[:], numba.float64, numba.complex128[:],
                     numba.float64[:], numba.float64[:], numba.complex128[:]), **JIT_OPTIONS)
def AmpNB(theta, lamda, n, d, sigma, Aout):
    # Thread id in a 1D block
    tx=cuda.threadIdx.x
    # Block id in a 1D grid
    ty=cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw=cuda.blockDim.x
    # Compute flattened index inside the array
    ai=tx+ty*bw
    if ai>=theta.shape[0]:
        return

    layers=d.shape[0]
    n0=n[-1]

    k=2.*pi/lamda

    pre1=2.*n0*k
    pre2=(pi/180.)

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
    Aout[ai]=Aj

def Refl(theta, lamda, n, d, sigma, return_int=True):
    threadsperblock=32
    blockspergrid=(theta.shape[0]+(threadsperblock-1))//threadsperblock

    Ctheta=cuda.to_device(theta)
    Cn=cuda.to_device(n.astype(complex128))
    Cd=cuda.to_device(d)
    Csigma=cuda.to_device(sigma)

    if return_int:
        CRout=cuda.device_array(shape=theta.shape, dtype=float64)
        ReflNB[blockspergrid, threadsperblock](Ctheta, lamda, Cn, Cd, Csigma, CRout)
        return CRout.copy_to_host()
    else:
        CAout=cuda.device_array(shape=theta.shape, dtype=complex128)
        AmpNB[blockspergrid, threadsperblock](Ctheta, lamda, Cn, Cd, Csigma, CAout)
        return CAout.copy_to_host()

@cuda.jit(numba.void(numba.float64[:], numba.float64, numba.complex128[:],
                     numba.float64[:], numba.float64[:], numba.float64[:]), **JIT_OPTIONS)
def ReflQNB(Q, Q0, n, d, sigma, Rout):
    # Thread id in a 1D block
    tx=cuda.threadIdx.x
    # Block id in a 1D grid
    ty=cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw=cuda.blockDim.x
    # Compute flattened index inside the array
    qi=tx+ty*bw
    if qi>=Q.shape[0]:
        return

    layers=d.shape[0]
    n0=n[-1]

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
    Rout[qi]=abs(Aj)**2

@cuda.jit(numba.void(numba.float64[:], numba.float64, numba.complex128[:],
                     numba.float64[:], numba.float64[:], numba.complex128[:]), **JIT_OPTIONS)
def AmpQNB(Q, Q0, n, d, sigma, Aout):
    # Thread id in a 1D block
    tx=cuda.threadIdx.x
    # Block id in a 1D grid
    ty=cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw=cuda.blockDim.x
    # Compute flattened index inside the array
    qi=tx+ty*bw
    if qi>=Q.shape[0]:
        return

    layers=d.shape[0]
    points=Q.shape[0]
    n0=n[-1]

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
    Aout[qi]=Aj

def ReflQ(Q, lamda, n, d, sigma, return_int=True):
    threadsperblock=32
    blockspergrid=(Q.shape[0]+(threadsperblock-1))//threadsperblock

    Q0=4.*pi/lamda

    CQ=cuda.to_device(Q)
    Cn=cuda.to_device(n.astype(complex128))
    Cd=cuda.to_device(d)
    Csigma=cuda.to_device(sigma)

    if return_int:
        CRout=cuda.device_array(shape=Q.shape, dtype=float64)
        ReflQNB[blockspergrid, threadsperblock](CQ, Q0, Cn, Cd, Csigma, CRout)
        return CRout.copy_to_host()
    else:
        CAout=cuda.device_array(shape=Q.shape, dtype=complex128)
        AmpQNB[blockspergrid, threadsperblock](CQ, Q0, Cn, Cd, Csigma, CAout)
        return CAout.copy_to_host()

@cuda.jit(numba.void(numba.float64[:], numba.float64[:], numba.complex128[:, :],
                     numba.float64[:], numba.float64[:], numba.float64[:]), **JIT_OPTIONS)
def Refl_nvary2NB(theta, lamda, n, d, sigma, Rout):
    # Thread id in a 1D block
    tx=cuda.threadIdx.x
    # Block id in a 1D grid
    ty=cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw=cuda.blockDim.x
    # Compute flattened index inside the array
    ai=tx+ty*bw
    if ai>=theta.shape[0]:
        return

    layers=d.shape[0]

    pre2=(pi/180.)

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
    Rout[ai]=abs(Aj)**2

@cuda.jit(numba.void(numba.float64[:], numba.float64[:], numba.complex128[:, :],
                     numba.float64[:], numba.float64[:], numba.complex128[:]), **JIT_OPTIONS)
def Amp_nvary2NB(theta, lamda, n, d, sigma, Aout):
    # Thread id in a 1D block
    tx=cuda.threadIdx.x
    # Block id in a 1D grid
    ty=cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw=cuda.blockDim.x
    # Compute flattened index inside the array
    ai=tx+ty*bw
    if ai>=theta.shape[0]:
        return

    layers=d.shape[0]

    pre2=(pi/180.)

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
    Aout[ai]=Aj

def Refl_nvary2(theta, lamda, n, d, sigma, return_int=True):
    threadsperblock=32
    blockspergrid=(theta.shape[0]+(threadsperblock-1))//threadsperblock

    Ctheta=cuda.to_device(theta)
    Clamda=cuda.to_device(lamda)
    Cn=cuda.to_device(n.astype(complex128))
    Cd=cuda.to_device(d)
    Csigma=cuda.to_device(sigma)

    if return_int:
        CRout=cuda.device_array(shape=theta.shape, dtype=float64)
        Refl_nvary2NB[blockspergrid, threadsperblock](Ctheta, Clamda, Cn, Cd, Csigma, CRout)
        return CRout.copy_to_host()
    else:
        CAout=cuda.device_array(shape=theta.shape, dtype=complex128)
        Amp_nvary2NB[blockspergrid, threadsperblock](Ctheta, Clamda, Cn, Cd, Csigma, CAout)
        return CAout.copy_to_host()
