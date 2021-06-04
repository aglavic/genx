"""
DWBA extension for offspecular from rough surfaces.

Rebuild of old C++ extension module.
"""
from numpy import *
from scipy import integrate
from scipy.special import factorial
from .elfield import *

import numba

@numba.jit(numba.float64[:](numba.float64[:], numba.complex128[:,:,:], numba.complex128[:,:,:], numba.float64,
                            numba.float64, numba.float64[:], numba.float64[:], numba.complex128[:], numba.float64[:],
                            numba.complex128[:], numba.float64[:], numba.float64[:], numba.float64),
            nopython=True, parallel=True, cache=True)
def dwba_interdiff_sum(qx, G, q, eta,
                       h, sigma, sigmaid, sqn, z,
                       table, q_min, fn, eta_z):
    Layers=G.shape[1]
    Points=qx.shape[0]

    s=0.0j
    I=zeros(qx.shape, dtype=float64)

    for m in numba.prange(Points):
        Itemp=0.0j
        for i in range(Layers):
            for j in range(Layers):
                for k in range(4):
                    for l in range(4):
                        nind=(k, i, m) # DWBA component, Layer, q-point for incident
                        npind=(l, j, m) # DWBA component, Layer, q-point for outgoing
                        for p in range(fn.shape[0]):
                            xnew=abs(qx[m]*eta/pow(p+1.0, 1.0/2.0/h))
                            lower=int((xnew-q_min[0])/(q_min[1]-q_min[0]))
                            s+=2.0*pow(q[nind]*conj(q[npind])*sigma[i]*sigma[j]*exp(-abs(z[i]-z[j])/eta_z), p+1.0)\
                               /fn[p]*eta/pow(p+1.0, 1.0/2.0/h)*\
                               ((table[lower+1]-table[lower])/(q_min[lower+1]-q_min[lower])*
                                (xnew-q_min[lower])+table[lower]);
                        Itemp+=(sqn[i]-sqn[i+1])*conj(sqn[j]-sqn[j+1])*\
                               G[nind]*exp(-0.5*pow(sigmaid[i]*q[nind], 2))*\
                               conj(G[npind]*exp(-0.5*pow(sigmaid[j]*q[npind], 2)))*\
                               exp(-0.5*(pow(q[nind]*sigma[i], 2)+pow(conj(q[npind])*sigma[j], 2)))\
                               /q[nind]/conj(q[npind])*s
                        s = 0
        I[m]=Itemp.real
    return I

@numba.jit(numba.float64[:](numba.float64[:], numba.complex128[:,:,:], numba.complex128[:,:,:], numba.float64,
                            numba.float64, numba.float64[:], numba.complex128[:], numba.float64[:],
                            numba.complex128[:], numba.float64[:], numba.float64[:], numba.float64),
            nopython=True, parallel=True, cache=True)
def dwba_sum(qx, G, q, eta,
             h, sigma, sqn, z,
             table, q_min, fn, eta_z):
    Layers=G.shape[1]
    Points=qx.shape[0]
    s=0.0j
    I=zeros(qx.shape, dtype=float64)

    for m in numba.prange(Points):
        Itemp=0.0j
        for i in range(Layers):
            for j in range(Layers):
                for k in range(4):
                    for l in range(4):
                        nind=(k, i, m) # DWBA component, Layer, q-point for incident
                        npind=(l, j, m) # DWBA component, Layer, q-point for outgoing
                        for p in range(fn.shape[0]):
                            xnew = fabs(qx[m]*eta/pow(p+1.0,1.0/2.0/h))
                            lower=int((xnew-q_min[0])/(q_min[1]-q_min[0]))
                            s+=2.0*pow(q[nind]*conj(q[npind])*sigma[i]*sigma[j]*exp(-abs(z[i]-z[j])/eta_z),p+1.0)\
                               /fn[p]*eta/pow(p+1.0,1.0/2.0/h)*\
                               ((table[lower+1]-table[lower])/(q_min[lower+1]-q_min[lower])*
                                (xnew-q_min[lower])+table[lower])
                        Itemp+=(sqn[i]-sqn[i+1])*conj(sqn[j]-sqn[j+1])*\
                               G[nind]*conj(G[npind])*\
                               exp(-0.5*(pow(q[nind]*sigma[i],2.0)+pow(conj(q[npind])*sigma[j],2.0)))\
                               /q[nind]/conj(q[npind])*s
                        s=0.0j
        I[m]=Itemp.real
    return I

@numba.jit(numba.float64[:](numba.float64[:], numba.float64[:], numba.float64,
                            numba.float64, numba.float64[:], numba.complex128[:], numba.float64[:],
                            numba.complex128[:], numba.float64[:], numba.float64[:], numba.float64),
            nopython=True, parallel=True, cache=True)
def born_ext(qx, qz, eta,
             h, sigma, sqn, z,
             table, q_min, fn, eta_z):
    Layers=sigma.shape[0]
    Points=qx.shape[0]
    s=0.0j
    I=zeros(qx.shape, dtype=float64)

    for m in numba.prange(Points):
        Itemp=0.0j
        for i in range(Layers):
            for j in range(Layers):
                for p in range(fn.shape[0]):
                    xnew=qx[m]*eta/pow(p+1.0, 1.0/2.0/h)
                    lower=int((xnew-q_min[0])/(q_min[1]-q_min[0]))
                    s+=2.0*pow(qz[m]*qz[m]*sigma[i]*sigma[j]*exp(-abs(z[i]-z[j])/eta_z), p+1.0)/fn[p]*eta/pow(
                        p+1.0, 1.0/2.0/h)*((table[lower+1]-table[lower])/(q_min[lower+1]-q_min[lower])*(
                                xnew-q_min[lower])+table[lower])

                Itemp+=(sqn[i]-sqn[i+1])*conj(sqn[j]-sqn[j+1])*exp(
                    -0.5*(pow(qz[m]*sigma[i], 2.0)+pow(qz[m]*sigma[j], 2.0)))/qz[m]/qz[m]*s;
                s=0.0j
        I[m]=Itemp.real
    return I


def vec_realsymint(F, omega, eta, h, eta_z, qz_n, qz_np, sigma_n, sigma_np, max_n):
    I=2*sum(
        [((qz_n*conj(qz_np)*(sigma_n*sigma_np))**n)/factorial(n)*eta/n**(1.0/2.0/h)*F(omega*eta/(n**(1.0/2.0/h))) for n
         in arange(1.0, max_n+1.0)])
    I=array(I)
    return I

def make_F(q, h):
    def f(x):
        return exp(-abs(x)**(2*h))

    q=q*(abs(q)>0.003)+0.003*(abs(q)<0.003)
    I=[integrate.quad(f, 0, inf, weight='cos', wvar=x)[0] for x in q]
    I=array(I)
    return I

def DWBA_Interdiff(qx, qz, lamda, n, z, sigma, sigmaid, eta, h, eta_z, d=[0],
                   taylor_n=1):
    # sigmaid-interdiffused part
    # sigma-roughness
    k=2*pi/lamda
    qx=array(qx, dtype=float64)
    # Calculating electrical fields
    omega=arctan(qx/qz)  # Not the real omega given by Fewster
    omegap=arcsin(sqrt(qx**2+qz**2)/2/k)
    kxi=k*cos(omega+omegap)
    kxf=k*cos(omegap-omega)

    # Calculate the electric fields
    (T, R, kz)=AmpElfield_q(k, r_[kxi, kxf], lamda, n[::-1], r_[0, d[1:][::-1]])

    lenq=len(qx)
    Ri=R[:, :lenq]
    Rf=R[:, lenq:]
    Ti=T[:, :lenq]
    Tf=T[:, lenq:]
    ki=kz[:, :lenq]
    kf=kz[:, lenq:]
    G=array([Ti*Tf, Ti*Rf, Ri*Tf, Ri*Rf], dtype=complex128)
    q=array([ki+kf, ki-kf, -ki+kf, -ki-kf], dtype=complex128)

    # Setting up for the Fourier integral as given by Pape et.al.
    maxn=taylor_n
    # New dqmin tested 20051201
    dqmin=min(abs(qx[:-1]-qx[1:]))*eta/(maxn**(1.0/2.0/h))
    if abs(dqmin)<1e-12:
        dqmin=1e-3
    # New q_min tested 20051201
    q_min=arange(min(abs(qx))*eta/(maxn**(1.0/2.0/h))-dqmin, max(abs(qx))*eta+2*dqmin, dqmin, dtype=float64)
    table=array(make_F(q_min, h), dtype=complex128)

    fn=factorial(arange(1, maxn+1))
    sqn=n**2
    sqn=array(sqn, dtype=complex128)
    sigma=array(sigma, dtype=float64)
    sigmaid=array(sigmaid, dtype=float64)
    s=dwba_interdiff_sum(qx, G, q, eta, h, sigma, sigmaid, sqn, z, table, q_min, fn, eta_z)

    return (s, omega+omegap, omegap-omega)

def DWBA(qx, qz, lamda, n, z, sigma, eta, h, eta_z, d=[0], taylor_n=1):
    k=2*pi/lamda
    qx=array(qx, dtype=float64)
    sqn=n**2
    sqn=array(sqn, dtype=complex)
    # Calculating electrical fields
    omega=arctan(qx/qz)  # Not the real omega given by Fewster
    omegap=arcsin(sqrt(qx**2+qz**2)/2/k)
    kxi=k*cos(omega+omegap)
    kxf=k*cos(omegap-omega)
    (T, R, kz)=AmpElfield_q(k, r_[kxi, kxf], lamda, n[::-1], r_[0, d[1:][::-1]])

    lenq=len(qx)
    Ri=R[:, :lenq]
    Rf=R[:, lenq:]
    Ti=T[:, :lenq]
    Tf=T[:, lenq:]
    ki=kz[:, :lenq]
    kf=kz[:, lenq:]
    G=array([Ti*Tf, Ti*Rf, Ri*Tf, Ri*Rf], dtype=complex)
    q=array([ki+kf, ki-kf, -ki+kf, -ki-kf], dtype=complex)

    # Setting up for the Fourier integral as given by Pape et.al.
    maxn=taylor_n
    dqmin=(qx[1]-qx[0])*eta/(maxn**(1.0/2.0/h))
    if abs(dqmin)<1e-12:
        dqmin=1e-3
    q_min=arange(qx[0]*eta/(maxn**(1.0/2.0/h)), qx[-1]*eta+2*dqmin, dqmin, dtype=float64)

    table=array(make_F(q_min, h), dtype=complex)
    fn=factorial(arange(1, maxn+1))
    s=dwba_sum(qx, G, q, eta, h, sigma, sqn, z, table, q_min, fn, eta_z)
    return (s, omega+omegap, omegap-omega)

def Born(qx, qz, lamda, n, z, sigma, eta, h, eta_z, d=[0], taylor_n=1):
    k=2*pi/lamda
    qx=array(qx, dtype=float64)
    sqn=n**2
    sqn=array(sqn, dtype=complex)
    # print 'Setup Complete'
    # Calculating electrical fields
    omega=arctan(qx/qz)  # Not the real omega given by Fewster
    omegap=arcsin(sqrt(qx**2+qz**2)/2/k)
    # Setting up for the Fourier integral as given by Pape et.al.
    maxn=taylor_n
    dqmin=(qx[1]-qx[0])*eta/(maxn**(1.0/2.0/h))
    q_min=arange(qx[0]*eta/(maxn**(1.0/2.0/h)), qx[-1]*eta+2*dqmin, dqmin, dtype=float64)

    table=make_F(q_min, h)
    table=array(table, dtype=complex128)

    fn=factorial(arange(1, maxn+1))
    s=born_ext(qx, qz, eta, h, sigma, sqn, z, table, q_min, fn, eta_z)

    return (s, omega+omegap, omegap-omega)

if __name__=='__main__':
    n=array([1]+[1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j]*10+[1-7.57e-6+1.73e-7j])
    # n=array([1,1-2.25e-5+2.9e-6j,1-7.5e-6+1.75e-7j])
    # n=array([1,1-2.25e-5,1-7.5e-6])
    d=array([0.0]+[80.0, 20.0]*10)
    # d=array([0.0,1000.0])
    z=-cumsum(d)
    sigma=array([5.0]+[5.0, 5.0]*10)
    # sigma=array([3.0,3.0])

    qx=arange(.1e-4, 0.005, 0.00002)
    qz=0.2*ones(qx.shape)

    # qx=arange(1e-5,3e-5,1e-5)
    eta=200.0
    eta_z=1.0
    h=1.0

    (I1, ain, aout)=Born(qx, qz, 1.540562, n, z, sigma, eta, h, eta_z)
    (I2, ain, aout)=DWBA(qx, qz, 1.540562, n, z, sigma, eta, h, eta_z, d)
    (I3, ain, aout)=DWBA_Interdiff(qx, qz, 1.540562, n, z, sigma, 0.3*sigma, eta, h, eta_z, d)

    from matplotlib import pyplot
    pyplot.plot(qx, real(I1), qx, real(I2), qx, real(I3))
    pyplot.show()
