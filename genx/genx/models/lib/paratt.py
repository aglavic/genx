from numpy import *
from functools import reduce
from genx.core.custom_logging import iprint

# "Ordinary" implementaion of Parrats recursion formula
# theta-vector, lamda- can be a vector,n-1Dvector, d-1Dvector, sigma-1Dvector
def Refl(theta, lamda, n, d, sigma, return_int=True):
    d=d[1:-1]
    sigma=sigma[:-1]
    # Length of k-vector in vaccum
    k=2*math.pi/lamda
    # Calculates the wavevector in each layer
    Qj=2*n[-1]*k*sqrt(n[:, newaxis]**2/n[-1]**2-cos(theta*math.pi/180)**2)
    # Fresnel reflectivity for the interfaces
    rp=(Qj[1:]-Qj[:-1])/(Qj[1:]+Qj[:-1])*exp(-Qj[1:]*Qj[:-1]/2*sigma[:, newaxis]**2)
    # Ignoring the top and bottom layer for the calc.
    p=exp(1.0j*d[:, newaxis]*Qj[1:-1])
    # Setting up a matrix for the reduce function. Reduce only takes one array
    # as argument
    rpp=array(list(map(lambda x, y: [x, y], rp[1:], p)))

    # Paratt's recursion formula
    def formula(rtot, rint):
        return (rint[0]+rtot*rint[1])/(1+rtot*rint[0]*rint[1])

    # Implement the recursion formula
    r=reduce(formula, rpp, rp[0])

    if return_int:
        return abs(r)**2
    else:
        return r

# "Ordinary" implementaion of Parrats recursion formula
# Q-vector,n-1Dvector, d-1Dvector, sigma-1Dvector
def ReflQ(Q, lamda, n, d, sigma, return_int=True):
    # Length of k-vector in vaccum
    d=d[1:-1]
    sigma=sigma[:-1]
    Q0=4*pi/lamda
    Q=Q.astype(complex128)
    # Calculates the wavevector in each layer
    Qj=sqrt((n[:, newaxis]**2-n[-1]**2)*Q0**2+n[-1]**2*Q**2)
    # Fresnel reflectivity for the interfaces
    rp=(Qj[1:]-Qj[:-1])/(Qj[1:]+Qj[:-1])*exp(-Qj[1:]*Qj[:-1]/2*sigma[:, newaxis]**2)
    # print rp.shape #For debugging
    # print d.shape
    # print Qj[1:-1].shape
    p=exp(1.0j*d[:, newaxis]*Qj[1:-1])  # Ignoring the top and bottom layer for the calc.
    # print p.shape #For debugging
    # Setting up a matrix for the reduce function. Reduce only takes one array
    # as argument
    rpp=array(list(map(lambda x, y: [x, y], rp[1:], p)))

    # print rpp.shape
    # Paratt's recursion formula
    def formula(rtot, rint):
        return (rint[0]+rtot*rint[1])/(1+rtot*rint[0]*rint[1])

    # Implement the recursion formula
    r=reduce(formula, rpp, rp[0])
    # print r.shape
    # return the reflectivity 
    if return_int:
        return abs(r)**2
    else:
        return r

# paratts algorithm for n as function of lamda or theta
def Refl_nvary2(theta, lamda, n_vector, d, sigma, return_int=True):
    d=d[1:-1]
    sigma=sigma[:-1]
    # Length of k-vector in vaccum
    k=2*math.pi/lamda
    # Calculates the wavevector in each layer
    # print n_func
    # ss=transpose((sin(theta[:,newaxis]*pi/180.0)/lamda)*ones(len(n_func)))
    # print ss.shape
    # print theta.shape
    # print len(n_func)
    # n=array(map(lambda f,val:f(val),n_func,ss))
    n=n_vector
    # print n
    Qj=2*n[-1]*k*sqrt(n**2/n[-1]**2-cos(theta*math.pi/180)**2)
    # print sigma.shape, Qj.shape
    # Fresnel reflectivity for the interfaces
    rp=(Qj[1:]-Qj[:-1])/(Qj[1:]+Qj[:-1])*exp(-Qj[1:]*Qj[:-1]/2*sigma[:, newaxis]**2)
    # print rp.shape #For debugging
    # print d.shape
    # print Qj[1:-1].shape
    p=exp(1.0j*d[:, newaxis]*Qj[1:-1])  # Ignoring the top and bottom layer for the calc.
    # print p.shape #For debugging
    # Setting up a matrix for the reduce function. Reduce only takes one array
    # as argument
    rpp=array(list(map(lambda x, y: [x, y], rp[1:], p)))

    # print rpp.shape
    # Paratt's recursion formula
    def formula(rtot, rint):
        return (rint[0]+rtot*rint[1])/(1+rtot*rint[0]*rint[1])

    # Implement the recursion formula
    r=reduce(formula, rpp, rp[0])
    if return_int:
        return abs(r)**2
    else:
        return r

def reflq_kin(q, lamda, n, d, sigma, correct_q=True, return_int=True):
    """Calculates the reflectivity in the kinematical approximation"""
    d=d[:-1]
    d[0]=0
    z=d.sum()-d.cumsum()
    sigma=sigma[:-1]
    q0=4*pi/lamda
    # Kinematical reflectivity for the interfaces
    iprint(n.shape, len(n.shape))
    if len(n.shape)==1:
        if correct_q:
            # The internal wave vector calacuted with the thickness averaged refractive index.
            n_mean=(n[1:-1]*d[1:]/d.sum()).sum()
            q_corr=sqrt((n_mean**2-n[-1]**2)*q0**2+(n[-1]*q)**2)
        else:
            q_corr=q
        rp=(n[:-1]-n[1:])[:, newaxis]*exp(-(q_corr*sigma[:, newaxis])**2/2)
    else:
        if correct_q:
            # The internal wave vector calacuted with the thickness averaged refractive index.
            n_mean=(n[1:-1]*d[1:][:, newaxis]/d.sum()).sum(axis=0)
            q_corr=sqrt((n_mean**2-n[-1]**2)*q0**2+(n[-1]*q)**2)
        else:
            q_corr=q
        rp=(n[:-1]-n[1:])*exp(-(q_corr*sigma[:, newaxis])**2/2)
    p=exp(1.0j*z[:, newaxis]*q_corr)

    r=(rp*p).sum(axis=0)*q0**2/q_corr**2/2.

    if return_int:
        return abs(r)**2
    else:
        return r

def reflq_pseudo_kin(q, lamda, n, d, sigma, return_int=True):
    """Calculates the reflectivity in a pseudo kinematical approximation.
    The mean refractive index of the film is simulated with the single reflection approximation and the deviation from
    the mean is simulated with the kinematical approximation.
    """
    d=d[:-1]
    d[0]=0
    z=d.sum()-d.cumsum()
    sigma=sigma[:-1]
    q0=4*pi/lamda
    # Q = Q.astype(complex128)
    # The internal wave vector calacuted with the thickness averaged refractive index.
    n_mean=(n[1:-1]*d[1:]/d.sum()).sum()
    q_corr=sqrt((n_mean**2-n[-1]**2)*q0**2+n[-1]**2*q**2)
    q_sub=sqrt((n[0]**2-n[-1]**2)*q0**2+n[-1]**2*q**2)
    q_amb=n[-1]*q
    # Top interface
    rp_top=(q_corr-q_amb)/(q_corr+q_amb)*exp(-q_corr*q/2*sigma[-1]**2)
    rp_sub=(q_sub-q_corr)/(q_sub+q_corr)*exp(-q_sub*q_corr/2*sigma[0]**2)
    # rp_top = -(n[-1] - n_mean)*exp(-(q_corr*sigma[-1])**2/2)*q0**2/q_corr**2/2.
    # rp_sub = -(n_mean - n[0])*exp(-(q_corr*sigma[0])**2/2)*q0**2/q_corr**2/2.
    # Kinematical reflectivity for the interfaces
    n_diff=n-n_mean
    n_diff[0]=0
    n_diff[-1]=0
    rp=(n_diff[:-1]-n_diff[1:])[:, newaxis]*exp(-(q_corr*sigma[:, newaxis])**2/2)
    p=exp(1.0j*z[:, newaxis]*q_corr)

    r_kin=(rp*p).sum(axis=0)*q0**2/q_corr**2/2.
    r_sra=rp_top+rp_sub*exp(1.0j*d.sum()*q_corr)
    r=r_kin+r_sra

    if return_int:
        return abs(r)**2
    else:
        return r

def reflq_sra(q, lamda, n, d, sigma, return_int=True):
    """Single reflection approximation calculation of the reflectivity"""
    # Length of k-vector in vaccum
    d=d[1:-1]
    sigma=sigma[:-1]
    q0=4*pi/lamda
    # Calculates the wavevector in each layer
    if len(n.shape)==1:
        qj=sqrt((n[:, newaxis]**2-n[-1]**2)*q0**2+(n[-1]*q)**2)
    else:
        qj=sqrt((n**2-n[-1]**2)*q0**2+(n[-1]*q)**2)
    # Fresnel reflectivity for the interfaces
    rp=(qj[:-1]-qj[1:])/(qj[1:]+qj[:-1])*exp(-qj[1:]*qj[:-1]/2*sigma[:, newaxis]**2)
    # The wave does not transverse the ambient and substrate - ignoring them
    # Also, the wave travels from top -> bottom, the array has the first element as the substrate
    # - need to reverse the order.
    phaseterm=(d[:, newaxis]*qj[1:-1])[::-1].cumsum(axis=0)[::-1]
    p=exp(1.0J*phaseterm)
    # Adding the first interface (top -> last in array) since p is not calculated for that layer (p = 1)
    r=rp[-1]+(rp[:-1]*p).sum(axis=0)
    if return_int:
        return abs(r)**2
    else:
        return r

from . import USE_NUMBA

if USE_NUMBA:
    # try to use numba to speed up the calculation intensive functions:
    try:
        from .paratt_numba import Refl, ReflQ, Refl_nvary2
    except Exception as e:
        iprint('Could not use numba, no speed up from JIT compiler:\n'+str(e))