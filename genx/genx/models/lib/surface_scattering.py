"""
Extension library for surface diffraction.

Rebuild of old C++ extension module.
"""

from logging import debug
from numpy import *
import numba


@numba.jit(numba.complex128[:](numba.float64[:], numba.float64[:], numba.float64[:],
                               numba.float64[:], numba.float64[:], numba.float64[:],
                               numba.float64[:], numba.float64[:], numba.complex128[:,:],
                               numba.float64[:,:,:], numba.float64[:]),
            nopython=True, parallel=True, cache=True)
def surface_lattice_sum(x, y, z,
                        h, k, l,
                        u, oc, f,
                        Pt, dinv):
    Nh=h.shape[0]
    Noc=oc.shape[0]
    fs=zeros(Nh, dtype=complex128)


    for i in numba.prange(Nh):
        pidi = -2.0*(pi*dinv[i])**2
        hi, ki, li = h[i], k[i], l[i]
        for j in range(Noc):
            # Loop over symmetry operations
            tmp=0.0j
            for m in range(Pt.shape[0]):
                tmp+=exp(2.0j*pi*(
                        hi*(Pt[m, 0, 0]*x[j] + Pt[m, 0, 1]*y[j] + Pt[m, 0, 2])+
                        ki*(Pt[m, 1, 0]*x[j] + Pt[m, 1, 1]*y[j]+ Pt[m, 1, 2]) +
                        li*z[j]))
            fs[i]+=oc[j]*f[i, j]*exp(pidi*u[j])*tmp
    return fs
