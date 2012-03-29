import numpy as np
import math_utils as mu
import isotropic_matrix as im
reload(im)
import pdb

_ctype = np.complex128

def ass_P(k, d, ctype = _ctype):
    ''' Assamble the propagation matrix for the layers'''
    P = np.zeros((2,2) + k.shape, dtype = ctype)
    P[0,0] = np.exp(-1.0J*k*d)
    P[1,1] = 1.0/P[0,0]
    return P

def ass_X(k, ctype = _ctype):
    '''Assemble the interface matrix for all interfaces'''
    k_j = k[...,1:]
    k_jm1 = k[...,:-1]

    X = np.empty((2,2) + k_j.shape, dtype = ctype)
    
    X[0,0] = (k_j + k_jm1)/k_jm1/2
    X[0,1] = (k_jm1 - k_j)/k_jm1/2
    X[1,0] = X[0,1]
    X[1,1] = X[0,0]
    return X

def ReflQ(Q, lamda, n, d, sigma):
    '''Calculate the reflectivity from a multilayer stack with
       Abeles matrix formalism. '''
    #pdb.set_trace()
    k_vac = 2*np.pi/lamda
    kz = Q[:, np.newaxis]/2.0
    n_amb2 = n[-1]*n[-1]
    k_j = np.sqrt((n*n - n_amb2)*k_vac*k_vac + n_amb2*kz*kz)
    
    #print d.shape, n.shape, k_j.dtype
    #X = ass_X(k_j)
    X = im.ass_X_interfacelayer4(k_j, k_j, k_j, k_j*0+5, k_j*0+5, sigma, sigma*0, sigma*0)
    P = ass_P(k_j, d-10)
    
    PX = mu.dot2_Adiag(P[...,1:-1], X[...,:-1])
    #print P.dtype, X.dtype, PX.dtype
    M = mu.dot2(X[...,-1], reduce(mu.dot2, rollaxis(PX, 3)[::-1]))
    
    r = M[1,0]/M[0,0]
    
    
    return np.abs(r)**2

def ReflQ_mag(Q, lamda, n, d, sigma, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l):
    '''Calculate the reflectivity from a multilayer stack with
       Abeles matrix formalism. '''
    #pdb.set_trace()
    k_vac = 2*np.pi/lamda
    kz = Q[:, np.newaxis]/2.0
    n_amb2 = (n[:,-1]*n[:,-1])[:,np.newaxis]
    #print n_amb2.shape
    #print n.shape, n_l.shape, n_u.shape, kz.shape
    k_j = np.sqrt((n*n - n_amb2)*k_vac*k_vac + n_amb2*kz*kz)
    k_jl = np.sqrt((n_l*n_l - n_amb2)*k_vac*k_vac + n_amb2*kz*kz)
    k_ju = np.sqrt((n_u*n_u - n_amb2)*k_vac*k_vac + n_amb2*kz*kz)
    
    #print np.all(n_u == n), np.all(n_l == n)
    #print np.all(k_ju == k_j), np.all(k_jl == k_j)
    
    #print d.shape, n.shape, k_j.dtype
    #X = ass_X(k_j)
    X = im.ass_X_interfacelayer4(k_j, k_ju, k_jl, dd_u, dd_l, sigma, sigma_l, sigma_u)
    P = ass_P(k_j, d)
    
    PX = mu.dot2_Adiag(P[...,1:-1], X[...,:-1])
    #print P.dtype, X.dtype, PX.dtype
    M = mu.dot2(X[...,-1], reduce(mu.dot2, np.rollaxis(PX, 3)[::-1]))
    
    r = M[1,0]/M[0,0]
    
    
    return np.abs(r)**2

def ReflQ_ref(Q,lamda,n,d,sigma):
    # Length of k-vector in vaccum
    d=d[1:-1]
    sigma=sigma[:-1]
    Q0=4*pi/lamda
    # Calculates the wavevector in each layer
    Qj=sqrt((n[:,newaxis]**2 - n[-1]**2)*Q0**2 + n[-1]**2*Q**2)
    # Fresnel reflectivity for the interfaces
    rp=(Qj[1:]-Qj[:-1])/(Qj[1:]+Qj[:-1])*exp(-Qj[1:]*Qj[:-1]/2*sigma[:,newaxis]**2)
    #print rp.shape #For debugging
    #print d.shape
    #print Qj[1:-1].shape
    p=exp(1.0j*d[:,newaxis]*Qj[1:-1]) # Ignoring the top and bottom layer for the calc.
    #print p.shape #For debugging
    # Setting up a matrix for the reduce function. Reduce only takes one array
    # as argument
    rpp=array(map(lambda x,y:[x,y],rp[1:],p))
    #print rpp.shape
    # Paratt's recursion formula
    def formula(rtot,rint):
        return (rint[0]+rtot*rint[1])/(1+rtot*rint[0]*rint[1])
    # Implement the recursion formula
    r=reduce(formula,rpp,rp[0])
    #print r.shape
    # return the reflectivity 
    return abs(r)**2


if __name__ == '__main__':
    from pylab import *
    theta = arange(0,5,0.01)+1e-13
    lamda = 1.54
    Q = 4*np.pi/lamda*np.sin(theta*np.pi/180)
    n_rep = 10
    n = array([1-7.57e-6+1.73e-7j,] + 
              [1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j]*n_rep +
              [1.0,], dtype = np.complex128)
    d = array([2,] +
              [80,20]*n_rep +
              [2,], dtype = np.complex128)
    n2 = array([1-7.57e-6+1.73e-7j,] + 
              [1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j]*n_rep +
              [1.0,], dtype = np.complex128)
    d2 = array([2,] +
              [80,0]*n_rep +
              [2,], dtype = np.complex128)
    n3 = array([1-7.57e-6+1.73e-7j,] + 
              [1-7.57e-6+1.73e-7j,1-7.57e-6+1.73e-7j]*n_rep +
              [1.0,], dtype = np.complex128)
    d3 = array([2,] +
              [40,40]*n_rep +
              [2,], dtype = np.complex128)
    
    sigma = array([0,] +
                  [0, 0]*n_rep +
                  [0])
    n_att = 1
    import time
    t1 = time.clock()
    for i in range(n_att):
        R_ref = ReflQ_ref(Q, 1.54, n, d, sigma)
        #R_ref = ReflQ(Q, 1.54, n2, d2, sigma)
    t2 = time.clock()
    for i in range(n_att):
        R_try = ReflQ(Q, 1.54, n, d, sigma)
        #R_try = ReflQ(Q, 1.54, n3, d3, sigma)
    t3 = time.clock()
    print 'Paratt: ', (t2 -t1)/n_att
    print 'Abeles: ', (t3 - t2)/n_att
    semilogy(theta, R_ref)
    semilogy(theta, R_try, '.')
    legend(('Parratt', 'Abeles with roughness/interface layer'))
    xlabel('Two Theta [deg.]')
    ylabel('Reflectivity')
    show()
