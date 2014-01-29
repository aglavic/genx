''' Library for reflectivity calculations with neutrons.
Programmed by Matts Bjorck
Last changed 2011-06-11
'''
from numpy import *
import math_utils as mu

def make_D(q_p,q_m):
    return mat([[1,1,0,0],[q_p,-q_p,0,0],[0,0,1,1],[0,0,q_m,-q_m]])
    
def make_P(q_p,q_m,d):
    return mat([[exp(-1.0j*q_p*d),0,0,0],[0,exp(1.0j*q_p*d),0,0],\
                [0,0,exp(-1.0j*q_m*d),0],[0,0,0,exp(1.0j*q_m*d)]])

def make_R(theta_diff):
    ct=cos(theta_diff/2.0)
    st=sin(theta_diff/2.0)
    return mat([[ct,0,st,0],[0,ct,0,st],[-st,0,ct,0],[0,-st,0,ct]])

def make_Sigma(q_p,q_m,sigma):
    ep=exp(-q_p**2*sigma**2)
    em=exp(-q_m**2*sigma**2)
    return mat([[ep,0,0,0],[0,1,0,0],[0,0,em,0],[0,0,0,1]])


def ReflOld(Q,Vp,Vm,d,M_ang):
    ''' Calculates spin-polarized reflectivity according to S.J. Blundell 
        and J.A.C. Bland Phys rev. B. vol 46 3391 (1992)
        Input parameters:   Q : Scattering vector in reciprocal 
                                angstroms Q=4*pi/lambda *sin(theta)
                            Vp: Neutron potential for spin up
                            Vm: Neutron potential for spin down
                            d: layer thickness
                            M_ang: Angle of the magnetic 
                                    moment(radians!) M_ang=0 =>M//nuetron spin
        No roughness is included!
        Returns:            (Ruu,Rdd,Rud,Rdu)
                            (up-up,down-down,up-down,down-up)
    '''
    # Assume first element=substrate and last=ambient!
    Q=Q/2.0
    # Wavevectors in the layers
    Qi_p=sqrt(Q[:,newaxis]**2-Vp)
    Qi_m=sqrt(Q[:,newaxis]**2-Vm)
    #print Qi_p.shape
    #print d.shape
    #print M_ang.shape
    #M_ang=M_ang[::-1]
    #d=d[::-1]
    #Angular difference between the magnetization
    theta_diff=M_ang[1:]-M_ang[:-1] #Unsure but think this is correct
    #theta_diff=theta_diff[::-1]
    #print theta_diff
    def calc_mat(q_p,q_m,d,theta_diff):
        return make_D(q_p,q_m)*make_P(q_p,q_m,d)*make_D(q_p,q_m)**-1*make_R(theta_diff)

    #Calculate the transfer matrix - this implementation is probably REALLY slow...
    M=[make_D(qi_p[-1],qi_m[-1])**-1*make_R(theta_diff[-1])*reduce(lambda x,y:y*x,[calc_mat(q_p,q_m,di,theta_diffi)
           for q_p,q_m,di,theta_diffi in zip(qi_p[1:-1],qi_m[1:-1],d[1:-1],theta_diff[:-1])]
           ,make_D(qi_p[0],qi_m[0]))
           for qi_p,qi_m in zip(Qi_p,Qi_m)]
    #M=[make_D(qi_p[0],qi_m[0])**-1*make_R(theta_diff[0])*reduce(lambda x,y:y*x,[calc_mat(q_p,q_m,di,theta_diffi)
    #     for q_p,q_m,di,theta_diffi in zip(qi_p[1:-1],qi_m[1:-1],d[1:-1],theta_diff[1:])]
    #       ,make_D(qi_p[-1],qi_m[-1]))
    #       for qi_p,qi_m in zip(Qi_p,Qi_m)]
    #print 'Matrix calculated'
    #print M[0]
    # transform M into an array - for fast indexing
    M=array([array(m) for m in M])
    #print M.shape
    Ruu=(M[:,1,0]*M[:,2,2]-M[:,1,2]*M[:,2,0])/(M[:,0,0]*M[:,2,2]-M[:,0,2]*M[:,2,0])
    Rud=(M[:,3,0]*M[:,2,2]-M[:,3,2]*M[:,2,0])/(M[:,0,0]*M[:,2,2]-M[:,0,2]*M[:,2,0])
    Rdu=(M[:,1,2]*M[:,0,0]-M[:,1,0]*M[:,0,2])/(M[:,0,0]*M[:,2,2]-M[:,0,2]*M[:,2,0])
    Rdd=(M[:,3,2]*M[:,0,0]-M[:,3,0]*M[:,0,2])/(M[:,0,0]*M[:,2,2]-M[:,0,2]*M[:,2,0])
    #print 'Reflectivites calculated'
    return abs(Ruu)**2,abs(Rdd)**2,abs(Rud)**2,abs(Rdu)**2

#======================================================================
# New quicker way of doing spin pol. calcs.

ctype = complex128

def ass_X(k_p, k_m, theta_diff):
    ''' Make the interface transmission matrix for neutron reflection from 
    a interface.
    '''
    k_pj1 = k_p[:, :-1]
    k_pj = k_p[:, 1:]
    k_mj1 = k_m[:, :-1]
    k_mj = k_m[:, 1:]
    costd = cos(theta_diff/2.0)
    sintd = sin(theta_diff/2.0)
    X = zeros((4,4) + k_pj.shape, dtype = ctype)
    X[0,0] = costd*(k_pj1 + k_pj)/2/k_pj
    X[0,1] = -costd*(k_pj1 - k_pj)/2/k_pj
    X[0,2] = sintd*(k_pj + k_mj1)/2/k_pj
    X[0,3] = sintd*(k_pj - k_mj1)/2/k_pj
    #X[0] = X[0]/2/k_pj
    X[1,0] = X[0,1]#-(costd*(k_pj1 - k_pj))/(2*k_pj)
    X[1,1] = X[0,0]#(costd*(k_pj1 + k_pj))/(2*k_pj)
    X[1,2] = X[0,3]#(sintd*(k_pj - k_mj1))/(2*k_pj)
    X[1,3] = X[0,2]#(sintd*(k_pj + k_mj1))/(2*k_pj)
    X[2,0] = -(sintd*(k_pj1 + k_mj))/(2*k_mj)
    X[2,1] = (sintd*(k_pj1 - k_mj))/(2*k_mj)
    X[2,2] = (costd*(k_mj1 + k_mj))/(2*k_mj)
    X[2,3] = -(costd*(k_mj1 - k_mj))/(2*k_mj)
    X[3,0] = X[2,1]#(sintd*(k_pj1 - k_mj))/(2*k_mj)
    X[3,1] = X[2,0]#-(sintd*(k_pj1 + k_mj))/(2*k_mj)
    X[3,2] = X[2,3]#-(costd*(k_mj1 - k_mj))/(2*k_mj)
    X[3,3] = X[2,2]#(costd*(k_mj1 + k_mj))/(2*k_mj)
    
    return X

def gauss(q, sigma2):
    '''Fourier transform of the interface roughness weight function
    '''
    return exp(-q**2*sigma2/2.0)
    
def include_sigma(X, k_p, k_m, sigma, w = gauss):
    '''Function to include roughness into the interface matrix.
    '''
    sigma2 = sigma[...,:-1]**2
    k_pj1 = k_p[:, :-1]
    k_pj = k_p[:, 1:]
    k_mj1 = k_m[:, :-1]
    k_mj = k_m[:, 1:]
    X[0,0] = X[0,0]*w(k_pj - k_pj1, sigma2)
    X[0,1] = X[0,1]*w(k_pj + k_pj1, sigma2)
    X[0,2] = X[0,2]*w(k_pj - k_mj1, sigma2)
    X[0,3] = X[0,3]*w(k_pj + k_mj1, sigma2)
    X[1,0] = X[0,1]#X[1,0]*w(k_pj + k_pj1, sigma2)
    X[1,1] = X[0,0]#X[1,1]*w(k_pj - k_pj1, sigma2)
    X[1,2] = X[0,3]#X[1,2]*w(k_pj + k_mj1, sigma2)
    X[1,3] = X[0,2]#X[1,3]*w(k_pj - k_mj1, sigma2)
    X[2,0] = X[2,0]*w(k_mj - k_pj1, sigma2)
    X[2,1] = X[2,1]*w(k_mj + k_pj1, sigma2)
    X[2,2] = X[2,2]*w(k_mj - k_mj1, sigma2)
    X[2,3] = X[2,3]*w(k_mj + k_mj1, sigma2)
    X[3,0] = X[2,1]#X[3,0]*w(k_mj + k_pj1, sigma)
    X[3,1] = X[2,0]#X[3,1]*w(k_mj - k_pj1, sigma)
    X[3,2] = X[2,3]#X[3,2]*w(k_mj + k_mj1, sigma)
    X[3,3] = X[2,2]#X[3,3]*w(k_mj - k_mj1, sigma)
    return X

def ass_P(k_p, k_m, d):
    ''' Make the layer proagation matrix for a layer.
    '''
    P = zeros((4,4) + k_p.shape, dtype=ctype)
    P[0,0] = exp(-1.0J*k_p*d)
    P[1,1] = 1/P[0,0]#exp(1.0J*k_p*d)
    P[2,2] = exp(-1.0J*k_m*d)
    P[3,3] = 1/P[2,2]#exp(1.0J*k_m*d)
    return P


def Refl(Q, Vp, Vm, d, M_ang, sigma = None):
    '''A quicker implementation than the ordinary slow implementaion in Refl
    Calculates spin-polarized reflectivity according to S.J. Blundell 
        and J.A.C. Blnd Phys rev. B. vol 46 3391 (1992)
        The algorithm assumes that the first element in the arrays represents
        the substrate and the last the ambient layer.
        Input parameters:   Q : Scattering vector in reciprocal 
                                angstroms Q=4*pi/lambda *sin(theta)
                            Vp: Neutron potential for spin up
                            Vm: Neutron potential for spin down
                            d: layer thickness
                            M_ang: Angle of the magnetic 
                                    moment(radians!) M_ang=0 =>M//nuetron spin
                            sigma: The roughness of the upper interface.
        Returns:            (Ruu,Rdd,Rud,Rdu)
                            (up-up,down-down,up-down,down-up)
    '''
    # Assume first element=substrate and last=ambient!
    k_amb = Q[:, newaxis]/2.0
    # Wavevectors in the layers
    k_p = sqrt(k_amb**2 - Vp).astype(complex128)
    k_m = sqrt(k_amb**2 - Vm).astype(complex128)
    #Angular difference between the magnetization
    theta_diff=M_ang[1:] - M_ang[:-1]
    #if sigma == None:
    #    sigma = zeros(d.shape)
    # Assemble the interface reflectivity matrix
    X = ass_X(k_p, k_m, theta_diff)
    if sigma != None:
        X = include_sigma(X, k_p, k_m, sigma)
    # Assemble the layer propagation matrices
    P = ass_P(k_p, k_m, d)
    # Multiply the propagation matrices with the interface matrix
    PX = mu.dot4_Adiag(P[...,1:-1], X[...,:-1])
    # Multiply up the sample matrix
    M = mu.dot4(X[...,-1], reduce(mu.dot4, rollaxis(PX, 3)[::-1]))
    #print M.shape
    denom = M[0,0]*M[2,2]-M[0,2]*M[2,0]
    Ruu = (M[1,0]*M[2,2]-M[1,2]*M[2,0])/denom
    Rud = (M[3,0]*M[2,2]-M[3,2]*M[2,0])/denom
    Rdu = (M[1,2]*M[0,0]-M[1,0]*M[0,2])/denom
    Rdd = (M[3,2]*M[0,0]-M[3,0]*M[0,2])/denom
    
    return abs(Ruu)**2,abs(Rdd)**2,abs(Rud)**2,abs(Rdu)**2
    

if __name__=='__main__':
    Q=arange(0.01,0.2,0.0005)
    sld_Fe=8e-6
    sld_Fe_p=12.9e-6
    sld_Fe_m=2.9e-6
    sld_Pt=6.22e-6
    def pot(sld):
        lamda=5.0
        #return (2*pi/lamda)**2*(1-(1-lamda**2/2/pi*sld)**2)
        return sld/pi
    Vp=array([pot(sld_Pt),pot(sld_Fe_p),pot(sld_Pt),pot(sld_Fe_p),0])
    Vm=array([pot(sld_Pt),pot(sld_Fe_m),pot(sld_Pt),pot(sld_Fe_m),0])
    d=array([3,100,50,100,3])
    M_ang=array([0.0,45*pi/180,0.0,90*pi/180,0.0,])
    sigma=array([10.,10.,10.,10.,10.0])*0
    import time
    t1 = time.time()
    for i in range(10):
        r = Refl(Q,Vp,Vm,d,M_ang, sigma)
    t2 = time.time()
    for i in range(10):
        r_orig = ReflOld(Q,Vp,Vm,d,M_ang)
    t3 = time.time()
    print 'Old version: ', t3 - t2
    print 'New version: ', t2 - t1
    print 'Speedup: ', (t3 - t2)/(t2 - t1)
    from pylab import *
    #plot(Q,log10(r[0]+1e-6),Q,log10(r[1]+1e-6),Q,log10(r[2]+1e-6),Q,log10(r[3]+1e-6))
    #io.write_array(open('test.dat','w'),array(zip(Q,abs(r[0]),abs(r[1]),abs(r[2]))))
    for rc in r:
        plot(Q,log10(abs(rc)))
    for rc in r_orig:   
        plot(Q,log10(abs(rc)),'.')
    print 'Done'
    show()
    if True:
        import profile
        profile.run('[Refl(Q,Vp,Vm,d,M_ang, sigma) for i in range(50)]')
    
