''' Library for reflectivity calculations with neutrons.
Programmed by Matts Bjorck
Last changed 2008-09-01
'''
from numpy import *

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


def Refl(Q, wavelength, np, nm, d, M_ang):
    ''' Calculates spin-polarized reflectivity according to S.J. Blundell 
        and J.A.C. Blnd Phys rev. B. vol 46 3391 (1992)
        Input parameters:   Q : Scattering vector in reciprocal 
                                angstroms Q=4*pi/lambda *sin(theta)
                            wavelength: wavelenght of the neutron
                            np: Neutron ref. index for spin up
                            nm: Neutron ref. index for spin down
                            d: layer thickness
                            M_ang: Angle of the magnetic 
                                    moment(radians!) M_ang=0 =>M//nuetron spin
        No roughness is included!
        Returns:            (Ruu,Rdd,Rud,Rdu)
                            (up-up,down-down,up-down,down-up)
    '''
    k0 = (2*pi/wavelength)
    Vp = k0**2*(1-np**2)
    Vm = k0**2*(1-nm**2)
    # Assume first element=substrate and last=ambient!
    Q=Q/2.0
    # Wavevectors in the layers
    #Qi_p=sqrt(Q[:,newaxis]**2-Vp)
    #Qi_m=sqrt(Q[:,newaxis]**2-Vm)
    Qi_p = sqrt(np[-1]**2*Q[:, newaxis]**2 + (np**2 - np[-1]**2)*k0**2)
    Qi_m = sqrt(nm[-1]**2*Q[:, newaxis]**2 + (nm**2 - nm[-1]**2)*k0**2)
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

if __name__=='__main__':
    Q=arange(0.001,0.2,0.0005)
    sld_Fe=8e-6
    sld_Fe_p=12.9e-6
    sld_Fe_m=2.9e-6
    sld_Pt=6.22e-6
    def pot(sld):
        lamda=5.0
        return (2*pi/lamda)**2*(1-(1-lamda**2/2/pi*sld)**2)
        return sld/pi
    Vp=array([pot(sld_Pt),pot(sld_Fe_p),pot(sld_Pt),pot(sld_Fe_p),0])
    Vm=array([pot(sld_Pt),pot(sld_Fe_m),pot(sld_Pt),pot(sld_Fe_m),0])
    d=array([3,100,50,100,3])
    M_ang=array([0.0,45*pi/180,0.0,90*pi/180,0.0,])
    sigma=array([100.,100.,50.,100.,0.0])
    r=Refl(Q,Vp,Vm,d,M_ang)
    gplt.plot(Q,log10(r[0]+1e-6),Q,log10(r[1]+1e-6),Q,log10(r[2]+1e-6),Q,log10(r[3]+1e-6))
    io.write_array(open('test.dat','w'),array(zip(Q,abs(r[0]),abs(r[1]),abs(r[2]))))
    #gplt.plot(Q,log10(abs(r[0])**2))

    
