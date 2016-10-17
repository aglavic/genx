#from scipy import *
from numpy import *

# "Ordinary" implementaion of Parrats recursion formula
# theta-vector, lamda- can be a vector,n-1Dvector, d-1Dvector, sigma-1Dvector
def Refl(theta,lamda,n,d,sigma):
    d=d[1:-1]
    sigma=sigma[:-1]
    # Length of k-vector in vaccum
    k=2*math.pi/lamda
    # Calculates the wavevector in each layer
    Qj=2*n[-1]*k*sqrt(n[:,newaxis]**2/n[-1]**2-cos(theta*math.pi/180)**2)
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
    #return r

# "Ordinary" implementaion of Parrats recursion formula
# Q-vector,n-1Dvector, d-1Dvector, sigma-1Dvector
def ReflQ(Q,lamda,n,d,sigma):
    # Length of k-vector in vaccum
    d=d[1:-1]
    sigma=sigma[:-1]
    Q0=4*pi/lamda
    Q = Q.astype(complex128)
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
    #return r

# Parrats recursion formula for varying n given by n_func-function
def Refl_nvary(theta,lamda,n_func,d,sigma):
    d=d[1:-1]
    sigma=sigma[:-1]
    # Length of k-vector in vaccum
    k=2*math.pi/lamda
    # Calculates the wavevector in each layer
    #print n_func
    ss=transpose((sin(theta[:,newaxis]*pi/180.0)/lamda)*ones(len(n_func)))
    #print ss.shape
    #print theta.shape
    #print len(n_func)
    n=array(map(lambda f,val:f(val),n_func,ss))
    #print n
    Qj=2*k*n[-1]*sqrt(n**2/n[-1]**2-cos(theta*math.pi/180)**2)
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
    #return r

def ReflProfiles(theta,lamda,n,d,sigma,profile):
    d=d[1:-1]
    sigma=sigma[:-1]
    # Length of k-vector in vaccum
    k=2*math.pi/lamda
    # Calculates the wavevector in each layer
    Qj=2*n[-1]*k*sqrt(n[:,newaxis]**2/n[-1]**2 - cos(theta*math.pi/180)**2)
    # Function to map caclulate the roughness values
    def w(QiQj,sigma,profile):
        # erf profile
        if profile==0:
            return exp(-QiQj*sigma**2/2)
        # exponential profile
        elif profile==1:
            return 1/(1+sqrt(QiQj)*sigma**2/2)
        # linear profile
        elif profile==2:
            return sin(sqrt(3)*sigma*sqrt(QiQj))/sqrt(3)/sigma/sqrt(QiQj)
        else:
            # If any wrong number are typed in return the erf profile
            return exp(-QiQj*sigma**2/2)

    interface=array(map(w,Qj[:-1]*Qj[1:],sigma,profile))
    # Fresnel reflectivity for the interfaces
    rp=(Qj[1:]-Qj[:-1])/(Qj[1:]+Qj[:-1])*interface
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
    #return r

# paratts algorithm for n as function of lamda or theta
def Refl_nvary2(theta,lamda,n_vector,d,sigma):
    d=d[1:-1]
    sigma=sigma[:-1]
    # Length of k-vector in vaccum
    k=2*math.pi/lamda
    # Calculates the wavevector in each layer
    #print n_func
    #ss=transpose((sin(theta[:,newaxis]*pi/180.0)/lamda)*ones(len(n_func)))
    #print ss.shape
    #print theta.shape
    #print len(n_func)
    #n=array(map(lambda f,val:f(val),n_func,ss))
    n=n_vector
    #print n
    Qj=2*n[-1]*k*sqrt(n**2/n[-1]**2-cos(theta*math.pi/180)**2)
    #print sigma.shape, Qj.shape
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
    #return r


def reflq_kin(q, lamda, n, d, sigma):
    """Calculates the reflectivity in the kinematical approximation"""
    d = d[:-1]
    d[0] = 0
    z = d.sum() - d.cumsum()
    sigma = sigma[:-1]
    q0 = 4*pi/lamda
    # Q = Q.astype(complex128)
    # The internal wave vector calacuted with the thickness averaged refractive index.
    n_mean = (n[1:-1]*d[1:]/d.sum()).sum()
    q_corr = sqrt((n_mean**2 - n[-1]**2)*q0**2 + (n[-1]*q)**2)
    # Uncomment this for a pure kinematical treatment
    # q_corr = q
    # Kinematical reflectivity for the interfaces
    rp = (n[:-1] - n[1:])[:, newaxis]*exp(-(q_corr*sigma[:, newaxis])**2/2)
    p = exp(1.0j*z[:, newaxis]*q_corr)

    r = (rp*p).sum(axis=0)*q0**2/q_corr**2/2.

    # return the reflectivity
    return abs(r)**2

def reflq_pseudo_kin(q, lamda, n, d, sigma):
    """Calculates the reflectivity in a pseudo kinematical approximation.
    The mean refractive index of the film is simulated with the single reflection approximation and the deviation from
    the mean is simulated with the kinematical approximation.
    """
    d = d[:-1]
    d[0] = 0
    z = d.sum() - d.cumsum()
    sigma = sigma[:-1]
    q0 = 4*pi/lamda
    # Q = Q.astype(complex128)
    # The internal wave vector calacuted with the thickness averaged refractive index.
    n_mean = (n[1:-1]*d[1:]/d.sum()).sum()
    q_corr = sqrt((n_mean**2 - n[-1]**2)*q0**2 + n[-1]**2*q**2)
    q_sub = sqrt((n[0]**2 - n[-1]**2)*q0**2 + n[-1]**2*q**2)
    q_amb = n[-1]*q
    # Top interface
    rp_top = (q_corr - q_amb) / (q_corr + q_amb) * exp(-q_corr*q/2 * sigma[-1]**2)
    rp_sub = (q_sub - q_corr) / (q_sub + q_corr) * exp(-q_sub*q_corr/2 * sigma[0]**2)
    #rp_top = -(n[-1] - n_mean)*exp(-(q_corr*sigma[-1])**2/2)*q0**2/q_corr**2/2.
    #rp_sub = -(n_mean - n[0])*exp(-(q_corr*sigma[0])**2/2)*q0**2/q_corr**2/2.
    # Kinematical reflectivity for the interfaces
    n_diff = n - n_mean
    n_diff[0] = 0
    n_diff[-1] = 0
    rp = (n_diff[:-1] - n_diff[1:])[:, newaxis]*exp(-(q_corr*sigma[:, newaxis])**2/2)
    p = exp(1.0j*z[:, newaxis]*q_corr)

    r_kin = (rp*p).sum(axis=0)*q0**2/q_corr**2/2.
    r_sra = rp_top + rp_sub*exp(1.0j*d.sum()*q_corr)

    # return the reflectivity
    return abs(r_kin + r_sra)**2


def reflq_sra(q, lamda, n, d, sigma):
    """Single reflection approximation calculation of the reflectivity"""
    # Length of k-vector in vaccum
    d = d[1:-1]
    sigma=sigma[:-1]
    q0 = 4*pi/lamda
    # Calculates the wavevector in each layer
    qj = sqrt((n[:, newaxis]**2 - n[-1]**2)*q0**2 + (n[-1]*q)**2)
    # Fresnel reflectivity for the interfaces
    rp = (qj[:-1] - qj[1:])/(qj[1:] + qj[:-1])*exp(-qj[1:]*qj[:-1]/2*sigma[:, newaxis]**2)
    # The wave does not transverse the ambient and substrate - ignoring them
    # Also, the wave travels from top -> bottom, the array has the first element as the substrate
    # - need to reverse the order.
    phaseterm = (d[:, newaxis]*qj[1:-1])[::-1].cumsum(axis=0)[::-1]
    p = exp(1.0J*phaseterm)
    # Adding the first interface (top -> last in array) since p is not calculated for that layer (p = 1)
    r = rp[-1] + (rp[:-1]*p).sum(axis=0)
    # return the reflectivity
    return abs(r)**2


if __name__=='__main__':
    theta=arange(0,5,0.01)+1e-13
    #c=paratt.Refl(theta,1.54,array([1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1]),array([80,20,80,20,80,20]),0)
    #c1=Refl(theta,1.54,array([1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1]),array([80,20,80,20,80,20]),4)
    c2=Refl(theta,1.54,array([1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1]),array([2,80,20,80,20,80,20,2]),array([0,0,0,0,0,0,0,0]))
    #print c.shape
    #gplt.plot(theta,log10(c1),theta,log10(c2))
    #gplt.plot(theta,log10(c2))
