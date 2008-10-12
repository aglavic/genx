#from scipy import *
from numpy import *

# "Ordinary" implementaion of Parrats recursion formula
# theta-vector, lamda- can be a vector,n-1Dvector, d-1Dvector, sigma-1Dvector
def Refl(theta,lamda,n,d,sigma):
    # Length of k-vector in vaccum
    k=2*math.pi/lamda
    # Calculates the wavevector in each layer
    Qj=2*k*sqrt(n[:,newaxis]**2-cos(theta*math.pi/180)**2)
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
    # Calculates the wavevector in each layer
    Qj=sqrt((n[:,newaxis]**2-1)*Q0**2+Q**2)
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
    Qj=2*k*sqrt(n**2-cos(theta*math.pi/180)**2)
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
    # Length of k-vector in vaccum
    k=2*math.pi/lamda
    # Calculates the wavevector in each layer
    Qj=2*k*sqrt(n[:,newaxis]**2-cos(theta*math.pi/180)**2)
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
    Qj=2*k*sqrt(n**2-cos(theta*math.pi/180)**2)
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


if __name__=='__main__':
    theta=arange(0,5,0.01)+1e-13
    #c=Refl(theta,1.54,array([1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1]),array([80,20,80,20,80]),0)
    #c1=Refl(theta,1.54,array([1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1]),array([80,20,80,20,80,20]),4)
    c2=Refl(theta,1.54,array([1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1]),array([80,20,80,20,80,20]),array([0,0,0,0,0,0,0]))
    #print c.shape
    #gplt.plot(theta,log10(c1),theta,log10(c2))
    gplt.plot(theta,log10(c2))
