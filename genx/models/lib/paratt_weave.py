'''File: paratt_weave.py an implementation of the paratt algorithm in weave
should yield a speed increase > 2.
Programmed by: Matts Bjorck
Last changed: 2008 11 23
'''

from numpy import *
import scipy.weave as weave

def Refl(theta,lamda,n,d,sigma):
    # Length of k-vector in vaccum
    k=2*math.pi/lamda
    theta=array(theta,dtype=float64)
    n = n.astype(complex128)
    d = d[1:-1].astype(float64)
    sigma = sigma[:-1].astype(float64)
    torad=math.pi/180.0
    Imag=1.0J
    R=array(zeros(theta.shape),dtype=complex128)
    Qj=1.01J
    Qj1=1.01J
    r=1.01J
    rp=1.01J
    p=1.01J
    # Calculates the wavevector in each layer
    code='''
    int points=Ntheta[0];
    int interfaces=Nsigma[0];
    int j=0;
    double costheta2;
    for(int i = 0; i < points; i++){
        costheta2 = cos(theta[i]*torad);
        costheta2 *= costheta2;
        Qj = 2.0*k*sqrt(n[0]*n[0] - costheta2);
        Qj1 = 2.0*k*sqrt(n[1]*n[1] - costheta2);
        rp=(Qj - Qj1)/(Qj1 + Qj)*exp(-Qj1*Qj/2.0*sigma[0]*sigma[0]);
        r = rp;
        for(j = 1; j < interfaces; j++){
            Qj = Qj1;
            Qj1 = 2.0*k*sqrt(n[j+1]*n[j+1] - costheta2);
            rp = (Qj-Qj1)/(Qj1+Qj)*exp(-Qj1*Qj/2.0*sigma[j]*sigma[j]);
            p = exp(Imag*d[j-1]*Qj);
            r = (rp+r*p)/(1.0 + rp*r*p);
        }
        
        R[i]=r*conj(r);
    }

    '''
    weave.inline(code,['theta','n','d','sigma','k','torad','Imag','R','Qj','Qj1','r','rp','p'],compiler='gcc')

    return real(R)

def Refl_nvary2(theta,lamda,n,d,sigma):
    # Length of k-vector in vaccum
    #print n.shape, theta.shape, d.shape
    theta=array(theta,dtype=float64)
    n = n.astype(complex128)
    d = d[1:-1].astype(float64)
    sigma = sigma[:-1].astype(float64)
    #print n.shape, theta.shape, d.shape
    R=array(zeros(theta.shape),dtype=float64)
    # Calculates the wavevector in each layer
    code='''
    int points=Ntheta[0];
    int layers = Nn[0];
    int interfaces=Nsigma[0];
    int j, m;
    double costheta2, k;
    double pi = 3.141592;
    double torad = pi/180.0; 
    std::complex<double> Qj, Qj1, r, rp, p, Imag(0.0, 1.0);
    for(int i = 0;i < points; i++){
        costheta2 = cos(theta[i]*torad);
        costheta2 *= costheta2;
        k = 4*pi/lamda[i];
        Qj = k*sqrt(n[0*points + i]*n[0*points + i]  - costheta2);
        Qj1 = k*sqrt(n[1*points + i]*n[1*points + i] - costheta2);
        rp = (Qj - Qj1)/(Qj1 + Qj)*exp(-Qj1*Qj/2.0*sigma[0]*sigma[0]);
        r = rp;
        for(j = 1; j < interfaces; j++){
            Qj = Qj1;
            Qj1 = k*sqrt(n[(j+1)*points + i]*n[(j+1)*points + i] - costheta2);
            rp = (Qj - Qj1)/(Qj1 + Qj)*exp(-Qj1*Qj/2.0*sigma[j]*sigma[j]);
            p = exp(Imag*d[j-1]*Qj);
            r = (rp+r*p)/(1.0+rp*r*p);
        }
        
       R[i] = real(r*conj(r));
    }

    '''
    #code = ''' int points = Ntheta[0];'''
    weave.inline(code,['theta','n','d','sigma','lamda','R'], compiler='gcc')#, headers = ['<complex>'])

    return R

def Refl_nvary2_nosigma(theta, lamda, n, d):
    # Length of k-vector in vaccum
    #print n.shape, theta.shape, d.shape
    theta=array(theta,dtype=float64)
    n = n.astype(complex128)
    d = d[1:-1].astype(float64)
    #print n.shape, theta.shape, d.shape
    R=array(zeros(theta.shape),dtype=float64)
    # Calculates the wavevector in each layer
    code='''
    int points = Ntheta[0];
    int layers = Nn[0];
    int interfaces = Nd[0] + 1 ;
    int j, m;
    double costheta2, k;
    double pi = 3.141592;
    double torad = pi/180.0; 
    std::complex<double> Qj, Qj1, r, rp, p, Imag(0.0, 1.0);
    for(int i = 0;i < points; i++){
        costheta2 = cos(theta[i]*torad);
        costheta2 *= costheta2;
        k = 4*pi/lamda[i];
        Qj = k*sqrt(n[0*points + i]*n[0*points + i]  - costheta2);
        Qj1 = k*sqrt(n[1*points + i]*n[1*points + i] - costheta2);
        rp = (Qj - Qj1)/(Qj1 + Qj);
        r = rp;
        for(j = 1; j < interfaces; j++){
            Qj = Qj1;
            Qj1 = k*sqrt(n[(j+1)*points + i]*n[(j+1)*points + i] - costheta2);
            rp = (Qj - Qj1)/(Qj1 + Qj);
            p = exp(Imag*d[j-1]*Qj);
            r = (rp+r*p)/(1.0+rp*r*p);
        }
        
       R[i] = real(r*conj(r));
    }

    '''
    #code = ''' int points = Ntheta[0];'''
    weave.inline(code,['theta','n','d','lamda','R'], compiler='gcc')#, headers = ['<complex>'])

    return R


if __name__=='__main__':
    import paratt
    import time
    import pylab as pl
    
    theta=arange(0,10,0.01)+1e-12
    rep = 1000
    n = array([1-7.57e-6+1.73e-7j] + [1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j]*rep +[1])
    d = array([1] + [80,20,80,20,80,20]*rep + [1])
    sigma = array([0] + [0,0,0,0,0,0]*rep + [0])
    print n.shape
    t1=time.clock()
    #c1=paratt.Refl_nvary2(theta, 1.54*ones(theta.shape), n[:, newaxis]*ones(theta.shape), d,sigma)
    c1=paratt.Refl_nvary2(theta, 1.54*ones(theta.shape), n[:, newaxis]*ones(theta.shape), d,sigma*0)
    #c1=paratt.Refl(theta, 1.54, n, d,sigma)
    t2=time.clock()
    #c2 = Refl_nvary2(theta, 1.54*ones(theta.shape), n[:, newaxis]*ones(theta.shape), d,sigma) 
    c2 = Refl_nvary2_nosigma(theta, 1.54*ones(theta.shape), n[:, newaxis]*ones(theta.shape), d) 
    #c2 = Refl(theta, 1.54, n, d, sigma)
    t3=time.clock()
    print t2-t1,t3-t2
    pl.plot(theta,log10(c1),'x',theta,log10(c2))
    pl.show()