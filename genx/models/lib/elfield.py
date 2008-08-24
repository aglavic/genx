from scipy import *


def IntElfield(theta,lamda,n,d):
    # Length of k-vector in vaccum
    k=2*math.pi/lamda
    # Calculates the wavevector in each layer
    Qj=2*k*sqrt(n[:,newaxis]**2-cos(theta*math.pi/180)**2)
    #print Qj
    # Fresnel reflectivity for the interfaces
    rp=(Qj[1:]-Qj[:-1])/(Qj[1:]+Qj[:-1])
    tp=1+rp
    #tp=2*Qj[1:]/(Qj[1:]+Qj[:-1])
    #print tp.shape
    #Beta=k*d*n[:-1]*sin(theta*math.pi/180)
    #print Beta.shape
    D=d[:,newaxis]*ones(theta.shape)
    #E_p=[1]
    #E_m=[0]
    E_p=[1*ones(theta.shape)]
    E_m=[0*ones(theta.shape)]
    #print E_p[0].shape
    #print D[0].shape
    #print Qj[0].shape
    #for i in range(0,len(rp)):
    #    E_p.append(1/tp[i]*(E_p[i]*exp(-1.0J*D[i]*Qj[i]/2)+E_m[i]*rp[i]*exp(1.0J*D[i]*Qj[i]/2)))
    #    E_m.append(1/tp[i]*(E_m[i]*exp(1.0J*D[i]*Qj[i]/2)+E_p[i]*rp[i]*exp(-1.0J*D[i]*Qj[i]/2)))
    for i in range(0,len(rp)):
        E_p.append(1/tp[i]*(E_p[i]*exp(-1.0J*D[i]*Qj[i]/2)+E_m[i]*rp[i]*exp(1.0J*D[i]*Qj[i]/2)))
        E_m.append(1/tp[i]*(E_m[i]*exp(1.0J*D[i]*Qj[i]/2)+E_p[i]*rp[i]*exp(-1.0J*D[i]*Qj[i]/2)))
    #print E_p
    E_p=array(E_p[1:])
    E_m=array(E_m[1:])
    return (E_p/E_p[-1],E_m/E_p[-1],Qj/2) 

def AmpElfield_q(k,kx,lamda,n,d):
    # Length of k-vector in vaccum
    k=2*math.pi/lamda
    # Calculates the wavevector in each layer
    #Qj=2*k*sqrt(n[:,newaxis]**2-cos(theta*math.pi/180)**2)
    Qj=2*sqrt(k**2*n[:,newaxis]**2-kx**2)
    # Fresnel reflectivity for the interfaces
    rp=(Qj[1:]-Qj[:-1])/(Qj[1:]+Qj[:-1])
    #print rp
    tp=1+rp
    #tp=2*Qj[1:]/(Qj[1:]+Qj[:-1])
    #print tp.shape
    #Beta=k*d*n[:-1]*sin(theta*math.pi/180)
    #print Beta.shape
    D=d[:,newaxis]*ones(kx.shape)
    #print D.shape
    #print rp.shape
    #E_p=[1]
    #E_m=[0]
    # old version
    E_p=[1*ones(kx.shape)]
    E_m=[0*ones(kx.shape)]
    for i in range(0,len(rp)):
        E_p.append(1/tp[i]*(E_p[i]*exp(-1.0J*D[i]*Qj[i]/2)+E_m[i]*rp[i]*exp(1.0J*D[i]*Qj[i]/2)))
        E_m.append(1/tp[i]*(E_m[i]*exp(1.0J*D[i]*Qj[i]/2)+E_p[i]*rp[i]*exp(-1.0J*D[i]*Qj[i]/2)))
    #Added 020905 for testing another T amplitude (The Transmitted amp. trough the interface)
    #print E_p
    # End old version
    E_p=array(E_p)
    E_m=array(E_m)
    #print E_p.shape
    #print Qj.shape
    dtemp=r_[d,0]
    #print dtemp
    dtemp=dtemp[:,newaxis]*ones(kx.shape)
    E_p=E_p*exp(-1.0J*dtemp*Qj/2)
    E_m=E_m*exp(1.0J*dtemp*Qj/2)
    E_p=list(E_p)
    E_m=list(E_m)
    # End addition
    E_p.reverse()
    E_m.reverse()
    Qj=list(Qj)
    Qj.reverse()
    Qj=array(Qj)
    # Removed 020905 as said above and replaced
    E_p=array(E_p) #Old version
    #E_p=array(E_p)
    # End addition
    E_m=array(E_m)
    q=Qj/2 # Old ver. replaced 020905 with:
    #return (E_p[:-1]/E_p[0],E_m[:-1]/E_p[0],q[:-1]) # Last correspond to kz
    #testar amplitud i lager j+1 istallet
    return (E_p[1:]/E_p[0],E_m[1:]/E_p[0],q[1:]) # Last correspond to kz

def AmpElfield2(kx,k,n,z):
    kz=sqrt(n[:,newaxis]**2*k**2-kx**2)
    r=(kz[:-1]-kz[1:])/(kz[:-1]+kz[1:])
    t=1+r
    X=0*ones(kx.shape)
    for i in range(len(n)-2,-1,-1):
        X=exp(-2j*kz[i]*z[i])*(r[i]+X*exp(2j*kz[i+1]*z[i]))/(1+r[i]*X*exp(2j*kz[i+1]*z[i]))
        print i
    # X=reflected amplitude...
    r=(kz[1:]-kz[:-1])/(kz[:-1]+kz[1:])
    t=1+r
    R=[X]
    T=[1*ones(X.shape)]
    for i in range(0,len(n)-1,1):
        R.append(1/t[i]*(T[i]*r[i]*exp(-1.0j*(kz[i+1]+kz[i])*z[i])+R[i]*exp(-1.0j*(kz[i+1]-kz[i])*z[i])))
        T.append(1/t[i]*(T[i]*exp(1.0j*(kz[i+1]-kz[i])*z[i])+R[i]*r[i]*exp(1.0j*(kz[i+1]+kz[i])*z[i])))
    #R[len(n)-1]=0*ones(X.shape)
    R=array(R)
    T=array(T)
    return (T,R,kz)

if __name__=='__main__':

    #R=[]
    #for x in arange(0,2,0.01):
    #    (E_p,E_m)=IntElfield(x,1.54,array([1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1]),array([0,80,20,80,20,80,20]))
    #    R.append(abs(E_m[-1]/E_p[-1])**2)

    theta=arange(0,1+0.001666670,0.001666670)
    #(E_p,E_m,Qj)=IntElfield(theta,1.54,array([1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1]),array([0,80,20,80,20,80,20]))
    #(E_p,E_m,Qj)=IntElfield(theta,1.540562,array([1-.15E-04/2+0.35E-06j/2,1-.45E-04/2+0.60E-05j/2,1]),array([0,1000]))
    from Paratt import Refl
    #c=Refl(theta,1.54,array([1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1]),array([80,20,80,20,80,20]),0)
    c=Refl(theta,1.54,array([1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1]),array([1000]),0)
    #gplt.plot(theta,log10(abs(E_m[-1]/E_p[-1])**2),theta,log10(abs(c**2)))
    #f=f=open('Specrefl.dat','r')
    #t=io.read_array(f)
    #t=transpose(t)
    #gplt.plot(theta,log10(abs(E_m[-1]/E_p[-1])**2),t[0],log10(t[1]))
    k=2*math.pi/1.54
    n=[1,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j]
    n=array(n)
    z=array([0,-1000])
    (T,R,k)=AmpElfield2(k*cos(theta*math.pi/180),k,n,z)
    gplt.plot(theta,log10(abs(R[0])**2),theta,log10(abs(c)**2))
