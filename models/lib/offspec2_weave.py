
from scipy import *
from scipy import integrate
from scipy.misc import factorial
import scipy.weave as weave
import time
from elfield import *

def vec_realsymint(F,omega,eta,h,eta_z,qz_n,qz_np,sigma_n,sigma_np,max_n):

    #I=2*sum([((qz_n*conj(qz_np))**(2*n)*(sigma_n*sigma_np)**n)/factorial(n)*eta*F(omega*eta/(n**(1.0/2/h))) for n in arange(1.0,max_n+1.0)])
    I=2*sum([((qz_n*conj(qz_np)*(sigma_n*sigma_np))**n)/factorial(n)*eta/n**(1.0/2.0/h)*F(omega*eta/(n**(1.0/2.0/h))) for n in arange(1.0,max_n+1.0)])
    I=array(I)
    print I.shape
    return I

def make_F(q,h):
    def f(x):
        return exp(-abs(x)**(2*h))
    #N=100
    #ws=workspace(N)
    #cws=workspace(N)
    #abserr=1e-9
    #func=gsl_function(f,h)
    q=q*(abs(q)>0.003)+0.003*(abs(q)<0.003)
    I=[integrate.quad(f,0,inf,weight='cos',wvar=x)[0] for x in q]
    #I=[qawf(func,0,abserr,N,ws,cws,qawo_table(x,0,COSINE,N)) for x in q]
    I=array(I)
    F=I
    return F

def DWBA_Interdiff(qx,qz,lamda,n,z,sigma,sigmaid,eta,h,eta_z,d=[0],\
                    taylor_n = 1):
    #sigmaid-interdiffused part
    #sigma-roughness
    k=2*math.pi/lamda
    qx=array(qx,dtype=float)
    sqn=n**2
    sqn=array(sqn,dtype=complex)
    #z=array(z,dtype=complex)
    #eta=complex(eta)
    #h=complex(h)
    #eta_z=complex(eta_z)
    sigma=array(sigma,dtype=complex)
    sigmaid=array(sigmaid,dtype=complex)
    #print 'Setup Complete'
    # Calculating electrical fields
    #print 'Calculating electrical fields...'
    omega=arctan(qx/qz) # Not the real omega given by Fewster
    omegap=arcsin(sqrt(qx**2+qz**2)/2/k)
    #print 2*omegap*180/pi
    #print omega*180/pi
    kxi=k*cos(omega+omegap)
    kxf=k*cos(omegap-omega)
    tn=list(n)
    tn.reverse()
    tn=array(tn)
    td=list(d[1:])
    td.reverse()
    td=array(td)
    td=r_[0,td]
    # Changed functin to calculate the electric fields
    t1=time.clock()
    
    (T,R,kz)=AmpElfield_q(k,r_[kxi,kxf],lamda,tn,td)
    t2=time.clock()
    #print 'AmpElfield_q'
    #print t2-t1
    
    lenq=len(qx)

    Ri=R[:,:lenq]
    Rf=R[:,lenq:]
    Ti=T[:,:lenq]
    Tf=T[:,lenq:]
    ki=kz[:,:lenq]
    kf=kz[:,lenq:]
    #print 'Done'
    G=array([Ti*Tf,Ti*Rf,Ri*Tf,Ri*Rf])
    q=array([ki+kf,ki-kf,-ki+kf,-ki-kf])
    #print q.shape
    #print G.shape
    # Setting up for the Fourier integral as given by Pape et.al.
    #print 'Prepearing the Fourier integral'
    maxn = taylor_n
    #print 'n= ', maxn
    # New dqmin tested 20051201
    dqmin=min(abs(qx[:-1]-qx[1:]))*eta/(maxn**(1.0/2.0/h))
    #dqmin=(qx[1]-qx[0])*eta/(maxn**(1.0/2.0/h))    
    #print 'dqmin = ',dqmin
    if abs(dqmin) < 1e-12:
        dqmin=1e-3
        #print 'Hej'
    #print 'Test'
    # New q_min tested 20051201
    q_min=arange(min(abs(qx))*eta/(maxn**(1.0/2.0/h))-dqmin,max(abs(qx))*eta+2*dqmin,dqmin,dtype=float)
    #print qx[0]
    #print qx[-1]
    #q_min=arange(qx[0]*eta/(maxn**(1.0/2.0/h)),qx[-1]*eta+2*dqmin,dqmin,typecode=Float64)
    #print q_min
    
    #print q_min.shape
    t1=time.clock()
    table=make_F(q_min,h)
    t2=time.clock()
    #print 'make_F '
    #print t2-t1
    table=array(table,dtype=complex)
    #print table
    G=array(G,dtype=complex)
    q=array(q,dtype=complex)
    #print 'Done'
    I=zeros(qx.shape)*(0.0+0.0J)
    I=array(I,dtype=complex)
    Im=1J
    fn=factorial(arange(1,maxn+1))
    #print fn
    s=array([0.0J],dtype=complex)
    t1=time.clock()
    code="""
    int Layers=NG[1];
    int Points=Nqx[0];
    int nind=0;
    int npind=0;
    double xnew=0;
    int lower=0;
    for(int m=0;m < Points;m++){
        for(int i=0;i < Layers;i++){
            for(int j=0;j < Layers;j++){
                for(int k=0;k < 4; k++){
                    for(int l=0;l < 4; l++){
                        nind=(k*Layers+i)*Points+m;
                        npind=(l*Layers+j)*Points+m;
                        for(int p=0; p < Nfn[0]; p++){
                            //added abs sigan 051201
                            xnew=fabs(qx[m]*eta/pow(p+1.0,1.0/2.0/h));
                            //xnew=qx[m]*eta/pow(p+1.0,1.0/2.0/h);
                            lower=int((xnew-q_min[0])/(q_min[1]-q_min[0]));
                            s[0]+=2.0*pow(q[nind]*conj(q[npind])*sigma[i]*sigma[j]*exp(-abs(z[i]-z[j])/eta_z),p+1.0)/fn[p]*eta/pow(p+1.0,1.0/2.0/h)*((table[lower+1]-table[lower])/(q_min[lower+1]-q_min[lower])*(xnew-q_min[lower])+table[lower]);
                            
                        }
                        
                        I[m]+=(sqn[i]-sqn[i+1])*conj(sqn[j]-sqn[j+1])*G[nind]*exp(-0.5*pow(sigmaid[i]*q[nind],2))*conj(G[npind]*exp(-0.5*pow(sigmaid[j]*q[npind],2)))*exp(-0.5*(pow(q[nind]*sigma[i],2)+pow(conj(q[npind])*sigma[j],2)))/q[nind]/conj(q[npind])*s[0];
                        //*exp(-Im*q[nind]*z[i]-conj(Im*q[npind])*z[j])
                        s[0]=Im-Im;
                        
                    }
                }
            }
        }
    }

    """    #eta,h,eta_z,qn,qnp,sigma_n,sigma_np
    weave.inline(code,['qx','G','q','eta','h','sigma','sigmaid','sqn','z','I','Im','table','q_min','fn','s','eta_z'],compiler='gcc')
    #print s
    t2=time.clock()
    #print 'code'
    #print t2-t1
    s=I
    return (s,omega+omegap,omegap-omega)



def DWBA(qx,qz,lamda,n,z,sigma,eta,h,eta_z,d=[0], taylor_n = 1):
    k=2*math.pi/lamda
    qx=array(qx,dtype=float64)
    sqn=n**2
    sqn=array(sqn,dtype=complex)
    #print 'Setup Complete'
    # Calculating electrical fields
    #print 'Calculating electrical fields...'
    omega=arctan(qx/qz) # Not the real omega given by Fewster
    omegap=arcsin(sqrt(qx**2+qz**2)/2/k)
    #print 2*omegap*180/pi
    #print omega*180/pi
    kxi=k*cos(omega+omegap)
    kxf=k*cos(omegap-omega)
    tn=list(n)
    tn.reverse()
    tn=array(tn)
    td=list(d[1:])
    td.reverse()
    td=array(td)
    td=r_[0,td]
    # Changed functin to calculate the electric fields
    t1=time.clock()
    #(T,R,kz)=AmpElfield2(r_[kxi,-kxf],k,n,z)
    (T,R,kz)=AmpElfield_q(k,r_[kxi,kxf],lamda,tn,td)
    t2=time.clock()
    #print 'AmpElfield_q'
    #print t2-t1
    #print R
    #print T
    #print kz
    lenq=len(qx)
    #print lenq
    #N=len(d)
    #print R.shape
    # pick out Amplitudes j+1 Needed for AmpEfield2
    #R=R[1:,:]
    #T=T[1:,:]
    #kz=kz[1:,:]
    #print R.shape
    Ri=R[:,:lenq]
    Rf=R[:,lenq:]
    Ti=T[:,:lenq]
    Tf=T[:,lenq:]
    ki=kz[:,:lenq]
    kf=kz[:,lenq:]
    #print 'Done'
    G=array([Ti*Tf,Ti*Rf,Ri*Tf,Ri*Rf])
    q=array([ki+kf,ki-kf,-ki+kf,-ki-kf])
    print q.shape
    print G.shape
    # Setting up for the Fourier integral as given by Pape et.al.
    #print 'Prepearing the Fourier integral'
    maxn = taylor_n
    #print 'n= ', maxn
    dqmin=(qx[1]-qx[0])*eta/(maxn**(1.0/2.0/h))    
    print dqmin
    #if abs(dqmin) < 1e-12:
    #    dqmin=1e-3
    #    print 'Hej'
    #print 'Test'
    q_min=arange(qx[0]*eta/(maxn**(1.0/2.0/h)),qx[-1]*eta+2*dqmin,dqmin,dtype=float64)
    
    #print q_min.shape
    t1=time.clock()
    table=make_F(q_min,h)
    t2=time.clock()
    #print 'make_F '
    #print t2-t1
    table=array(table,dtype=complex)
    G=array(G,dtype=complex)
    q=array(q,dtype=complex)
    #print 'Done'
    I=zeros(qx.shape)*(0.0+0.0J)
    I=array(I,dtype=complex)
    Im=1J
    fn=factorial(arange(1,maxn+1))
    #print fn
    s=array([0.0J],dtype=complex)
    t1=time.clock()
    code="""
    int Layers=NG[1];
    int Points=Nqx[0];
    int nind=0;
    int npind=0;
    double xnew=0;
    int lower=0;
    for(int m=0;m < Points;m++){
        for(int i=0;i < Layers;i++){
            for(int j=0;j < Layers;j++){
                for(int k=0;k < 4; k++){
                    for(int l=0;l < 4; l++){
                        nind=(k*Layers+i)*Points+m;
                        npind=(l*Layers+j)*Points+m;
                        for(int p=0; p < Nfn[0]; p++){
                            xnew=qx[m]*eta/pow(p+1.0,1.0/2.0/h);
                            lower=int((xnew-q_min[0])/(q_min[1]-q_min[0]));
                            s[0]+=2.0*pow(q[nind]*conj(q[npind])*sigma[i]*sigma[j]*exp(-abs(z[i]-z[j])/eta_z),p+1.0)/fn[p]*eta/pow(p+1.0,1.0/2.0/h)*((table[lower+1]-table[lower])/(q_min[lower+1]-q_min[lower])*(xnew-q_min[lower])+table[lower]);
                            
                        }
                        
                        I[m]+=(sqn[i]-sqn[i+1])*conj(sqn[j]-sqn[j+1])*G[nind]*conj(G[npind])*exp(-0.5*(pow(q[nind]*sigma[i],2.0)+pow(conj(q[npind])*sigma[j],2.0)))/q[nind]/conj(q[npind])*s[0];
                        //*exp(-Im*q[nind]*z[i]-conj(Im*q[npind])*z[j])
                        s[0]=Im-Im;
                        
                    }
                }
            }
        }
    }

    """
    #eta,h,eta_z,qn,qnp,sigma_n,sigma_np
    weave.inline(code,['qx','G','q','eta','h','sigma','sqn','z','I','Im','table','q_min','fn','s','eta_z'],compiler='gcc')
    t2=time.clock()
    #print 'code'
    #print t2-t1
    s=I
    return (s,omega+omegap,omegap-omega)

def Born(qx,qz,lamda,n,z,sigma,eta,h,eta_z,d=[0], taylor_n = 1):
    k=2*math.pi/lamda
    qx=array(qx,dtype=float64)
    sqn=n**2
    sqn=array(sqn,dtype=complex)
    #print 'Setup Complete'
    # Calculating electrical fields
    #print 'Calculating electrical fields...'
    omega=arctan(qx/qz) # Not the real omega given by Fewster
    omegap=arcsin(sqrt(qx**2+qz**2)/2/k)
    #print 2*omegap*180/pi
    #print omega*180/pi
    #kxi=k*cos(omega+omegap)
    #kxf=k*cos(omegap-omega)
    #tn=list(n)
    #tn.reverse()
    #tn=array(tn)
    #td=list(d[1:])
    #td.reverse()
    #td=array(td)
    #td=r_[0,td]
    # Changed functin to calculate the electric fields
    #t1=time.clock()
    #(T,R,kz)=AmpElfield2(r_[kxi,-kxf],k,n,z)
    #(T,R,kz)=AmpElfield_q(k,r_[kxi,kxf],lamda,tn,td)
    #t2=time.clock()
    #print 'AmpElfield2'
    #print t2-t1
    #print R
    #print T
    #print kz
    lenq=len(qx)
    #print lenq
    #N=len(d)
    #print R.shape
    # pick out Amplitudes j+1 Needed for AmpEfield2
    #R=R[1:,:]
    #T=T[1:,:]
    #kz=kz[1:,:]
    #print R.shape
    #Ri=R[:,:lenq]
    #Rf=R[:,lenq:]
    #Ti=T[:,:lenq]
    #Tf=T[:,lenq:]
    #ki=kz[:,:lenq]
    #kf=kz[:,lenq:]
    #print 'Done'
    #G=array([Ti*Tf,Ti*Rf,Ri*Tf,Ri*Rf])
    #q=array([ki+kf,ki-kf,-ki+kf,-ki-kf])
    #print q.shape
    #print G.shape
    # Setting up for the Fourier integral as given by Pape et.al.
    #print 'Prepearing the Fourier integral'
    maxn=taylor_n
    dqmin=(qx[1]-qx[0])*eta/(maxn**(1.0/2.0/h))
    q_min=arange(qx[0]*eta/(maxn**(1.0/2.0/h)),qx[-1]*eta+2*dqmin,dqmin,dtype=float64)
    #print q_min.shape
    t1=time.clock()
    table=make_F(q_min,h)
    t2=time.clock()
    #print 'make_F '
    #print t2-t1
    table=array(table,dtype=complex)
    #G=array(G,typecode=complex)
    #q=array(q,typecode=complex)
    #print 'Done'
    I=zeros(qx.shape)*(0.0+0.0J)
    I=array(I,dtype=complex)
    Im=1J
    fn=factorial(arange(1,maxn+1))
    #print fn
    s=array([0.0J],dtype=complex)
    t1=time.clock()
    code="""
    int Layers=Nsigma[0];
    int Points=Nqx[0];
    double xnew=0;
    int lower=0;
    for(int m=0;m < Points;m++){
        for(int i=0;i < Layers;i++){
            for(int j=0;j < Layers;j++){
                for(int p=0; p < Nfn[0]; p++){
                    xnew=qx[m]*eta/pow(p+1.0,1.0/2.0/h);
                    lower=int((xnew-q_min[0])/(q_min[1]-q_min[0]));
                    s[0]+=2.0*pow(qz[m]*qz[m]*sigma[i]*sigma[j]*exp(-abs(z[i]-z[j])/eta_z),p+1.0)/fn[p]*eta/pow(p+1.0,1.0/2.0/h)*((table[lower+1]-table[lower])/(q_min[lower+1]-q_min[lower])*(xnew-q_min[lower])+table[lower]);
                            
                }
                        
                I[m]+=(sqn[i]-sqn[i+1])*conj(sqn[j]-sqn[j+1])*exp(-0.5*(pow(qz[m]*sigma[i],2.0)+pow(qz[m]*sigma[j],2.0)))/qz[m]/qz[m]*s[0];
                s[0]=Im-Im;
                        
            }
        }
    }
    """
    #eta,h,eta_z,qn,qnp,sigma_n,sigma_np
    weave.inline(code,['qx','qz','eta','h','sigma','sqn','z','I','Im','table','q_min','fn','s','eta_z'],compiler='gcc')
    t2=time.clock()
    #print 'code'
    #print t2-t1
    s=I
    return (s,omega+omegap,omegap-omega)
    
def saveOrigin(filename):
    f=open(filename,'w')
    io.write_array(f,transpose(array([qx,real(I1)])))
    

if __name__=='__main__':

    n=array([1]+[1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j]*10+[1-7.57e-6+1.73e-7j])
    #n=array([1,1-2.25e-5+2.9e-6j,1-7.5e-6+1.75e-7j])
    #n=array([1,1-2.25e-5,1-7.5e-6])
    d=array([0.0]+[80.0,20.0]*10)
    #d=array([0.0,1000.0])
    z=-cumsum(d)
    sigma=array([5.0]+[5.0,5.0]*10)
    #sigma=array([3.0,3.0])

    qx=arange(.1e-4,0.005,0.00002)
    qz=0.2*ones(qx.shape)
    
    #qx=arange(1e-5,3e-5,1e-5)
    eta=200.0
    eta_z=1.0
    h=1.0
    
    (I1,ain,aout)=Born(qx,qz,1.540562,n,z,sigma,eta,h,eta_z)
    (I2,ain,aout)=DWBA(qx,qz,1.540562,n,z,sigma,eta,h,eta_z,d)
    gplt.plot(qx,real(I1),qx,real(I2))
    
    
    



    





    
