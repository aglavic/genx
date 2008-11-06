# Functions for instrumental resolution corrections and area of illumination
# these are to be included in the Models class and Refl class
# Note that these function assumes a Gaussian spread in both the angular as well
# as in real space

# THis one really needs scipy
from scipy import *
from numpy import *
from scipy.special import erf

# Area correction
def GaussArea(alpha,s1,s2,sigma_x):
    alpha=alpha*(abs(alpha)>1e-7)+1e-7*(abs(alpha)<=1e-7)
    sinalpha=sin(alpha*pi/180)
    A=sqrt(pi/2.0)*sigma_x*(erf(s2*sinalpha/sqrt(2.0)/sigma_x)+erf(s1*sinalpha/sqrt(2.0)/sigma_x))/sinalpha
    return A

# Intensity correction Gaussian beamprofile
def GaussIntensity(alpha,s1,s2,sigma_x):
    sinalpha=sin(alpha*pi/180)
    return (erf(s2*sinalpha/sqrt(2.0)/sigma_x)+erf(s1*sinalpha/sqrt(2.0)/sigma_x))/2.0

# Diffuse correction: Area corr
def GaussDiffCorrection(alpha,s1,s2,sigma_x):
    return GaussArea(alpha,s1,s2,sigma_x)#*GaussIntensity(alpha,s1,s2,sigma_x)

# Specular foorprintcorrections square beamprofile
def SquareIntensity(alpha,slen,beamwidth):
    F=slen/beamwidth*sin(alpha*pi/180)
    return where(F<=1.0,F,ones(F.shape))

# Function to calculate the instrumental resolution in incomming, alpha, and
# outgoing, beta, directions
def ResVariance(alpha,beta,sigma_x,sigma_xp,samplewidth,slitwidth,DetGCdist):
    alpha=alpha*pi/180.0
    beta=beta*pi/180.0
    sigma_sample2=samplewidth**2/12
    sigma_slit2=slitwidth**2/12
    sigma_t2=sigma_x**2*sigma_sample2*sin(beta)**2/(sigma_sample2*sin(alpha)**2+sigma_x**2)
    sigma_beta=sqrt(sigma_slit2+sigma_t2)/DetGCdist
    sigma_alpha=ones(beta.shape)*sigma_xp
    return (sigma_alpha,sigma_beta)

# Function to convolve the tth scan with a gaussian for simulating
# the instrumental resolution
def SpecularRes(tth,I,sigma_alpha,sigma_beta,points):
    # cutoff is between 0 and 1 and describes how much of the gaussian should
    # be incorporated in the resolution
    sigmainv2=(1/sigma_alpha**2+1/sigma_beta**2)/4
    #print sigmainv2
    #tthcutoff=max(sqrt(-2.0/sigmainv2*log(cutoff)))
    dtth=arange(-points,points+1)*(tth[1]-tth[0])
    Iconv=ones(I.shape)
    resfunc=exp(-(dtth*pi/180.0)**2/2*min(sigmainv2))
    resfunc=resfunc/sum(resfunc)
    #print sum(resfunc)
    Iconv=convolve(I,resfunc,mode=1)

    return Iconv

####################################################################
## Resolution Functions (Normal Distributed=Gaussian)
#####################################################################

#Full Convlutions - vayring resolution

# Function to create a 1D vector for the resolution with the 
# positions to calculate the reflectivity Qret and the weight
# of each point weight
# Inputs: Q - the Q values 
#         dQ - the resolution
#         points - the number of points for the convolution
#         range how far the gaussian should be convoluted
def ResolutionVector(Q,dQ,points,range=3):
    #if type(dQ)!=type(array([])):
    #    dQ=dQ*ones(Q.shape)
    Qstep=2*range*dQ/points
    Qres=Q+(arange(points)-(points-1)/2)[:,newaxis]*Qstep
    
    weight=1/sqrt(2*pi)/dQ*exp(-(transpose(Q[:,newaxis])-Qres)**2/(dQ)**2/2)
    Qret = Qres.flatten()#reshape(Qres,(1,Qres.shape[0]*Qres.shape[1]))[0]
    #print Qres
    #print Qres.shape
    #print Qret.shape
    return (Qret,weight)

# Include the resolution with Qret and weight calculated from ResolutionVector
# and I the calculated intensity at each point. returns the intensity
def ConvoluteResolutionVector(Qret,I,weight):
    Qret2 = Qret.reshape(weight.shape[0], weight.shape[1])
    #print Qret.shape,weight.shape
    I2 = I.reshape(weight.shape[0], weight.shape[1])
    #print (I*weight).shape,Qret.shape
    Int = integrate.trapz(I2*weight, x = Qret2, axis = 0)
    #print Int.shape
    return Int

# Fast convlution - constant resolution
# constant spacing between data!
def ConvoluteFast(Q,I,dQ,range=3):
    Qstep=Q[1]-Q[0]
    resvector=arange(-range*dQ,range*dQ+Qstep,Qstep)
    weight=1/sqrt(2*pi)/dQ*exp(-(resvector)**2/(dQ)**2/2)
    Iconv=convolve(r_[ones(resvector.shape)*I[0],I,ones(resvector.shape)*I[-1]], weight, mode=1)[resvector.shape[0]:-resvector.shape[0]]
    return Iconv

# Fast convolution - varying resolution
# constant spacing between the dat.
def ConvoluteFastVar(Q,I,dQ,range=3):
    Qstep=Q[1]-Q[0]
    steps=max(dQ*ones(Q.shape))*range/Qstep
    
    weight=1/sqrt(2*pi)/dQ*exp(-(Q[:,newaxis]-Q)**2/(dQ)**2/2)
    Itemp=I[:,newaxis]*ones(I.shape)
    Int=integrate.trapz(Itemp*weight,axis=0)
    return Int


# Add functions for determining s1,s2,sigma_x,sigma_xp

