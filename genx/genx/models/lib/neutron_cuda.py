from numpy import *
from numba import cuda
import numba
import math, cmath

if numba.__version__.split('.')>['0', '55', '0']:
    JIT_OPTIONS = dict(cache=True)
else:
    JIT_OPTIONS = {}

##################### not yet correct ###############################

@cuda.jit(numba.void(numba.complex128[:, ::1], numba.complex128[:, ::1], numba.complex128[:, ::1]),
          device=True, **JIT_OPTIONS)
def dot4(A, B, D):
    D[0, 0]=(A[0, 0]*B[0, 0]+A[0, 1]*B[1, 0]+A[0, 2]*B[2, 0]+
             A[0, 3]*B[3, 0])
    D[0, 1]=(A[0, 0]*B[0, 1]+A[0, 1]*B[1, 1]+A[0, 2]*B[2, 1]+
             A[0, 3]*B[3, 1])
    D[0, 2]=(A[0, 0]*B[0, 2]+A[0, 1]*B[1, 2]+A[0, 2]*B[2, 2]+
             A[0, 3]*B[3, 2])
    D[0, 3]=(A[0, 0]*B[0, 3]+A[0, 1]*B[1, 3]+A[0, 2]*B[2, 3]+
             A[0, 3]*B[3, 3])

    D[1, 0]=(A[1, 0]*B[0, 0]+A[1, 1]*B[1, 0]+A[1, 2]*B[2, 0]+
             A[1, 3]*B[3, 0])
    D[1, 1]=(A[1, 0]*B[0, 1]+A[1, 1]*B[1, 1]+A[1, 2]*B[2, 1]+
             A[1, 3]*B[3, 1])
    D[1, 2]=(A[1, 0]*B[0, 2]+A[1, 1]*B[1, 2]+A[1, 2]*B[2, 2]+
             A[1, 3]*B[3, 2])
    D[1, 3]=(A[1, 0]*B[0, 3]+A[1, 1]*B[1, 3]+A[1, 2]*B[2, 3]+
             A[1, 3]*B[3, 3])

    D[2, 0]=(A[2, 0]*B[0, 0]+A[2, 1]*B[1, 0]+A[2, 2]*B[2, 0]+
             A[2, 3]*B[3, 0])
    D[2, 1]=(A[2, 0]*B[0, 1]+A[2, 1]*B[1, 1]+A[2, 2]*B[2, 1]+
             A[2, 3]*B[3, 1])
    D[2, 2]=(A[2, 0]*B[0, 2]+A[2, 1]*B[1, 2]+A[2, 2]*B[2, 2]+
             A[2, 3]*B[3, 2])
    D[2, 3]=(A[2, 0]*B[0, 3]+A[2, 1]*B[1, 3]+A[2, 2]*B[2, 3]+
             A[2, 3]*B[3, 3])

    D[3, 0]=(A[3, 0]*B[0, 0]+A[3, 1]*B[1, 0]+A[3, 2]*B[2, 0]+
             A[3, 3]*B[3, 0])
    D[3, 1]=(A[3, 0]*B[0, 1]+A[3, 1]*B[1, 1]+A[3, 2]*B[2, 1]+
             A[3, 3]*B[3, 1])
    D[3, 2]=(A[3, 0]*B[0, 2]+A[3, 1]*B[1, 2]+A[3, 2]*B[2, 2]+
             A[3, 3]*B[3, 2])
    D[3, 3]=(A[3, 0]*B[0, 3]+A[3, 1]*B[1, 3]+A[3, 2]*B[2, 3]+
             A[3, 3]*B[3, 3])

@cuda.jit(numba.void(numba.float64[:], numba.complex128[:], numba.complex128[:],
                     numba.float64[:], numba.float64[:], numba.float64[:],
                     numba.float64[:, ::1], numba.complex128[:, :, :, ::1]), **JIT_OPTIONS)
def ReflNBSigma(Q, Vp, Vm, d, M_ang, sigma, Rout, PX):
    '''A quicker implementation than the ordinary slow implementaion in Refl
    Calculates spin-polarized reflectivity according to S.J. Blundell
        and J.A.C. Bland Phys rev. B. vol 46 3391 (1992)
        The algorithm assumes that the first element in the arrays represents
        the substrate and the last the ambient layer.
        Input parameters:   Q : Scattering vector in reciprocal
                                angstroms Q=4*pi/lambda *sin(theta)
                            Vp: Neutron potential for spin up
                            Vm: Neutron potential for spin down
                            d: layer thickness
                            M_ang: Angle of the magnetic
                                    moment(radians!) M_ang=0 =>M//neutron spin
                            sigma: The roughness of the upper interface.
                            return_int: Flag for returning the instensity, default=True. If False return the amplitudes.
        Returns:            (Ruu,Rdd,Rud,Rdu)
                            (up-up,down-down,up-down,down-up)
    '''
    layers=Vp.shape[0]

    # Thread id in a 1D block
    tx=cuda.threadIdx.x
    # Block id in a 1D grid
    ty=cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw=cuda.blockDim.x
    # Compute flattened index inside the array
    qi=tx+ty*bw
    if qi>=Q.shape[0]:
        return

    X=cuda.local.array((4, 4), dtype=numba.complex128)
    P=cuda.local.array((4,), dtype=numba.complex128)
    M=cuda.local.array((4, 4), dtype=numba.complex128)
    Mtmp=cuda.local.array((4, 4), dtype=numba.complex128)

    # Assume first element=substrate and last=ambient!
    k_amb2=(Q[qi]/2.0)**2

    k_pi=cmath.sqrt(k_amb2-Vp[0])
    k_mi=cmath.sqrt(k_amb2-Vm[0])

    for lj in range(1, layers):
        # Wavevectors in the layers
        k_pj=cmath.sqrt(k_amb2-Vp[lj])
        k_mj=cmath.sqrt(k_amb2-Vm[lj])

        theta_diff=M_ang[lj]-M_ang[lj-1]

        ##### ass_X ####
        # Angular difference between the magnetization
        # Assemble the interface reflectivity matrix
        costd=math.cos(theta_diff/2.0)
        sintd=math.sin(theta_diff/2.0)
        X[0, 0]=costd*(k_pi+k_pj)/2./k_pj
        X[0, 1]=-costd*(k_pi-k_pj)/2./k_pj
        X[0, 2]=sintd*(k_pj+k_mi)/2./k_pj
        X[0, 3]=sintd*(k_pj-k_mi)/2./k_pj
        #X[1, 0]=X[0, 1]  # -(costd*(k_pj1 - k_pj))/(2*k_pj)
        #X[1, 1]=X[0, 0]  # (costd*(k_pj1 + k_pj))/(2*k_pj)
        #X[1, 2]=X[0, 3]  # (sintd*(k_pj - k_mj1))/(2*k_pj)
        #X[1, 3]=X[0, 2]  # (sintd*(k_pj + k_mj1))/(2*k_pj)
        X[2, 0]=-(sintd*(k_pi+k_mj))/(2.*k_mj)
        X[2, 1]=(sintd*(k_pi-k_mj))/(2.*k_mj)
        X[2, 2]=(costd*(k_mi+k_mj))/(2.*k_mj)
        X[2, 3]=-(costd*(k_mi-k_mj))/(2.*k_mj)
        #X[3, 0]=X[2, 1]  # (sintd*(k_pj1 - k_mj))/(2*k_mj)
        #X[3, 1]=X[2, 0]  # -(sintd*(k_pj1 + k_mj))/(2*k_mj)
        #X[3, 2]=X[2, 3]  # -(costd*(k_mj1 - k_mj))/(2*k_mj)
        #X[3, 3]=X[2, 2]  # (costd*(k_mj1 + k_mj))/(2*k_mj)

        ##### include_sigma #####
        sigma2=sigma[lj-1]**2/2.0
        X[0, 0]*=cmath.exp(-(k_pj-k_pi)**2*sigma2)
        X[0, 1]*=cmath.exp(-(k_pj+k_pi)**2*sigma2)
        X[0, 2]*=cmath.exp(-(k_pj-k_mi)**2*sigma2)
        X[0, 3]*=cmath.exp(-(k_pj+k_mi)**2*sigma2)
        X[1, 0]=X[0, 1]  # X[1,0]*w(k_pj + k_pj1, sigma2)
        X[1, 1]=X[0, 0]  # X[1,1]*w(k_pj - k_pj1, sigma2)
        X[1, 2]=X[0, 3]  # X[1,2]*w(k_pj + k_mj1, sigma2)
        X[1, 3]=X[0, 2]  # X[1,3]*w(k_pj - k_mj1, sigma2)
        X[2, 0]*=cmath.exp(-(k_mj-k_pi)**2*sigma2)
        X[2, 1]*=cmath.exp(-(k_mj+k_pi)**2*sigma2)
        X[2, 2]*=cmath.exp(-(k_mj-k_mi)**2*sigma2)
        X[2, 3]*=cmath.exp(-(k_mj+k_mi)**2*sigma2)
        X[3, 0]=X[2, 1]  # X[3,0]*w(k_mj + k_pj1, sigma)
        X[3, 1]=X[2, 0]  # X[3,1]*w(k_mj - k_pj1, sigma)
        X[3, 2]=X[2, 3]  # X[3,2]*w(k_mj + k_mj1, sigma)
        X[3, 3]=X[2, 2]  # X[3,3]*w(k_mj - k_mj1, sigma)

        ##### ass_P ####
        P[0]=cmath.exp(-1.0J*k_pj*d[lj])
        P[1]=1./P[0]  # exp(1.0J*k_p*d)
        P[2]=cmath.exp(-1.0J*k_mj*d[lj])
        P[3]=1./P[2]  # exp(1.0J*k_m*d)

        # Assemble the layer propagation matrices

        # Multiply the propagation matrices with the interface matrix
        for i in range(4):
            for j in range(4):
                PX[qi, lj, i, j]=P[i]*X[i, j]
        k_pi=k_pj
        k_mi=k_mj

    ##### ass_P ####
    for i in range(4):
        for j in range(4):
            M[i, j]=PX[qi, -2, i, j]
    for linv in range(layers-3):
        # Multiply up the sample matrix
        dot4(M, PX[qi, layers-3-linv], Mtmp)
        for i in range(4):
            for j in range(4):
                M[i, j]=Mtmp[i, j]
    dot4(X, M, Mtmp)
    for i in range(4):
        for j in range(4):
            M[i, j]=Mtmp[i, j]

    denom=M[0, 0]*M[2, 2]-M[0, 2]*M[2, 0]
    Ruu=(M[1, 0]*M[2, 2]-M[1, 2]*M[2, 0])/denom
    Rud=(M[3, 0]*M[2, 2]-M[3, 2]*M[2, 0])/denom
    Rdu=(M[1, 2]*M[0, 0]-M[1, 0]*M[0, 2])/denom
    Rdd=(M[3, 2]*M[0, 0]-M[3, 0]*M[0, 2])/denom
    Rout[0, qi]=abs(Ruu)**2
    Rout[1, qi]=abs(Rdd)**2
    Rout[2, qi]=abs(Rud)**2
    Rout[3, qi]=abs(Rdu)**2

@cuda.jit(numba.void(numba.float64[:], numba.complex128[:], numba.complex128[:],
                     numba.float64[:], numba.float64[:],
                     numba.float64[:, ::1], numba.complex128[:, :, :, ::1]), **JIT_OPTIONS)
def ReflNB(Q, Vp, Vm, d, M_ang, Rout, PX):
    '''A quicker implementation than the ordinary slow implementaion in Refl
    Calculates spin-polarized reflectivity according to S.J. Blundell
        and J.A.C. Bland Phys rev. B. vol 46 3391 (1992)
        The algorithm assumes that the first element in the arrays represents
        the substrate and the last the ambient layer.
        Input parameters:   Q : Scattering vector in reciprocal
                                angstroms Q=4*pi/lambda *sin(theta)
                            Vp: Neutron potential for spin up
                            Vm: Neutron potential for spin down
                            d: layer thickness
                            M_ang: Angle of the magnetic
                                    moment(radians!) M_ang=0 =>M//neutron spin
                            sigma: The roughness of the upper interface.
                            return_int: Flag for returning the instensity, default=True. If False return the amplitudes.
        Returns:            (Ruu,Rdd,Rud,Rdu)
                            (up-up,down-down,up-down,down-up)
    '''
    layers=Vp.shape[0]

    # Thread id in a 1D block
    tx=cuda.threadIdx.x
    # Block id in a 1D grid
    ty=cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw=cuda.blockDim.x
    # Compute flattened index inside the array
    qi=tx+ty*bw
    if qi>=Q.shape[0]:
        return

    X=cuda.local.array((4, 4), dtype=numba.complex128)
    P=cuda.local.array((4,), dtype=numba.complex128)
    M=cuda.local.array((4, 4), dtype=numba.complex128)
    Mtmp=cuda.local.array((4, 4), dtype=numba.complex128)

    # Assume first element=substrate and last=ambient!
    k_amb2=(Q[qi]/2.0)**2

    k_pi=cmath.sqrt(k_amb2-Vp[0])
    k_mi=cmath.sqrt(k_amb2-Vm[0])

    for lj in range(1, layers):
        # Wavevectors in the layers
        k_pj=cmath.sqrt(k_amb2-Vp[lj])
        k_mj=cmath.sqrt(k_amb2-Vm[lj])

        theta_diff=M_ang[lj]-M_ang[lj-1]

        ##### ass_X ####
        # Angular difference between the magnetization
        # Assemble the interface reflectivity matrix
        costd=math.cos(theta_diff/2.0)
        sintd=math.sin(theta_diff/2.0)
        X[0, 0]=costd*(k_pi+k_pj)/2./k_pj
        X[0, 1]=-costd*(k_pi-k_pj)/2./k_pj
        X[0, 2]=sintd*(k_pj+k_mi)/2./k_pj
        X[0, 3]=sintd*(k_pj-k_mi)/2./k_pj
        X[1, 0]=X[0, 1]  # -(costd*(k_pj1 - k_pj))/(2*k_pj)
        X[1, 1]=X[0, 0]  # (costd*(k_pj1 + k_pj))/(2*k_pj)
        X[1, 2]=X[0, 3]  # (sintd*(k_pj - k_mj1))/(2*k_pj)
        X[1, 3]=X[0, 2]  # (sintd*(k_pj + k_mj1))/(2*k_pj)
        X[2, 0]=-(sintd*(k_pi+k_mj))/(2.*k_mj)
        X[2, 1]=(sintd*(k_pi-k_mj))/(2.*k_mj)
        X[2, 2]=(costd*(k_mi+k_mj))/(2.*k_mj)
        X[2, 3]=-(costd*(k_mi-k_mj))/(2.*k_mj)
        X[3, 0]=X[2, 1]  # (sintd*(k_pj1 - k_mj))/(2*k_mj)
        X[3, 1]=X[2, 0]  # -(sintd*(k_pj1 + k_mj))/(2*k_mj)
        X[3, 2]=X[2, 3]  # -(costd*(k_mj1 - k_mj))/(2*k_mj)
        X[3, 3]=X[2, 2]  # (costd*(k_mj1 + k_mj))/(2*k_mj)

        ##### ass_P ####
        P[0]=cmath.exp(-1.0J*k_pj*d[lj])
        P[1]=1./P[0]  # exp(1.0J*k_p*d)
        P[2]=cmath.exp(-1.0J*k_mj*d[lj])
        P[3]=1./P[2]  # exp(1.0J*k_m*d)

        # Assemble the layer propagation matrices

        # Multiply the propagation matrices with the interface matrix
        for i in range(4):
            for j in range(4):
                PX[qi, lj, i, j]=P[i]*X[i, j]
        k_pi=k_pj
        k_mi=k_mj

    ##### ass_P ####
    for i in range(4):
        for j in range(4):
            M[i, j]=PX[qi, -2, i, j]
    for linv in range(layers-3):
        # Multiply up the sample matrix
        dot4(M, PX[qi, layers-3-linv], Mtmp)
        for i in range(4):
            for j in range(4):
                M[i, j]=Mtmp[i, j]
    dot4(X, M, Mtmp)
    for i in range(4):
        for j in range(4):
            M[i, j]=Mtmp[i, j]

    denom=M[0, 0]*M[2, 2]-M[0, 2]*M[2, 0]
    Ruu=(M[1, 0]*M[2, 2]-M[1, 2]*M[2, 0])/denom
    Rud=(M[3, 0]*M[2, 2]-M[3, 2]*M[2, 0])/denom
    Rdu=(M[1, 2]*M[0, 0]-M[1, 0]*M[0, 2])/denom
    Rdd=(M[3, 2]*M[0, 0]-M[3, 0]*M[0, 2])/denom
    Rout[0, qi]=abs(Ruu)**2
    Rout[1, qi]=abs(Rdd)**2
    Rout[2, qi]=abs(Rud)**2
    Rout[3, qi]=abs(Rdu)**2

def Refl(Q, Vp, Vm, d, M_ang, sigma=None, return_int=True):
    if M_ang[-1]!=0:
        raise ValueError("The magnetization in the ambient layer has to be in polarization direction")
    if Vp[-1]!=0 or Vm[-1]!=0:
        # Ambient not vacuum
        raise ValueError("The SLD in the ambient layer has to be zero, apply renormalization first")
    if len(Vp)==2:
        # Algorithm breaks without a layer, so add an empty one
        Vp=hstack([Vp, [Vp[-1]]])
        Vm=hstack([Vm, [Vm[-1]]])
        M_ang=array([M_ang[0], 0., 0.], dtype=float64)
        d=array([d[0], 10., d[1]], dtype=float64)
        if sigma is not None:
            sigma=array([sigma[0], sigma[0], sigma[1]], dtype=float64)

    threadsperblock=32
    blockspergrid=(Q.shape[0]+(threadsperblock-1))//threadsperblock

    CQ=cuda.to_device(Q)
    CVp=cuda.to_device(Vp.astype(complex128))
    CVm=cuda.to_device(Vm.astype(complex128))
    Cd=cuda.to_device(d)
    CM_ang=cuda.to_device(M_ang)
    CRout=cuda.device_array(shape=(4, Q.shape[0]), dtype=float64)
    CPX=cuda.device_array(shape=(Q.shape[0], Vp.shape[0], 4, 4), dtype=complex128)

    if sigma is not None:
        Csigma=cuda.to_device(sigma)
        ReflNBSigma[blockspergrid, threadsperblock](CQ, CVp, CVm, Cd, CM_ang, Csigma, CRout, CPX)
    else:
        ReflNB[blockspergrid, threadsperblock](CQ, CVp, CVm, Cd, CM_ang, CRout, CPX)

    return CRout.copy_to_host()
