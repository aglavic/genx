from numpy import append, arange, array, c_, cos, exp, log10, newaxis, ones, pi, r_, rollaxis, sin, sqrt

from genx.core.custom_logging import iprint


def IntElfield(theta, lamda, n, d):
    # Length of k-vector in vaccum
    k = 2 * pi / lamda
    # Calculates the wavevector in each layer
    Qj = 2 * k * sqrt(n[:, newaxis] ** 2 - cos(theta * pi / 180) ** 2)
    # print Qj
    # Fresnel reflectivity for the interfaces
    rp = (Qj[1:] - Qj[:-1]) / (Qj[1:] + Qj[:-1])
    tp = 1 + rp
    # tp=2*Qj[1:]/(Qj[1:]+Qj[:-1])
    # print tp.shape
    # Beta=k*d*n[:-1]*sin(theta*pi/180)
    # print Beta.shape
    D = d[:, newaxis] * ones(theta.shape)
    # E_p=[1]
    # E_m=[0]
    E_p = [1 * ones(theta.shape)]
    E_m = [0 * ones(theta.shape)]
    # print E_p[0].shape
    # print D[0].shape
    # print Qj[0].shape
    # for i in range(0,len(rp)):
    #    E_p.append(1/tp[i]*(E_p[i]*exp(-1.0J*D[i]*Qj[i]/2)+E_m[i]*rp[i]*exp(1.0J*D[i]*Qj[i]/2)))
    #    E_m.append(1/tp[i]*(E_m[i]*exp(1.0J*D[i]*Qj[i]/2)+E_p[i]*rp[i]*exp(-1.0J*D[i]*Qj[i]/2)))
    for i in range(0, len(rp)):
        E_p.append(1 / tp[i] * (E_p[i] * exp(-1.0j * D[i] * Qj[i] / 2) + E_m[i] * rp[i] * exp(1.0j * D[i] * Qj[i] / 2)))
        E_m.append(1 / tp[i] * (E_m[i] * exp(1.0j * D[i] * Qj[i] / 2) + E_p[i] * rp[i] * exp(-1.0j * D[i] * Qj[i] / 2)))
    # print E_p
    E_p = array(E_p[1:])
    E_m = array(E_m[1:])
    return E_p / E_p[-1], E_m / E_p[-1], Qj / 2


def AmpElfield_q(k, kx, lamda, n, d):
    # Length of k-vector in vaccum
    k = 2 * pi / lamda
    # Calculates the wavevector in each layer
    # Qj=2*k*sqrt(n[:,newaxis]**2-cos(theta*pi/180)**2)
    Qj = 2 * sqrt(k**2 * n[:, newaxis] ** 2 - kx**2)
    # Fresnel reflectivity for the interfaces
    rp = (Qj[1:] - Qj[:-1]) / (Qj[1:] + Qj[:-1])
    # print rp
    tp = 1 + rp
    # tp=2*Qj[1:]/(Qj[1:]+Qj[:-1])
    # print tp.shape
    # Beta=k*d*n[:-1]*sin(theta*pi/180)
    # print Beta.shape
    D = d[:, newaxis] * ones(kx.shape)
    # print D.shape
    # print rp.shape
    # E_p=[1]
    # E_m=[0]
    # old version
    E_p = [1 * ones(kx.shape)]
    E_m = [0 * ones(kx.shape)]
    for i in range(0, len(rp)):
        E_p.append(1 / tp[i] * (E_p[i] * exp(-1.0j * D[i] * Qj[i] / 2) + E_m[i] * rp[i] * exp(1.0j * D[i] * Qj[i] / 2)))
        E_m.append(1 / tp[i] * (E_m[i] * exp(1.0j * D[i] * Qj[i] / 2) + E_p[i] * rp[i] * exp(-1.0j * D[i] * Qj[i] / 2)))
    # Added 020905 for testing another T amplitude (The Transmitted amp. trough the interface)
    # print E_p
    # End old version
    E_p = array(E_p)
    E_m = array(E_m)
    # print E_p.shape
    # print Qj.shape
    dtemp = r_[d, 0]
    # print dtemp
    dtemp = dtemp[:, newaxis] * ones(kx.shape)
    E_p = E_p * exp(-1.0j * dtemp * Qj / 2)
    E_m = E_m * exp(1.0j * dtemp * Qj / 2)
    E_p = list(E_p)
    E_m = list(E_m)
    # End addition
    E_p.reverse()
    E_m.reverse()
    Qj = list(Qj)
    Qj.reverse()
    Qj = array(Qj)
    # Removed 020905 as said above and replaced
    E_p = array(E_p)  # Old version
    # E_p=array(E_p)
    # End addition
    E_m = array(E_m)
    q = Qj / 2  # Old ver. replaced 020905 with:
    # return (E_p[:-1]/E_p[0],E_m[:-1]/E_p[0],q[:-1]) # Last correspond to kz
    # testar amplitud i lager j+1 istallet
    return E_p[1:] / E_p[0], E_m[1:] / E_p[0], q[1:]  # Last correspond to kz


def AmpElfield2(kx, k, n, z):
    kz = sqrt(n[:, newaxis] ** 2 * k**2 - kx**2)
    r = (kz[:-1] - kz[1:]) / (kz[:-1] + kz[1:])
    t = 1 + r
    X = 0 * ones(kx.shape)
    for i in range(len(n) - 2, -1, -1):
        X = (
            exp(-2j * kz[i] * z[i])
            * (r[i] + X * exp(2j * kz[i + 1] * z[i]))
            / (1 + r[i] * X * exp(2j * kz[i + 1] * z[i]))
        )
        iprint(i)
    # X=reflected amplitude...
    r = (kz[1:] - kz[:-1]) / (kz[:-1] + kz[1:])
    t = 1 + r
    R = [X]
    T = [1 * ones(X.shape)]
    for i in range(0, len(n) - 1, 1):
        R.append(
            1
            / t[i]
            * (T[i] * r[i] * exp(-1.0j * (kz[i + 1] + kz[i]) * z[i]) + R[i] * exp(-1.0j * (kz[i + 1] - kz[i]) * z[i]))
        )
        T.append(
            1
            / t[i]
            * (T[i] * exp(1.0j * (kz[i + 1] - kz[i]) * z[i]) + R[i] * r[i] * exp(1.0j * (kz[i + 1] + kz[i]) * z[i]))
        )
    # R[len(n)-1]=0*ones(X.shape)
    R = array(R)
    T = array(T)
    return T, R, kz


from .xrmr import dot2, inv2


def AmpElfield_test(th, lam, n, d, dz=0.1):
    """Added 20110317 to see if I can do calcs of the fields inside a multilayer"""
    kz = 2 * pi / lam * sin(th * pi / 180)
    k = 2 * pi / lam
    # n = r_[n, n[-1]]
    # d = r_[d, d[-1]]
    kz = sqrt((n[:, newaxis] ** 2 - 1) * k**2 + kz**2)
    # print kz.shape, d.shape
    a = exp(-1.0j * kz * d[:, newaxis])[1:, :]
    # print a.shape
    # print kz[newaxis, newaxis, :, :].shape
    r = (kz[:-1] - kz[1:]) / (kz[:-1] + kz[1:])
    M = (
        1
        / kz[newaxis, newaxis, :-1, :]
        / 2
        * array([[(kz[1:] + kz[:-1]) * a, (kz[:-1] - kz[1:]) / a], [(kz[:-1] - kz[1:]) * a, (kz[1:] + kz[:-1]) / a]])
    )
    # M = array([[a, r*a],[r/a, 1/a]])
    M = rollaxis(M, 2, 0)[::, :, :, :]
    # "quick" version
    # MM = reduce(dot2, M)
    MMcum = [M[0]]
    for i in range(1, M.shape[0]):
        MMcum.append(dot2(MMcum[-1], M[i]))
    MM = MMcum[-1]

    R0 = MM[1, 0] / MM[0, 0]
    TN = 1 / M[0, 0]
    R = abs(R0) ** 2
    T = abs(TN) ** 2

    # Calc el. fileds as a function of z
    Rlay = R0
    Tlay = 1.0
    z = array([0])
    iprint("R0 ", R0.shape)
    E = array(abs(R0 + 1.0))[:, newaxis] * ones((th.shape[0], 1))
    # E = array([])
    iprint(kz.shape)
    iprint((th.shape[0], 1))
    iprint(ones((th.shape[0], 1)).shape)
    iprint(E.shape)
    for (
        di,
        kzi,
        M,
    ) in zip(d[1:], kz[1:], MMcum):
        Mlay = inv2(M)
        Rlay_new = Mlay[0, 0] * Tlay + Mlay[0, 1] * Rlay
        Tlay_new = Mlay[1, 0] * Tlay + Mlay[1, 1] * Rlay
        znew = arange(0, di, dz)
        z = append(z, znew + z[-1])
        # print kzi.shape
        a = exp(-1.0j * kzi[:, newaxis] * (di - znew))
        # print (abs(Rlay_new[:, newaxis]*a + Tlay_new[:, newaxis]/a)**2).shape
        E = c_[E, abs(Rlay_new[:, newaxis] * a + Tlay_new[:, newaxis] / a)]
        Rlay = Rlay_new
        Tlay = Tlay_new

    # print MM.shape
    iprint(E.shape)
    # E = E.reshape((th.shape[0], z.shape[0]))
    return T, R, z, E


if __name__ == "__main__":
    # R=[]
    # for x in arange(0,2,0.01):
    #    (E_p,E_m)=IntElfield(x,1.54,array([1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1]),array([0,80,20,80,20,80,20]))
    #    R.append(abs(E_m[-1]/E_p[-1])**2)
    import pylab

    theta = arange(0, 1 + 0.001666670, 0.005)
    # (E_p,E_m,Qj)=IntElfield(theta,1.54,array([1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1]),array([0,80,20,80,20,80,20]))
    # (E_p,E_m,Qj)=IntElfield(theta,1.540562,array([1-.15E-04/2+0.35E-06j/2,1-.45E-04/2+0.60E-05j/2,1]),array([0,1000]))
    from .paratt import Refl

    # c=Refl(theta,1.54,array([1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1-2.24e-5+2.89e-6j,1-7.57e-6+1.73e-7j,1]),array([80,20,80,20,80,20]),0)
    c = Refl(
        theta, 1.54, array([1 - 7.57e-6 + 1.73e-7j, 1 - 2.24e-5 + 2.89e-6j, 1]), array([0, 1000, 0]), array([0, 0, 0])
    )
    # gplt.plot(theta,log10(abs(E_m[-1]/E_p[-1])**2),theta,log10(abs(c**2)))
    # with open('Specrefl.dat','r') as f:
    #     t=io.read_array(f)
    #     t=transpose(t)
    # gplt.plot(theta,log10(abs(E_m[-1]/E_p[-1])**2),t[0],log10(t[1]))
    k = 2 * pi / 1.54
    n = [1, 1 - 2.24e-5 + 2.89e-6j, 1 - 7.57e-6 + 1.73e-7j]
    n = array(n)
    z = array([0, -1000])
    # (T,R,k)=AmpElfield2(k*cos(theta*pi/180),k*ones(theta.shape), n,z)
    # (T,R,k)=IntElfield(theta,1.54, n, array([0,1000]))
    (T, R, zc, E) = AmpElfield_test(theta, 1.54, n, array([0, 1000, 0]))
    # print T[0]
    pylab.subplot(211)
    pylab.plot(theta, log10(abs(R)), theta, log10(abs(c)), ".-")
    pylab.ylabel("R")
    pylab.xlabel("theta [deg]")
    pylab.legend(("Abeles", "Parratt"))
    pylab.subplot(212)
    pylab.contourf(theta, zc, E.transpose())
    # print zc.shape, zc[-1]
    # print log10(E[0,:]).shape
    # pylab.plot(zc, (E[1,:]))
    pylab.show()
