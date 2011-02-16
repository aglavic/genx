'''Library that implements an effective density model for composition 
profiles (as described by Tolan) and also a home constructed model for
magnetic moment profiles (element specific). 
'''
from pylab import *
from numpy import *

from scipy.special import erf

def erf_profile(z, z0, d, sigma0, sigma1):
    eta = (sigma1*z0 + sigma0*(z0 + d))/(sigma0 + sigma1)
    p = 0.5*(1 + where(z <= eta, erf((z - z0)/sqrt(2.)/sigma0), 
                       -erf((z - z0 - d)/sqrt(2.)/sigma1)))
    return p

def _lin(z, sigma):
    p = zeros(z.shape)
    p = ((z > sqrt(3)*sigma)*1.0 + 
         (abs(z) <= sqrt(3)*sigma)*(0.5 + z/2/sqrt(3)/sigma))
    return p

def lin_profile(z, z0, d, sigma0, sigma1):
    eta = (sigma1*z0 + sigma0*(z0 + d))/(sigma0 + sigma1)
    p =  where(z < eta, _lin(z - z0, sigma0), 
                       _lin(-(z - z0 - d), sigma1))
    return p

def _exp(z, sigma):
    p = ((z <= 0)*(0.5*exp(sqrt(2)*z/sigma)) + 
         (z > 0)*(1 - 0.5*exp(-sqrt(2)*z/sigma)))
    return p

def exp_profile(z, z0, d, sigma0, sigma1):
    eta = (sigma1*z0 + sigma0*(z0 + d))/(sigma0 + sigma1)
    p =  where(z < eta, _exp(z - z0, sigma0), 
                       _exp(-(z - z0 - d), sigma1))
    return p

def _sin(z, sigma):
    a = pi/sqrt(pi**2 - 8)*sigma
    p = (z > a)*1.0 + (abs(z) <= a)*(0.5 + 0.5*sin(pi*z/2/a))
    return p

def sin_profile(z, z0, d, sigma0, sigma1):
    eta = (sigma1*z0 + sigma0*(z0 + d))/(sigma0 + sigma1)
    p =  where(z < eta, _sin(z - z0, sigma0), 
                       _sin(-(z - z0 - d), sigma1))
    return p


def compress_profile_old(z, p, delta_max):
    pnew = p.copy()
    znew = z.copy()
    i = 0
    index = array([False])
    while any(bitwise_not(index)):
        print i
        i += 1
        index = abs(pnew[:-1] - pnew[1:]) > delta_max
        index[::2] = True
        index = r_[index, True]
        print pnew.shape, index.shape
        pnew = pnew[index]
        znew = znew[index]

    return znew, pnew

def compress_profile(z, p, delta_max):
    ''' Compresses a profile by merging the neighbouring layers that
    has an density difference of maximum delta_max'''
    pnew = p.copy()
    znew = z.copy()
    inew = arange(len(p))
    i = 0
    index = array([False])
    while any(bitwise_not(index)):
        #print i
        i += 1
        index = array([True]*len(pnew))
        index[1:-1:2] = abs(pnew[:-2:2] - pnew[2::2]) > delta_max
        #print pnew.shape, index.shape
        pnew = pnew[index]
        znew = znew[index]
        inew = inew[index]
    #print inew
    inew = append(inew, inew[-1])
    pret = array([p[inew[i]:inew[i+1]+1].mean() for i in range(len(inew)-1)])
    return znew, pret

def compress_profile2(z, p1, p2, delta_max):
    ''' Compresses a profile by merging the neighbouring layers that
    has an density difference of maximum delta_max'''
    p1new = p1.copy()
    p2new = p2.copy()
    znew = z.copy()
    inew = arange(len(p1))
    i = 0
    index = array([False])
    while any(bitwise_not(index)):
        #print i
        i += 1
        index = array([True]*len(p1new))
        index[1:-1:2] = bitwise_or(abs(p1new[:-2:2] - p1new[2::2]) > delta_max
                                    ,abs(p2new[:-2:2] - p2new[2::2]) > delta_max
                                    )
        #print pnew.shape, index.shape
        p1new = p1new[index]
        p2new = p2new[index]
        znew = znew[index]
        inew = inew[index]
    #print inew
    inew = append(inew, inew[-1])
    p1ret = array([p1[inew[i]:inew[i+1]+1].mean() for i in range(len(inew)-1)])
    p2ret = array([p2[inew[i]:inew[i+1]+1].mean() for i in range(len(inew)-1)])
    return znew, p1ret, p2ret

def create_profile(d, sigma, dens, prof_funcs, dz = 0.01, 
                   mult = 3, buffer = 0, delta = 20):
    zlay = r_[0, cumsum(d)]
    z = arange(-sigma[0]*mult - buffer, zlay[-1]+sigma[-1]*mult + buffer, dz)
    ptot = z*0
    pdens = z*0
    ps = []
    d = r_[delta+sigma[0]*mult + buffer, d, sigma[-1]*mult+delta + buffer]
    sigma = r_[0, sigma, 0]
    zlay = cumsum(d)
    print zlay
    z0 = buffer + sigma[1]*mult
    zlay = r_[0, zlay] - z0
    print zlay
    #for i in range(len(d)):
    #    print z0
    #    p = make_profile(z, z0, d[i], sigma[i], sigma[i+1])
    #    z0 += d[i]
    #    ptot += p
    #    pdens += p*dens[i]
    #    ps.append(p)
    ps = array([prof(z, zp, di, s_low, s_up) 
                for prof, zp, di, s_low, s_up in zip(prof_funcs, zlay, 
                                                     d, sigma[:-1], sigma[1:])])
    ptot = ps.sum(0)
    pdens_indiv = (ps*dens[:,newaxis])/ptot
    pdens = pdens_indiv.sum(0)

    return z, pdens, pdens_indiv

def create_profile_cm(d, sigma_c, sigma_m, dens_c, dens_m, prof_funcs, dz = 0.01, 
                   mult = 3, buffer = 0, delta = 20):
    zlay = r_[0, cumsum(d)]
    s_max_bot = max(sigma_c[0], sigma_m[0])
    s_max_up = max(sigma_c[-1], sigma_m[-1])
    z = arange(-s_max_bot*mult - buffer, zlay[-1]+s_max_up*mult + buffer + dz, dz)
    ptot = z*0
    pdens = z*0
    ps = []
    d = r_[delta+s_max_bot*mult + buffer, d, s_max_up*mult+delta + buffer]
    sigma_c = r_[0, sigma_c, 0] + 1e-20
    sigma_m = r_[0, sigma_m, 0] + 1e-20
    zlay = cumsum(d)
    z0 = delta + buffer + s_max_bot*mult
    zlay = r_[0, zlay] - z0
    ps = array([r_[prof(z, zp, di, s_c_low, s_c_up), 
                   prof(z, zp, di, s_m_low, s_m_up)]
                for prof, zp, di, s_c_low, s_c_up, s_m_low, s_m_up 
                   in zip(prof_funcs, zlay, d, sigma_c[:-1], sigma_c[1:], 
                          sigma_m[:-1], sigma_m[1:])])
    ptot = ps.sum(0)
    pdens_indiv = (ps[:, :len(z)])/ptot[:len(z)]
    pdens_indiv_c = pdens_indiv*dens_c[:,newaxis]
    pdens_indiv_m = ps[:, len(z):]*dens_m[:,newaxis]
    pdens_c = pdens_indiv_c.sum(0)
    pdens_m = (pdens_indiv_m*pdens_indiv).sum(0)

    return (z, pdens_c, pdens_m, 
            pdens_indiv_c, pdens_indiv_m)

if __name__ == '__main__':
    #sigma0 = 5.
    #sigma1 = 1.
    #mult = 3
    #buffer = 0
    #delta = 20
    #z0 = sigma0*mult
    #d = 9.#sigma0*mult + sigma1*mult
    #z = arange(0, z0 + d + sigma1*mult, 0.1)
    #p = make_profile(z, z0, d, sigma0, sigma1)
    #plot(z,p)
    sigma_c = array([7])*1.0#sigma_c = array([2,8,2])*1.0
    sigma_m = array([7])*1.0#sigma_m = array([2,5,4])*1.0
    d = array([])*1.0#d = array([5, 60])*1.0
    dens_c = array([-1.79e-3 + 0.0029J, -4e-23])#dens_c = array([5,50,25,0])
    dens_m = array([1.4458e-3 - 0.00144J, 2.12e-23])#dens_m = array([0,6,10,0])
    prof_funcs = [erf_profile]*2#prof_funcs = [erf_profile]*4
    #z, pdens, pdens_indiv = create_profile(d, sigma_c, dens_cprof_funcs, dz = 0.1,
    #                                       mult = 5.0, 
    #                                       buffer = 0, delta = 5.)
    z, pdens_c, pdens_m, pdens_indiv_c, pdens_indiv_m = create_profile_cm(d, sigma_c, sigma_m, dens_c, dens_m, prof_funcs, dz = 1.0,
                                           mult = 4.0, 
                                           buffer = 20, delta = 5.)
    print z.shape
    #pdens = pdens/ptot
    #zn, pn = compress_profile(z, pdens, 1.)
    zn, pn_c, pn_m = compress_profile2(z, pdens_c, pdens_m, 1.)
    dn = zn[1:] - zn[:-1]
    subplot(211)
    #plot(z, pdens, zn, pn, drawstyle = 'steps-post',lw = 2.0)
    #bar(zn, pn_c, r_[dn,1], zorder = -10)
    bar(zn[:], pn_c[:], r_[dn,1], zorder = -10)
    plot(z, pdens_indiv_m[0], lw = 2.0, c = 'r')
    #bar(zn, pn_m, r_[dn,1], color = 'g')
    #plot(z, pdens_m, lw = 2.0, c = 'r')
    #subplot(212)
    #plot(z, pdens_indiv_c[0], 'b-', z, pdens_indiv_c[1], 'r-', z, pdens_indiv_c[2],'g-')
    #plot(z, pdens_indiv_m[0], 'b-.', z, pdens_indiv_m[1], 'r-.', z, pdens_indiv_m[2],'g-.')
    #subplot(313)
    #plot(z,ps[0], z, ps[1], z, ps[2])
    show()