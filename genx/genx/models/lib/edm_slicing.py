'''Library that implements an effective density model for composition 
profiles (as described by Tolan) and also a home constructed model for
magnetic moment profiles (element specific). 
'''
# from pylab import *
from numpy import *

import numpy as np

from scipy.special import erf

def erf_profile(z, z0, d, sigma0, sigma1):
    eta=(sigma1*z0+sigma0*(z0+d))/(sigma0+sigma1)
    p=0.5*(1+where(z<=eta, erf((z-z0)/sqrt(2.)/sigma0),
                   -erf((z-z0-d)/sqrt(2.)/sigma1)))
    return p

def erf_interf(z, sigma):
    return 0.5+0.5*erf(z/sqrt(2.)/sigma)

def _lin(z, sigma):
    p=zeros(z.shape)
    p=((z>sqrt(3)*sigma)*1.0+
       (abs(z)<=sqrt(3)*sigma)*(0.5+z/2/sqrt(3)/sigma))
    return p

def lin_profile(z, z0, d, sigma0, sigma1):
    eta=(sigma1*z0+sigma0*(z0+d))/(sigma0+sigma1)
    p=where(z<eta, _lin(z-z0, sigma0),
            _lin(-(z-z0-d), sigma1))
    return p

def _exp(z, sigma):
    p=((z<=0)*(0.5*exp(sqrt(2)*z/sigma))+
       (z>0)*(1-0.5*exp(-sqrt(2)*z/sigma)))
    return p

def exp_profile(z, z0, d, sigma0, sigma1):
    eta=(sigma1*z0+sigma0*(z0+d))/(sigma0+sigma1)
    p=where(z<eta, _exp(z-z0, sigma0),
            _exp(-(z-z0-d), sigma1))
    return p

def _sin(z, sigma):
    a=pi/sqrt(pi**2-8)*sigma
    p=(z>a)*1.0+(abs(z)<=a)*(0.5+0.5*sin(pi*z/2/a))
    return p

def sin_profile(z, z0, d, sigma0, sigma1):
    eta=(sigma1*z0+sigma0*(z0+d))/(sigma0+sigma1)
    p=where(z<eta, _sin(z-z0, sigma0),
            _sin(-(z-z0-d), sigma1))
    return p

def compress_profile_old(z, p, delta_max):
    pnew=p.copy()
    znew=z.copy()
    i=0
    index=array([False])
    while any(bitwise_not(index)):
        # print i
        i+=1
        index=abs(pnew[:-1]-pnew[1:])>delta_max
        index[::2]=True
        index=r_[index, True]
        # print pnew.shape, index.shape
        pnew=pnew[index]
        znew=znew[index]

    return znew, pnew

def compress_profile(z, p, delta_max):
    ''' Compresses a profile by merging the neighbouring layers that
    has an density difference of maximum delta_max'''
    pnew=p.copy()
    znew=z.copy()
    inew=arange(len(p))
    i=0
    index=array([False])
    while any(bitwise_not(index)):
        # print i
        i+=1
        index=array([True]*len(pnew))
        index[1:-1:2]=abs(pnew[:-2:2]-pnew[2::2])>delta_max
        # print pnew.shape, index.shape
        pnew=pnew[index]
        znew=znew[index]
        inew=inew[index]
    # print inew
    inew=append(inew, inew[-1])
    pret=array([p[inew[i]:inew[i+1]+1].mean() for i in range(len(inew)-1)])
    return znew, pret

def compress_profile2(z, p1, p2, delta_max):
    ''' Compresses a profile by merging the neighbouring layers that
    has an density difference of maximum delta_max'''
    p1new=p1.copy()
    p2new=p2.copy()
    znew=z.copy()
    inew=arange(len(p1))
    i=0
    index=array([False])
    while any(bitwise_not(index)):
        # print i
        i+=1
        index=array([True]*len(p1new))
        index[1:-1:2]=bitwise_or(abs(p1new[:-2:2]-p1new[2::2])>delta_max
                                 , abs(p2new[:-2:2]-p2new[2::2])>delta_max
                                 )
        # print pnew.shape, index.shape
        p1new=p1new[index]
        p2new=p2new[index]
        znew=znew[index]
        inew=inew[index]
    # print inew
    inew=append(inew, inew[-1])
    p1ret=array([p1[inew[i]:inew[i+1]+1].mean() for i in range(len(inew)-1)])
    p2ret=array([p2[inew[i]:inew[i+1]+1].mean() for i in range(len(inew)-1)])
    return znew, p1ret, p2ret

def compress_profile_n(z, ps, delta_maxs):
    ''' Compresses multiple profiles, each profile: ps[i], 
    by merging the neighbouring layers that
    has an density difference of maximum delta_maxs[i]'''
    inew, znew=compress_profile_index_n(z, ps, delta_maxs)
    psnew=create_compressed_profile(ps, inew)
    return znew, psnew

def compare_layers(ps, delta_max):
    """Compare consecutive layers and see if ps is smaller than delta_max"""
    if len(ps.shape)>1:
        return (abs(ps[:-2:2]-ps[2::2])>delta_max).any(axis=1)
    else:
        return abs(ps[:-2:2]-ps[2::2])>delta_max

def compress_profile_index_n(z, ps, delta_maxs):
    znew=z.copy()
    inew=arange(len(ps[0]))
    psnew=[ps_i.copy() for ps_i in ps]
    i=0
    index=array([False])
    while any(bitwise_not(index)):
        i+=1
        index=array([True]*len(psnew[0]))
        test=array([compare_layers(ps_i, delta_maxs_i) for ps_i, delta_maxs_i in zip(psnew, delta_maxs)])
        index[1:-1:2]=test.any(axis=0)

        psnew=[ps_i[index] for ps_i in psnew]
        znew=znew[index]
        inew=inew[index]
    inew=append(inew, inew[-1])
    return inew, znew

def create_compressed_profile(ps, inew):
    psret=[array([ps_i[inew[i]:inew[i+1]+1].mean(0) for i in range(len(inew)-1)]) for ps_i in ps]
    return psret

def create_profile(d, sigma, dens, prof_funcs, dz=0.01,
                   mult=3, buffer=0, delta=20):
    zlay=r_[0, cumsum(d)]
    z=arange(-sigma[0]*mult-buffer, zlay[-1]+sigma[-1]*mult+buffer, dz)
    ptot=z*0
    pdens=z*0
    ps=[]
    d=r_[delta+sigma[0]*mult+buffer, d, sigma[-1]*mult+delta+buffer]
    sigma=r_[0, sigma, 0]
    zlay=cumsum(d)
    # print zlay
    z0=buffer+sigma[1]*mult
    zlay=r_[0, zlay]-z0
    # print zlay
    # for i in range(len(d)):
    #    print z0
    #    p = make_profile(z, z0, d[i], sigma[i], sigma[i+1])
    #    z0 += d[i]
    #    ptot += p
    #    pdens += p*dens[i]
    #    ps.append(p)
    ps=array([prof(z, zp, di, s_low, s_up)
              for prof, zp, di, s_low, s_up in zip(prof_funcs, zlay,
                                                   d, sigma[:-1], sigma[1:])])
    ptot=ps.sum(0)
    pdens_indiv=(ps*dens[:, newaxis])/ptot
    pdens=pdens_indiv.sum(0)

    return z, pdens, pdens_indiv

def create_profile_cm(d, sigma_c, sigma_m, prof_funcs,
                      prof_funcs_mag, dmag_dens_l, dmag_dens_u, mag_dens, dd_m,
                      dz=0.01, mult=3, buffer=0, delta=20):
    '''Create a scattering length profile for magnetic x-ray scattering for the charge and
    magnetic part of the scattering lengths. sigma is roughnesses.
    Note that prof_funcs relate to layer profiles and prof_funcs_mag relates to 
    magnetic interface profiles.
    '''
    zlay=r_[0, cumsum(d)]
    s_max_bot=max(sigma_c[0], sigma_m[0])
    s_max_up=max(sigma_c[-1], sigma_m[-1])
    z=arange(-s_max_bot*mult-buffer, zlay[-1]+s_max_up*mult+buffer+dz, dz)
    ptot=z*0
    pdens=z*0
    ps=[]
    d=r_[delta+s_max_bot*mult+buffer, d, s_max_up*mult+delta+buffer]
    sigma_c=r_[0, sigma_c, 0]+1e-20
    sigma_m=r_[0, sigma_m, 0]+1e-20
    dd_m=r_[0, dd_m, 0]+1e-20
    prof_funcs_mag=[prof_funcs_mag[0]]+prof_funcs_mag+[prof_funcs_mag[-1]]
    zlay=cumsum(d)
    z0=delta+buffer+s_max_bot*mult
    zlay=r_[0, zlay]-z0
    ps=array([r_[prof(z, zp, di, s_c_low, s_c_up),
                 # prof(z, zp, di, s_m_low, s_m_up)]
                 prof_m_l(-(z-zp-dd_low), s_m_low)*dm_l+prof_m_u(z-zp-di-dd_up, s_m_up)*dm_u+m]
              for prof, zp, di, s_c_low, s_c_up, s_m_low, s_m_up,
                  prof_m_l, prof_m_u, dm_l, dm_u, m, dd_low, dd_up
              in zip(prof_funcs, zlay, d, sigma_c[:-1], sigma_c[1:],
                     sigma_m[:-1], sigma_m[1:],
                     prof_funcs_mag[:-1], prof_funcs_mag[1:],
                     dmag_dens_l, dmag_dens_u, mag_dens, dd_m[:-1], dd_m[1:]
                     )])
    ptot=ps.sum(0)
    # print ps.shape, ptot.shape
    pdens_indiv=(ps[:, :len(z)])/ptot[:len(z)]
    pdens_indiv_c=pdens_indiv  # *dens_c[:,newaxis]
    pdens_indiv_m=ps[:, len(z):]  # *dens_m[:,newaxis]
    # pdens_c = pdens_indiv_c.sum(0)
    # pdens_m = (pdens_indiv_m*pdens_indiv).sum(0)

    return z, pdens_indiv_c, pdens_indiv_m

def create_profile_cm2(d, sigma_c, sigma_ml, sigma_mu, prof_funcs,
                       prof_funcs_mag, dmag_dens_l, dmag_dens_u, mag_dens, dd_l, dd_u,
                       dz=0.01, mult=3, buffer=0, delta=20):
    '''Create a scattering length profile for magnetic x-ray scattering for the charge and
    magnetic part of the scattering lengths. sigma is roughnesses.
    Note that prof_funcs relate to layer profiles and prof_funcs_mag relates to 
    magnetic interface profiles.
    '''
    zlay=r_[0, cumsum(d)]
    s_max_bot=max(sigma_c[0], sigma_ml[0])
    s_max_up=max(sigma_c[-1], sigma_mu[-1])
    z=arange(-s_max_bot*mult-buffer, zlay[-1]+s_max_up*mult+buffer+dz, dz)
    ptot=z*0
    pdens=z*0
    ps=[]
    d=r_[delta+s_max_bot*mult+buffer, d, s_max_up*mult+delta+buffer]
    sigma_c=r_[0, sigma_c, 0]+1e-20
    sigma_mu=r_[sigma_mu, 0]+1e-20
    sigma_ml=r_[0, sigma_ml]+1e-20

    dd_l=r_[dd_l, 0]+1e-20
    dd_u=r_[dd_u, 0]+1e-20

    prof_funcs_mag=[prof_funcs_mag[0]]+prof_funcs_mag+[prof_funcs_mag[-1]]
    zlay=cumsum(d)
    z0=delta+buffer+s_max_bot*mult
    zlay=r_[0, zlay]-z0
    ps=array([r_[prof(z, zp, di, s_c_low, s_c_up),
                 # prof(z, zp, di, s_m_low, s_m_up)]
                 prof_m_l(-(z-zp-dd_low), s_m_low)*dm_l*m+prof_m_u((z-zp-di+dd_up), s_m_up)*dm_u*m+m]
              for prof, zp, di, s_c_low, s_c_up, s_m_low, s_m_up,
                  prof_m_l, prof_m_u, dm_l, dm_u, m, dd_low, dd_up
              in zip(prof_funcs, zlay, d, sigma_c[:-1], sigma_c[1:],
                     sigma_ml, sigma_mu,
                     prof_funcs_mag[:-1], prof_funcs_mag[1:],
                     dmag_dens_l, dmag_dens_u, mag_dens, dd_l, dd_u
                     )])
    ptot=ps.sum(0)
    # print ps.shape, ptot.shape
    pdens_indiv=(ps[:, :len(z)])/ptot[:len(z)]
    pdens_indiv_c=pdens_indiv  # *dens_c[:,newaxis]
    pdens_indiv_m=ps[:, len(z):]  # *dens_m[:,newaxis]
    # pdens_c = pdens_indiv_c.sum(0)
    # pdens_m = (pdens_indiv_m*pdens_indiv).sum(0)

    return z, pdens_indiv_c, pdens_indiv_m

if __name__=='__main__':
    from pylab import *
    from mpl_toolkits.axes_grid import Grid
    from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

    # sigma0 = 5.
    # sigma1 = 1.
    # mult = 3
    # buffer = 0
    # delta = 20
    # z0 = sigma0*mult
    # d = 9.#sigma0*mult + sigma1*mult
    # z = arange(0, z0 + d + sigma1*mult, 0.1)
    # p = make_profile(z, z0, d, sigma0, sigma1)
    # plot(z,p)
    sigma_c=array([4, 4, 2])*1.0  # sigma_c = array([2,8,2])*1.0
    sigma_m=array([1, 1, 1])*1.0  # sigma_m = array([2,5,4])*1.0
    dmag_l=array([0.0, 0.0, 0.0])
    dmag_u=array([0.0, 0.0, 0.0])
    mag_dens=array([0.0, 2.25, 0.0])
    # ddm = array([10.0, -10.0, 0.0])
    ddm=array([10.0, -10.0, 10.0])
    d=array([0.0, 50, 0.0])*1.0  # d = array([5, 60])*1.0
    dens_c=array([-1.79e-3+0.0029J, -4e-23])  # dens_c = array([5,50,25,0])
    dens_m=array([1.4458e-3-0.00144J, 2.12e-23])  # dens_m = array([0,6,10,0])
    prof_funcs=[erf_profile]*3  # prof_funcs = [erf_profile]*4
    mag_prof_funcs=[erf_interf]*3
    # z, pdens, pdens_indiv = create_profile(d, sigma_c, dens_cprof_funcs, dz = 0.1,
    #                                       mult = 5.0, 
    #                                       buffer = 0, delta = 5.)
    z, c_den, m_den=create_profile_cm(d[1:-1], sigma_c[:-1], sigma_m[:-1],
                                      prof_funcs, mag_prof_funcs,
                                      dmag_l, dmag_u, mag_dens,
                                      ddm, dz=1.0, mult=4.0,
                                      buffer=20, delta=5.)

    def plot_fig(filename, fig_title=''):
        fig=figure(1)
        grid=Grid(fig, 111, nrows_ncols=(3, 1), axes_pad=0.15)
        # subplot(311)
        grid[0].set_title(fig_title)
        grid[0].plot(z, c_den[0], lw=2.0, c='b')
        grid[0].plot(z, c_den[1], lw=2.0, c='r')
        grid[0].plot(z, c_den[2], lw=2.0, c='g')
        grid[0].axvline(0, linestyle='--', c='k')
        grid[0].axvline(d[1], linestyle='--', c='k')
        grid[0].set_ylabel('Comp.')
        grid[0].legend(('Substrate', 'Fe layer', 'Ambient'))
        # subplot(312)
        grid[1].plot(z, m_den[1], lw=2.0, c='r')
        grid[1].axvline(0, linestyle='--', c='k')
        grid[1].axvline(d[1], linestyle='--', c='k')
        grid[1].axvline(0+ddm[0], linestyle=':', c='k')
        grid[1].axvline(d[1]+ddm[1], linestyle=':', c='k')
        # grid[1].annotate('Sub.dd_m = %.1f'%ddm[0], xy=(ddm[0], mag_dens[1]), xycoords = 'data',
        #                 xytext = (-50, 0), textcoords = 'offset points', 
        #                 arrowprops=dict(arrowstyle="->", connectionstyle = 'arc3'))
        # grid[1].annotate('Fe.dd_m = %.1f'%ddm[1], xy=(d[1]+ddm[1], mag_dens[1]), xycoords = 'data',
        #                 xytext = (50, 0), textcoords = 'offset points', 
        #                 arrowprops=dict(arrowstyle="<-", connectionstyle = 'arc3'))
        at=AnchoredText("dd_ml = %.1f"%(ddm[0]),
                        prop=dict(size=10), frameon=True,
                        loc=6,
                        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        grid[1].add_artist(at)
        at=AnchoredText("dd_mu = %.1f"%(ddm[1]),
                        prop=dict(size=10), frameon=True,
                        loc=7,
                        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        grid[1].add_artist(at)
        grid[1].set_ylabel('Mag. mom.\n per res. atom.')
        grid[1].set_ylim(0, 3.1)
        # subplot(313)
        grid[2].plot(z, m_den[1]*c_den[1], 'r', z, 2.25*c_den[1], 'b', lw=2.0)
        grid[2].axvline(0, linestyle='--', c='k')
        grid[2].axvline(d[1], linestyle='--', c='k')
        grid[2].axvline(0+ddm[0], linestyle=':', c='k')
        grid[2].axvline(d[1]+ddm[1], linestyle=':', c='k')
        grid[2].set_ylabel('Tot. Mag mom.')
        grid[2].legend(('Profile', 'Bulk Fe'))
        grid[2].set_xlabel('Depth [AA]')
        savefig(filename)
        clf()

    plot_fig('Bulk_Fe_sigma', 'Bulk Fe with no change at the interfaces')
    dmag_l=array([0.0, 0.75, 0.0])
    dmag_u=array([0.0, 0.75, 0.0])
    z, c_den, m_den=create_profile_cm(d[1:-1], sigma_c[:-1], sigma_m[:-1],
                                      prof_funcs, mag_prof_funcs,
                                      dmag_l, dmag_u, mag_dens,
                                      ddm, dz=1.0, mult=4.0,
                                      buffer=20, delta=5.)
    plot_fig('Enhanced_Fe_sigma', 'Bulk Fe with enhanced interfaces')
    dmag_l=array([0.0, -1., 0.0])
    dmag_u=array([0.0, -1., 0.0])
    z, c_den, m_den=create_profile_cm(d[1:-1], sigma_c[:-1], sigma_m[:-1],
                                      prof_funcs, mag_prof_funcs,
                                      dmag_l, dmag_u, mag_dens,
                                      ddm, dz=1.0, mult=4.0,
                                      buffer=20, delta=5.)
    plot_fig('Depleted_Fe_sigma', 'Bulk Fe with depleted interfaces')
