"""Library to calculate the reflectivity with the formalism presented in
S. A. Stepanov and S. K. Sinha, Phys. Rev. B 61, 15302.

All the matrices/arrays in this module has the following index order:
index: Meaning
1: matrix row
2: matrix column
3: Layer index
4: scanning angle (g0, incident angle)

"""

from functools import reduce

from genx.core.custom_logging import iprint

from .int_lay_xmean import calc_xrmr_Xmean
from .math_utils import *


# mpy_limit = 1e-9 corresponds to an angle of 89.998 deg.
def calc_refl(g_0, lamda, chi0, A, B, C, M, d, mag_limit=1e-8, mpy_limit=1e-9):
    """Calculate the reflectivity according to the
    recursion matrix formalism as given by
    S. A. Stepanov and S. K. Sinha, Phys. Rev. B 61, 15302.
    https://doi.org/10.1103/PhysRevB.61.15302
    Note that this formalism has the first item in the parameters
    describing the layers (chi0, A, B, C, d) as the vacuum.
    Any ambient layer is ignored!
    g_0 = sin Phi_0 where Phi_0 is the incident angle
    lamda - the wavelength of the experiment
    chi0 - the non-resonant suceptibility of the layers
    A - the resonant non-magnetic term of the suceptibility
    B - the resonant magnetic term linear to the magnetization (circulair dichroism)
    C - the resonant magnetic term quadratic to the magnetization.
    M - the UNIT vector of the magnetization vector direction
    d - the thickness of the layers
    mag_limit - the limit where B and C are assumed to be zero and the
    material are assumed to be non-magnetic
    mpy_limit - magnet parallel to y limit the angle to the y-axis where the sample is assumed
    to be parallel to the y-direction

    Sizes of input arrays:
    g_0: # data points
    lamda: Either a scalar or a array with size # data points
    d: An array of size # layers
    A, B, C, chi0: either all arrays with size # layers or 2 D arrays of size (# layers, # data points)
    M: either an array of size (3, # layers) or (3, # layers, # data points)
    The if the case of more dimensions is used for one array it has to applied to all the other arrays.

    Note that the first layer should be the ambient and the last the substrate!
    """
    chi, non_mag, mpy = create_chi(g_0, lamda, chi0, A, B, C, M, d, mag_limit=mag_limit, mpy_limit=mpy_limit)

    return do_calc(g_0, lamda, chi, d, non_mag, mpy)


def calc_refl_int_lay(
    g_0, lamda, chi0, A, B, C, M, d, sigma, sigma_l, sigma_u, dd_l, dd_u, dmag_l, dmag_u, mag_limit=1e-8, mpy_limit=1e-9
):
    """Calculate the reflectivity according to the recursion matrix formalism as calc_refl.
    This function incorporates a interface layer for the magnetism of thicknesses dd_l and dd_u.
    the interfaces has rouhnesses sigma_l, sigma_u and sigma

    Note that the first layer should be the ambient and the last the substrate!
    """
    chi_b, non_mag_b, mpy_b = create_chi(g_0, lamda, chi0, A, B, C, M, d, mag_limit=mag_limit, mpy_limit=mpy_limit)
    scale = 1.0 + dmag_l
    if len(B.shape) == 2:
        scale = scale[:, np.newaxis] * np.ones(g_0.shape)
    chi_l, non_mag_l, mpy_l = create_chi(
        g_0, lamda, chi0, A, B * scale, C * scale**2, M, dd_l, mag_limit=mag_limit, mpy_limit=mpy_limit
    )
    scale = 1.0 + dmag_u
    if len(B.shape) == 2:
        scale = scale[:, np.newaxis] * np.ones(g_0.shape)
    chi_u, non_mag_u, mpy_u = create_chi(
        g_0, lamda, chi0, A, B * scale, C * scale**2, M, dd_u, mag_limit=mag_limit, mpy_limit=mpy_limit
    )

    d = d - dd_l - dd_u

    # Setting up g0 to be the last index
    g_0 = g_0 * np.ones(chi_b[0][0].shape[0], dtype=np.complex128)[:, np.newaxis]

    u_b, S_b = calc_u_S(chi_b, g_0, mpy_b, non_mag_b)
    u_l, S_l = calc_u_S(chi_l, g_0, mpy_l, non_mag_l)
    u_u, S_u = calc_u_S(chi_u, g_0, mpy_u, non_mag_u)

    X_lu = dot4(inv4(S_l[:, :, :-1, :]), S_u[:, :, 1:, :])
    X_l = dot4(inv4(S_b[:, :, :-1, :]), S_l[:, :, :-1, :])
    X_u = dot4(inv4(S_u[:, :, 1:, :]), S_b[:, :, 1:, :])

    d = d[:, np.newaxis] * np.ones(g_0.shape[1], dtype=np.complex128)
    dd_u = dd_u[:, np.newaxis] * np.ones(g_0.shape[1], dtype=np.complex128)
    dd_l = dd_l[:, np.newaxis] * np.ones(g_0.shape[1], dtype=np.complex128)
    sigma = sigma[:, np.newaxis] * np.ones(g_0.shape[1], dtype=np.complex128)
    sigma_l = sigma_l[:, np.newaxis] * np.ones(g_0.shape[1], dtype=np.complex128)
    sigma_u = sigma_u[:, np.newaxis] * np.ones(g_0.shape[1], dtype=np.complex128)

    Xmean = calc_xrmr_Xmean(lamda, X_l, X_lu, X_u, u_b, u_l, u_u, dd_u, dd_l, sigma, sigma_l, sigma_u)

    W_tot = recursion_matrix(Xmean, chi_b, d, g_0, lamda, u_b)
    return W_tot[2]


def create_chi(g_0, lamda, chi0, A, B, C, M, d, mag_limit=1e-8, mpy_limit=1e-9):
    A = A.astype(np.complex128)
    B = B.astype(np.complex128)
    C = C.astype(np.complex128)

    m_x = M[:, 0]
    m_y = M[:, 1]
    m_z = M[:, 2]
    # print 'my: ', m_y, m_x, m_z
    chi_xx = chi0 + A + C * m_x * m_x
    chi_yy = chi0 + A + C * m_y * m_y
    chi_zz = chi0 + A + C * m_z * m_z
    chi_xy = -1.0j * B * m_z + C * m_x * m_y
    chi_yx = 1.0j * B * m_z + C * m_x * m_y
    chi_xz = 1.0j * B * m_y + C * m_x * m_z
    chi_zx = -1.0j * B * m_y + C * m_x * m_z
    chi_yz = -1.0j * B * m_x + C * m_y * m_z
    chi_zy = 1.0j * B * m_x + C * m_y * m_z
    chi = ((chi_xx, chi_xy, chi_xz), (chi_yx, chi_yy, chi_yz), (chi_zx, chi_zy, chi_zz))

    # Take into account non-magnetic materials:
    non_mag = np.bitwise_and(np.abs(B) < mag_limit, np.abs(C) < mag_limit)
    # Ignore the ambient (vacuum)
    non_mag[0] = False

    # Take into account the matrix singularity arising when M||Y
    mpy = np.bitwise_and(1.0 - np.abs(m_y) < mpy_limit, np.bitwise_not(non_mag))

    return chi, non_mag, mpy


def calc_u_S(chi, g_0, mpy, non_mag):
    """Calculates the interface matrix S and the u values."""
    chi_xx, chi_xy, chi_xz = chi[0]
    chi_yx, chi_yy, chi_yz = chi[1]
    chi_zx, chi_zy, chi_zz = chi[2]
    # Check if the chi_xx has the correct format already or not
    if len(chi_xx.shape) < 2:
        trans = np.ones(g_0.shape, dtype=np.complex128)
        chi_xx = trans * chi_xx[:, np.newaxis]
        chi_xy = trans * chi_xy[:, np.newaxis]
        chi_xz = trans * chi_xz[:, np.newaxis]
        chi_yx = trans * chi_yx[:, np.newaxis]
        chi_yy = trans * chi_yy[:, np.newaxis]
        chi_yz = trans * chi_yz[:, np.newaxis]
        chi_zx = trans * chi_zx[:, np.newaxis]
        chi_zy = trans * chi_zy[:, np.newaxis]
        chi_zz = trans * chi_zz[:, np.newaxis]

    n_x = np.sqrt(1.0 - g_0**2)
    q1 = 1 + chi_zz
    q2 = n_x * (chi_xz + chi_zx)
    q3 = chi_xz * chi_zx + chi_yz * chi_zy - (1 + chi_zz) * (g_0**2 + chi_yy) - (1 + chi_xx) * (g_0**2 + chi_zz)
    q4 = n_x * (chi_xy * chi_yz + chi_yx * chi_zy - (chi_xz + chi_zx) * (g_0**2 + chi_yy))
    q5 = (
        (1 + chi_xx) * ((g_0**2 + chi_yy) * (g_0**2 + chi_zz) - chi_yz * chi_zy)
        - chi_xy * chi_yx * (g_0**2 + chi_zz)
        - chi_xz * chi_zx * (g_0**2 + chi_yy)
        + chi_xy * chi_zx * chi_yz
        + chi_yx * chi_xz * chi_zy
    )
    # Note assuming C = 0 => q2 = 0 and q4 = 0
    # And this yields the following eigenstates
    # c = np.sqrt(q3**2 - 4*q1*q5)
    # u1 = np.sqrt((-q3 - c)/2.0/q1)
    # u2 = np.sqrt((-q3 + c)/2.0/q1)
    # u3 = -u1
    # u4 = -u2
    # End simplifaction
    # Proper solution to a 4'th degree polynomial
    u1, u2, u3, u4 = roots4thdegree(q1, q2, q3, q4, q5)
    # print u1[:,ind];print u2[:,ind];print u3[:,ind];print u4[:,ind]
    # Special case M||X for finding errors in the code:
    # u1 = -np.sqrt(g_0**2 + trans*(chi0 + A - B)[:, np.newaxis])
    # u2 = np.sqrt(g_0**2 + trans*(chi0 + A + B)[:, np.newaxis])
    # u3 = -np.sqrt(g_0**2 + trans*(chi0 + A + B)[:, np.newaxis])
    # u4 = np.sqrt(g_0**2 + trans*(chi0 + A - B)[:, np.newaxis])
    # End special case
    # I am lazy and simply sort the roots in ascending order to get the
    # right direction of the light later
    u_temp = np.array((u1, u2, u3, u4))
    pos = np.argsort(u_temp.imag, axis=0)
    u_temp = np.choose(pos, u_temp)
    u = np.zeros(u_temp.shape, dtype=np.complex128)
    u[0] = u_temp[3]
    u[1] = u_temp[2]
    u[2] = u_temp[1]
    u[3] = u_temp[0]

    D = (chi_xz + u * n_x) * (chi_zx + u * n_x) - (1.0 - u**2 + chi_xx) * (g_0**2 + chi_zz)
    # These expressions can throw a runtime warning if D has an incorrect value
    # The different special cases are handled later by non-magnetic flags as well
    # as m//y flag.
    old_error_vals = np.seterr(invalid="ignore")
    P_x = (chi_xy * (g_0**2 + chi_zz) - chi_zy * (chi_xz + u * n_x)) / D
    P_z = (chi_zy * (1.0 - u**2 + chi_xx) - chi_xy * (chi_zx + u * n_x)) / D
    np.seterr(invalid=old_error_vals["invalid"])

    S = np.zeros((4, 4, chi_xx.shape[0], g_0.shape[1]), dtype=np.complex128)
    # The ambient layer
    S[0, 0, 0] = 1.0
    S[0, 2, 0] = 1.0
    S[1, 1, 0] = 1.0
    S[1, 3, 0] = 1.0
    S[2, 0, 0] = g_0[0]
    S[2, 2, 0] = -g_0[0]
    S[3, 1, 0] = g_0[0]
    S[3, 3, 0] = -g_0[0]
    u[0, 0] = g_0[0]
    u[1, 0] = g_0[0]
    u[2, 0] = -g_0[0]
    u[3, 0] = -g_0[0]
    # The rest of the multilayers
    v = (u * P_x - n_x * P_z)[:, 1:]
    w = P_x[:, 1:]
    S[0, :, 1:] = 1.0
    S[1, :, 1:] = v
    S[2, :, 1:] = u[:, 1:]
    S[3, :, 1:] = w

    # Handle the layers that are non-magnetic
    # Removing to test if roughness calcs are affected
    chi = chi_xx[non_mag]
    nm_u1 = np.sqrt(g_0[non_mag] ** 2 + chi)
    nm_u2 = -nm_u1
    sqr_eps = np.sqrt(1 + chi)
    S[0, 0, non_mag] = 1.0
    S[0, 1, non_mag] = 0.0
    S[0, 2, non_mag] = 1.0
    S[0, 3, non_mag] = 0.0
    S[1, 0, non_mag] = 0.0
    S[1, 1, non_mag] = sqr_eps
    S[1, 2, non_mag] = 0.0
    S[1, 3, non_mag] = sqr_eps
    S[2, 0, non_mag] = nm_u1
    S[2, 1, non_mag] = 0.0
    S[2, 2, non_mag] = nm_u2
    S[2, 3, non_mag] = 0.0
    S[3, 0, non_mag] = 0.0
    S[3, 1, non_mag] = nm_u1 / sqr_eps
    S[3, 2, non_mag] = 0.0
    S[3, 3, non_mag] = nm_u2 / sqr_eps
    u[0, non_mag] = nm_u1
    u[1, non_mag] = nm_u1
    u[2, non_mag] = nm_u2
    u[3, non_mag] = nm_u2

    # Take into account the matrix singularity arising when M||Y
    if np.any(mpy):
        # print 'M||Y calcs activated'
        delta = chi_xz[mpy] ** 2 * (1 + chi_xx[mpy])
        nx = n_x[mpy]
        mpy_u1 = np.sqrt(g_0[mpy] ** 2 + chi_yy[mpy])
        mpy_u3 = -mpy_u1
        mpy_u2 = np.sqrt(g_0[mpy] ** 2 + chi_zz[mpy])
        mpy_u4 = -mpy_u2
        S[0, 0, mpy] = 1.0
        S[0, 1, mpy] = 0.0
        S[0, 2, mpy] = 1.0
        S[0, 3, mpy] = 0.0
        S[1, 0, mpy] = 0.0
        S[1, 1, mpy] = -(mpy_u2 * chi_xz[mpy] + nx * (1 + chi_xx[mpy])) / (nx**2 - delta)
        S[1, 2, mpy] = 0.0
        S[1, 3, mpy] = -(mpy_u4 * chi_xz[mpy] + nx * (1 + chi_xx[mpy])) / (nx**2 - delta)
        S[2, 0, mpy] = mpy_u1
        S[2, 1, mpy] = 0.0
        S[2, 2, mpy] = mpy_u3
        S[2, 3, mpy] = 0.0
        S[3, 0, mpy] = 0.0
        S[3, 1, mpy] = -(mpy_u2 * nx + chi_xz[mpy]) / (nx**2 - delta)
        S[3, 2, mpy] = 0.0
        S[3, 3, mpy] = -(mpy_u4 * nx + chi_xz[mpy]) / (nx**2 - delta)
        u[0, mpy] = mpy_u1
        u[1, mpy] = mpy_u2
        u[2, mpy] = mpy_u3
        u[3, mpy] = mpy_u4

    return u, S


def recursion_matrix(X, chi, d, g_0, lamda, u):
    """Use the recursion matrix algorithm to form the W matrix for a multilayer"""
    kappa = 2 * np.pi / lamda
    Fp = np.zeros((2, 2, chi[0][0].shape[0], g_0.shape[1]), dtype=np.complex128)
    Fm = np.zeros((2, 2, chi[0][0].shape[0], g_0.shape[1]), dtype=np.complex128)
    Fp[0, 0] = np.exp(-1.0j * u[0] * kappa * d)
    Fp[1, 1] = np.exp(-1.0j * u[1] * kappa * d)
    Fm[0, 0] = np.exp(-1.0j * u[2] * kappa * d)
    Fm[1, 1] = np.exp(-1.0j * u[3] * kappa * d)
    Fp = Fp[:, :, 1:]
    Fm = Fm[:, :, 1:]

    Xtt = X[:2, :2]
    Xtr = X[:2, 2:]
    Xrt = X[2:, :2]
    Xrr = X[2:, 2:]
    Mtt = dot2(inv2(Fp), inv2(Xtt))
    Mtr = -dot2(Mtt, dot2(Xtr, Fm))
    Mrt = dot2(Xrt, inv2(Xtt))
    Mrr = dot2(Xrr - dot2(Mrt, Xtr), Fm)

    def reduce_func(W, i):
        return calc_W(W[0], W[1], W[2], W[3], Mtt[:, :, i, :], Mtr[:, :, i, :], Mrt[:, :, i, :], Mrr[:, :, i, :])

    W0tt = np.zeros((2, 2, g_0.shape[1]), dtype=np.complex128)
    W0tt[0, 0] = 1.0
    W0tt[1, 1] = 1.0
    W0rr = np.zeros((2, 2, g_0.shape[1]), dtype=np.complex128)
    W0rr[0, 0] = 1.0
    W0rr[1, 1] = 1.0
    W0rt = np.zeros((2, 2, g_0.shape[1]), dtype=np.complex128)
    W0tr = np.zeros((2, 2, g_0.shape[1]), dtype=np.complex128)

    W = reduce(reduce_func, list(range(Mtt.shape[2])), (W0tt, W0tr, W0rt, W0rr))
    return W


def do_calc(g_0, lamda, chi, d, non_mag, mpy):
    """Do the calculation to form the mulatilayer reflectivity matrix, W"""
    # Setting up g0 to be the last index
    g_0 = g_0 * np.ones(chi[0][0].shape[0], dtype=np.complex128)[:, np.newaxis]
    u, S = calc_u_S(chi, g_0, mpy, non_mag)
    X = dot4(inv4(S[:, :, :-1, :]), S[:, :, 1:, :])
    d = d[:, np.newaxis] * np.ones(g_0.shape[1], dtype=np.complex128)
    W_tot = recursion_matrix(X, chi, d, g_0, lamda, u)
    return W_tot[2]


def calc_W(Wtt, Wtr, Wrt, Wrr, Mtt, Mtr, Mrt, Mrr):
    """Calculate the Wn+1 matrices for recursion formalism"""

    ident = np.zeros(Wtt.shape, dtype=np.complex128)
    ident[0, 0] = 1.0
    ident[1, 1] = 1.0

    A = dot2(Mtt, inv2(ident - dot2(Wtr, Mrt)))
    B = dot2(Wrr, inv2(ident - dot2(Mrt, Wtr)))

    Wtt_next = dot2(A, Wtt)
    Wtr_next = Mtr + dot2(dot2(A, Wtr), Mrr)
    Wrt_next = Wrt + dot2(dot2(B, Mrt), Wtt)
    Wrr_next = dot2(B, Mrr)
    return Wtt_next, Wtr_next, Wrt_next, Wrr_next


def calc_nonres(g_0, lamda, chi0, d):
    trans = np.ones(g_0.shape, dtype=np.complex128)
    chi_xx = trans * chi0[:, np.newaxis]
    epsilon = 1.0 + chi_xx
    kappa = 2 * np.pi / lamda
    d = np.ones(g_0.shape) * d[:, np.newaxis]
    g_0 = g_0 * np.ones(chi0.shape)[:, np.newaxis]
    u1 = np.sqrt(g_0**2 + chi0[:, np.newaxis])
    u2 = -u1

    S = np.zeros((4, 4, chi0.shape[0], g_0.shape[1]), dtype=np.complex128)
    S[0, 0, 0] = 1
    S[0, 2, 0] = 1
    S[1, 1, 0] = 1
    S[1, 3, 0] = 1
    S[2, 0, 0] = g_0[0]
    S[2, 2, 0] = -g_0[0]
    S[3, 1, 0] = g_0[0]
    S[3, 3, 0] = -g_0[0]
    # S_v = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
    #                [g_0, 0, -g_0, 0], [0, g_0, 0, -g_0]])
    e = np.sqrt(epsilon[1:])
    S[0, 0, 1:] = 1
    S[0, 2, 1:] = 1
    S[1, 1, 1:] = e
    S[1, 3, 1:] = e
    S[2, 0, 1:] = u1[1:]
    S[2, 2, 1:] = u2[1:]
    S[3, 1, 1:] = u1[1:] / e
    S[3, 3, 1:] = u2[1:] / e
    X = dot4(inv4(S[:, :, :-1, :]), S[:, :, 1:, :])
    # print 'X: ', X

    Fp = np.zeros((2, 2, chi0.shape[0], g_0.shape[1]), dtype=np.complex128)
    Fm = np.zeros((2, 2, chi0.shape[0], g_0.shape[1]), dtype=np.complex128)
    Fp[0, 0] = np.exp(-1.0j * u1 * kappa * d)
    Fp[1, 1] = np.exp(-1.0j * u1 * kappa * d)
    Fm[0, 0] = np.exp(-1.0j * u2 * kappa * d)
    Fm[1, 1] = np.exp(-1.0j * u2 * kappa * d)
    Fp = Fp[:, :, 1:, :]
    Fm = Fm[:, :, 1:, :]
    # print 'F: ', Fp
    Xtt = X[:2, :2]
    Xtr = X[:2, 2:]
    Xrt = X[2:, :2]
    Xrr = X[2:, 2:]
    # print Xrr.shape, Xtr.shape, Fm.shape
    Mtt = dot2(inv2(Fp), inv2(Xtt))
    # print det2(Xtt), det2(Xtr), det2(Xtr), det2(Xrr)
    Mtr = -dot2(Mtt, dot2(Xtr, Fm))
    Mrt = dot2(Xrt, inv2(Xtt))
    # print Mrt
    Mrr = dot2(Xrr - dot2(Mrt, Xtr), Fm)

    return Mrt


if __name__ == "__main__":
    import time

    # alpha = np.arange(0.01, 10.0, 0.01)
    alpha = np.arange(0.25, 30.0, 0.025)
    g_0 = np.sin(alpha * np.pi / 180.0)

    F10 = 16.0 + 10.0j
    F11 = 12.0 + 6.0j
    F1m1 = 20.0 + 14.0j
    fco = 27.0 - 0.001j
    fpt = 78.0 - 0.001j
    n_a_co = 2.0 / 2.86**3
    n_a_pt = 4.0 / 3.92**3
    l = 15.77
    l_array = l * np.ones(g_0.shape)
    # l = 1.07
    re = 2.8179402894e-5
    chi0_co = -(l**2) * re / np.pi * n_a_co * fco
    chi0_pt = -(l**2) * re / np.pi * n_a_pt * fpt

    N = 100
    # Single layer BEGIN
    A = np.array(
        [
            0,
        ]
        + [
            l**2 * re / np.pi * n_a_co * (F11 + F1m1),
        ]
        * N
        + [
            0,
        ],
        dtype=np.complex128,
    )  # [:, np.newaxis]*np.ones(g_0.shape)
    B = np.array(
        [
            0,
        ]
        + [
            l**2 * re / np.pi * n_a_co * (F11 - F1m1),
        ]
        * N
        + [0.0],
        dtype=np.complex128,
    )  # [:, np.newaxis]*np.ones(g_0.shape)
    C = np.array(
        [
            0,
        ]
        + [
            l**2 * re / np.pi * n_a_co * (2 * F10 - F11 - F1m1),
        ]
        * N
        + [0],
        dtype=np.complex128,
    )  # [:, np.newaxis]*np.ones(g_0.shape)

    chi0 = np.array(
        [
            0.0,
        ]
        + [
            chi0_co,
        ]
        * N
        + [
            chi0_pt,
        ]
    )  # [:,np.newaxis]*np.ones(g_0.shape)
    d = np.array(
        [
            1.0,
        ]
        + [
            100.0,
        ]
        * N
        + [10.0]
    )
    M = np.array([[1.0, 0.0, 0.0]] + [[1.0, 0.0, 0.0]] * N + [[1.0, 0.0, 0.0]])  # [:,:,np.newaxis]*np.ones(g_0.shape)
    # SINGLE LAYER END

    # Single interface BEGIN
    # A = np.array([1e-8, l**2*re/np.pi*n_a_co*(F11 + F1m1)], dtype = np.complex128)
    # B = np.array([1e-8,l**2*re/np.pi*n_a_co*(F11 - F1m1)], dtype = np.complex128)
    # C = np.array([0,l**2*re/np.pi*n_a_co*(2*F10 - F11 - F1m1)*0], dtype = np.complex128)
    #
    # chi0 = np.array([0.0, chi0_co])
    # d = np.array([0.0, 100.0])
    # theta = 0.0*np.pi/180.0
    # phi = 0.0*np.pi/180.0
    # M = np.array([[1.0, 0.0, 0.0], [np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), np.sin(phi)]])
    # SINGLE interface END

    ##N = 1
    ##A = np.array([1e-9] + [1e-9, l**2*re/np.pi*n_a_pt*(F11 + F1m1)/5., 1e-9, l**2*re/np.pi*n_a_pt*(F11 + F1m1)/5.]*N + [1e-9], dtype = np.complex128)
    ##B = np.array([1e-9] + [1e-9, l**2*re/np.pi*n_a_pt*(F11 - F1m1)/5., 1e-9, l**2*re/np.pi*n_a_pt*(F11 - F1m1)/5.]*N + [1e-9], dtype = np.complex128)

    ##chi0 = np.array([0.0] + [chi0_pt, chi0_pt, chi0_co, chi0_pt]*N + [chi0_pt])
    ##d = np.array([0.0] + [9.0, 5.0, 10.0, 5.0]*N + [0.0])
    ##M = np.array([[1.0, 0.0, 0.0]]  + [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]*N + [[1.0, 0.0, 0.0]])

    t1 = time.time()
    W = calc_refl(g_0, l_array, chi0, A, B, C, M, d)
    t2 = time.time()
    W = calc_refl_int_lay(g_0, l_array, chi0, A, B, C, M, d, d * 0, d * 0, d * 0, d * 0, d * 0, d * 0, d * 0)
    t3 = time.time()
    iprint("No roughness, simulation time: ", t2 - t1, "s")
    iprint("Roughness factors, simulation time ", t3 - t2, "s")

    # W = calc_nonres(g_0, l, chi0, d)
    Ias = (
        2
        * (W[0, 0] * W[0, 1].conj() + W[1, 0] * W[1, 1].conj()).imag
        / (np.abs(W[0, 0]) ** 2 + np.abs(W[1, 0]) ** 2 + np.abs(W[0, 1]) ** 2 + np.abs(W[1, 1]) ** 2)
    )
    Itot = (np.abs(W[0, 0]) ** 2 + np.abs(W[1, 0]) ** 2 + np.abs(W[0, 1]) ** 2 + np.abs(W[1, 1]) ** 2) / 2
    I = np.abs(W) ** 2
    trans = np.ones(W.shape, dtype=np.complex128)
    trans[0, 1] = 1.0j
    trans[1, 1] = -1.0j
    trans = trans / np.sqrt(2)
    Wc = dot2(trans, dot2(W, np.conj(inv2(trans))))
    Ic = np.abs(Wc) ** 2
    # I2 = np.abs(W2)**2
    ref = np.loadtxt("MAG09235.dat")
    from pylab import *

    figure()
    subplot(211)
    title("Verification of calculations (full lines)\n against Stephanov's x-ray server(symbols)")
    semilogy(
        alpha,
        Itot,
        "r",
        alpha,
        (np.abs(Wc[0, 0]) ** 2 + np.abs(Wc[1, 0]) ** 2),
        "b",
        alpha,
        (np.abs(Wc[0, 1]) ** 2 + np.abs(Wc[1, 1]) ** 2),
        "g",
    )
    semilogy(ref[:, 0], (ref[:, 2] + ref[:, 3]) / 2.0, "r.", ref[:, 0], ref[:, 2], "b.", ref[:, 0], ref[:, 3], "g.")
    legend(("Itot", "circ+", "circ-"))
    ylabel("Reflectivity [arb. u.]")
    subplot(212)
    plot(alpha, Ias, "r")
    plot(ref[:, 0], ref[:, -1], "r.")
    ylabel("Asymmetry")
    xlabel("Incident angle [deg]")
    # print Ias
    # show()

    # try to do some profiling
    import pstats

    calc_refl_int_lay(g_0, l_array, chi0, A, B, C, M, d, d * 0, d * 0, d * 0, d * 0, d * 0, d * 0, d * 0)
    # cProfile.runctx("calc_refl(g_0, l_array, chi0, A, B, C, M, d)", globals(), locals(), "Profile.prof")
    # cProfile.runctx("calc_refl_int_lay(g_0, l_array, chi0, A, B, C, M, d, d*0, d*0, d*0, d*0, d*0, d*0, d*0)",
    #                globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
