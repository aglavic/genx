#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
.. module:: grating
   :synopsis: Models to calcualte the reflectivity from a grating

.. moduleauthor:: Matts Björck <matts.bjorck@gmail.com>

This module provide functions to calculate the reflectivity from gratings using the distorted wave Born approximation
(DWBA) and the kinematical theory.
"""

from functools import reduce

import numpy as np

from scipy.special import j1

from genx.core.custom_logging import iprint

_ctype = np.complex128


def dot2(A, B):
    D = np.zeros(A.shape, dtype=_ctype)
    D[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
    D[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
    D[1, 0] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
    D[1, 1] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]
    return D


def ass_P(k, d, ctype=_ctype):
    """Assemble the propagation matrix for the layers"""
    P = np.zeros((2, 2) + k.shape, dtype=ctype)
    P[0, 0] = np.exp(-1.0j * k * d)
    P[1, 1] = 1.0 / P[0, 0]
    return P


def ass_X(k, ctype=_ctype):
    """Assemble the interface matrix for all interfaces"""
    # k_j = k[..., 1:]
    # k_jm1 = k[..., :-1]
    k_j = k[..., :-1]
    k_jm1 = k[..., 1:]

    X = np.empty((2, 2) + k_j.shape, dtype=ctype)

    X[0, 0] = (k_j + k_jm1) / k_jm1 / 2
    X[0, 1] = (k_jm1 - k_j) / k_jm1 / 2
    X[1, 0] = X[0, 1]
    X[1, 1] = X[0, 0]
    return X


def calc_fields(kz, lamda, v, d, return_sub=False):
    """Calculate the fields inside a multilayer"""
    k_vac = 2 * np.pi / lamda
    kz = kz[:, np.newaxis]
    v_amb = v[-1]
    k_j = np.sqrt(kz * kz - v + v_amb * (1 - (kz / k_vac) ** 2))

    X = ass_X(k_j)
    P = ass_P(k_j, d)

    M = np.ones_like(X)
    # Iterate over all layers - starting at the bottom (index 0)
    M[..., 0] = X[..., 0]
    for i in range(1, X.shape[-1]):
        M[..., i] = dot2(dot2(X[..., i], P[..., i]), M[..., i - 1])

    # Calculations of the wavefields above each interface
    E_T_sub = 1.0 / M[0, 0, :, -1][:, np.newaxis]
    E_T = M[0, 0] * E_T_sub
    E_R = M[1, 0] * E_T_sub

    # r = M[1, 0, :, -1] / M[0, 0, :, -1]
    # r = E_R[:, -1]/E_T[:, -1]
    if return_sub:
        return E_T, E_R, k_j, E_T_sub
    else:
        return E_T, E_R, k_j


def coherent_nonspec_refl(k_in, k_out, lamda, vf_layers, v_mean, d, kin=False, spec_refl=False):
    """Calculates the coherent matrix element T_12 for non-specular rods using DWBA or kinematical theory.

    Parameters:
        k_in (array): The wave vector of the incoming beam (k_x, k_y, k_z), size 3xM.
        k_out (array): The wave vector of the reflected beam (k_x, k_y, k_z), size 3xM.
        lamda (array): The wavelength of the radiation used, size M.
        vf_layers (array): The in-plane fourier transform function of scattering
                                potential for the layers, size MxN.
        v_mean (array): The mean scattering potential for the layers, size N.
        d (array): The thickness of layers. First layer is the substrate, size N.
        kin (bool): Flag for the use of kinematical theory (defualt: False)
        spec_refl (bool): Flag for the return of the specular reflected amplitude.

    Returns:
        T_12 (array): The scattering matrix element
    """
    # vf_func = assemble_vf_func(vf)
    # v_mean = vf_func(np.zeros(2))/area_uc
    T_1, R_1, kz_1 = calc_fields(k_in[2], lamda, v_mean, d)
    T_2, R_2, kz_2 = calc_fields(-k_out[2], lamda, v_mean, d)

    # Note that the ambient layer, index -1, and the substrate, index 0, is not used and must be removed.
    q0 = -(kz_1 + kz_2)[:, 1:-1]
    q1 = -(kz_1 - kz_2)[:, 1:-1]
    q2 = -(-kz_1 + kz_2)[:, 1:-1]
    q3 = -(-kz_2 - kz_1)[:, 1:-1]

    # The in-plane scattering vector - check that this is correct!!!!
    # q_ip = k_out[:2] - k_in[:2]
    d = d[np.newaxis, 1:-1] * np.ones_like(R_2)[:, :-1]
    # vf = vf_func(q_ip).T[:, 1:-1]
    vf_layers = vf_layers[:, 1:-1]
    # The wave amplitudes T_x, R_x does not contain the amplitudes in the substrate, therefore :-1
    w = vf_layers * (
        T_2[:, :-1] * T_1[:, :-1] * ftz_layer(q0, d)
        + R_2[:, :-1] * T_1[:, :-1] * ftz_layer(q1, d)
        + T_2[:, :-1] * R_1[:, :-1] * ftz_layer(q2, d)
        + R_2[:, :-1] * R_1[:, :-1] * ftz_layer(q3, d)
    )
    T_12 = w.sum(axis=-1)

    if kin:
        q = k_out[2] - k_in[2]
        z = d.cumsum(axis=1)
        z = z[:, -1][:, np.newaxis] - z + d
        w = vf_layers * ftz_layer(-q[:, np.newaxis], d) * np.exp(1.0j * q[:, np.newaxis] * z)
        T_12 = w.sum(axis=-1)
        # Adding the reflectivity from the substrate if at the specular rod
        T_12 += np.where(is_spec(k_in, k_out), vf[:, 0] * 1.0j / q * np.exp(1.0j * q * z[:, 0]), 0.0)

    if spec_refl:
        return T_12, R_1[:, -1]  # /T_1[:, -1]
    else:
        return T_12


def coherent_refl(k_in, k_out, lamda, vf, v_mean, d, area_uc, kin=False):
    """Calculates the coherent reflectivity using DWBA.

    Parameters:
        k_in (array): The wave vector of the incoming beam (k_x, k_y, k_z), size 3xM.
        k_out (array): The wave vector of the reflected beam (k_x, k_y, k_z), size 3xM.
        lamda (array): The wavelength of the radiation used, size M.
        vf (array): The in-plane fourier transform function of scattering
                                potential for the layers, size MxN.
        v_mean(array): The mean scattering potential for the layers, size N
        d (array): The thickness of layers. First layer is the substrate, size N.
        area_uc (float): The area of the unit cell in  AA^2.

    Returns:
        T_12 (array): The scattering matrix element
    """
    T_12, r = coherent_nonspec_refl(k_in, k_out, lamda, vf, v_mean, d, kin=kin, spec_refl=True)
    R = np.where(
        np.bitwise_and(is_spec(k_in, k_out), not kin),
        np.abs(r) ** 2,
        1.0 / np.abs(k_in[2]) / np.abs(k_out[2]) / 4 / area_uc**2 * np.abs(T_12) ** 2,
    )
    return R


def is_spec(k_in, k_out):
    return np.bitwise_and(
        np.isclose(k_in[0], k_out[0]), np.bitwise_and(np.isclose(k_in[1], 0), np.isclose(k_out[1], 0))
    )


def ftz_layer(qz, d):
    """The one-diminesional fourier transform for a homogenous layer of thickness d"""
    return (1.0 - np.exp(1.0j * qz * d)) / -1.0j / qz


def assemble_vf_func(vf_list):
    """Creates a function to evaluate vf

    Parameters:
        vf_list (list): A list of vf functions, one for each layer

    Returns:
        vf_func (function): A function that evaluates the vf functions given q-coords
    """

    def vf_func(q):
        return np.array([vf(q) for vf in vf_list])

    return vf_func


# from refl import ReflFunction as ArithFunc
# This is a GenX dependency which is not strictly needed but makes the life easier
# def ft_math(func):
#    """Decorator that enables a fourier transform function to used in arithmetic expressions"""
#    return ArithFunc(func, np.zeros(2), {}, id='ft(q)')


def mult(factor, func):
    """Multilies factor with func"""

    def mult_func(*args, **kwargs):
        return factor * func(*args, **kwargs)

    return mult_func


def ft_circle(radius=1.0):
    """Creates a function that evaluates the fourier transform of a circle with radius"""

    # @ft_math
    def circle(q):
        q_r = np.sqrt(q[0] ** 2 + q[1] ** 2)
        return 2 * np.pi * radius**2 * jinc(q_r * radius)

    return circle


def ft_layer(area_uc):
    """Creates a function that returns the in-plane fourier transform of an in-plane infinite layer"""

    # @ft_math
    def layer(q):
        return np.where(np.bitwise_and(q[0] == 0, q[1] == 0), 1.0, 0.0) * area_uc

    return layer


def jinc(x):
    """The jinc function J_1(x)/x, where J_1 is the Bessel function of the first kind of order 1"""
    # TODO: Check if 0.5*x == 0 is ok - seems to be j1(x) < x
    # print 'jinc:', x
    is_zero = 0.5 * x == 0
    result = j1(x) / np.where(is_zero, 1, x)
    return np.where(is_zero, 0.5, result)


def calc_kin_kout(h, qz_max, qz_step, a, lamda):
    """Calcualtes the k_in and k_out vectors for a reflectivity setup, assumin qy=0.

    Parameters:
        h(int): The inplane reflection index.
        qz_max (float): The maximum qz value wanted.
        qz_step (float): The step length in qz.
        a (float): The unit cell size in the x-direction.
        lamda (float): The wavelength of radiation used.

    Returns:
        k_in (array): A 3xM array representing the incident beam.
        k_out (array): A 3xM array represting the reflected beam.
    """
    qx = h * 2 * np.pi / a
    qz = np.arange(np.sqrt(2 * 2 * np.pi / lamda * np.abs(qx) - qx**2) + qz_step, qz_max, qz_step)
    q_abs = np.sqrt(qx**2 + qz**2)
    delta = np.arcsin(qx / q_abs)
    theta = np.arcsin(q_abs / 4 / np.pi * lamda)
    alpha = theta + delta
    beta = theta - delta
    k_in = 2 * np.pi / 1.54 * np.c_[np.cos(alpha), np.zeros_like(alpha), np.sin(alpha)].T
    k_out = 2 * np.pi / 1.54 * np.c_[np.cos(beta), np.zeros_like(beta), -np.sin(beta)].T
    return k_in, k_out


def ReflQ_ref(Q, lamda, n, d):
    """Parratt's recursion algorithm used as reference method in tests"""
    # Length of k-vector in vaccum
    d = d[1:-1]
    Q0 = 4 * np.pi / lamda
    # Calculates the wavevector in each layer
    Qj = np.sqrt((n[:, np.newaxis] ** 2 - n[-1] ** 2) * Q0**2 + n[-1] ** 2 * Q**2)
    # Fresnel reflectivity for the interfaces
    rp = (Qj[1:] - Qj[:-1]) / (Qj[1:] + Qj[:-1])
    p = np.exp(1.0j * d[:, np.newaxis] * Qj[1:-1])  # Ignoring the top and bottom layer for the calc.
    # Setting up a matrix for the reduce function. Reduce only takes one array
    # as argument
    rpp = np.array(list(map(lambda x, y: [x, y], rp[1:], p)))

    # Paratt's recursion formula

    def formula(rtot, rint):
        return (rint[0] + rtot * rint[1]) / (1 + rtot * rint[0] * rint[1])

    # Implement the recursion formula
    r = reduce(formula, rpp, rp[0])
    # return the reflectivity
    return np.abs(r) ** 2


if __name__ == "__main__":
    from pylab import *

    atype = np.complex128
    lamda = 1.54
    if True:
        # Tests for the calculation of the fields at the vacuum interface.
        theta = arange(0, 5, 0.01) + 1e-13
        Q = 4 * np.pi / lamda * np.sin(theta * np.pi / 180)
        n_rep = 1
        n = array(
            [
                1 - 7.57e-6 + 1.73e-7j,
            ]
            + [1 - 7.57e-6 + 1.73e-7j, 1 - 2.24e-5 + 2.89e-6j] * n_rep
            + [
                1.0,
            ],
            dtype=atype,
        )
        d = array(
            [
                2,
            ]
            + [80, 20] * n_rep
            + [
                2,
            ],
            dtype=atype,
        )
        sigma = array(
            [
                0,
            ]
            + [0, 0] * n_rep
            + [0]
        )
        n_att = 1
        import time

        t1 = time.clock()
        for i in range(n_att):
            R_ref = ReflQ_ref(Q, 1.54, n, d)
        t2 = time.clock()
        for i in range(n_att):
            E_T, E_R, k_j = calc_fields(Q / 2.0, 1.54, (2 * np.pi / 1.54) ** 2 * (1 - n**2), d)
        t3 = time.clock()
        iprint("Paratt: ", (t2 - t1) / n_att)
        iprint("Abeles: ", (t3 - t2) / n_att)
        R_try = np.abs(E_R[:, -1]) ** 2
        semilogy(theta, R_ref)
        semilogy(theta, R_try, ".")

        legend(("Parratt", "Abeles with roughness/interface layer"))
        xlabel("Two Theta [deg.]")
        ylabel("Reflectivity")
        show()

    if False:
        # Test case with a single rod
        a = 2e4
        b = 2e4

        v_lay = (2 * pi / 1.54) ** 2 * (1 - (1 - (1.57 - 0.4j) * r_e * lamda**2 / 2 / np.pi) ** 2)
        v_sub = (2 * pi / 1.54) ** 2 * (1 - (1 - (0.71 - 0.09j) * r_e * lamda**2 / 2 / np.pi) ** 2)
        # print v_lay
        v_lay = (2 * pi / 1.54) ** 2 * (1 - (1 - 7.57e-6 + 1.73e-7j) ** 2)
        # print v_lay
        d = array([0, 1, 0])
        vf_list = [mult(v_lay, ft_layer(a * b)), mult(v_lay, ft_circle(radius=a / 4.0)), mult(0, ft_layer(a * b))]
        vf_func = assemble_vf_func(vf_list)

        h = 0
        qz_max = 0.4
        qz_step = 0.001

        k_in, k_out = calc_kin_kout(h, qz_max, qz_step, a, lamda)
        v_mean = vf_func([0, 0, 0]) / a / b
        iprint(v_mean)
        vf = vf_func(k_out - k_in).T
        R = coherent_refl(k_in, k_out, lamda, vf, v_mean, d, a * b)
        R_kin = coherent_refl(k_in, k_out, lamda, vf, v_mean, d, a * b, kin=True)
        T_1, R_1, kz_1 = calc_fields(k_in[2], lamda, v_mean, d)
        # T = coherent_nonspec_refl(k_in, k_out, lamda, vf, v_mean, d)
        # T_kin = coherent_nonspec_refl(k_in, k_out, lamda, vf, v_mean, d, kin=True)

        figure()
        qz = -(k_out[2] - k_in[2])
        semilogy(qz, np.abs(R))
        semilogy(qz, np.abs(R_1[:, -1]) ** 2, ".")
        semilogy(qz, np.abs(R_kin))
        show()

    if False:
        # Do a h mapping
        a = 2e4
        b = 2e4

        v_lay = (2 * pi / 1.54) ** 2 * (1 - (1 - (1.57 - 0.4j) * r_e * lamda**2 / 2 / np.pi) ** 2)
        v_sub = (2 * pi / 1.54) ** 2 * (1 - (1 - (0.71 - 0.09j) * r_e * lamda**2 / 2 / np.pi) ** 2)
        # print v_lay
        # v_lay = (2*pi/1.54)**2*(1 - (1 - 7.57e-6 + 1.73e-7j)**2)
        # print v_lay
        d = array([2, 500, 50, 1])
        vf_list = [
            mult(v_lay, ft_layer(a * b)),
            mult(v_lay, ft_circle(radius=a / 4.0 * 1.5)),
            mult(v_lay, ft_circle(radius=a / 4.0 * 1.5)),
            mult(0, ft_layer(a * b)),
        ]
        vf_func = assemble_vf_func(vf_list)

        h_max = 6
        qz_max = 0.4
        qz_step = 0.001
        # k_in, k_out = calc_kin_kout(0, qz_max, qz_step, a, lamda)
        # print k_in
        # print k_out
        figure()
        # locator_params(axis='y', nbins=2)
        for h in range(0, h_max):
            k_in, k_out = calc_kin_kout(h, qz_max, qz_step, a, lamda)
            v_mean = vf_func([0, 0, 0]) / a / b
            vf = vf_func(k_out - k_in).T
            R = coherent_refl(k_in, k_out, lamda, vf, v_mean, d, a * b)
            R_kin = coherent_refl(k_in, k_out, lamda, vf, v_mean, d, a * b, kin=True)
            qz = -(k_out[2] - k_in[2])
            ax = subplot(h_max, 1, h + 1)
            plot(qz, R)
            plot(qz, R_kin, "--")
            yloc = plt.MaxNLocator(3)
            ax.yaxis.set_major_locator(yloc)

            # ax.locator_params(nbins=3, axis='y')
            ax.set_yscale("log")
            ax.set_xlim([0, qz_max])
            ax.set_ylim([1e-14, 10])
            ax.yaxis.set_ticks([1e-12, 1e-2])
            ax.text(
                0.95,
                0.95,
                "h = %d" % h,
                verticalalignment="top",
                horizontalalignment="right",
                transform=ax.transAxes,
                color="black",
                fontsize=15,
                bbox={"facecolor": "white", "alpha": 0.5, "pad": 3},
            )
            if h != h_max - 1:
                ax.xaxis.set_ticklabels([])
        suptitle("Comparison between DWBA (full lines) and BA (dashed)")
        subplots_adjust(hspace=0)
        xlabel("Qz [1/AA]")
        show()
